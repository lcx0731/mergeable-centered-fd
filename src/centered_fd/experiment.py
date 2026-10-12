"""An in-memory oracle audit of a streaming core, using seeded synthetic shards."""

import argparse
import json
from pathlib import Path

import numpy as np

from .core import CenteredFD, FrequentDirections


def synthetic_shards(seed=20261012):
    rng = np.random.default_rng(seed)
    dimension, rows_per_shard = 40, 300
    rotation = np.linalg.qr(rng.normal(size=(dimension, dimension)))[0]
    levels = [-8, -6, -4, -2, 2, 4, 6, 8]
    shards = []
    for level in levels:
        factors = rng.normal(size=(rows_per_shard, 3)) * [3, 2, 1]
        noise = .2 * rng.normal(size=(rows_per_shard, dimension))
        shards.append(factors @ rotation[:, 1:4].T + noise + level * rotation[:, 0])
    return shards, levels


def ingest(shard, ell, batch_size):
    sketch = CenteredFD(shard.shape[1], ell)
    for start in range(0, len(shard), batch_size):
        sketch.update(shard[start:start + batch_size])
    return sketch


def balanced_merge(sketches):
    if not sketches:
        raise ValueError("at least one sketch is required")
    level = list(sketches)
    while len(level) > 1:
        next_level = [level[j].merged(level[j + 1]) for j in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            next_level.append(level[-1])
        level = next_level
    return level[0]


def omit_mean_correction_baseline(sketches):
    """Intentionally WRONG global covariance target; isolated from production merge.

    The returned bound concerns only the sum of within-shard scatters, NOT the
    scatter around the global mean. It must not be labeled a global guarantee.
    """
    raw = FrequentDirections(sketches[0].dimension, sketches[0].ell)
    for sketch in sketches:
        state = sketch.snapshot()
        raw.shrinkage += state.shrinkage
        raw.compressions += state.compressions
        raw.append(state.rows)
    return raw.snapshot()


def run_experiment(seed=20261012, batch_size=75):
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    shards, levels = synthetic_shards(seed)
    data = np.vstack(shards)  # Oracle validation only; the core never does this.
    n, dimension = data.shape
    mean = data.mean(axis=0)
    centered = data - mean
    gram = centered.T @ centered
    covariance = gram / (n - 1)
    singular = np.linalg.svd(centered, compute_uv=False)
    k, widths = 4, [6, 10, 16]
    optimal_residual = float(np.sum(singular[k:]**2))
    within = sum((shard - shard.mean(axis=0)).T @ (shard - shard.mean(axis=0)) for shard in shards)
    between = sum(len(shard) * np.outer(shard.mean(axis=0) - mean, shard.mean(axis=0) - mean) for shard in shards)
    rows, spectra = [], {}
    for ell in widths:
        sequential = CenteredFD(dimension, ell)
        for shard in shards:
            for start in range(0, len(shard), batch_size):
                sequential.update(shard[start:start + batch_size])
        workers = [ingest(shard, ell, batch_size) for shard in shards]
        merged = balanced_merge(workers)
        states = {
            "sequential_corrected": sequential.snapshot(),
            "balanced_merge_corrected": merged.snapshot(),
            "omit_mean_correction": omit_mean_correction_baseline(workers),
        }
        for method, state in states.items():
            valid = method != "omit_mean_correction"
            approximate = state.rows.T @ state.rows / (n - 1)
            error_eigenvalues = np.linalg.eigvalsh(covariance - approximate)
            operator_error = float(np.max(np.abs(error_eigenvalues)))
            directions = np.linalg.svd(state.rows, full_matrices=False)[2][:k]
            residual = centered - (centered @ directions.T) @ directions
            ratio = float(np.sum(residual**2) / optimal_residual)
            bound = state.shrinkage / (n - 1)
            tail_bound = optimal_residual / ((ell - k) * (n - 1))
            tolerance = 1e-9 * float(np.linalg.norm(covariance, ord=2))
            if valid:
                assert error_eigenvalues.min() >= -tolerance
                assert operator_error <= bound + tolerance
                assert bound <= tail_bound + tolerance
                assert ratio <= 1 + k / (ell - k) + tolerance
                assert state.count == n
            rows.append({
                "ell": ell, "method": method,
                "global_covariance_bound_applicable": valid,
                "covariance_operator_error": operator_error,
                "minimum_error_eigenvalue": float(error_eigenvalues.min()),
                "accumulated_global_error_bound": bound if valid else None,
                "bound_for_wrong_within_shard_target_only": bound if not valid else None,
                "global_tail_energy_bound": tail_bound if valid else None,
                "rank_4_reconstruction_error_ratio": ratio,
                "rank_4_ratio_upper_bound": 1 + k / (ell - k) if valid else None,
                "mean_error_l2": float(np.linalg.norm(state.mean - mean)) if valid else None,
                "trace_error_absolute": abs(state.scatter_trace - float(np.trace(gram))) if valid else None,
                "compressions": state.compressions,
                "snapshot_rows": len(state.rows),
                "single_updater_persistent_array_bytes": sequential.persistent_array_bytes,
                "snapshot_rows_plus_mean_bytes": state.rows.nbytes + mean.nbytes,
            })
            if ell == 10:
                spectra[method] = np.linalg.eigvalsh(approximate)[::-1].tolist()
    return {
        "data": "Seeded synthetic mean-shifted shards; no private or external dataset.",
        "seed": seed, "n": n, "dimension": dimension, "shards": len(shards),
        "rows_per_shard": len(shards[0]), "batch_size": batch_size,
        "mean_shift_levels": levels, "within_factor_scales": [3, 2, 1], "noise_sd": .2,
        "widths": widths, "projection_rank": k, "ddof": 1,
        "offline_centered_data_bytes": centered.nbytes,
        "dense_covariance_bytes": covariance.nbytes,
        "optimal_rank_4_residual_energy": optimal_residual,
        "centering_identity": {
            "relative_frobenius_residual": float(np.linalg.norm(gram - within - between) / np.linalg.norm(gram)),
            "total_scatter_trace": float(np.trace(gram)),
            "within_shard_scatter_trace": float(np.trace(within)),
            "between_shard_scatter_trace": float(np.trace(between)),
            "between_shard_covariance_operator_norm": float(np.linalg.norm(between / (n - 1), ord=2)),
        },
        "exact_covariance_spectrum": np.linalg.eigvalsh(covariance)[::-1].tolist(),
        "ell_10_approximate_spectra": spectra,
        "results": rows,
    }


def make_figures(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"sequential_corrected": "#166d75", "balanced_merge_corrected": "#376aab", "omit_mean_correction": "#c26c34"}
    labels = {"sequential_corrected": "Sequential, mean-corrected", "balanced_merge_corrected": "Balanced merge, corrected", "omit_mean_correction": "Incorrect merge: no mean correction"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for method in labels:
        rows = [r for r in summary["results"] if r["method"] == method]
        widths = [r["ell"] for r in rows]
        axes[0].plot(widths, [r["covariance_operator_error"] for r in rows], "o-", color=colors[method], label=labels[method])
        axes[1].plot(widths, [r["rank_4_reconstruction_error_ratio"] for r in rows], "o-", color=colors[method], label=labels[method])
        if method != "omit_mean_correction":
            axes[0].plot(widths, [r["accumulated_global_error_bound"] for r in rows], ":", color=colors[method])
    axes[0].set(title="Covariance error (dotted: valid online bounds)", ylabel="Operator-norm error", yscale="log")
    axes[1].set(title="Rank-4 reconstruction loss", ylabel="Loss / offline optimal rank-4 loss", yscale="log")
    axes[1].axhline(1, color="#808080", linestyle=":", label="Offline optimum")
    for ax in axes:
        ax.set_xticks(summary["widths"])
        ax.set_xlabel("Sketch width ell (snapshot retains at most ell − 1 rows)")
        ax.grid(alpha=.18)
    axes[1].legend(fontsize=8)
    fig.suptitle("Compression cannot repair discarded between-shard variation")
    fig.savefig(output / "error_and_reconstruction.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    positions = np.arange(1, summary["dimension"] + 1)
    axes[0].semilogy(positions, summary["exact_covariance_spectrum"], color="#242f42", label="Exact globally centered covariance")
    for method, values in summary["ell_10_approximate_spectra"].items():
        # Show only retained eigenvalues; zeros are not assigned fake log heights.
        keep = np.asarray(values) > 1e-10
        axes[0].semilogy(positions[keep], np.asarray(values)[keep], "o--", color=colors[method], label=labels[method], markersize=3)
    axes[0].set(title="Covariance spectra at ell = 10", xlabel="Eigenvalue index (sorted separately)", ylabel="Eigenvalue")
    axes[0].legend(fontsize=7)
    rows = [r for r in summary["results"] if r["method"] == "sequential_corrected"]
    index = np.arange(len(rows))
    denominator = summary["dense_covariance_bytes"]
    axes[1].bar(index - .18, [r["single_updater_persistent_array_bytes"] / denominator for r in rows], width=.36, color="#166d75", label="Updater buffer + mean")
    axes[1].bar(index + .18, [r["snapshot_rows_plus_mean_bytes"] / denominator for r in rows], width=.36, color="#376aab", label="Detached snapshot rows + mean")
    axes[1].axhline(1, color="#808080", linestyle=":", label="Dense covariance array")
    axes[1].set_xticks(index, summary["widths"])
    axes[1].set(title="Array payload only; not peak process memory", xlabel="Sketch width ell", ylabel="Bytes / dense covariance bytes")
    axes[1].legend(fontsize=8)
    fig.suptitle("Centering and storage diagnostics · 2,400 rows × 40 features")
    fig.savefig(output / "spectra_and_storage.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--batch-size", type=int, default=75)
    args = parser.parse_args()
    if args.seed < 0 or args.batch_size < 1:
        parser.error("require seed >= 0 and batch-size >= 1")
    summary = run_experiment(args.seed, args.batch_size)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    make_figures(summary, args.output)
    print(f"Wrote summary.json and two figures to {args.output}")


if __name__ == "__main__":
    main()
