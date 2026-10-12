import json
import subprocess
import sys

import numpy as np
from numpy.testing import assert_allclose
import pytest

from centered_fd import CenteredFD
from centered_fd.experiment import balanced_merge, omit_mean_correction_baseline, run_experiment, synthetic_shards


def test_synthetic_generator_repeats_without_external_data():
    first, levels = synthetic_shards(13)
    second, _ = synthetic_shards(13)
    assert levels == [-8, -6, -4, -2, 2, 4, 6, 8]
    assert len(first) == 8
    assert all(part.shape == (300, 40) for part in first)
    for a, b in zip(first, second):
        assert np.array_equal(a, b)


def test_wrong_baseline_really_drops_between_mean_scatter():
    parts = [np.tile([i, 0., 0.], (10, 1)) for i in (-2, 2)]
    summaries = [CenteredFD(3, 3).update(part) for part in parts]
    wrong = omit_mean_correction_baseline(summaries)
    good = balanced_merge(summaries).snapshot()
    assert_allclose(wrong.rows.T @ wrong.rows, np.zeros((3, 3)))
    assert wrong.shrinkage == 0  # Correct for the WRONG (within-only) target.
    assert_allclose(good.rows.T @ good.rows, np.diag([80, 0, 0]), atol=1e-12)


def test_odd_sized_merge_tree():
    summaries = [CenteredFD(3, 2).update([[i, 1, 0]]) for i in range(5)]
    result = balanced_merge(summaries).snapshot()
    assert result.count == 5
    assert_allclose(result.mean, [2, 1, 0])
    assert_allclose(result.scatter_trace, 10)
    with pytest.raises(ValueError):
        balanced_merge([])


def test_full_audit_checks_bounds_and_marks_wrong_target():
    summary = run_experiment(seed=20261012)
    assert len(summary["results"]) == 9
    assert summary["centering_identity"]["relative_frobenius_residual"] < 1e-12
    for row in summary["results"]:
        assert row["rank_4_reconstruction_error_ratio"] >= 1 - 1e-10
        if row["global_covariance_bound_applicable"]:
            assert row["covariance_operator_error"] <= row["accumulated_global_error_bound"] + 1e-9
            assert row["mean_error_l2"] < 1e-12
        else:
            assert row["accumulated_global_error_bound"] is None
            assert row["covariance_operator_error"] > row["bound_for_wrong_within_shard_target_only"]
            assert row["rank_4_reconstruction_error_ratio"] > 5


@pytest.mark.parametrize("batch_size", [0, -1, True, 2.5])
def test_bad_batch_size(batch_size):
    with pytest.raises(ValueError):
        run_experiment(batch_size=batch_size)


def test_cli_json_and_figures(tmp_path):
    subprocess.run([
        sys.executable, "-W", "error", "-m", "centered_fd.experiment", "--output", str(tmp_path),
    ], check=True, capture_output=True, text=True)
    content = (tmp_path / "summary.json").read_text()
    assert "Infinity" not in content and "NaN" not in content
    assert json.loads(content)["n"] == 2400
    for name in ("error_and_reconstruction.png", "spectra_and_storage.png"):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
