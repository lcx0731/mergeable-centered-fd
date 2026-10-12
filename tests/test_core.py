import numpy as np
import pytest
from numpy.testing import assert_allclose

from centered_fd import CenteredFD, FrequentDirections


def check_bound(data, state, ell, centered=True):
    matrix = data - data.mean(axis=0) if centered else data
    gram = matrix.T @ matrix
    error = gram - state.rows.T @ state.rows
    tolerance = 1e-9 * max(1.0, np.linalg.norm(gram, 2))
    eigenvalues = np.linalg.eigvalsh(error)
    assert eigenvalues.min() >= -tolerance
    assert eigenvalues.max() <= state.shrinkage + tolerance
    assert np.trace(error) >= ell * state.shrinkage - tolerance
    singular = np.linalg.svd(matrix, compute_uv=False)
    for k in range(ell):
        tail = float(np.sum(singular[k:]**2))
        assert state.shrinkage <= tail / (ell - k) + tolerance


def test_single_compression_matches_diagonal_hand_calculation():
    state = FrequentDirections(3, 2).append(np.diag([5., 3., 1.])).snapshot()
    assert state.shrinkage == 9
    assert_allclose(state.rows.T @ state.rows, np.diag([16, 0, 0]), atol=1e-13)
    assert state.compressions == 1


@pytest.mark.parametrize("ell", [1, 2, 4, 7])
def test_uncentered_psd_spectral_and_tail_bounds(ell):
    data = np.random.default_rng(19).normal(size=(93, 7))
    sketch = FrequentDirections(7, ell)
    for end in range(0, len(data), 13):
        sketch.append(data[end:end + 13])
        assert sketch.buffer_bytes == 2 * ell * 7 * 8
        check_bound(data[:min(len(data), end + 13)], sketch.snapshot(), ell, centered=False)


@pytest.mark.parametrize("ell", [2, 4, 7])
@pytest.mark.parametrize("batch_size", [1, 8, 101])
def test_centered_bounds_and_exact_moments(ell, batch_size):
    data = np.random.default_rng(37).normal(size=(101, 7)) + np.arange(101)[:, None] / 9
    sketch = CenteredFD(7, ell)
    for start in range(0, len(data), batch_size):
        sketch.update(data[start:start + batch_size])
    state = sketch.snapshot()
    assert state.count == len(data)
    assert_allclose(state.mean, data.mean(axis=0), atol=1e-13)
    assert_allclose(state.scatter_trace, np.sum((data - data.mean(axis=0))**2), rtol=1e-13)
    check_bound(data, state, ell)
    assert len(state.rows) < ell


def test_low_rank_data_has_negligible_loss():
    rng = np.random.default_rng(51)
    data = rng.normal(size=(120, 2)) @ rng.normal(size=(2, 9)) + np.arange(9)
    state = CenteredFD(9, 4).update(data).snapshot()
    exact = (data - data.mean(axis=0)).T @ (data - data.mean(axis=0))
    assert_allclose(state.rows.T @ state.rows, exact, atol=2e-10)
    assert state.shrinkage < 1e-20


def test_constant_rows_have_zero_scatter():
    data = np.tile([2., -3., 4.], (30, 1))
    state = CenteredFD(3, 2).update(data[:7]).update(data[7:]).snapshot()
    assert state.scatter_trace == 0
    assert state.shrinkage == 0
    assert_allclose(state.covariance(), np.zeros((3, 3)))


def test_mean_correction_recovers_variation_between_constant_shards():
    a = np.tile([0., 1., 0.], (6, 1))
    b = np.tile([2., 1., 0.], (6, 1))
    first, second = CenteredFD(3, 3).update(a), CenteredFD(3, 3).update(b)
    assert first.snapshot().scatter_trace == second.snapshot().scatter_trace == 0
    combined = first.merged(second).snapshot()
    assert combined.count == 12
    assert_allclose(combined.mean, [1, 1, 0])
    assert_allclose(combined.rows.T @ combined.rows, np.diag([12, 0, 0]), atol=1e-13)
    assert_allclose(combined.scatter_trace, 12)
    assert_allclose(combined.covariance(), np.diag([12 / 11, 0, 0]), atol=1e-13)


def test_merge_trees_keep_bounds_not_identical_sketches():
    rng = np.random.default_rng(22)
    parts = [rng.normal(size=(47, 8)) + i * np.arange(8) for i in range(4)]
    sketches = [CenteredFD(8, 4).update(part) for part in parts]
    a, b, c, d = sketches
    balanced = a.merged(b).merged(c.merged(d)).snapshot()
    left_deep = a.merged(b).merged(c).merged(d).snapshot()
    all_data = np.vstack(parts)
    for state in (balanced, left_deep):
        check_bound(all_data, state, 4)
        assert_allclose(state.mean, all_data.mean(axis=0), atol=1e-13)
        assert_allclose(state.scatter_trace, np.sum((all_data - all_data.mean(axis=0))**2), rtol=1e-13)
    assert not np.allclose(balanced.rows.T @ balanced.rows, left_deep.rows.T @ left_deep.rows)


def test_merge_does_not_mutate_inputs_and_empty_is_identity():
    data = np.random.default_rng(1).normal(size=(80, 6))
    a, b = CenteredFD(6, 3).update(data[:40]), CenteredFD(6, 3).update(data[40:])
    before_a, before_b = a.snapshot(), b.snapshot()
    a.merged(b)
    assert_allclose(a.snapshot().rows, before_a.rows)
    assert_allclose(b.snapshot().rows, before_b.rows)
    empty = CenteredFD(6, 3)
    for state in (a.merged(empty).snapshot(), empty.merged(a).snapshot()):
        assert state.count == before_a.count
        assert state.shrinkage == before_a.shrinkage
        assert_allclose(state.rows.T @ state.rows, before_a.rows.T @ before_a.rows, atol=1e-13)
        assert_allclose(state.mean, before_a.mean)
    assert empty.merged(empty).snapshot().count == 0


def test_snapshot_queries_do_not_change_future_updates():
    data = np.random.default_rng(6).normal(size=(67, 5))
    queried, unqueried = CenteredFD(5, 3), CenteredFD(5, 3)
    for start in range(0, len(data), 7):
        queried.update(data[start:start + 7])
        queried.snapshot()
        queried.snapshot()
        unqueried.update(data[start:start + 7])
    assert_allclose(queried.snapshot().rows, unqueried.snapshot().rows, atol=0)
    assert queried.snapshot().shrinkage == unqueried.snapshot().shrinkage
    assert not queried.snapshot().rows.flags.writeable
    assert not queried.snapshot().mean.flags.writeable


def test_translation_invariance_at_moderate_large_offset():
    data = np.random.default_rng(7).normal(size=(100, 5))
    states = []
    for shift in (np.zeros(5), np.arange(5) * 1e5):
        sketch = CenteredFD(5, 3)
        for part in np.array_split(data + shift, 5):
            sketch.update(part)
        states.append(sketch.snapshot())
    assert_allclose(states[0].rows.T @ states[0].rows, states[1].rows.T @ states[1].rows, atol=2e-8)
    assert_allclose(states[0].scatter_trace, states[1].scatter_trace, atol=2e-8)


def test_projection_bound_and_orthonormal_components():
    rng = np.random.default_rng(13)
    data = rng.normal(size=(140, 10)) * np.arange(1, 11)
    centered = data - data.mean(axis=0)
    state = CenteredFD(10, 6).update(data).snapshot()
    singular = np.linalg.svd(centered, compute_uv=False)
    for k in (1, 3, 5):
        components = state.components(k)
        assert_allclose(components @ components.T, np.eye(k), atol=1e-13)
        residual = centered - centered @ components.T @ components
        actual = np.sum(residual**2)
        optimum = np.sum(singular[k:]**2)
        assert actual >= optimum - 1e-8
        assert actual <= (1 + k / (6 - k)) * optimum + 1e-8


def test_ddof_scaling_and_invalid_small_sample_operations():
    state = CenteredFD(3, 2).update(np.arange(18).reshape(6, 3)).snapshot()
    assert_allclose(state.covariance(ddof=0) * 6, state.covariance(ddof=1) * 5)
    assert_allclose(state.covariance_error_bound(0) * 6, state.covariance_error_bound(1) * 5)
    for ddof in (-1, 6, 1.5, True):
        with pytest.raises(ValueError):
            state.covariance(ddof)
    with pytest.raises(ValueError):
        CenteredFD(3, 2).snapshot().covariance()
    with pytest.raises(ValueError):
        CenteredFD(3, 2).update([[1, 2, 3]]).snapshot().covariance()
    for k in (0, 2, True):
        with pytest.raises(ValueError):
            state.components(k)


@pytest.mark.parametrize("dimension,ell", [(0, 1), (3, 0), (3, 4), (3, 1.5), (3, True), (2.5, 1)])
def test_invalid_sizes(dimension, ell):
    with pytest.raises(ValueError):
        CenteredFD(dimension, ell)


@pytest.mark.parametrize("batch", [np.zeros(3), np.zeros((2, 4)), [[1, 2, np.nan]], [[1, 2, np.inf]]])
def test_invalid_rows_leave_state_unchanged(batch):
    sketch = CenteredFD(3, 2)
    with pytest.raises(ValueError):
        sketch.update(batch)
    assert sketch.snapshot().count == 0


def test_empty_updates_and_mismatched_merges():
    sketch = CenteredFD(3, 2).update(np.empty((0, 3)))
    assert sketch.snapshot().count == 0
    for other in (CenteredFD(4, 2), CenteredFD(3, 3), object()):
        with pytest.raises(ValueError):
            sketch.merged(other)
