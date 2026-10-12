"""Frequent Directions with bounded buffers and mean-corrected summary merges."""

from dataclasses import dataclass
from operator import index

import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not boolean")
    try:
        return index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _rows(value, dimension):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != dimension or not np.all(np.isfinite(array)):
        raise ValueError(f"rows must be a finite two-dimensional array with {dimension} columns")
    return array


def _readonly(value):
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


def _shrink(rows, ell):
    """Subtract sigma_ell^2 from the largest ell squared singular values.

    The ell-th output row is zero, so only ell-1 rows are stored. This works
    for ell <= number of input rows, even with up to 2*ell buffered rows.
    """
    _, singular, right = np.linalg.svd(rows, full_matrices=False)
    delta = float(singular[ell - 1] ** 2)
    retained = np.sqrt(np.maximum(singular[:ell - 1] ** 2 - delta, 0))
    return retained[:, None] * right[:ell - 1], delta


@dataclass(frozen=True)
class FDSnapshot:
    rows: np.ndarray
    shrinkage: float
    compressions: int


class FrequentDirections:
    """Uncentered row sketch. Persistent buffer contains exactly 2*ell*d floats."""

    def __init__(self, dimension, ell):
        self.dimension = _integer(dimension, "dimension")
        self.ell = _integer(ell, "ell")
        if not 1 <= self.ell <= self.dimension:
            raise ValueError("require 1 <= ell <= dimension")
        self._buffer = np.zeros((2 * self.ell, self.dimension))
        self._used = 0
        self.shrinkage = 0.0
        self.compressions = 0

    @property
    def buffer_bytes(self):
        return self._buffer.nbytes

    def append(self, rows):
        rows = _rows(rows, self.dimension)
        offset = 0
        while offset < len(rows):
            take = min(len(rows) - offset, len(self._buffer) - self._used)
            self._buffer[self._used:self._used + take] = rows[offset:offset + take]
            self._used += take
            offset += take
            if self._used == len(self._buffer):
                compressed, delta = _shrink(self._buffer, self.ell)
                self._used = len(compressed)
                self._buffer[:self._used] = compressed
                self.shrinkage += delta
                self.compressions += 1
        return self

    def snapshot(self):
        """Return a detached read-only sketch, without changing future updates."""
        rows = self._buffer[:self._used]
        delta, extra = 0.0, 0
        if self._used >= self.ell:
            rows, delta = _shrink(rows, self.ell)
            extra = 1
        return FDSnapshot(_readonly(rows), self.shrinkage + delta, self.compressions + extra)


@dataclass(frozen=True)
class CenteredSnapshot:
    count: int
    mean: np.ndarray
    scatter_trace: float
    rows: np.ndarray
    shrinkage: float
    compressions: int

    def covariance(self, ddof=1):
        """Materialize d-by-d covariance for small diagnostics, not streaming."""
        denominator = self._denominator(ddof)
        return self.rows.T @ self.rows / denominator

    def covariance_error_bound(self, ddof=1):
        return self.shrinkage / self._denominator(ddof)

    def _denominator(self, ddof):
        ddof = _integer(ddof, "ddof")
        if ddof < 0 or self.count <= ddof:
            raise ValueError("require 0 <= ddof < count")
        return self.count - ddof

    def components(self, k):
        """Top k right singular vectors (rows), for projection of centered data."""
        k = _integer(k, "k")
        if not 1 <= k <= min(self.rows.shape):
            raise ValueError("k must be positive and no larger than stored sketch row count")
        return np.linalg.svd(self.rows, full_matrices=False)[2][:k]


class CenteredFD:
    """Global-centered scatter sketch with counts, means and exact trace updates.

    Means/trace are exact-arithmetic identities, computed in float64 here.
    Summary queries and merges never mutate either input summary. Different
    batch partitions or merge trees can produce different valid approximations.
    """

    def __init__(self, dimension, ell):
        self._fd = FrequentDirections(dimension, ell)
        self.dimension, self.ell = self._fd.dimension, self._fd.ell
        self._count = 0
        self._mean = np.zeros(self.dimension)
        self._trace = 0.0

    @property
    def persistent_array_bytes(self):
        return self._fd.buffer_bytes + self._mean.nbytes

    def update(self, batch):
        batch = _rows(batch, self.dimension)
        size = len(batch)
        if size == 0:
            return self
        batch_mean = batch.mean(axis=0)
        centered = batch - batch_mean
        total = self._count + size
        difference = batch_mean - self._mean
        correction = np.sqrt(self._count * (size / total)) * difference
        trace = self._trace + float(np.sum(centered**2)) + float(correction @ correction)
        if not np.isfinite(trace):
            raise ValueError("scatter trace exceeds floating-point range")
        self._fd.append(centered)
        if self._count:
            self._fd.append(correction[None, :])
        self._mean = batch_mean.copy() if self._count == 0 else self._mean + (size / total) * difference
        self._count = total
        self._trace = trace
        return self

    def snapshot(self):
        state = self._fd.snapshot()
        return CenteredSnapshot(self._count, _readonly(self._mean), self._trace,
                                state.rows, state.shrinkage, state.compressions)

    def merged(self, other):
        """Return a new sketch of the multiset union; overlapping inputs double-count."""
        if not isinstance(other, CenteredFD) or (self.dimension, self.ell) != (other.dimension, other.ell):
            raise ValueError("merging requires matching dimension and ell")
        a, b = self.snapshot(), other.snapshot()
        result = CenteredFD(self.dimension, self.ell)
        total = a.count + b.count
        if total == 0:
            return result
        difference = b.mean - a.mean
        correction = np.sqrt(a.count * (b.count / total)) * difference
        result._count = total
        result._mean = a.mean + (b.count / total) * difference
        result._trace = a.scatter_trace + b.scatter_trace + float(correction @ correction)
        result._fd.shrinkage = a.shrinkage + b.shrinkage
        result._fd.compressions = a.compressions + b.compressions
        result._fd.append(a.rows).append(b.rows)
        if a.count and b.count:
            result._fd.append(correction[None, :])
        return result
