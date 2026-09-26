"""Expected macro F0.5 decisions under independent pair probabilities."""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=128)
def _quadrature(size: int) -> tuple[np.ndarray, np.ndarray]:
    # The largest integrand degree is 5*size-1. Gaussian quadrature integrates
    # this polynomial exactly with ceil(5*size/2) points.
    points, weights = np.polynomial.legendre.leggauss((5 * size + 1) // 2)
    points, weights = (points + 1) / 2, weights / 2
    sizes = np.arange(1, size + 1, dtype=np.float64)
    weighted = 5 * weights[None, :] * points[None, :] ** (4 * sizes[:, None] - 1)
    return points, weighted


def expected_f05_curve(probabilities: np.ndarray) -> np.ndarray:
    """Expected F0.5 for each prefix, including the empty prediction.

    Every supplied probability contributes to the true-count distribution,
    including candidates outside the chosen prefix. Probability order determines
    the prefixes; callers should sort descending before choosing a prefix.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be a finite one-dimensional array")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must lie between 0 and 1")
    size = len(probabilities)
    result = np.empty(size + 1, dtype=np.float64)
    result[0] = np.prod(1 - probabilities)
    if not size:
        return result
    points, weighted = _quadrature(size)
    factors = 1 - probabilities[:, None] + probabilities[:, None] * points
    generating = np.prod(factors, axis=0)
    selected = np.cumsum(probabilities[:, None] * points / factors, axis=0)
    # E[F0.5] = 5*integral(t**(4*k-1) * E[TP*t**A], t=0..1).
    result[1:] = (weighted * generating[None, :] * selected).sum(axis=1)
    return result


def decode_expected_exact(si: np.ndarray, targets: np.ndarray, probabilities: np.ndarray,
                          threshold: float = 0.05) -> dict[int, list]:
    """Choose each entity's prefix with the greatest exact expected F0.5."""
    if not (len(si) == len(targets) == len(probabilities)):
        raise ValueError("source, target and probability lengths differ")
    if not len(si):
        return {}
    order = np.lexsort((targets, -probabilities, si))
    si, targets, probabilities = si[order], targets[order], probabilities[order]
    starts = np.flatnonzero(np.r_[True, si[1:] != si[:-1]])
    ends = np.r_[starts[1:], len(si)]
    chosen = {}
    for start, end in zip(starts, ends):
        scores = probabilities[start:end]
        eligible = int((scores >= threshold).sum())
        if eligible:
            # Zero-probability tails cannot increase the optimal prefix and do
            # not affect the true-count distribution.
            live = scores[scores > 0]
            curve = expected_f05_curve(live)
            count = int(np.argmax(curve[:eligible + 1]))
        else:
            count = 0
        chosen[int(si[start])] = targets[start:start + count].tolist()
    return chosen
