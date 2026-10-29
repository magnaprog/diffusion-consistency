"""Score predictive draws and combine results with explicit weighting.

The energy score compares whole coordinate vectors. The continuous ranked
probability score (CRPS) here compares each coordinate separately and leaves
aggregation over channels or time to the caller. Both use distinct pairs of
draws for their pairwise term. Coverage uses inclusive bounds selected by
one-based order-statistic ranks.

Monthly averages give each represented month equal weight. For bootstrap
resampling, each sampled occurrence of a year keeps its own month weights,
so repeated years contribute repeatedly. The observed-mean superiority
helper checks only its stated score thresholds; callers must separately
check uncertainty intervals and agreement across trained models (plan Sections 9, 11).
"""

from __future__ import annotations

import numpy as np

from prigm.config import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    ENERGY_PREFACTOR,
    FINAL_SEEDS,
    MU0,
    N_DRAWS,
    PRIMARY_CHANNELS,
)


def pairwise_l2_offdiag(X: np.ndarray) -> float:
    """Mean ||X_k - X_l||_2 over ordered k != l, X: (K, D)."""
    X = np.asarray(X, dtype=np.float64)
    k = X.shape[0]
    if k < 2:
        raise ValueError("need K>=2")
    d = X[:, None, :] - X[None, :, :]
    nrm = np.linalg.norm(d, axis=-1)
    return float((nrm.sum() - np.trace(nrm)) / (k * (k - 1)))


def energy_score(samples: np.ndarray, target: np.ndarray) -> float:
    """ES = mean_k ||Xk-y|| - sum_{k!=l} ||Xk-Xl|| / (2 K (K-1))."""
    X = np.asarray(samples, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if X.ndim != 2:
        raise ValueError("samples (K, D)")
    if X.shape[1] != y.size:
        raise ValueError("sample and target coordinates must match")
    term1 = np.mean(np.linalg.norm(X - y[None, :], axis=-1))
    term2 = pairwise_l2_offdiag(X) / 2.0
    return float(term1 - term2)


def crps_univariate(samples: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Coordinatewise CRPS using off-diagonal draw pairs: (K, D) -> (D,).

    The caller handles any aggregation over time positions or channels.
    """
    X = np.asarray(samples, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if X.ndim != 2:
        raise ValueError("samples (K, D)")
    if X.shape[1] != y.size:
        raise ValueError("sample and target coordinates must match")
    k = X.shape[0]
    if k < 2:
        raise ValueError("need K>=2")
    term1 = np.mean(np.abs(X - y[None, :]), axis=0)
    diff = np.abs(X[:, None, :] - X[None, :, :])
    off = diff.sum(axis=(0, 1))  # diagonal is 0
    term2 = off / (2.0 * k * (k - 1))
    return term1 - term2


def order_interval(samples: np.ndarray, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """1-based order statistics: 2nd/38th -> indices 1 and 37. Inclusive."""
    X = np.sort(np.asarray(samples, dtype=np.float64), axis=0)
    return X[lo - 1], X[hi - 1]


def coverage_inclusive(samples: np.ndarray, target: np.ndarray, lo: int, hi: int) -> np.ndarray:
    lo_v, hi_v = order_interval(samples, lo, hi)
    y = np.asarray(target, dtype=np.float64)
    return (y >= lo_v) & (y <= hi_v)


def interval_width(samples: np.ndarray, lo: int, hi: int) -> np.ndarray:
    lo_v, hi_v = order_interval(samples, lo, hi)
    return hi_v - lo_v


def month_balanced_mean(values: np.ndarray, month_ids: np.ndarray) -> float:
    """Average within month, then equal weight over represented months.

    Repeated month IDs count as one month. For bootstrap multiplicity, use
    month_balanced_mean_replicated, which labels each sampled occurrence separately.
    """
    values = np.asarray(values, dtype=np.float64)
    month_ids = np.asarray(month_ids)
    months = np.unique(month_ids)
    if months.size == 0:
        raise ValueError("empty")
    acc = 0.0
    n = 0
    for m in months:
        acc += float(np.mean(values[month_ids == m]))
        n += 1
    return acc / n


def month_balanced_mean_replicated(
    values: np.ndarray,
    month_ids: np.ndarray,
    year_of,
    years_draw: np.ndarray,
) -> float:
    """With-replacement year (or two-year) blocks. Repeat years, keep Aug+Sep together."""
    values = np.asarray(values, dtype=np.float64)
    month_ids = np.asarray(month_ids)
    years = np.array([year_of(m) for m in month_ids], dtype=np.int64)
    parts_v = []
    parts_m = []
    replica = 0
    for y in years_draw:
        mask = years == int(y)
        if not np.any(mask):
            continue
        # Distinct replica ids so duplicated years remain separate months.
        m_ids = month_ids[mask].astype(np.int64) + replica * 100000
        parts_v.append(values[mask])
        parts_m.append(m_ids)
        replica += 1
    if not parts_v:
        return float("nan")
    return month_balanced_mean(np.concatenate(parts_v), np.concatenate(parts_m))


def flatten_primary(z_hidden: np.ndarray) -> np.ndarray:
    """(12, 6) or (K, 12, 6) -> (60,) or (K, 60) using first five channels."""
    z = np.asarray(z_hidden, dtype=np.float64)
    if z.shape[-2:] != (12, 6):
        raise ValueError("expected hidden summaries with shape (..., 12, 6)")
    return np.reshape(z[..., :PRIMARY_CHANNELS], z.shape[:-2] + (12 * PRIMARY_CHANNELS,))


def energy_density_proxy(z_hidden: np.ndarray) -> np.ndarray:
    """Gap-mean resolved energy-density proxy. z_hidden (..., 12, 6)."""
    z = np.asarray(z_hidden, dtype=np.float64)
    q = z[..., 3] ** 2 + z[..., 4]
    return ENERGY_PREFACTOR * np.sum(q, axis=-1) / (24.0 * MU0)


def variogram_score(samples: np.ndarray, target: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    """Bias-corrected variogram, exponent 1/2. samples (K, D). Can be negative."""
    X = np.asarray(samples, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    k = X.shape[0]
    acc = 0.0
    for i, j in pairs:
        a = np.abs(y[i] - y[j]) ** 0.5
        b = np.abs(X[:, i] - X[:, j]) ** 0.5
        bbar = b.mean()
        sb2 = b.var(ddof=1)
        acc += (a - bbar) ** 2 - sb2 / k
    return float(acc / len(pairs))


def primary_variogram_pairs(n_bins: int = 12, n_chan: int = 5) -> list[tuple[int, int]]:
    """Adjacent-time pairs within each of 5 channels, plus same-time cross-channel."""
    pairs: list[tuple[int, int]] = []

    def idx(t: int, c: int) -> int:
        return t * n_chan + c

    for c in range(n_chan):
        for t in range(n_bins - 1):
            pairs.append((idx(t, c), idx(t + 1, c)))
    for t in range(n_bins):
        for c1 in range(n_chan):
            for c2 in range(c1 + 1, n_chan):
                pairs.append((idx(t, c1), idx(t, c2)))
    return pairs


def year_blocks() -> np.ndarray:
    return np.arange(2005, 2025, dtype=np.int64)


def two_year_blocks() -> list[tuple[int, int]]:
    years = year_blocks()
    return [(int(years[i]), int(years[i + 1])) for i in range(0, 20, 2)]


def bootstrap_indices(n_blocks: int, n_rep: int, seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.integers(0, n_blocks, size=(n_rep, n_blocks), dtype=np.int64)


def percentile_interval(samples: np.ndarray, level: float = 95.0) -> tuple[float, float]:
    lo = (100.0 - level) / 2.0
    hi = 100.0 - lo
    return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))


def superiority_observed(
    score_os: float,
    score_d: float,
    score_d_corr: float,
    score_g5: float,
    score_g5_corr: float,
    nonneural: list[float],
    rel_thresh: float = 0.02,
) -> dict[str, bool | float]:
    """Observed-mean 2% gate only; intervals and seeds checked by caller."""
    denoms = {
        "D": score_d,
        "D_corr": score_d_corr,
        "G5": score_g5,
        "G5_corr": score_g5_corr,
    }
    beats = {}
    for name, base in denoms.items():
        if not (base > 0.0):
            beats[name] = False
        else:
            beats[name] = (base - score_os) / base >= rel_thresh
    nn_ok = all(score_os < s for s in nonneural)
    return {
        "beats_all_neural": all(beats.values()),
        "beats_nonneural": nn_ok,
        **{f"rel_vs_{k}": (denoms[k] - score_os) / denoms[k] if denoms[k] > 0 else np.nan for k in denoms},
    }
