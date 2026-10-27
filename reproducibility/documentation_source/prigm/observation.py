"""Physical-unit magnetic summaries and support (plan Section 4)."""

from __future__ import annotations

import numpy as np

from prigm.config import N_MINUTES_PER_BIN


def summaries_from_vectors(B: np.ndarray) -> np.ndarray:
    """B: (..., 5, 3) float64 GSE vectors -> (..., 6) z = (mux,muy,muz,m,vs,vv)."""
    B = np.asarray(B, dtype=np.float64)
    if B.shape[-2] != N_MINUTES_PER_BIN or B.shape[-1] != 3:
        raise ValueError(f"expected (..., 5, 3), got {B.shape}")
    mu = B.mean(axis=-2)
    mag = np.linalg.norm(B, axis=-1)
    m = mag.mean(axis=-1)
    vs = ((mag - m[..., None]) ** 2).mean(axis=-1)
    vv = ((B - mu[..., None, :]) ** 2).sum(axis=-1).mean(axis=-1)
    return np.stack((mu[..., 0], mu[..., 1], mu[..., 2], m, vs, vv), axis=-1)


def r_from_z(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return np.sqrt(z[..., 0] ** 2 + z[..., 1] ** 2 + z[..., 2] ** 2)


def residual_obs(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    r2 = z[..., 0] ** 2 + z[..., 1] ** 2 + z[..., 2] ** 2
    return z[..., 3] ** 2 + z[..., 4] - r2 - z[..., 5]


def residual_inst(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    r2 = z[..., 0] ** 2 + z[..., 1] ** 2 + z[..., 2] ** 2
    return z[..., 3] ** 2 - r2


def vs_upper_bound(m: np.ndarray, r: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    return 1.5 * m ** 2 + 2.5 * r ** 2


def support_gaps(z: np.ndarray) -> dict[str, np.ndarray]:
    """Signed gaps: m-r, vs, vs_upper-vs. Positive means inside the inequality."""
    z = np.asarray(z, dtype=np.float64)
    r = r_from_z(z)
    m = z[..., 3]
    vs = z[..., 4]
    return {
        "m_minus_r": m - r,
        "vs": vs,
        "vs_upper_minus_vs": vs_upper_bound(m, r) - vs,
        "r_obs": residual_obs(z),
        "r_inst": residual_inst(z),
    }


def is_realizable(z: np.ndarray, atol: float = 0.0) -> np.ndarray:
    gaps = support_gaps(z)
    r = r_from_z(z)
    m = np.asarray(z, dtype=np.float64)[..., 3]
    vs = np.asarray(z, dtype=np.float64)[..., 4]
    eq = np.abs(residual_obs(z)) <= atol
    return (
        (m + atol >= r)
        & (vs + atol >= 0.0)
        & (vs <= vs_upper_bound(m, r) + atol)
        & eq
    )


def closure_correct(z: np.ndarray) -> np.ndarray:
    """Idempotent support repair retaining mu (plan Section 4)."""
    z = np.asarray(z, dtype=np.float64).copy()
    r = r_from_z(z)
    m = np.maximum(z[..., 3], r)
    vs_hi = vs_upper_bound(m, r)
    vs = np.minimum(np.maximum(z[..., 4], 0.0), vs_hi)
    vv = m ** 2 + vs - r ** 2
    out = z.copy()
    out[..., 3] = m
    out[..., 4] = vs
    out[..., 5] = vv
    return out


def derive_vv(z5: np.ndarray) -> np.ndarray:
    """G5: six-channel from first five physical coordinates."""
    z5 = np.asarray(z5, dtype=np.float64)
    r2 = z5[..., 0] ** 2 + z5[..., 1] ** 2 + z5[..., 2] ** 2
    vv = z5[..., 3] ** 2 + z5[..., 4] - r2
    return np.concatenate((z5, vv[..., None]), axis=-1)
