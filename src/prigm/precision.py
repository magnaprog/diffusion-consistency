"""Separate floating-point rounding limits from magnetic constraint violations.

The arithmetic tolerance scales float64 machine precision by the relevant
physical magnitude. The representation bands start from half the spacing
between adjacent float32 values in standardized coordinates, convert that
spacing to physical units, and propagate it through the support inequalities.
ULP means a unit in the last place: here it is measured using the larger
finite adjacent float32 spacing. These bands describe numerical
representation, not uncertainty in a predicted magnetic field (plan Section 5).
"""

from __future__ import annotations

import numpy as np

from prigm.config import ARITH_EPS_MULT


F64_EPS = np.float64(np.finfo(np.float64).eps)


def arithmetic_tolerance(a: np.ndarray, b: np.ndarray, unit: float) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mag = np.maximum(np.maximum(np.abs(a), np.abs(b)), np.float64(unit))
    return ARITH_EPS_MULT * F64_EPS * mag


def ulp32(u: np.ndarray) -> np.ndarray:
    """Maximum finite adjacent float32 spacing at nearest even float32, in float64."""
    u = np.asarray(u, dtype=np.float64)
    u32 = u.astype(np.float32)
    x = u32.astype(np.float64)
    # Next after toward +inf and -inf in float32, then take the larger spacing.
    nxt = np.nextafter(u32, np.float32(np.inf)).astype(np.float64)
    prv = np.nextafter(u32, np.float32(-np.inf)).astype(np.float64)
    d_hi = np.abs(nxt - x)
    d_lo = np.abs(x - prv)
    # At finite extremes nextafter may be inf; treat that side as non-finite.
    d_hi = np.where(np.isfinite(nxt), d_hi, d_lo)
    d_lo = np.where(np.isfinite(prv), d_lo, d_hi)
    return np.maximum(d_hi, d_lo)


def half_ulp_physical(u_std: np.ndarray, scale: np.ndarray) -> np.ndarray:
    u_std = np.asarray(u_std, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    return 0.5 * scale * ulp32(u_std)


def representation_bands(
    z_phys: np.ndarray,
    u_std: np.ndarray,
    scales: np.ndarray,
    unit_m: float,
    unit_vs: float,
) -> dict[str, np.ndarray]:
    """Propagated half-ULP bands plus arithmetic tolerance (plan 227-233)."""
    z_phys = np.asarray(z_phys, dtype=np.float64)
    u_std = np.asarray(u_std, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    b = half_ulp_physical(u_std, scales)
    b_mx, b_my, b_mz, b_m, b_vs = b[..., 0], b[..., 1], b[..., 2], b[..., 3], b[..., 4]
    b_r = np.sqrt(b_mx ** 2 + b_my ** 2 + b_mz ** 2)
    m = z_phys[..., 3]
    r = np.sqrt(z_phys[..., 0] ** 2 + z_phys[..., 1] ** 2 + z_phys[..., 2] ** 2)
    h_m = 2.0 * np.abs(m) * b_m + b_m ** 2
    h_r = 2.0 * r * b_r + b_r ** 2
    tol_m = arithmetic_tolerance(m, r, unit_m)
    vs = z_phys[..., 4]
    vs_hi = 1.5 * m ** 2 + 2.5 * r ** 2
    tol_vs = arithmetic_tolerance(vs, vs, unit_vs)
    tol_hi = arithmetic_tolerance(vs_hi, vs, unit_vs)
    return {
        "m_minus_r": b_m + b_r + tol_m,
        "vs": b_vs + tol_vs,
        "vs_upper_minus_vs": b_vs + 1.5 * h_m + 2.5 * h_r + tol_hi,
    }


def store_standardized_f32(u: np.ndarray) -> np.ndarray:
    return np.asarray(u, dtype=np.float32)


def promote_f32_to_f64(u32: np.ndarray) -> np.ndarray:
    return np.asarray(u32, dtype=np.float64)
