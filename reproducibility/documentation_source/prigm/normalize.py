"""Training-union affine standardization (plan Section 5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prigm.config import N_CHANNELS, PRIMARY_CHANNELS


@dataclass(frozen=True)
class AffineNormalizer:
    mean: np.ndarray  # (6,)
    scale: np.ndarray  # (6,)
    b_ref_sq: float
    n_bins: int

    def transform(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        return (z - self.mean) / self.scale

    def inverse(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        return u * self.scale + self.mean

    def primary_scale(self) -> np.ndarray:
        return self.scale[:PRIMARY_CHANNELS]


def fit_normalizer(z_union: np.ndarray) -> AffineNormalizer:
    """Population mean/std on the deduplicated training-bin union.

    z_union: (N, 6) physical. Divisor N, not N-1.
    Zero std: scale = B_ref for mean-field channels 0-3, B_ref^2 for variances.
    """
    z = np.asarray(z_union, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != N_CHANNELS:
        raise ValueError(f"expected (N, 6), got {z.shape}")
    n = z.shape[0]
    if n == 0:
        raise ValueError("empty training-bin union")
    mean = z.mean(axis=0)
    # population std
    var = ((z - mean) ** 2).mean(axis=0)
    std = np.sqrt(var)
    b_ref_sq = float(np.mean(z[:, 3] ** 2 + z[:, 4]))
    b_ref = float(np.sqrt(b_ref_sq)) if b_ref_sq > 0.0 else 1.0
    scale = std.copy()
    for i in range(4):
        if scale[i] == 0.0:
            scale[i] = b_ref
    for i in (4, 5):
        if scale[i] == 0.0:
            scale[i] = b_ref_sq if b_ref_sq > 0.0 else 1.0
    return AffineNormalizer(mean=mean, scale=scale, b_ref_sq=b_ref_sq, n_bins=n)


def mask_hidden(u: np.ndarray, hidden_start: int = 18, hidden_end: int = 30) -> np.ndarray:
    """Zero hidden entries of the conditioning array after standardization."""
    out = np.asarray(u, dtype=np.float64).copy()
    out[..., hidden_start:hidden_end, :] = 0.0
    return out
