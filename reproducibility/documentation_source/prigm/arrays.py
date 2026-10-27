"""Window arrays in NCHW standardized layout."""

from __future__ import annotations

import numpy as np

from prigm.config import HIDDEN_END, HIDDEN_START, N_CHANNELS
from prigm.data import Window
from prigm.normalize import AffineNormalizer, mask_hidden


def windows_to_physical(windows: list[Window]) -> np.ndarray:
    return np.stack([w.z for w in windows], axis=0).astype(np.float64)


def physical_to_model(
    z: np.ndarray,
    nrm: AffineNormalizer,
    n_generated: int = N_CHANNELS,
) -> tuple[np.ndarray, np.ndarray]:
    """z (N, 48, 6) physical -> u0 (N, C, 48), obs (N, 6, 48) float32."""
    u = nrm.transform(z.reshape(-1, 6)).reshape(z.shape)
    obs = mask_hidden(u)
    u0 = np.transpose(u[..., :n_generated], (0, 2, 1)).astype(np.float32)
    obs_nchw = np.transpose(obs, (0, 2, 1)).astype(np.float32)
    return u0, obs_nchw


def window_ids(windows: list[Window]) -> list[str]:
    return [f"{w.year:04d}-{w.month:02d}-{str(w.start_left)}-s{w.source}" for w in windows]
