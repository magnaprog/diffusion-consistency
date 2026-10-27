"""Non-neural interpolation and joint Gaussian controls (plan Section 7)."""

from __future__ import annotations

import numpy as np

from prigm.config import GAUSSIAN_JITTER, GAUSSIAN_RHO, HIDDEN_END, HIDDEN_START, N_CHANNELS
from prigm.observation import closure_correct


def interpolation_weights() -> np.ndarray:
    """Weight on index 30 at hidden index i is (i-17)/13."""
    i = np.arange(HIDDEN_START, HIDDEN_END, dtype=np.float64)
    return (i - 17.0) / 13.0


def linear_interpolate(z_window: np.ndarray) -> np.ndarray:
    """Point-mass prediction for hidden bins from observed indices 17 and 30."""
    z = np.asarray(z_window, dtype=np.float64)
    w = interpolation_weights()
    left = z[..., 17, :]
    right = z[..., 30, :]
    hid = (1.0 - w)[:, None] * left[..., None, :] + w[:, None] * right[..., None, :]
    out = z.copy()
    out[..., HIDDEN_START:HIDDEN_END, :] = hid
    return out


def gaussian_fit(
    residuals: np.ndarray,
    rho: float,
    jitter: float = GAUSSIAN_JITTER,
) -> tuple[np.ndarray, np.ndarray]:
    """residuals (N, 72) standardized hidden coords. Population cov + shrinkage."""
    R = np.asarray(residuals, dtype=np.float64)
    mu = R.mean(axis=0)
    xc = R - mu
    n = R.shape[0]
    cov = (xc.T @ xc) / n
    d = np.diag(np.diag(cov))
    shrunk = (1.0 - rho) * cov + rho * d + jitter * np.eye(cov.shape[0])
    return mu, shrunk


def gaussian_sample(
    mean_ref: np.ndarray,
    resid_mean: np.ndarray,
    cov: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """mean_ref and resid_mean (72,); return (K, 72)."""
    loc = np.asarray(mean_ref, dtype=np.float64) + np.asarray(resid_mean, dtype=np.float64)
    return rng.multivariate_normal(loc, cov, size=n_draws)


def flatten_hidden(z: np.ndarray) -> np.ndarray:
    return np.reshape(np.asarray(z[..., HIDDEN_START:HIDDEN_END, :], dtype=np.float64), z.shape[:-2] + (12 * z.shape[-1],))


def unflatten_hidden(flat: np.ndarray, n_ch: int = N_CHANNELS) -> np.ndarray:
    return np.reshape(np.asarray(flat, dtype=np.float64), flat.shape[:-1] + (12, n_ch))


def apply_correction_hidden(z_window: np.ndarray) -> np.ndarray:
    z = np.asarray(z_window, dtype=np.float64).copy()
    hid = z[..., HIDDEN_START:HIDDEN_END, :]
    hid_c = closure_correct(hid)
    z[..., HIDDEN_START:HIDDEN_END, :] = hid_c
    return z
