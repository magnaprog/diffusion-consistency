"""Linear and Gaussian interpolation comparison methods (plan Section 7).

Linear interpolation fills hidden bins 18 through 29 from observed bins 17
and 30. Gaussian interpolation adds a fitted residual vector to that
interpolation. In run_controls, each training residual contains all six
quantities at all 12 hidden positions, so its covariance can include
dependence across both time and channels. Here, controls means comparison
methods; the term does not imply a causal experiment.
"""

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
    """Fit a residual mean and a covariance shrunk toward its diagonal.

    Rows of residuals are observations and columns are standardized coordinates.
    The joint hidden-hour fit uses shape (N, 72), for 12 positions times six
    quantities; this helper also accepts other coordinate counts. Covariance
    uses divisor N. rho mixes that covariance with its diagonal, then jitter
    is added to every diagonal entry. The caller supplies the residuals and
    chooses rho; this function does not select or validate that choice.
    """
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
