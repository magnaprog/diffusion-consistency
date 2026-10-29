"""Cosine diffusion schedule with noisy timesteps 1 through T (plan Section 6).

Array index 0 stores the clean state at time 0, with alpha_bar[0] = 1.
Noisy timesteps 1 through T use the remaining array indices. PIDM commit
733aeb0fe8e56f8d1b3aaf9ff24f340438c20d9a uses noisy tensor indices starting
at 0: its index t maps to timestep t + 1 here. Sigma_1 := Sigma_2 because
the written posterior variance is zero at timestep 1.
"""

from __future__ import annotations

import numpy as np

from prigm.config import BETA_CAP, COSINE_OFFSET, N_DIFFUSION_STEPS, SNR_CLIP


def cosine_alpha_bar_grid(n_steps: int = N_DIFFUSION_STEPS, s: float = COSINE_OFFSET) -> np.ndarray:
    """Nichol-Dhariwal tilde alpha_bar on x = 0..n_steps, length n_steps+1.

    PIDM: x = linspace(0, n_timesteps, n_timesteps+1), then
    f(x) = cos(((x/T)+s)/(1+s)*pi/2)^2, normalized by f(0).
    """
    n_steps = int(n_steps)
    x = np.linspace(0.0, n_steps, n_steps + 1, dtype=np.float64)
    f = np.cos(((x / n_steps) + s) / (1.0 + s) * (0.5 * np.pi)) ** 2
    return f / f[0]


def build_schedule(
    n_steps: int = N_DIFFUSION_STEPS,
    s: float = COSINE_OFFSET,
    beta_cap: float = BETA_CAP,
) -> dict[str, np.ndarray]:
    """Return arrays of length n_steps+1 with index 0 for the clean state.

    betas[1:] has length n_steps; betas[0] unused (nan).
    alpha_bar[t] for t=1..n_steps is PIDM alphas_prod[t-1].
    """
    n_steps = int(n_steps)
    tilde = cosine_alpha_bar_grid(n_steps, s)
    # PIDM: betas = 1 - tilde[1:]/tilde[:-1], clip to [0, beta_cap]
    betas_pidm = 1.0 - (tilde[1:] / tilde[:-1])
    betas_pidm = np.clip(betas_pidm, 0.0, beta_cap)
    alpha_bar = np.empty(n_steps + 1, dtype=np.float64)
    alpha_bar[0] = 1.0
    alpha_bar[1:] = np.cumprod(1.0 - betas_pidm)
    betas = np.empty(n_steps + 1, dtype=np.float64)
    betas[0] = np.nan
    betas[1:] = betas_pidm
    a = np.sqrt(alpha_bar)
    sm = np.sqrt(np.maximum(1.0 - alpha_bar, 0.0))
    snr = np.empty(n_steps + 1, dtype=np.float64)
    snr[0] = np.inf
    snr[1:] = alpha_bar[1:] / (1.0 - alpha_bar[1:])
    w = np.minimum(snr, SNR_CLIP)
    w[0] = np.nan
    sigma = np.empty(n_steps + 1, dtype=np.float64)
    sigma[0] = np.nan
    # Sigma_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
    one_m = 1.0 - alpha_bar
    sigma[1:] = betas[1:] * one_m[:-1] / one_m[1:]
    # Written formula is 0 at t=1; replace with Sigma_2.
    sigma[1] = sigma[2]
    return {
        "n_steps": n_steps,
        "betas": betas,
        "alpha_bar": alpha_bar,
        "a": a,
        "s": sm,
        "snr": snr,
        "w": w,
        "sigma": sigma,
    }


def ddim_timesteps(n_eval: int, n_steps: int = N_DIFFUSION_STEPS) -> np.ndarray:
    """t_i = n_steps - floor((n_steps-1)*i/(N-1)), i=0..N-1."""
    if n_eval < 2:
        raise ValueError("N>=2 required")
    i = np.arange(n_eval, dtype=np.int64)
    return n_steps - np.floor((n_steps - 1) * i / (n_eval - 1)).astype(np.int64)


def pidm_index(t_1based: int) -> int:
    """Map plan timestep 1..T to PIDM 0-based tensor index."""
    if t_1based < 1:
        raise ValueError("t=0 is not a noisy PIDM index")
    return int(t_1based) - 1
