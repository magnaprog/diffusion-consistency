"""Generate hidden-bin estimates with DDIM at eta=0 (plan Section 6).

Each draw begins with a keyed Gaussian noise array. Subsequent DDIM steps
add no new random noise, so draw-to-draw variation comes from initialization.
Observed bins before and after the gap remain conditioning inputs while only
hidden coordinates evolve. The returned array stores the final clean-state
estimate in bins 18:30 and zeros elsewhere; those outer zeros are storage
placeholders, not predictions of the observed magnetic field.
"""

from __future__ import annotations

import numpy as np
import torch

from prigm.config import HIDDEN_END, HIDDEN_START, N_CHANNELS, N_DRAWS
from prigm.keys import noise_key, pcg64
from prigm.schedule import build_schedule, ddim_timesteps


def initial_noise(
    n_generated: int,
    length: int,
    key: int,
    pair_from: np.ndarray | None = None,
) -> np.ndarray:
    """Return a (length, n_generated) array for noise initialization.

    Generate six standard-normal coordinates at every time position, then keep
    the requested leading channels so G5 shares the first five of six. If
    pair_from is supplied, reuse its leading channels instead of drawing noise.
    ddim_generate uses only the hidden positions of this full-length array.
    """
    if not 1 <= n_generated <= N_CHANNELS:
        raise ValueError(f"n_generated must be between 1 and {N_CHANNELS}")
    if pair_from is not None:
        pair_from = np.asarray(pair_from)
        if (
            pair_from.ndim != 2
            or pair_from.shape[0] != length
            or pair_from.shape[1] < n_generated
        ):
            raise ValueError(
                f"pair_from must have shape ({length}, C) with C >= {n_generated}, "
                f"got {pair_from.shape}"
            )
        return np.asarray(pair_from[:, :n_generated], dtype=np.float64)
    rng = pcg64(key)
    z = rng.standard_normal((length, N_CHANNELS)).astype(np.float64)
    return z[:, :n_generated]


def ddim_generate(
    model,
    observed: torch.Tensor,
    mask: torch.Tensor,
    relpos: torch.Tensor,
    noise: torch.Tensor,
    n_eval: int,
    sched: dict | None = None,
    n_generated: int = N_CHANNELS,
) -> torch.Tensor:
    """Use standardized noise of shape (B, C, L); only hidden coordinates evolve.

    Return the clean-state estimate from the t=1 denoiser call in the hidden
    slice, with zero flanks.
    """
    if sched is None:
        sched = build_schedule()
    steps = ddim_timesteps(n_eval, sched["n_steps"])
    a = torch.as_tensor(sched["a"], device=noise.device, dtype=noise.dtype)
    s = torch.as_tensor(sched["s"], device=noise.device, dtype=noise.dtype)
    hid = slice(HIDDEN_START, HIDDEN_END)
    # Generated state is n_generated channels. Observed context is a separate 6-channel tensor.
    u = observed[:, :n_generated].clone()
    u[:, :, hid] = noise[:, :, hid]
    b = u.shape[0]
    for i, t_i in enumerate(steps):
        t = torch.full((b,), int(t_i), device=u.device, dtype=torch.long)
        u0_hat = model(u, observed, mask, relpos, t)
        a_t = a[int(t_i)]
        s_t = s[int(t_i)]
        eps = (u[:, :, hid] - a_t * u0_hat[:, :, hid]) / torch.clamp(s_t, min=1e-12)
        if i == len(steps) - 1:
            out = torch.zeros_like(noise)
            out[:, :, hid] = u0_hat[:, :, hid]
            return out
        t_next = int(steps[i + 1])
        u[:, :, hid] = a[t_next] * u0_hat[:, :, hid] + s[t_next] * eps
    raise RuntimeError("unreachable")


def generate_ensemble(
    model,
    observed: torch.Tensor,
    mask: torch.Tensor,
    relpos: torch.Tensor,
    window_ids: list,
    split: str,
    eval_seed: int,
    n_eval: int,
    n_draws: int = N_DRAWS,
    n_generated: int = N_CHANNELS,
    pair_noise: np.ndarray | None = None,
    sched: dict | None = None,
) -> np.ndarray:
    """Return standardized float32 draws as (n_windows, n_draws, C, L).

    C is n_generated and L is the number of time bins. This is the stored axis
    order used by the scoring code. Each draw contains a generated hidden
    slice and zero entries outside it; decode_ensemble reads only that slice.
    """
    device = observed.device
    length = observed.shape[-1]
    n_win = len(window_ids)
    noises = []
    for wi, wid in enumerate(window_ids):
        for d in range(n_draws):
            key = noise_key("eval", split, wid, d, eval_seed)
            src = None if pair_noise is None else pair_noise[wi, d]
            npy = initial_noise(n_generated, length, key, pair_from=src)
            noises.append(np.transpose(npy, (1, 0)))
    noise = torch.as_tensor(np.stack(noises, axis=0), device=device, dtype=observed.dtype)
    obs_b = observed.repeat_interleave(n_draws, dim=0)
    mask_b = mask.repeat_interleave(n_draws, dim=0)
    rel_b = relpos.repeat_interleave(n_draws, dim=0)
    with torch.no_grad():
        u = ddim_generate(
            model, obs_b, mask_b, rel_b, noise, n_eval, sched=sched, n_generated=n_generated
        )
    arr = u.detach().cpu().numpy().reshape(n_win, n_draws, n_generated, length).astype(np.float32)
    return arr
