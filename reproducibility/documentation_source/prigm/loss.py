"""Data MSE plus optional physical residual (plan Section 6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from prigm.config import HIDDEN_END, HIDDEN_START, N_CHANNELS, N_HIDDEN
from prigm.model import hidden_mask
from prigm.observation import residual_inst, residual_obs
from prigm.schedule import build_schedule


RESIDUAL_OBS = "obs"
RESIDUAL_INST = "inst"
ESTIMATOR_MEAN = "M"
ESTIMATOR_SAMPLE = "S"


@dataclass
class LossConfig:
    c_r: float
    residual: str | None  # None, "obs", "inst"
    estimator: str  # "M" or "S"
    n_generated: int = N_CHANNELS
    b_ref_sq: float = 1.0
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None


def _decode_physical(u: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    """u: (B, C, L) standardized float32 network tensor -> physical float32."""
    scale = torch.as_tensor(cfg.scale, device=u.device, dtype=u.dtype)
    mean = torch.as_tensor(cfg.mean, device=u.device, dtype=u.dtype)
    c = u.shape[1]
    return u * scale[:c][None, :, None] + mean[:c][None, :, None]


def _residual_from_physical(z: torch.Tensor, kind: str) -> torch.Tensor:
    """z (B, C, L) physical; residual per hidden bin (B, 12)."""
    hid = z[:, :, HIDDEN_START:HIDDEN_END]
    mux, muy, muz = hid[:, 0], hid[:, 1], hid[:, 2]
    m = hid[:, 3]
    vs = hid[:, 4]
    r2 = mux ** 2 + muy ** 2 + muz ** 2
    if kind == RESIDUAL_INST:
        r = m ** 2 - r2
    elif kind == RESIDUAL_OBS:
        if hid.shape[1] < 6:
            raise ValueError("obs residual needs six channels")
        vv = hid[:, 5]
        r = m ** 2 + vs - r2 - vv
    else:
        raise ValueError(kind)
    return r


def two_step_u0(
    model,
    u_t: torch.Tensor,
    t: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor,
    relpos: torch.Tensor,
    sched: dict,
    u0_hat: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (u0_hat, u0_star). Reuse u0_hat at t==1. Hidden coords only."""
    if u0_hat is None:
        u0_hat = model(u_t, observed, mask, relpos, t)
    a = torch.as_tensor(sched["a"], device=u_t.device, dtype=u_t.dtype)
    s = torch.as_tensor(sched["s"], device=u_t.device, dtype=u_t.dtype)
    a_t = a[t][:, None, None]
    s_t = s[t][:, None, None]
    hid = slice(HIDDEN_START, HIDDEN_END)
    eps_hat = (u_t[:, :, hid] - a_t * u0_hat[:, :, hid]) / s_t.clamp_min(1e-12)
    u1_star = u_t.clone()
    u1_star[:, :, hid] = a[1] * u0_hat[:, :, hid] + s[1] * eps_hat
    reuse = t == 1
    if reuse.all():
        return u0_hat, u0_hat
    if (~reuse).all():
        t1 = torch.ones_like(t)
        return u0_hat, model(u1_star, observed, mask, relpos, t1)
    u0_star = u0_hat.clone()
    idx = ~reuse
    t1 = torch.ones_like(t[idx])
    u0_star[idx] = model(u1_star[idx], observed[idx], mask[idx], relpos[idx], t1)
    return u0_hat, u0_star


def data_mse(u0_hat: torch.Tensor, u0: torch.Tensor, n_generated: int) -> torch.Tensor:
    """Mean over generated hidden coordinates. (B,) per-example."""
    hid = hidden_mask(n_generated, device=u0.device, dtype=u0.dtype)
    diff = (u0_hat - u0) * hid
    denom = n_generated * N_HIDDEN
    return (diff ** 2).sum(dim=(1, 2)) / denom


def physics_term(
    z_est: torch.Tensor,
    kind: str,
    c_r: float,
    sigma_t: torch.Tensor,
    b_ref_sq: float,
) -> torch.Tensor:
    """Mean of squared normalized residuals over 12 bins. (B,)"""
    r = _residual_from_physical(z_est, kind)
    sq = (r / b_ref_sq) ** 2
    mean_sq = sq.mean(dim=1)
    coef = (c_r / 2.0) / sigma_t
    return coef * mean_sq


def prigm_loss(
    model,
    u0: torch.Tensor,
    u_t: torch.Tensor,
    t: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor,
    relpos: torch.Tensor,
    cfg: LossConfig,
    sched: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Per-example data + physics, then caller takes batch mean.

    Zero c_r bypasses the residual branch entirely.
    """
    if sched is None:
        sched = build_schedule()
    u0_hat = model(u_t, observed, mask, relpos, t)
    w = torch.as_tensor(sched["w"], device=u_t.device, dtype=u_t.dtype)[t]
    data = w * data_mse(u0_hat, u0, cfg.n_generated)
    phys = torch.zeros_like(data)
    if cfg.c_r != 0.0 and cfg.residual is not None:
        if cfg.estimator == ESTIMATOR_SAMPLE:
            _, u0_star = two_step_u0(
                model, u_t, t, observed, mask, relpos, sched, u0_hat=u0_hat
            )
            u_est = u0_star
        elif cfg.estimator == ESTIMATOR_MEAN:
            u_est = u0_hat
        else:
            raise ValueError(cfg.estimator)
        z_est = _decode_physical(u_est, cfg)
        if z_est.shape[1] == 5:
            r2 = z_est[:, 0] ** 2 + z_est[:, 1] ** 2 + z_est[:, 2] ** 2
            vv = z_est[:, 3] ** 2 + z_est[:, 4] - r2
            z_est = torch.cat((z_est, vv[:, None, :]), dim=1)
        sigma = torch.as_tensor(sched["sigma"], device=u_t.device, dtype=u_t.dtype)[t]
        phys = physics_term(z_est, cfg.residual, cfg.c_r, sigma, cfg.b_ref_sq)
    per = data + phys
    return {
        "per_example": per,
        "batch": per.mean(),
        "data": data.mean(),
        "phys": phys.mean(),
        "u0_hat": u0_hat,
    }
