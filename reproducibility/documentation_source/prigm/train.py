"""Training loop with example-weighted accumulation and recovery hooks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from prigm.config import (
    ADAMW_BETAS,
    ADAMW_EPS,
    ADAMW_LR,
    ADAMW_WD,
    BATCH_SIZE,
    GRAD_CLIP,
    N_DIFFUSION_STEPS,
)
from prigm.loss import LossConfig, prigm_loss
from prigm.schedule import build_schedule


@dataclass
class TrainState:
    updates: int = 0
    epochs: int = 0
    cursor: int = 0


def require_cuda_if_requested() -> torch.device:
    import os

    if os.environ.get("PISCES_REQUIRE_CUDA") == "1" or os.environ.get("PRIGM_REQUIRE_CUDA") == "1":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required but unavailable")
        # occupancy check: allocate and verify device
        t = torch.zeros(1, device="cuda")
        if t.device.type != "cuda":
            raise RuntimeError("tensor not on CUDA")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=ADAMW_LR,
        betas=ADAMW_BETAS,
        eps=ADAMW_EPS,
        weight_decay=ADAMW_WD,
    )


def milestone_epochs(e: int, e_p: int) -> list[int]:
    """q_k = E_p + floor(k (E - E_p)/4), k=0..4, unique, sorted."""
    if e < e_p:
        raise ValueError("E >= E_p required")
    vals = []
    for k in range(5):
        vals.append(e_p + (k * (e - e_p)) // 4)
    out = []
    for v in vals:
        if v not in out:
            out.append(v)
    return out


def accumulate_step(
    model,
    optimizer,
    batch_u0: torch.Tensor,
    batch_obs: torch.Tensor,
    mask: torch.Tensor,
    relpos: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    cfg: LossConfig,
    sched: dict,
    n_examples: int,
    accum_denom: int,
    clip: float = GRAD_CLIP,
) -> dict:
    """One microbatch. Caller zeros grad at accum start and steps at end."""
    a = torch.as_tensor(sched["a"], device=batch_u0.device, dtype=batch_u0.dtype)[t]
    s = torch.as_tensor(sched["s"], device=batch_u0.device, dtype=batch_u0.dtype)[t]
    a = a[:, None, None]
    s = s[:, None, None]
    hid = slice(18, 30)
    u_t = batch_u0.clone()
    u_t[:, :, hid] = a * batch_u0[:, :, hid] + s * noise[:, :, hid]
    out = prigm_loss(model, batch_u0, u_t, t, batch_obs, mask, relpos, cfg, sched)
    # actual-example-weighted accumulation
    (out["batch"] * (n_examples / accum_denom)).backward()
    return out


def clip_and_step(model, optimizer, clip: float = GRAD_CLIP) -> float:
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm)
