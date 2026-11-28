"""Fit the two five-output models in protocol.json. Each checkpoint is restartable."""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from prigm.config import BATCH_SIZE, HIDDEN_END, HIDDEN_START, N_BINS, N_DIFFUSION_STEPS, UNET_WIDTHS
from prigm.loop import configure_determinism
from prigm.loss import LossConfig, prigm_loss
from prigm.model import ConditionalUNet
from prigm.recovery import sha256_file
from prigm.run_fit import load_real_arrays
from prigm.schedule import build_schedule
from prigm.train import clip_and_step, make_optimizer

PROTOCOL = Path(__file__).with_name("protocol.json")
EPOCHS = (3, 10, 17, 24, 31)
HIDDEN = slice(HIDDEN_START, HIDDEN_END)


def contract(arm, seed):
    protocol = json.loads(PROTOCOL.read_text())
    if arm not in protocol["arms"] or seed not in protocol["seeds"]:
        raise ValueError("arm or seed outside protocol")
    module = Path(__file__).resolve()
    root = module.parents[2]
    paths = [module, PROTOCOL, *sorted((root / "prigm").glob("*.py"))]
    sources = {str(path.relative_to(root)): sha256_file(path) for path in paths}
    return {"arm": arm, "seed": seed, "sources": sources, "input": protocol["training_inputs"]["train.npz"]}


def variance_loss(clean, target, mean, scale, weights):
    physical = clean[:, :, HIDDEN] * scale[:5][None, :, None] + mean[:5][None, :, None]
    vector_norm_sq = physical[:, :3].square().sum(dim=1)
    derived = physical[:, 3].square() + physical[:, 4] - vector_norm_sq
    return (weights * ((derived - target) / scale[5]).square().mean(dim=1) / 5).mean()


def context_features(n, device):
    mask = torch.ones(1, 1, N_BINS, device=device)
    mask[:, :, HIDDEN] = 0
    mid = (N_BINS - 1) / 2
    rel = torch.from_numpy(((np.arange(N_BINS, dtype=np.float32) - mid) / mid)).to(device)
    return mask.expand(n, -1, -1), rel.view(1, 1, -1).expand(n, -1, -1)


def atomic_save(path, state):
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def fit(arm, seed, arrays, folder):
    configure_determinism()
    device = torch.device("cuda")
    binding = contract(arm, seed)
    if sha256_file(arrays) != binding["input"]:
        raise ValueError("training input digest mismatch")
    clean_np, observed_np, nrm = load_real_arrays(arrays, 5)
    with np.load(arrays, allow_pickle=False) as f:
        target_np = f["z"][:, HIDDEN, 5].astype(np.float32)
    if sha256_file(arrays) != binding["input"] or len(clean_np) != 122309:
        raise ValueError("training input changed or wrong length")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = ConditionalUNet(n_generated=5, widths=UNET_WIDTHS).to(device)
    optimizer = make_optimizer(model)
    noise_gen = torch.Generator(device="cpu").manual_seed(seed + 17)
    cfg = LossConfig(0.0, None, "S", 5, nrm.b_ref_sq, nrm.mean, nrm.scale)
    sched = build_schedule()
    a, s, weight = [torch.as_tensor(sched[k], dtype=torch.float32, device=device) for k in ("a", "s", "w")]
    mean = torch.as_tensor(nrm.mean, dtype=torch.float32, device=device)
    scale = torch.as_tensor(nrm.scale, dtype=torch.float32, device=device)
    mask, rel = context_features(BATCH_SIZE, device)
    folder.mkdir(parents=True, exist_ok=True)
    epochs = sorted(folder.glob("epoch-??.pt"))
    latest = folder / "latest.pt"
    candidates = epochs + ([latest] if latest.exists() else [])
    start = 0
    for path in candidates:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["contract"] != binding:
            raise ValueError("resume identity mismatch")
        if state["epoch"] >= start:
            start = state["epoch"]
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            noise_gen.set_state(state["noise_rng"])
            torch.set_rng_state(state["cpu_rng"])
            torch.cuda.set_rng_state(state["gpu_rng"])
    for epoch in range(start + 1, EPOCHS[-1] + 1):
        total = 0.
        for i in range(0, len(clean_np), BATCH_SIZE):
            sl = slice(i, i + BATCH_SIZE)
            clean = torch.as_tensor(clean_np[sl], device=device)
            observed = torch.as_tensor(observed_np[sl], device=device)
            n = len(clean)
            t = torch.randint(1, N_DIFFUSION_STEPS + 1, (n,), generator=noise_gen).to(device)
            noise = torch.randn(clean.shape, generator=noise_gen).to(device)
            u = clean.clone()
            u[:, :, HIDDEN] = a[t, None, None] * clean[:, :, HIDDEN] + s[t, None, None] * noise[:, :, HIDDEN]
            optimizer.zero_grad(set_to_none=True)
            out = prigm_loss(model, clean, u, t, observed, mask[:n], rel[:n], cfg, sched)
            loss = out["batch"]
            if arm == "G5aux":
                target = torch.as_tensor(target_np[sl], device=device)
                loss = loss + variance_loss(out["u0_hat"], target, mean, scale, weight[t])
            if not torch.isfinite(loss).item():
                raise ValueError("nonfinite loss")
            loss.backward()
            if not math.isfinite(clip_and_step(model, optimizer)):
                raise ValueError("nonfinite gradient")
            total += float(loss.detach()) * n
        state = {"contract": binding, "epoch": epoch, "updates": epoch * 956,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "noise_rng": noise_gen.get_state(), "cpu_rng": torch.get_rng_state(),
                 "gpu_rng": torch.cuda.get_rng_state()}
        atomic_save(folder / (f"epoch-{epoch:02}.pt" if epoch in EPOCHS else "latest.pt"), state)
        print(json.dumps({"arm": arm, "seed": seed, "epoch": epoch, "mean_loss": total / len(clean_np)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("arm")
    p.add_argument("seed", type=int)
    p.add_argument("arrays", type=Path)
    p.add_argument("folder", type=Path)
    args = p.parse_args()
    fit(args.arm, args.seed, args.arrays, args.folder)
