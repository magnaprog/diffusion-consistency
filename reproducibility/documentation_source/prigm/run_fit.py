"""Train one PriGM arm into a fresh run root. CUDA_VISIBLE_DEVICES set outside."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from prigm.arrays import physical_to_model
from prigm.checkpoint import frozen_source_snapshot, normalization_hash, source_hashes
from prigm.config import (
    BATCH_SIZE,
    BETA_CAP,
    COSINE_OFFSET,
    E_PILOT,
    GRAD_CLIP,
    HIDDEN_END,
    HIDDEN_START,
    MAX_COMPLETE_EPOCHS,
    MAX_FINAL_UPDATES,
    N_BINS,
    N_CHANNELS,
    N_DIFFUSION_STEPS,
    N_EPOCH_UPDATES,
    PARAM_CEILING,
    RECOVERY_EVERY,
    SNR_CLIP,
    TOY_WIDTHS,
    UNET_WIDTHS,
)
from prigm.loop import RecoverableTrainer, RunContract, configure_determinism
from prigm.model import ConditionalUNet, count_parameters
from prigm.normalize import AffineNormalizer, fit_normalizer
from prigm.toy import pooled_toy_observations
from prigm.toy_windows import make_toy_training
from prigm.recovery import ContractMismatch, sha256_bytes, sha256_file
from prigm.schedule import build_schedule
from prigm.train import make_optimizer, milestone_epochs, require_cuda_if_requested


def parse_milestones(value: str) -> list[int]:
    try:
        values = [int(v) for v in value.split(",")] if value else []
    except ValueError as exc:
        raise argparse.ArgumentTypeError("milestones must be comma-separated integer updates") from exc
    if any(v <= 0 for v in values) or values != sorted(set(values)):
        raise argparse.ArgumentTypeError("milestones must be positive, unique, and increasing")
    return values


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", required=True)
    p.add_argument("--arm", required=True, choices=["D", "I-S", "O-S", "G5", "I-M", "O-M"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--c-r", type=float, default=0.0)
    p.add_argument("--updates", type=int, required=True, help="total completed-update target, including on resume")
    p.add_argument("--budget-updates", type=int, default=MAX_FINAL_UPDATES,
                   help="frozen lifetime budget; keep unchanged across stage targets")
    p.add_argument("--milestones", type=parse_milestones, default=None,
                   help="frozen comma-separated update milestones; default: registered real-data milestones")
    p.add_argument("--toy", action="store_true")
    p.add_argument("--arrays", default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--profile", action="store_true")
    args = p.parse_args(argv)
    if not 0 <= args.updates <= args.budget_updates or args.batch_size <= 0:
        p.error("require 0 <= --updates <= --budget-updates and positive --batch-size")
    if args.milestones is None:
        args.milestones = [e * N_EPOCH_UPDATES for e in milestone_epochs(MAX_COMPLETE_EPOCHS, E_PILOT)
                           if e * N_EPOCH_UPDATES <= args.budget_updates]
    if any(v > args.budget_updates for v in args.milestones):
        p.error("milestones exceed --budget-updates")
    if args.profile and args.resume:
        p.error("profiling requires a fresh run; resume uses the total --updates target")
    return args


def arm_cfg(arm: str, c_r: float):
    if arm == "D":
        return 6, 0.0, None, "S"
    if arm == "G5":
        return 5, 0.0, None, "S"
    if arm == "I-S":
        return 6, c_r, "inst", "S"
    if arm == "O-S":
        return 6, c_r, "obs", "S"
    if arm == "I-M":
        return 6, c_r, "inst", "M"
    if arm == "O-M":
        return 6, c_r, "obs", "M"
    raise ValueError(arm)


def load_real_arrays(path: Path, n_generated: int):
    with np.load(path, allow_pickle=False) as blob:
        z = blob["z"]
        if z.ndim != 3 or z.shape[1:] != (N_BINS, N_CHANNELS) or not np.isfinite(z).all():
            raise ContractMismatch("invalid physical training arrays")
        ids = blob["ids"]
        if (ids.ndim != 1 or len(ids) != len(z) or ids.dtype.kind != "U"
                or len(set(ids.tolist())) != len(ids) or np.any(ids == "")):
            raise ContractMismatch("training arrays require unique string IDs")
        if str(blob["split"].item()) != "TRAIN":
            raise ContractMismatch("training arrays must have split TRAIN")
        n_bins = blob["n_bins"].item()
        if type(n_bins) is not int:
            raise ContractMismatch("normalization training-bin count must be an integer")
        nrm = AffineNormalizer(mean=blob["mean"], scale=blob["scale"],
                               b_ref_sq=float(blob["b_ref_sq"]), n_bins=n_bins)
    if nrm.n_bins <= 0:
        raise ContractMismatch("normalization requires a positive training-bin count")
    normalization_hash(nrm.mean, nrm.scale, nrm.b_ref_sq, nrm.n_bins)
    u0, obs = physical_to_model(z, nrm, n_generated=n_generated)
    return u0, obs, nrm


def training_settings(device: torch.device, milestones: list[int]) -> dict:
    """Record the actual optimizer defaults and schedule, without drawing RNG."""
    dummy = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(()))])
    optimizer = {k: v for k, v in make_optimizer(dummy).param_groups[0].items() if k != "params"}
    optimizer = json.loads(json.dumps(optimizer))
    schedule = build_schedule()
    return {
        "optimizer": {"class": "torch.optim.AdamW", **optimizer},
        "lr_schedule": "constant",
        "grad_clip": {"max_norm": GRAD_CLIP, "norm_type": 2.0},
        "accumulation_steps": 1,
        "chronological": True,
        "drop_last": False,
        "hidden_slice": [HIDDEN_START, HIDDEN_END],
        "initialization": {"torch_seed": "contract.seed", "noise_seed_offset": 17, "numpy_seed_offset": 31},
        "noise": {"device": "cpu", "generator": "torch.Generator", "t_shape": "actual_batch",
                  "noise_shape": f"actual_batch,n_generated,{N_BINS}", "dtype": "float32"},
        "diffusion": {"steps": N_DIFFUSION_STEPS, "cosine_offset": COSINE_OFFSET,
                      "beta_cap": BETA_CAP, "snr_clip": SNR_CLIP, "sigma_1": "sigma_2",
                      "array_hashes": {k: sha256_bytes(np.asarray(v, dtype="<f8").tobytes())
                                       for k, v in schedule.items() if isinstance(v, np.ndarray)}},
        "precision": {"model": "float32", "normalization": "float64", "amp": False,
                      "default_dtype": str(torch.get_default_dtype()),
                      "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                      "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                      "cudnn_benchmark": torch.backends.cudnn.benchmark,
                      "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                      "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                      "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
        "milestones": milestones,
        "recovery_every": RECOVERY_EVERY,
        "environment": {"python": platform.python_version(), "numpy": str(np.__version__),
                        "torch": str(torch.__version__), "cuda": torch.version.cuda,
                        "cudnn": torch.backends.cudnn.version(), "platform": platform.platform(),
                        "device_type": device.type, "torch_threads": torch.get_num_threads(),
                        "torch_interop_threads": torch.get_num_interop_threads(),
                        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    configure_determinism()
    os.environ.setdefault("PRIGM_REQUIRE_CUDA", "1")
    device = require_cuda_if_requested()
    n_gen, c_r, residual, est = arm_cfg(args.arm, args.c_r)
    source_root = Path(__file__).resolve().parent.parent
    hashes, snapshot = frozen_source_snapshot(source_root)
    settings = training_settings(device, args.milestones)
    settings["source_snapshot_sha256"] = sha256_bytes(snapshot)
    settings["data"] = {"kind": "toy" if args.toy else "real_npz", "ids_covered_by_arrays_hash": not args.toy}
    if args.profile:
        settings["recovery_every"] = 10**9
    data_hashes = {}
    widths = TOY_WIDTHS if args.toy else UNET_WIDTHS
    n = count_parameters(ConditionalUNet(n_generated=n_gen, widths=widths))
    if n >= PARAM_CEILING:
        raise SystemExit(f"param count {n} exceeds ceiling")
    if args.toy:
        settings["data"].update({"seed": args.seed + 99, "n_examples": max(4096, args.batch_size * 8)})
        z, _, _ = make_toy_training(max(4096, args.batch_size * 8), seed=args.seed + 99)
        data_hashes["toy_arrays_sha256"] = sha256_bytes(np.ascontiguousarray(z, dtype="<f8").tobytes())
        nrm = fit_normalizer(pooled_toy_observations())
        u0, obs = physical_to_model(z, nrm, n_generated=n_gen)
    else:
        if not args.arrays:
            raise SystemExit("--arrays required for real data")
        data_hashes["arrays_sha256"] = sha256_file(Path(args.arrays))
        u0, obs, nrm = load_real_arrays(Path(args.arrays), n_gen)
        if sha256_file(Path(args.arrays)) != data_hashes["arrays_sha256"]:
            raise ContractMismatch("source arrays changed while loading")
    # The contract identifies the shared source dataset. Arm-specific normalized
    # input hashes are stored separately by RecoverableTrainer in state.pt.
    if source_hashes(sorted((source_root / "prigm").rglob("*.py")), relative_to=source_root) != hashes:
        raise ContractMismatch("source changed while preparing the fit")
    contract = RunContract(
        arm=args.arm,
        seed=args.seed,
        n_generated=n_gen,
        n_aux=0,
        widths=tuple(widths),
        c_r=c_r,
        residual=residual,
        estimator=est,
        n_examples=int(u0.shape[0]),
        batch_size=args.batch_size,
        max_updates=args.budget_updates,
        source_hashes=hashes,
        data_hashes=data_hashes,
        normalization_hash=normalization_hash(nrm.mean, nrm.scale, nrm.b_ref_sq, nrm.n_bins),
        settings=settings,
    )
    t0 = time.time()
    trainer = RecoverableTrainer(
        Path(args.run_root),
        contract,
        u0,
        obs,
        None,
        nrm.mean,
        nrm.scale,
        nrm.b_ref_sq,
        device=device,
        normalization_n_bins=nrm.n_bins,
        source_snapshot=snapshot,
    )
    if args.profile:
        trainer.snapshot_every = 10**9
        n_prof = min(20, args.updates)
        trainer.train(n_prof, start_new=not args.resume)
        elapsed = time.time() - t0
        rec = {
            "arm": args.arm,
            "params": n,
            "device": str(device),
            "updates": n_prof,
            "seconds": elapsed,
            "sec_per_update": elapsed / max(n_prof, 1),
            "cuda": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        }
        print(json.dumps(rec, indent=2))
        Path(args.run_root).mkdir(parents=True, exist_ok=True)
        (Path(args.run_root) / "profile.json").write_text(json.dumps(rec, indent=2) + "\n")
        return 0
    trainer.train(args.updates, start_new=not args.resume)
    print(json.dumps({"arm": args.arm, "seed": args.seed, "updates": trainer.updates, "epochs": trainer.epochs, "params": n, "seconds": time.time() - t0}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
