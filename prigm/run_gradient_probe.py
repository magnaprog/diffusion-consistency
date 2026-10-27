"""Bounded CPU diagnostic of weighted loss gradients at one pinned checkpoint.

This measures local gradient magnitudes and alignment, not the cause of an
observed score change. No optimizer, parameter updates, sampling scores or TEST
arrays are used. Run with python -m prigm.run_gradient_probe --help.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
from pathlib import Path

import numpy as np
import torch

from prigm.artifacts import id_hash, json_bytes, load_arrays
from prigm.arrays import physical_to_model
from prigm.checkpoint import array_hash, load_torch_payload, normalization_hash, source_hashes
from prigm.config import BETA_CAP, COSINE_OFFSET, HIDDEN_END, HIDDEN_START, N_BINS, SNR_CLIP
from prigm.loss import LossConfig, prigm_loss
from prigm.model import ConditionalUNet, count_parameters
from prigm.recovery import ContractMismatch, PublicationError, load_committed, sha256_bytes, sha256_file
from prigm.schedule import build_schedule

TIMESTEPS = (1, 25, 50, 100)


def fixed_window_indices(n_available: int, n_windows: int) -> np.ndarray:
    """Floor-spaced indices over the full ordered split, independent of scores."""
    if n_available <= 0 or n_windows <= 0:
        raise ValueError("window counts must be positive")
    return np.linspace(0, n_available - 1, min(n_available, n_windows), dtype=np.int64)


def corrupt_gap(u0: torch.Tensor, noise: torch.Tensor, step: int, sched: dict) -> torch.Tensor:
    """Keep clean flanks, matching the training state convention."""
    if noise.shape != u0.shape or noise.dtype != u0.dtype or noise.device != u0.device:
        raise ValueError("noise must match the clean state shape, dtype and device")
    a = torch.as_tensor(sched["a"][step], dtype=u0.dtype, device=u0.device)
    s = torch.as_tensor(sched["s"][step], dtype=u0.dtype, device=u0.device)
    hidden = slice(HIDDEN_START, HIDDEN_END)
    u_t = u0.clone()
    u_t[:, :, hidden] = a * u0[:, :, hidden] + s * noise[:, :, hidden]
    return u_t


def gradient_statistics(data: torch.Tensor, phys: torch.Tensor) -> dict:
    """Euclidean geometry of full parameter gradients, reduced in float64."""
    data = data.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    phys = phys.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if data.shape != phys.shape or not data.numel():
        raise ValueError("gradient vectors must have the same nonempty shape")
    if not torch.isfinite(data).all() or not torch.isfinite(phys).all():
        raise PublicationError("nonfinite loss gradient")
    data_norm = torch.linalg.vector_norm(data).item()
    phys_norm = torch.linalg.vector_norm(phys).item()
    ratio = phys_norm / data_norm if data_norm != 0 else None
    cosine = None
    angle = None
    if data_norm != 0 and phys_norm != 0:
        # Unit vectors avoid multiplying norms; clamp only acos roundoff.
        cosine = max(-1.0, min(1.0, torch.dot(data / data_norm, phys / phys_norm).item()))
        angle = math.degrees(math.acos(cosine))
    return {
        "data_gradient_norm": data_norm,
        "physics_gradient_norm": phys_norm,
        "physics_to_data_norm_ratio": ratio,
        "gradient_cosine": cosine,
        "gradient_angle_degrees": angle,
    }


def probe_case(model, u0, u_t, t, observed, mask, relpos, cfg, sched) -> dict:
    """Differentiate the actual weighted batch means, including both sample calls."""
    params = tuple(model.parameters())
    losses = prigm_loss(model, u0, u_t, t, observed, mask, relpos, cfg, sched)
    values = {"weighted_data_mean": losses["data"].item(),
              "weighted_physics_mean": losses["phys"].item(),
              "total_mean": losses["batch"].item()}
    if not all(math.isfinite(value) for value in values.values()):
        raise PublicationError("nonfinite weighted loss")
    data_grad = torch.autograd.grad(losses["data"], params, retain_graph=True, allow_unused=True)
    # prigm_loss deliberately returns a disconnected zero when c_r == 0.
    phys_grad = (torch.autograd.grad(losses["phys"], params, allow_unused=True)
                 if losses["phys"].requires_grad else (None,) * len(params))

    def vector(grads):
        return torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).reshape(-1)
                          for g, p in zip(grads, params)])

    return {**values, **gradient_statistics(vector(data_grad), vector(phys_grad))}


def schedule_record(sched: dict) -> dict:
    return {
        "steps": sched["n_steps"], "cosine_offset": COSINE_OFFSET,
        "beta_cap": BETA_CAP, "snr_clip": SNR_CLIP, "sigma_1": "sigma_2",
        "array_hashes": {k: sha256_bytes(np.asarray(v, dtype="<f8").tobytes())
                         for k, v in sched.items() if isinstance(v, np.ndarray)},
    }


def load_probe_model(run_root: Path, generation: int, checkpoint_sha256: str,
                     array_identity: dict, sched: dict):
    """Strict pinned inference load; never resume or migrate a legacy payload."""
    meta = load_committed(run_root, generation=generation)
    if meta["files"].get("state.pt") != checkpoint_sha256:
        raise ContractMismatch("selected checkpoint hash mismatch")
    contract = meta.get("contract", {})
    if contract.get("contract_version") != 2:
        raise ContractMismatch("gradient probe requires contract version 2; no legacy migration")
    if contract.get("n_generated") != 6 or contract.get("arm") == "G5":
        raise ContractMismatch("G5 is intentionally excluded: its physics identity is derived")
    if contract.get("n_aux", 0) != 0:
        raise ContractMismatch("auxiliary conditioning is unsupported")
    if contract.get("normalization_hash") != array_identity["normalization_sha256"]:
        raise ContractMismatch("checkpoint and array normalization hashes differ")
    training_diffusion = contract.get("settings", {}).get("diffusion")
    if training_diffusion is not None and training_diffusion != schedule_record(sched):
        raise ContractMismatch("current diffusion schedule differs from training contract")
    state_path = run_root / f"gen_{generation:08d}" / "state.pt"
    data = state_path.read_bytes()
    if sha256_bytes(data) != checkpoint_sha256:
        raise ContractMismatch("checkpoint changed while loading")
    payload = load_torch_payload(data, map_location="cpu")
    if (payload.get("contract") != contract or payload.get("updates") != generation
            or meta.get("updates", generation) != generation):
        raise ContractMismatch("checkpoint payload and committed identity disagree")
    if "normalization" in payload:
        nrm = payload["normalization"]
        if normalization_hash(nrm["mean"], nrm["scale"], nrm["b_ref_sq"], nrm["n_bins"]) != contract["normalization_hash"]:
            raise ContractMismatch("checkpoint normalization payload differs from contract")
    # Initialization is overwritten completely; do not consume the caller's RNG.
    with torch.random.fork_rng(devices=[]):
        model = ConditionalUNet(n_generated=6, widths=tuple(contract["widths"])).cpu().float()
    state = payload["model"]
    expected = model.state_dict()
    if set(state) != set(expected) or any(
        not torch.is_tensor(state[k]) or state[k].shape != expected[k].shape
        or state[k].dtype != expected[k].dtype or not torch.isfinite(state[k]).all()
        for k in expected
    ):
        raise ContractMismatch("model state keys, shapes, dtype or finite values disagree")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, contract, state


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", required=True, type=Path)
    p.add_argument("--generation", required=True, type=int)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--arrays", required=True, type=Path)
    p.add_argument("--split", required=True, choices=("TRAIN", "VAL"))
    p.add_argument("--c-r", required=True, type=float, help="probe coefficient, not necessarily the training coefficient")
    p.add_argument("--output", required=True, type=Path, help="fresh JSON path with an existing parent")
    p.add_argument("--n-windows", default=8, type=int)
    p.add_argument("--seed", default=7003, type=int)
    args = p.parse_args(argv)
    if args.generation < 0 or args.n_windows <= 0 or not 0 <= args.seed < 2**63:
        p.error("require nonnegative generation, positive window count and seed in [0, 2**63)")
    if not math.isfinite(args.c_r) or args.c_r < 0:
        p.error("--c-r must be finite and nonnegative")
    if len(args.checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.checkpoint_sha256):
        p.error("--checkpoint-sha256 must be a lowercase SHA256 digest")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise PublicationError(f"output exists: {args.output}")
    if not args.output.parent.is_dir():
        raise PublicationError(f"output parent does not exist: {args.output.parent}")
    source_root = Path(__file__).resolve().parent.parent
    paths = sorted((source_root / "prigm").rglob("*.py"))
    current_source = source_hashes(paths, relative_to=source_root)
    z, ordered_ids, nrm, identity = load_arrays(args.arrays, args.split)
    if sha256_file(args.arrays) != identity["sha256"]:
        raise ContractMismatch("arrays changed while loading")
    # Fixed before any model call. No scores, labels or test arrays select windows.
    indices = fixed_window_indices(len(z), args.n_windows)
    ids = [ordered_ids[i] for i in indices]
    u0_array, obs_array = physical_to_model(z[indices], nrm, n_generated=6)
    u0, observed = torch.from_numpy(u0_array), torch.from_numpy(obs_array)
    if not torch.isfinite(u0).all() or not torch.isfinite(observed).all():
        raise ContractMismatch("nonfinite standardized probe input")
    sched = build_schedule()
    model, contract, original_state = load_probe_model(
        args.run_root, args.generation, args.checkpoint_sha256, identity, sched,
    )
    mask = torch.ones(len(ids), 1, N_BINS, dtype=u0.dtype, device="cpu")
    mask[:, :, HIDDEN_START:HIDDEN_END] = 0.0
    center = 0.5 * (N_BINS - 1)
    relpos = torch.from_numpy(((np.arange(N_BINS, dtype=np.float32) - center) / center)[None, None, :])
    relpos = relpos.expand(len(ids), -1, -1)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(u0.shape, generator=generator, dtype=u0.dtype, device="cpu")
    records = []
    for step in TIMESTEPS:
        u_t = corrupt_gap(u0, noise, step, sched)
        t = torch.full((len(ids),), step, dtype=torch.long, device="cpu")
        for estimator in ("M", "S"):
            for residual in ("obs", "inst"):
                cfg = LossConfig(c_r=args.c_r, residual=residual, estimator=estimator,
                                 n_generated=6, mean=nrm.mean, scale=nrm.scale, b_ref_sq=nrm.b_ref_sq)
                records.append({
                    "t": step, "estimator": estimator, "residual": residual,
                    **probe_case(model, u0, u_t, t, observed, mask, relpos, cfg, sched),
                })
    if any(not torch.equal(value, original_state[key]) for key, value in model.state_dict().items()):
        raise PublicationError("probe modified model state")
    if any(p.grad is not None for p in model.parameters()):
        raise PublicationError("probe accumulated parameter gradients")
    if source_hashes(sorted((source_root / "prigm").rglob("*.py")), relative_to=source_root) != current_source:
        raise ContractMismatch("source changed while running probe")
    result = {
        "schema_version": 1, "diagnostic": "weighted_physics_data_parameter_gradients",
        "scope": "Local checkpoint diagnostic on fixed TRAIN or VAL windows; not causal evidence for observed score changes.",
        "limitations": [
            "Small deterministic subset and one shared noise realization, not population uncertainty.",
            "Gradients of weighted batch means, not means of per-window gradient norms.",
            "No optimizer step, training, parameter update, model substitution or TEST evaluation.",
            "The fixed index rule does not certify that the checkpoint or hypothesis was chosen before prior test exposure.",
            "G5 excluded because its six-channel physics identity is derived.",
        ],
        "gradient_definition": "Full parameter vector; Euclidean norms, dot-product cosine and angle in degrees; float64 reductions.",
        "undefined": "Ratio is null when the data norm is zero; cosine and angle are null when either norm is zero. No epsilon denominators.",
        "run_root": str(args.run_root.resolve()), "generation_updates": args.generation,
        "checkpoint_sha256": args.checkpoint_sha256,
        "training_contract": contract, "training_source_hashes": contract.get("source_hashes", {}),
        "current_source_hashes": current_source,
        "source_comparison": "Training source is recorded, not required to equal current probe source.",
        "model": {"class": "prigm.model.ConditionalUNet", "widths": contract["widths"],
                  "n_generated": 6, "n_parameters": count_parameters(model), "mode": "eval"},
        "arrays_path": str(args.arrays.resolve()), "arrays": identity,
        "selection": {"rule": "np.linspace(0, n_available - 1, min(n_available, n_windows), dtype=int64); first index if count is one",
                      "n_requested": args.n_windows, "n_selected": len(ids),
                      "indices": indices.tolist(), "ids": ids, "ids_sha256": id_hash(ids)},
        "inputs": {"u0_sha256": array_hash(u0_array), "observed_sha256": array_hash(obs_array),
                   "state_convention": "clean_observed_flanks_hidden_diffusion",
                   "hidden_slice": [HIDDEN_START, HIDDEN_END]},
        "noise": {"seed": args.seed, "generator": "torch.Generator", "device": "cpu",
                  "dtype": str(noise.dtype), "shape": list(noise.shape), "sha256": array_hash(noise.numpy()),
                  "reuse": "One full noise tensor shared across all timesteps, estimators and residuals; only the gap is corrupted."},
        "normalization": {"mean": nrm.mean.tolist(), "scale": nrm.scale.tolist(),
                          "b_ref_sq": nrm.b_ref_sq, "n_bins": nrm.n_bins,
                          "sha256": identity["normalization_sha256"]},
        "c_r": args.c_r,
        "schedule": {**schedule_record(sched), "probe_timesteps": list(TIMESTEPS),
                     "probe_values": {k: [float(sched[k][t]) for t in TIMESTEPS]
                                      for k in ("a", "s", "w", "sigma")}},
        "environment": {"python": platform.python_version(), "numpy": str(np.__version__),
                        "torch": str(torch.__version__), "platform": platform.platform(),
                        "device": "cpu", "model_dtype": str(next(model.parameters()).dtype),
                        "torch_threads": torch.get_num_threads(),
                        "torch_interop_threads": torch.get_num_interop_threads(),
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "mkldnn_enabled": torch.backends.mkldnn.enabled,
                        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS")},
        "records": records,
    }
    data = json_bytes(result)
    # Exclusive creation refuses concurrent outputs and dangling symlinks too.
    with args.output.open("xb") as output:
        output.write(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
