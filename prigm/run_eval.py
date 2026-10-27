"""Generate predictive draws for a bounded subset of windows from a saved model.

The caller identifies the checkpoint generation and its SHA256 hash. Before
sampling, the command checks the recorded source, data, normalization,
training counters, and sampler settings against the declared protocol.
TEST work additionally requires the recorded model selection and exposure
receipt. A chunk contains at most SAMPLE_CHUNK_WINDOWS consecutive windows,
with their IDs saved alongside the draws.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from prigm.artifacts import (
    NOISE_CONVENTION, STATE_CONVENTION, load_arrays, publish_ensemble, require_source_match,
)
from prigm.arrays import physical_to_model
from prigm.barrier import load_protocol, require_test_started
from prigm.checkpoint import load_torch_payload, normalization_hash, verify_source_snapshot
from prigm.config import HIDDEN_END, HIDDEN_START, N_BINS, N_DRAWS, SAMPLE_CHUNK_WINDOWS
from prigm.loop import configure_determinism
from prigm.model import ConditionalUNet
from prigm.recovery import ContractMismatch, PublicationError, load_committed, sha256_file
from prigm.sampler import generate_ensemble
from prigm.schedule import build_schedule
from prigm.train import require_cuda_if_requested


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", required=True)
    p.add_argument("--generation", type=int, required=True)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--out", required=True, help="fresh committed chunk directory")
    p.add_argument("--arrays", required=True)
    p.add_argument("--split", required=True, choices=("VAL", "TEST", "TOY"))
    p.add_argument("--n-eval", type=int, required=True)
    p.add_argument("--eval-seed", type=int, default=7)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=SAMPLE_CHUNK_WINDOWS)
    p.add_argument("--study-root", required=True)
    p.add_argument("--protocol-sha256", required=True)
    args = p.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise PublicationError(f"output root exists: {out}")
    protocol = load_protocol(Path(args.study_root), args.protocol_sha256)
    require_source_match(protocol)
    selection = require_test_started(Path(args.study_root), args.split, args.protocol_sha256)
    if args.offset < 0 or args.limit <= 0 or args.limit > SAMPLE_CHUNK_WINDOWS:
        raise ContractMismatch("invalid fixed-layout chunk bounds")
    requested_sampler = {"n_eval": args.n_eval, "n_draws": N_DRAWS, "eval_seed": args.eval_seed}
    if requested_sampler != protocol["samplers"][args.split]:
        raise ContractMismatch("sampler does not match frozen split protocol")
    if selection is not None and args.checkpoint_sha256 not in selection["test_checkpoint_hashes"]:
        raise ContractMismatch("checkpoint was not selected for test evaluation")
    z, all_ids, nrm, array_identity = load_arrays(
        Path(args.arrays), args.split, protocol["arrays"][args.split]["sha256"]
    )
    if array_identity != protocol["arrays"][args.split]:
        raise ContractMismatch("array identity does not match follow-up protocol")
    if args.offset >= len(z):
        raise ContractMismatch("chunk starts beyond dataset")
    stop = min(args.offset + args.limit, len(z))
    ids = all_ids[args.offset:stop]
    run_root = Path(args.run_root)
    meta = load_committed(run_root, generation=args.generation)
    if meta["files"].get("state.pt") != args.checkpoint_sha256:
        raise ContractMismatch("selected checkpoint hash mismatch")
    contract = meta["contract"]
    if contract.get("contract_version") != 2:
        raise ContractMismatch("historical checkpoint requires explicit verified migration")
    if (contract.get("source_hashes") != protocol["training_source_hashes"]
            or contract.get("data_hashes") != protocol["training_data_hashes"]
            or contract.get("normalization_hash") != array_identity["normalization_sha256"]):
        raise ContractMismatch("training source, data or normalization differs from protocol")
    state_path = run_root / f"gen_{args.generation:08d}" / "state.pt"
    payload = load_torch_payload(state_path.read_bytes(), map_location="cpu")
    if payload["contract"] != contract or payload["updates"] != args.generation:
        raise ContractMismatch("checkpoint payload and committed identity disagree")
    if contract["n_aux"] != 0 or contract["batch_size"] <= 0 or contract["n_examples"] <= 0:
        raise ContractMismatch("invalid model conditioning or training layout")
    steps_per_epoch = (contract["n_examples"] + contract["batch_size"] - 1) // contract["batch_size"]
    epochs, remainder = divmod(args.generation, steps_per_epoch)
    if (payload["epochs"] != epochs or payload["cursor"] != remainder * contract["batch_size"]
            or meta["updates"] != payload["updates"] or meta["epochs"] != payload["epochs"]
            or meta["cursor"] != payload["cursor"] or not 0 <= args.generation <= contract["max_updates"]):
        raise ContractMismatch("checkpoint counters do not match training layout")
    saved_normalizer = payload["normalization"]
    digest = normalization_hash(
        saved_normalizer["mean"].numpy(), saved_normalizer["scale"].numpy(),
        saved_normalizer["b_ref_sq"], saved_normalizer["n_bins"],
    )
    if digest != contract["normalization_hash"]:
        raise ContractMismatch("saved normalization differs from its contract")
    verify_source_snapshot(
        (run_root / "source_snapshot.zip").read_bytes(), contract["source_hashes"],
        contract["settings"]["source_snapshot_sha256"],
    )
    if any(not torch.is_tensor(value) or not torch.isfinite(value).all().item()
           for value in payload["model"].values()):
        raise ContractMismatch("checkpoint contains invalid model weights")
    configure_determinism()
    device = require_cuda_if_requested()
    model = ConditionalUNet(n_generated=contract["n_generated"], widths=tuple(contract["widths"]))
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    _, observed = physical_to_model(z[args.offset:stop], nrm, n_generated=contract["n_generated"])
    obs = torch.as_tensor(observed, device=device)
    mask = torch.ones(len(ids), 1, N_BINS, device=device)
    mask[:, :, HIDDEN_START:HIDDEN_END] = 0.0
    center = 0.5 * (N_BINS - 1)
    rel = torch.as_tensor(((np.arange(N_BINS, dtype=np.float32) - center) / center)[None, None, :], device=device)
    rel = rel.expand(len(ids), -1, -1)
    ensemble = generate_ensemble(
        model, obs, mask, rel, ids, args.split, args.eval_seed, args.n_eval,
        n_draws=N_DRAWS, n_generated=contract["n_generated"], sched=build_schedule(),
    )
    record = {
        "protocol_sha256": args.protocol_sha256, "arrays": array_identity,
        "checkpoint_sha256": args.checkpoint_sha256, "generation_updates": args.generation,
        "arm": contract["arm"], "seed": contract["seed"],
        "n_generated": contract["n_generated"], "n_params": sum(p.numel() for p in model.parameters()),
        "sampler": requested_sampler,
        "state_convention": STATE_CONVENTION, "noise_convention": NOISE_CONVENTION,
        "offset": args.offset, "count": len(ids),
        "inference_source_sha256": sha256_file(Path(__file__)),
        "device": str(device), "torch_version": str(torch.__version__),
    }
    publish_ensemble(out, ensemble, ids, record)
    print(json.dumps({"out": str(out), "shape": list(ensemble.shape), "committed": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
