"""Fresh-process CUDA recovery fixtures on owned synthetic runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys

import numpy as np
import torch

from prigm.arrays import physical_to_model
from prigm.checkpoint import load_torch_payload, normalization_hash, save_torch_payload
from prigm.loop import RecoverableTrainer, RunContract, configure_determinism
from prigm.normalize import fit_normalizer
from prigm.recovery import sha256_file
from prigm.toy import pooled_toy_observations
from prigm.toy_windows import make_toy_training
from prigm.train import require_cuda_if_requested

ARMS = {"D": (6, 0.0, None), "G5": (5, 0.0, None), "O-S": (6, 0.0005, "obs")}


def child(root, arm, target, mode):
    random.seed(7)
    configure_determinism()
    device = require_cuda_if_requested()
    if device.type != "cuda":
        raise RuntimeError("CUDA required")
    channels, c_r, residual = ARMS[arm]
    z, _, _ = make_toy_training(20, seed=11)
    nrm = fit_normalizer(pooled_toy_observations())
    u0, obs = physical_to_model(z, nrm, n_generated=channels)
    contract = RunContract(
        arm=arm, seed=7, n_generated=channels, n_aux=0, widths=(8, 16, 16),
        c_r=c_r, residual=residual, estimator="S", n_examples=20, batch_size=8,
        max_updates=8, source_hashes={"run_cuda_restart.py": sha256_file(Path(__file__))},
        normalization_hash=normalization_hash(nrm.mean, nrm.scale, nrm.b_ref_sq, nrm.n_bins),
        settings={"milestones": [3, 7]},
    )
    trainer = RecoverableTrainer(
        root, contract, u0, obs, None, nrm.mean, nrm.scale, nrm.b_ref_sq,
        device=device, normalization_n_bins=nrm.n_bins,
    )
    trainer.snapshot_every = 1
    if mode in ("kill_mid", "graceful"):
        original_update = trainer._one_update

        def update():
            result = original_update()
            if trainer.updates == 2:
                if mode == "graceful":
                    os.kill(os.getpid(), signal.SIGTERM)
                else:
                    print("FIXTURE_READY", flush=True)
                    sys.stdin.readline()
            return result

        trainer._one_update = update
    elif mode == "kill_publish":
        import prigm.recovery as recovery
        original_write = recovery.atomic_write_bytes

        def write(path, data):
            if Path(path).name == "COMMIT.json" and Path(path).parent.name == ".tmp_gen_00000003":
                print("FIXTURE_READY", flush=True)
                sys.stdin.readline()
            return original_write(path, data)

        recovery.atomic_write_bytes = write
    trainer.train(target, start_new=mode != "resume")
    if trainer.updates == 7:
        before = load_torch_payload(trainer.snapshot_payload()["state.pt"])
        next_output = trainer._one_update()
        probe = {
            "before": before,
            "next_losses": {k: float(next_output[k].detach().cpu()) for k in ("batch", "data", "phys")},
            "after": load_torch_payload(trainer.snapshot_payload()["state.pt"]),
        }
        with root.with_suffix(".probe.pt").open("xb") as handle:
            handle.write(save_torch_payload(probe))


def compare(left, right, path="state"):
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or not torch.equal(left, right):
            raise AssertionError(f"CUDA resume mismatch at {path}")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"dictionary keys differ at {path}")
        for key in left:
            compare(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"sequence lengths differ at {path}")
        for i, (a, b) in enumerate(zip(left, right)):
            compare(a, b, f"{path}[{i}]")
    elif left != right:
        raise AssertionError(f"CUDA resume mismatch at {path}: {left!r} != {right!r}")


def launch_child(root, arm, target, mode):
    command = [sys.executable, "-B", "-u", "-m", "prigm.run_cuda_restart", "--child-root", str(root),
               "--arm", arm, "--target", str(target), "--mode", mode]
    if mode.startswith("kill_"):
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        ready = False
        for line in process.stdout:
            if line.strip() == "FIXTURE_READY":
                ready = True
                process.kill()
                break
        code = process.wait(timeout=60)
        if not ready or code != -signal.SIGKILL:
            raise RuntimeError(f"owned crash fixture never reached injection point: {mode}, exit {code}")
    else:
        subprocess.run(command, check=True, timeout=120)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--child-root")
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--target", type=int, default=7)
    parser.add_argument("--mode", choices=("new", "resume", "kill_mid", "kill_publish", "graceful"), default="new")
    args = parser.parse_args(argv)
    if args.child_root:
        return child(Path(args.child_root), args.arm, args.target, args.mode)
    if args.output is None:
        parser.error("--output required")
    output = Path(args.output)
    parent = output.with_suffix(".runs")
    if output.exists() or parent.exists():
        raise FileExistsError(output if output.exists() else parent)
    parent.mkdir()
    records = []
    for arm in ARMS:
        baseline = parent / f"{arm}-baseline"
        launch_child(baseline, arm, 7, "new")
        expected = load_torch_payload(baseline.with_suffix(".probe.pt").read_bytes())
        for mode in ("target", "kill_mid", "kill_publish", "graceful"):
            root = parent / f"{arm}-{mode}"
            launch_child(root, arm, 3 if mode == "target" else 7, "new" if mode == "target" else mode)
            launch_child(root, arm, 7, "resume")
            actual = load_torch_payload(root.with_suffix(".probe.pt").read_bytes())
            compare(expected, actual)
            records.append({"arm": arm, "mode": mode, "exact_full_state_and_next_loss": True})
            print(json.dumps(records[-1]), flush=True)
    result = {
        "schema_version": 2, "arms": records, "all_match": True, "n_checks": len(records),
        "fresh_process": True, "examples": 20, "batch_size": 8, "final_batch": 4,
        "completed_updates": 7, "comparison": "exact parameters, optimizer, RNG, cursor, and next losses/update",
        "source_sha256": sha256_file(Path(__file__)),
        "scope": "owned synthetic CUDA fixtures; CPU unit tests separately cover concurrent writers and malformed contracts",
    }
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
