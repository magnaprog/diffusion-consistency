"""Coordinate the fixed-recipe PriGM follow-up after historical TEST exposure.

Publish a protocol that records the data and source hashes, four procedures,
training seeds, saved updates and sampling settings. The stages train the
models, evaluate validation predictions, choose checkpoints, and evaluate
TEST. Raw predictions and predictions after the physical clipping map select
checkpoints independently
by the mean of represented-month energy scores; an exact tie chooses the
earlier saved update.

Selection and TEST-start records bind subsequent work to those choices.
The coordinator lock and immutable completion records preserve the sequence
across staged execution. This procedure does not restore an unseen TEST set
or introduce a new coefficient search.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prigm.artifacts import (
    json_bytes, load_arrays, merge_ensembles, read_ensemble,
)
from prigm.barrier import load_protocol, require_test_started
from prigm.checkpoint import source_hashes
from prigm.config import FINAL_SEEDS, GAUSSIAN_JITTER, SAMPLE_CHUNK_WINDOWS, UNET_WIDTHS
from prigm.recovery import (
    ContractMismatch, PublicationError, RunLock, atomic_write_bytes, load_committed, sha256_file,
)
from prigm.scoring import month_balanced_mean

ARMS = ("D", "I-S", "O-S", "G5")
MILESTONES = (2868, 9560, 16252, 22944, 29636)


def fresh_json(path, value):
    """Publish under the coordinator lock without leaving a partial receipt."""
    path = Path(path)
    if path.exists():
        raise PublicationError(f"receipt exists: {path}")
    atomic_write_bytes(path, json_bytes(value))


def source_identity():
    return source_hashes(sorted((ROOT / "prigm").rglob("*.py")), relative_to=ROOT)


def preserve_uncommitted(root):
    """Keep an incomplete owned attempt before a sampling chunk is restarted.

    The caller holds the coordinator lock. If the directory has no committed
    ensemble generation, acquire its run lock, rename it to an unused
    .incomplete-N sibling and record why it was preserved. Leave completed
    generations in place. This retains the failed attempt for inspection before
    a deterministic redraw creates a new chunk.
    """
    if not root.exists() or (root / "gen_00000000/COMMIT.json").exists():
        return
    with RunLock(root):
        index = 1
        destination = root.with_name(f"{root.name}.incomplete-{index}")
        while destination.exists():
            index += 1
            destination = root.with_name(f"{root.name}.incomplete-{index}")
        root.rename(destination)
        fresh_json(destination / "PRESERVED.json", {
            "reason": "no committed ensemble generation; retained before deterministic redraw",
            "original": root.name,
        })


def launch(command, log, gpu):
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(ROOT),
        "PRIGM_REQUIRE_CUDA": "1", "PISCES_REQUIRE_CUDA": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1",
    })
    with Path(log).open("ab") as handle:
        handle.write((json.dumps(command) + "\n").encode())
        handle.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def base_protocol(study, arrays):
    identities = {split: load_arrays(arrays / f"{split.lower()}.npz", split)[3]
                  for split in ("TRAIN", "VAL", "TEST")}
    if len({row["normalization_sha256"] for row in identities.values()}) != 1:
        raise ContractMismatch("all splits must use identical training normalization")
    return {
        "schema_version": 2, "exposure": "historical_test_exposed",
        "study": "fixed_recipe_corrected_sampling_followup",
        "arrays": identities, "array_root": str(arrays.resolve()),
        "training_source_hashes": source_identity(), "inference_source_hashes": source_identity(),
        "training_data_hashes": {"arrays_sha256": identities["TRAIN"]["sha256"]},
        "arms": list(ARMS), "seeds": list(FINAL_SEEDS), "milestones": list(MILESTONES),
        "c_r": {arm: 0.0005 if arm in ("I-S", "O-S") else 0.0 for arm in ARMS},
        "batch_size": 128, "budget_updates": MILESTONES[-1],
        "samplers": {
            "VAL": {"n_eval": 25, "n_draws": 39, "eval_seed": 7},
            "TEST": {"n_eval": 50, "n_draws": 39, "eval_seed": 7},
        },
        "sampling_chunk_windows": SAMPLE_CHUNK_WINDOWS,
        "controls": {
            "interpolation": {"conditioning": "two_boundaries"},
            "persistence": {"conditioning": "left_boundary"},
            "gaussian": {"rho": 0.0, "jitter": GAUSSIAN_JITTER, "seed": 13},
        },
        "selection": "minimum validation month-balanced primary energy, earlier milestone on exact tie; raw and support-corrected selected independently",
        "claim_scope": "post-exposure corrected procedure; no restored blindness or new coefficient search",
        "coordinator_sha256": sha256_file(Path(__file__)),
    }


def verify_checkpoint(study, name, generation, protocol):
    root = study / "runs" / name
    meta = load_committed(root, generation=generation)
    contract = meta["contract"]
    arm, seed = name.rsplit("_s", 1)
    if (arm not in protocol["arms"] or int(seed) not in protocol["seeds"]
            or contract["arm"] != arm or contract["seed"] != int(seed)
            or contract["c_r"] != protocol["c_r"][arm]
            or contract["n_generated"] != (5 if arm == "G5" else 6)
            or contract["batch_size"] != protocol["batch_size"]
            or contract.get("contract_version") != 2 or contract.get("n_aux") != 0
            or contract.get("widths") != list(UNET_WIDTHS)
            or contract.get("n_examples") != protocol["arrays"]["TRAIN"]["n_windows"]
            or contract.get("estimator") != "S"
            or contract.get("residual") != {"D": None, "G5": None, "I-S": "inst", "O-S": "obs"}[arm]
            or contract["settings"]["milestones"] != protocol["milestones"]):
        raise ContractMismatch(f"arm, seed or training recipe mismatch: {name}")
    if (contract.get("source_hashes") != protocol["training_source_hashes"]
            or contract.get("data_hashes") != protocol["training_data_hashes"]
            or contract.get("normalization_hash") != protocol["arrays"]["TRAIN"]["normalization_sha256"]
            or contract.get("max_updates") != protocol["budget_updates"]):
        raise ContractMismatch(f"checkpoint does not match study: {name}/{generation}")
    return meta


def train(study, protocol, gpus, target):
    if target not in protocol["milestones"]:
        raise ContractMismatch("training target must be a frozen common milestone")

    def arm_worker(arm, gpu):
        for seed in protocol["seeds"]:
            name = f"{arm}_s{seed}"
            root = study / "runs" / name
            if (root / f"gen_{target:08d}/COMMIT.json").is_file():
                verify_checkpoint(study, name, target, protocol)
                continue
            command = [sys.executable, "-B", "-u", "-m", "prigm.run_fit",
                       "--run-root", str(root), "--arm", arm, "--seed", str(seed),
                       "--c-r", str(protocol["c_r"][arm]), "--updates", str(target),
                       "--budget-updates", str(protocol["budget_updates"]),
                       "--milestones", ",".join(map(str, protocol["milestones"])),
                       "--arrays", str(Path(protocol["array_root"]) / "train.npz"),
                       "--batch-size", str(protocol["batch_size"])]
            if root.exists():
                # The trainer checks any fallback generation and holds its own lifetime lock.
                load_committed(root)
                command.append("--resume")
            launch(command, study / "logs" / f"train-{name}.log", gpu)
            verify_checkpoint(study, name, target, protocol)

    def device_worker(index, gpu):
        for arm in ARMS[index::len(gpus)]:
            arm_worker(arm, gpu)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(device_worker, i, gpu) for i, gpu in enumerate(gpus)]
        for future in futures:
            future.result()


def evaluation_jobs(study, protocol, split):
    if split == "VAL":
        return [(f"{arm}_s{seed}", gen) for arm in ARMS for seed in FINAL_SEEDS for gen in MILESTONES]
    receipt = json.loads((study / "SELECTION.json").read_text())
    return [(name, generation) for name, generation in receipt["test_jobs"]]


def validate_ensemble_identity(meta, sampled_ids, ids, identity, checkpoint, protocol, protocol_hash, split):
    contract = checkpoint["contract"]
    if (sampled_ids != ids or meta.get("arrays") != identity
            or meta.get("checkpoint_sha256") != checkpoint["files"]["state.pt"]
            or meta.get("protocol_sha256") != protocol_hash
            or meta.get("sampler") != protocol["samplers"][split]
            or meta.get("arm") != contract["arm"] or meta.get("seed") != contract["seed"]
            or meta.get("n_generated") != contract["n_generated"]
            or meta.get("generation_updates") != checkpoint["generation"]):
        raise ContractMismatch("existing ensemble differs from expected contract")


def validate_score(path, merged, ids, identity, checkpoint, protocol_hash, corrected):
    row = json.loads(path.read_text())
    values = np.asarray(row["scores"], dtype=np.float64)
    months = np.array([int(w[:4]) * 100 + int(w[5:7]) for w in ids], dtype=np.int64)
    if (row.get("schema_version") != 2 or row.get("procedure") != "diffusion"
            or row.get("arrays") != identity or row.get("ids") != ids
            or type(row.get("n")) is not int or row["n"] != len(ids)
            or values.shape != (len(ids),) or not np.isfinite(values).all()
            or row.get("months") != months.tolist()
            or row.get("corrected") is not corrected
            or row.get("checkpoint_sha256") != checkpoint["files"]["state.pt"]
            or row.get("protocol_sha256") != protocol_hash
            or row.get("ensemble_commit_sha256") != sha256_file(merged / "gen_00000000/COMMIT.json")
            or not math.isfinite(row["month_balanced"]) or not math.isfinite(row["mean"])
            or not np.isclose(row["month_balanced"], month_balanced_mean(values, months), rtol=1e-12, atol=0)
            or not np.isclose(row["mean"], values.mean(), rtol=1e-12, atol=0)):
        raise ContractMismatch("score values or provenance differ from the expected population")
    return row


def evaluate(study, protocol, protocol_hash, gpus, split):
    require_test_started(study, split, protocol_hash)
    arrays = Path(protocol["array_root"]) / f"{split.lower()}.npz"
    _, ids, _, identity = load_arrays(arrays, split, protocol["arrays"][split]["sha256"])
    jobs = evaluation_jobs(study, protocol, split)

    def evaluate_one(name, generation, gpu):
        meta = verify_checkpoint(study, name, generation, protocol)
        checkpoint_hash = meta["files"]["state.pt"]
        parent = study / "artifacts" / split.lower() / f"{name}-g{generation}"
        parent.mkdir(parents=True, exist_ok=True)
        roots = []
        for offset in range(0, len(ids), SAMPLE_CHUNK_WINDOWS):
            chunk = parent / f"chunk-{offset:04d}"
            preserve_uncommitted(chunk)
            expected_ids = ids[offset:offset + SAMPLE_CHUNK_WINDOWS]
            if chunk.exists():
                _, sampled_ids, info = read_ensemble(chunk)
                validate_ensemble_identity(info, sampled_ids, expected_ids, identity, meta,
                                           protocol, protocol_hash, split)
            else:
                sampler = protocol["samplers"][split]
                launch([
                    sys.executable, "-B", "-u", "-m", "prigm.run_eval",
                    "--run-root", str(study / "runs" / name), "--generation", str(generation),
                    "--checkpoint-sha256", checkpoint_hash, "--out", str(chunk),
                    "--arrays", str(arrays), "--split", split, "--n-eval", str(sampler["n_eval"]),
                    "--eval-seed", str(sampler["eval_seed"]), "--offset", str(offset),
                    "--limit", str(SAMPLE_CHUNK_WINDOWS), "--study-root", str(study),
                    "--protocol-sha256", protocol_hash,
                ], study / "logs" / f"sample-{split}-{name}-g{generation}.log", gpu)
            roots.append(chunk)
        merged = parent / "ensemble"
        preserve_uncommitted(merged)
        if merged.exists():
            _, sampled_ids, info = read_ensemble(merged)
            validate_ensemble_identity(info, sampled_ids, ids, identity, meta, protocol, protocol_hash, split)
            expected_chunks = {str(root.resolve()): sha256_file(root / "gen_00000000/COMMIT.json") for root in roots}
            if info.get("chunks") != expected_chunks:
                raise ContractMismatch("merged ensemble does not bind the verified chunk inventory")
        else:
            merge_ensembles(roots, merged, ids)
        for corrected in (False, True):
            output = parent / ("score-corrected.json" if corrected else "score-raw.json")
            if output.exists():
                validate_score(output, merged, ids, identity, meta, protocol_hash, corrected)
                continue
            command = [sys.executable, "-B", "-m", "prigm.run_score", "--ensemble", str(merged),
                       "--arrays", str(arrays), "--output", str(output), "--split", split,
                       "--study-root", str(study), "--protocol-sha256", protocol_hash]
            if corrected:
                command.append("--correct")
            launch(command, study / "logs" / f"score-{split}-{name}-g{generation}.log", gpu)

    def worker(index, gpu):
        for name, generation in jobs[index::len(gpus)]:
            evaluate_one(name, generation, gpu)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, i, gpu) for i, gpu in enumerate(gpus)]
        for future in futures:
            future.result()


def select(study, protocol, protocol_hash):
    if (study / "TEST_STARTED.json").exists():
        require_test_started(study, "TEST", protocol_hash)
    _, ids, _, identity = load_arrays(
        Path(protocol["array_root"]) / "val.npz", "VAL", protocol["arrays"]["VAL"]["sha256"]
    )
    if identity != protocol["arrays"]["VAL"]:
        raise ContractMismatch("validation identity differs from protocol")
    selected, jobs, hashes, score_hashes = {}, set(), set(), {}
    verified_ensembles = set()
    for arm in ARMS:
        for seed in FINAL_SEEDS:
            name = f"{arm}_s{seed}"
            selected[name] = {}
            for kind in ("raw", "corrected"):
                values = []
                for gen in MILESTONES:
                    parent = study / "artifacts/val" / f"{name}-g{gen}"
                    path = parent / f"score-{kind}.json"
                    checkpoint = verify_checkpoint(study, name, gen, protocol)
                    merged = parent / "ensemble"
                    if merged not in verified_ensembles:
                        _, sampled_ids, info = read_ensemble(merged)
                        validate_ensemble_identity(info, sampled_ids, ids, identity, checkpoint,
                                                   protocol, protocol_hash, "VAL")
                        verified_ensembles.add(merged)
                    score = validate_score(path, merged, ids, identity, checkpoint, protocol_hash,
                                           kind == "corrected")
                    values.append((score["month_balanced"], gen, score["checkpoint_sha256"]))
                    score_hashes[str(path.relative_to(study))] = sha256_file(path)
                value, gen, digest = min(values)
                selected[name][kind] = {"generation": gen, "score": value, "checkpoint_sha256": digest}
                jobs.add((name, gen))
                hashes.add(digest)
    selection = {
        "schema_version": 2, "protocol_sha256": protocol_hash, "selected": selected,
        "test_jobs": sorted(jobs), "test_checkpoint_hashes": sorted(hashes),
        "validation_score_sha256": score_hashes,
        "exposure": "historical_test_exposed",
        "scope": "fresh fixed-recipe runs; historical checkpoints require separately labeled exploratory evaluation",
    }
    selection_path = study / "SELECTION.json"
    if selection_path.exists():
        if selection_path.read_bytes() != json_bytes(selection):
            raise ContractMismatch("published selection differs from verified validation results")
    else:
        fresh_json(selection_path, selection)
    if not (study / "TEST_STARTED.json").exists():
        fresh_json(study / "TEST_STARTED.json", {
            "phase": "TEST_STARTED", "previous_test_exposure": True,
            "protocol_sha256": protocol_hash, "selection_sha256": sha256_file(selection_path),
        })
    require_test_started(study, "TEST", protocol_hash)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("protocol", "train", "validation", "select", "test"))
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--arrays-root")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--target", type=int, default=MILESTONES[-1])
    parser.add_argument("--protocol-sha256")
    args = parser.parse_args(argv)
    study = Path(args.study_root).resolve()
    if not study.is_dir():
        raise PublicationError("create and verify an owned study root before execution")
    gpus = [int(gpu) for gpu in args.gpus.split(",")]
    if len(gpus) != len(set(gpus)) or not gpus or any(gpu < 0 for gpu in gpus):
        raise ValueError("GPU IDs must be distinct nonnegative integers")
    with RunLock(study / "coordinator"):
        if args.stage == "protocol":
            if args.arrays_root is None:
                parser.error("protocol requires --arrays-root")
            fresh_json(study / "PROTOCOL.json", base_protocol(study, Path(args.arrays_root)))
            print(sha256_file(study / "PROTOCOL.json"))
            return 0
        if not args.protocol_sha256:
            parser.error("execution requires --protocol-sha256")
        protocol = load_protocol(study, args.protocol_sha256)
        if (source_identity() != protocol["inference_source_hashes"]
                or sha256_file(Path(__file__)) != protocol["coordinator_sha256"]):
            raise ContractMismatch("execution source differs from frozen protocol")
        if args.stage == "train":
            train(study, protocol, gpus, args.target)
        elif args.stage in ("validation", "test"):
            evaluate(study, protocol, args.protocol_sha256, gpus, "VAL" if args.stage == "validation" else "TEST")
        else:
            select(study, protocol, args.protocol_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
