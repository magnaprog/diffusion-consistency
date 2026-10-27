"""Predetermined 29636-update coefficient response, with no VAL selection.

Create/verify an empty owned study root externally. Publish a protocol, run
profile-train (four ordinary resumable fits), then train, freeze-endpoints,
sample, aggregate. Admission and throughput estimation belong to the caller.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Event
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_prigm_repair import fresh_json, source_identity, validate_ensemble_identity
from prigm.artifacts import json_bytes, load_arrays, merge_ensembles, read_ensemble
from prigm.barrier import load_protocol, require_test_started
from prigm.checkpoint import load_torch_payload, normalization_hash, verify_source_snapshot
from prigm.config import RECOVERY_EVERY, UNET_WIDTHS
from prigm.recovery import ContractMismatch, PublicationError, RunLock, load_committed, sha256_file
from prigm.run_score import decode_ensemble
from prigm.run_secondaries import summarize_diagnostics, window_diagnostics
from prigm.observation import closure_correct

ORIGINAL_PROTOCOL = "25d0cdefdd5dbb160c32c7b5f4a6053bab99e4a9f64832d9eb6502232bf24e6b"
MILESTONES = (2868, 9560, 16252, 22944, 29636)
FINAL = MILESTONES[-1]
TEST_WINDOWS = 480
SEEDS = (7, 13, 42)
COEFFICIENTS = ("0", "0.00005", "0.0005", "0.005")
SAMPLERS = {"VAL": {"n_eval": 25, "n_draws": 39, "eval_seed": 7},
            "TEST": {"n_eval": 50, "n_draws": 39, "eval_seed": 7}}
SELECTION_RULE = "predetermined_final_update_29636_not_VAL_selected"
METRICS = ("es5", "r_obs_mean", "frac_support_inequality_violation",
           "frac_support_violation_beyond_rounding", "cov90_mean", "width90_mean")


class DeadlineReached(TimeoutError):
    pass


def parse_deadline(value):
    try:
        result = float(value)
    except ValueError:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.utcoffset() is None:
            raise ValueError("deadline needs an explicit UTC offset")
        result = dt.timestamp()
    if not math.isfinite(result):
        raise ValueError("deadline must be finite")
    return result


def remaining(deadline):
    seconds = deadline - time.time()
    if seconds <= 0:
        raise DeadlineReached("absolute deadline reached; partial artifacts retained")
    return seconds


def producers():
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in
            (Path(__file__).resolve(), ROOT / "experiments/run_prigm_repair.py")}


def fixed_jobs(study, original):
    jobs = []
    for coefficient in COEFFICIENTS:
        for seed in SEEDS:
            arm = "D" if coefficient == "0" else "O-S"
            new_fit = coefficient in ("0.00005", "0.005")
            reuse = (coefficient, seed) in (("0", 7), ("0.0005", 7), ("0.0005", 13))
            name = f"c{coefficient}_s{seed}"
            run = study / "runs" / name if new_fit else original / "runs" / f"{arm}_s{seed}"
            ensemble = (original / "artifacts/test" / f"{arm}_s{seed}-g{FINAL}" / "ensemble"
                        if reuse else study / "artifacts/test" / name / "ensemble")
            jobs.append({"id": name, "arm": arm, "seed": seed, "c_r": float(coefficient),
                         "generation": FINAL, "new_fit": new_fit, "reuse": reuse,
                         "run_root": str(run), "ensemble_root": str(ensemble)})
    return jobs


def exact_json(path, value):
    if path.exists():
        if path.read_bytes() != json_bytes(value):
            raise ContractMismatch(f"immutable receipt differs: {path}")
    else:
        fresh_json(path, value)


def owned_path(study, path):
    """Do not follow substituted output directories or write outside the owned root."""
    path = Path(path)
    if not path.is_relative_to(study) or path.resolve() != path:
        raise ContractMismatch(f"noncanonical owned path: {path}")
    return path


def recipe(settings):
    # Runtime environment is recorded in every checkpoint, but is not a hyperparameter.
    return {k: v for k, v in settings.items() if k != "environment"}


def verify_checkpoint(job, protocol, generation=FINAL):
    """Check actual coefficient and payload, not the arm-only run_eval gate."""
    root = Path(job["run_root"])
    meta = load_committed(root, generation=generation)
    c = meta["contract"]
    expected = {
        "contract_version": 2, "arm": job["arm"], "seed": job["seed"], "c_r": job["c_r"],
        "n_generated": 6, "n_aux": 0, "widths": list(UNET_WIDTHS), "estimator": "S",
        "residual": None if job["arm"] == "D" else "obs", "batch_size": 128,
        "n_examples": protocol["arrays"]["TRAIN"]["n_windows"], "max_updates": FINAL,
        "source_hashes": protocol["training_source_hashes"],
        "data_hashes": protocol["training_data_hashes"],
        "normalization_hash": protocol["arrays"]["TRAIN"]["normalization_sha256"],
    }
    if any(c.get(k) != v for k, v in expected.items()):
        raise ContractMismatch(f"coefficient, seed, source, data or recipe mismatch: {job['id']}")
    settings = c["settings"]
    if (settings.get("milestones") != list(MILESTONES)
            or settings.get("recovery_every") != RECOVERY_EVERY
            or settings.get("data") != {"kind": "real_npz", "ids_covered_by_arrays_hash": True}
            or ("training_settings" in protocol and recipe(settings) != protocol["training_settings"])):
        raise ContractMismatch("checkpoint optimizer/precision/recovery recipe differs")
    folder = root / f"gen_{generation:08d}"
    if (set(meta["files"]) != {"state.pt", "meta.json"}
            or json.loads((folder / "meta.json").read_text()) != c):
        raise ContractMismatch("checkpoint metadata inventory differs")
    payload = load_torch_payload((folder / "state.pt").read_bytes())
    epochs, step = divmod(generation, (c["n_examples"] + 127) // 128)
    if (payload["contract"] != c or not 0 <= generation <= FINAL
            or any(payload[k] != v or meta[k] != v for k, v in
                   (("updates", generation), ("epochs", epochs), ("cursor", step * 128)))):
        raise ContractMismatch("checkpoint payload/final endpoint counters differ")
    nrm = payload["normalization"]
    if normalization_hash(nrm["mean"], nrm["scale"], nrm["b_ref_sq"], nrm["n_bins"]) != c["normalization_hash"]:
        raise ContractMismatch("checkpoint payload normalization differs")
    verify_source_snapshot((root / "source_snapshot.zip").read_bytes(), c["source_hashes"],
                           settings["source_snapshot_sha256"])
    import torch
    from prigm.model import ConditionalUNet
    with torch.device("meta"):
        shapes = ConditionalUNet(n_generated=6, widths=UNET_WIDTHS).state_dict()
    weights = payload["model"]
    if (weights.keys() != shapes.keys() or any(
            not torch.is_tensor(v) or v.shape != shapes[k].shape or v.dtype != shapes[k].dtype
            or not torch.isfinite(v).all().item() for k, v in weights.items())):
        raise ContractMismatch("checkpoint model architecture/weights differ")
    return meta


def checkpoint_binding(job, checkpoint):
    root = Path(job["run_root"])
    return {"run_root": str(root), "generation": FINAL,
            "checkpoint_sha256": checkpoint["files"]["state.pt"],
            "commit_sha256": sha256_file(root / f"gen_{FINAL:08d}/COMMIT.json"),
            "source_snapshot_sha256": sha256_file(root / "source_snapshot.zip")}


def checked_ensemble(path, job, checkpoint, protocol, digest, ids):
    ensemble, sampled_ids, meta = read_ensemble(path)
    validate_ensemble_identity(meta, sampled_ids, ids, protocol["arrays"]["TEST"],
                               checkpoint, protocol, digest, "TEST")
    if (meta.get("procedure", "diffusion") != "diffusion"
            or meta.get("inference_source_sha256") != protocol["inference_source_hashes"]["prigm/run_eval.py"]):
        raise ContractMismatch("ensemble inference producer differs")
    return ensemble, meta


def original_inputs(protocol, ids):
    original = Path(protocol["original_root"])
    old = load_protocol(original, ORIGINAL_PROTOCOL)
    selection = require_test_started(original, "TEST", ORIGINAL_PROTOCOL)
    if (old["arrays"] != protocol["arrays"] or old["samplers"] != SAMPLERS
            or old["training_source_hashes"] != protocol["training_source_hashes"]
            or old["inference_source_hashes"] != protocol["inference_source_hashes"]):
        raise ContractMismatch("original protocol source/data/sampler differs")
    bindings = {"receipts": {name: sha256_file(original / name) for name in
                            ("PROTOCOL.json", "SELECTION.json", "TEST_STARTED.json")},
                "checkpoints": {}, "ensembles": {}}
    for job in protocol["jobs"]:
        if job["new_fit"]:
            continue
        checkpoint = verify_checkpoint(job, protocol)
        bindings["checkpoints"][job["id"]] = checkpoint_binding(job, checkpoint)
        if not job["reuse"]:
            continue
        name = f"{job['arm']}_s{job['seed']}"
        kind = "corrected" if job["arm"] == "D" else "raw"
        chosen = selection["selected"][name][kind]
        if (chosen["generation"] != FINAL or chosen["checkpoint_sha256"] != checkpoint["files"]["state.pt"]
                or [name, FINAL] not in selection["test_jobs"]
                or chosen["checkpoint_sha256"] not in selection["test_checkpoint_hashes"]):
            raise ContractMismatch("original reuse is not the expected final endpoint")
        path = Path(job["ensemble_root"])
        _, meta = checked_ensemble(path, job, checkpoint, old, ORIGINAL_PROTOCOL, ids)
        bindings["ensembles"][job["id"]] = {
            "ensemble_root": str(path), "commit_sha256": sha256_file(path / "gen_00000000/COMMIT.json"),
            "metadata": meta, "protocol_sha256": ORIGINAL_PROTOCOL,
        }
    return bindings


def base_protocol(study, original, arrays, deadline):
    if study == original or study.is_relative_to(original) or original.is_relative_to(study):
        raise ContractMismatch("owned and original roots must be disjoint")
    old = load_protocol(original, ORIGINAL_PROTOCOL)
    identities = {split: load_arrays(arrays / f"{split.lower()}.npz", split)[3]
                  for split in ("TRAIN", "VAL", "TEST")}
    if (identities != old["arrays"] or identities["TEST"]["n_windows"] != TEST_WINDOWS
            or len({v["normalization_sha256"] for v in identities.values()}) != 1
            or source_identity() != old["training_source_hashes"]
            or source_identity() != old["inference_source_hashes"]
            or old["batch_size"] != 128 or old["milestones"] != list(MILESTONES)
            or old["budget_updates"] != FINAL or old["sampling_chunk_windows"] != 8):
        raise ContractMismatch("original array/source/fixed-recipe contract differs")
    protocol = {
        "schema_version": 2, "exposure": "historical_test_exposed",
        "study": "fixed_final_coefficient_response", "study_root": str(study),
        "original_root": str(original), "original_protocol_sha256": ORIGINAL_PROTOCOL,
        "array_root": str(arrays), "arrays": identities,
        "training_source_hashes": old["training_source_hashes"],
        "inference_source_hashes": old["inference_source_hashes"],
        "training_data_hashes": {"arrays_sha256": identities["TRAIN"]["sha256"]},
        "producers": producers(), "milestones": list(MILESTONES), "batch_size": 128,
        "budget_updates": FINAL, "samplers": SAMPLERS, "sampling_chunk_windows": 8,
        "jobs": fixed_jobs(study, original), "selection": SELECTION_RULE,
        "deadline_epoch": deadline, "claim_scope": "exposed equal-update sensitivity; no coefficient selection",
    }
    reference = verify_checkpoint(protocol["jobs"][0], protocol)
    protocol["training_settings"] = recipe(reference["contract"]["settings"])
    _, ids, _, _ = load_arrays(arrays / "test.npz", "TEST")
    protocol["original_inputs"] = original_inputs(protocol, ids)
    remaining(deadline)
    return protocol


def verify_protocol(study, digest):
    protocol = load_protocol(study, digest)
    if (protocol.get("study_root") != str(study)
            or protocol.get("original_protocol_sha256") != ORIGINAL_PROTOCOL
            or protocol.get("jobs") != fixed_jobs(study, Path(protocol["original_root"]))
            or protocol.get("selection") != SELECTION_RULE or protocol.get("samplers") != SAMPLERS
            or protocol.get("milestones") != list(MILESTONES) or protocol.get("budget_updates") != FINAL
            or protocol.get("batch_size") != 128 or protocol.get("sampling_chunk_windows") != 8
            or protocol.get("producers") != producers()
            or protocol["training_source_hashes"] != source_identity()
            or protocol["inference_source_hashes"] != source_identity()):
        raise ContractMismatch("fixed design, root or execution source differs from protocol")
    for job in protocol["jobs"]:
        if job["new_fit"]:
            owned_path(study, job["run_root"])
        if not job["reuse"]:
            owned_path(study, job["ensemble_root"])
    for split, identity in protocol["arrays"].items():
        actual = load_arrays(Path(protocol["array_root"]) / f"{split.lower()}.npz", split, identity["sha256"])[3]
        if actual != identity:
            raise ContractMismatch("array identity differs from protocol")
    return protocol


def endpoint_selection(protocol, digest, ids):
    if protocol["jobs"] != fixed_jobs(Path(protocol["study_root"]), Path(protocol["original_root"])):
        raise ContractMismatch("endpoint authorization requires the complete fixed grid")
    if original_inputs(protocol, ids) != protocol["original_inputs"]:
        raise ContractMismatch("original input bridge changed")
    endpoints = {job["id"]: checkpoint_binding(job, verify_checkpoint(job, protocol))
                 for job in protocol["jobs"]}
    return {"schema_version": 2, "protocol_sha256": digest, "selection": SELECTION_RULE,
            "exposure": "historical_test_exposed", "endpoints": endpoints,
            "test_jobs": [[j["id"], FINAL] for j in protocol["jobs"] if not j["reuse"]],
            "test_checkpoint_hashes": sorted({v["checkpoint_sha256"] for v in endpoints.values()})}


def freeze_endpoints(study, protocol, digest, deadline=None):
    deadline = protocol["deadline_epoch"] if deadline is None else min(deadline, protocol["deadline_epoch"])
    remaining(deadline)
    _, ids, _, _ = load_arrays(Path(protocol["array_root"]) / "test.npz", "TEST")
    value = endpoint_selection(protocol, digest, ids)
    remaining(deadline)
    selection = owned_path(study, study / "SELECTION.json")
    phase = owned_path(study, study / "TEST_STARTED.json")
    if phase.exists():
        require_test_started(study, "TEST", digest)
    exact_json(selection, value)
    remaining(deadline)
    exact_json(phase, {"phase": "TEST_STARTED", "previous_test_exposure": True,
                       "protocol_sha256": digest, "selection_sha256": sha256_file(selection),
                       "selection": SELECTION_RULE})
    require_test_started(study, "TEST", digest)


def authorize_test(study, protocol, digest, ids):
    selection = require_test_started(study, "TEST", digest)
    if selection != endpoint_selection(protocol, digest, ids):
        raise ContractMismatch("selection differs from complete predetermined endpoints")
    return selection


def launch(command, study, name, gpu, deadline):
    """Bound only this owned child; time startup, verification and publication too."""
    remaining(deadline)
    token = f"{name}-{uuid4().hex}"
    logs = owned_path(study, study / "logs")
    logs.mkdir(exist_ok=True)
    start, tick = time.time(), time.monotonic()
    receipt = {"command": command, "gpu": gpu, "started_epoch": start, "deadline_epoch": deadline}
    fresh_json(logs / f"{token}.started.json", receipt)
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(ROOT),
                "PRIGM_REQUIRE_CUDA": "1", "PISCES_REQUIRE_CUDA": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"})
    status, returncode = "failed", None
    try:
        with (logs / f"{token}.log").open("xb") as handle:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                                    stderr=subprocess.STDOUT, check=True, timeout=remaining(deadline))
        returncode, status = result.returncode, "complete"
    except (subprocess.TimeoutExpired, DeadlineReached) as exc:
        status = "deadline"
        raise DeadlineReached("owned child deadline; committed/partial outputs retained") from exc
    except subprocess.CalledProcessError as exc:
        returncode = exc.returncode
        raise
    finally:
        fresh_json(logs / f"{token}.finished.json", {
            **receipt, "finished_epoch": time.time(), "wall_seconds": time.monotonic() - tick,
            "status": status, "returncode": returncode})


def work_queue(jobs, gpus, deadline, action):
    if not 1 <= len(gpus) <= 4 or len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus):
        raise ValueError("use one to four distinct nonnegative GPU IDs")
    remaining(deadline)
    queue, stop = Queue(), Event()
    for job in jobs:
        queue.put(job)

    def worker(gpu):
        while not stop.is_set():
            try:
                remaining(deadline)
                job = queue.get_nowait()
            except Empty:
                return
            try:
                action(job, gpu)
            except BaseException:
                stop.set()
                raise
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in gpus]
        for future in futures:
            future.result()


def training_command(job, protocol, target, resume=False):
    if not job["new_fit"] or target not in (MILESTONES[0], FINAL):
        raise ContractMismatch("only six declared fits and two stage targets are permitted")
    command = [sys.executable, "-B", "-u", "-m", "prigm.run_fit", "--run-root", job["run_root"],
               "--arm", "O-S", "--c-r", str(job["c_r"]), "--seed", str(job["seed"]),
               "--updates", str(target), "--budget-updates", str(FINAL),
               "--milestones", ",".join(map(str, MILESTONES)), "--batch-size", "128",
               "--arrays", str(Path(protocol["array_root"]) / "train.npz")]
    return command + (["--resume"] if resume else [])


def train(study, protocol, gpus, deadline, profile=False):
    remaining(deadline)
    _, ids, _, _ = load_arrays(Path(protocol["array_root"]) / "test.npz", "TEST")
    if original_inputs(protocol, ids) != protocol["original_inputs"]:
        raise ContractMismatch("original input bridge changed before fitting")
    jobs = [j for j in protocol["jobs"] if j["new_fit"] and (not profile or j["seed"] in (7, 13))]
    target = MILESTONES[0] if profile else FINAL

    def action(job, gpu):
        root = owned_path(study, job["run_root"])
        if (root / f"gen_{target:08d}/COMMIT.json").exists():
            verify_checkpoint(job, protocol, target)
            return
        if root.exists():
            meta = load_committed(root)
            verify_checkpoint(job, protocol, meta["generation"])
            if meta["generation"] > target:
                raise ContractMismatch("cannot rewind a training stage")
        launch(training_command(job, protocol, target, root.exists()), study,
               f"train-{job['id']}-g{target}", gpu, deadline)
        verify_checkpoint(job, protocol, target)
    work_queue(jobs, gpus, deadline, action)


def chunks_for(job, ids):
    parent = Path(job["ensemble_root"]).parent
    return [(offset, parent / f"chunk-{offset:04d}", ids[offset:offset + 8])
            for offset in range(0, len(ids), 8)]


def complete_ensemble(job, checkpoint, protocol, digest, ids):
    roots = []
    for _, path, subset in chunks_for(job, ids):
        checked_ensemble(path, job, checkpoint, protocol, digest, subset)
        roots.append(path)
    merged = Path(job["ensemble_root"])
    inventory = {str(p): sha256_file(p / "gen_00000000/COMMIT.json") for p in roots}
    if not merged.exists():
        merge_ensembles(roots, merged, ids)
    _, meta = checked_ensemble(merged, job, checkpoint, protocol, digest, ids)
    if meta.get("chunks") != inventory:
        raise ContractMismatch("merged ensemble differs from complete chunk inventory")
    return meta


def sample(study, protocol, digest, gpus, deadline):
    _, ids, _, _ = load_arrays(Path(protocol["array_root"]) / "test.npz", "TEST")
    authorize_test(study, protocol, digest, ids)
    checkpoints = {j["id"]: verify_checkpoint(j, protocol) for j in protocol["jobs"] if not j["reuse"]}
    tasks = [(j, offset, path, subset) for j in protocol["jobs"] if not j["reuse"]
             for offset, path, subset in chunks_for(j, ids)]

    def action(task, gpu):
        job, offset, path, subset = task
        owned_path(study, path)
        checkpoint = checkpoints[job["id"]]
        if not path.exists():
            command = [sys.executable, "-B", "-u", "-m", "prigm.run_eval",
                       "--run-root", job["run_root"], "--generation", str(FINAL),
                       "--checkpoint-sha256", checkpoint["files"]["state.pt"], "--out", str(path),
                       "--arrays", str(Path(protocol["array_root"]) / "test.npz"), "--split", "TEST",
                       "--n-eval", "50", "--eval-seed", "7", "--offset", str(offset), "--limit", "8",
                       "--study-root", str(study), "--protocol-sha256", digest]
            launch(command, study, f"sample-{job['id']}-{offset:04d}", gpu, deadline)
        # Existing incomplete roots fail closed, never deleted or renamed implicitly.
        checked_ensemble(path, job, checkpoint, protocol, digest, subset)
    work_queue(tasks, gpus, deadline, action)
    for job in protocol["jobs"]:
        if not job["reuse"]:
            remaining(deadline)
            complete_ensemble(job, checkpoints[job["id"]], protocol, digest, ids)


def aggregate(study, protocol, digest, deadline):
    z, ids, nrm, _ = load_arrays(Path(protocol["array_root"]) / "test.npz", "TEST")
    selection = authorize_test(study, protocol, digest, ids)
    # Validate all twelve full populations before calculating any metrics.
    inputs = {}
    for job in protocol["jobs"]:
        remaining(deadline)
        checkpoint = verify_checkpoint(job, protocol)
        own_digest = ORIGINAL_PROTOCOL if job["reuse"] else digest
        if not job["reuse"]:
            complete_ensemble(job, checkpoint, protocol, digest, ids)
        _, meta = checked_ensemble(Path(job["ensemble_root"]), job, checkpoint, protocol, own_digest, ids)
        inputs[job["id"]] = {"protocol_sha256": own_digest, "metadata": meta,
                            "ensemble_root": job["ensemble_root"],
                            "ensemble_commit_sha256": sha256_file(Path(job["ensemble_root"]) / "gen_00000000/COMMIT.json")}
    months = [int(w[:4]) * 100 + int(w[5:7]) for w in ids]
    rows = []
    for job in protocol["jobs"]:
        ensemble, _, _ = read_ensemble(Path(job["ensemble_root"]))
        results = {}
        for corrected in (False, True):
            records = []
            for draws, truth in zip(ensemble, z):
                remaining(deadline)
                physical = decode_ensemble(draws, nrm)
                if corrected:
                    physical = closure_correct(physical)
                diagnostics = window_diagnostics(physical, truth, nrm, ensemble.dtype)
                records.append({k: diagnostics[k] for k in METRICS})
            results["corrected" if corrected else "raw"] = {
                "per_window": records, "summary": summarize_diagnostics(records, months)}
        rows.append({**job, **inputs[job["id"]], "results": results})
    remaining(deadline)
    # Recheck the bridge/authorization after reduction, before a single publication.
    authorize_test(study, protocol, digest, ids)
    for row in rows:
        if sha256_file(Path(row["ensemble_root"]) / "gen_00000000/COMMIT.json") != row["ensemble_commit_sha256"]:
            raise ContractMismatch("ensemble changed during reduction")
    output = {"schema_version": 2, "protocol_sha256": digest, "selection": SELECTION_RULE,
              "selection_sha256": sha256_file(study / "SELECTION.json"), "endpoints": selection["endpoints"],
              "original_inputs": protocol["original_inputs"], "producers": protocol["producers"],
              "arrays": protocol["arrays"]["TEST"], "ids": ids, "months": months,
              "complete_grid": True, "rows": rows}
    remaining(deadline)
    exact_json(owned_path(study, study / "SUMMARY.json"), output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("protocol", "profile-train", "train",
                                                         "freeze-endpoints", "sample", "aggregate"))
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--original-root", help="original study containing PROTOCOL.json and runs/")
    parser.add_argument("--arrays-root")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--deadline", type=parse_deadline, help="UTC epoch or offset-aware ISO; may only shorten frozen deadline")
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args(argv)
    study = Path(args.study_root).resolve()
    if not study.is_dir():
        raise PublicationError("create and verify a fresh owned root externally")
    if args.stage == "protocol" and any(study.iterdir()):
        raise PublicationError("protocol publication requires an empty owned root")
    with RunLock(owned_path(study, study / "coordinator")):
        if args.stage == "protocol":
            if args.original_root is None or args.arrays_root is None or args.deadline is None:
                parser.error("protocol requires --original-root, --arrays-root and --deadline")
            remaining(args.deadline)
            protocol = base_protocol(study, Path(args.original_root).resolve(), Path(args.arrays_root).resolve(), args.deadline)
            fresh_json(study / "PROTOCOL.json", protocol)
            print(sha256_file(study / "PROTOCOL.json"))
            return 0
        if args.protocol_sha256 is None:
            parser.error("execution requires --protocol-sha256")
        # Check time before expensive payload validation and before any child can launch.
        raw = load_protocol(study, args.protocol_sha256)
        deadline = raw["deadline_epoch"] if args.deadline is None else min(args.deadline, raw["deadline_epoch"])
        remaining(deadline)
        protocol = verify_protocol(study, args.protocol_sha256)
        remaining(deadline)
        if args.stage in ("profile-train", "train"):
            train(study, protocol, [int(g) for g in args.gpus.split(",")], deadline, args.stage == "profile-train")
        elif args.stage == "freeze-endpoints":
            freeze_endpoints(study, protocol, args.protocol_sha256, deadline)
        elif args.stage == "sample":
            sample(study, protocol, args.protocol_sha256, [int(g) for g in args.gpus.split(",")], deadline)
        else:
            aggregate(study, protocol, args.protocol_sha256, deadline)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DeadlineReached as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(124)
