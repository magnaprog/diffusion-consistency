"""Bounded retrospective checkpoint study; inference only, with immutable receipts.

Run with PYTHONPATH pointing at the authenticated original inference source.
The original run_eval, run_score, source archives, protocols and data stay read only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
import torch
import prigm
from experiments.run_prigm_repair import verify_checkpoint
from prigm.artifacts import (NOISE_CONVENTION, STATE_CONVENTION, json_bytes,
                            load_arrays, merge_ensembles, read_ensemble,
                            require_source_match)
from prigm.barrier import require_test_started
from prigm.checkpoint import verify_source_snapshot
from prigm.recovery import ContractMismatch, RunLock, sha256_file
from prigm.scoring import month_balanced_mean

ORIGINAL = "25d0cdefdd5dbb160c32c7b5f4a6053bab99e4a9f64832d9eb6502232bf24e6b"
SELECTION = "b2fa1d9233b94c9db22472e644de1fc84f217baaa3d7385faaab2208e0a9c351"
EXTENSION = "1408cd52f306af0095f808dfe1d34aa8493fb26db9a4e75dae422839157bc3b4"
SUMMARY = "be795841a48d9ef1864c7adcbeb52e499e25a3bd6cc8a60909c5fbd557884b34"
REAL_MANIFEST = "17ac806ec1d3af99764fd4ce23bce0d0502cd49fd89719504f0e2926389967c0"
ARMS, SEEDS = ("D", "O-S"), (7, 13, 42)
UPDATES = (2868, 9560, 16252, 22944, 29636)
SAMPLER = {"n_eval": 50, "n_draws": 39, "eval_seed": 7}
COUNTS = {"VAL": 319, "TEST": 480}
SOURCE = Path(prigm.__file__).resolve().parent.parent
CHILD_SECONDS, STAGE_SECONDS = 120, 3600
STOP = threading.Event()
DEADLINE = float("inf")


def check(condition, message):
    if not condition:
        raise ContractMismatch(message)


def read(path, digest=None):
    path = Path(path)
    payload = path.read_bytes()
    if digest is not None:
        check(hashlib.sha256(payload).hexdigest() == digest, f"hash differs: {path}")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            check(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(payload, object_pairs_hook=unique)


def owned(root, path):
    root, path = Path(root).resolve(), Path(path)
    check(path.is_absolute(), "output must be absolute")
    check(path.resolve().is_relative_to(root) and path.resolve() != root, "output escapes study")
    for part in (path, *path.parents):
        if part == root:
            break
        check(not part.is_symlink(), "output must not traverse symlinks")
    return path


def publish(root, path, value):
    path = owned(root, path)
    payload = json_bytes(value)
    if path.exists():
        check(path.read_bytes() == payload, f"immutable receipt differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o444)
    return sha256_file(path)


def job_key(arm, seed, generation):
    return f"{arm}_s{seed}-g{generation}"


def validate_scope(scope):
    check(scope.get("exposure") == "historical_test_exposed", "exposure scope differs")
    check(scope.get("arms") == list(ARMS) and scope.get("training_seeds") == list(SEEDS)
          and scope.get("milestones") == list(UPDATES), "candidate grid differs")
    check(scope.get("coefficients") == {"D": 0.0, "O-S": 0.0005}, "coefficient scope differs")
    for split, count in COUNTS.items():
        row = scope[split]
        check(all(type(row.get(k)) is int and row[k] == v for k, v in
                  {**SAMPLER, "n_windows": count, "chunk_windows": 8, "jobs": 30}.items()),
              f"{split} scope differs")
    check(scope["TEST"].get("authenticated_reuse_jobs") == 10
          and scope["TEST"].get("new_jobs") == 20, "reuse scope differs")
    check(scope.get("no_training") is True and scope.get("accepted_Table1_unchanged") is True,
          "study must preserve training and original table")
    check(scope.get("original_protocol_sha256") == ORIGINAL
          and scope.get("original_selection_sha256") == SELECTION
          and scope.get("extension_summary_sha256") == SUMMARY, "ancestry differs")
    check(scope.get("policies") == {
        "P25": "original_raw_VAL25_selection",
        "P50": "minimum_complete_VAL50_month_balanced_ES_earlier_exact_tie",
        "F": "update_29636"}, "selection policies differ")


def initialize(root, scope_hash, real_manifest):
    scope = read(root / "SCOPE.json", scope_hash)
    validate_scope(scope)
    original, extension = Path(scope["original_root"]), Path(scope["extension_root"])
    check(not root.is_relative_to(original) and not root.is_relative_to(extension), "original root is protected")
    old = read(original / "PROTOCOL.json", ORIGINAL)
    selection = read(original / "SELECTION.json", SELECTION)
    ext = read(extension / "PROTOCOL.json", EXTENSION)
    final = read(extension / "SUMMARY.json", SUMMARY)
    manifest = read(real_manifest, REAL_MANIFEST)
    require_source_match(old)
    check(ext["inference_source_hashes"] == old["inference_source_hashes"], "extension inference differs")
    require_test_started(original, "TEST", ORIGINAL)
    require_test_started(extension, "TEST", EXTENSION)
    arrays = {}
    for split in COUNTS:
        path = Path(old["array_root"]) / f"{split.lower()}.npz"
        _, ids, _, identity = load_arrays(path, split, old["arrays"][split]["sha256"])
        check(identity == old["arrays"][split] and len(ids) == COUNTS[split], "array population differs")
        arrays[split] = identity
    jobs = []
    for arm in ARMS:
        for seed in SEEDS:
            name = f"{arm}_s{seed}"
            for generation in UPDATES:
                meta = verify_checkpoint(original, name, generation, old)
                run = original / "runs" / name
                archive = run / "source_snapshot.zip"
                if generation == UPDATES[0]:
                    verify_source_snapshot(archive.read_bytes(), meta["contract"]["source_hashes"],
                                           meta["contract"]["settings"]["source_snapshot_sha256"])
                key = job_key(arm, seed, generation)
                job = {"id": key, "arm": arm, "seed": seed, "generation": generation,
                       "run_root": str(run), "checkpoint_sha256": meta["files"]["state.pt"],
                       "checkpoint_commit_sha256": sha256_file(run / f"gen_{generation:08d}/COMMIT.json"),
                       "source_snapshot_sha256": meta["contract"]["settings"]["source_snapshot_sha256"]}
                parent = original / "artifacts/val" / key
                relative = (parent / "score-raw.json").relative_to(original).as_posix()
                job["val25"] = {"path": str(parent / "score-raw.json"),
                                "sha256": selection["validation_score_sha256"][relative]}
                if [name, generation] in selection["test_jobs"]:
                    ensemble = original / "artifacts/test" / key / "ensemble"
                    prefix = ensemble.relative_to(original).as_posix() + "/gen_00000000/"
                    job["reuse"] = {"root": str(ensemble), "study_root": str(original),
                                    "protocol_sha256": ORIGINAL,
                                    "commit_sha256": manifest["input_sha256"][prefix + "COMMIT.json"],
                                    "payload_sha256": manifest["input_sha256"][prefix + "ensemble.npz"]}
                elif generation == UPDATES[-1]:
                    row = next(r for r in final["rows"] if r["arm"] == arm and r["seed"] == seed
                               and r["c_r"] == old["c_r"][arm])
                    ensemble = Path(row["ensemble_root"])
                    check(ensemble.is_relative_to(extension), "unexpected final reuse root")
                    check(row["metadata"]["checkpoint_sha256"] == job["checkpoint_sha256"], "final weights differ")
                    job["reuse"] = {"root": str(ensemble), "study_root": str(extension),
                                    "protocol_sha256": EXTENSION, "commit_sha256": row["ensemble_commit_sha256"],
                                    "payload_sha256": row["metadata"]["files"]["ensemble.npz"]}
                jobs.append(job)
    check(len(jobs) == 30 and sum("reuse" in j for j in jobs) == 10, "incomplete grid or reuse inventory")
    protocol = {
        "schema_version": 2, "exposure": "historical_test_exposed", "study": "checkpoint_followup",
        "scope_sha256": scope_hash, "runner_sha256": sha256_file(Path(__file__)),
        "source_root": str(SOURCE), "original_root": str(original), "extension_root": str(extension),
        "original_protocol_sha256": ORIGINAL, "original_selection_sha256": SELECTION,
        "extension_protocol_sha256": EXTENSION, "extension_summary_sha256": SUMMARY,
        "accepted_real_manifest_sha256": REAL_MANIFEST,
        "checkpoint_verifier_sha256": sha256_file(SOURCE / "experiments/run_prigm_repair.py"),
        "execution_limits": {"child_seconds": CHILD_SECONDS, "stage_seconds": STAGE_SECONDS},
        "array_root": old["array_root"], "arrays": {**old["arrays"], **arrays},
        "training_source_hashes": old["training_source_hashes"],
        "training_data_hashes": old["training_data_hashes"],
        "inference_source_hashes": old["inference_source_hashes"],
        "samplers": {split: dict(SAMPLER) for split in COUNTS}, "controls": old["controls"],
        "jobs": jobs, "P25": {f"{a}_s{s}": selection["selected"][f"{a}_s{s}"]["raw"]
                              for a in ARMS for s in SEEDS},
        "environment": {"python": sys.version, "executable": sys.executable, "numpy": np.__version__,
                        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
                        "platform": platform.platform(), "hostname": platform.node()},
    }
    publish(root, root / "PROTOCOL.json", protocol)
    return protocol


def load(root, digest):
    p = read(root / "PROTOCOL.json", digest)
    validate_scope(read(root / "SCOPE.json", p["scope_sha256"]))
    check(p["runner_sha256"] == sha256_file(Path(__file__)), "runner changed after protocol freeze")
    check(p["source_root"] == str(SOURCE), "wrong imported inference source")
    check(p["environment"]["python"] == sys.version and p["environment"]["numpy"] == np.__version__
          and p["environment"]["torch"] == str(torch.__version__)
          and p["environment"]["executable"] == sys.executable, "execution environment changed")
    expected = [job_key(a, s, g) for a in ARMS for s in SEEDS for g in UPDATES]
    check([j["id"] for j in p["jobs"]] == expected, "complete ordered grid required")
    return p


def identity(p, split):
    z, ids, nrm, ident = load_arrays(Path(p["array_root"]) / f"{split.lower()}.npz",
                                   split, p["arrays"][split]["sha256"])
    check(ident == p["arrays"][split], "array identity differs")
    return z, ids, nrm


def recheck_inputs(root, p, stage):
    require_source_match(p)
    check(sha256_file(SOURCE / "experiments/run_prigm_repair.py") == p["checkpoint_verifier_sha256"],
          "checkpoint verifier changed")
    pins = {}
    for job in p["jobs"]:
        run = Path(job["run_root"])
        for path, expected in ((run / f"gen_{job['generation']:08d}/COMMIT.json", job["checkpoint_commit_sha256"]),
                               (run / f"gen_{job['generation']:08d}/state.pt", job["checkpoint_sha256"]),
                               (run / "source_snapshot.zip", job["source_snapshot_sha256"])):
            if str(path) not in pins:
                pins[str(path)] = sha256_file(path)
            check(pins[str(path)] == expected, f"original checkpoint input changed: {path}")
    for split in COUNTS:
        identity(p, split)
        pins[str(Path(p["array_root"]) / f"{split.lower()}.npz")] = p["arrays"][split]["sha256"]
    publish(root, root / f"INPUTS_RECHECKED-{stage}.json", {
        "protocol_sha256": sha256_file(root / "PROTOCOL.json"), "input_sha256": pins,
        "inference_source_hashes": p["inference_source_hashes"]})


def metadata(meta, job, p, split, protocol_hash):
    expected = {"arm": job["arm"], "seed": job["seed"], "generation_updates": job["generation"],
                "checkpoint_sha256": job["checkpoint_sha256"], "n_generated": 6,
                "protocol_sha256": protocol_hash, "arrays": p["arrays"][split],
                "sampler": p["samplers"][split], "state_convention": STATE_CONVENTION,
                "noise_convention": NOISE_CONVENTION,
                "inference_source_sha256": p["inference_source_hashes"]["prigm/run_eval.py"]}
    for key, value in expected.items():
        check(json_bytes({key: meta.get(key)}) == json_bytes({key: value}), f"ensemble {key} differs")
    check(meta.get("procedure", "diffusion") == "diffusion", "neural ensemble required")


def verify_ensemble(root, job, p, split, protocol_hash, ids, complete=False):
    draws, actual, meta = read_ensemble(root)
    check(actual == ids, "ensemble ordered IDs differ")
    metadata(meta, job, p, split, protocol_hash)
    check(np.count_nonzero(draws[..., :18]) == 0 and np.count_nonzero(draws[..., 30:]) == 0,
          "non-hidden storage must be zero placeholders")
    if not complete:
        return draws, meta
    expected_roots = [root.parent / f"chunk-{offset:04d}" for offset in range(0, len(ids), 8)]
    check(set(meta.get("chunks", {})) == {str(x.resolve()) for x in expected_roots}, "chunk inventory differs")
    for offset, chunk in zip(range(0, len(ids), 8), expected_roots):
        check(sha256_file(chunk / "gen_00000000/COMMIT.json") == meta["chunks"][str(chunk.resolve())],
              "chunk receipt changed")
        block, cm = verify_ensemble(chunk, job, p, split, protocol_hash, ids[offset:offset + 8])
        check(type(cm.get("offset")) is int and cm["offset"] == offset
              and type(cm.get("count")) is int and cm["count"] == len(block), "chunk layout differs")
        check(np.array_equal(block.view(np.uint8), draws[offset:offset + 8].view(np.uint8)),
              "merged values differ from exact chunks")
    return draws, meta


def authenticate_reuse(root, p, digest):
    _, ids, _ = identity(p, "TEST")
    records = {}
    for job in p["jobs"]:
        if "reuse" not in job:
            continue
        r = job["reuse"]
        path = Path(r["root"])
        check(sha256_file(path / "gen_00000000/COMMIT.json") == r["commit_sha256"], "accepted reuse commit differs")
        check(sha256_file(path / "gen_00000000/ensemble.npz") == r["payload_sha256"], "accepted reuse payload differs")
        require_test_started(Path(r["study_root"]), "TEST", r["protocol_sha256"])
        _, meta = verify_ensemble(path, job, p, "TEST", r["protocol_sha256"], ids, complete=True)
        records[job["id"]] = {**r, "chunks": meta["chunks"], "n_chunks": len(meta["chunks"]),
                              "checkpoint_sha256": job["checkpoint_sha256"]}
    check(len(records) == 10, "ten complete reuse jobs required")
    result = {"protocol_sha256": digest, "scope_sha256": p["scope_sha256"], "jobs": records,
              "scope": "Full payload hashes, IDs, sampler/noise/state metadata and bitwise chunk union; no TEST statistics."}
    publish(root, root / "REUSE_AUTHENTICATED.json", result)
    return result


def validate_score(row, job, p, split, protocol_hash, ids, commit_hash):
    values = np.asarray(row.get("scores"), dtype=float)
    months = np.asarray([int(w[:4]) * 100 + int(w[5:7]) for w in ids])
    check(values.shape == (len(ids),) and np.isfinite(values).all(), "score population/nonfinite value")
    check(row.get("ids") == ids and row.get("months") == months.tolist()
          and type(row.get("n")) is int and row["n"] == len(ids), "score IDs/count differ")
    check(row.get("corrected") is False and row.get("procedure") == "diffusion", "raw neural scores required")
    check(row.get("arrays") == p["arrays"][split] and row.get("protocol_sha256") == protocol_hash
          and row.get("checkpoint_sha256") == job["checkpoint_sha256"]
          and row.get("ensemble_commit_sha256") == commit_hash, "score provenance differs")
    check(np.isclose(row["mean"], values.mean(), rtol=0, atol=1e-12)
          and np.isclose(row["month_balanced"], month_balanced_mean(values, months), rtol=0, atol=1e-12),
          "score reductions differ")
    return values, months


def curve_statistics(values, months):
    values, months = np.asarray(values), np.asarray(months)
    check(values.shape == (5, len(months)) and np.isfinite(values).all(), "five finite curves required")
    represented, years = np.unique(months), np.unique(months // 100)
    check(len(years) == 20 and all(np.sum(represented // 100 == y) == 2 for y in years), "twenty two-month years required")
    by_month = np.stack([values[:, months == m].mean(axis=1) for m in represented], axis=1)
    by_year = np.stack([by_month[:, represented // 100 == y].mean(axis=1) for y in years], axis=1)
    means = by_month.mean(axis=1)
    draws = np.random.Generator(np.random.PCG64(20261001)).integers(0, 20, (10000, 20))
    winners = by_year[:, draws].mean(axis=2).argmin(axis=0)
    loo = [int(np.delete(by_year, i, axis=1).mean(axis=1).argmin()) for i in range(20)]
    ordered = sorted(range(5), key=lambda i: (means[i], UPDATES[i]))
    return {"means": means.tolist(), "selected_generation": UPDATES[ordered[0]],
            "runner_up_gap": float(means[ordered[1]] - means[ordered[0]]),
            "bootstrap_selection_counts": {str(g): int(np.sum(winners == i)) for i, g in enumerate(UPDATES)},
            "leave_year_out_counts": {str(g): loo.count(i) for i, g in enumerate(UPDATES)}}


def saved_val25(root, p, digest):
    _, ids, _ = identity(p, "VAL")
    oldp = {**p, "samplers": {**p["samplers"], "VAL": {**SAMPLER, "n_eval": 25}}}
    values, inputs = {}, {}
    for job in p["jobs"]:
        item = job["val25"]
        row = read(item["path"], item["sha256"])
        commit = Path(item["path"]).parent / "ensemble/gen_00000000/COMMIT.json"
        meta = read(commit, row["ensemble_commit_sha256"])
        metadata(meta, job, oldp, "VAL", ORIGINAL)
        v, months = validate_score(row, job, p, "VAL", ORIGINAL, ids, sha256_file(commit))
        values[job["id"]] = v.tolist()
        inputs[item["path"]] = item["sha256"]
    curves = {}
    for arm in ARMS:
        for seed in SEEDS:
            name = f"{arm}_s{seed}"
            curves[name] = curve_statistics([values[job_key(arm, seed, g)] for g in UPDATES], months)
            check(curves[name]["selected_generation"] == p["P25"][name]["generation"], "original raw selection does not reproduce")
    result = {"protocol_sha256": digest, "scope_sha256": p["scope_sha256"], "input_sha256": inputs,
              "ids": ids, "months": months.tolist(), "scores": values, "curves": curves,
              "bootstrap": {"replicates": 10000, "seed": 20261001, "generator": "PCG64",
                            "scope": "Paired validation years, fixed weights/draws; not MC or refitting uncertainty."}}
    publish(root, root / "SAVED_VAL25.json", result)
    return result


def gpu_admission(root, digest, stage, path, expected):
    check(path is not None and expected is not None, "main GPU admission receipt required")
    record = read(owned(root, Path(path)), expected)
    check(record.get("protocol_sha256") == digest and record.get("approved_by") == "main"
          and stage in record.get("stages", []), "GPU stage not admitted by main")
    devices = record.get("devices", [])
    check(0 < len(devices) <= 4 and all(type(d.get("index")) is int and isinstance(d.get("uuid"), str) for d in devices)
          and len({d["index"] for d in devices}) == len(devices)
          and len({d["uuid"] for d in devices}) == len(devices), "unique assigned devices required")
    observed = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True, timeout=10)
    actual = {int(line.split(",")[0]): line.split(",")[1].strip() for line in observed.splitlines()}
    check(all(actual.get(d["index"]) == d["uuid"] for d in devices), "GPU UUID assignment differs")
    applications = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
                                           "--format=csv,noheader,nounits"], text=True, timeout=10)
    assigned = {d["uuid"] for d in devices}
    # Graphics-only desktop processes are absent from this compute query; accept only
    # these exact desktop executable names if the driver also lists them here.
    desktop = {"/usr/lib/xorg/Xorg", "/usr/bin/gnome-shell", "/usr/bin/kwin_wayland",
               "/usr/libexec/gnome-remote-desktop-daemon"}
    for line in applications.splitlines():
        gpu, pid, name = [part.strip() for part in line.split(",", 2)]
        check(gpu not in assigned or name in desktop, f"unexpected compute on allocated GPU: {line}")
    publish(root, root / "receipts" / f"admission-{uuid.uuid4().hex}.json", {
        "protocol_sha256": digest, "admission_sha256": expected, "stage": stage,
        "devices": devices, "compute_applications": applications,
        "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version",
                                            "--format=csv,noheader,nounits"], text=True, timeout=10)})
    return devices


def launch(root, p, command, label, device=None):
    token = uuid.uuid4().hex
    log = owned(root, root / "logs" / f"{label}-{token}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    overrides = {"PYTHONPATH": p["source_root"], "PYTHONDONTWRITEBYTECODE": "1",
                 "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                 "CUDA_VISIBLE_DEVICES": "" if device is None else device["uuid"],
                 "PRIGM_REQUIRE_CUDA": "0" if device is None else "1", "PISCES_REQUIRE_CUDA": "0"}
    env.update(overrides)
    started, tick, code = datetime.now(timezone.utc).isoformat(), time.monotonic(), None
    child, failure = None, None
    try:
        check(not STOP.is_set() and tick < DEADLINE, "stage stopped or budget exhausted")
        with log.open("xb") as handle:
            child = subprocess.Popen(command, cwd=p["source_root"], env=env, stdout=handle,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                check(not STOP.is_set(), "stage interrupted")
                remaining = min(tick + CHILD_SECONDS, DEADLINE) - time.monotonic()
                check(remaining > 0, "child timeout or stage budget exhausted")
                try:
                    code = child.wait(timeout=min(1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        check(code == 0, f"child failed ({code}); retained log {log}")
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        if child is not None:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    pass
                if sig == signal.SIGTERM:
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
            code = child.wait(timeout=5)
        raise
    finally:
        record = {"command": command, "cwd": p["source_root"], "environment": overrides,
                  "gpu": device, "started_utc": started, "elapsed_seconds": time.monotonic() - tick,
                  "exit_code": code, "failure": failure, "child_seconds": CHILD_SECONDS,
                  "log": str(log), "log_sha256": sha256_file(log) if log.exists() else None,
                  "runner_sha256": p["runner_sha256"], "inference_source_hashes": p["inference_source_hashes"]}
        publish(root, root / "receipts" / f"{label}-{token}.json", record)


def sample_command(root, p, digest, job, split, offset, output):
    # Call the unchanged entry point, then report allocator peaks without changing sampling.
    measured_entry = ("from prigm.run_eval import main; import torch,json; result=main(); "
                      "print(json.dumps({'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),"
                      "'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved()})); raise SystemExit(result)")
    return [sys.executable, "-B", "-c", measured_entry, "--run-root", job["run_root"],
            "--generation", str(job["generation"]), "--checkpoint-sha256", job["checkpoint_sha256"],
            "--out", str(output), "--arrays", str(Path(p["array_root"]) / f"{split.lower()}.npz"),
            "--split", split, "--n-eval", str(p["samplers"][split]["n_eval"]), "--eval-seed", "7",
            "--offset", str(offset), "--limit", "8", "--study-root", str(root), "--protocol-sha256", digest]


def score_job(root, p, digest, job, split, ids):
    parent = root / "artifacts" / split.lower() / job["id"]
    protocol_root, protocol_hash = root, digest
    if split == "TEST" and "reuse" in job:
        ensemble = Path(job["reuse"]["root"])
        check(sha256_file(ensemble / "gen_00000000/COMMIT.json") == job["reuse"]["commit_sha256"]
              and sha256_file(ensemble / "gen_00000000/ensemble.npz") == job["reuse"]["payload_sha256"],
              "accepted reuse changed after authentication")
        protocol_root, protocol_hash = Path(job["reuse"]["study_root"]), job["reuse"]["protocol_sha256"]
    else:
        ensemble = parent / "ensemble"
        chunks = [parent / f"chunk-{o:04d}" for o in range(0, len(ids), 8)]
        if not ensemble.exists():
            merge_ensembles(chunks, owned(root, ensemble), ids)
    verify_ensemble(ensemble, job, p, split, protocol_hash, ids, complete=True)
    output = owned(root, parent / "score-raw.json")
    if not output.exists():
        launch(root, p, [sys.executable, "-B", "-m", "prigm.run_score", "--ensemble", str(ensemble),
                        "--arrays", str(Path(p["array_root"]) / f"{split.lower()}.npz"), "--split", split,
                        "--output", str(output), "--study-root", str(protocol_root),
                        "--protocol-sha256", protocol_hash], f"score-{split}-{job['id']}")
    values, months = validate_score(read(output), job, p, split, protocol_hash, ids,
                                   sha256_file(ensemble / "gen_00000000/COMMIT.json"))
    return values, months, sha256_file(output)


def selection_from_scores(p, scores):
    check(set(scores) == {j["id"] for j in p["jobs"]}, "all thirty VAL scores required")
    policies = {}
    for arm in ARMS:
        for seed in SEEDS:
            name = f"{arm}_s{seed}"
            candidates = [j for j in p["jobs"] if j["arm"] == arm and j["seed"] == seed]
            check(len(candidates) == 5 and all(np.isfinite(scores[j["id"]]) for j in candidates), "complete finite candidates required")
            winner = min(candidates, key=lambda j: (scores[j["id"]], j["generation"]))
            policies[name] = {"P25": p["P25"][name]["generation"], "P50": winner["generation"], "F": UPDATES[-1]}
    return policies


def freeze_selection(root, p, digest):
    reuse = read(root / "REUSE_AUTHENTICATED.json")
    check(reuse["protocol_sha256"] == digest and reuse["scope_sha256"] == p["scope_sha256"]
          and set(reuse["jobs"]) == {j["id"] for j in p["jobs"] if "reuse" in j}
          and len(reuse["jobs"]) == 10, "reuse must be authenticated first")
    _, ids, _ = identity(p, "VAL")
    scores, pins = {}, {}
    for job in p["jobs"]:
        path = root / "artifacts/val" / job["id"] / "score-raw.json"
        ensemble = path.parent / "ensemble"
        verify_ensemble(ensemble, job, p, "VAL", digest, ids, complete=True)
        values, months = validate_score(read(path), job, p, "VAL", digest, ids,
                                       sha256_file(ensemble / "gen_00000000/COMMIT.json"))
        scores[job["id"]] = month_balanced_mean(values, months)
        pins[str(path)] = sha256_file(path)
    result = {"schema_version": 2, "protocol_sha256": digest, "scope_sha256": p["scope_sha256"],
              "policies": selection_from_scores(p, scores), "validation_score_sha256": pins,
              "test_jobs": [[f"{j['arm']}_s{j['seed']}", j["generation"]] for j in p["jobs"]],
              "test_checkpoint_hashes": sorted(j["checkpoint_sha256"] for j in p["jobs"]),
              "reuse_authentication_sha256": sha256_file(root / "REUSE_AUTHENTICATED.json"),
              "selection": "P50 solely complete raw VAL50; TEST covers complete predetermined grid"}
    selection_hash = publish(root, root / "SELECTION.json", result)
    publish(root, root / "TEST_STARTED.json", {"phase": "TEST_STARTED", "protocol_sha256": digest,
            "selection_sha256": selection_hash, "previous_test_exposure": True})
    return result


def test_gate(root, p, digest):
    result = require_test_started(root, "TEST", digest)
    check(result is not None and result.get("scope_sha256") == p["scope_sha256"], "missing study selection")
    check(result["test_checkpoint_hashes"] == sorted(j["checkpoint_sha256"] for j in p["jobs"]), "TEST must cover all thirty checkpoints")
    check(result["test_jobs"] == [[f"{j['arm']}_s{j['seed']}", j["generation"]] for j in p["jobs"]], "TEST job grid changed")
    check(sha256_file(root / "REUSE_AUTHENTICATED.json") == result["reuse_authentication_sha256"], "reuse receipt changed")
    for job in p["jobs"]:
        if "reuse" in job:
            check(sha256_file(Path(job["reuse"]["root"]) / "gen_00000000/COMMIT.json") == job["reuse"]["commit_sha256"],
                  "accepted reuse commit changed before TEST")
    scores = {}
    for job in p["jobs"]:
        path = str(root / "artifacts/val" / job["id"] / "score-raw.json")
        row = read(path, result["validation_score_sha256"][path])
        scores[job["id"]] = row["month_balanced"]
    check(result["policies"] == selection_from_scores(p, scores), "policies differ from complete VAL50")
    return result


def sample_grid(root, p, digest, split, devices):
    if split == "TEST":
        test_gate(root, p, digest)
    _, ids, _ = identity(p, split)
    jobs = [j for j in p["jobs"] if not (split == "TEST" and "reuse" in j)]
    tasks = [(job, offset) for job in jobs for offset in range(0, len(ids), 8)]
    failed = threading.Event()
    def worker(index, device):
        try:
            for job, offset in tasks[index::len(devices)]:
                if failed.is_set():
                    return
                out = owned(root, root / "artifacts" / split.lower() / job["id"] / f"chunk-{offset:04d}")
                if not out.exists():
                    launch(root, p, sample_command(root, p, digest, job, split, offset, out),
                           f"sample-{split}-{job['id']}-{offset:04d}", device)
                _, meta = verify_ensemble(out, job, p, split, digest, ids[offset:offset + 8])
                check(meta["offset"] == offset and meta["count"] == len(ids[offset:offset + 8]), "chunk offset/count differs")
        except BaseException:
            failed.set()
            raise
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(worker, i, d) for i, d in enumerate(devices)]
        for future in futures:
            future.result()
    for job in p["jobs"]:
        score_job(root, p, digest, job, split, ids)


def pilot(root, p, digest, device):
    job = next(j for j in p["jobs"] if j["id"] == "D_s7-g9560")
    _, ids, _ = identity(p, "VAL")
    pilot_root = owned(root, root / "pilot25")
    p25 = {**p, "samplers": {**p["samplers"], "VAL": {**SAMPLER, "n_eval": 25}}}
    h25 = publish(root, pilot_root / "PROTOCOL.json", p25)
    out25 = pilot_root / "chunk-0000"
    if not out25.exists():
        launch(root, p25, sample_command(pilot_root, p25, h25, job, "VAL", 0, out25), "pilot-VAL25", device)
    actual, _ = verify_ensemble(out25, job, p25, "VAL", h25, ids[:8])
    old = Path(job["val25"]["path"]).parent / "chunk-0000"
    old_score = read(job["val25"]["path"], job["val25"]["sha256"])
    old_commit = read(old.parent / "ensemble/gen_00000000/COMMIT.json", old_score["ensemble_commit_sha256"])
    check(sha256_file(old / "gen_00000000/COMMIT.json") == old_commit["chunks"][str(old.resolve())],
          "pilot reference chunk differs from frozen validation history")
    expected, _ = verify_ensemble(old, job, p25, "VAL", ORIGINAL, ids[:8])
    equal = np.array_equal(actual.view(np.uint8), expected.view(np.uint8))
    publish(root, root / "PILOT25.json", {"protocol_sha256": digest, "bitwise_equal": equal,
            "max_absolute_difference": float(np.max(np.abs(actual.astype(float) - expected.astype(float)))),
            "reference_commit_sha256": sha256_file(old / "gen_00000000/COMMIT.json"),
            "replay_commit_sha256": sha256_file(out25 / "gen_00000000/COMMIT.json")})
    check(equal, "pilot VAL25 replay differs; no VAL50 launch")
    out50 = owned(root, root / "artifacts/val" / job["id"] / "chunk-0000")
    if not out50.exists():
        launch(root, p, sample_command(root, p, digest, job, "VAL", 0, out50), "pilot-VAL50", device)
    verify_ensemble(out50, job, p, "VAL", digest, ids[:8])
    publish(root, root / "PILOT_COMPLETED.json", {"protocol_sha256": digest, "gpu": device,
            "VAL25_receipt_sha256": sha256_file(root / "PILOT25.json"),
            "VAL50_commit_sha256": sha256_file(out50 / "gen_00000000/COMMIT.json")})


def budget_comparison(saved, curves50):
    result = {}
    for arm in ARMS:
        for seed in SEEDS:
            name = f"{arm}_s{seed}"
            v25, v50 = np.asarray(saved["curves"][name]["means"]), np.asarray(curves50[name])
            check(v25.shape == v50.shape == (5,) and np.isfinite(v25).all() and np.isfinite(v50).all(),
                  "complete paired validation curves required")
            ranks = lambda v: (np.argsort(np.argsort(v, kind="stable"), kind="stable") + 1).tolist()
            result[name] = {"VAL25": v25.tolist(), "VAL50": v50.tolist(),
                            "VAL50_minus_VAL25": (v50 - v25).tolist(),
                            "VAL25_ranks_earlier_ties": ranks(v25), "VAL50_ranks_earlier_ties": ranks(v50),
                            "P25": UPDATES[int(v25.argmin())], "P50": UPDATES[int(v50.argmin())]}
    return result


def summarize(root, p, digest):
    selected = test_gate(root, p, digest)
    result = {"protocol_sha256": digest, "selection_sha256": sha256_file(root / "SELECTION.json"),
              "policies": selected["policies"], "scores": {}, "curves": {}, "input_sha256": {},
              "scope": "Raw complete retrospective grid. No TEST selection or training-cause inference."}
    for split in COUNTS:
        _, ids, _ = identity(p, split)
        rows = {}
        for job in p["jobs"]:
            values, months, pin = score_job(root, p, digest, job, split, ids)
            rows[job["id"]] = values.tolist()
            result["input_sha256"][str(root / "artifacts" / split.lower() / job["id"] / "score-raw.json")] = pin
        result["scores"][split] = {"ids": ids, "months": months.tolist(), "data": rows}
        # Rank/selection diagnostics apply only to VAL, never TEST.
        result["curves"][split] = {f"{a}_s{s}": [month_balanced_mean(rows[job_key(a, s, g)], months)
                                                   for g in UPDATES] for a in ARMS for s in SEEDS}
    policy_means = {}
    for policy in ("P25", "P50", "F"):
        policy_means[policy] = {}
        for arm in ARMS:
            values = [result["curves"]["TEST"][f"{arm}_s{s}"][UPDATES.index(selected["policies"][f"{arm}_s{s}"][policy])]
                      for s in SEEDS]
            policy_means[policy][arm] = {"seedwise": values, "mean": float(np.mean(values))}
        policy_means[policy]["O-S_minus_D"] = policy_means[policy]["O-S"]["mean"] - policy_means[policy]["D"]["mean"]
    result["policy_means"] = policy_means
    result["contrast_changes"] = {f"{policy}_minus_P25": policy_means[policy]["O-S_minus_D"] - policy_means["P25"]["O-S_minus_D"]
                                  for policy in ("P50", "F")}
    result["common_update_TEST_contrasts"] = {
        str(g): {"seedwise": [result["curves"]["TEST"][f"O-S_s{s}"][i] - result["curves"]["TEST"][f"D_s{s}"][i] for s in SEEDS]}
        for i, g in enumerate(UPDATES)}
    for row in result["common_update_TEST_contrasts"].values():
        row["mean"] = float(np.mean(row["seedwise"]))
    val = result["scores"]["VAL"]
    result["VAL50_selection_diagnostics"] = {
        f"{a}_s{s}": curve_statistics([val["data"][job_key(a, s, g)] for g in UPDATES], val["months"])
        for a in ARMS for s in SEEDS}
    saved = saved_val25(root, p, digest)
    result["input_sha256"][str(root / "SAVED_VAL25.json")] = sha256_file(root / "SAVED_VAL25.json")
    result["validation_budget_comparison"] = budget_comparison(saved, result["curves"]["VAL"])
    result["final_minus_selected_TEST"] = {
        policy: {a: (np.asarray(policy_means["F"][a]["seedwise"]) - policy_means[policy][a]["seedwise"]).tolist()
                 for a in ARMS} for policy in ("P25", "P50")}
    publish(root, root / "SUMMARY.json", result)
    return result


def main(argv=None):
    global DEADLINE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("init", "saved-val25", "authenticate-reuse", "pilot",
                        "sample-val", "freeze-selection", "sample-test", "summarize"))
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--scope-sha256")
    parser.add_argument("--accepted-real-manifest", type=Path)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--gpu-admission", type=Path)
    parser.add_argument("--gpu-admission-sha256")
    args = parser.parse_args(argv)
    root = args.study_root.resolve()
    check(root.is_dir() and not args.study_root.is_symlink(), "fresh externally owned study root required")
    started, tick, failure = datetime.now(timezone.utc).isoformat(), time.monotonic(), None
    STOP.clear()
    DEADLINE = tick + STAGE_SECONDS
    def interrupted(signum, frame):
        STOP.set()
        raise ContractMismatch(f"stage interrupted or budget exhausted: signal {signum}")
    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)}
    signal.setitimer(signal.ITIMER_REAL, STAGE_SECONDS)
    try:
        with RunLock(root / "coordinator"):
            if args.stage == "init":
                check(args.scope_sha256 is not None and args.accepted_real_manifest is not None, "scope and accepted manifest required")
                initialize(root, args.scope_sha256, args.accepted_real_manifest)
            else:
                check(args.protocol_sha256 is not None, "pinned protocol required")
                p = load(root, args.protocol_sha256)
                if args.stage in ("pilot", "sample-val", "sample-test"):
                    devices = gpu_admission(root, args.protocol_sha256, args.stage, args.gpu_admission, args.gpu_admission_sha256)
                    if args.stage == "pilot":
                        pilot(root, p, args.protocol_sha256, devices[0])
                    else:
                        receipt = read(root / "PILOT_COMPLETED.json")
                        check(receipt["protocol_sha256"] == args.protocol_sha256, "successful pilot required")
                        sample_grid(root, p, args.protocol_sha256, "VAL" if args.stage == "sample-val" else "TEST", devices)
                else:
                    {"saved-val25": saved_val25, "authenticate-reuse": authenticate_reuse,
                     "freeze-selection": freeze_selection, "summarize": summarize}[args.stage](root, p, args.protocol_sha256)
                if args.stage in ("pilot", "sample-val", "sample-test", "summarize"):
                    recheck_inputs(root, p, args.stage)
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        outputs = {name: sha256_file(root / name) for name in
                   ("PROTOCOL.json", "SAVED_VAL25.json", "REUSE_AUTHENTICATED.json", "SELECTION.json", "TEST_STARTED.json", "SUMMARY.json")
                   if (root / name).is_file()}
        publish(root, root / "receipts" / f"stage-{args.stage}-{uuid.uuid4().hex}.json", {
            "command": [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
            "started_utc": started, "elapsed_seconds": time.monotonic() - tick,
            "environment": {key: os.environ.get(key) for key in
                            ("PYTHONPATH", "CUDA_VISIBLE_DEVICES", "PYTHONDONTWRITEBYTECODE",
                             "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "failure": failure, "output_sha256": outputs, "runner_sha256": sha256_file(Path(__file__))})
    print(json.dumps({"stage": args.stage, "protocol_sha256": sha256_file(root / "PROTOCOL.json"), "status": "complete"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
