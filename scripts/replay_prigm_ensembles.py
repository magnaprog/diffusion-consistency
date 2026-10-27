"""Regenerate the thirteen originally selected TEST prediction sets separately.

Write results in a separate directory owned by this replay. The stages are
validate, pilot, and run. They perform no training, new model selection, input
repair or automatic retry. Preserve earlier attempts and failed outputs so the
record shows what was tried and what completed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import signal
import subprocess
import sys
import threading
import time
import zipfile

import numpy as np


PUBLIC_MANIFEST = "17ac806ec1d3af99764fd4ce23bce0d0502cd49fd89719504f0e2926389967c0"
PROTOCOL = "25d0cdefdd5dbb160c32c7b5f4a6053bab99e4a9f64832d9eb6502232bf24e6b"
COMMIT = "3e75e96c3db031d8ea94f65c062adbafc9e2644c"
SAMPLER = {"n_eval": 50, "n_draws": 39, "eval_seed": 7}
PILOTS = ("D_s7-g9560/chunk-0000", "G5_s7-g29636/chunk-0000")
ENV_KEYS = ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
            "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "PRIGM_REQUIRE_CUDA",
            "PISCES_REQUIRE_CUDA", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TMPDIR")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def publish(path, value):
    """Write a new JSON record while the caller holds the runner lock.

    Refuse an existing destination. Keep any temporary file left by an
    interrupted write so that the incomplete attempt remains inspectable.
    """
    path = Path(path)
    require(not path.exists(), f"refusing existing receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".pending-{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def pin(path, expected, pins):
    path = Path(path).resolve(strict=True)
    require(path.is_file() and sha(path) == expected, f"input hash mismatch: {path}")
    require(str(path) not in pins or pins[str(path)] == expected, f"conflicting pin: {path}")
    pins[str(path)] = expected
    return path


def member(root, name):
    root = Path(root).resolve(strict=True)
    path = (root / name).resolve(strict=True)
    require(path.is_relative_to(root), f"input escapes its root: {path}")
    return path


def ensemble(path):
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == {"ensemble", "ids"}, f"unexpected NPZ fields: {path}")
        values, ids = data["ensemble"], data["ids"]
    require(values.dtype == np.float32 and np.isfinite(values).all(), f"invalid ensemble: {path}")
    return values, ids


def compare_payload(actual, original, expected_ids):
    left, left_ids = ensemble(actual)
    right, right_ids = ensemble(original)
    require(np.array_equal(left_ids, expected_ids) and np.array_equal(right_ids, expected_ids),
            "ordered IDs differ")
    require(left.shape == right.shape and left.dtype == right.dtype
            and left.tobytes() == right.tobytes(), "raw ensemble arrays differ")
    require(sha(actual) == sha(original), "NPZ bytes differ")
    return {"npz_sha256": sha(actual), "shape": list(left.shape), "dtype": str(left.dtype),
            "ordered_ids_exact": True, "raw_array_bytes_exact": True, "npz_bytes_exact": True}


def command(python, study, output, name, generation, checkpoint, arrays, offset):
    return [str(python), "-B", "-u", "-m", "prigm.run_eval", "--run-root", str(study / "runs" / name),
            "--generation", str(generation), "--checkpoint-sha256", checkpoint,
            "--out", str(output), "--arrays", str(arrays), "--split", "TEST",
            "--n-eval", "50", "--eval-seed", "7", "--offset", str(offset), "--limit", "8",
            "--study-root", str(study), "--protocol-sha256", PROTOCOL]


def task_layout(jobs):
    require(len(jobs) == 13 and len({(n, g) for n, g in jobs}) == 13, "expected thirteen unique selections")
    return [(name, generation, offset) for name, generation in jobs for offset in range(0, 480, 8)]


def validate(args):
    study, source, public, root = map(Path, (args.study, args.source, args.public_repo, args.root))
    pins = {}
    manifest_path = pin(public / "paper/workshops/prigm/evidence/real/MANIFEST.json", PUBLIC_MANIFEST, pins)
    manifest = read(manifest_path)
    published = manifest["input_sha256"]
    require(published["PROTOCOL.json"] == PROTOCOL, "published protocol changed")
    for name in ("PROTOCOL.json", "SELECTION.json", "TEST_STARTED.json"):
        pin(study / name, published[name], pins)
    protocol, selection, receipt = (read(study / name) for name in ("PROTOCOL.json", "SELECTION.json", "TEST_STARTED.json"))
    require(protocol["samplers"]["TEST"] == SAMPLER and protocol["sampling_chunk_windows"] == 8,
            "sampling layout changed")
    require(receipt["protocol_sha256"] == PROTOCOL and receipt["selection_sha256"] == published["SELECTION.json"]
            and receipt["previous_test_exposure"] is True and receipt["phase"] == "TEST_STARTED", "invalid TEST receipt")
    hashes = protocol["inference_source_hashes"]
    actual_sources = {p.relative_to(source).as_posix(): sha(p) for p in (source / "prigm").rglob("*.py")}
    require(len(hashes) == 43 and actual_sources == hashes, "frozen inference source inventory mismatch")
    for name, digest in hashes.items():
        pin(member(source, name), digest, pins)
    gradients_path = manifest_path.parent / "gradients.json"
    pin(gradients_path, manifest["output_sha256"]["gradients.json"], pins)
    reports = [r for grid in read(gradients_path)["grids"].values() for r in grid["reports"].values()]
    require(len(reports) == 12 and all(r["current_source_hashes"] == hashes
            and r["training_source_hashes"] == protocol["training_source_hashes"] for r in reports),
            "published source receipts disagree")
    for split in ("TRAIN", "VAL", "TEST"):
        path = member(study, Path(protocol["array_root"]) / f"{split.lower()}.npz")
        pin(path, protocol["arrays"][split]["sha256"], pins)
    arrays = Path(protocol["array_root"]) / "test.npz"
    with np.load(arrays, allow_pickle=False) as data:
        ids = data["ids"]
        require(data["z"].shape == (480, 48, 6) and str(data["split"].item()) == "TEST", "invalid TEST arrays")
    require(ids.shape == (480,) and len(set(ids.tolist())) == 480, "invalid TEST IDs")
    selected = {(name, row["generation"]) for name, kinds in selection["selected"].items() for row in kinds.values()}
    jobs = selection["test_jobs"]
    require(set(map(tuple, jobs)) == selected, "selected union changed")
    expected_merged = {f"artifacts/test/{name}-g{gen}/ensemble/gen_00000000/ensemble.npz" for name, gen in jobs}
    require(expected_merged == {k for k in published if k.startswith("artifacts/test/")
            and k.endswith("/ensemble/gen_00000000/ensemble.npz")}, "public ensemble inventory differs")
    task_layout(jobs)
    tasks = []
    for name, generation in jobs:
        parent = study / f"artifacts/test/{name}-g{generation}"
        merged_commit = parent / "ensemble/gen_00000000/COMMIT.json"
        pin(merged_commit, published[merged_commit.relative_to(study).as_posix()], pins)
        merged = read(merged_commit)
        original = parent / "ensemble/gen_00000000/ensemble.npz"
        pin(original, published[original.relative_to(study).as_posix()], pins)
        require(merged["files"] == {"ensemble.npz": sha(original)}, "merged payload pin differs")
        values, merged_ids = ensemble(original)
        channels = 5 if name.startswith("G5_") else 6
        require(values.shape == (480, 39, channels, 48) and np.array_equal(merged_ids, ids), "merged layout differs")
        chosen = [r["checkpoint_sha256"] for r in selection["selected"][name].values() if r["generation"] == generation]
        require(len(set(chosen)) == 1 and merged["checkpoint_sha256"] == chosen[0], "checkpoint selection differs")
        checkpoint = chosen[0]
        run = study / "runs" / name
        state = run / f"gen_{generation:08d}/state.pt"
        pin(state, checkpoint, pins)
        state_commit = state.with_name("COMMIT.json")
        pin(state_commit, sha(state_commit), pins)
        saved = read(state_commit)
        contract = saved["contract"]
        require(saved["files"].get("state.pt") == checkpoint and saved["generation"] == generation
                and contract["source_hashes"] == protocol["training_source_hashes"]
                and contract["data_hashes"] == protocol["training_data_hashes"]
                and contract["normalization_hash"] == protocol["arrays"]["TEST"]["normalization_sha256"],
                "checkpoint contract differs")
        for filename, digest in saved["files"].items():
            pin(member(state.parent, filename), digest, pins)
        snapshot = pin(run / "source_snapshot.zip", contract["settings"]["source_snapshot_sha256"], pins)
        with zipfile.ZipFile(snapshot) as archive:
            require(sorted(archive.namelist()) == sorted(contract["source_hashes"]), "snapshot inventory differs")
            require(all(hashlib.sha256(archive.read(k)).hexdigest() == v for k, v in contract["source_hashes"].items()),
                    "snapshot source bytes differ")
        expected_chunks = {str(parent / f"chunk-{offset:04d}") for offset in range(0, 480, 8)}
        require(set(merged["chunks"]) == expected_chunks, "original chunk inventory differs")
        for offset in range(0, 480, 8):
            original_root = parent / f"chunk-{offset:04d}"
            original_commit = original_root / "gen_00000000/COMMIT.json"
            pin(original_commit, merged["chunks"][str(original_root)], pins)
            meta = read(original_commit)
            original_npz = original_commit.with_name("ensemble.npz")
            pin(original_npz, meta["files"]["ensemble.npz"], pins)
            chunk, chunk_ids = ensemble(original_npz)
            require(chunk.shape == (8, 39, channels, 48) and np.array_equal(chunk_ids, ids[offset:offset + 8])
                    and chunk.tobytes() == values[offset:offset + 8].tobytes(), "chunk differs from published merged ensemble")
            require(meta["checkpoint_sha256"] == checkpoint and meta["protocol_sha256"] == PROTOCOL
                    and meta["sampler"] == SAMPLER and meta["arrays"] == protocol["arrays"]["TEST"]
                    and meta["offset"] == offset and meta["count"] == 8 and meta["n_generated"] == channels,
                    "original chunk contract differs")
            task_id = f"{name}-g{generation}/chunk-{offset:04d}"
            output = root / "chunks" / task_id
            tasks.append({"id": task_id, "output": str(output), "original": str(original_npz),
                          "original_commit": str(original_commit), "ids": ids[offset:offset + 8].tolist(),
                          "command": command(args.python, study, output, name, generation, checkpoint, arrays, offset)})
    require(len(tasks) == 780 and len({t["output"] for t in tasks}) == 780, "incomplete task plan")
    origin = read(root / "ORIGIN.json")
    require(origin["base_commit"] == COMMIT and origin["runner_sha256"] == sha(__file__), "runner origin differs")
    require(origin["tracked_diff"] == "", "tracked checkout was not clean at runner freeze")
    pin(root / "ORIGIN.json", sha(root / "ORIGIN.json"), pins)
    pin(__file__, sha(__file__), pins)
    plan = {"schema_version": 1, "protocol_sha256": PROTOCOL, "script_sha256": sha(__file__),
            "source": str(source), "study": str(study), "root": str(root), "python": args.python,
            "inputs": pins, "tasks": tasks, "pilots": list(PILOTS), "max_task_seconds": 120,
            "max_stage_seconds": 3600, "scope": "original selected TEST replay; no training or selection"}
    publish(root / "PLAN.json", plan)
    print(json.dumps({"validated": True, "tasks": len(tasks), "input_files": len(pins), "plan_sha256": sha(root / "PLAN.json")}), flush=True)


def recheck(plan):
    for path, digest in plan["inputs"].items():
        require(sha(path) == digest, f"input changed: {path}")


def gpu_inventory():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version",
                             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=10)
    return [{"index": int(row[0]), "uuid": row[1].strip(), "name": row[2].strip(),
             "total_mib": int(row[3]), "driver": row[4].strip()} for row in csv.reader(result.stdout.splitlines())]


def gpu_processes():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name,used_memory",
                             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=10)
    return [{"pid": int(row[0]), "uuid": row[1].strip(), "name": row[2].strip(), "mib": int(row[3])}
            for row in csv.reader(result.stdout.splitlines()) if row]


def child_env(plan, gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu["uuid"], PYTHONPATH=plan["source"],
               PRIGM_REQUIRE_CUDA="1", PISCES_REQUIRE_CUDA="1", PYTHONDONTWRITEBYTECODE="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", TMPDIR=str(Path(plan["root"]) / "tmp"))
    return env


def run_task(task, plan, gpu, stop, deadline, stage):
    root = Path(plan["root"])
    record = root / "tasks" / task["id"]
    output = Path(task["output"])
    try:
        require(not output.exists() and not record.exists(), f"existing attempt retained: {task['id']}")
        record.mkdir(parents=True)
    except BaseException:
        stop.set()
        raise
    env = child_env(plan, gpu)
    started = time.time()
    monotonic = time.monotonic()
    process = None
    result = {"task": task["id"], "stage": stage, "gpu": gpu, "started_epoch": started,
              "command": task["command"], "cwd": plan["source"], "pid": None,
              "environment": {k: env[k] for k in ENV_KEYS if k in env}, "peak_process_vram_mib": None}
    try:
        require(not stop.is_set() and time.monotonic() < deadline, "stage stopped")
        require(sha(__file__) == plan["script_sha256"], "runner changed during execution")
        with (record / "child.log").open("xb") as log:
            process = subprocess.Popen(task["command"], cwd=plan["source"], env=env,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            result["pid"] = process.pid
            publish(record / "START.json", result)
            print(json.dumps({"task": task["id"], "pid": process.pid, "gpu": gpu["index"], "output": str(output)}), flush=True)
            while process.poll() is None:
                require(not stop.is_set(), "another task failed or runner interrupted")
                require(time.monotonic() < min(deadline, monotonic + plan["max_task_seconds"]), "task/stage deadline exceeded")
                for row in gpu_processes():
                    if row["pid"] == process.pid:
                        require(row["uuid"] == gpu["uuid"], "child used an unassigned GPU")
                        result["peak_process_vram_mib"] = max(result["peak_process_vram_mib"] or 0, row["mib"])
                time.sleep(0.25)
        result["exit_code"] = process.returncode
        require(process.returncode == 0, f"child exit {process.returncode}")
        require(result["peak_process_vram_mib"] is not None
                and result["peak_process_vram_mib"] < gpu["total_mib"], "missing or invalid VRAM measurement")
        actual_commit = output / "gen_00000000/COMMIT.json"
        original_commit = Path(task["original_commit"])
        require(sha(original_commit) == plan["inputs"][str(original_commit)], "original commit changed")
        require(read(actual_commit) == read(original_commit), "replayed chunk metadata differs")
        actual_npz = actual_commit.with_name("ensemble.npz")
        require(sha(task["original"]) == plan["inputs"][task["original"]], "original payload changed")
        result["comparison"] = compare_payload(actual_npz, task["original"], np.asarray(task["ids"]))
        result["output_commit_sha256"] = sha(actual_commit)
        result["status"] = "passed"
    except BaseException as error:
        stop.set()
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        result["exit_code"] = None if process is None else process.returncode
        result["ended_epoch"] = time.time()
        result["wall_seconds"] = time.monotonic() - monotonic
        result["log_sha256"] = sha(record / "child.log") if (record / "child.log").exists() else None
        publish(record / "RESULT.json", result)
        print(json.dumps({"task": task["id"], "status": result.get("status"), "seconds": result["wall_seconds"]}), flush=True)
    return result


def verify_result(task, plan):
    record = Path(plan["root"]) / "tasks" / task["id"]
    result = read(record / "RESULT.json")
    require(result["status"] == "passed", f"missing passed task: {task['id']}")
    actual_commit = Path(task["output"]) / "gen_00000000/COMMIT.json"
    require(sha(actual_commit) == result["output_commit_sha256"]
            and read(actual_commit) == read(task["original_commit"]), "output commit changed")
    comparison = compare_payload(actual_commit.with_name("ensemble.npz"), task["original"], np.asarray(task["ids"]))
    require(comparison == result["comparison"] and sha(record / "child.log") == result["log_sha256"],
            "output or execution log changed")
    return sha(record / "RESULT.json")


def execute(args):
    root = Path(args.root)
    plan = read(root / "PLAN.json")
    require(plan["root"] == str(root) and plan["script_sha256"] == sha(__file__), "plan/runner mismatch")
    require(not (root / f"{args.stage.upper()}_START.json").exists(), "stage already attempted; preserve and inspect it")
    recheck(plan)
    gpus = gpu_inventory()
    require([g["index"] for g in gpus] == list(range(4))
            and all(g["total_mib"] == 12288 and "3080 Ti" in g["name"] for g in gpus), "GPU allocation differs")
    background = gpu_processes()
    require(all(p["uuid"] == gpus[0]["uuid"] and p["name"].endswith("/gnome-remote-desktop-daemon")
                for p in background), "unexpected existing GPU process")
    env = child_env(plan, gpus[0])
    version_code = "import json,platform,numpy,torch; print(json.dumps(dict(python=platform.python_version(),numpy=numpy.__version__,torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())))"
    versions = json.loads(subprocess.check_output([plan["python"], "-B", "-c", version_code], env={**env, "CUDA_VISIBLE_DEVICES": ""}, text=True, timeout=30))
    require(versions == {"python": "3.11.15", "numpy": "2.4.3", "torch": "2.6.0+cu124", "cuda": "12.4", "cudnn": 90100}, "execution environment differs")
    tasks = {task["id"]: task for task in plan["tasks"]}
    if args.stage == "pilot":
        selected = [tasks[key] for key in PILOTS]
    else:
        pilot = read(root / "PILOT_COMPLETE.json")
        require(pilot["status"] == "passed" and set(pilot["results"]) == set(PILOTS)
                and pilot["plan_sha256"] == sha(root / "PLAN.json"), "pilots did not pass this plan")
        for key in PILOTS:
            require(verify_result(tasks[key], plan) == pilot["results"][key], "pilot receipt changed")
        selected = [task for key, task in tasks.items() if key not in PILOTS]
    stop = threading.Event()
    def interrupt(signum, frame):
        stop.set()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupt)
    start = {"pid": os.getpid(), "started_epoch": time.time(), "host": platform.node(),
             "script_sha256": sha(__file__), "plan_sha256": sha(root / "PLAN.json"),
             "gpu_inventory": gpus, "background": background, "versions": versions,
             "tasks": [task["id"] for task in selected]}
    publish(root / f"{args.stage.upper()}_START.json", start)
    deadline = time.monotonic() + (300 if args.stage == "pilot" else plan["max_stage_seconds"])
    pending = queue.Queue()
    for task in selected:
        pending.put(task)
    def worker(gpu):
        while not stop.is_set():
            try:
                task = pending.get_nowait()
            except queue.Empty:
                return
            run_task(task, plan, gpu, stop, deadline, args.stage)
    try:
        with ThreadPoolExecutor(max_workers=2 if args.stage == "pilot" else 4) as pool:
            if args.stage == "pilot":
                futures = [pool.submit(run_task, task, plan, gpus[i], stop, deadline, args.stage)
                           for i, task in enumerate(selected)]
            else:
                futures = [pool.submit(worker, gpu) for gpu in gpus]
            for future in futures:
                future.result()
        require(not stop.is_set(), "execution interrupted")
        recheck(plan)
        completed = PILOTS if args.stage == "pilot" else tasks
        results = {}
        for key in completed:
            results[key] = verify_result(tasks[key], plan)
        publish(root / f"{args.stage.upper()}_COMPLETE.json", {
            "status": "passed", "plan_sha256": sha(root / "PLAN.json"), "results": results,
            "original_inputs_unchanged": True, "ended_epoch": time.time(),
            "wall_seconds": time.time() - start["started_epoch"]})
    except BaseException as error:
        stop.set()
        publish(root / f"{args.stage.upper()}_FAILED.json", {"error": f"{type(error).__name__}: {error}",
                "ended_epoch": time.time(), "inputs_and_attempts_preserved": True})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate", "pilot", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--public-repo", type=Path)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    require(args.root.is_dir() and args.root.absolute() == args.root.resolve(), "owned root must exist without symlink redirection")
    args.root = args.root.resolve()
    # Children run from the source directory. Keep the caller's interpreter path,
    # including a virtual environment's symlink, while leaving bare names on PATH.
    if os.path.dirname(args.python):
        args.python = str(Path(args.python).absolute())
    if args.stage == "validate":
        require(all((args.study, args.source, args.public_repo)), "validate needs study, source and public repo")
        for name in ("study", "source", "public_repo"):
            path = getattr(args, name)
            require(path.is_dir() and path.absolute() == path.resolve(), "input root redirects through symlinks")
            path = path.resolve()
            require(not args.root.is_relative_to(path), "replay output must be outside input roots")
            setattr(args, name, path)
    with (args.root / "RUNNER.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (args.root / "tmp").mkdir(exist_ok=True)
        if args.stage == "validate":
            validate(args)
        else:
            execute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
