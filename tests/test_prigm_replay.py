"""Check the ensemble-replay runner with local files and controlled execution.

A matching replay must retain the ordered window IDs, the raw ensemble
values and the exact NPZ file bytes. The tests distinguish these requirements:
changing compression can preserve the arrays while changing the file.
Other cases exercise declared input hashes and path restrictions.

These offline fixtures check the runner's comparison rules; they do not
themselves reproduce the original trained models' predictions.
"""

import importlib.util
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "replay_prigm_ensembles", Path(__file__).resolve().parents[1] / "scripts/replay_prigm_ensembles.py")
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def payload(path, values=None, ids=None, compressed=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2) if values is None else values
    ids = np.array(["a", "b"]) if ids is None else np.asarray(ids)
    save = np.savez_compressed if compressed else np.savez
    save(path, ensemble=values, ids=ids)


def test_fixed_unique_780_chunks():
    jobs = [(f"model{i}", i) for i in range(13)]
    layout = replay.task_layout(jobs)
    assert len(layout) == len(set(layout)) == 780
    for name, generation in jobs:
        offsets = [offset for n, g, offset in layout if (n, g) == (name, generation)]
        assert offsets == list(range(0, 480, 8))
    with pytest.raises(RuntimeError, match="thirteen unique"):
        replay.task_layout(jobs[:-1] + jobs[:1])


def test_original_command_and_uuid_environment(monkeypatch, tmp_path):
    study, out = tmp_path / "original", tmp_path / "replay"
    cmd = replay.command("/env/python", study, out, "D_s7", 9560, "abc", study / "test.npz", 472)
    assert cmd[:5] == ["/env/python", "-B", "-u", "-m", "prigm.run_eval"]
    options = dict(zip(cmd[5::2], cmd[6::2]))
    assert options == {"--run-root": str(study / "runs/D_s7"), "--generation": "9560",
                       "--checkpoint-sha256": "abc", "--out": str(out),
                       "--arrays": str(study / "test.npz"), "--split": "TEST", "--n-eval": "50",
                       "--eval-seed": "7", "--offset": "472", "--limit": "8",
                       "--study-root": str(study), "--protocol-sha256": replay.PROTOCOL}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "wrong")
    monkeypatch.setenv("PYTHONPATH", "current-source")
    env = replay.child_env({"root": str(out), "source": "/frozen/source"}, {"uuid": "GPU-owned"})
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-owned"
    assert env["PYTHONPATH"] == "/frozen/source"
    assert env["TMPDIR"] == str(out / "tmp")
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


@pytest.mark.parametrize("change,error", [("ids", "ordered IDs"), ("values", "raw ensemble"),
                                           ("archive", "NPZ bytes"), ("dtype", "invalid ensemble"),
                                           ("nan", "invalid ensemble")])
def test_payload_requires_all_three_equalities(tmp_path, change, error):
    original, actual = tmp_path / "original.npz", tmp_path / "actual.npz"
    payload(original)
    actual.write_bytes(original.read_bytes())
    assert replay.compare_payload(actual, original, ["a", "b"])["npz_bytes_exact"]
    values, ids = replay.ensemble(original)
    if change == "ids":
        ids = ids[::-1]
    elif change == "values":
        values[0, 0, 0, 0] += 1
    elif change == "dtype":
        values = values.astype(np.float64)
    elif change == "nan":
        values[0, 0, 0, 0] = np.nan
    payload(actual, values, ids, compressed=change == "archive")
    with pytest.raises(RuntimeError, match=error):
        replay.compare_payload(actual, original, ["a", "b"])


def test_input_pins_and_root_escape(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    source = root / "file.py"
    source.write_text("original")
    pins = {}
    replay.pin(source, replay.sha(source), pins)
    replay.recheck({"inputs": pins})
    source.write_text("changed")
    with pytest.raises(RuntimeError, match="input changed"):
        replay.recheck({"inputs": pins})
    outside = tmp_path / "outside.py"
    outside.write_text("outside")
    (root / "linked.py").symlink_to(outside)
    for name in ("../outside.py", "linked.py"):
        with pytest.raises(RuntimeError, match="escapes"):
            replay.member(root, name)


def test_receipts_never_overwrite(tmp_path):
    path = tmp_path / "receipt.json"
    replay.publish(path, {"original": True})
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match="refusing existing receipt"):
        replay.publish(path, {"original": False})
    assert path.read_bytes() == original


@pytest.fixture
def task_fixture(tmp_path, monkeypatch):
    original = tmp_path / "original/ensemble.npz"
    payload(original)
    commit = original.with_name("COMMIT.json")
    commit.write_text(json.dumps({"files": {"ensemble.npz": replay.sha(original)}}))
    root = tmp_path / "owned"
    root.mkdir()
    task = {"id": "model/chunk-0000", "output": str(root / "chunks/model/chunk-0000"),
            "original": str(original), "original_commit": str(commit), "ids": ["a", "b"],
            "command": ["/fake/python", "-m", "prigm.run_eval"]}
    plan = {"root": str(root), "source": "/frozen/source", "script_sha256": replay.sha(replay.__file__),
            "inputs": {str(original): replay.sha(original), str(commit): replay.sha(commit)},
            "max_task_seconds": 10}
    gpu = {"uuid": "GPU-owned", "index": 0, "total_mib": 12288}
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    return task, plan, gpu


class Child:
    pid = 12345

    def __init__(self, exit_code=0, running=False):
        self.returncode = None
        self.exit_code = exit_code
        self.running = running
        self.polls = 0
        self.terminated = False

    def poll(self):
        self.polls += 1
        if self.polls > 1 and not self.running:
            self.returncode = self.exit_code
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15
        self.running = False
        self.exit_code = -15

    def wait(self, timeout):
        return self.returncode


@pytest.mark.parametrize("failure", [None, "exit", "bytes", "device", "interrupt"])
def test_task_proof_or_fail_closed(task_fixture, monkeypatch, failure):
    task, plan, gpu = task_fixture
    stop = threading.Event()
    child = Child(exit_code=7 if failure == "exit" else 0, running=failure in {"device", "interrupt"})

    def launch(cmd, **kwargs):
        assert cmd == task["command"]
        assert kwargs["cwd"] == plan["source"]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == gpu["uuid"]
        output = Path(task["output"]) / "gen_00000000"
        output.mkdir(parents=True)
        for name in ("ensemble.npz", "COMMIT.json"):
            (output / name).write_bytes((Path(task["original"]).parent / name).read_bytes())
        if failure == "bytes":
            values, ids = replay.ensemble(task["original"])
            values[0, 0, 0, 0] += 1
            payload(output / "ensemble.npz", values, ids)
        if failure == "interrupt":
            stop.set()
        return child

    monkeypatch.setattr(replay.subprocess, "Popen", launch)
    monkeypatch.setattr(replay, "gpu_processes", lambda: [{"pid": child.pid,
                        "uuid": "GPU-wrong" if failure == "device" else gpu["uuid"], "mib": 1000}])
    if failure:
        with pytest.raises(RuntimeError):
            replay.run_task(task, plan, gpu, stop, time.monotonic() + 30, "pilot")
        assert stop.is_set()
        assert child.terminated == (failure in {"device", "interrupt"})
    else:
        result = replay.run_task(task, plan, gpu, stop, time.monotonic() + 30, "pilot")
        assert result["comparison"]["raw_array_bytes_exact"]
        assert result["peak_process_vram_mib"] == 1000
        assert replay.verify_result(task, plan)
        (Path(plan["root"]) / "tasks" / task["id"] / "child.log").write_text("changed")
        with pytest.raises(RuntimeError, match="log changed"):
            replay.verify_result(task, plan)
    receipt = Path(plan["root"]) / "tasks" / task["id"] / "RESULT.json"
    assert replay.read(receipt)["status"] == ("failed" if failure else "passed")
    assert Path(task["output"]).exists()
    replay.recheck(plan)


def test_existing_attempt_stops_siblings_without_touching_output(task_fixture):
    task, plan, gpu = task_fixture
    output = Path(task["output"])
    output.mkdir(parents=True)
    retained = output / "partial"
    retained.write_text("preserve")
    stop = threading.Event()
    with pytest.raises(RuntimeError, match="existing attempt retained"):
        replay.run_task(task, plan, gpu, stop, time.monotonic() + 30, "run")
    assert stop.is_set() and retained.read_text() == "preserve"


@pytest.mark.parametrize("protected", ["study", "source", "public_repo"])
@pytest.mark.parametrize("relative", ["root", "input"])
def test_cli_rejects_mixed_path_containment_before_lock(tmp_path, monkeypatch, protected, relative):
    monkeypatch.chdir(tmp_path)
    inputs = {name: tmp_path / name for name in ("study", "source", "public_repo")}
    for path in inputs.values():
        path.mkdir()
    root = inputs[protected] / "owned"
    root.mkdir()
    root_arg = root.relative_to(tmp_path) if relative == "root" else root
    if relative == "input":
        inputs[protected] = inputs[protected].relative_to(tmp_path)
    def forbidden_lock(*args):
        pytest.fail("contained output reached lock acquisition")
    monkeypatch.setattr(replay.fcntl, "flock", forbidden_lock)
    argv = ["validate", "--root", str(root_arg)]
    for name, path in inputs.items():
        argv += ["--" + name.replace("_", "-"), str(path)]
    with pytest.raises(RuntimeError, match="outside input roots"):
        replay.main(argv)
    assert not (root / "RUNNER.lock").exists()
    assert not (root / "tmp").exists()


@pytest.mark.parametrize("argument", ["root", "study", "source", "public_repo"])
@pytest.mark.parametrize("redirection", ["parent", "symlink"])
def test_cli_retains_path_redirection_guards(tmp_path, monkeypatch, argument, redirection):
    monkeypatch.chdir(tmp_path)
    paths = {name: tmp_path / name for name in ("root", "study", "source", "public_repo")}
    for path in paths.values():
        path.mkdir()
    if redirection == "parent":
        (tmp_path / "spare").mkdir()
        paths[argument] = tmp_path / "spare" / ".." / argument
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(paths[argument], target_is_directory=True)
        paths[argument] = alias
    def forbidden_lock(*args):
        pytest.fail("redirected path reached lock acquisition")
    monkeypatch.setattr(replay.fcntl, "flock", forbidden_lock)
    argv = ["validate"]
    for name, path in paths.items():
        argv += ["--" + name.replace("_", "-"), str(path)]
    with pytest.raises(RuntimeError, match="symlink"):
        replay.main(argv)
    assert not (tmp_path / "root/RUNNER.lock").exists()


@pytest.mark.parametrize("python_mode", ["absolute", "bare", "relative_symlink"])
def test_relative_cli_roots_keep_cpu_child_output_under_owned_root(tmp_path, monkeypatch, python_mode):
    monkeypatch.chdir(tmp_path)
    root, source, study, public = [tmp_path / name for name in ("owned", "source", "study", "public")]
    for path in (root, source, study, public):
        path.mkdir()
    python_arg = sys.executable
    entry = None
    if python_mode == "relative_symlink":
        entry = tmp_path / "env/bin/python"
        entry.parent.mkdir(parents=True)
        entry.symlink_to(sys.executable)
        python_arg = str(entry.relative_to(tmp_path))
    elif python_mode == "bare":
        path_bin = tmp_path / "path-bin"
        path_bin.mkdir()
        (path_bin / "python3").symlink_to(sys.executable)
        monkeypatch.setenv("PATH", str(path_bin))
        python_arg = "python3"
    package = source / "prigm"
    package.mkdir()
    (package / "__init__.py").write_text("")
    original = tmp_path / "original/ensemble.npz"
    payload(original)
    commit = original.with_name("COMMIT.json")
    commit.write_text(json.dumps({"files": {"ensemble.npz": replay.sha(original)}}))
    (package / "run_eval.py").write_text(
        "import argparse,json,os,pathlib,shutil,sys,time\n"
        "p=argparse.ArgumentParser()\n"
        "for flag in ('out','study-root','run-root','arrays'): p.add_argument('--'+flag)\n"
        "a,_=p.parse_known_args(); assert os.environ['CUDA_VISIBLE_DEVICES']==''\n"
        "out=pathlib.Path(a.out)/'gen_00000000'; out.mkdir(parents=True)\n"
        f"original=pathlib.Path({str(original.parent)!r})\n"
        "for name in ('ensemble.npz','COMMIT.json'): shutil.copyfile(original/name,out/name)\n"
        "(out/'where.json').write_text(json.dumps(dict(cwd=os.getcwd(),options=vars(a),executable=sys.executable)))\n"
        "time.sleep(0.3)\n")
    captured = {}
    def fixture_validate(args):
        captured["args"] = args
        output = args.root / "chunks/model/chunk-0000"
        task = {"id": "model/chunk-0000", "output": str(output), "original": str(original),
                "original_commit": str(commit), "ids": ["a", "b"],
                "command": replay.command(args.python, args.study, output, "D_s7", 9560,
                                          "fixture", args.study / "test.npz", 0)}
        plan = {"root": str(args.root), "source": str(args.source), "study": str(args.study),
                "script_sha256": replay.sha(replay.__file__), "max_task_seconds": 5,
                "inputs": {str(p): replay.sha(p) for p in (original, commit)}, "tasks": [task]}
        replay.publish(args.root / "PLAN.json", plan)
    real_env, real_popen = replay.child_env, replay.subprocess.Popen
    def cpu_env(plan, gpu):
        return {**real_env(plan, gpu), "CUDA_VISIBLE_DEVICES": "", "PRIGM_REQUIRE_CUDA": "0",
                "PISCES_REQUIRE_CUDA": "0"}
    def launch(*args, **kwargs):
        captured["process"] = real_popen(*args, **kwargs)
        return captured["process"]
    gpu = {"index": 0, "uuid": "GPU-fixture", "total_mib": 12288}
    def fixture_execute(args):
        captured["execute_root"] = args.root
        plan = replay.read(args.root / "PLAN.json")
        captured["result"] = replay.run_task(plan["tasks"][0], plan, gpu, threading.Event(),
                                             time.monotonic() + 10, args.stage)
    monkeypatch.setattr(replay, "validate", fixture_validate)
    monkeypatch.setattr(replay, "execute", fixture_execute)
    monkeypatch.setattr(replay, "child_env", cpu_env)
    monkeypatch.setattr(replay.subprocess, "Popen", launch)
    monkeypatch.setattr(replay, "gpu_processes", lambda: [
        {"pid": captured["process"].pid, "uuid": gpu["uuid"], "mib": 1}])
    replay.main(["validate", "--root", "owned", "--study", "study", "--source", "source",
                 "--public-repo", "public", "--python", python_arg])
    replay.main(["pilot", "--root", "owned"])
    args = captured["args"]
    assert (args.root, args.study, args.source, args.public_repo) == (root, study, source, public)
    assert captured["execute_root"] == root
    assert captured["result"]["status"] == "passed"
    output = root / "chunks/model/chunk-0000/gen_00000000"
    assert (output / "ensemble.npz").read_bytes() == original.read_bytes()
    where = replay.read(output / "where.json")
    assert where["cwd"] == str(source)
    assert all(Path(value).is_absolute() for value in where["options"].values())
    assert not (source / "owned").exists()
    if python_mode == "relative_symlink":
        assert entry.is_symlink()
        assert args.python == str(entry) and args.python != str(entry.resolve())
        assert captured["result"]["command"][0] == str(entry)
        assert where["executable"] == str(entry)
    else:
        assert args.python == python_arg
        assert captured["result"]["command"][0] == python_arg
