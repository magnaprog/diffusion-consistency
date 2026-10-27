"""Check selection and reuse rules for the retrospective checkpoint study on CPU.

Construct local records to test which saved model updates may be reused and
how validation scores determine the P25 and P50 selections before TEST is
authorized. Mutations must not change the methods, seeds, milestones, draw
counts or validation-only selection rules.

Some fixtures replace expensive ensemble checks after separate synthetic
ensemble tests cover those checks. The results concern coordinator behavior;
they do not rerun the historical study or restore an unexposed TEST set.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/prigm_checkpoint_followup.py"
spec = importlib.util.spec_from_file_location("checkpoint_followup", SOURCE)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


@pytest.fixture(autouse=True)
def reset_stage_state(monkeypatch):
    study.STOP.clear()
    monkeypatch.setattr(study, "DEADLINE", float("inf"))
    yield
    study.STOP.clear()


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(study.json_bytes(value))
    return study.sha256_file(path)


@pytest.fixture
def scope():
    return {"exposure": "historical_test_exposed", "arms": list(study.ARMS),
            "training_seeds": list(study.SEEDS), "milestones": list(study.UPDATES),
            "coefficients": {"D": 0.0, "O-S": .0005},
            "VAL": {**study.SAMPLER, "n_windows": 319, "chunk_windows": 8, "jobs": 30},
            "TEST": {**study.SAMPLER, "n_windows": 480, "chunk_windows": 8, "jobs": 30,
                     "authenticated_reuse_jobs": 10, "new_jobs": 20},
            "no_training": True, "accepted_Table1_unchanged": True,
            "original_protocol_sha256": study.ORIGINAL, "original_selection_sha256": study.SELECTION,
            "extension_summary_sha256": study.SUMMARY,
            "policies": {"P25": "original_raw_VAL25_selection", "P50": "minimum_complete_VAL50_month_balanced_ES_earlier_exact_tie", "F": "update_29636"}}


@pytest.fixture
def protocol():
    jobs = [{"id": study.job_key(a, s, g), "arm": a, "seed": s, "generation": g,
             "run_root": f"/original/runs/{a}_s{s}", "checkpoint_sha256": digest((a, s, g))}
            for a in study.ARMS for s in study.SEEDS for g in study.UPDATES]
    return {"jobs": jobs, "P25": {f"{a}_s{s}": {"generation": 9560} for a in study.ARMS for s in study.SEEDS},
            "scope_sha256": "a" * 64, "samplers": {k: dict(study.SAMPLER) for k in ("VAL", "TEST")},
            "arrays": {"TEST": {"sha256": "b" * 64}, "VAL": {"sha256": "c" * 64}},
            "array_root": "/original/arrays", "inference_source_hashes": {"prigm/run_eval.py": "d" * 64}}


def test_approved_scope_and_complete_grid(scope):
    study.validate_scope(scope)


@pytest.mark.parametrize("damage", [
    lambda s: s.update(arms=["D", "G5"]),
    lambda s: s.update(training_seeds=[7, 13]),
    lambda s: s.update(milestones=[9560, 29636]),
    lambda s: s["VAL"].update(n_eval=25),
    lambda s: s["VAL"].update(n_draws=40),
    lambda s: s["VAL"].update(eval_seed=13),
    lambda s: s["TEST"].update(jobs=6),
    lambda s: s["TEST"].update(authenticated_reuse_jobs=13),
    lambda s: s.update(no_training=False),
    lambda s: s["policies"].update(P50="minimum_TEST_score"),
    lambda s: s.update(original_protocol_sha256="0" * 64),
])
def test_scope_cannot_expand_or_change_selection(scope, damage):
    damage(scope)
    with pytest.raises(study.ContractMismatch):
        study.validate_scope(scope)


def test_selection_uses_all_val_scores_and_earlier_exact_tie(protocol):
    scores = {j["id"]: 2.0 for j in protocol["jobs"]}
    selected = study.selection_from_scores(protocol, scores)
    assert all(v == {"P25": 9560, "P50": 2868, "F": 29636} for v in selected.values())
    scores[study.job_key("D", 7, 22944)] = 1.0
    changed = study.selection_from_scores(protocol, scores)
    assert changed["D_s7"]["P50"] == 22944
    assert changed["O-S_s7"]["P50"] == 2868


@pytest.mark.parametrize("damage", ["missing", "extra", "nan", "inf"])
def test_incomplete_or_nonfinite_val_cannot_select(protocol, damage):
    scores = {j["id"]: 2.0 for j in protocol["jobs"]}
    if damage == "missing":
        scores.pop(next(iter(scores)))
    elif damage == "extra":
        scores["G5_s7-g29636"] = 0
    else:
        scores[next(iter(scores))] = float(damage)
    with pytest.raises(study.ContractMismatch):
        study.selection_from_scores(protocol, scores)


def test_bootstrap_matches_manual_year_occurrences_and_unequal_month_windows():
    months = np.array([y * 100 + m for y in range(2005, 2025) for m in (11, 11, 12)])
    x = np.arange(len(months), dtype=float)
    values = np.array([np.sin(x / (i + 1)) + i / 50 for i in range(5)])
    result = study.curve_statistics(values, months)
    draws = np.random.Generator(np.random.PCG64(20261001)).integers(0, 20, (10000, 20))
    ym = np.array([[np.mean([row[months == y * 100 + m].mean() for m in (11, 12)])
                    for y in range(2005, 2025)] for row in values])
    counts = np.bincount(ym[:, draws].mean(axis=2).argmin(axis=0), minlength=5)
    assert list(result["bootstrap_selection_counts"].values()) == counts.tolist()
    np.testing.assert_allclose(result["means"], ym.mean(axis=1))
    assert sum(result["bootstrap_selection_counts"].values()) == 10000
    assert sum(result["leave_year_out_counts"].values()) == 20


@pytest.fixture
def ensemble(tmp_path, protocol):
    ids = [f"2005-08-{i+1:02d}T00:00-s52" for i in range(9)]
    job = protocol["jobs"][0]
    p = copy.deepcopy(protocol)
    parent = tmp_path / "original" / job["id"]
    rng = np.random.default_rng(401)
    values = np.zeros((9, 39, 6, 48), dtype=np.float32)
    values[..., 18:30] = rng.normal(size=(9, 39, 6, 12))
    roots = []
    for offset in (0, 8):
        path = parent / f"chunk-{offset:04d}"
        meta = {"arm": job["arm"], "seed": job["seed"], "generation_updates": job["generation"],
                "checkpoint_sha256": job["checkpoint_sha256"], "n_generated": 6,
                "protocol_sha256": "e" * 64, "arrays": p["arrays"]["TEST"],
                "sampler": p["samplers"]["TEST"], "state_convention": study.STATE_CONVENTION,
                "noise_convention": study.NOISE_CONVENTION, "inference_source_sha256": "d" * 64,
                "offset": offset, "count": len(ids[offset:offset + 8])}
        from prigm.artifacts import publish_ensemble
        publish_ensemble(path, values[offset:offset + 8], ids[offset:offset + 8], meta)
        roots.append(path)
    merged = parent / "ensemble"
    study.merge_ensembles(roots, merged, ids)
    return merged, job, p, ids, values


def test_complete_reuse_authenticates_exact_chunks(ensemble):
    root, job, p, ids, values = ensemble
    actual, meta = study.verify_ensemble(root, job, p, "TEST", "e" * 64, ids, complete=True)
    np.testing.assert_array_equal(actual, values)
    assert len(meta["chunks"]) == 2


@pytest.mark.parametrize("field,value", [
    ("sampler", {"n_eval": 25, "n_draws": 39, "eval_seed": 7}),
    ("sampler", {"n_eval": 50, "n_draws": 39, "eval_seed": 13}),
    ("noise_convention", "different-noise"), ("state_convention", "noisy_flanks"),
    ("checkpoint_sha256", "0" * 64), ("generation_updates", 9560),
    ("seed", True), ("arm", "O-S"), ("inference_source_sha256", "0" * 64),
])
def test_reuse_rejects_wrong_metadata(ensemble, field, value):
    root, job, p, ids, _ = ensemble
    commit = root / "gen_00000000/COMMIT.json"
    meta = study.read(commit)
    meta[field] = value
    save(commit, meta)
    with pytest.raises(study.ContractMismatch):
        study.verify_ensemble(root, job, p, "TEST", "e" * 64, ids, complete=True)


def test_reuse_detects_rebound_wrong_chunk_offset(ensemble):
    root, job, p, ids, _ = ensemble
    chunk = root.parent / "chunk-0008/gen_00000000/COMMIT.json"
    meta = study.read(chunk)
    meta["offset"] = 0
    pin = save(chunk, meta)
    commit = root / "gen_00000000/COMMIT.json"
    merged = study.read(commit)
    merged["chunks"][str(chunk.parent.parent.resolve())] = pin
    save(commit, merged)
    with pytest.raises(study.ContractMismatch, match="layout"):
        study.verify_ensemble(root, job, p, "TEST", "e" * 64, ids, complete=True)


def test_reuse_rejects_missing_chunk(ensemble):
    root, job, p, ids, _ = ensemble
    (root.parent / "chunk-0008/gen_00000000/ensemble.npz").unlink()
    with pytest.raises((study.ContractMismatch, OSError, RuntimeError)):
        study.verify_ensemble(root, job, p, "TEST", "e" * 64, ids, complete=True)


def test_no_gpu_admission_means_no_device_probe_or_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: pytest.fail("unexpected GPU probe"))
    with pytest.raises(study.ContractMismatch, match="admission"):
        study.gpu_admission(tmp_path, "a" * 64, "pilot", None, None)


@pytest.mark.parametrize("damage", ["stage", "protocol", "uuid", "duplicate"])
def test_gpu_admission_must_match_main_stage_protocol_and_device(tmp_path, monkeypatch, damage):
    row = {"approved_by": "main", "protocol_sha256": "a" * 64, "stages": ["pilot"],
           "devices": [{"index": 0, "uuid": "GPU-one"}]}
    if damage == "stage": row["stages"] = ["sample-val"]
    if damage == "protocol": row["protocol_sha256"] = "b" * 64
    if damage == "uuid": row["devices"][0]["uuid"] = "GPU-two"
    if damage == "duplicate": row["devices"].append(dict(row["devices"][0]))
    path = tmp_path / "admission.json"
    pin = save(path, row)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "0, GPU-one\n")
    with pytest.raises(study.ContractMismatch):
        study.gpu_admission(tmp_path, "a" * 64, "pilot", path, pin)


def test_sampling_command_preserves_weights_noise_layout_and_original_entrypoint(tmp_path, protocol):
    job = protocol["jobs"][1]
    command = study.sample_command(tmp_path, protocol, "a" * 64, job, "VAL", 312, tmp_path / "chunk")
    assert "from prigm.run_eval import main" in command[3]
    options = dict(zip(command[4::2], command[5::2]))
    assert options["--n-eval"] == "50" and options["--eval-seed"] == "7"
    assert options["--limit"] == "8" and options["--offset"] == "312"
    assert options["--run-root"] == job["run_root"] and options["--generation"] == "9560"
    assert options["--checkpoint-sha256"] == job["checkpoint_sha256"]


def test_test_work_requires_selection_before_loading_arrays_or_launching(tmp_path, protocol, monkeypatch):
    monkeypatch.setattr(study, "identity", lambda *a: pytest.fail("TEST read before receipt"))
    monkeypatch.setattr(study, "launch", lambda *a: pytest.fail("GPU launch before receipt"))
    with pytest.raises(RuntimeError):
        study.sample_grid(tmp_path, protocol, "a" * 64, "TEST", [{"index": 0, "uuid": "GPU-one"}])


def test_new_receipts_are_immutable_and_cannot_escape(tmp_path):
    path = tmp_path / "receipt.json"
    pin = study.publish(tmp_path, path, {"value": 1})
    assert study.publish(tmp_path, path, {"value": 1}) == pin
    with pytest.raises(study.ContractMismatch, match="immutable"):
        study.publish(tmp_path, path, {"value": 2})
    with pytest.raises(study.ContractMismatch, match="escapes"):
        study.publish(tmp_path, tmp_path.parent / "outside.json", {})
    linked = tmp_path / "linked.json"
    linked.symlink_to(path)
    with pytest.raises(study.ContractMismatch, match="symlink"):
        study.publish(tmp_path, linked, {})


def test_failed_child_keeps_log_command_timing_and_exit_status(tmp_path, protocol):
    p = {**protocol, "source_root": str(tmp_path), "runner_sha256": "f" * 64}
    with pytest.raises(study.ContractMismatch, match="child failed"):
        study.launch(tmp_path, p, [study.sys.executable, "-c", "raise SystemExit(17)"], "failure")
    receipt = study.read(next((tmp_path / "receipts").glob("*.json")))
    assert receipt["exit_code"] == 17 and receipt["elapsed_seconds"] >= 0
    assert receipt["gpu"] is None and receipt["environment"]["CUDA_VISIBLE_DEVICES"] == ""
    assert study.sha256_file(Path(receipt["log"])) == receipt["log_sha256"]


def test_score_validation_keeps_raw_population_and_reduction(protocol):
    job = protocol["jobs"][0]
    ids = ["2005-08-01", "2005-08-02", "2005-09-01"]
    row = {"ids": ids, "months": [200508, 200508, 200509], "scores": [1., 3., 10.],
           "n": 3, "mean": 14/3, "month_balanced": 6., "corrected": False,
           "procedure": "diffusion", "arrays": protocol["arrays"]["TEST"],
           "protocol_sha256": "a" * 64, "checkpoint_sha256": job["checkpoint_sha256"],
           "ensemble_commit_sha256": "b" * 64}
    values, months = study.validate_score(row, job, protocol, "TEST", "a" * 64, ids, "b" * 64)
    assert study.month_balanced_mean(values, months) == 6.
    row["corrected"] = True
    with pytest.raises(study.ContractMismatch, match="raw"):
        study.validate_score(row, job, protocol, "TEST", "a" * 64, ids, "b" * 64)


@pytest.fixture
def selected_study(tmp_path, protocol, monkeypatch):
    p = copy.deepcopy(protocol)
    p.update(schema_version=2, exposure="historical_test_exposed")
    for job in p["jobs"][:10]:
        old = tmp_path / "historical" / job["id"]
        pin = save(old / "gen_00000000/COMMIT.json", {"identity": job["id"]})
        job["reuse"] = {"root": str(old), "commit_sha256": pin}
    pin = save(tmp_path / "PROTOCOL.json", p)
    save(tmp_path / "REUSE_AUTHENTICATED.json", {"protocol_sha256": pin,
         "scope_sha256": p["scope_sha256"], "jobs": {j["id"]: j["reuse"] for j in p["jobs"] if "reuse" in j}})
    ids = ["2005-08-01", "2005-09-01"]
    monkeypatch.setattr(study, "identity", lambda *a: (None, ids, None))
    # Complete chunk authentication is covered by the real synthetic ensembles above.
    monkeypatch.setattr(study, "verify_ensemble", lambda *a, **k: None)
    for job in p["jobs"]:
        parent = tmp_path / "artifacts/val" / job["id"]
        cp = save(parent / "ensemble/gen_00000000/COMMIT.json", {"identity": job["id"]})
        mean = float(abs(job["generation"] - 16252))
        save(parent / "score-raw.json", {"ids": ids, "months": [200508, 200509],
             "scores": [mean, mean], "n": 2, "mean": mean, "month_balanced": mean,
             "corrected": False, "procedure": "diffusion", "arrays": p["arrays"]["VAL"],
             "protocol_sha256": pin, "checkpoint_sha256": job["checkpoint_sha256"],
             "ensemble_commit_sha256": cp})
    study.freeze_selection(tmp_path, p, pin)
    return tmp_path, p, pin


def test_complete_val_selection_satisfies_unchanged_test_barrier(selected_study):
    root, p, pin = selected_study
    receipt = study.test_gate(root, p, pin)
    assert len(receipt["test_jobs"]) == len(receipt["test_checkpoint_hashes"]) == 30
    assert all(v == {"P25": 9560, "P50": 16252, "F": 29636} for v in receipt["policies"].values())


@pytest.mark.parametrize("damage", ["score", "selection", "reuse", "old_commit"])
def test_test_gate_rejects_postselection_drift(selected_study, damage):
    root, p, pin = selected_study
    path = {"score": root / "artifacts/val" / p["jobs"][0]["id"] / "score-raw.json",
            "selection": root / "SELECTION.json", "reuse": root / "REUSE_AUTHENTICATED.json",
            "old_commit": Path(p["jobs"][0]["reuse"]["root"]) / "gen_00000000/COMMIT.json"}[damage]
    path.chmod(0o644)
    save(path, {"tampered": True})
    with pytest.raises(RuntimeError):
        study.test_gate(root, p, pin)


def test_sampling_queue_uses_30_val_and_only_20_missing_test(protocol, tmp_path, monkeypatch):
    p = copy.deepcopy(protocol)
    for job in p["jobs"][:10]:
        job["reuse"] = {}
    commands, scored = [], []
    monkeypatch.setattr(study, "test_gate", lambda *a: None)
    monkeypatch.setattr(study, "identity", lambda p, s: (None, list(range(study.COUNTS[s])), None))
    monkeypatch.setattr(study, "launch", lambda r, p, cmd, label, d: commands.append(cmd))
    monkeypatch.setattr(study, "verify_ensemble", lambda out, j, p, s, h, ids:
                        (None, {"offset": int(out.name.split('-')[1]), "count": len(ids)}))
    monkeypatch.setattr(study, "score_job", lambda r, p, h, j, s, ids: scored.append(j["id"]))
    for split in ("VAL", "TEST"):
        commands.clear(); scored.clear()
        study.sample_grid(tmp_path, p, "a" * 64, split, [{"index": 0, "uuid": "GPU-one"}])
        assert len(commands) == 1200 and len(scored) == 30
        names = {cmd[cmd.index("--out") + 1].split('/')[-2] for cmd in commands}
        assert len(names) == (30 if split == "VAL" else 20)
        if split == "TEST":
            assert not names.intersection(j["id"] for j in p["jobs"] if "reuse" in j)


def test_budget_comparison_reports_all_candidates_without_test_values():
    names = [f"{a}_s{s}" for a in study.ARMS for s in study.SEEDS]
    saved = {"curves": {n: {"means": [5, 2, 2, 3, 4]} for n in names}}
    new = {n: [4, 2, 1, 3, 5] for n in names}
    result = study.budget_comparison(saved, new)
    for row in result.values():
        assert row["VAL50_minus_VAL25"] == [-1, 0, -1, 0, 1]
        assert row["VAL25_ranks_earlier_ties"] == [5, 1, 2, 3, 4]
        assert row["VAL50_ranks_earlier_ties"] == [4, 2, 1, 3, 5]
        assert row["P25"] == 9560 and row["P50"] == 16252


@pytest.mark.parametrize("reason", ["timeout", "interrupt", "stage_budget"])
def test_child_group_cleanup_retains_failure_receipt(tmp_path, protocol, monkeypatch, reason):
    p = {**protocol, "source_root": str(tmp_path), "runner_sha256": "f" * 64}
    pid_path = tmp_path / "grandchild.pid"
    command = [study.sys.executable, "-c",
               "import subprocess,sys,time,pathlib; "
               "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
               "pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)", str(pid_path)]
    timer = None
    if reason == "timeout":
        monkeypatch.setattr(study, "CHILD_SECONDS", .5)
    elif reason == "stage_budget":
        monkeypatch.setattr(study, "DEADLINE", time.monotonic() + .5)
    else:
        timer = threading.Timer(.5, study.STOP.set)
        timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(study.ContractMismatch, match="timeout|budget|interrupted"):
            study.launch(tmp_path, p, command, reason, {"index": 0, "uuid": "GPU-test"})
    finally:
        if timer is not None:
            timer.join()
    assert time.monotonic() - started < 8
    receipt = study.read(next((tmp_path / "receipts").glob("*.json")))
    assert receipt["failure"] and receipt["exit_code"] < 0
    assert receipt["environment"]["CUDA_VISIBLE_DEVICES"] == "GPU-test"
    proc = Path('/proc') / pid_path.read_text() / 'stat'
    if proc.exists():
        assert proc.read_text().split()[2] in {"Z", "X"}


@pytest.mark.parametrize("busy,allowed", [
    ("GPU-one, 123, /opt/python\n", False),
    ("GPU-two, 123, /opt/python\n", True),
    ("GPU-one, 123, /usr/lib/xorg/Xorg\n", True),
    ("GPU-one, 123, /usr/libexec/gnome-remote-desktop-daemon\n", True),
    ("GPU-one, 123, /home/guest/gnome-remote-desktop-daemon\n", False),
    ("", True),
])
def test_admission_checks_allocated_uuid_compute_only(tmp_path, monkeypatch, busy, allowed):
    row = {"approved_by": "main", "protocol_sha256": "a" * 64, "stages": ["pilot"],
           "devices": [{"index": 0, "uuid": "GPU-one"}]}
    path = tmp_path / "admission.json"
    pin = save(path, row)
    def query(args, **kwargs):
        assert kwargs["timeout"] == 10
        return busy if "--query-compute-apps=" in args[1] else "0, GPU-one\n"
    monkeypatch.setattr(subprocess, "check_output", query)
    if allowed:
        assert study.gpu_admission(tmp_path, "a" * 64, "pilot", path, pin) == row["devices"]
    else:
        with pytest.raises(study.ContractMismatch, match="unexpected compute"):
            study.gpu_admission(tmp_path, "a" * 64, "pilot", path, pin)


def test_exhausted_stage_never_starts_child(tmp_path, protocol, monkeypatch):
    p = {**protocol, "source_root": str(tmp_path), "runner_sha256": "f" * 64}
    monkeypatch.setattr(study, "DEADLINE", 0)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("child started after budget"))
    with pytest.raises(study.ContractMismatch, match="budget"):
        study.launch(tmp_path, p, ["unused"], "expired")


@pytest.fixture
def figure_module():
    path = SOURCE.parents[1] / "paper/workshops/prigm/fig_checkpoints.py"
    module_spec = importlib.util.spec_from_file_location("checkpoint_figure", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


@pytest.fixture
def public_inputs(tmp_path, figure_module, monkeypatch):
    root = tmp_path / "private"
    records = {name: {"numeric": [1, 1.125, 1e-12, -2.5], "source_hashes": {"prigm/a.py": "a" * 64}}
               for name in figure_module.FILES}
    records["PROTOCOL.json"].update(original_root="/original-study-root", extension_root="/extension-study-root",
                                    environment={"executable": "/known/python", "hostname": "private-host", "version": "3.11.15"})
    records["SUMMARY.json"].update(input_sha256={str(root / "score.json"): "b" * 64},
                                  original_path="/original-study-root/runs/D_s7")
    for name, value in records.items(): save(root / name, value)
    monkeypatch.setattr(figure_module, "PROTOCOL", study.sha256_file(root / "PROTOCOL.json"))
    monkeypatch.setattr(figure_module, "SCOPE", study.sha256_file(root / "SCOPE.json"))
    return root, records


def test_figure_public_projection_preserves_numbers_hashes_and_original_files(public_inputs, figure_module, tmp_path):
    root, records = public_inputs
    before = {name: (root / name).read_bytes() for name in figure_module.FILES}
    output = tmp_path / "public"
    manifest = figure_module.public_export(root, output)
    assert set(p.name for p in output.iterdir()) == {*figure_module.FILES, "MANIFEST.json"}
    for name in figure_module.FILES:
        result = json.loads((output / name).read_bytes())
        assert result["numeric"] == records[name]["numeric"]
        assert result["source_hashes"] == records[name]["source_hashes"]
        assert (root / name).read_bytes() == before[name]
        assert manifest["files"][name] == study.sha256_file(output / name)
        assert manifest["original_sha256"][name] == study.sha256_file(root / name)
        assert b'/original-study-root' not in (output / name).read_bytes()
        assert b'/known/python' not in (output / name).read_bytes()
        assert b'private-host' not in (output / name).read_bytes()
    p = json.loads((output / "PROTOCOL.json").read_bytes())
    assert p["environment"] == {"executable": "environment-python", "hostname": "execution-host", "version": "3.11.15"}
    assert json.loads((output / "SUMMARY.json").read_bytes())["input_sha256"] == {"checkpoint-study/score.json": "b" * 64}


def test_figure_public_projection_rejects_unknown_absolute_path(public_inputs, figure_module, tmp_path):
    root, records = public_inputs
    records["SUMMARY.json"]["unexpected"] = "/unapproved/location"
    save(root / "SUMMARY.json", records["SUMMARY.json"])
    with pytest.raises(ValueError, match="unknown absolute"):
        figure_module.public_export(root, tmp_path / "public")


def test_figure_reduction_checks_complete_grid_month_weight_and_finite(figure_module):
    ids = ["2005-08-01", "2005-08-02", "2005-09-01"]
    months = [200508, 200508, 200509]
    rows = {key: [1., 3., 10.] for key in figure_module.KEYS}
    reduced = figure_module.reduce_curves(rows, ids, months, 3)
    assert all(v == [6.] * 5 for v in reduced.values())
    rows[next(iter(rows))][0] = float('nan')
    with pytest.raises(ValueError, match="invalid raw"):
        figure_module.reduce_curves(rows, ids, months, 3)
