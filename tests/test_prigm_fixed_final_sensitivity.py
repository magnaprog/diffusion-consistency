"""CPU-only synthetic tests for the bounded fixed-final coordinator."""
import copy
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pytest
import torch

from experiments import run_prigm_fixed_final_sensitivity as c
from experiments import run_prigm_repair as original_coordinator
from prigm.artifacts import NOISE_CONVENTION, STATE_CONVENTION, json_bytes, publish_ensemble, read_ensemble
from prigm.barrier import TestBarrierError as BarrierError, require_test_started
from prigm.checkpoint import frozen_source_snapshot, save_torch_payload
from prigm.model import ConditionalUNet
from prigm.recovery import ContractMismatch, PublicationError, publish_generation, sha256_bytes, sha256_file
from prigm.run_fit import training_settings


@pytest.fixture
def study(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "TEST_WINDOWS", 9)
    monkeypatch.setattr(c, "UNET_WIDTHS", (8, 16, 16))
    original, root, arrays = (tmp_path / name for name in ("original", "extension", "arrays"))
    for path in (original, root, arrays):
        path.mkdir()
    ids = [f"2020-{1 + i // 5:02d}-window-{i}" for i in range(9)]
    for split in ("TRAIN", "VAL", "TEST"):
        np.savez(arrays / f"{split.lower()}.npz", z=np.ones((9, 48, 6)),
                 ids=np.array(ids), split=np.array(split), mean=np.zeros(6), scale=np.ones(6),
                 b_ref_sq=np.array(1.), n_bins=np.array(100))
    old = original_coordinator.base_protocol(original, arrays)
    c.fresh_json(original / "PROTOCOL.json", old)
    old_digest = sha256_file(original / "PROTOCOL.json")
    monkeypatch.setattr(c, "ORIGINAL_PROTOCOL", old_digest)
    _, snapshot = frozen_source_snapshot(c.ROOT)
    settings = training_settings(torch.device("cpu"), list(c.MILESTONES))
    settings.update(source_snapshot_sha256=sha256_bytes(snapshot),
                    data={"kind": "real_npz", "ids_covered_by_arrays_hash": True})
    jobs = c.fixed_jobs(root, original)
    weights = ConditionalUNet(n_generated=6, widths=c.UNET_WIDTHS).state_dict()

    def checkpoint(job, protocol, generation=c.FINAL):
        contract = {
            "contract_version": 2, "arm": job["arm"], "seed": job["seed"], "c_r": job["c_r"],
            "n_generated": 6, "n_aux": 0, "widths": list(c.UNET_WIDTHS), "estimator": "S",
            "residual": None if job["arm"] == "D" else "obs", "batch_size": 128,
            "n_examples": 9, "max_updates": c.FINAL, "source_hashes": protocol["training_source_hashes"],
            "data_hashes": protocol["training_data_hashes"],
            "normalization_hash": protocol["arrays"]["TRAIN"]["normalization_sha256"],
            "settings": copy.deepcopy(settings),
        }
        meta = {"contract": contract, "updates": generation, "epochs": generation, "cursor": 0}
        payload = {**meta, "model": weights, "normalization": {
            "mean": torch.zeros(6, dtype=torch.float64), "scale": torch.ones(6, dtype=torch.float64),
            "b_ref_sq": 1., "n_bins": 100}}
        run = Path(job["run_root"])
        run.mkdir(parents=True, exist_ok=True)
        (run / "source_snapshot.zip").write_bytes(snapshot)
        publish_generation(run, generation, {"state.pt": save_torch_payload(payload),
                                            "meta.json": json_bytes(contract)}, meta)
        return c.verify_checkpoint(job, protocol, generation)

    def ensemble(job, protocol, digest, path=None, offset=0, count=None):
        stop = len(ids) if count is None else min(offset + count, len(ids))
        subset = ids[offset:stop]
        ckpt = c.verify_checkpoint(job, protocol)
        draws = np.stack([np.full((39, 6, 48), i / 100 + job["c_r"], np.float32)
                          for i in range(offset, stop)])
        meta = {"protocol_sha256": digest, "checkpoint_sha256": ckpt["files"]["state.pt"],
                "generation_updates": c.FINAL, "arm": job["arm"], "seed": job["seed"],
                "n_generated": 6, "arrays": protocol["arrays"]["TEST"], "sampler": c.SAMPLERS["TEST"],
                "state_convention": STATE_CONVENTION, "noise_convention": NOISE_CONVENTION,
                "inference_source_sha256": protocol["inference_source_hashes"]["prigm/run_eval.py"],
                "offset": offset, "count": len(subset)}
        publish_ensemble(Path(job["ensemble_root"]) if path is None else path, draws, subset, meta)

    selected, hashes, old_jobs = {}, [], []
    for job in jobs:
        if job["new_fit"]:
            continue
        ckpt = checkpoint(job, old)
        if job["reuse"]:
            name = f"{job['arm']}_s{job['seed']}"
            item = {"generation": c.FINAL, "checkpoint_sha256": ckpt["files"]["state.pt"]}
            selected[name] = {"raw": item, "corrected": item}
            hashes.append(item["checkpoint_sha256"])
            old_jobs.append([name, c.FINAL])
            ensemble(job, old, old_digest)
    c.fresh_json(original / "SELECTION.json", {"selected": selected, "test_jobs": old_jobs,
                                               "test_checkpoint_hashes": hashes})
    c.fresh_json(original / "TEST_STARTED.json", {
        "phase": "TEST_STARTED", "previous_test_exposure": True, "protocol_sha256": old_digest,
        "selection_sha256": sha256_file(original / "SELECTION.json")})
    protocol = c.base_protocol(root, original, arrays, time.time() + 3600)
    c.fresh_json(root / "PROTOCOL.json", protocol)
    digest = sha256_file(root / "PROTOCOL.json")
    return root, protocol, digest, checkpoint, ensemble, ids


def new_checkpoints(study):
    _, protocol, _, checkpoint, _, _ = study
    for job in protocol["jobs"]:
        if job["new_fit"]:
            checkpoint(job, protocol)


def change_json(path, mutate):
    value = json.loads(path.read_text())
    mutate(value)
    path.write_bytes(json_bytes(value))


def test_protocol_binds_exact_workload_and_reuse(study):
    root, p, digest, *_ = study
    assert c.verify_protocol(root, digest) == p
    assert len(p["jobs"]) == 12
    assert sum(j["new_fit"] for j in p["jobs"]) == 6
    assert sum(not j["reuse"] for j in p["jobs"]) == 9
    assert len(p["original_inputs"]["checkpoints"]) == 6
    assert len(p["original_inputs"]["ensembles"]) == 3
    assert all(j["arm"] == "O-S" and f"c{j['c_r']}" in j["id"]
               for j in p["jobs"] if j["new_fit"] and j["c_r"] == .005)
    assert all(j["generation"] == 29636 for j in p["jobs"])
    assert p["selection"] == c.SELECTION_RULE
    assert not (root / "TEST_STARTED.json").exists()


@pytest.mark.parametrize("key,value", [
    ("c_r", .5), ("seed", 42), ("arm", "I-S"), ("n_generated", 5), ("n_aux", 1),
    ("widths", [1, 2, 3]), ("n_examples", 10), ("batch_size", 64), ("max_updates", 2868),
    ("normalization_hash", "wrong"), ("source_hashes", {}), ("data_hashes", {}),
    ("residual", "inst"), ("estimator", "M"),
])
def test_wrong_checkpoint_identity_rejected(study, key, value):
    _, p, _, *_ = study
    job = next(j for j in p["jobs"] if j["c_r"] == .0005 and j["seed"] == 7)
    path = Path(job["run_root"]) / f"gen_{c.FINAL:08d}/COMMIT.json"
    change_json(path, lambda row: row["contract"].__setitem__(key, value))
    with pytest.raises(ContractMismatch):
        c.verify_checkpoint(job, p)


def test_optimizer_and_profile_recovery_contract_rejected(study):
    _, p, _, *_ = study
    job = p["jobs"][0]
    path = Path(job["run_root"]) / f"gen_{c.FINAL:08d}/COMMIT.json"
    change_json(path, lambda row: row["contract"]["settings"]["optimizer"].__setitem__("lr", 100.))
    with pytest.raises(ContractMismatch, match="optimizer"):
        c.verify_checkpoint(job, p)


def test_saved_source_snapshot_is_read(study):
    _, p, _, *_ = study
    job = p["jobs"][0]
    (Path(job["run_root"]) / "source_snapshot.zip").write_bytes(b"bad")
    with pytest.raises(PublicationError, match="snapshot hash"):
        c.verify_checkpoint(job, p)


def test_nonfinal_payload_fails_before_test_authorization(study):
    root, p, digest, *_ = study
    new_checkpoints(study)
    job = p["jobs"][0]
    path = Path(job["run_root"]) / f"gen_{c.FINAL:08d}/COMMIT.json"
    change_json(path, lambda row: row.__setitem__("updates", 2868))
    with pytest.raises(ContractMismatch, match="counters"):
        c.freeze_endpoints(root, p, digest)
    assert not (root / "TEST_STARTED.json").exists()


@pytest.mark.parametrize("field,value", [("run_root", "/tmp/wrong"), ("c_r", .0005),
                                        ("seed", 123), ("generation", 9560), ("reuse", True)])
def test_rehashed_protocol_still_rejects_changed_design(study, field, value):
    root, _, _, *_ = study
    path = root / "PROTOCOL.json"
    change_json(path, lambda row: row["jobs"][3].__setitem__(field, value))
    with pytest.raises(ContractMismatch, match="fixed design"):
        c.verify_protocol(root, sha256_file(path))


def test_protocol_hash_and_array_tampering(study):
    root, p, digest, *_ = study
    path = root / "PROTOCOL.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(BarrierError):
        c.verify_protocol(root, digest)
    path.write_bytes(json_bytes(p))
    arrays = Path(p["array_root"]) / "train.npz"
    arrays.write_bytes(arrays.read_bytes() + b"tamper")
    with pytest.raises(ContractMismatch, match="array hash"):
        c.verify_protocol(root, digest)


def test_source_producer_and_symlink_identity(study, monkeypatch, tmp_path):
    root, p, digest, *_ = study
    with monkeypatch.context() as patch:
        patch.setattr(c, "source_identity", lambda: {})
        with pytest.raises(ContractMismatch, match="execution source"):
            c.verify_protocol(root, digest)
    with monkeypatch.context() as patch:
        patch.setattr(c, "producers", lambda: {})
        with pytest.raises(ContractMismatch, match="execution source"):
            c.verify_protocol(root, digest)
    (root / "runs").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    with pytest.raises(ContractMismatch, match="noncanonical"):
        c.verify_protocol(root, digest)


@pytest.mark.parametrize("field,value", [("noise_convention", "legacy"), ("sampler", {}),
                                        ("generation_updates", 9560), ("inference_source_sha256", "bad"),
                                        ("protocol_sha256", "bad"), ("arrays", {})])
def test_invalid_reuse_fails_without_enlarging_workload(study, field, value):
    root, p, digest, _, _, ids = study
    job = next(j for j in p["jobs"] if j["reuse"])
    path = Path(job["ensemble_root"]) / "gen_00000000/COMMIT.json"
    change_json(path, lambda row: row.__setitem__(field, value))
    with pytest.raises(ContractMismatch):
        c.original_inputs(p, ids)
    assert not (root / "TEST_STARTED.json").exists()
    assert sum(not j["reuse"] for j in p["jobs"]) == 9


def test_missing_old_checkpoint_does_not_train_replacement(study, monkeypatch):
    root, p, digest, *_ = study
    path = Path(p["jobs"][1]["run_root"]) / f"gen_{c.FINAL:08d}/state.pt"
    path.unlink()
    monkeypatch.setattr(c, "launch", lambda *a: pytest.fail("must not launch"))
    with pytest.raises(PublicationError):
        c.freeze_endpoints(root, p, digest)
    with pytest.raises(PublicationError):
        c.train(root, p, [0], time.time() + 60)
    assert not (root / "SELECTION.json").exists()


def test_all_endpoints_required_and_freeze_is_immutable(study):
    root, p, digest, *_ = study
    with pytest.raises(PublicationError):
        c.freeze_endpoints(root, p, digest)
    assert not (root / "SELECTION.json").exists()
    new_checkpoints(study)
    c.freeze_endpoints(root, p, digest)
    before = (root / "SELECTION.json").read_bytes()
    c.freeze_endpoints(root, p, digest)
    assert (root / "SELECTION.json").read_bytes() == before
    selection = require_test_started(root, "TEST", digest)
    assert len(selection["endpoints"]) == 12
    assert len(selection["test_jobs"]) == 9
    assert "validation_score_sha256" not in selection


def test_cannot_freeze_a_truncated_grid(study):
    root, p, digest, *_ = study
    p["jobs"].pop()
    with pytest.raises(ContractMismatch, match="complete fixed grid"):
        c.freeze_endpoints(root, p, digest)
    assert not (root / "SELECTION.json").exists()


def test_deadline_during_endpoint_checks_does_not_authorize_test(study, monkeypatch):
    root, p, digest, *_ = study
    clock = [100.]
    monkeypatch.setattr(c.time, "time", lambda: clock[0])

    def slow_check(*args):
        clock[0] = 102.
        return {}
    monkeypatch.setattr(c, "endpoint_selection", slow_check)
    with pytest.raises(c.DeadlineReached):
        c.freeze_endpoints(root, p, digest, deadline=101.)
    assert not (root / "SELECTION.json").exists()
    assert not (root / "TEST_STARTED.json").exists()


def test_selection_tampering_fails_even_if_phase_rehashed(study):
    root, p, digest, _, _, ids = study
    new_checkpoints(study)
    c.freeze_endpoints(root, p, digest)
    change_json(root / "SELECTION.json", lambda row: row["test_jobs"].pop())
    with pytest.raises(BarrierError):
        c.authorize_test(root, p, digest, ids)
    change_json(root / "TEST_STARTED.json", lambda row: row.__setitem__(
        "selection_sha256", sha256_file(root / "SELECTION.json")))
    with pytest.raises(ContractMismatch, match="complete predetermined"):
        c.authorize_test(root, p, digest, ids)


def test_partial_grid_never_computes_or_publishes_metrics(study, monkeypatch):
    root, p, digest, *_ = study
    new_checkpoints(study)
    c.freeze_endpoints(root, p, digest)
    monkeypatch.setattr(c, "window_diagnostics", lambda *a: pytest.fail("partial grid must not score"))
    with pytest.raises(PublicationError):
        c.aggregate(root, p, digest, time.time() + 60)
    assert not (root / "SUMMARY.json").exists()


def test_profile_train_four_normal_jobs_then_resume_all_six(study, monkeypatch):
    root, p, _, checkpoint, *_ = study
    commands = []

    def fake_launch(command, study_root, name, gpu, deadline):
        commands.append(command)
        job = next(j for j in p["jobs"] if j["run_root"] == command[command.index("--run-root") + 1])
        target = int(command[command.index("--updates") + 1])
        checkpoint(job, p, target)
    monkeypatch.setattr(c, "launch", fake_launch)
    c.train(root, p, [0], time.time() + 60, profile=True)
    assert len(commands) == 4
    assert all("--profile" not in cmd and "--resume" not in cmd for cmd in commands)
    assert all(cmd[cmd.index("--updates") + 1] == "2868" for cmd in commands)
    assert all(cmd[cmd.index("--budget-updates") + 1] == "29636" for cmd in commands)
    c.train(root, p, [0], time.time() + 60)
    assert len(commands) == 10
    assert sum("--resume" in cmd for cmd in commands[4:]) == 4
    assert all(cmd[cmd.index("--arm") + 1] == "O-S" for cmd in commands)
    assert all(cmd[cmd.index("--milestones") + 1] == "2868,9560,16252,22944,29636" for cmd in commands)
    c.train(root, p, [0], time.time() + 60)
    assert len(commands) == 10


def test_deadline_prevents_launch_and_further_queue_jobs(monkeypatch):
    called = []
    with pytest.raises(c.DeadlineReached):
        c.work_queue([1, 2], [0], time.time() - 1, lambda *args: called.append(args))
    assert not called
    clock = [100.]
    monkeypatch.setattr(c.time, "time", lambda: clock[0])

    def action(job, gpu):
        called.append(job)
        clock[0] = 102.
    with pytest.raises(c.DeadlineReached):
        c.work_queue([1, 2, 3], [0], 101., action)
    assert called == [1]


def test_launcher_timeout_receipts_and_owned_environment(tmp_path, monkeypatch):
    seen = []

    def run(command, **kwargs):
        seen.append(kwargs)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(c.subprocess, "run", run)
    with pytest.raises(c.DeadlineReached):
        c.launch(["synthetic-owned-child"], tmp_path, "test", 3, time.time() + 20)
    assert 0 < seen[0]["timeout"] <= 20
    env = seen[0]["env"]
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["PRIGM_REQUIRE_CUDA"] == env["PISCES_REQUIRE_CUDA"] == "1"
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == "2"
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    receipt = json.loads(next((tmp_path / "logs").glob("*.finished.json")).read_text())
    assert receipt["status"] == "deadline" and receipt["wall_seconds"] >= 0
    with pytest.raises(c.DeadlineReached):
        c.launch(["never"], tmp_path, "test", 0, time.time() - 1)
    assert len(seen) == 1


def test_successful_job_has_separate_immutable_whole_wall_receipts(tmp_path, monkeypatch):
    clock = [100.]
    monkeypatch.setattr(c.time, "time", lambda: clock[0])
    monkeypatch.setattr(c.time, "monotonic", lambda: clock[0])

    def run(command, **kwargs):
        clock[0] += 7.
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(c.subprocess, "run", run)
    c.launch(["synthetic"], tmp_path, "same-job", 0, 200.)
    c.launch(["synthetic"], tmp_path, "same-job", 0, 200.)
    receipts = list((tmp_path / "logs").glob("*.finished.json"))
    assert len(receipts) == 2
    for path in receipts:
        row = json.loads(path.read_text())
        assert row["status"] == "complete" and row["returncode"] == 0
        assert row["wall_seconds"] == 7.


def test_cli_expired_deadline_does_not_start_a_job(study, monkeypatch):
    root, _, digest, *_ = study
    monkeypatch.setattr(c, "launch", lambda *a: pytest.fail("must not launch"))
    with pytest.raises(c.DeadlineReached):
        c.main(["--stage", "train", "--study-root", str(root), "--protocol-sha256", digest,
                "--deadline", "2020-01-01T00:00:00Z"])
    assert c.parse_deadline("2026-09-08T16:40:50Z") == c.parse_deadline("1788885650")
    with pytest.raises(ValueError):
        c.parse_deadline("2026-09-08T16:40:50")


def test_complete_chunk_merge_scoring_and_historical_bridge(study, monkeypatch):
    root, p, digest, _, ensemble, ids = study
    new_checkpoints(study)
    c.freeze_endpoints(root, p, digest)
    commands = []

    def fake_launch(command, study_root, name, gpu, deadline):
        commands.append(command)
        job = next(j for j in p["jobs"] if j["run_root"] == command[command.index("--run-root") + 1])
        assert not job["reuse"]
        offset = int(command[command.index("--offset") + 1])
        ensemble(job, p, digest, Path(command[command.index("--out") + 1]), offset, 8)
    monkeypatch.setattr(c, "launch", fake_launch)
    c.sample(root, p, digest, [0, 1, 2, 3], time.time() + 120)
    assert len(commands) == 18
    assert all(cmd[cmd.index("--split") + 1] == "TEST" for cmd in commands)
    c.sample(root, p, digest, [0], time.time() + 120)
    assert len(commands) == 18
    for job in p["jobs"]:
        if job["reuse"]:
            continue
        draws, sampled_ids, _ = read_ensemble(Path(job["ensemble_root"]))
        assert sampled_ids == ids
        for i in range(len(ids)):
            np.testing.assert_array_equal(draws[i], np.full((39, 6, 48), i / 100 + job["c_r"], np.float32))
    output = c.aggregate(root, p, digest, time.time() + 120)
    assert output["complete_grid"] and len(output["rows"]) == 12
    assert {r["protocol_sha256"] for r in output["rows"]} == {digest, c.ORIGINAL_PROTOCOL}
    for row in output["rows"]:
        assert len(row["results"]["raw"]["per_window"]) == len(ids)
        assert row["protocol_sha256"] == (c.ORIGINAL_PROTOCOL if row["reuse"] else digest)
        assert "es5" in row["results"]["corrected"]["summary"]
    before = (root / "SUMMARY.json").read_bytes()
    c.aggregate(root, p, digest, time.time() + 120)
    assert (root / "SUMMARY.json").read_bytes() == before


def test_incomplete_chunk_is_retained_not_relabelled(study, monkeypatch):
    root, p, digest, *_ = study
    new_checkpoints(study)
    c.freeze_endpoints(root, p, digest)
    job = next(j for j in p["jobs"] if not j["reuse"])
    path = Path(job["ensemble_root"]).parent / "chunk-0000"
    path.mkdir(parents=True)
    marker = path / "partial.bin"
    marker.write_bytes(b"owned partial work")
    monkeypatch.setattr(c, "launch", lambda *a: pytest.fail("must not overwrite partial chunk"))
    with pytest.raises(PublicationError):
        c.sample(root, p, digest, [0], time.time() + 60)
    assert marker.read_bytes() == b"owned partial work"
    assert not list(path.parent.glob("*.incomplete-*"))
