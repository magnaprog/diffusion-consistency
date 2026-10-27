"""CPU-only restricted-checkpoint, provenance, and replay regression tests."""

from __future__ import annotations

import io
import json
import os
import random
import signal
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prigm.checkpoint import (
    array_hash,
    frozen_source_snapshot,
    load_torch_payload,
    normalization_hash,
    save_torch_payload,
    source_hashes,
    verify_source_snapshot,
)
from prigm.loop import RecoverableTrainer, RunContract, configure_determinism
from prigm.recovery import (
    ContractMismatch,
    PublicationError,
    RunLock,
    load_committed,
    previous_generation,
    sha256_bytes,
    sha256_file,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    configure_determinism()
    yield
    torch.set_num_threads(old)


def _fixture(root, arm="D", *, n=10):
    random.seed(918)
    all_u = np.random.default_rng(123).normal(size=(n, 6, 48)).astype(np.float32)
    obs = all_u.copy()
    obs[:, :, 18:30] = 0
    u = all_u[:, :5].copy() if arm == "G5" else all_u
    mean, scale = np.zeros(6), np.ones(6)
    c = RunContract(arm=arm, seed=7, n_generated=u.shape[1], n_aux=0, widths=(4, 8, 8),
                    c_r=5e-4 if arm == "O-S" else 0.0,
                    residual="obs" if arm == "O-S" else None, estimator="S",
                    n_examples=n, batch_size=4, max_updates=12, source_hashes={"fixture.py": "fixture"},
                    data_hashes={"u0_sha256": array_hash(u), "observed_sha256": array_hash(obs)},
                    normalization_hash=normalization_hash(mean, scale, 2.5, 123),
                    settings={"milestones": [2, 5]})
    return RecoverableTrainer(root, c, u, obs, None, mean, scale, 2.5,
                              device=torch.device("cpu"), normalization_n_bins=123)


def _blob(root, generation):
    return load_torch_payload((Path(root) / f"gen_{generation:08d}" / "state.pt").read_bytes())


def _equal(a, b):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            _equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            _equal(x, y)
    else:
        assert a == b


def _subprocess_train(root, arm, target, fresh):
    # Running this file as a script starts a new interpreter, not a forked trainer.
    subprocess.run([sys.executable, str(Path(__file__).resolve()), str(root), arm,
                    str(target), "fresh" if fresh else "resume"], check=True,
                   cwd=PROJECT_ROOT, timeout=90,
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})


@pytest.mark.parametrize("arm", ["D", "G5", "O-S"])
def test_full_state_matches_fresh_subprocess_resume(tmp_path, arm):
    a, b = tmp_path / "full", tmp_path / "resumed"
    _subprocess_train(a, arm, 7, True)
    _subprocess_train(b, arm, 2, True)
    # Resume crosses the smaller final batch (2 examples), two epoch boundaries,
    # a required milestone, and finishes at a non-periodic target.
    _subprocess_train(b, arm, 7, False)
    full, resumed = _blob(a, 7), _blob(b, 7)
    _equal(full, resumed)  # model, optimizer moments/steps, every RNG, and all metadata
    assert (resumed["updates"], resumed["epochs"], resumed["cursor"]) == (7, 2, 4)
    assert (b / "gen_00000003").is_dir()
    assert load_committed(b, 5)["kind"] == "milestone"
    assert load_committed(b, 7)["kind"] == "target"


def test_restricted_optimizer_rng_roundtrip_and_no_unsafe_fallback(tmp_path, monkeypatch):
    trainer = _fixture(tmp_path / "run")
    trainer.train(1, start_new=True)
    blob = _blob(trainer.run_root, 1)
    assert blob["optimizer"]["state"]
    _equal(load_torch_payload(save_torch_payload(blob)), blob)
    raw = io.BytesIO()
    torch.save({"numpy_object": np.arange(3)}, raw, pickle_protocol=2)
    calls = []
    real_load = torch.load

    def checked_load(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    with pytest.raises(PublicationError, match="restricted checkpoint"):
        load_torch_payload(raw.getvalue())
    assert calls == [True]
    with pytest.raises(PublicationError, match="unsupported checkpoint type"):
        save_torch_payload({"numpy_object": np.arange(3)})
    with pytest.raises(PublicationError):
        load_torch_payload(b"corrupt bytes")


def test_hashes_are_canonical_and_source_snapshot_is_verifiable(tmp_path):
    mean = np.arange(6, dtype=np.float32)
    scale = np.ones(6, dtype=np.float32)
    digest = normalization_hash(mean, scale, np.float32(2.5), np.int64(123))
    assert digest == normalization_hash(mean.astype(">f8"), scale.astype(np.float64), 2.5, 123)
    assert digest != normalization_hash(mean, scale, 2.5, 124)
    assert digest != normalization_hash(mean, scale, 2.6, 123)
    assert digest != normalization_hash(mean + 1, scale, 2.5, 123)
    hashes, snapshot = frozen_source_snapshot(PROJECT_ROOT)
    assert hashes == source_hashes(sorted((PROJECT_ROOT / "prigm").rglob("*.py")), relative_to=PROJECT_ROOT)
    assert "prigm/loop.py" in hashes and all(not Path(k).is_absolute() for k in hashes)
    assert all(len(v) == 64 for v in hashes.values())
    verify_source_snapshot(snapshot, hashes, sha256_bytes(snapshot))
    with pytest.raises(PublicationError, match="snapshot hash mismatch"):
        verify_source_snapshot(snapshot + b"changed", hashes, sha256_bytes(snapshot))
    trainer = _fixture(tmp_path / "run")
    trainer.contract = replace(trainer.contract, source_hashes=hashes, settings={
        **trainer.contract.settings, "source_snapshot_sha256": sha256_bytes(snapshot),
    })
    trainer.source_snapshot = snapshot
    trainer.train(0, start_new=True)
    assert (trainer.run_root / "source_snapshot.zip").read_bytes() == snapshot
    checkpoint = _blob(trainer.run_root, 0)
    nrm = checkpoint["normalization"]
    assert nrm["mean"].dtype == torch.float64
    assert normalization_hash(**nrm) == checkpoint["contract"]["normalization_hash"]
    (trainer.run_root / "source_snapshot.zip").write_bytes(snapshot + b"changed")
    with pytest.raises(PublicationError, match="snapshot hash mismatch"):
        trainer.load(0)


@pytest.mark.parametrize("change", ["source", "data_contract", "data_values", "normalization", "settings"])
def test_changed_scientific_contract_cannot_fallback(tmp_path, change):
    root = tmp_path / "run"
    _fixture(root).train(2, start_new=True)
    trainer = _fixture(root)
    if change == "source":
        trainer.contract = replace(trainer.contract, source_hashes={"fixture.py": "changed"})
    elif change == "data_contract":
        trainer.contract = replace(trainer.contract, data_hashes={**trainer.contract.data_hashes, "arrays_sha256": "changed"})
    elif change == "data_values":
        trainer.u0_np[0, 0, 0] += 1
        trainer.data_hashes["u0_sha256"] = array_hash(trainer.u0_np)
        trainer.contract = replace(trainer.contract, data_hashes=dict(trainer.data_hashes))
    elif change == "normalization":
        trainer.normalization_digest = normalization_hash(np.ones(6), np.ones(6), 2.5, 123)
    else:
        trainer.contract = replace(trainer.contract, settings={"milestones": [1, 2]})
    with pytest.raises(ContractMismatch):
        trainer.train(3, start_new=False)
    assert not list(root.glob("recovery_*.json"))
    assert load_committed(root)["generation"] == 2


def test_corrupt_latest_falls_back_records_replay_and_preserves_bad(tmp_path):
    root, reference = tmp_path / "run", tmp_path / "reference"
    _fixture(reference).train(5, start_new=True)
    trainer = _fixture(root)
    trainer.snapshot_every = 1
    trainer.train(3, start_new=True)
    bad = root / "gen_00000003" / "state.pt"
    bad.write_bytes(b"broken checkpoint")
    (root / "LATEST").write_text("0\n")  # stale hints cannot hide either good or bad generations
    with pytest.raises(PublicationError, match="hash mismatch"):
        _fixture(root).load(3, inference=True)
    assert load_committed(root)["generation"] == 2
    assert previous_generation(root, 4) == 2
    recovered = _fixture(root)
    recovered.train(5, start_new=False)
    _equal(_blob(root, 5), _blob(reference, 5))
    records = list(root.glob("recovery_*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["latest_hint"] == "0" and record["replay_from_update"] == 2
    assert record["total_target"] == 5
    preserved = root / record["preserved"][0]["path"]
    assert (preserved / "state.pt").read_bytes() == b"broken checkpoint"
    assert (root / "gen_00000003" / "state.pt").read_bytes() != b"broken checkpoint"


@pytest.mark.parametrize("hint", [None, "0\n", "999\n", "invalid\n"])
def test_latest_is_advisory_and_target_already_saved_is_not_republished(tmp_path, hint):
    root = tmp_path / "run"
    _fixture(root).train(2, start_new=True)
    before = sha256_file(root / "gen_00000002" / "COMMIT.json")
    if hint is None:
        (root / "LATEST").unlink()
    else:
        (root / "LATEST").write_text(hint)
    assert load_committed(root)["generation"] == 2
    _fixture(root).train(2, start_new=False)
    assert (root / "LATEST").read_text() == "2\n"
    assert sha256_file(root / "gen_00000002" / "COMMIT.json") == before


def _rewrite_payload(root, generation, edit):
    gen = root / f"gen_{generation:08d}"
    state = _blob(root, generation)
    edit(state)
    (gen / "state.pt").write_bytes(save_torch_payload(state))
    meta = json.loads((gen / "COMMIT.json").read_text())
    meta["files"]["state.pt"] = sha256_file(gen / "state.pt")
    (gen / "COMMIT.json").write_text(json.dumps(meta))


@pytest.mark.parametrize("change", ["contract", "cursor", "epochs", "step", "nonfinite", "optimizer", "normalization", "rng"])
def test_hash_valid_but_inconsistent_checkpoint_is_rejected(tmp_path, change):
    root = tmp_path / "run"
    _fixture(root).train(2, start_new=True)

    def corrupt(state):
        if change == "contract":
            state["contract"]["seed"] += 1
        elif change == "cursor":
            state["cursor"] = 1
        elif change == "epochs":
            state["epochs"] += 1
        elif change == "step":
            state["updates"] += 1
        elif change == "nonfinite":
            next(iter(state["model"].values())).flatten()[0] = float("nan")
        elif change == "optimizer":
            next(iter(state["optimizer"]["state"].values()))["step"] += 1
        elif change == "rng":
            state["rng"]["noise_gen"] = "00" * torch.get_rng_state().numel()
        else:
            state["normalization"]["mean"][0] += 1

    _rewrite_payload(root, 2, corrupt)
    with pytest.raises(PublicationError):
        _fixture(root).load(2, inference=True)
    meta = _fixture(root).load()
    assert meta["generation"] == 0
    assert meta["recovery"]["skipped"][0]["generation"] == 2


@pytest.mark.parametrize("damage", ["missing_contract", "wrong_generation", "malformed_json"])
def test_corrupt_commit_metadata_falls_back_but_explicit_load_fails(tmp_path, damage):
    root = tmp_path / "run"
    _fixture(root).train(2, start_new=True)
    path = root / "gen_00000002" / "COMMIT.json"
    meta = json.loads(path.read_text())
    if damage == "missing_contract":
        del meta["contract"]
    elif damage == "wrong_generation":
        meta["generation"] = 1
    path.write_text("{" if damage == "malformed_json" else json.dumps(meta))
    with pytest.raises(PublicationError):
        _fixture(root).load(2, inference=True)
    assert _fixture(root).load()["generation"] == 0


def test_inference_uses_training_cursor_theory_not_eval_row_count(tmp_path):
    root = tmp_path / "run"
    _fixture(root).train(2, start_new=True)
    trainer = _fixture(root, n=2)
    saved = _blob(root, 2)["contract"]
    trainer.contract = RunContract(**{**saved, "widths": tuple(saved["widths"]), "n_examples": 2})
    trainer.load(2, inference=True)
    assert (trainer.updates, trainer.epochs, trainer.cursor) == (2, 0, 0)
    with pytest.raises(ContractMismatch):
        trainer.load(2, inference=False)


def test_writer_lock_covers_updates_not_just_publication(tmp_path, monkeypatch):
    root = tmp_path / "run"
    trainer = _fixture(root)
    update = trainer._one_update
    code = """
import sys
from pathlib import Path
from prigm.recovery import RunLock, PublicationError
try:
    with RunLock(Path(sys.argv[1])):
        raise SystemExit('concurrent writer entered')
except PublicationError:
    pass
"""
    checks = []

    def checked_update():
        subprocess.run([sys.executable, "-c", code, str(root)], cwd=PROJECT_ROOT,
                       check=True, timeout=30)
        checks.append(trainer.updates)
        return update()

    monkeypatch.setattr(trainer, "_one_update", checked_update)
    trainer.train(2, start_new=True)
    assert checks == [0, 1]
    with RunLock(root):
        pass  # lock released after training
    with pytest.raises(PublicationError, match="requires the training run lock"):
        trainer.publish(2, "external")


@pytest.mark.parametrize("occupied", ["unrelated.txt", "WRITE.lock", ".tmp_gen_00000000", "gen_00000000"])
def test_fresh_root_rejects_any_existing_contents(tmp_path, occupied):
    root = tmp_path / "run"
    root.mkdir()
    path = root / occupied
    path.mkdir() if "gen_" in occupied else path.write_text("keep")
    with pytest.raises(PublicationError, match="not fresh"):
        _fixture(root).train(0, start_new=True)
    assert path.exists()
    assert not (root / "LATEST").exists()


def test_total_target_bounds_and_periodic_final_no_duplicate(tmp_path):
    root = tmp_path / "run"
    trainer = _fixture(root)
    trainer.snapshot_every = 1
    trainer.train(3, start_new=True)
    assert sorted(p.name for p in root.glob("gen_*")) == [f"gen_{g:08d}" for g in range(4)]
    with pytest.raises(ValueError, match="precedes"):
        _fixture(root).train(2, start_new=False)
    with pytest.raises(ValueError, match="frozen training budget"):
        _fixture(root).train(13, start_new=False)
    resumed = _fixture(root)
    resumed.train(6, start_new=False)
    assert resumed.updates == 6 and resumed.epochs == 2


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_graceful_signal_saves_once_and_restores_handler(tmp_path, monkeypatch, sig):
    trainer = _fixture(tmp_path / "run")
    original_handler = signal.getsignal(sig)
    update = trainer._one_update

    def interrupted_update():
        out = update()
        os.kill(os.getpid(), sig)
        return out

    monkeypatch.setattr(trainer, "_one_update", interrupted_update)
    trainer.train(7, start_new=True)
    assert trainer.updates == 1
    assert load_committed(trainer.run_root)["kind"] == "graceful"
    assert signal.getsignal(sig) == original_handler
    with RunLock(trainer.run_root):
        pass


def test_precommit_write_failure_stops_and_incomplete_generation_is_preserved(tmp_path, monkeypatch):
    import prigm.recovery as recovery

    root = tmp_path / "run"
    trainer = _fixture(root)
    trainer.snapshot_every = 1
    write = recovery.atomic_write_bytes

    def fail(path, data):
        if path.parent.name == ".tmp_gen_00000001" and path.name == "COMMIT.json":
            raise OSError("fixture disk publication failure")
        return write(path, data)

    with monkeypatch.context() as patch:
        patch.setattr(recovery, "atomic_write_bytes", fail)
        with pytest.raises(OSError, match="fixture disk publication failure"):
            trainer.train(5, start_new=True)
    assert trainer.updates == 1
    assert load_committed(root)["generation"] == 0
    assert (root / ".tmp_gen_00000001" / "state.pt").is_file()
    assert not (root / "gen_00000001").exists()
    _fixture(root).train(2, start_new=False)
    record = json.loads(next(root.glob("recovery_*.json")).read_text())
    preserved = root / record["preserved"][0]["path"]
    assert (preserved / "state.pt").is_file() and not (preserved / "COMMIT.json").exists()


def test_latest_write_failure_stops_but_committed_generation_remains_recoverable(tmp_path, monkeypatch):
    import prigm.recovery as recovery

    root = tmp_path / "run"
    trainer = _fixture(root)
    trainer.snapshot_every = 1
    write = recovery.atomic_write_bytes

    def fail(path, data):
        if path.name == "LATEST" and data == b"1\n":
            raise OSError("fixture advisory hint failure")
        return write(path, data)

    with monkeypatch.context() as patch:
        patch.setattr(recovery, "atomic_write_bytes", fail)
        with pytest.raises(OSError, match="advisory hint failure"):
            trainer.train(5, start_new=True)
    assert trainer.updates == 1 and (root / "LATEST").read_text() == "0\n"
    assert load_committed(root)["generation"] == 1
    _fixture(root).train(2, start_new=False)
    record = json.loads(next(root.glob("recovery_*.json")).read_text())
    assert record["selected_generation"] == 1 and record["skipped"] == []


def test_fallback_does_not_swallow_unrelated_errors(tmp_path, monkeypatch):
    import prigm.recovery as recovery

    root = tmp_path / "run"
    _fixture(root).train(2, start_new=True)

    def denied(path):
        raise PermissionError("fixture denied")

    monkeypatch.setattr(recovery, "sha256_file", denied)
    with pytest.raises(PermissionError, match="fixture denied"):
        load_committed(root)
    with pytest.raises(PermissionError, match="fixture denied"):
        previous_generation(root, 3)


def test_aux_and_nonfinite_state_rejected_before_publication(tmp_path):
    trainer = _fixture(tmp_path / "run")
    with pytest.raises(ContractMismatch, match="auxiliary"):
        RecoverableTrainer(tmp_path / "aux", replace(trainer.contract, n_aux=1),
                           trainer.u0_np, trainer.obs_np, None, np.zeros(6), np.ones(6), 2.5,
                           device=torch.device("cpu"), normalization_n_bins=123)
    with torch.no_grad():
        next(trainer.model.parameters()).flatten()[0] = float("inf")
    with pytest.raises(PublicationError, match="nonfinite"):
        trainer.train(1, start_new=True)
    assert not (trainer.run_root / "gen_00000000").exists()


def test_cli_frozen_budget_and_real_array_provenance(tmp_path, monkeypatch):
    import prigm.run_fit as run_fit

    path = tmp_path / "train.npz"
    z = np.random.default_rng(0).normal(size=(5, 48, 6))
    np.savez(path, z=z, ids=np.array([f"train:{i}" for i in range(5)]), split=np.array("TRAIN"),
             mean=np.zeros(6), scale=np.ones(6), b_ref_sq=np.array(2.5), n_bins=np.array(123))
    monkeypatch.setenv("PRIGM_REQUIRE_CUDA", "0")
    monkeypatch.setattr(run_fit, "UNET_WIDTHS", (4, 8, 8))
    monkeypatch.setattr(run_fit, "require_cuda_if_requested", lambda: torch.device("cpu"))
    args = ["--run-root", str(tmp_path / "run"), "--arm", "D", "--seed", "7",
            "--arrays", str(path), "--batch-size", "4", "--budget-updates", "7", "--milestones", "2,5"]
    assert run_fit.main([*args, "--updates", "2"]) == 0
    assert run_fit.main([*args, "--updates", "5", "--resume"]) == 0
    state = _blob(tmp_path / "run", 5)
    contract = state["contract"]
    assert contract["max_updates"] == 7 and state["updates"] == 5
    assert contract["data_hashes"] == {"arrays_sha256": sha256_file(path)}
    assert set(state["data_hashes"]) == {"u0_sha256", "observed_sha256"}
    assert contract["settings"]["data"]["ids_covered_by_arrays_hash"] is True
    assert contract["settings"]["optimizer"]["betas"] == [0.9, 0.999]
    assert contract["settings"]["precision"]["amp"] is False
    assert contract["settings"]["environment"]["torch"] == str(torch.__version__)
    # The NPZ digest covers row identities even when every physical value is unchanged.
    np.savez(path, z=z, ids=np.array([f"changed:{i}" for i in range(5)]), split=np.array("TRAIN"),
             mean=np.zeros(6), scale=np.ones(6), b_ref_sq=np.array(2.5), n_bins=np.array(123))
    with pytest.raises(ContractMismatch, match="scientific contract changed"):
        run_fit.main([*args, "--updates", "6", "--resume"])
    parsed = run_fit.parse_args(["--run-root", "unused", "--arm", "D", "--seed", "7", "--updates", "2868"])
    assert parsed.budget_updates == 29636
    assert parsed.milestones == [2868, 9560, 16252, 22944, 29636]
    with pytest.raises(SystemExit):
        run_fit.parse_args([*args, "--updates", "8"])
    with pytest.raises(SystemExit):
        run_fit.parse_args([*args, "--updates", "5", "--milestones", "2,1"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    configure_determinism()
    _fixture(Path(sys.argv[1]), sys.argv[2]).train(int(sys.argv[3]), start_new=sys.argv[4] == "fresh")
