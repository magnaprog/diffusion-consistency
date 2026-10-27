"""Selection and recovery gates use committed inputs, not cached summary numbers."""

import json

import numpy as np
import pytest

from experiments import run_prigm_repair as coordinator
from prigm.artifacts import NOISE_CONVENTION, STATE_CONVENTION, json_bytes, publish_ensemble
from prigm.barrier import TestBarrierError as BarrierError, require_test_started
from prigm.recovery import ContractMismatch, PublicationError, publish_generation, sha256_file


@pytest.fixture
def study(tmp_path, monkeypatch):
    monkeypatch.setattr(coordinator, "ARMS", ("D",))
    monkeypatch.setattr(coordinator, "FINAL_SEEDS", (7,))
    monkeypatch.setattr(coordinator, "MILESTONES", (1, 2))
    arrays = tmp_path / "arrays"
    arrays.mkdir()
    ids = ["2005-11-a", "2005-11-b", "2005-11-c", "2005-12-d"]
    for split in ("TRAIN", "VAL", "TEST"):
        np.savez(arrays / f"{split.lower()}.npz", z=np.ones((4, 48, 6)),
                 ids=np.array(ids), split=np.array(split), mean=np.zeros(6),
                 scale=np.ones(6), b_ref_sq=np.array(1.), n_bins=np.array(100))
    protocol = coordinator.base_protocol(tmp_path, arrays)
    coordinator.fresh_json(tmp_path / "PROTOCOL.json", protocol)
    digest = sha256_file(tmp_path / "PROTOCOL.json")
    for generation in (1, 2):
        contract = {
            "contract_version": 2, "arm": "D", "seed": 7, "c_r": 0.,
            "n_generated": 6, "n_aux": 0, "widths": list(coordinator.UNET_WIDTHS),
            "n_examples": 4, "batch_size": 128, "max_updates": 2,
            "estimator": "S", "residual": None,
            "source_hashes": protocol["training_source_hashes"],
            "data_hashes": protocol["training_data_hashes"],
            "normalization_hash": protocol["arrays"]["TRAIN"]["normalization_sha256"],
            "settings": {"milestones": [1, 2]},
        }
        run = tmp_path / "runs/D_s7"
        publish_generation(run, generation, {"state.pt": f"fixture-{generation}".encode()},
                           {"contract": contract})
        checkpoint = sha256_file(run / f"gen_{generation:08d}/state.pt")
        parent = tmp_path / f"artifacts/val/D_s7-g{generation}"
        ensemble = parent / "ensemble"
        publish_ensemble(ensemble, np.zeros((4, 39, 6, 48), np.float32), ids, {
            "protocol_sha256": digest, "checkpoint_sha256": checkpoint,
            "generation_updates": generation, "arm": "D", "seed": 7,
            "n_generated": 6, "arrays": protocol["arrays"]["VAL"],
            "sampler": protocol["samplers"]["VAL"],
            "state_convention": STATE_CONVENTION, "noise_convention": NOISE_CONVENTION,
        })
        for kind in ("raw", "corrected"):
            # Raw exact ties choose the earlier milestone; corrected chooses 2.
            value = 1. if kind == "raw" else 5. - generation
            row = {
                "schema_version": 2, "procedure": "diffusion", "protocol_sha256": digest,
                "checkpoint_sha256": checkpoint, "arrays": protocol["arrays"]["VAL"],
                "ensemble_commit_sha256": sha256_file(ensemble / "gen_00000000/COMMIT.json"),
                "ids": ids, "corrected": kind == "corrected", "n": 4,
                "scores": [value] * 4, "months": [200511, 200511, 200511, 200512],
                "mean": value, "month_balanced": value,
            }
            coordinator.fresh_json(parent / f"score-{kind}.json", row)
    return tmp_path, protocol, digest


def test_selection_uses_earlier_tie_and_independent_correction(study):
    root, protocol, digest = study
    coordinator.select(root, protocol, digest)
    selection = require_test_started(root, "TEST", digest)
    assert selection["selected"]["D_s7"]["raw"]["generation"] == 1
    assert selection["selected"]["D_s7"]["corrected"]["generation"] == 2
    assert selection["test_jobs"] == [["D_s7", 1], ["D_s7", 2]]
    assert len(selection["validation_score_sha256"]) == 4
    before = sha256_file(root / "SELECTION.json")
    coordinator.select(root, protocol, digest)
    assert sha256_file(root / "SELECTION.json") == before


@pytest.mark.parametrize("field,value", [
    ("month_balanced", -100.), ("mean", 100.), ("months", [200511] * 4),
    ("scores", [float("nan")] * 4), ("checkpoint_sha256", "0" * 64),
])
def test_selection_rejects_stale_or_inconsistent_score(study, field, value):
    root, protocol, digest = study
    path = root / "artifacts/val/D_s7-g1/score-raw.json"
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row))
    with pytest.raises(ContractMismatch):
        coordinator.select(root, protocol, digest)
    assert not (root / "SELECTION.json").exists()


def test_selection_reads_ensemble_payload_not_only_commit_file(study):
    root, protocol, digest = study
    payload = root / "artifacts/val/D_s7-g1/ensemble/gen_00000000/ensemble.npz"
    payload.write_bytes(b"corrupted after scoring")
    with pytest.raises(PublicationError, match="hash mismatch"):
        coordinator.select(root, protocol, digest)
    assert not (root / "SELECTION.json").exists()


def test_interruption_between_selection_and_phase_is_recoverable(study, monkeypatch):
    root, protocol, digest = study
    original = coordinator.fresh_json

    def interrupted(path, value):
        if path.name == "TEST_STARTED.json":
            raise OSError("owned publication fixture")
        return original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(coordinator, "fresh_json", interrupted)
        with pytest.raises(OSError, match="owned publication"):
            coordinator.select(root, protocol, digest)
    before = sha256_file(root / "SELECTION.json")
    assert not (root / "TEST_STARTED.json").exists()
    coordinator.select(root, protocol, digest)
    assert sha256_file(root / "SELECTION.json") == before
    assert require_test_started(root, "TEST", digest)["test_jobs"]


def test_test_receipt_required_before_any_reuse_or_array_read(study):
    root, protocol, digest = study
    with pytest.raises(BarrierError, match="committed selection"):
        coordinator.evaluate(root, protocol, digest, [0], "TEST")


@pytest.mark.parametrize("field,value", [
    ("widths", [8, 16, 16]), ("residual", "obs"), ("estimator", "M"),
    ("n_examples", 5), ("n_aux", 1),
])
def test_checkpoint_requires_actual_scientific_recipe(study, field, value):
    root, protocol, _ = study
    path = root / "runs/D_s7/gen_00000001/COMMIT.json"
    row = json.loads(path.read_text())
    row["contract"][field] = value
    path.write_bytes(json_bytes(row))
    with pytest.raises(ContractMismatch, match="recipe mismatch"):
        coordinator.verify_checkpoint(root, "D_s7", 1, protocol)


def test_fresh_receipt_does_not_replace_existing_bytes(tmp_path):
    path = tmp_path / "receipt.json"
    coordinator.fresh_json(path, {"first": True})
    with pytest.raises(PublicationError, match="exists"):
        coordinator.fresh_json(path, {"second": True})
    assert json.loads(path.read_text()) == {"first": True}
