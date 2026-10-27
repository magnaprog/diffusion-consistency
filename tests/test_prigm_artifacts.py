"""Check that saved predictions identify the data, model and sampling settings.

Local fixtures record ordered window IDs, array bytes, normalization values,
checkpoint identity, draw count and sampler settings. Tests alter those
records or their contents and require readers and scorers to reject the
mismatch. Successful checks establish consistency of these constructed
files; they do not authenticate a historical training run.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from prigm.artifacts import (
    NOISE_CONVENTION, STATE_CONVENTION, id_hash, json_bytes, load_arrays,
    merge_ensembles, publish_ensemble, read_ensemble, require_array_match,
)
from prigm.barrier import TestBarrierError as BarrierError, load_protocol, require_test_started
from prigm.recovery import ContractMismatch, PublicationError, sha256_file


def array_fixture(tmp_path, ids=None):
    ids = ids or ["2005-08-a", "2005-09-b", "2006-08-c"]
    path = tmp_path / "arrays.npz"
    np.savez(path, z=np.ones((len(ids), 48, 6)), ids=np.array(ids), split=np.array("VAL"),
             mean=np.zeros(6), scale=np.ones(6), b_ref_sq=np.array(1.), n_bins=np.array(100))
    return path


def ensemble_meta(identity):
    return {
        "arrays": identity, "protocol_sha256": "a" * 64, "checkpoint_sha256": "b" * 64,
        "n_generated": 6, "state_convention": STATE_CONVENTION,
        "noise_convention": NOISE_CONVENTION, "sampler": {"n_eval": 25, "n_draws": 39, "eval_seed": 7},
    }


def test_arrays_bind_split_unique_ids_normalization_and_bytes(tmp_path):
    path = array_fixture(tmp_path)
    z, ids, nrm, identity = load_arrays(path, "VAL", sha256_file(path))
    assert z.shape == (3, 48, 6)
    assert identity["ids_sha256"] == id_hash(ids)
    assert nrm.n_bins == 100
    with pytest.raises(ContractMismatch, match="split"):
        load_arrays(path, "TEST")
    with pytest.raises(ContractMismatch, match="hash"):
        load_arrays(path, "VAL", "0" * 64)
    data = dict(np.load(path))
    data["ids"][1] = data["ids"][0]
    np.savez(path, **data)
    with pytest.raises(ContractMismatch, match="duplicate"):
        load_arrays(path, "VAL")


def test_ensemble_corruption_and_nonfinite_values_fail_closed(tmp_path):
    path = array_fixture(tmp_path)
    _, ids, _, identity = load_arrays(path, "VAL")
    ensemble = np.zeros((3, 39, 6, 48), dtype=np.float32)
    root = tmp_path / "chunk"
    publish_ensemble(root, ensemble, ids, ensemble_meta(identity))
    actual, actual_ids, meta = read_ensemble(root)
    np.testing.assert_array_equal(actual, ensemble)
    assert actual_ids == ids
    require_array_match(meta, identity)
    with pytest.raises(ContractMismatch):
        require_array_match(meta, {**identity, "sha256": "0" * 64})
    with pytest.raises(PublicationError, match="exists"):
        publish_ensemble(root, ensemble, ids, ensemble_meta(identity))
    bad = ensemble.copy()
    bad[0, 0, 0, 20] = np.nan
    with pytest.raises(PublicationError, match="nonfinite"):
        publish_ensemble(tmp_path / "bad", bad, ids, ensemble_meta(identity))
    assert not (tmp_path / "bad").exists()
    payload = root / "gen_00000000/ensemble.npz"
    payload.write_bytes(b"corrupt")
    with pytest.raises(PublicationError, match="hash mismatch"):
        read_ensemble(root)


def test_merge_requires_exact_population_and_restores_order(tmp_path):
    path = array_fixture(tmp_path)
    _, ids, _, identity = load_arrays(path, "VAL")
    meta = ensemble_meta(identity)
    roots = [tmp_path / "second", tmp_path / "first"]
    publish_ensemble(roots[0], np.full((1, 39, 6, 48), 2, np.float32), ids[2:], {**meta, "offset": 2, "count": 1})
    publish_ensemble(roots[1], np.zeros((2, 39, 6, 48), np.float32), ids[:2], {**meta, "offset": 0, "count": 2})
    output = tmp_path / "merged"
    merge_ensembles(roots, output, ids)
    ensemble, actual_ids, _ = read_ensemble(output)
    assert actual_ids == ids
    assert ensemble[2].mean() == 2
    with pytest.raises(ContractMismatch, match="duplicate"):
        merge_ensembles([roots[0], roots[0], roots[1]], tmp_path / "dup", ids)
    with pytest.raises(ContractMismatch, match="exactly cover"):
        merge_ensembles([roots[0]], tmp_path / "partial", ids)


def test_merge_rejects_changed_checkpoint(tmp_path):
    path = array_fixture(tmp_path)
    _, ids, _, identity = load_arrays(path, "VAL")
    meta = ensemble_meta(identity)
    roots = [tmp_path / "one", tmp_path / "two"]
    publish_ensemble(roots[0], np.zeros((1, 39, 6, 48), np.float32), ids[:1], meta)
    publish_ensemble(roots[1], np.zeros((2, 39, 6, 48), np.float32), ids[1:], {**meta, "checkpoint_sha256": "c" * 64})
    with pytest.raises(ContractMismatch, match="incompatible"):
        merge_ensembles(roots, tmp_path / "mixed", ids)


def test_phase_binds_protocol_selection_and_previous_exposure(tmp_path):
    protocol = {"schema_version": 2, "exposure": "historical_test_exposed"}
    (tmp_path / "PROTOCOL.json").write_bytes(json_bytes(protocol))
    protocol_hash = sha256_file(tmp_path / "PROTOCOL.json")
    assert load_protocol(tmp_path, protocol_hash) == protocol
    with pytest.raises(BarrierError):
        require_test_started(tmp_path, "TEST", protocol_hash)
    selection = {"test_checkpoint_hashes": ["a" * 64]}
    (tmp_path / "SELECTION.json").write_bytes(json_bytes(selection))
    receipt = {
        "phase": "TEST_STARTED", "protocol_sha256": protocol_hash,
        "selection_sha256": sha256_file(tmp_path / "SELECTION.json"),
        "previous_test_exposure": True,
    }
    (tmp_path / "TEST_STARTED.json").write_bytes(json_bytes(receipt))
    assert require_test_started(tmp_path, "TEST", protocol_hash) == selection
    (tmp_path / "SELECTION.json").write_text(json.dumps({"test_checkpoint_hashes": []}))
    with pytest.raises(BarrierError, match="receipt"):
        require_test_started(tmp_path, "TEST", protocol_hash)
    with pytest.raises(BarrierError, match="unknown split"):
        require_test_started(tmp_path, "training-test")


def test_existing_evaluation_output_refused_before_loading(tmp_path):
    from prigm.run_eval import main

    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(PublicationError, match="exists"):
        main([
            "--run-root", "missing", "--generation", "1", "--checkpoint-sha256", "0" * 64,
            "--out", str(output), "--arrays", "missing", "--split", "VAL", "--n-eval", "25",
            "--study-root", "missing", "--protocol-sha256", "0" * 64,
        ])


def test_diagnostics_keep_position_channel_and_month_axes():
    from prigm.run_secondaries import summarize_diagnostics

    records = [
        {"score": 0., "position": [[0., 1.], [2., 3.]]},
        {"score": 0., "position": [[0., 1.], [2., 3.]]},
        {"score": 9., "position": [[10., 11.], [12., 13.]]},
    ]
    result = summarize_diagnostics(records, np.array([200508, 200508, 200509]))
    assert result["score"]["month_balanced"] == 4.5
    np.testing.assert_equal(result["position"]["month_balanced"], [[5., 6.], [7., 8.]])


def test_control_ensembles_use_common_scoring_and_preserve_float64(tmp_path):
    from prigm.config import GAUSSIAN_JITTER
    from prigm.run_controls import main as controls_main
    from prigm.run_score import main as score_main
    from prigm.run_secondaries import main as secondaries_main

    evaluation = array_fixture(tmp_path)
    train = tmp_path / "train.npz"
    train_blob = dict(np.load(evaluation))
    train_blob["split"] = np.array("TRAIN")
    np.savez(train, **train_blob)
    _, ids, _, val_identity = load_arrays(evaluation, "VAL")
    _, _, _, train_identity = load_arrays(train, "TRAIN")
    protocol = {
        "schema_version": 2, "exposure": "historical_test_exposed",
        "arrays": {"VAL": val_identity, "TRAIN": train_identity},
        "inference_source_hashes": {
            f"prigm/{p.name}": sha256_file(p)
            for p in (Path(__file__).resolve().parents[1] / "prigm").glob("*.py")
        },
        "controls": {
            "interpolation": {"conditioning": "two_boundaries"},
            "persistence": {"conditioning": "left_boundary"},
            "gaussian": {"rho": 0.0, "jitter": GAUSSIAN_JITTER, "seed": 13},
        },
    }
    (tmp_path / "PROTOCOL.json").write_bytes(json_bytes(protocol))
    digest = sha256_file(tmp_path / "PROTOCOL.json")
    controls = tmp_path / "controls"
    shared = ["--study-root", str(tmp_path), "--protocol-sha256", digest, "--split", "VAL"]
    controls_main(["--train-arrays", str(train), "--eval-arrays", str(evaluation),
                   "--output", str(controls), *shared])
    for name in protocol["controls"]:
        ensemble, sampled_ids, meta = read_ensemble(controls / name)
        assert ensemble.dtype == np.float64
        assert sampled_ids == ids
        assert meta["procedure"] == name
        output = tmp_path / f"{name}.json"
        score_main(["--ensemble", str(controls / name), "--arrays", str(evaluation),
                    "--output", str(output), *shared])
        result = json.loads(output.read_text())
        assert result["n"] == 3 and result["checkpoint_sha256"] is None
        if name != "gaussian":
            assert result["month_balanced"] == 0.0
    details = tmp_path / "details.json"
    secondaries_main(["--ensemble", str(controls / "interpolation"), "--arrays", str(evaluation),
                      "--output", str(details), *shared])
    result = json.loads(details.read_text())
    assert result["es5"]["month_balanced"] == 0.0
    assert np.asarray(result["cov90_by_position_channel"]["month_balanced"]).shape == (12, 5)
    assert result["cov90_mean"]["month_balanced"] == 1.0


@pytest.mark.parametrize("channels", [5, 6])
def test_real_evaluation_path_ignores_hidden_truth(tmp_path, channels):
    import torch
    import prigm
    from prigm.checkpoint import frozen_source_snapshot, save_torch_payload
    from prigm.model import ConditionalUNet
    from prigm.recovery import publish_generation, sha256_bytes
    from prigm.run_eval import main

    arrays = array_fixture(tmp_path)
    _, ids, _, identity = load_arrays(arrays, "VAL")
    source, snapshot = frozen_source_snapshot(Path(prigm.__file__).resolve().parents[1])
    contract = {
        "contract_version": 2, "arm": "G5" if channels == 5 else "D", "seed": 7,
        "n_generated": channels, "widths": [8, 16, 16], "source_hashes": source,
        "data_hashes": {"fixture": "fixed_training_input"},
        "normalization_hash": identity["normalization_sha256"],
        "n_aux": 0, "n_examples": 20, "batch_size": 8, "max_updates": 7,
        "settings": {"source_snapshot_sha256": sha256_bytes(snapshot)},
    }
    torch.manual_seed(7)
    model = ConditionalUNet(n_generated=channels, widths=(8, 16, 16))
    payload = save_torch_payload({
        "model": model.state_dict(), "contract": contract, "updates": 0, "epochs": 0, "cursor": 0,
        "normalization": {"mean": torch.zeros(6, dtype=torch.float64),
                          "scale": torch.ones(6, dtype=torch.float64), "b_ref_sq": 1., "n_bins": 100},
    })
    run = tmp_path / "run"
    run.mkdir()
    (run / "source_snapshot.zip").write_bytes(snapshot)
    publish_generation(run, 0, {"state.pt": payload}, {"contract": contract, "updates": 0, "epochs": 0, "cursor": 0})
    state_hash = sha256_file(run / "gen_00000000/state.pt")
    protocol = {
        "schema_version": 2, "exposure": "historical_test_exposed",
        "inference_source_hashes": source, "training_source_hashes": source,
        "training_data_hashes": contract["data_hashes"], "arrays": {"VAL": identity},
        "samplers": {"VAL": {"n_eval": 2, "n_draws": 39, "eval_seed": 7}},
    }
    protocol_path = tmp_path / "PROTOCOL.json"
    predictions = []
    for attempt in range(2):
        if attempt:
            values = dict(np.load(arrays))
            values["z"][:, 18:30] += 1000
            np.savez(arrays, **values)
            protocol["arrays"]["VAL"] = load_arrays(arrays, "VAL")[3]
        protocol_path.write_bytes(json_bytes(protocol))
        output = tmp_path / f"draws-{attempt}"
        main([
            "--run-root", str(run), "--generation", "0", "--checkpoint-sha256", state_hash,
            "--out", str(output), "--arrays", str(arrays), "--split", "VAL", "--n-eval", "2",
            "--limit", "3", "--study-root", str(tmp_path),
            "--protocol-sha256", sha256_file(protocol_path),
        ])
        ensemble, actual_ids, _ = read_ensemble(output)
        assert actual_ids == ids
        predictions.append(ensemble)
    np.testing.assert_array_equal(*predictions)


def test_interrupted_chunk_is_preserved_before_redraw(tmp_path):
    from experiments.run_prigm_repair import preserve_uncommitted
    from prigm.recovery import RunLock

    root = tmp_path / "chunk-0000"
    temporary = root / ".tmp_gen_00000000"
    temporary.mkdir(parents=True)
    (temporary / "ensemble.npz").write_bytes(b"unfinished payload")
    with RunLock(root):
        with pytest.raises(PublicationError, match="locked"):
            preserve_uncommitted(root)
    preserve_uncommitted(root)
    assert not root.exists()
    preserved = root.with_name("chunk-0000.incomplete-1")
    assert (preserved / ".tmp_gen_00000000/ensemble.npz").read_bytes() == b"unfinished payload"
    assert (preserved / "PRESERVED.json").is_file()
    meta = ensemble_meta({"fixture": True})
    values = np.zeros((1, 39, 6, 48), np.float32)
    publish_ensemble(root, values, ["window"], meta)
    before = sha256_file(root / "gen_00000000/COMMIT.json")
    preserve_uncommitted(root)
    assert sha256_file(root / "gen_00000000/COMMIT.json") == before


@pytest.mark.parametrize("entrypoint", ["prigm.run_score", "prigm.run_secondaries"])
@pytest.mark.parametrize("damage", [None, "n_eval", "n_draws", "eval_seed", "missing"])
def test_scoring_requires_protocol_sampler(tmp_path, entrypoint, damage):
    from importlib import import_module

    arrays = array_fixture(tmp_path)
    _, ids, _, identity = load_arrays(arrays, "VAL")
    meta = ensemble_meta(identity)
    sampler = dict(meta["sampler"])
    protocol = {
        "schema_version": 2, "exposure": "historical_test_exposed",
        "arrays": {"VAL": identity}, "samplers": {"VAL": sampler},
        "inference_source_hashes": {
            f"prigm/{p.name}": sha256_file(p)
            for p in (Path(__file__).resolve().parents[1] / "prigm").glob("*.py")
        },
    }
    protocol_path = tmp_path / "PROTOCOL.json"
    protocol_path.write_bytes(json_bytes(protocol))
    digest = sha256_file(protocol_path)
    meta["protocol_sha256"] = digest
    if damage == "missing":
        del meta["sampler"]
    elif damage is not None:
        meta["sampler"][damage] += 1
    ensemble = tmp_path / "ensemble"
    publish_ensemble(ensemble, np.zeros((len(ids), 39, 6, 48), np.float32), ids, meta)
    output = tmp_path / "score.json"
    argv = ["--ensemble", str(ensemble), "--arrays", str(arrays), "--output", str(output),
            "--split", "VAL", "--study-root", str(tmp_path), "--protocol-sha256", digest]
    main = import_module(entrypoint).main
    if damage is not None:
        with pytest.raises(ContractMismatch, match="sampler"):
            main(argv)
        assert not output.exists()
    else:
        assert main(argv) == 0
        result = json.loads(output.read_text())
        actual = result["month_balanced"] if entrypoint == "prigm.run_score" else result["es5"]["month_balanced"]
        assert actual == pytest.approx(np.sqrt(60))
