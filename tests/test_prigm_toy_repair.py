"""Local synthetic tests for the toy coordinator; never launch real training."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import run_prigm_toy_repair as toy
from prigm.artifacts import id_hash, load_arrays, publish_ensemble, read_ensemble
from prigm.barrier import TestBarrierError as BarrierError, load_protocol, require_test_started
from prigm.checkpoint import frozen_source_snapshot, save_torch_payload
from prigm.keys import noise_key, pcg64
from prigm.recovery import ContractMismatch, PublicationError, publish_generation, sha256_bytes, sha256_file
from prigm.sampler import initial_noise
from prigm.toy import ALPHAS, PS, pooled_toy_observations, state_summary, transition_matrix
from prigm.toy_windows import make_toy_training, sample_boundaries


@pytest.fixture(scope="module")
def training_source():
    # Real deterministic source construction is inexpensive; no optimizer steps.
    return toy.training_identity()


@pytest.fixture
def study(tmp_path, monkeypatch, training_source):
    root = tmp_path / "owned-study"
    root.mkdir()
    monkeypatch.setattr(toy, "training_identity", lambda: copy.deepcopy(training_source))
    monkeypatch.setattr(toy, "launch", lambda *args: pytest.fail("unexpected subprocess launch"))
    toy.main(["protocol", "--study-root", str(root), "--gpus", "0,1,2,3"])
    digest = sha256_file(root / "PROTOCOL.json")
    return root, load_protocol(root, digest), digest


def write_checkpoint(study, protocol, fit, *, mutate=None, payload_mutate=None, generation=5000):
    """Publish a synthetic state with authentic shapes and counters, no training."""
    root = study / "runs" / fit["name"]
    root.mkdir(parents=True, exist_ok=True)
    hashes, snapshot = frozen_source_snapshot(toy.ROOT)
    (root / "source_snapshot.zip").write_bytes(snapshot)
    contract = {k: v for k, v in protocol["recipe"].items() if k not in ("fits", "milestones")}
    contract.update({k: v for k, v in fit.items() if k != "name"})
    contract.update(contract_version=2, source_hashes=hashes,
                    data_hashes=protocol["training_data_hashes"],
                    normalization_hash=protocol["training_identity"]["normalization_sha256"],
                    settings={**copy.deepcopy(protocol["training_settings"]),
                              "environment": {"device_type": "cuda", "torch_threads": 2}})
    contract = copy.deepcopy(contract)
    if mutate:
        mutate(contract)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = toy.ConditionalUNet(n_generated=6, widths=(32, 64, 64))
    nrm = protocol["training_identity"]["normalization"]
    counters = {"updates": generation, "epochs": generation // 32, "cursor": (generation % 32) * 128}
    payload = {"contract": contract, "model": model.state_dict(), **counters,
               "optimizer": {}, "rng": {},
               "data_hashes": protocol["training_identity"]["normalized_data_hashes"],
               "normalization": {"mean": torch.tensor(nrm["mean"], dtype=torch.float64),
                                 "scale": torch.tensor(nrm["scale"], dtype=torch.float64),
                                 "b_ref_sq": nrm["b_ref_sq"], "n_bins": nrm["n_bins"]}}
    if payload_mutate:
        payload_mutate(payload)
    publish_generation(root, generation,
                       {"state.pt": save_torch_payload(payload), "meta.json": toy.json_bytes(contract)},
                       {"contract": contract, **counters})
    return root


def reduce_fixture_to_one_fit(study):
    """Small evaluate fixture, not a user-facing protocol or alternative CLI recipe."""
    root, protocol, _ = study
    protocol = copy.deepcopy(protocol)
    protocol["recipe"]["fits"] = protocol["recipe"]["fits"][:1]
    (root / "PROTOCOL.json").write_bytes(toy.json_bytes(protocol))
    digest = sha256_file(root / "PROTOCOL.json")
    (root / "PROTOCOL_RECEIPT.json").write_bytes(toy.json_bytes({
        "protocol_sha256": digest, "published_before": ["train", "evaluate"],
    }))
    (root / "logs").mkdir(exist_ok=True)
    return root, protocol, digest


def prepare_final_receipts(study, protocol, digest):
    directory = study / "artifacts/training"
    directory.mkdir(exist_ok=True)
    rows = {}
    for fit in protocol["recipe"]["fits"]:
        toy.fresh_json(directory / f"{fit['name']}-request.json", toy.fit_request(study, fit, digest))
        write_checkpoint(study, protocol, fit)
        row = toy.completed_fit(study, fit, protocol, digest)
        toy.fresh_json(directory / f"{fit['name']}-completed.json", row)
        rows[fit["name"]] = row
    toy.fresh_json(study / "TRAINED.json", {"protocol_sha256": digest, "completed_runs": rows})
    return rows


def option(command, flag):
    return command[command.index(flag) + 1]


def test_training_source_is_canonical_run_fit_toy_not_test_derived(training_source):
    record, nrm = training_source
    z, alpha, p = make_toy_training(4096, seed=7 + 99)
    assert record["data_hashes"] == {"toy_arrays_sha256": sha256_bytes(np.ascontiguousarray(z, dtype="<f8").tobytes())}
    assert record["seed"] == 106 and record["n_examples"] == 4096
    assert record["normalization"]["n_bins"] == 6
    np.testing.assert_array_equal(nrm.mean, pooled_toy_observations().mean(axis=0))
    assert sum(r["n"] for r in record["regime_counts"]) == 4096
    assert sorted(r["n"] for r in record["regime_counts"]) == [455] * 8 + [456]
    assert np.isfinite(z).all()
    for a in ALPHAS:
        rows = z[alpha == a]
        states = np.array([state_summary(a, s) for s in (0, 1)])
        assert np.all(np.any(np.all(rows[..., None, :] == states, axis=-1), axis=-1))
        np.testing.assert_array_equal(rows[:, :18], np.repeat(rows[:, :1], 18, axis=1))
        np.testing.assert_array_equal(rows[:, 30:], np.repeat(rows[:, -1:], 18, axis=1))
    assert set(p) == set(PS)


def test_stationary_boundary_law_and_canonical_oracle_noise():
    class BoundaryProbe:
        def integers(self, low, high):
            assert (low, high) == (0, 2)
            return 1

        def choice(self, n, p):
            assert n == 2
            np.testing.assert_allclose(p, np.linalg.matrix_power(transition_matrix(0.1), 13)[1])
            return 0

    assert sample_boundaries(0.1, BoundaryProbe()) == (1, 0)
    key = noise_key("eval", "TOY", "TOY:287", 38, 7)
    expected = pcg64(key).standard_normal((48, 6))
    np.testing.assert_array_equal(initial_noise(6, 48, key), expected)


def test_protocol_freezes_all_nine_fits_source_and_stationary_arrays(study, training_source):
    root, protocol, digest = study
    fits = protocol["recipe"]["fits"]
    assert len(fits) == 9 and len({fit["name"] for fit in fits}) == 9
    assert [(r["arm"], r["c_r"]) for r in fits] == [
        ("D", 0.0), ("I-M", .0005), ("I-M", .005), ("I-S", .0005), ("I-S", .005),
        ("O-M", .0005), ("O-M", .005), ("O-S", .0005), ("O-S", .005),
    ]
    assert [(r["residual"], r["estimator"]) for r in fits] == [
        (None, "S"), ("inst", "M"), ("inst", "M"), ("inst", "S"), ("inst", "S"),
        ("obs", "M"), ("obs", "M"), ("obs", "S"), ("obs", "S"),
    ]
    assert protocol["training_data_hashes"] == training_source[0]["data_hashes"]
    assert protocol["training_source_hashes"] == protocol["inference_source_hashes"] == toy.source_identity()
    assert protocol["coordinator_sha256"] == sha256_file(Path(toy.__file__))
    assert "experiments/run_prigm_repair.py" not in protocol["training_source_hashes"]
    assert protocol["samplers"] == {"TOY": {"n_eval": 50, "n_draws": 39, "eval_seed": 7}}
    assert protocol["noise_followup_change"]["previous_proposal_eval_seed"] == 7002
    assert "not selected from scores" in protocol["noise_followup_change"]["reason"]
    assert require_test_started(root, "TOY", digest) is None
    assert not (root / "TEST_STARTED.json").exists()
    assert not (root / "runs").exists()
    z, ids, nrm, identity = load_arrays(root / "artifacts/toy.npz", "TOY")
    assert ids == [f"TOY:{i}" for i in range(288)] and id_hash(ids) == identity["ids_sha256"]
    ez, ea, ep = toy.build_eval_windows(32)
    np.testing.assert_array_equal(z, ez)
    with np.load(root / "artifacts/toy.npz", allow_pickle=False) as arrays:
        assert arrays["boundary_law"].item() == "stationary"
        np.testing.assert_array_equal(arrays["alpha"], ea)
        np.testing.assert_array_equal(arrays["p"], ep)
        assert all(((arrays["alpha"] == a) & (arrays["p"] == p)).sum() == 32 for a in ALPHAS for p in PS)
    np.testing.assert_array_equal(nrm.scale, training_source[1].scale)
    assert toy.execution_protocol(root, digest) == protocol


def test_old_outputs_and_unpublished_or_wrong_protocol_refused(tmp_path, study):
    old = tmp_path / "old"
    old.mkdir()
    sentinel = old / "historical-results.json"
    sentinel.write_text("keep unchanged")
    with pytest.raises(PublicationError, match="fresh empty"):
        toy.main(["protocol", "--study-root", str(old)])
    assert sentinel.read_text() == "keep unchanged"
    root, _, digest = study
    with pytest.raises(PublicationError, match="fresh empty"):
        toy.main(["protocol", "--study-root", str(root)])
    for stage in ("train", "evaluate"):
        with pytest.raises(SystemExit):
            toy.main([stage, "--study-root", str(root)])
        with pytest.raises(BarrierError, match="protocol"):
            toy.main([stage, "--study-root", str(root), "--protocol-sha256", "0" * 64])
        with pytest.raises(BarrierError, match="protocol"):
            toy.main([stage, "--study-root", str(old), "--protocol-sha256", digest])
    assert not (root / "runs").exists()
    assert set(old.iterdir()) == {sentinel}


@pytest.mark.parametrize("field", ["training_source_hashes", "inference_source_hashes", "coordinator_sha256"])
def test_protocol_source_pins_are_enforced(study, field):
    root, protocol, _ = study
    protocol[field] = "0" * 64 if field == "coordinator_sha256" else {}
    (root / "PROTOCOL.json").write_bytes(toy.json_bytes(protocol))
    with pytest.raises(ContractMismatch, match="source"):
        toy.execution_protocol(root, sha256_file(root / "PROTOCOL.json"))


def test_in_process_source_edit_is_rejected(monkeypatch):
    monkeypatch.setattr(toy, "source_identity", lambda: {"prigm/run_fit.py": "changed"})
    with pytest.raises(ContractMismatch, match="after coordinator import"):
        toy.require_sources()


@pytest.mark.parametrize("field,value", [
    ("widths", [64, 128, 128]), ("residual", "inst"), ("estimator", "M"), ("c_r", .005),
    ("seed", 13), ("n_generated", 5), ("n_aux", 1), ("n_examples", 4095),
    ("batch_size", 64), ("max_updates", 30000), ("contract_version", 1),
    ("source_hashes", {}), ("data_hashes", {"toy_arrays_sha256": "0" * 64}),
    ("normalization_hash", "0" * 64),
])
def test_checkpoint_recipe_mismatch_is_not_sampling_eligible(study, field, value):
    root, protocol, _ = study
    fit = protocol["recipe"]["fits"][0]
    write_checkpoint(root, protocol, fit, mutate=lambda c: c.update({field: value}))
    with pytest.raises(ContractMismatch, match="recipe mismatch"):
        toy.verify_checkpoint(root, fit, protocol)


@pytest.mark.parametrize("change", [
    lambda c: c["settings"].update(milestones=[1000, 5000]),
    lambda c: c["settings"]["data"].update(seed=105),
    lambda c: c["settings"]["optimizer"].update(lr=1e-3),
    lambda c: c["settings"]["precision"].update(matmul_tf32=True),
    lambda c: c["settings"]["environment"].update(device_type="cpu"),
])
def test_checkpoint_settings_are_checked(study, change):
    root, protocol, _ = study
    fit = protocol["recipe"]["fits"][0]
    write_checkpoint(root, protocol, fit, mutate=change)
    with pytest.raises(ContractMismatch, match="settings mismatch"):
        toy.verify_checkpoint(root, fit, protocol)


@pytest.mark.parametrize("change,match", [
    (lambda p: p.update(updates=4999), "counters"),
    (lambda p: p.update(data_hashes={}), "training data"),
    (lambda p: p["normalization"].update(n_bins=288), "normalization"),
    (lambda p: next(iter(p["model"].values())).fill_(float("nan")), "model weights"),
])
def test_checkpoint_payload_is_verified_not_only_commit_label(study, change, match):
    root, protocol, _ = study
    fit = protocol["recipe"]["fits"][0]
    write_checkpoint(root, protocol, fit, payload_mutate=change)
    with pytest.raises(ContractMismatch, match=match):
        toy.verify_checkpoint(root, fit, protocol)


def test_only_exact_final_generation_and_snapshot_are_accepted(study):
    root, protocol, _ = study
    fit = protocol["recipe"]["fits"][0]
    run = write_checkpoint(root, protocol, fit, generation=4992)
    with pytest.raises(PublicationError, match="JSON"):
        toy.verify_checkpoint(root, fit, protocol)
    write_checkpoint(root, protocol, fit)
    (run / "LATEST").write_text("4992\n")
    assert toy.verify_checkpoint(root, fit, protocol)["updates"] == 5000
    (run / "source_snapshot.zip").write_bytes(b"old source")
    with pytest.raises(PublicationError, match="snapshot hash"):
        toy.verify_checkpoint(root, fit, protocol)


def test_mocked_train_launches_exact_nine_commands_and_records_pins(study, monkeypatch):
    root, protocol, digest = study
    calls = []
    fits = {fit["name"]: fit for fit in protocol["recipe"]["fits"]}

    def fake_launch(command, log, gpu):
        assert load_protocol(root, digest) == protocol
        name = Path(option(command, "--run-root")).name
        assert (root / f"artifacts/training/{name}-request.json").is_file()
        assert command[4] == "prigm.run_fit"
        assert "--toy" in command and "--arrays" not in command and "--resume" not in command
        for flag, value in (("--seed", "7"), ("--batch-size", "128"), ("--updates", "5000"),
                            ("--budget-updates", "5000"), ("--milestones", "5000")):
            assert option(command, flag) == value
        assert float(option(command, "--c-r")) == fits[name]["c_r"]
        write_checkpoint(root, protocol, fits[name])
        calls.append((name, gpu))

    monkeypatch.setattr(toy, "launch", fake_launch)
    toy.main(["train", "--study-root", str(root), "--protocol-sha256", digest, "--gpus", "0,1,2,3"])
    assert len(calls) == 9
    assert dict(calls) == {fit["name"]: i % 4 for i, fit in enumerate(protocol["recipe"]["fits"])}
    trained = json.loads((root / "TRAINED.json").read_text())
    assert set(trained["completed_runs"]) == set(fits)
    assert {toy.verify_checkpoint(root, fit, protocol)["contract"]["data_hashes"]["toy_arrays_sha256"]
            for fit in fits.values()} == {protocol["training_data_hashes"]["toy_arrays_sha256"]}
    for name, row in trained["completed_runs"].items():
        assert row["generation_updates"] == 5000
        assert row["checkpoint_sha256"] == sha256_file(root / f"runs/{name}/gen_00005000/state.pt")
    # Idempotence means reading and verifying owned committed outputs, not relaunching.
    toy.main(["train", "--study-root", str(root), "--protocol-sha256", digest])
    assert len(calls) == 9


def test_train_refuses_unowned_old_run(study):
    root, protocol, digest = reduce_fixture_to_one_fit(study)
    fit = protocol["recipe"]["fits"][0]
    write_checkpoint(root, protocol, fit)
    with pytest.raises(PublicationError, match="pre-existing run"):
        toy.train(root, protocol, digest, [0])
    assert not (root / f"artifacts/training/{fit['name']}-request.json").exists()


def test_owned_training_resume_keeps_the_same_total_target(study, monkeypatch):
    root, protocol, digest = reduce_fixture_to_one_fit(study)
    fit = protocol["recipe"]["fits"][0]
    requests = root / "artifacts/training"
    requests.mkdir()
    toy.fresh_json(requests / f"{fit['name']}-request.json", toy.fit_request(root, fit, digest))
    write_checkpoint(root, protocol, fit, generation=100)
    calls = []

    def fake_launch(command, log, gpu):
        assert "--resume" in command
        assert option(command, "--updates") == option(command, "--budget-updates") == "5000"
        write_checkpoint(root, protocol, fit)
        calls.append(command)

    monkeypatch.setattr(toy, "launch", fake_launch)
    toy.train(root, protocol, digest, [0])
    assert len(calls) == 1 and (root / "TRAINED.json").is_file()


def test_sampling_requires_manifest_pins_for_every_final_checkpoint(study):
    root, protocol, digest = reduce_fixture_to_one_fit(study)
    prepare_final_receipts(root, protocol, digest)
    trained = json.loads((root / "TRAINED.json").read_text())
    next(iter(trained["completed_runs"].values()))["checkpoint_sha256"] = "0" * 64
    (root / "TRAINED.json").write_bytes(toy.json_bytes(trained))
    with pytest.raises(ContractMismatch, match="pin the nine final"):
        toy.evaluate(root, protocol, digest, [0])
    assert not (root / "artifacts/toy").exists()


def test_last_fit_recipe_mismatch_stops_all_sampling(study):
    root, protocol, digest = study
    prepare_final_receipts(root, protocol, digest)
    name = protocol["recipe"]["fits"][-1]["name"]
    commit = root / f"runs/{name}/gen_00005000/COMMIT.json"
    meta = json.loads(commit.read_text())
    meta["contract"]["estimator"] = "M"
    commit.write_bytes(toy.json_bytes(meta))
    with pytest.raises(ContractMismatch, match="recipe mismatch"):
        toy.main(["evaluate", "--study-root", str(root), "--protocol-sha256", digest])
    assert not (root / "artifacts/toy").exists()


def test_mocked_evaluation_covers_288_ids_and_both_common_scoring_clis(study, monkeypatch):
    root, protocol, digest = reduce_fixture_to_one_fit(study)
    rows = prepare_final_receipts(root, protocol, digest)
    fit = protocol["recipe"]["fits"][0]
    arrays_identity = protocol["arrays"]["TOY"]
    ids = [f"TOY:{i}" for i in range(288)]
    sampled, scored = [], []

    def fake_launch(command, log, gpu):
        assert option(command, "--split") == "TOY"
        assert option(command, "--protocol-sha256") == digest
        assert option(command, "--arrays") == str(root / "artifacts/toy.npz")
        if "prigm.run_eval" in command:
            offset, limit = int(option(command, "--offset")), int(option(command, "--limit"))
            assert limit == 8 and option(command, "--generation") == "5000"
            assert option(command, "--eval-seed") == "7" and option(command, "--n-eval") == "50"
            assert option(command, "--checkpoint-sha256") == rows[fit["name"]]["checkpoint_sha256"]
            out = Path(option(command, "--out"))
            meta = {"protocol_sha256": digest, "arrays": arrays_identity,
                    "checkpoint_sha256": rows[fit["name"]]["checkpoint_sha256"],
                    "generation_updates": 5000, "arm": fit["arm"], "seed": 7, "n_generated": 6,
                    "sampler": protocol["samplers"]["TOY"], "state_convention": toy.STATE_CONVENTION,
                    "noise_convention": toy.NOISE_CONVENTION, "offset": offset, "count": limit,
                    "inference_source_sha256": protocol["inference_source_hashes"]["prigm/run_eval.py"]}
            publish_ensemble(out, np.zeros((8, 39, 6, 48), np.float32), ids[offset:offset + limit], meta)
            sampled.append(offset)
        else:
            secondary = "prigm.run_secondaries" in command
            assert secondary or "prigm.run_score" in command
            corrected = "--correct" in command
            ensemble = Path(option(command, "--ensemble"))
            assert read_ensemble(ensemble)[1] == ids
            # Exercise the real common scoring CLIs on synthetic ensembles only.
            from prigm.run_score import main as score_main
            from prigm.run_secondaries import main as secondaries_main

            entry_point = secondaries_main if secondary else score_main
            assert entry_point(command[command.index("-m") + 2:]) == 0
            output = json.loads(Path(option(command, "--output")).read_text())
            if secondary:
                assert len(output["per_window"]) == 288
                assert np.asarray(output["per_window"][0]["bias_by_position_channel"]).shape == (12, 5)
            else:
                assert len(output["scores"]) == 288
            scored.append((secondary, corrected))

    monkeypatch.setattr(toy, "launch", fake_launch)
    toy.evaluate(root, protocol, digest, [0])
    assert sampled == list(range(0, 288, 8)) and sampled[-1] == 280
    assert scored == [(False, False), (False, True), (True, False), (True, True)]
    manifest_path = root / "EVALUATED.json"
    before = sha256_file(manifest_path)
    result = json.loads(manifest_path.read_text())["completed_runs"][fit["name"]]
    assert result["checkpoint_sha256"] == rows[fit["name"]]["checkpoint_sha256"]
    for rec in result["outputs"].values():
        assert rec["sha256"] == sha256_file(root / rec["path"])
    toy.evaluate(root, protocol, digest, [0])
    assert len(sampled) == 36 and len(scored) == 4 and sha256_file(manifest_path) == before
    # Metadata alone is not enough for reuse: retain every finite per-window score.
    output = root / result["outputs"]["score-raw"]["path"]
    row = json.loads(output.read_text())
    row["scores"] = row["scores"][:-1]
    output.write_bytes(toy.json_bytes(row))
    with pytest.raises(ContractMismatch, match="incomplete energy"):
        toy.evaluate(root, protocol, digest, [0])
    assert len(sampled) == 36


@pytest.mark.parametrize("field,value", [
    ("sampler", {"n_eval": 50, "n_draws": 39, "eval_seed": 7002}),
    ("generation_updates", 4992), ("offset", 8),
    ("protocol_sha256", "0" * 64), ("checkpoint_sha256", "0" * 64),
])
def test_reused_chunk_must_match_exact_sampler_and_final_pin(study, field, value):
    root, protocol, digest = study
    fit = protocol["recipe"]["fits"][0]
    ids = [f"TOY:{i}" for i in range(8)]
    checkpoint_hash = "a" * 64
    meta = {"protocol_sha256": digest, "arrays": protocol["arrays"]["TOY"],
            "checkpoint_sha256": checkpoint_hash, "generation_updates": 5000,
            "arm": fit["arm"], "seed": 7, "n_generated": 6,
            "sampler": protocol["samplers"]["TOY"], "state_convention": toy.STATE_CONVENTION,
            "noise_convention": toy.NOISE_CONVENTION, "offset": 0, "count": 8,
            "inference_source_sha256": protocol["inference_source_hashes"]["prigm/run_eval.py"]}
    meta[field] = value
    chunk = root / "artifacts/chunk-fixture"
    publish_ensemble(chunk, np.zeros((8, 39, 6, 48), np.float32), ids, meta)
    with pytest.raises(ContractMismatch, match="pinned toy procedure"):
        toy.check_ensemble(chunk, ids, protocol, digest, fit, checkpoint_hash, offset=0)


def test_existing_evaluation_outputs_need_owned_request(study):
    root, protocol, digest = reduce_fixture_to_one_fit(study)
    rows = prepare_final_receipts(root, protocol, digest)
    name = next(iter(rows))
    parent = root / f"artifacts/toy/{name}-g5000"
    parent.mkdir(parents=True)
    (parent / "score-raw.json").write_text("historical")
    with pytest.raises(PublicationError, match="pre-existing evaluation outputs"):
        toy.evaluate(root, protocol, digest, [0])
    assert (parent / "score-raw.json").read_text() == "historical"


def test_preserve_failed_owned_chunk_but_never_committed_or_locked(tmp_path):
    root = tmp_path / "chunk-0000"
    temporary = root / ".tmp_gen_00000000"
    temporary.mkdir(parents=True)
    (temporary / "ensemble.npz").write_bytes(b"unfinished")
    with toy.RunLock(root):
        with pytest.raises(PublicationError, match="locked"):
            toy.preserve_uncommitted(root)
    toy.preserve_uncommitted(root)
    preserved = tmp_path / "chunk-0000.incomplete-1"
    assert (preserved / ".tmp_gen_00000000/ensemble.npz").read_bytes() == b"unfinished"
    assert (preserved / "PRESERVED.json").exists() and not root.exists()
    (root / "gen_00000000").mkdir(parents=True)
    (root / "gen_00000000/COMMIT.json").write_text("even a corrupt commit must not be silently discarded")
    toy.preserve_uncommitted(root)
    assert (root / "gen_00000000/COMMIT.json").exists()


def test_chunk_committed_while_acquiring_lock_is_not_preserved(tmp_path, monkeypatch):
    root = tmp_path / "chunk"
    root.mkdir()
    original_lock = toy.RunLock

    class ConcurrentCommit(original_lock):
        def __enter__(self):
            (root / "gen_00000000").mkdir()
            (root / "gen_00000000/COMMIT.json").write_text("published by the previous writer")
            return super().__enter__()

    monkeypatch.setattr(toy, "RunLock", ConcurrentCommit)
    toy.preserve_uncommitted(root)
    assert (root / "gen_00000000/COMMIT.json").exists()
    assert not root.with_name("chunk.incomplete-1").exists()


def test_gpu_launch_is_local_and_requires_cuda(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(toy.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    toy.launch(["python", "-m", "prigm.run_fit"], tmp_path / "run.log", 3)
    _, call = calls[0]
    assert call["cwd"] == toy.ROOT and call["check"] is True
    env = call["env"]
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["PRIGM_REQUIRE_CUDA"] == env["PISCES_REQUIRE_CUDA"] == "1"
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == "2"


def test_symlink_paths_cannot_redirect_owned_outputs(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "artifacts").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PublicationError, match="symlinks"):
        toy.owned_path(root, "artifacts/toy.npz")


def test_receipt_failure_does_not_publish_partial_state(tmp_path, monkeypatch):
    import prigm.recovery as recovery
    path = tmp_path / "receipt.json"
    original = recovery.os.replace

    def fail_publication(source, destination):
        if Path(destination) == path:
            raise OSError("owned receipt publication failure")
        return original(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(recovery.os, "replace", fail_publication)
        with pytest.raises(OSError, match="owned receipt"):
            toy.record_or_check(path, {"complete": True})
    assert not path.exists()
    toy.record_or_check(path, {"complete": True})
    before = path.read_bytes()
    toy.record_or_check(path, {"complete": True})
    assert path.read_bytes() == before
    with pytest.raises(ContractMismatch, match="existing receipt"):
        toy.record_or_check(path, {"complete": False})
