"""Check preparation of public toy-study evidence without training or denoising.

The complete fixture contains nine synthetic checkpoints with the expected
state structure and 36 saved ensemble chunks per fit. Its learned-model
numbers are invented and reduced into the required reports. Source, array,
checkpoint, completion-record and ensemble validators still run.

A separate limited check uses the actual public array and oracle to
reconstruct the existing direct draws from the conditional distribution.
It does not run the learned denoiser, create new scientific results, use
the network or execute on a GPU.
"""

import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from analysis import export_prigm_toy_evidence as export
from experiments import run_prigm_toy_repair as toy
from experiments import verify_prigm_paper as gate
from prigm.artifacts import id_hash, json_bytes, merge_ensembles, publish_ensemble
from prigm.checkpoint import save_torch_payload
from prigm.recovery import ContractMismatch, PublicationError, sha256_file
from prigm.run_secondaries import summarize_diagnostics
from tests.test_prigm_toy_repair import write_checkpoint

PUBLIC = Path(__file__).resolve().parents[1] / "paper/workshops/prigm/evidence/toy"
ORACLE_SHA = "fee403a2d4f823181ea08a7d726193d2ffcc50e5c85e4ded5810426877397c5f"
ARRAY_SHA = "dbf80b22d72ddeaf8b609455be797cc7862edf9c585d5b76f7abdc3d6f4dcaf7"
FAILURES = (ContractMismatch, PublicationError, FileNotFoundError)


def save(path, value):
    path.write_bytes(json_bytes(value))
    return sha256_file(path)


def read(path):
    return json.loads(path.read_bytes())


def current_oracle_fixture():
    """Synthetic current-source metadata with the published numerical values."""
    record = read(PUBLIC / "oracle-stationary-01.json")
    record["source_hashes"] = {
        name: sha256_file(Path(export.oracle_module.__file__).parent / name)
        for name in record["source_hashes"]
    }
    return record


@pytest.fixture(autouse=True)
def no_experiments(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("curation must never train, denoise, evaluate or launch a process")

    from prigm import sampler
    for name in ("train", "evaluate", "launch"):
        monkeypatch.setattr(toy, name, forbidden)
    for name in ("main", "ddim_exact", "two_step_exact"):
        monkeypatch.setattr(export.oracle_module, name, forbidden)
    monkeypatch.setattr(sampler, "generate_ensemble", forbidden)
    monkeypatch.setattr(sampler, "ddim_generate", forbidden)
    monkeypatch.setattr(toy.ConditionalUNet, "forward", forbidden)


def synthetic_scores(ids, fit_index, corrected):
    """Every scalar/matrix is finite; retain a deliberately negative variogram."""
    records = []
    for i in range(len(ids)):
        base = float(1 + fit_index + corrected + i / len(ids))
        record = {key: base for key in export.SCALAR_METRICS}
        record.update({key: np.full((12, 5), base).tolist() for key in export.ARRAY_METRICS})
        record["variogram"] = -0.25
        record["cov90_by_position_channel"] = np.full((12, 5), float(i % 2)).tolist()
        for scalar, matrix in (("crps5_mean", "crps_by_position_channel"),
                               ("cov90_mean", "cov90_by_position_channel"),
                               ("width90_mean", "width90_by_position_channel")):
            record[scalar] = float(np.asarray(record[matrix]).mean())
        records.append(record)
    values = [row["es5"] for row in records]
    return values, records


@pytest.fixture(scope="module")
def complete_study(tmp_path_factory):
    root = tmp_path_factory.mktemp("toy-curation") / "private-study-name"
    root.mkdir()
    # Cache the deterministic source construction only; all checkpoint, source,
    # array, receipt and ensemble validators still execute unmocked.
    training = toy.training_identity()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(toy, "training_identity", lambda: copy.deepcopy(training))
        toy.publish_protocol(root)
        protocol = read(root / "PROTOCOL.json")
        protocol["private_execution_path"] = "/home/fixture-owner/private-location"
        digest = save(root / "PROTOCOL.json", protocol)
        save(root / "PROTOCOL_RECEIPT.json", {"protocol_sha256": digest, "published_before": ["train", "evaluate"]})
        oracle = root.parent / "private-oracle.json"
        save(oracle, current_oracle_fixture())
        requests = root / "artifacts/training"
        requests.mkdir()
        ids = [f"TOY:{i}" for i in range(288)]
        trained, evaluated = {}, {}
        for index, fit in enumerate(protocol["recipe"]["fits"]):
            name = fit["name"]
            save(requests / f"{name}-request.json", toy.fit_request(root, fit, digest))
            write_checkpoint(root, protocol, fit, mutate=lambda c: c["settings"]["environment"].update(
                private_path="/home/checkpoint-owner/not-public", hostname="private-host"))
            row = toy.completed_fit(root, fit, protocol, digest)
            save(requests / f"{name}-completed.json", row)
            trained[name] = row
            parent = root / f"artifacts/toy/{name}-g5000"
            parent.mkdir(parents=True)
            save(parent / "REQUEST.json", row)
            chunks = []
            for offset in range(0, 288, 8):
                chunk = parent / f"chunk-{offset:04d}"
                meta = {"protocol_sha256": digest, "arrays": protocol["arrays"]["TOY"],
                        "checkpoint_sha256": row["checkpoint_sha256"], "generation_updates": 5000,
                        "arm": fit["arm"], "seed": 7, "n_generated": 6,
                        "sampler": protocol["samplers"]["TOY"], "state_convention": toy.STATE_CONVENTION,
                        "noise_convention": toy.NOISE_CONVENTION, "offset": offset, "count": 8,
                        "inference_source_sha256": protocol["inference_source_hashes"]["prigm/run_eval.py"]}
                samples = np.zeros((8, 39, 6, 48), np.float32)
                samples[..., 18:30] = index + offset / 288
                publish_ensemble(chunk, samples, ids[offset:offset + 8], meta)
                chunks.append(chunk)
            merge_ensembles(chunks, parent / "ensemble", ids)
            commit = sha256_file(parent / "ensemble/gen_00000000/COMMIT.json")
            outputs = {}
            for corrected in (False, True):
                kind = "corrected" if corrected else "raw"
                scores, records = synthetic_scores(ids, index, corrected)
                common = {"schema_version": 2, "protocol_sha256": digest, "ids": ids, "n": 288,
                          "arrays": protocol["arrays"]["TOY"], "checkpoint_sha256": row["checkpoint_sha256"],
                          "ensemble_commit_sha256": commit, "procedure": "diffusion",
                          "control_fit_sha256": None, "months": [0] * 288}
                for secondary, payload in ((False, {"corrected": corrected, "scores": scores,
                        "mean": float(np.mean(scores)), "month_balanced": float(np.mean(scores))}),
                        (True, {"correct": corrected, "interval_protocol": export.INTERVAL_PROTOCOL,
                                "per_window": records, **summarize_diagnostics(records, [0] * 288)})):
                    key = f"{'secondaries' if secondary else 'score'}-{kind}"
                    path = parent / f"{key}.json"
                    outputs[key] = {"path": path.relative_to(root).as_posix(), "sha256": save(path, {**common, **payload})}
            evaluated[name] = {**row, "ensemble_commit_sha256": commit, "outputs": outputs}
            save(parent / "COMPLETED.json", evaluated[name])
        trained_hash = save(root / "TRAINED.json", {"protocol_sha256": digest, "completed_runs": trained})
        save(root / "EVALUATED.json", {"protocol_sha256": digest, "trained_manifest_sha256": trained_hash,
                                      "completed_runs": evaluated})
        yield SimpleNamespace(root=root, protocol=protocol, digest=digest, oracle=oracle,
                              oracle_hash=sha256_file(oracle))


@pytest.fixture
def study(complete_study, tmp_path):
    """Restore only files explicitly mutated by each test; do not copy 234 MB per case."""
    state = SimpleNamespace(**vars(complete_study), output=tmp_path / "public")
    backups = {}

    def write(path, payload):
        path = Path(path)
        if not path.is_absolute():
            path = state.root / path
        if path not in backups:
            backups[path] = path.read_bytes()
        if payload is None:
            path.unlink()
        else:
            path.write_bytes(payload)
        return path

    def change(path, mutate):
        absolute = Path(path) if Path(path).is_absolute() else state.root / path
        value = read(absolute)
        mutate(value)
        write(absolute, json_bytes(value))
        return value

    state.write, state.change = write, change
    yield state
    for path, payload in backups.items():
        if path.is_symlink():
            path.unlink()
        path.write_bytes(payload)


def run(study):
    return export.export_study(study.root, study.digest, study.oracle, study.oracle_hash, study.output)


def parent(study, index=0):
    return f"artifacts/toy/{study.protocol['recipe']['fits'][index]['name']}-g5000"


def rebind_score(study, key, index=0):
    name = study.protocol["recipe"]["fits"][index]["name"]
    directory = parent(study, index)
    completion = study.change(f"{directory}/COMPLETED.json", lambda row:
        row["outputs"][key].update(sha256=sha256_file(study.root / f"{directory}/{key}.json")))
    study.change("EVALUATED.json", lambda row: row["completed_runs"].update({name: completion}))


def change_protocol(study, mutate):
    study.change("PROTOCOL.json", mutate)
    study.digest = sha256_file(study.root / "PROTOCOL.json")
    study.change("PROTOCOL_RECEIPT.json", lambda row: row.update(protocol_sha256=study.digest))


def test_cli_complete_portable_layout_and_original_numbers(study, capsys):
    before = {p: sha256_file(p) for p in study.root.rglob("*") if p.is_file()}
    assert export.main(["--study-root", str(study.root), "--protocol-sha256", study.digest,
                        "--oracle", str(study.oracle), "--oracle-sha256", study.oracle_hash,
                        "--output-dir", str(study.output)]) == 0
    manifest = read(study.output / "MANIFEST.json")
    assert manifest["schema_version"] == 1
    assert manifest["exporter_source_sha256"] == sha256_file(Path(export.__file__))
    assert set(p.name for p in study.output.iterdir()) == {"MANIFEST.json", "learned.json", "reference-calibration.json"}
    for name, digest in manifest["output_sha256"].items():
        assert sha256_file(study.output / name) == digest
    for relative, digest in manifest["input_sha256"].items():
        assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
        assert sha256_file(study.root / relative) == digest
    assert len([p for p in manifest["input_sha256"] if "/chunk-" in p and p.endswith("COMMIT.json")]) == 9 * 36
    assert len([p for p in manifest["input_sha256"] if p.endswith("state.pt")]) == 9
    learned = read(study.output / "learned.json")
    assert learned["schema_version"] == 1 and learned["n"] == 288 and learned["n_draws"] == 39
    assert learned["ids"] == [f"TOY:{i}" for i in range(288)]
    assert learned["fit_order"] == [f["name"] for f in study.protocol["recipe"]["fits"]]
    assert len(learned["fits"]) == 9 and "D_c0p0_s7" in learned["fits"]
    assert "not original execution records" in learned["scope"]
    for index, name in enumerate(learned["fit_order"]):
        fit = learned["fits"][name]
        assert fit["config"]["arm"] == study.protocol["recipe"]["fits"][index]["arm"]
        assert "environment" not in fit["config"]["settings"]
        assert fit["generation_updates"] == 5000
        assert fit["checkpoint_sha256"] == sha256_file(study.root / f"runs/{name}/gen_00005000/state.pt")
        for kind in ("raw", "corrected"):
            primary = read(study.root / f"{parent(study, index)}/score-{kind}.json")
            secondary = read(study.root / f"{parent(study, index)}/secondaries-{kind}.json")
            result = fit[kind]
            assert result["scores"] == primary["scores"]
            assert result["mean_energy"] == primary["mean"]
            assert result["month_balanced_energy"] == primary["month_balanced"]
            assert result["per_window"] == secondary["per_window"]
            assert result["secondary"] == {k: secondary[k] for k in export.SCALAR_METRICS + export.ARRAY_METRICS}
            assert result["secondary"]["variogram"]["mean"] == -0.25
    calibration = read(study.output / "reference-calibration.json")
    assert calibration["ids"] == learned["ids"] and calibration["arrays"] == learned["arrays"]
    assert calibration["source_oracle_sha256"] == study.oracle_hash
    assert calibration["source_protocol_sha256"] == study.digest
    for text in [p.read_text() for p in study.output.iterdir()] + [capsys.readouterr().out]:
        for forbidden in (str(study.root), "/home/", "fixture-owner", "checkpoint-owner", "private-host", "private-study-name"):
            assert forbidden not in text
    assert {p: sha256_file(p) for p in before} == before


@pytest.mark.parametrize("relative", [
    "TRAINED.json", "EVALUATED.json", "artifacts/training/D_c0p0_s7-completed.json",
    "artifacts/toy/D_c0p0_s7-g5000/COMPLETED.json",
    "runs/D_c0p0_s7/gen_00005000/state.pt", "runs/D_c0p0_s7/source_snapshot.zip",
    "artifacts/toy/D_c0p0_s7-g5000/chunk-0280/gen_00000000/COMMIT.json",
    "artifacts/toy/D_c0p0_s7-g5000/chunk-0280/gen_00000000/ensemble.npz",
    "artifacts/toy/O-S_c0p005_s7-g5000/score-corrected.json",
])
def test_every_completion_and_last_chunk_and_last_fit_are_required(study, relative):
    study.write(relative, None)
    with pytest.raises(FAILURES):
        run(study)
    assert not study.output.exists()


@pytest.mark.parametrize("relative", [
    "runs/D_c0p0_s7/gen_00005000/state.pt", "runs/D_c0p0_s7/source_snapshot.zip",
    "artifacts/toy/D_c0p0_s7-g5000/chunk-0280/gen_00000000/ensemble.npz",
    "artifacts/toy/D_c0p0_s7-g5000/ensemble/gen_00000000/ensemble.npz",
    "artifacts/toy/D_c0p0_s7-g5000/score-corrected.json", "artifacts/toy/D_c0p0_s7-g5000/secondaries-raw.json",
])
def test_mutated_payload_hashes_are_not_adopted(study, relative):
    path = study.root / relative
    study.write(relative, path.read_bytes() + b" ")
    with pytest.raises(FAILURES):
        run(study)
    assert not study.output.exists()


@pytest.mark.parametrize("receipt", ["TRAINED.json", "EVALUATED.json", "artifacts/toy/D_c0p0_s7-g5000/COMPLETED.json",
                                     "artifacts/training/D_c0p0_s7-request.json"])
def test_completion_generation_flags_are_type_strict(study, receipt):
    def mutate(row):
        target = row["completed_runs"]["D_c0p0_s7"] if "completed_runs" in row else row
        target["generation_updates"] = 5000.0
    study.change(receipt, mutate)
    with pytest.raises(ContractMismatch, match="type differs"):
        run(study)


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(boundary_law="iid"),
    lambda p: p.update(state_convention="noisy_flanks"),
    lambda p: p.update(noise_convention="old_channel_noise"),
    lambda p: p.update(training_source_hashes={}),
    lambda p: p.update(inference_source_hashes={}),
    lambda p: p["recipe"].update(seed=13),
    lambda p: p["recipe"].update(fits=p["recipe"]["fits"][:-1]),
    lambda p: p["samplers"]["TOY"].update(n_draws=39.0),
    lambda p: p["training_settings"]["precision"].update(deterministic_algorithms=1),
])
def test_recipe_source_law_and_type_contracts(study, mutate):
    change_protocol(study, mutate)
    with pytest.raises(FAILURES):
        run(study)
    assert not study.output.exists()


@pytest.mark.parametrize("key,mutate,match", [
    ("score-raw", lambda r: r.update(mean=r["mean"] + 1e-10), "primary mean"),
    ("score-raw", lambda r: r.update(month_balanced=r["mean"] + 1e-10), "primary month"),
    ("score-raw", lambda r: r.update(corrected=0), "type differs"),
    ("secondaries-corrected", lambda r: r.update(correct=1), "type differs"),
    ("score-corrected", lambda r: r.update(corrected=False), "provenance"),
    ("score-raw", lambda r: r["ids"].reverse(), "provenance"),
    ("secondaries-raw", lambda r: r["r_obs_mean"].update(mean=r["r_obs_mean"]["mean"] + 1e-10), "secondary r_obs_mean"),
    ("secondaries-raw", lambda r: r["per_window"][-1].update(r_obs_mean=float("inf")), "nonfinite"),
    ("secondaries-raw", lambda r: r["per_window"][-1].update(es5=r["per_window"][-1]["es5"] + 1e-10), "primary versus es5"),
    ("secondaries-raw", lambda r: r["per_window"][-1]["width90_by_position_channel"][0].__setitem__(
        0, r["per_window"][-1]["width90_by_position_channel"][0][0] + 1e-8), "coordinate mean"),
    ("secondaries-raw", lambda r: r["per_window"][-1].pop("es6"), None),
    ("score-raw", lambda r: r["months"].__setitem__(-1, False), "type differs"),
])
def test_strict_score_reductions_and_identities_even_when_receipts_are_rebound(study, key, mutate, match):
    # Use nonstandard JSON only for the explicit infinity corruption case.
    path = study.root / f"{parent(study)}/{key}.json"
    value = read(path)
    mutate(value)
    study.write(path, json.dumps(value).encode())
    rebind_score(study, key)
    with pytest.raises((ContractMismatch, KeyError), match=match):
        run(study)
    assert not study.output.exists()


def rewrite_ensemble(study, relative, mutate):
    path = study.root / f"{relative}/gen_00000000/ensemble.npz"
    with np.load(path, allow_pickle=False) as blob:
        values = {key: blob[key] for key in blob.files}
    mutate(values)
    buffer = io.BytesIO()
    np.savez(buffer, **values)
    study.write(path, buffer.getvalue())
    study.change(f"{relative}/gen_00000000/COMMIT.json", lambda row:
                 row["files"].update({"ensemble.npz": sha256_file(path)}))


def rebind_chunk(study, relative):
    study.change(f"{parent(study)}/ensemble/gen_00000000/COMMIT.json", lambda row:
        row["chunks"].update({str(study.root / relative): sha256_file(study.root / f"{relative}/gen_00000000/COMMIT.json")}))


def test_merged_payload_must_equal_all_36_actual_chunks(study):
    relative = f"{parent(study)}/chunk-0280"
    rewrite_ensemble(study, relative, lambda v: v["ensemble"].__setitem__((-1, -1, -1, 29), 123.))
    rebind_chunk(study, relative)
    with pytest.raises(ContractMismatch, match="36-chunk union"):
        run(study)


def test_union_comparison_preserves_signed_zero_bytes(study):
    relative = f"{parent(study)}/chunk-0000"
    rewrite_ensemble(study, relative, lambda v: v["ensemble"].__setitem__((0, 0, 0, 18), -0.))
    rebind_chunk(study, relative)
    with pytest.raises(ContractMismatch, match="36-chunk union"):
        run(study)


def test_actual_checkpoint_payload_contract_is_type_checked(study):
    relative = "runs/D_c0p0_s7/gen_00005000/state.pt"
    path = study.root / relative
    payload = export.load_torch_payload(path.read_bytes(), map_location="cpu")
    payload["contract"]["settings"]["precision"]["deterministic_algorithms"] = 1
    study.write(path, save_torch_payload(payload))
    study.change("runs/D_c0p0_s7/gen_00005000/COMMIT.json", lambda row:
                 row["files"].update({"state.pt": sha256_file(path)}))
    with pytest.raises(ContractMismatch, match="checkpoint payload contract.*type differs"):
        run(study)


@pytest.mark.parametrize("field,value", [("offset", 280.0), ("count", 8.0), ("seed", 7.0),
                                         ("inference_source_sha256", "0" * 64)])
def test_chunk_identity_types_and_source(study, field, value):
    relative = f"{parent(study)}/chunk-0280"
    study.change(f"{relative}/gen_00000000/COMMIT.json", lambda row: row.update({field: value}))
    rebind_chunk(study, relative)
    with pytest.raises(ContractMismatch):
        run(study)


def test_reordered_chunk_ids_fail_even_with_rebound_id_and_payload_hash(study):
    relative = f"{parent(study)}/chunk-0280"
    rewrite_ensemble(study, relative, lambda v: v.update(ids=v["ids"][::-1]))
    study.change(f"{relative}/gen_00000000/COMMIT.json", lambda row:
                 row.update(ids_sha256=id_hash([f"TOY:{i}" for i in range(287, 279, -1)])))
    rebind_chunk(study, relative)
    with pytest.raises(ContractMismatch, match="exact IDs"):
        run(study)


def test_hidden_only_ensemble_storage_is_not_confused_with_clean_context(study):
    relative = f"{parent(study)}/chunk-0280"
    rewrite_ensemble(study, relative, lambda v: v["ensemble"].__setitem__((-1, 0, 0, 0), 1.))
    rewrite_ensemble(study, f"{parent(study)}/ensemble", lambda v: v["ensemble"].__setitem__((-1, 0, 0, 0), 1.))
    rebind_chunk(study, relative)
    with pytest.raises(ContractMismatch, match="nonzero non-hidden"):
        run(study)


def test_actual_public_reference_same_stream_matches_all_288_scores(monkeypatch):
    assert sha256_file(PUBLIC / "toy.npz") == ARRAY_SHA
    assert sha256_file(PUBLIC / "oracle-stationary-01.json") == ORACLE_SHA
    original = read(PUBLIC / "oracle-stationary-01.json")
    calls, streams = [], []
    posterior = export.oracle_module.posterior_hidden
    sample = export.sample_mixture_paths

    def checked_posterior(alpha, p, a, b, nrm, **kwargs):
        assert p is None and kwargs == {"conditioning": "pooled", "boundary_law": "stationary"}
        calls.append((alpha, a, b))
        return posterior(alpha, p, a, b, nrm, **kwargs)

    def checked_sample(post, rng, n):
        assert n == 39
        streams.append(rng)
        return sample(post, rng, n)

    monkeypatch.setattr(export.oracle_module, "posterior_hidden", checked_posterior)
    monkeypatch.setattr(export, "sample_mixture_paths", checked_sample)
    result = gate.replay_frozen_toy_reference(PUBLIC / "toy.npz", original, ORACLE_SHA)
    assert result["oracle_metadata"]["source_hashes"] == original["source_hashes"]
    assert sha256_file(PUBLIC / "oracle-stationary-01.json") == ORACLE_SHA
    assert len(calls) == len(streams) == 288 and all(rng is streams[0] for rng in streams)
    assert result["scores"] == original["methods"]["exact_reference"]["scores"]
    assert result["mean_energy"] == original["methods"]["exact_reference"]["mean_energy"]
    assert result["secondary"]["cov90_mean"]["mean"] == 1.0
    assert result["secondary"]["cov50_mean"]["mean"] == pytest.approx(0.9749421296296297, abs=1e-12)
    assert result["secondary"]["width90_mean"]["mean"] == pytest.approx(1.4580513587358999, abs=1e-12)
    assert result["secondary"]["width50_mean"]["mean"] == pytest.approx(1.348480688839868, abs=1e-12)
    assert result["secondary"]["crps5_mean"]["mean"] == pytest.approx(0.34613405118380225, abs=1e-12)
    for key, reductions in result["secondary"].items():
        actual = np.asarray([r[key] for r in result["per_window"]]).mean(axis=0)
        np.testing.assert_allclose(reductions["mean"], actual, atol=1e-12, rtol=0)
        np.testing.assert_allclose(reductions["month_balanced"], actual, atol=1e-12, rtol=0)
    assert "atomic conditional law" in result["scope"] and "not a Bayes floor" in result["scope"]


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(reference_seed=r["reference_seed"] + 1),
    lambda r: r.update(reference_seed=float(r["reference_seed"])),
    lambda r: r.update(n_draws=38),
    lambda r: r.update(n=287),
    lambda r: r.update(conditioning="regime-known"),
    lambda r: r.update(boundary_law="iid"),
    lambda r: r["p_prior"].update({"0.1": 0.5}),
    lambda r: r["source_hashes"].update({"toy.py": "0" * 64}),
    lambda r: r["contexts"].reverse(),
    lambda r: r["methods"]["exact_reference"]["scores"].reverse(),
    lambda r: r["methods"]["exact_reference"]["scores"].__setitem__(0, 99.),
    lambda r: r["methods"]["exact_reference"].update(mean_energy=99.),
])
def test_oracle_metadata_order_and_frozen_values_are_not_replaceable(mutate):
    original = read(PUBLIC / "oracle-stationary-01.json")
    mutate(original)
    with pytest.raises((ContractMismatch, ValueError)):
        gate.replay_frozen_toy_reference(PUBLIC / "toy.npz", original, ORACLE_SHA)


@pytest.mark.parametrize("change", ["ids", "array_order", "law"])
def test_reference_array_order_and_law_checked_even_with_rebound_array_sha(tmp_path, change):
    with np.load(PUBLIC / "toy.npz", allow_pickle=False) as blob:
        values = {key: blob[key] for key in blob.files}
    if change == "ids":
        values["ids"] = values["ids"][::-1]
    elif change == "array_order":
        values["z"] = values["z"][::-1]
    else:
        values["boundary_law"] = np.array("iid")
    path = tmp_path / "toy.npz"
    np.savez(path, **values)
    original = read(PUBLIC / "oracle-stationary-01.json")
    original["toy_eval_sha256"] = sha256_file(path)
    with pytest.raises((ContractMismatch, ValueError)):
        gate.replay_frozen_toy_reference(path, original, ORACLE_SHA)


def test_bad_reference_cannot_publish_any_calibration(study):
    study.change(study.oracle, lambda r: r["methods"]["exact_reference"].update(mean_energy=99.))
    study.oracle_hash = sha256_file(study.oracle)
    with pytest.raises(ContractMismatch, match="oracle mean"):
        run(study)
    assert not study.output.exists()


@pytest.mark.parametrize("case", ["existing", "partial", "missing_parent", "dangling_symlink"])
def test_output_must_be_fresh_with_an_existing_parent(study, tmp_path, case):
    if case in ("existing", "partial"):
        study.output.mkdir()
        marker = study.output / ("MANIFEST.json" if case == "existing" else "learned.json")
        marker.write_bytes(b"do not replace")
    elif case == "missing_parent":
        study.output = tmp_path / "absent" / "output"
    else:
        study.output.symlink_to(tmp_path / "absent")
    with pytest.raises((FileExistsError, FileNotFoundError)):
        run(study)
    if case in ("existing", "partial"):
        assert marker.read_bytes() == b"do not replace"


@pytest.mark.parametrize("target", ["protocol", "oracle", "array", "checkpoint", "chunk", "score", "source", "exporter"])
def test_inputs_rehashed_after_outputs_before_manifest(study, monkeypatch, target):
    real_write = export._write
    target_paths = {"protocol": study.root / "PROTOCOL.json", "oracle": study.oracle,
                    "array": study.root / "artifacts/toy.npz",
                    "checkpoint": study.root / "runs/D_c0p0_s7/gen_00005000/state.pt",
                    "chunk": study.root / f"{parent(study)}/chunk-0280/gen_00000000/ensemble.npz",
                    "score": study.root / f"{parent(study)}/score-corrected.json"}

    def write(path, payload):
        real_write(path, payload)
        if path.name == "reference-calibration.json":
            if target == "source":
                monkeypatch.setattr(toy, "source_identity", lambda: {})
            elif target == "exporter":
                real_hash = export.sha256_file
                monkeypatch.setattr(export, "sha256_file", lambda p:
                    "0" * 64 if p == Path(export.__file__).resolve() else real_hash(p))
            else:
                changed = target_paths[target]
                study.write(changed, changed.read_bytes() + b" ")

    monkeypatch.setattr(export, "_write", write)
    with pytest.raises(FAILURES):
        run(study)
    assert (study.output / "learned.json").is_file()
    assert (study.output / "reference-calibration.json").is_file()
    assert not (study.output / "MANIFEST.json").exists()
    before = {p: p.read_bytes() for p in study.output.iterdir()}
    with pytest.raises(FileExistsError):
        run(study)
    assert {p: p.read_bytes() for p in study.output.iterdir()} == before


def test_partial_output_and_failed_manifest_publication_are_preserved(study, monkeypatch):
    real_write = export._write

    def fail(path, payload):
        if path.name == "reference-calibration.json":
            path.write_bytes(b"partial output")
            raise OSError("simulated write failure")
        real_write(path, payload)

    monkeypatch.setattr(export, "_write", fail)
    with pytest.raises(OSError, match="simulated"):
        run(study)
    assert (study.output / "learned.json").is_file()
    assert (study.output / "reference-calibration.json").read_bytes() == b"partial output"
    assert not (study.output / "MANIFEST.json").exists()


def test_manifest_is_atomic_and_published_last(study, monkeypatch):
    def fail_link(source, destination):
        assert Path(source).name == ".MANIFEST.json.partial"
        assert not Path(destination).exists()
        pending = read(Path(source))
        for name, digest in pending["output_sha256"].items():
            assert sha256_file(study.output / name) == digest
        raise OSError("manifest publication interrupted")

    monkeypatch.setattr(export.os, "link", fail_link)
    with pytest.raises(OSError, match="manifest publication"):
        run(study)
    assert (study.output / ".MANIFEST.json.partial").is_file()
    assert not (study.output / "MANIFEST.json").exists()


def test_symlink_input_refused_without_modifying_target(study, tmp_path):
    path = study.root / "runs/D_c0p0_s7/source_snapshot.zip"
    outside = tmp_path / "snapshot.zip"
    outside.write_bytes(path.read_bytes())
    study.write(path, None)
    path.symlink_to(outside)
    before = outside.read_bytes()
    with pytest.raises(ContractMismatch, match="redirected"):
        run(study)
    assert outside.read_bytes() == before


@pytest.mark.parametrize("text", ["/home/person/model.pt", "person", "../private.json", "C:\\Users\\person\\run", "ssh://host/run"])
def test_public_provenance_rejects_private_paths_and_detached_names(text):
    with pytest.raises(ContractMismatch, match="private"):
        export._safe({"provenance": text}, {"person"})


def test_exporter_must_still_match_its_imported_source(study, monkeypatch):
    monkeypatch.setattr(export, "IMPORTED_EXPORTER_SHA256", "0" * 64)
    with pytest.raises(ContractMismatch, match="exporter changed after import"):
        run(study)
    assert not study.output.exists()


def test_duplicate_json_keys_are_not_silently_adopted(study):
    relative = f"{parent(study)}/score-raw.json"
    path = study.root / relative
    study.write(path, path.read_bytes().replace(b'"corrected": false', b'"corrected": true, "corrected": false'))
    with pytest.raises(ContractMismatch, match="duplicate"):
        run(study)
