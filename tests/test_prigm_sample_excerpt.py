"""CPU-only synthetic provenance and excerpt contract checks; no generation runs."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from analysis import export_prigm_sample_excerpt as ex
from prigm.artifacts import id_hash, load_arrays, STATE_CONVENTION, NOISE_CONVENTION
from prigm.recovery import ContractMismatch, PublicationError, sha256_file
from prigm.run_score import decode_ensemble, hidden_primary_std
from prigm.scoring import energy_score


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def read_json(path):
    return json.loads(path.read_text())


def pin(root, relative, value):
    path = root / relative
    record = read_json(path)
    record.update(value)
    write_json(path, record)


def repin_real(paper, name):
    manifest = paper / "evidence/real/MANIFEST.json"
    data = read_json(manifest)
    data["output_sha256"][name] = sha256_file(manifest.parent / name)
    write_json(manifest, data)


def repin_ensemble(paper, ensembles, name="D_s7-g9560"):
    gen = ensembles / name / "gen_00000000"
    commit = read_json(gen / "COMMIT.json")
    commit["files"]["ensemble.npz"] = sha256_file(gen / "ensemble.npz")
    write_json(gen / "COMMIT.json", commit)
    manifest = paper / "evidence/real/MANIFEST.json"
    data = read_json(manifest)
    for filename in ("COMMIT.json", "ensemble.npz"):
        data["input_sha256"][f"artifacts/test/{name}/ensemble/gen_00000000/{filename}"] = sha256_file(gen / filename)
    write_json(manifest, data)


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    base = tmp_path_factory.mktemp("sample-excerpt-fixture")
    paper, ensembles = base / "paper", base / "ensembles"
    obs = paper / "evidence/observations/test.npz"
    obs.parent.mkdir(parents=True)
    ids = [f"test-window-{i:03d}" for i in range(480)]
    mean = np.array([1., 2., 3., 4., 5., 6.])
    scale = np.array([2., 3., 4., 5., 6., 7.])
    z = np.broadcast_to(mean, (480, 48, 6)).copy()
    z += np.arange(480)[:, None, None] / 1000
    np.savez_compressed(obs, z=z, ids=np.asarray(ids), split="TEST", mean=mean,
                        scale=scale, b_ref_sq=1., n_bins=100)
    _, _, nrm, identity = load_arrays(obs, "TEST")
    comparison = {"ids": ids, "arrays": identity, "protocol_sha256": ex.PROTOCOL, "selected": {}}
    scores = {"ids": ids, "protocol_sha256": ex.PROTOCOL, "data": {}, "row_ids": {}}
    real = {"input_sha256": {}, "output_sha256": {}}
    deadline = {"files": {"gaussian/FIT.npz": "fit-payload"}}
    for index, method in enumerate(ex.METHODS):
        neural = method in ex.STEPS
        channels = 5 if method == "G5" else 6
        ens = np.zeros((480, 39, channels, 48), dtype=np.float32 if neural else np.float64)
        ens[..., 18:30] = (np.arange(39)[None, :, None, None] / 100 + index / 10)
        meta = {"schema_version": 2, "generation": 0, "n_generated": channels,
                "arrays": identity, "ids_sha256": id_hash(ids),
                "chunks": {"/private/unpublished/path": "do-not-export"}}
        if neural:
            step = ex.STEPS[method]
            name = f"{method}_s7-g{step}"
            root = ensembles / name
            public = f"artifacts/test/{name}/ensemble"
            selected = {"generation": step, "checkpoint_sha256": f"checkpoint-{method}"}
            comparison["selected"][f"{method}_s7"] = {"raw": selected,
                "corrected": {"generation": 29636, "checkpoint_sha256": "wrong-final"}}
            meta.update(arm=method, seed=7, generation_updates=step,
                        checkpoint_sha256=selected["checkpoint_sha256"], protocol_sha256=ex.PROTOCOL,
                        sampler={"eval_seed": 7, "n_draws": 39, "n_eval": 50},
                        state_convention=STATE_CONVENTION, noise_convention=NOISE_CONVENTION)
            pins = real["input_sha256"]
        else:
            root = paper / "evidence/deadline" / ex.GAUSSIAN
            public = ex.GAUSSIAN
            meta.update(procedure="gaussian", protocol_sha256=ex.GAUSSIAN_PROTOCOL,
                        source_protocol_sha256=ex.PROTOCOL, fit_select_sha256=ex.FIT_SELECT,
                        control_fit_sha256="fit-payload", noise={"seed": 13, "n_draws": 39},
                        control={"family": "full_flank_linear_gaussian", "jitter": 1e-6,
                                 "joint_gaussian_channels": 5, "lambda": .1, "rho": 0.},
                        state_convention="physical_control_standardized_float64")
            pins = deadline["files"]
        gen = root / "gen_00000000"
        gen.mkdir(parents=True)
        np.savez_compressed(gen / "ensemble.npz", ensemble=ens, ids=np.asarray(ids))
        meta["files"] = {"ensemble.npz": sha256_file(gen / "ensemble.npz")}
        write_json(gen / "COMMIT.json", meta)
        for filename in ("COMMIT.json", "ensemble.npz"):
            pins[f"{public}/gen_00000000/{filename}"] = sha256_file(gen / filename)
        # Only the two fixed cases are scored by the exporter, all score entries are validated.
        values = np.zeros(480)
        for i in ex.INDICES:
            values[i] = energy_score(hidden_primary_std(decode_ensemble(ens[i], nrm), nrm),
                                     hidden_primary_std(z[i, 18:30][None], nrm)[0])
        if neural:
            scores["data"][f"{method}_raw"] = np.tile(values, (3, 1)).tolist()
            scores["row_ids"][f"{method}_raw"] = ["7", "13", "42"]
        else:
            gaussian_scores = {"ids": ids, "arrays": identity, "n": 480, "n_draws": 39,
                "extension_protocol_sha256": ex.GAUSSIAN_PROTOCOL, "source_protocol_sha256": ex.PROTOCOL,
                "fit_select_sha256": ex.FIT_SELECT, "fit_payload_sha256": "fit-payload",
                "candidates": {"lambda_0.1_rho_0": {"raw": {"per_window": [
                    {"primary_es": float(v)} for v in values]}}}}
    for name, data in (("comparison.json", comparison), ("scores.json", scores)):
        path = paper / "evidence/real" / name
        write_json(path, data)
        real["output_sha256"][name] = sha256_file(path)
    path = paper / "evidence/deadline/gaussian/test/SCORES.json"
    write_json(path, gaussian_scores)
    deadline["files"]["gaussian/test/SCORES.json"] = sha256_file(path)
    write_json(paper / "evidence/real/MANIFEST.json", real)
    write_json(paper / "evidence/deadline/MANIFEST.json", deadline)
    return base


@pytest.fixture
def inputs(frozen, tmp_path, monkeypatch):
    shutil.copytree(frozen, tmp_path / "inputs")
    paper, ensembles = tmp_path / "inputs/paper", tmp_path / "inputs/ensembles"
    monkeypatch.setattr(ex, "OBS_SHA256", sha256_file(paper / "evidence/observations/test.npz"))
    return paper, ensembles, tmp_path / "excerpt"


def test_contract_truth_draws_methods_and_receipt(inputs):
    paper, ensembles, out = inputs
    manifest = ex.export_excerpt(*inputs)
    assert {p.name for p in out.iterdir()} == {"sample-excerpt.npz", "MANIFEST.json"}
    assert manifest == read_json(out / "MANIFEST.json")
    assert manifest["files"]["sample-excerpt.npz"] == sha256_file(out / "sample-excerpt.npz")
    assert manifest["selection"]["zero_based_indices"] == [120, 360]
    assert manifest["selection"]["after_test_exposure"] is True
    assert manifest["method_order"] == list(ex.METHODS)
    assert len(manifest["per_case_primary_energy"]) == 4
    assert "private" not in (out / "MANIFEST.json").read_text()
    assert "chunks" not in (out / "MANIFEST.json").read_text()
    assert "analysis/export_prigm_sample_excerpt.py" in manifest["source_sha256"]
    with np.load(out / "sample-excerpt.npz", allow_pickle=False) as blob:
        assert set(blob.files) == {"truth", "samples", "ids", "methods"}
        assert blob["truth"].shape == (2, 48, 6)
        assert blob["samples"].shape == (4, 2, 39, 12, 5)
        assert blob["truth"].dtype == blob["samples"].dtype == np.float64
        assert blob["ids"].dtype.kind == blob["methods"].dtype.kind == "U"
        assert blob["methods"].tolist() == list(ex.METHODS)
        truth, ids, nrm, _ = load_arrays(paper / "evidence/observations/test.npz", "TEST")
        np.testing.assert_array_equal(blob["truth"], truth[[120, 360]])
        assert np.all(blob["truth"][:, :18] != 0)  # Stored NN flanks are zero, truth is not.
        assert blob["ids"].tolist() == [ids[120], ids[360]]
        for m, method in enumerate(ex.METHODS):
            root = (ensembles / f"{method}_s7-g{ex.STEPS[method]}" if method in ex.STEPS
                    else paper / "evidence/deadline" / ex.GAUSSIAN)
            ens, _, _ = ex.read_ensemble(root)
            for j, i in enumerate(ex.INDICES):
                np.testing.assert_array_equal(blob["samples"][m, j], decode_ensemble(ens[i], nrm)[..., :5])
                assert np.any(blob["samples"][m, j, 38] != blob["samples"][m, j, 0])
                rec = manifest["per_case_primary_energy"][method][j]
                assert abs(rec["recomputed"] - rec["published"]) <= 1e-12


@pytest.mark.parametrize("filename", ["COMMIT.json", "ensemble.npz"])
def test_hash_mismatch(inputs, filename):
    paper, ensembles, out = inputs
    with (ensembles / "D_s7-g9560/gen_00000000" / filename).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ContractMismatch, match="hash mismatch"):
        ex.export_excerpt(*inputs)
    assert not out.exists()


@pytest.mark.parametrize("field,value", [
    ("generation_updates", 29636), ("checkpoint_sha256", "wrong-final"),
    ("arm", "O-S"), ("seed", 13), ("protocol_sha256", "wrong"),
    ("sampler", {"eval_seed": 7, "n_draws": 39, "n_eval": 100}),
])
def test_reject_wrong_raw_lineage(inputs, field, value):
    paper, ensembles, out = inputs
    pin(ensembles, "D_s7-g9560/gen_00000000/COMMIT.json", {field: value})
    repin_ensemble(paper, ensembles)
    with pytest.raises(ContractMismatch, match="raw-selected"):
        ex.export_excerpt(*inputs)
    assert not out.exists()


def test_normalization_identity(inputs):
    paper, ensembles, out = inputs
    path = ensembles / "D_s7-g9560/gen_00000000/COMMIT.json"
    meta = read_json(path)
    meta["arrays"]["normalization_sha256"] = "wrong"
    write_json(path, meta)
    repin_ensemble(paper, ensembles)
    with pytest.raises(ContractMismatch, match="identities differ"):
        ex.export_excerpt(*inputs)
    assert not out.exists()


@pytest.mark.parametrize("change", ["ids", "shape", "nonfinite"])
def test_full_ensemble_validation(inputs, change):
    paper, ensembles, out = inputs
    gen = ensembles / "D_s7-g9560/gen_00000000"
    with np.load(gen / "ensemble.npz", allow_pickle=False) as blob:
        ens, ids = blob["ensemble"], blob["ids"]
    if change == "ids":
        ids[[0, 1]] = ids[[1, 0]]  # Neither selected case changes.
        pin(gen, "COMMIT.json", {"ids_sha256": id_hash(ids.tolist())})
    elif change == "shape":
        ens = ens[:, :38]
    else:
        ens[0, 0, 0, 0] = np.nan  # Outside the excerpt, still rejected.
    np.savez_compressed(gen / "ensemble.npz", ensemble=ens, ids=ids)
    repin_ensemble(paper, ensembles)
    with pytest.raises((ContractMismatch, PublicationError)):
        ex.export_excerpt(*inputs)
    assert not out.exists()


@pytest.mark.parametrize("change", ["ids", "rows", "energy", "nonfinite"])
def test_published_score_validation(inputs, change):
    paper, _, out = inputs
    path = paper / "evidence/real/scores.json"
    scores = read_json(path)
    if change == "ids":
        scores["ids"][0], scores["ids"][1] = scores["ids"][1], scores["ids"][0]
    elif change == "rows":
        scores["row_ids"]["D_raw"] = ["13", "7", "42"]
    elif change == "energy":
        scores["data"]["D_raw"][0][120] += 1e-8
    else:
        scores["data"]["D_raw"][2][0] = float("nan")
    write_json(path, scores)
    repin_real(paper, "scores.json")
    with pytest.raises(ContractMismatch):
        ex.export_excerpt(*inputs)
    assert not out.exists()


def test_nonfinite_observations(inputs, monkeypatch):
    paper, _, out = inputs
    path = paper / "evidence/observations/test.npz"
    with np.load(path, allow_pickle=False) as blob:
        data = {key: blob[key] for key in blob.files}
    data["z"][0, 0, 0] = np.inf
    np.savez_compressed(path, **data)
    monkeypatch.setattr(ex, "OBS_SHA256", sha256_file(path))
    with pytest.raises(ContractMismatch, match="nonfinite"):
        ex.export_excerpt(*inputs)
    assert not out.exists()


def test_no_overwrite(inputs):
    out = inputs[2]
    out.mkdir()
    (out / "sentinel").write_text("untouched")
    with pytest.raises(FileExistsError):
        ex.export_excerpt(*inputs)
    assert [p.name for p in out.iterdir()] == ["sentinel"]
    assert (out / "sentinel").read_text() == "untouched"


@pytest.mark.parametrize("target", ["source", "input"])
def test_rehash_before_publication(inputs, monkeypatch, target):
    original = ex.sha256_file
    seen = {}
    selected = Path(ex.__file__).resolve() if target == "source" else inputs[0] / "evidence/real/comparison.json"

    def changed(path):
        path = Path(path)
        seen[path] = seen.get(path, 0) + 1
        if path == selected and seen[path] > 1:
            return "changed-during-export"
        return original(path)

    monkeypatch.setattr(ex, "sha256_file", changed)
    with pytest.raises(ContractMismatch, match="changed before publication"):
        ex.export_excerpt(*inputs)
    assert not inputs[2].exists()


def test_cli(inputs, capsys):
    assert ex.main(["--paper-dir", str(inputs[0]), "--ensemble-dir", str(inputs[1]),
                    "--output-dir", str(inputs[2])]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["manifest_sha256"] == sha256_file(inputs[2] / "MANIFEST.json")
