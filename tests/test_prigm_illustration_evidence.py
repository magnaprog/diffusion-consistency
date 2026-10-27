"""Focused acceptance/tampering tests on the small, committed illustration excerpt.

Copies use hardlinks for read-only large inputs. Every mutation replaces a file
atomically, so no test can modify the repository through a shared inode. Hash-pin
bypass is test-only; all public hash chains are refreshed rather than disabled.
"""
import io
import json
import os
import shutil
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from experiments import _prigm_illustrations as illustrations
from experiments import verify_prigm_paper as gate
from prigm.artifacts import load_arrays
from prigm.recovery import ContractMismatch


EXCERPT = "evidence/imputation/sample-excerpt.npz"
PROVENANCE = "evidence/imputation/MANIFEST.json"
FIXED = "figures/fixed_final/fig_fixed_final.manifest.json"
CASE = "figures/imputation/case_120/fig_imputation.manifest.json"
ERRORS = (ValueError, ContractMismatch)


@pytest.fixture(scope="module")
def observations():
    return load_arrays(gate.PAPER / "evidence/observations/test.npz", "TEST")


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    # Copy once onto the test filesystem; repository and tmpfs cannot hardlink.
    root = tmp_path_factory.mktemp("illustration-snapshot") / "paper"
    shutil.copytree(gate.PAPER, root, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    yield root
    shutil.rmtree(root)


@pytest.fixture
def paper(tmp_path, snapshot):
    root = tmp_path / "paper"
    shutil.copytree(snapshot, root, copy_function=os.link)
    yield root
    shutil.rmtree(root)


@pytest.fixture
def unpinned(monkeypatch):
    # Only the six independent pins are bypassed. No parsing, source, lineage,
    # file inventory, output hash or numeric comparison is monkeypatched.
    monkeypatch.setattr(illustrations, "_pin", lambda *args: None)


def replace(path, payload):
    temporary = path.with_name(path.name + ".test-replacement")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_json(path, obj):
    replace(path, (json.dumps(obj, allow_nan=False) + "\n").encode())


def mutate_json(root, name, keys, value):
    obj = gate.read_json(root / name)
    node = obj
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    write_json(root / name, obj)


def rebind(root):
    """Refresh real public dependencies, never replace independent pin checks."""
    real = gate.read_json(root / "evidence/real/MANIFEST.json")
    for name in real["output_sha256"]:
        real["output_sha256"][name] = gate.digest(root / "evidence/real" / name)
    write_json(root / "evidence/real/MANIFEST.json", real)
    deadline = gate.read_json(root / "evidence/deadline/MANIFEST.json")
    for name in deadline["files"]:
        deadline["files"][name] = gate.digest(root / "evidence/deadline" / name)
    write_json(root / "evidence/deadline/MANIFEST.json", deadline)
    record = gate.read_json(root / PROVENANCE)
    record["files"]["sample-excerpt.npz"] = gate.digest(root / EXCERPT)
    for name in record["input_sha256"]:
        if name.startswith("evidence/"):
            record["input_sha256"][name] = gate.digest(root / name)
    write_json(root / PROVENANCE, record)
    for name in (FIXED, CASE, CASE.replace("120", "360"),
                 "figures/imputation/detail/fig_imputation.manifest.json"):
        manifest = gate.read_json(root / name)
        for spec in manifest["inputs"].values():
            spec["sha256"] = gate.digest(root / spec["path"])
        write_json(root / name, manifest)


def alter_npz(root, edit):
    with np.load(root / EXCERPT, allow_pickle=False) as blob:
        data = {key: blob[key] for key in blob.files}
    edit(data)
    buffer = io.BytesIO()
    np.savez(buffer, **data)
    replace(root / EXCERPT, buffer.getvalue())
    rebind(root)


def test_committed_evidence(observations):
    result = illustrations.validate_illustrations(gate.PAPER, observations)
    assert result["case_indices"] == [120, 360]
    assert result["methods"] == ["D", "O-S", "G5", "full_flank_gaussian"]
    assert result["fixed_final_rows"] == 12
    assert len(result["fixed_final_means"]) == 4
    assert sum(map(len, result["primary_energies"].values())) == 8
    assert len(result["figure_manifests"]) == 4
    assert "not full neural replay" in result["scope"]


@pytest.mark.parametrize("keys,value", [
    (["cases", 0, "y_limits", "D", 0], -99.),
    (["cases", 1, "x_limits", 1], 70),
    (["cases", 0, "y_limits", "full_flank_gaussian", 0], 0.),
    (["axis_sides", "gaussian"], "left"),
    (["cases", 0, "coordinates", "methods", "D", "median", 0, 4], 999.),
    (["y_scale"], "symlog"),
])
def test_detail_tampering_rejected(paper, observations, unpinned, keys, value):
    mutate_json(paper, "figures/imputation/detail/fig_imputation.manifest.json", keys, value)
    with pytest.raises(ERRORS):
        illustrations.validate_illustrations(paper, observations)


def test_independent_pins_are_literal():
    assert illustrations.PINS == {
        EXCERPT: "f3b270d52ea52088fa84414fcb77b6954624913ed474618be245de417034a1cc",
        PROVENANCE: "4416bd1c76aa849de65595e445edc09d5230f2e0bfd2afc75c30c888697c3c66",
        FIXED: "db2436baa4ae0229a870eafef05c867ab1e77d94426c5d0b65fd6111cc721f44",
        CASE: "5d2fbf52337f17d0722cfe19d6c1c975ae1ac4746832e50bc1cbb3f842e1b1f1",
        CASE.replace("120", "360"): "d5823fe76c1deecde13210b32f87ada619716079ac36d12ba1a3a2bbcf3d236f",
        "figures/imputation/detail/fig_imputation.manifest.json": "29802ad07eebd26ab31262441e7441aa78f5e3492e54bd4677a76d601c4d4a40",
    }


@pytest.mark.parametrize("name", list(illustrations.PINS))
def test_pin_rejects_rehashed_mutation(paper, observations, name):
    # Even harmless byte changes cannot redefine the independent audit anchor.
    replace(paper / name, (paper / name).read_bytes() + b" ")
    with pytest.raises(ERRORS, match="independent illustration pin"):
        illustrations.validate_illustrations(paper, observations)


def test_copy_and_hash_rebinding_are_valid(paper, observations, unpinned):
    rebind(paper)
    assert illustrations.validate_illustrations(paper, observations)["fixed_final_rows"] == 12


@pytest.mark.parametrize("edit,match", [
    (lambda d: d["truth"].__setitem__((0, 0, 0), 123.), "physical truth"),
    (lambda d: d.__setitem__("truth", d["truth"][::-1]), "physical truth"),
    (lambda d: d.__setitem__("ids", d["ids"][::-1]), "excerpt ids"),
    (lambda d: d.__setitem__("methods", d["methods"][::-1]), "excerpt methods"),
    (lambda d: d.__setitem__("truth", d["truth"].astype(np.float32)), "dtype"),
    (lambda d: d.__setitem__("samples", d["samples"][:, :, :38]), "shape"),
    (lambda d: d["samples"].__setitem__((0, 0, 0, 0, 0), np.nan), "finite"),
    (lambda d: d.__setitem__("ids", d["ids"].astype("S")), "Unicode"),
    (lambda d: d.__setitem__("extra", np.zeros(1)), "NPZ fields"),
    (lambda d: d["samples"].__setitem__((0, 0, 0, 0, 0), 123.), "primary energy"),
    (lambda d: d["samples"].__setitem__((3, 0, 0, 0, 0), 123.), "Gaussian excerpt decode"),
])
def test_npz_semantics_beyond_pins(paper, observations, unpinned, edit, match):
    alter_npz(paper, edit)
    with pytest.raises(ERRORS, match=match):
        illustrations.validate_illustrations(paper, observations)


@pytest.mark.parametrize("keys,value,match", [
    (["selection", "zero_based_indices"], [119, 360], "selection"),
    (["selection", "after_test_exposure"], False, "selection"),
    (["units", 4], "nT", "units"),
    (["methods", "D", "step"], 29636, "step"),
    (["methods", "D", "selection"], "corrected", "selection"),
    (["methods", "D", "seed"], 13, "seed"),
    (["methods", "D", "seed"], True, "seed"),
    (["methods", "D", "checkpoint_sha256"], "0" * 64, "checkpoint_sha256"),
    (["methods", "D", "ensemble_commit_sha256"], "0" * 64, "ensemble_commit_sha256"),
    (["methods", "G5", "n_generated"], 6, "n_generated"),
    (["methods", "full_flank_gaussian", "seed"], 7, "seed"),
    (["methods", "full_flank_gaussian", "control", "lambda"], 1., "lambda"),
    (["methods", "full_flank_gaussian", "control", "rho"], .5, "rho"),
    (["source_sha256", "prigm/run_score.py"], "0" * 64, "source_sha256"),
    (["source_sha256", "analysis/export_prigm_sample_excerpt.py"], "0" * 64, "source_sha256"),
    (["input_sha256", "artifacts/test/D_s7-g9560/ensemble/gen_00000000/ensemble.npz"], "0" * 64, "input_sha256"),
    (["per_case_primary_energy", "D", 0, "recomputed"], 0., "energy"),
    (["energy_absolute_tolerance"], .01, "energy_absolute_tolerance"),
])
def test_excerpt_manifest_semantics(paper, observations, unpinned, keys, value, match):
    mutate_json(paper, PROVENANCE, keys, value)
    rebind(paper)
    with pytest.raises(ERRORS, match=match):
        illustrations.validate_illustrations(paper, observations)


@pytest.mark.parametrize("filename,keys,value,match", [
    (CASE, ["case_index"], 360, "case_index"),
    (CASE, ["case_id"], "wrong", "case_id"),
    (CASE, ["units", 4], "nT", "units"),
    (CASE, ["coordinates", "ranks", "zero_based_low"], 0, "ranks"),
    (CASE, ["coordinates", "ranks", "zero_based_median"], 20, "ranks"),
    (CASE, ["coordinates", "ranks", "zero_based_high"], 38, "ranks"),
    (CASE, ["coordinates", "ranks", "interpolation"], "linear", "interpolation"),
    (CASE, ["coordinates", "time_bin_start_minutes", 0], -85, "time_bin_start"),
    (CASE, ["coordinates", "gap_minutes", 1], 55, "gap_minutes"),
    (CASE, ["coordinates", "observed_left", "y", 0, 0], 123., "observed_left"),
    (CASE, ["coordinates", "withheld_truth", "x", 0], 5, "withheld_truth"),
    (CASE, ["coordinates", "observed_right", "x", 0], 65, "observed_right"),
    (CASE, ["coordinates", "methods", "D", "median", 0, 0], 123., "median"),
    (CASE, ["coordinates", "methods", "D", "draws", 0, 0, 0], 123., "draws"),
    (CASE, ["y_limits", 0], [0., .01], "clip"),
    (CASE, ["y_limits", 0], [1., -1.], "limits"),
    (CASE, ["y_scale", 4, "scale"], "linear", "y_scale"),
    (CASE, ["y_scale", 4, "linthresh"], 0.5, "y_scale"),
    (FIXED, ["plotted_rows", 0, "energy"], 0., "energy"),
    (FIXED, ["plotted_rows", 0, "residual"], 0., "residual"),
    (FIXED, ["plotted_rows", 0, "seed"], 13, "seed"),
    (FIXED, ["mean_rows", 0, "energy"], 0., "energy"),
    (FIXED, ["mean_rows", 0, "residual"], 0., "residual"),
    (FIXED, ["axes", "x", "scale"], "log", "scale"),
    (FIXED, ["source", "path"], "fig_imputation.py", "source"),
    (FIXED, ["source", "sha256"], "0" * 64, "source"),
    (FIXED, ["scope"], "Interpolated coefficient optimum", "scope"),
])
def test_figure_semantics_beyond_pins(paper, observations, unpinned, filename, keys, value, match):
    mutate_json(paper, filename, keys, value)
    rebind(paper)
    with pytest.raises(ERRORS, match=match):
        illustrations.validate_illustrations(paper, observations)


@pytest.mark.parametrize("filename", [PROVENANCE, CASE, FIXED])
@pytest.mark.parametrize("suffix,match", [(b', "schema_version": 1}', "duplicate"),
                                        (b', "unused": NaN}', "JSON compliant"),
                                        (b', "unused": 1e999}', "JSON compliant")])
def test_strict_json(paper, observations, unpinned, filename, suffix, match):
    payload = (paper / filename).read_bytes().rstrip()
    replace(paper / filename, payload[:-1] + suffix)
    with pytest.raises(ERRORS, match=match):
        illustrations.validate_illustrations(paper, observations)


@pytest.mark.parametrize("name", ["fig_fixed_final.py", "fig_imputation.py"])
def test_supplied_source_must_match_repository(paper, observations, unpinned, name):
    replace(paper / name, (paper / name).read_bytes() + b"\n# altered\n")
    with pytest.raises(ERRORS, match="supplied figure source"):
        illustrations.validate_illustrations(paper, observations)


@pytest.mark.parametrize("directory,name", [("fixed_final", "fig_fixed_final"),
                                           ("imputation/case_120", "fig_imputation"),
                                           ("imputation/case_360", "fig_imputation")])
@pytest.mark.parametrize("suffix", ["pdf", "png"])
def test_output_bytes(paper, observations, unpinned, directory, name, suffix):
    path = paper / "figures" / directory / (name + "." + suffix)
    replace(path, path.read_bytes() + b"tampered")
    with pytest.raises(ERRORS, match="figure output hash"):
        illustrations.validate_illustrations(paper, observations)


@pytest.mark.parametrize("directory", ["evidence/imputation", "figures/fixed_final", "figures/imputation/case_120"])
def test_exact_directory_inventory(paper, observations, unpinned, directory):
    (paper / directory / "extra.partial").write_bytes(b"partial")
    with pytest.raises(ERRORS, match="inventory"):
        illustrations.validate_illustrations(paper, observations)


def test_no_external_symlink(paper, observations, unpinned):
    path = paper / EXCERPT
    path.unlink()
    path.symlink_to(gate.PAPER / EXCERPT)
    with pytest.raises(ERRORS, match="symlink"):
        illustrations.validate_illustrations(paper, observations)


def test_duplicate_npz_field(paper, observations, unpinned):
    buffer = io.BytesIO((paper / EXCERPT).read_bytes())
    with zipfile.ZipFile(buffer, "a") as archive:
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("truth.npy", archive.read("truth.npy"))
    replace(paper / EXCERPT, buffer.getvalue())
    rebind(paper)
    with pytest.raises(ERRORS, match="NPZ fields"):
        illustrations.validate_illustrations(paper, observations)


@pytest.mark.parametrize("what,match", [("truth", "observation arrays"), ("order", "score IDs"),
                                       ("scale", "normalization scale")])
def test_observation_argument_not_just_identity(paper, observations, what, match):
    z, ids, nrm, identity = observations
    if what == "truth":
        z = z.copy()
        z[0, 0, 0] += 1
    elif what == "order":
        ids = list(ids)
        ids[0], ids[1] = ids[1], ids[0]
    else:
        nrm = SimpleNamespace(mean=nrm.mean.copy(), scale=nrm.scale * 2,
                              b_ref_sq=nrm.b_ref_sq, n_bins=nrm.n_bins)
    with pytest.raises(ERRORS, match=match):
        illustrations.validate_illustrations(paper, (z, ids, nrm, identity))


@pytest.mark.parametrize("filename,keys,value,match", [
    ("evidence/real/scores.json", ["data", "D_raw", 0, 120], 0., "primary energy"),
    ("evidence/real/scores.json", ["row_ids", "D_raw"], ["13", "7", "42"], "seed order"),
    ("evidence/real/comparison.json", ["selected", "D_s7", "raw", "generation"], 29636, "raw-selected step"),
    ("evidence/deadline/gaussian/test/SCORES.json",
     ["candidates", "lambda_0.1_rho_0", "raw", "per_window", 120, "primary_es"], 0., "primary energy"),
])
def test_original_score_bindings(paper, observations, unpinned, filename, keys, value, match):
    mutate_json(paper, filename, keys, value)
    rebind(paper)
    with pytest.raises(ERRORS, match=match):
        illustrations.validate_illustrations(paper, observations)


def test_energy_unbiased_offdiagonal_and_standardized_coordinates():
    # One varying coordinate, with nontrivial mean/scale. All pair distances
    # are absolute integer differences; count each unordered pair once.
    mean, scale = np.arange(6, dtype=float), np.arange(1, 7, dtype=float)
    samples = np.broadcast_to(mean[:5], (39, 12, 5)).copy()
    samples[:, 0, 4] += np.arange(39) * scale[4]
    truth = np.broadcast_to(mean[:5], (12, 5)).copy()
    expected = np.arange(39).mean() - sum(j - i for i in range(39) for j in range(i + 1, 39)) / (39 * 38)
    assert illustrations._energy(samples, truth, mean, scale) == pytest.approx(expected, abs=1e-14)


def test_exact_comparison_tuple_list_and_nonfinite():
    illustrations._exact(gate, [[1, 2.]], [(1, 2.)], "tuple/list")
    for value in (True, float("nan"), float("inf"), 2. + 1e-13):
        with pytest.raises(ERRORS):
            illustrations._exact(gate, [value], (2.,), "finite exact coordinate")
