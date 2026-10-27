"""Check score grouping and physical-feature calculations with synthetic data.

Calculate the specified quantities for hidden bins and observed context
separately, then compare their window averages with direct formulas. In
particular, averaging ratios within bins is not the same as taking a ratio
after averaging. Deliberately infeasible inputs test the zero-denominator
convention and ensure the calculation does not silently clip values.

The study fixture mocks the expensive complete primary-score loader, whose
identity checks are covered in test_prigm_comparison.py. Array loading,
completion records, source checks, reductions and exclusive output creation
remain active. These CPU fixtures do not train models or supply scientific
scores.
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from analysis import prigm_training_strata as strata
from prigm.artifacts import id_hash, json_bytes, load_arrays
from prigm.recovery import ContractMismatch, sha256_bytes, sha256_file


def matrices(n):
    data = {f"{arm}_{kind}": np.arange(n, dtype=float)[None, :] + np.array([1., 3., 8.])[:, None]
            for arm in strata.bootstrap.ARMS
            for kind in (*strata.bootstrap.KINDS, "corrected_at_raw_checkpoint")}
    data.update({f"{name}_{kind}": (np.arange(n, dtype=float) + 7)[None, :]
                 for name in strata.bootstrap.CONTROLS for kind in strata.bootstrap.KINDS})
    return data


def windows(n):
    z = np.zeros((n, 48, 6))
    z[:, :, 3] = np.arange(1, n + 1)[:, None]
    z[:, :, 0] = z[:, :, 3] / 2
    z[:, :, 4] = np.arange(48)[None, :] / 48
    z[:, :, 5] = 500  # Redundant channel is not used by these formulas.
    return z


def scalar_features(values):
    return {key: np.asarray(values, dtype=float).copy() for key in strata.FEATURES}


def manual_mean(values, months, drawn_years):
    monthly = []
    for year in drawn_years:
        for month in sorted(set(m for m in months if m // 100 == year)):
            selected = [v for v, m in zip(values, months) if m == month]
            monthly.append(sum(selected) / len(selected))
    return sum(monthly) / len(monthly)


def test_direct_bin_formulas_before_window_means_and_exact_slices():
    z = windows(2)
    z[:, :, :3] = np.arange(2 * 48 * 3).reshape(2, 48, 3) / 100
    z[:, :, 3] = np.arange(1, 97).reshape(2, 48) / 10
    before = z.copy()
    result = strata.physical_features(z, 4.)
    for population, indices in (("target", range(18, 30)), ("context", [*range(18), *range(30, 48)])):
        for w in range(2):
            q, c, a = [], [], []
            for i in indices:
                m2 = float(z[w, i, 3]) ** 2
                num = m2 - sum(float(x) ** 2 for x in z[w, i, :3])
                q.append((m2 + z[w, i, 4]) / 4)
                c.append(num / m2)
                a.append(num / 4)
            assert result[f"{population}_energy"][w] == pytest.approx(sum(q) / len(q))
            assert result[f"{population}_cancellation"][w] == pytest.approx(sum(c) / len(c))
            if population == "target":
                assert result["target_absolute_cancellation"][w] == pytest.approx(sum(a) / len(a))
    np.testing.assert_array_equal(z, before)
    # The binwise ratio mean is not a ratio of window means.
    target = z[0, 18:30]
    ratio_of_means = (np.mean(target[:, 3] ** 2) - np.mean(np.sum(target[:, :3] ** 2, axis=1))) / np.mean(target[:, 3] ** 2)
    assert result["target_cancellation"][0] != pytest.approx(ratio_of_means, rel=1e-7)


def test_zero_field_convention_and_no_physical_clipping():
    zero = strata.physical_features(np.zeros((1, 48, 6)), 1.)
    assert all(v.tolist() == [0.] for v in zero.values())
    z = np.zeros((1, 48, 6))
    z[:, :, 0] = 2
    z[:, :, 4] = -3
    got = strata.physical_features(z, 2.)
    assert got["target_cancellation"].tolist() == [0.]  # m^2=0 ratio convention only.
    assert got["target_absolute_cancellation"].tolist() == [-2.]
    assert got["context_energy"].tolist() == [-1.5]
    z[:, :, 3] = 1
    assert strata.physical_features(z, 2.)["target_cancellation"].tolist() == [-3.]


@pytest.mark.parametrize("b_ref_sq", [0., -1., np.nan, np.inf, "bad", [1.]])
def test_invalid_reference_refuses(b_ref_sq):
    with pytest.raises(ContractMismatch, match="finite"):
        strata.physical_features(windows(2), b_ref_sq)


@pytest.mark.parametrize("damage", ["nan", "inf", "overflow", "shape", "empty", "strings"])
def test_nonfinite_or_invalid_physical_windows_refuse(damage):
    z = windows(2)
    if damage in ("nan", "inf", "overflow"):
        z[0, 1, 0] = {"nan": np.nan, "inf": np.inf, "overflow": 1e308}[damage]
    elif damage == "shape":
        z = z[:, :47]
    elif damage == "empty":
        z = z[:0]
    else:
        z = z.astype(str)
    with pytest.raises(ContractMismatch):
        strata.physical_features(z, 1.)


def test_training_cutoffs_hidden_test_independence_and_training_perturbation():
    train = windows(8)
    test = windows(6)
    cutoffs = strata.fit_cutoffs(strata.physical_features(train, 4.))
    unchanged = copy.deepcopy(cutoffs)
    original, membership = strata.group_masks(strata.physical_features(test, 4.), cutoffs)
    hidden = test.copy()
    hidden[:, 18:30, 3] += 100
    altered, new_membership = strata.group_masks(strata.physical_features(hidden, 4.), cutoffs)
    assert cutoffs == unchanged == strata.fit_cutoffs(strata.physical_features(train, 4.))
    for key in original:
        if key.startswith("context_") or key == "full_test":
            np.testing.assert_array_equal(original[key], altered[key])
    assert not np.array_equal(original["target_high"], altered["target_high"])
    assert not np.array_equal(membership["target_energy_quartile"], new_membership["target_energy_quartile"])
    for key in ("context_energy_quartile", "context_high_cancellation"):
        np.testing.assert_array_equal(membership[key], new_membership[key])
    train[:, :, 3] += 20
    revised = strata.fit_cutoffs(strata.physical_features(train, 4.))
    assert all(cutoffs[key]["values"] != revised[key]["values"] for key in cutoffs)
    _, after_train = strata.group_masks(strata.physical_features(test, 4.), revised)
    assert not np.array_equal(membership["context_energy_quartile"], after_train["context_energy_quartile"])


def test_linear_quantiles_use_all_overlap_windows_not_unique_feature_values():
    train = scalar_features([0., 0., 0., 0., 4., 8., 12., 16.])
    got = strata.fit_cutoffs(train)
    assert got["target_energy"]["values"] == [0., 2., 9.]
    assert got["context_energy"]["values"] != np.quantile(np.unique(train["context_energy"]), [.25, .5, .75]).tolist()
    for name, row in got.items():
        assert row["n_training_windows"] == 8
        assert row["quantile_method"] == "linear"
        assert row["values"] == np.quantile(train[name], row["quantiles"], method="linear").tolist()


def test_exact_threshold_ties_high_is_strict_and_quartiles_go_right():
    cutoffs = strata.fit_cutoffs(scalar_features([0., 1., 2., 3., 4.]))
    masks, membership = strata.group_masks(scalar_features([0., 1., 2., 3., 4.]), cutoffs)
    assert membership["target_energy_quartile"].tolist() == [1, 2, 3, 4, 4]
    assert membership["context_high_cancellation"].tolist() == [False, False, False, False, True]
    assert masks["target_rest"].tolist() == [True, True, True, True, False]
    cutoffs = strata.fit_cutoffs(scalar_features([7.] * 8))
    masks, membership = strata.group_masks(scalar_features([7.] * 3), cutoffs)
    assert membership["context_energy_quartile"].tolist() == [4, 4, 4]
    assert cutoffs["target_energy"]["training_edge_tie_counts"] == [8, 8, 8]
    assert len(masks) == 21
    for population in ("target", "context"):
        joint = [mask for key, mask in masks.items() if key.startswith(f"{population}_q")]
        assert len(joint) == 8
        np.testing.assert_array_equal(np.sum(joint, axis=0), np.ones(3))


def test_empty_single_year_groups_keep_ids_methods_and_descriptive_means(monkeypatch):
    ids = ["2005-08-a-s52", "2005-09-b-s71", "2006-08-c-s50"]
    features = scalar_features([0., 0., 10.])
    cutoffs = strata.fit_cutoffs(scalar_features([0.] * 4))
    groups = strata.summarize_groups(matrices(3), [200508, 200509, 200608], ids, features, cutoffs,
                                     replicates=19)["groups"]
    assert len(groups) == 21
    for row in groups.values():
        assert row["ids_sha256"] == id_hash(row["ids"])
        assert len(row["methods"]) == 18
        assert sum(row["month_counts"].values()) == row["n_windows"]
        assert sum(row["year_counts"].values()) == row["n_windows"]
        assert sum(row["source_counts"].values()) == row["n_windows"]
        if row["status"] == "empty_stratum":
            assert row["comparisons"] == {}
            assert row["ids"] == row["months"] == []
            assert all(m["month_balanced"] is None for m in row["methods"].values())
        else:
            assert len(row["comparisons"]) == 4
        if row["status"] == "single_year":
            assert all(c["year_95"] is None and c["replicates_used"] == 0 for c in row["comparisons"].values())
    assert groups["target_high"]["ids"] == [ids[2]]
    assert groups["target_high"]["status"] == "single_year"
    assert groups["target_q1_high"]["status"] == "empty_stratum"
    assert groups["target_high"]["realized_feature_means"]["target_cancellation"]["mean"] == 10.
    assert groups["context_high"]["membership_scope"] == "conditioning_inputs_exploratory"


def test_unequal_months_repeated_aab_and_fixed_seed_control_rows():
    months = np.array([200508, 200508, 200509, 200608])
    ids = ["2005-08-a-s52", "2005-08-b-s52", "2005-09-c-s71", "2006-08-d-s71"]
    values = np.array([1., 3., 10., 20.])
    data = matrices(4)
    data["O-S_raw"] = np.stack([values, 2 * values, values + 9])
    data["D_raw"] = np.stack([values + 1, values + 5, values + 10])
    data["G5_raw"] = data["D_raw"] * 2
    data["gaussian_raw"] = (3 * values + 1)[None, :]
    data["interpolation_raw"] = (values / 2)[None, :]
    features = scalar_features([1.] * 4)
    row = strata.summarize_groups(data, months, ids, features, strata.fit_cutoffs(features),
                                  seed=43, replicates=37)["groups"]["full_test"]
    assert row["methods"]["O-S_raw"]["seedwise"]["7"]["month_balanced"] == pytest.approx(32 / 3)
    assert row["methods"]["O-S_raw"]["seedwise"]["7"]["mean"] == 8.5
    assert set(row["methods"]["gaussian_raw"]["seedwise"]) == {"shared"}
    assert len(row["methods"]["O-S_raw"]["seedwise"]) == 3
    weights = strata.bootstrap.month_occurrence_weights(months, [[2005, 2005, 2006]])
    np.testing.assert_allclose(weights, [[2 / 5, 2 / 5, 1 / 5]])
    assert (weights @ strata.bootstrap.monthly_means(values[None, :], months).T).item() == pytest.approx(44 / 5)
    rng = np.random.Generator(np.random.PCG64(43))
    draws = np.array([2005, 2006])[rng.integers(2, size=(37, 2))]
    for method, baseline in strata.summary.PAIRS:
        actual = row["comparisons"][f"{method}_minus_{baseline}"]
        boot = [np.mean([manual_mean(v, months, draw) for v in data[method]])
                - np.mean([manual_mean(v, months, draw) for v in data[baseline]]) for draw in draws]
        assert actual["year_95"] == pytest.approx(np.quantile(boot, [.025, .975]))
        left = np.array([manual_mean(v, months, [2005, 2006]) for v in data[method]])
        right = np.array([manual_mean(v, months, [2005, 2006]) for v in data[baseline]])
        assert list(actual["seed_diffs"].values()) == pytest.approx(left - right)
        assert actual["baseline_realizations"] == len(data[baseline])


@pytest.mark.parametrize("damage", ["seed", "control", "method", "nan", "ids", "months"])
def test_matrix_and_calendar_identity_refuses(damage):
    data = matrices(2)
    ids, months = ["2005-08-a-s52", "2006-08-b-s71"], [200508, 200608]
    if damage == "seed":
        data["D_raw"] = data["D_raw"][:2]
    elif damage == "control":
        data["gaussian_raw"] = np.repeat(data["gaussian_raw"], 3, axis=0)
    elif damage == "method":
        del data["D_corrected"]
    elif damage == "nan":
        data["D_raw"][0, 0] = np.nan
    elif damage == "ids":
        ids[1] = ids[0]
    else:
        months[0] = 200507
    features = scalar_features([1., 2.])
    with pytest.raises(ContractMismatch):
        strata.summarize_groups(data, months, ids, features, strata.fit_cutoffs(features), replicates=3)


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(record))
    return sha256_file(path)


@pytest.fixture
def study(tmp_path, monkeypatch):
    """Mock primary gate only, retaining real 480-ID arrays and local hashes."""
    root = tmp_path / "private-study-name"
    inventory = root / "artifacts/inventory"
    inventory.mkdir(parents=True)
    months = np.repeat([200508, 200509, 200608], [160, 80, 240])
    ids = [f"{month // 100}-{month % 100:02d}-{i:04d}-s{52 if i % 2 else 71}" for i, month in enumerate(months)]
    identities = {}
    for split, z, wids in (("TRAIN", windows(8), [f"train-{i}" for i in range(8)]),
                           ("TEST", windows(480), ids)):
        path = inventory / f"{split.lower()}.npz"
        np.savez(path, z=z, ids=np.asarray(wids), split=split, mean=np.arange(6), scale=np.arange(1, 7),
                 b_ref_sq=4., n_bins=64)
        identities[split] = load_arrays(path, split)[3]
    identities["VAL"] = {**identities["TEST"], "split": "VAL", "n_windows": 3}
    protocol = {"array_root": str(inventory), "arrays": identities,
                "training_data_hashes": {"arrays_sha256": identities["TRAIN"]["sha256"]},
                "inference_source_hashes": {f"prigm/{p.name}": sha256_file(p)
                                            for p in Path(strata.bootstrap.__file__).parent.glob("*.py")},
                "private_name": "private-study-name", "path": "/home/hidden-name/data"}
    digest = save(root / "PROTOCOL.json", protocol)
    hashes = {str(root / "PROTOCOL.json"): digest}
    for name in ("SELECTION.json", "TEST_STARTED.json"):
        hashes[str(root / name)] = save(root / name, {"fixture": True})
    for split in ("TRAIN", "TEST"):
        hashes[str(inventory / f"{split.lower()}.npz")] = identities[split]["sha256"]
    save(inventory / "summary.json", {
        "train_starts": 8, "train_windows_saved": 8, "train_bin_union": 64, "test_selected": 480,
        "val_selected": 3, "hashes": {str(y): "e" * 64 for y in strata.YEARS}, "path": "/home/hidden-name/raw"})
    comparison = save(root / "artifacts/comparison.json", {"fixture": True})
    original = save(root / "artifacts/diagnostic-summary-01.json", {"fixture": True})
    launchers = {name: save(root / name, {"fixture": True}) for name in ("run_fixed_followup.py", "run_cpu_after_real.py")}
    save(root / "PIPELINE_COMPLETED.json", {"protocol_sha256": digest, "comparison_sha256": comparison,
         "selection_sha256": hashes[str(root / "SELECTION.json")], "launcher_sha256": launchers["run_fixed_followup.py"]})
    save(root / "CPU_FOLLOWUP_COMPLETED.json", {"protocol_sha256": digest, "summary_sha256": original,
         "analysis_source_sha256": sha256_file(Path(strata.summary.__file__)), "launcher_sha256": launchers["run_cpu_after_real.py"]})
    fit_path = root / "artifacts/controls_test/gaussian_fit.npz"
    fit_hash = save(fit_path, {"fixture_payload": True})
    hashes[str(fit_path)] = fit_hash
    data, methods = matrices(480), {}
    for label, values in data.items():
        arm, variant = label.split("_", 1)
        selection = "raw" if variant == "corrected_at_raw_checkpoint" else variant
        scoring = "raw" if variant == "raw" else "corrected"
        generation = 10 if selection == "raw" else 20
        keys = ["shared"] if len(values) == 1 else list(map(str, strata.FINAL_SEEDS))
        files = {}
        for key in keys:
            parent = f"artifacts/controls_test/{arm}" if key == "shared" else f"artifacts/test/{arm}_s{key}-g{generation}"
            path = root / parent / f"score-{scoring}.json"
            ensemble = root / parent
            if key != "shared":
                ensemble /= "ensemble"
            ensemble /= "gen_00000000"
            for filename in ("COMMIT.json", "ensemble.npz"):
                hashes[str(ensemble / filename)] = save(ensemble / filename, {"fixture_payload": parent})
            hashes[str(path)] = save(path, {
                "checkpoint_sha256": None if key == "shared" else sha256_bytes(parent.encode()),
                "control_fit_sha256": fit_hash if arm == "gaussian" else None,
                "ensemble_commit_sha256": hashes[str(ensemble / "COMMIT.json")],
            })
            files[key] = str(path)
        methods[label] = {"procedure": arm if len(values) == 1 else "diffusion", "score_files": files,
                          "scoring_kind": scoring, "selection_kind": selection,
                          "generations": {k: generation for k in keys} if len(values) == 3 else {}}
    provenance = {"ids": ids, "protocol_sha256": digest, "arrays": identities["TEST"], "methods": methods,
                  "input_sha256": hashes, "selection_sha256": hashes[str(root / "SELECTION.json")],
                  "test_started_sha256": hashes[str(root / "TEST_STARTED.json")]}

    def load(actual_root, actual_hash):
        assert actual_root == root and actual_hash == digest
        return data, months, provenance

    monkeypatch.setattr(strata.bootstrap, "load_study", load)
    full_helper = strata.summarize_groups

    def bounded(*args, seed, replicates):
        assert seed == 7003 and replicates == 10000  # Production must request fixed settings.
        return full_helper(*args, seed=seed, replicates=17)

    monkeypatch.setattr(strata, "summarize_groups", bounded)
    return SimpleNamespace(root=root, digest=digest, data=data, months=months, ids=ids,
                           provenance=provenance, protocol=protocol, output=tmp_path / "report.json")


def cli(study):
    return ["--study-root", str(study.root), "--protocol-sha256", study.digest, "--output", str(study.output)]


def test_cli_schema_hashes_anonymity_readonly_and_fresh_publication(study, capsys):
    before = {p: sha256_file(p) for p in study.root.rglob("*") if p.is_file()}
    assert strata.main(cli(study)) == 0
    payload = study.output.read_text()
    result = json.loads(payload)
    assert len(result["groups"]) == 21 and len(result["methods"]) == 18
    assert result["schema_version"] == 1
    assert result["ids"] == study.ids and result["ids_sha256"] == id_hash(study.ids)
    assert result["months"] == study.months.tolist()
    assert result["analysis_source_sha256"] == sha256_file(Path(strata.__file__))
    assert result["producer_source_sha256"]["analysis/prigm_followup_summary.py"] == sha256_file(Path(strata.summary.__file__))
    assert result["normalization"]["b_ref_sq"] == 4.
    assert result["inventory_counts"]["train_windows_saved"] == 8
    for value in result["cutoffs"].values():
        assert value["n_training_windows"] == 8
    assert result["features_in_test_id_order"] == {k: v.tolist() for k, v in strata.physical_features(windows(480), 4.).items()}
    for path, digest in result["input_sha256"].items():
        assert not Path(path).is_absolute() and ".." not in Path(path).parts
        assert sha256_file(study.root / path) == digest
    for private in (str(study.root), "private-study-name", "hidden-name", "/home/"):
        assert private not in payload and private not in capsys.readouterr().out
    assert {p: sha256_file(p) for p in before} == before
    assert not study.output.with_name(f".{study.output.name}.partial").exists()
    d = result["methods"]
    assert d["D_raw"]["files"]["7"]["checkpoint_sha256"] == d["D_corrected_at_raw_checkpoint"]["files"]["7"]["checkpoint_sha256"]
    assert d["D_corrected"]["files"]["7"]["checkpoint_sha256"] != d["D_raw"]["files"]["7"]["checkpoint_sha256"]
    assert set(d["gaussian_raw"]["files"]) == {"shared"}
    assert "after inspection" in result["scope"]["timing"]
    assert "training-threshold uncertainty" in result["bootstrap"]["scope"]
    with pytest.raises(FileExistsError):
        strata.main(cli(study))
    study.output = study.output.with_name("second.json")
    assert strata.main(cli(study)) == 0
    assert study.output.read_text() == payload


@pytest.mark.parametrize("relative", ["PIPELINE_COMPLETED.json", "CPU_FOLLOWUP_COMPLETED.json"])
def test_missing_completion_refuses(study, relative):
    (study.root / relative).unlink()
    with pytest.raises(FileNotFoundError):
        strata.main(cli(study))
    assert not study.output.exists()


@pytest.mark.parametrize("relative,key", [("PIPELINE_COMPLETED.json", "protocol_sha256"),
                                         ("CPU_FOLLOWUP_COMPLETED.json", "analysis_source_sha256"),
                                         ("CPU_FOLLOWUP_COMPLETED.json", "summary_sha256"),
                                         ("PIPELINE_COMPLETED.json", "launcher_sha256")])
def test_bad_completion_identity_refuses(study, relative, key):
    path = study.root / relative
    row = json.loads(path.read_bytes())
    row[key] = "a" * 64
    save(path, row)
    with pytest.raises(ContractMismatch):
        strata.main(cli(study))
    assert not study.output.exists()


def test_loader_identity_failure_propagates_without_publication(study, monkeypatch):
    def fail(*args):
        raise ContractMismatch("checkpoint/score identity rejected by primary gate")

    monkeypatch.setattr(strata.bootstrap, "load_study", fail)
    with pytest.raises(ContractMismatch, match="primary gate"):
        strata.main(cli(study))
    assert not study.output.exists()


@pytest.mark.parametrize("damage", ["population", "ids", "normalization", "train_count", "nan", "b_ref", "array_hash", "source"])
def test_bad_array_population_transform_source_and_finiteness_refuse(study, monkeypatch, damage):
    original = strata.load_arrays

    def damaged(path, split, digest):
        z, ids, nrm, identity = original(path, split, digest)
        if split == "TEST":
            if damage == "ids":
                ids = ids[::-1]
            elif damage == "normalization":
                nrm.mean[0] += 1
            elif damage == "nan":
                z[0, 0, 0] = np.nan
            elif damage == "b_ref":
                nrm = type(nrm)(nrm.mean, nrm.scale, 0., nrm.n_bins)
        return z, ids, nrm, identity

    monkeypatch.setattr(strata, "load_arrays", damaged)
    if damage == "population":
        study.provenance["ids"] = study.ids[:-1]
    elif damage == "train_count":
        path = study.root / "artifacts/inventory/summary.json"
        row = json.loads(path.read_bytes())
        row["train_starts"] -= 1
        save(path, row)
    elif damage == "array_hash":
        with (study.root / "artifacts/inventory/train.npz").open("ab") as handle:
            handle.write(b"changed")
    elif damage == "source":
        def fail(*args):
            raise ContractMismatch("source changed")
        monkeypatch.setattr(strata, "require_source_match", fail)
    with pytest.raises(ContractMismatch):
        strata.main(cli(study))
    assert not study.output.exists()


@pytest.mark.parametrize("relative", ["artifacts/inventory/train.npz", "artifacts/inventory/test.npz",
    "artifacts/inventory/summary.json", "artifacts/test/D_s7-g10/score-raw.json", "PROTOCOL.json",
    "SELECTION.json", "TEST_STARTED.json", "PIPELINE_COMPLETED.json", "CPU_FOLLOWUP_COMPLETED.json",
    "artifacts/comparison.json", "artifacts/diagnostic-summary-01.json",
    "artifacts/test/D_s7-g10/ensemble/gen_00000000/COMMIT.json",
    "artifacts/test/D_s7-g10/ensemble/gen_00000000/ensemble.npz",
    "artifacts/controls_test/gaussian_fit.npz"])
def test_changed_input_during_analysis_cannot_publish(study, monkeypatch, relative):
    original = strata.summarize_groups

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        with (study.root / relative).open("ab") as handle:
            handle.write(b" ")
        return result

    monkeypatch.setattr(strata, "summarize_groups", mutate)
    with pytest.raises(ContractMismatch, match="input changed"):
        strata.main(cli(study))
    assert not study.output.exists()


@pytest.mark.parametrize("damage", ["input", "producer", "partial", "race"])
def test_last_publication_recheck_preserves_partial_without_success(study, monkeypatch, damage):
    original = strata.fsync_dir
    partial = study.output.with_name(f".{study.output.name}.partial")

    def mutate(path):
        original(path)
        if damage == "input":
            with (study.root / "CPU_FOLLOWUP_COMPLETED.json").open("ab") as handle:
                handle.write(b" ")
        elif damage == "partial":
            partial.write_bytes(b"changed")
        elif damage == "race":
            study.output.write_bytes(b"other publisher")
        else:
            original_hash = strata.sha256_file
            monkeypatch.setattr(strata, "sha256_file", lambda p: "a" * 64 if p == Path(strata.__file__) else original_hash(p))

    monkeypatch.setattr(strata, "fsync_dir", mutate)
    with pytest.raises((ContractMismatch, FileExistsError)):
        strata.main(cli(study))
    assert partial.exists()
    if damage == "race":
        assert study.output.read_bytes() == b"other publisher"
    else:
        assert not study.output.exists()
    with pytest.raises(FileExistsError):
        strata.main(cli(study))


@pytest.mark.parametrize("damage", ["output", "partial", "symlink", "partial_symlink", "parent"])
def test_fresh_output_refusal_precedes_study_reads(study, monkeypatch, damage):
    def forbidden(*args):
        raise AssertionError("must reject before reading study")

    monkeypatch.setattr(strata, "summarize_study", forbidden)
    partial = study.output.with_name(f".{study.output.name}.partial")
    if damage in ("output", "partial"):
        (study.output if damage == "output" else partial).write_bytes(b"existing")
    elif damage in ("symlink", "partial_symlink"):
        (study.output if damage == "symlink" else partial).symlink_to(study.output.parent / "absent")
    else:
        study.output = study.output.parent / "absent" / "report.json"
    with pytest.raises((FileExistsError, FileNotFoundError)):
        strata.main(cli(study))


def test_external_and_retargetable_input_references_refuse(study):
    with pytest.raises(ContractMismatch, match="escapes study"):
        strata._input_path(study.root.parent / "outside", study.root)
    (study.root / "alias").symlink_to(study.root / "artifacts", target_is_directory=True)
    with pytest.raises(ContractMismatch, match="symlinks"):
        strata._input_path("alias/comparison.json", study.root)


@pytest.mark.parametrize("relative", ["artifacts/test/D_s7-g10/ensemble/gen_00000000/ensemble.npz",
                                      "artifacts/controls_test/gaussian_fit.npz"])
def test_resolved_loader_inventory_cannot_hide_internal_payload_alias(study, relative):
    path = study.root / relative
    canonical = path.with_name("alternate.npz")
    path.rename(canonical)
    path.symlink_to(canonical)
    # This is the loader's resolved-path inventory, not the original reference.
    hashes = study.provenance["input_sha256"]
    hashes[str(canonical)] = hashes.pop(str(path))
    with pytest.raises(ContractMismatch, match="not bound by loader"):
        strata.main(cli(study))
    assert not study.output.exists()


@pytest.mark.parametrize("argv", [[], ["--study-root", "absent", "--output", "unused"],
                                    ["--study-root", "absent", "--protocol-sha256", "a" * 64,
                                     "--output", "unused", "--replicates", "3"]])
def test_cli_requires_protocol_and_has_no_bootstrap_override(argv):
    with pytest.raises(SystemExit):
        strata.main(argv)
