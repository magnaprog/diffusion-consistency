"""Check paired score comparisons using local saved-ensemble fixtures.

Independently average windows within each represented month, then average
those month means. When a resampled year appears twice, include its month
means twice; missing months contribute no invented observations. Apply the
same resampled years to the compared procedures and retain the fixed model
seeds in the recorded comparison.

The fixtures also exercise array, source and ensemble hashes. Reported
intervals concern variation across test years for fixed fitted models, not
training-seed or selection uncertainty. They have no familywise correction
and do not turn historically exposed TEST data into a blind confirmation.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from prigm import run_bootstrap as comparison
from prigm.artifacts import (
    NOISE_CONVENTION, STATE_CONVENTION, json_bytes, load_arrays, publish_ensemble,
)
from prigm.recovery import ContractMismatch, PublicationError, sha256_bytes, sha256_file
from prigm.scoring import month_balanced_mean, month_balanced_mean_replicated


def write_json(path, row):
    path.write_bytes(json_bytes(row))


def manual_mean(values, months, drawn_years):
    """Independent month averaging, retaining every drawn occurrence."""
    means = []
    for year in drawn_years:
        for month in sorted(set(months[months // 100 == year])):
            selected = values[months == month].tolist()
            means.append(sum(selected) / len(selected))
    return sum(means) / len(means)


@pytest.mark.parametrize("missing_september", [False, True])
def test_vectorized_occurrences_match_manual_and_unchanged_helper(missing_september):
    months = np.array([200508, 200508, 200509, 200608, 200609, 200609, 200609, 200708])
    values = np.array([1., 3., 10., 20., 26., 30., 34., 9999.])
    if missing_september:
        values, months = values[months != 200609], months[months != 200609]
    rows = np.stack([values, 2 * values, values + 7])
    draws = np.array([[2005, 2005, 2006], [2006, 2006, 2005], [2024, 2005, 2006]])
    weights = comparison.month_occurrence_weights(months, draws)
    actual = weights @ comparison.monthly_means(rows, months).T
    for r, draw in enumerate(draws):
        for s, row in enumerate(rows):
            expected = manual_mean(row, months, draw)
            assert actual[r, s] == pytest.approx(expected, rel=1e-15)
            assert month_balanced_mean_replicated(row, months, lambda m: int(m) // 100, draw) == pytest.approx(expected)
    # Pooling windows differs from equal weighting of represented months.
    # With September missing, averaging drawn-year means also differs.
    assert actual[0, 0] != pytest.approx(np.concatenate([values[months // 100 == y] for y in draws[0]]).mean())
    if missing_september:
        annual_average = (2 * manual_mean(values, months, [2005]) + manual_mean(values, months, [2006])) / 3
        assert actual[0, 0] != pytest.approx(annual_average)


def test_empty_bootstrap_population_is_rejected_not_nanmean():
    with pytest.raises(ContractMismatch, match="no represented months"):
        comparison.month_occurrence_weights(np.array([200508]), [[2006, 2006]])
    with pytest.raises(ContractMismatch, match="finite"):
        comparison.monthly_means([[1., np.nan]], [200508, 200509])


def summary(points, year, twoyear=None):
    return {"seed_means": np.asarray(points, dtype=float), "year": np.asarray(year, dtype=float),
            "twoyear": np.asarray(year if twoyear is None else twoyear, dtype=float)}


def test_relative_interval_uses_replicate_denominator_and_single_control():
    baseline = summary([25.], [1., 100., 2., 50.])
    method = summary([20., 21., 22.], [0.9, 90., 1.8, 45.])
    result = comparison.paired_summary(method, baseline)
    assert result["seed_diffs"] == [-5., -4., -3.]
    assert result["mean_diff"] == -4.
    assert result["rel_mean_vs_base"] == -4 / 25
    assert result["rel_year_95"] == pytest.approx([-0.1, -0.1])
    assert result["rel_twoyear_95"] == pytest.approx([-0.1, -0.1])
    assert result["rel_year_95"] != pytest.approx(np.quantile((method["year"] - baseline["year"]) / 25, [.025, .975]))
    assert result["method_fixed_models"] == 3
    assert result["baseline_fixed_models"] == 1


@pytest.mark.parametrize("base_point,base_draws", [(0., [0., 1., 2.]), (-1., [-1., -2., 3.]), (1., [0., 1., 2.])])
def test_nonpositive_relative_denominators_are_explicit_not_dropped(base_point, base_draws):
    result = comparison.paired_summary(summary([2.] * 3, [1., 2., 3.]), summary([base_point], base_draws))
    assert result["rel_mean_vs_base"] == (None if base_point <= 0 else 1.)
    for scheme in ("year", "twoyear"):
        assert result[f"rel_{scheme}_95"] is None
        assert result[f"rel_{scheme}_undefined_replicates"] == sum(x <= 0 for x in base_draws)
        assert result[f"{scheme}_95"] is not None
    assert result["descriptive_gates"]["strict_relative_equivalence_within_2_percent_both_intervals"] is None


@pytest.mark.parametrize("value,equivalent,superior", [(98., False, True), (99., True, False), (102., False, False)])
def test_two_percent_gate_uses_strict_interval_bounds(value, equivalent, superior):
    result = comparison.paired_summary(summary([value] * 3, [value] * 4), summary([100.], [100.] * 4))
    assert result["descriptive_gates"]["strict_relative_equivalence_within_2_percent_both_intervals"] is equivalent
    assert result["descriptive_gates"]["observed_improvement_at_least_2_percent"] is superior


def test_equal_point_means_do_not_establish_equivalence():
    result = comparison.paired_summary(summary([100.] * 3, [50., 150.]), summary([100.], [100., 100.]))
    assert result["mean_diff"] == 0
    assert result["descriptive_gates"]["strict_relative_equivalence_within_2_percent_both_intervals"] is False
    assert not any("p_value" in key for key in result)


@pytest.fixture
def study(tmp_path, monkeypatch):
    # Small population, unequal month counts, all fixed model seeds. The protocol
    # hashes real local source files; ensemble commits and payload hashes are real.
    monkeypatch.setattr(comparison, "YEARS", (2005, 2006))
    monkeypatch.setattr(comparison, "year_blocks", lambda: np.array([2005, 2006]))
    monkeypatch.setattr(comparison, "two_year_blocks", lambda: [(2005, 2006)])
    monkeypatch.setattr(comparison, "BOOTSTRAP_REPLICATES", 61)
    ids = ["2005-08-01T00:00", "2005-08-02T00:00", "2005-09-01T00:00",
           "2006-08-01T00:00", "2006-09-01T00:00", "2006-09-02T00:00", "2006-09-03T00:00"]
    months = np.array([int(w[:4]) * 100 + int(w[5:7]) for w in ids])
    arrays = tmp_path / "arrays"
    arrays.mkdir()
    identities = {}
    for split in ("TRAIN", "TEST"):
        path = arrays / f"{split.lower()}.npz"
        np.savez(path, z=np.ones((len(ids), 48, 6)), ids=np.asarray(ids), split=np.array(split),
                 mean=np.zeros(6), scale=np.ones(6), b_ref_sq=np.array(1.), n_bins=np.array(100))
        identities[split] = load_arrays(path, split)[3]
    protocol = {
        "schema_version": 2, "exposure": "historical_test_exposed", "array_root": str(arrays),
        "arrays": identities, "arms": list(comparison.ARMS), "seeds": list(comparison.FINAL_SEEDS),
        "milestones": [10, 20], "samplers": {"TEST": {"n_eval": 50, "n_draws": 39, "eval_seed": 7}},
        "controls": {"interpolation": {"conditioning": "two_boundaries"},
                     "persistence": {"conditioning": "left_boundary"},
                     "gaussian": {"rho": 0.0, "jitter": 1e-6, "seed": 13}},
        "inference_source_hashes": {f"prigm/{p.name}": sha256_file(p)
                                    for p in Path(comparison.__file__).parent.glob("*.py")},
    }
    write_json(tmp_path / "PROTOCOL.json", protocol)
    protocol_hash = sha256_file(tmp_path / "PROTOCOL.json")
    selection = {"schema_version": 2, "protocol_sha256": protocol_hash, "exposure": "historical_test_exposed",
                 "selected": {}, "test_jobs": [], "test_checkpoint_hashes": [], "validation_score_sha256": {}}
    expected = {}

    def publish_scores(parent, ensemble_root, meta, raw_values):
        channels = meta["n_generated"]
        dtype = np.float32 if meta.get("procedure", "diffusion") == "diffusion" else np.float64
        publish_ensemble(ensemble_root, np.zeros((len(ids), 39, channels, 48), dtype=dtype), ids, meta)
        commit_hash = sha256_file(ensemble_root / "gen_00000000/COMMIT.json")
        # Deliberately synthetic finite score records: comparisons must check
        # their identities and reductions, never regenerate predictive draws.
        for kind in comparison.KINDS:
            values = raw_values - (0.5 if kind == "corrected" else 0.0)
            row = {
                "schema_version": 2, "protocol_sha256": protocol_hash, "arrays": identities["TEST"],
                "checkpoint_sha256": meta.get("checkpoint_sha256"), "ensemble_commit_sha256": commit_hash,
                "procedure": meta.get("procedure", "diffusion"), "control_fit_sha256": meta.get("control_fit_sha256"),
                "ids": ids, "months": months.tolist(), "n": len(ids), "corrected": kind == "corrected",
                "scores": values.tolist(), "mean": float(values.mean()),
                "month_balanced": month_balanced_mean(values, months),
            }
            path = parent / f"score-{kind}.json"
            write_json(path, row)
            expected[str(path)] = values

    pattern = np.array([1., 3., 10., 20., 26., 30., 34.])
    for arm_index, arm in enumerate(comparison.ARMS):
        for seed_index, seed in enumerate(comparison.FINAL_SEEDS):
            name = f"{arm}_s{seed}"
            selection["selected"][name] = {}
            for kind, generation in (("raw", 10), ("corrected", 20)):
                checkpoint = sha256_bytes(f"{name}:{generation}".encode())
                selection["selected"][name][kind] = {"generation": generation, "score": 1., "checkpoint_sha256": checkpoint}
                selection["test_jobs"].append([name, generation])
                selection["test_checkpoint_hashes"].append(checkpoint)
                parent = tmp_path / "artifacts/test" / f"{name}-g{generation}"
                parent.mkdir(parents=True)
                meta = {
                    "protocol_sha256": protocol_hash, "arrays": identities["TEST"],
                    "checkpoint_sha256": checkpoint, "arm": arm, "seed": seed, "generation_updates": generation,
                    "n_generated": 5 if arm == "G5" else 6, "sampler": protocol["samplers"]["TEST"],
                    "state_convention": STATE_CONVENTION, "noise_convention": NOISE_CONVENTION,
                }
                values = 10 + arm_index + seed_index / 10 + pattern + (2 if generation == 20 else 0)
                publish_scores(parent, parent / "ensemble", meta, values)
    control_root = tmp_path / "artifacts/controls_test"
    control_root.mkdir()
    fit = control_root / "gaussian_fit.npz"
    np.savez(fit, residual_mean=np.zeros(72), covariance=np.eye(72))
    for i, name in enumerate(comparison.CONTROLS):
        meta = {
            "protocol_sha256": protocol_hash, "arrays": identities["TEST"], "procedure": name,
            "training_arrays": identities["TRAIN"], "control": protocol["controls"][name],
            "n_generated": 6, "state_convention": "physical_control_standardized_float64",
            "control_fit_sha256": sha256_file(fit) if name == "gaussian" else None,
        }
        publish_scores(control_root / name, control_root / name, meta, 20 + i + pattern)
    write_json(tmp_path / "SELECTION.json", selection)
    write_json(tmp_path / "TEST_STARTED.json", {
        "phase": "TEST_STARTED", "previous_test_exposure": True, "protocol_sha256": protocol_hash,
        "selection_sha256": sha256_file(tmp_path / "SELECTION.json"),
    })
    return tmp_path, protocol_hash, months, expected


def cli(study, output):
    root, digest, _, _ = study
    return ["--study-root", str(root), "--protocol-sha256", digest, "--output", str(output)]


def test_complete_cli_provenance_pair_set_estimands_and_manual_bootstrap(study, monkeypatch):
    from prigm import run_score

    def no_rescoring(*args, **kwargs):
        raise AssertionError("bootstrap must not rescore ensembles")

    monkeypatch.setattr(run_score, "energy_score", no_rescoring)
    root, digest, months, expected = study
    output = root / "comparison.json"
    assert comparison.main(cli(study, output)) == 0
    rec = json.loads(output.read_text())
    assert rec["protocol_sha256"] == digest
    assert rec["selection_sha256"] == sha256_file(root / "SELECTION.json")
    assert rec["test_started_sha256"] == sha256_file(root / "TEST_STARTED.json")
    for path, sha in rec["input_sha256"].items():
        assert Path(path).is_absolute()
        assert sha256_file(Path(path)) == sha
    assert any(path.endswith("ensemble.npz") for path in rec["input_sha256"])
    assert str(root / "artifacts/controls_test/gaussian_fit.npz") in rec["input_sha256"]
    assert len(rec["seed_means"]) == 18
    for label, method in rec["methods"].items():
        for seed, path in method["score_files"].items():
            assert rec["seed_means"][label][seed] == pytest.approx(manual_mean(expected[path], months, [2005, 2006]))
    assert set(rec["seed_means"]["gaussian_raw"]) == {"shared"}
    assert rec["month_counts"] == {"200508": 2, "200509": 1, "200608": 1, "200609": 3}
    pairs = [frozenset((c["method"], c["baseline"])) for c in rec["comparisons"].values()]
    assert len(pairs) == len(set(pairs))
    for arm in comparison.ARMS:
        for kind in comparison.KINDS:
            label = f"{arm}_{kind}"
            for base in ("D_raw", "D_corrected", "gaussian_raw", "gaussian_corrected", "G5_raw", "G5_corrected"):
                if label != base:
                    assert frozenset((label, base)) in pairs
    assert rec["comparisons"]["O-S_raw_minus_D_raw"]["family"] == "primary_O-S_repair"
    assert rec["comparisons"]["O-S_raw_minus_I-S_raw"]["family"] == "preplanned_secondary_repair"
    assert rec["comparisons"]["I-S_raw_minus_D_raw"]["family"] == "preplanned_secondary_repair"
    within = rec["comparisons"]["D_corrected_at_raw_checkpoint_minus_D_raw"]
    selected = rec["comparisons"]["D_corrected_minus_D_raw"]
    assert within["mean_diff"] == pytest.approx(-0.5)
    assert selected["mean_diff"] == pytest.approx(1.5)
    assert within["estimand"] == "support_correction_minus_raw_at_raw_selected_weights"
    assert selected["estimand"] == "independently_validation_selected_corrected_minus_raw"
    assert rec["methods"]["D_corrected_at_raw_checkpoint"]["generations"] == {"7": 10, "13": 10, "42": 10}

    rng = np.random.Generator(np.random.PCG64(comparison.BOOTSTRAP_SEED))
    year_draws = np.array([2005, 2006])[rng.integers(2, size=(comparison.BOOTSTRAP_REPLICATES, 2))]
    # The one two-year block in this fixture always includes both years.
    rng.integers(1, size=(comparison.BOOTSTRAP_REPLICATES, 1))
    method = [expected[p] for p in rec["methods"]["O-S_raw"]["score_files"].values()]
    base = expected[rec["methods"]["gaussian_raw"]["score_files"]["shared"]]
    diff_draws, rel_draws = [], []
    for draw in year_draws:
        base_mean = manual_mean(base, months, draw)
        difference = sum(manual_mean(v, months, draw) - base_mean for v in method) / 3
        diff_draws.append(difference)
        rel_draws.append(difference / base_mean)
    pair = rec["comparisons"]["O-S_raw_minus_gaussian_raw"]
    assert pair["year_95"] == pytest.approx(np.quantile(diff_draws, [.025, .975]))
    assert pair["rel_year_95"] == pytest.approx(np.quantile(rel_draws, [.025, .975]))
    assert pair["twoyear_95"] == pytest.approx([pair["mean_diff"]] * 2)
    assert pair["baseline_fixed_models"] == 1
    assert "Not training-seed" in rec["uncertainty_scope"]
    assert "not familywise corrected" in rec["comparison_scope"]
    assert "not a restored blind confirmation" in rec["comparison_scope"]
    second = root / "comparison-second.json"
    assert comparison.main(cli(study, second)) == 0
    assert output.read_bytes() == second.read_bytes()
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        comparison.main(cli(study, output))
    assert output.read_bytes() == before


def test_real_config_bootstrap_draws_match_manual_in_both_block_schemes():
    months = np.array([y * 100 + m for y in comparison.YEARS for m in (8, 9)])
    rows = np.stack([np.arange(len(months)) + 1., np.arange(len(months)) ** 2 + 3., np.arange(len(months)) + 11.])
    weights = comparison.bootstrap_weights(months)
    assert comparison.BOOTSTRAP_REPLICATES >= 2000
    rng = np.random.Generator(np.random.PCG64(comparison.BOOTSTRAP_SEED))
    years = comparison.year_blocks()
    draws = years[rng.integers(len(years), size=(comparison.BOOTSTRAP_REPLICATES, len(years)))]
    blocks = np.array(comparison.two_year_blocks())
    block_draws = blocks[rng.integers(len(blocks), size=(comparison.BOOTSTRAP_REPLICATES, len(blocks)))].reshape(comparison.BOOTSTRAP_REPLICATES, -1)
    for scheme, year_draws in (("year", draws), ("twoyear", block_draws)):
        observed = weights[scheme] @ comparison.monthly_means(rows, months).mean(axis=0)
        for index in (0, 1, 19, comparison.BOOTSTRAP_REPLICATES - 1):
            assert observed[index] == pytest.approx(sum(manual_mean(v, months, year_draws[index]) for v in rows) / 3)


@pytest.mark.parametrize("damage", ["reordered_ids", "missing_id", "duplicate_id", "months", "n", "short_scores",
                                    "nan", "infinity", "month_balanced", "mean", "split", "population", "checkpoint",
                                    "correction", "commit", "protocol", "procedure"])
def test_bad_score_records_fail_closed(study, damage):
    root, digest, _, _ = study
    path = root / "artifacts/test/D_s7-g10/score-raw.json"
    row = json.loads(path.read_text())
    if damage == "reordered_ids":
        row["ids"] = row["ids"][::-1]
    elif damage == "missing_id":
        row["ids"].pop()
    elif damage == "duplicate_id":
        row["ids"][1] = row["ids"][0]
    elif damage == "months":
        row["months"][0] = 200509
    elif damage == "n":
        row["n"] -= 1
    elif damage == "short_scores":
        row["scores"].pop()
    elif damage in ("nan", "infinity"):
        row["scores"][0] = float("nan" if damage == "nan" else "inf")
    elif damage in ("month_balanced", "mean"):
        row[damage] += 1
    elif damage == "split":
        row["arrays"]["split"] = "VAL"
    elif damage == "population":
        row["arrays"]["ids_sha256"] = "0" * 64
    elif damage == "checkpoint":
        row["checkpoint_sha256"] = "0" * 64
    elif damage == "correction":
        row["corrected"] = True
    elif damage == "commit":
        row["ensemble_commit_sha256"] = "0" * 64
    elif damage == "protocol":
        row["protocol_sha256"] = "0" * 64
    elif damage == "procedure":
        row["procedure"] = "gaussian"
    path.write_text(json.dumps(row))  # Permit nonfinite JSON solely for rejection tests.
    output = root / "must-not-exist.json"
    with pytest.raises(ContractMismatch):
        comparison.main(cli(study, output))
    assert not output.exists()


@pytest.mark.parametrize("damage", ["missing_seed", "duplicate_seed", "duplicate_job", "extra_job", "duplicate_checkpoint", "changed_receipt"])
def test_selection_and_exposure_receipt_fail_closed(study, damage):
    root, digest, _, _ = study
    path = root / "SELECTION.json"
    row = json.loads(path.read_text())
    if damage == "missing_seed":
        del row["selected"]["O-S_s42"]
    elif damage == "duplicate_seed":
        row["selected"]["O-S_s7_again"] = row["selected"]["O-S_s7"]
    elif damage == "duplicate_job":
        row["test_jobs"].append(row["test_jobs"][0])
    elif damage == "extra_job":
        row["test_jobs"].append(["D_s7", 30])
    elif damage == "duplicate_checkpoint":
        row["test_checkpoint_hashes"].append(row["test_checkpoint_hashes"][0])
    elif damage == "changed_receipt":
        row["selected"]["O-S_s42"]["raw"]["score"] += 1
    write_json(path, row)
    if damage != "changed_receipt":
        phase_path = root / "TEST_STARTED.json"
        phase = json.loads(phase_path.read_text())
        phase["selection_sha256"] = sha256_file(path)
        write_json(phase_path, phase)
    error = PublicationError if damage == "changed_receipt" else ContractMismatch
    with pytest.raises(error):
        comparison.load_study(root, digest)


def test_duplicate_json_selection_key_is_not_silently_overwritten(study):
    root, digest, _, _ = study
    path = root / "SELECTION.json"
    path.write_text(path.read_text().replace('"selected": {', '"selected": {}, "selected": {', 1))
    phase_path = root / "TEST_STARTED.json"
    phase = json.loads(phase_path.read_text())
    phase["selection_sha256"] = sha256_file(path)
    write_json(phase_path, phase)
    with pytest.raises(ContractMismatch, match="duplicate JSON key"):
        comparison.load_study(root, digest)


@pytest.mark.parametrize("damage", ["arm", "seed", "generation_updates", "checkpoint_sha256", "sampler", "control", "training_arrays"])
def test_ensemble_metadata_must_match_selected_path_and_control_procedure(study, damage):
    root, digest, _, _ = study
    control = damage in ("control", "training_arrays")
    parent = root / ("artifacts/controls_test/gaussian" if control else "artifacts/test/D_s7-g10")
    ensemble = parent if control else parent / "ensemble"
    commit = ensemble / "gen_00000000/COMMIT.json"
    meta = json.loads(commit.read_text())
    replacements = {"arm": "I-S", "seed": 13, "generation_updates": 20,
                    "checkpoint_sha256": "0" * 64, "sampler": {}, "control": {}, "training_arrays": {}}
    meta[damage] = replacements[damage]
    write_json(commit, meta)
    for kind in comparison.KINDS:
        path = parent / f"score-{kind}.json"
        row = json.loads(path.read_text())
        row["ensemble_commit_sha256"] = sha256_file(commit)
        write_json(path, row)
    with pytest.raises(ContractMismatch):
        comparison.load_study(root, digest)


@pytest.mark.parametrize("relative", ["artifacts/test/D_s7-g10/ensemble/gen_00000000/ensemble.npz",
                                      "artifacts/controls_test/gaussian_fit.npz", "arrays/train.npz", "arrays/test.npz"])
def test_changed_payload_bytes_are_rejected(study, relative):
    root, digest, _, _ = study
    path = root / relative
    path.write_bytes(path.read_bytes() + b"changed")
    error = PublicationError if relative.endswith("ensemble.npz") else ContractMismatch
    with pytest.raises(error, match="hash"):
        comparison.load_study(root, digest)


def test_missing_same_checkpoint_corrected_score_is_not_substituted(study):
    root, digest, _, _ = study
    (root / "artifacts/test/D_s7-g10/score-corrected.json").unlink()
    with pytest.raises(FileNotFoundError):
        comparison.load_study(root, digest)


def test_source_and_required_test_barrier_are_checked(study):
    root, digest, _, _ = study
    (root / "TEST_STARTED.json").unlink()
    with pytest.raises(PublicationError, match="exposure receipt"):
        comparison.load_study(root, digest)
    protocol_path = root / "PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["inference_source_hashes"]["prigm/run_bootstrap.py"] = "0" * 64
    write_json(protocol_path, protocol)
    with pytest.raises(ContractMismatch, match="source changed"):
        comparison.load_study(root, sha256_file(protocol_path))


@pytest.mark.parametrize("argv", [[], ["--study-root", "missing", "--output", "unused"],
                                  ["--scores-dir", "old", "--selection", "old", "--output", "unused"]])
def test_only_protocol_bound_cli_is_accepted(argv):
    with pytest.raises(SystemExit):
        comparison.main(argv)
