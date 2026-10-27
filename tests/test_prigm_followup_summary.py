"""Synthetic local fixtures, not scientific follow-up results."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from analysis import prigm_followup_summary as summary
from prigm.artifacts import json_bytes, load_arrays
from prigm.recovery import ContractMismatch, sha256_bytes, sha256_file


def manual_mean(values, months, drawn_years):
    """Average represented months independently, retaining resampled years.

    For each occurrence of a drawn year, calculate a mean for every month
    present in that year. Average the resulting month means with equal weight.
    Repeated years repeat their month contributions; absent months contribute
    nothing. This explicit loop provides a check on the vectorized calculation.
    """
    monthly = []
    for year in drawn_years:
        for month in sorted(set(m for m in months if m // 100 == year)):
            selected = [v for v, m in zip(values, months) if m == month]
            monthly.append(sum(selected) / len(selected))
    return sum(monthly) / len(monthly)


def matrices(n):
    data = {f"{arm}_{kind}": np.tile(np.arange(n, dtype=float), (3, 1))
            for arm in ("D", "I-S", "O-S", "G5")
            for kind in ("raw", "corrected", "corrected_at_raw_checkpoint")}
    data.update({f"{name}_{kind}": np.arange(n, dtype=float)[None, :]
                 for name in ("interpolation", "persistence", "gaussian")
                 for kind in ("raw", "corrected")})
    return data


def test_unequal_month_scalar_vector_tensor_reductions_are_manual():
    months = [200508, 200508, 200509, 200608]
    values = np.array([1., 3., 10., 20.])
    # Month means are 2, 10, 20; the window mean is 8.5.
    for shape in ((), (2,), (2, 3)):
        expanded = values.reshape((4,) + (1,) * len(shape)) * np.ones((1,) + shape)
        result = summary.reductions(expanded, months)
        np.testing.assert_allclose(result["month_balanced"], np.full(shape, 32 / 3), rtol=1e-15)
        np.testing.assert_allclose(result["mean"], np.full(shape, 8.5), rtol=1e-15)


def test_paired_year_ci_preserves_occurrences_distinct_months_and_shared_control():
    months = np.array([200508, 200508, 200509, 200608, 200708, 200708])
    values = np.array([1., 3., 10., 20., 27., 33.])
    data = matrices(len(months))
    data["O-S_raw"] = np.stack([values, values * 2, values + 9])
    data["D_raw"] = np.stack([values + 5, values + 7, values + 15])
    data["G5_raw"] = np.stack([values ** 2, values ** 2 + 4, values ** 2 + 9])
    data["gaussian_raw"] = (values * 3 + 1)[None, :]
    data["interpolation_raw"] = (values / 2)[None, :]
    seed, replicates = 43, 109
    actual = summary.paired_year_summary(data, months, seed=seed, replicates=replicates)
    rng = np.random.Generator(np.random.PCG64(seed))
    years = np.array([2005, 2006, 2007])
    draws = years[rng.integers(3, size=(replicates, 3))]
    assert any(len(set(draw)) < 3 for draw in draws)
    for method, base in summary.PAIRS:
        boot = []
        for draw in draws:
            left = sum(manual_mean(v, months, draw) for v in data[method]) / 3
            right = sum(manual_mean(v, months, draw) for v in data[base]) / len(data[base])
            boot.append(left - right)
        pair = actual[f"{method}_minus_{base}"]
        assert pair["year_95"] == pytest.approx(np.quantile(boot, [.025, .975]), rel=1e-13)
        expected = [manual_mean(v, months, years) for v in data[method]]
        baseline = [manual_mean(v, months, years) for v in data[base]]
        diff = np.asarray(expected) - np.asarray(baseline)
        assert list(pair["seed_diffs"].values()) == pytest.approx(diff)
        assert pair["mean_diff"] == pytest.approx(sum(diff) / 3)
    assert actual["O-S_raw_minus_gaussian_raw"]["baseline_realizations"] == 1
    weights = summary.bootstrap.month_occurrence_weights(months, [[2005, 2005, 2006]])
    # Four month occurrences from 2005 and one from 2006, not 2/3 and 1/3 annual weights.
    np.testing.assert_allclose(weights, [[2 / 5, 2 / 5, 1 / 5, 0]])
    assert manual_mean(values, months, [2005, 2005, 2006]) == pytest.approx(44 / 5)
    with pytest.raises(ContractMismatch, match="no represented months"):
        summary.bootstrap.month_occurrence_weights(months, [[2004, 2004, 2004]])


def test_single_year_has_no_uncertainty_and_no_bootstrap_draws(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("one-year uncertainty must not be manufactured")

    monkeypatch.setattr(summary.bootstrap, "month_occurrence_weights", forbidden)
    comparisons = summary.paired_year_summary(matrices(3), [200508, 200508, 200509])
    for row in comparisons.values():
        assert row["year_95"] is None
        assert row["ci_status"] == "unavailable_one_represented_year"
        assert row["replicates_used"] == 0


@pytest.mark.parametrize("damage", ["seed_missing", "control_tripled", "window_missing", "method_missing", "nan"])
def test_actual_480_window_matrix_contract(damage):
    data = matrices(480)
    ids = [f"2005-08-{i}-s52" for i in range(480)]
    months = np.full(480, 200508)
    summary.validate_matrices(data, months, ids)
    if damage == "seed_missing":
        data["D_raw"] = data["D_raw"][:2]
    elif damage == "control_tripled":
        data["gaussian_raw"] = np.repeat(data["gaussian_raw"], 3, axis=0)
    elif damage == "window_missing":
        data["G5_raw"] = data["G5_raw"][:, :-1]
    elif damage == "method_missing":
        del data["O-S_corrected_at_raw_checkpoint"]
    else:
        data["D_raw"][0, 0] = np.nan
    with pytest.raises(ContractMismatch):
        summary.validate_matrices(data, months, ids)


def test_source_parsing_uses_terminal_documented_codes():
    assert summary.source_codes([f"2005-08-x-s{code}" for code in (52, 71, 50, 60)]).tolist() == [52, 71, 50, 60]
    assert summary.SOURCE_LABELS[52] == "Wind"
    assert summary.SOURCE_LABELS[71] == "ACE"
    assert summary.SOURCE_LABELS[50] == "IMF source 50"
    assert summary.SOURCE_LABELS[60] == "IMF source 60"
    for wid in ("2005-08-s52-extra", "2005-08-52", "2005-08-s99", "2005-08-s71/",
                "2005-08-s53", "2005-08-s52\n", "2005-08-s052"):
        with pytest.raises(ContractMismatch, match="terminal IMF source"):
            summary.source_codes([wid])


def test_tertile_ties_and_empty_bins_are_not_forced_equal():
    bins, rec = summary.tertiles([0., 0., 0., 0., 1., 2.])
    assert rec["edges"] == pytest.approx([0, 1 / 3])
    assert bins.tolist() == [1, 1, 1, 1, 2, 2]
    assert rec["counts"] == [0, 4, 2]
    assert rec["edge_tie_counts"] == [4, 0]
    assert rec["windows_in_tied_values"] == 4
    bins, rec = summary.tertiles([7.] * 480)
    assert rec["counts"] == [0, 0, 480]
    assert rec["edge_tie_counts"] == [480, 480]
    assert bins.min() == 2


def make_secondary(primary):
    months = primary["months"]
    records = []
    for i, value in enumerate(primary["scores"]):
        record = {key: float(value + j / 100) for j, key in enumerate(summary.SCALARS)}
        record.update({key: (np.arange(60).reshape(12, 5) / 10 + value).tolist()
                       for key in summary.POSITION_CHANNEL})
        record["extra_vector"] = [value, value * 2]
        records.append(record)
    row = {key: copy.deepcopy(primary[key]) for key in summary.SECONDARY_IDENTITY}
    row.update(correct=primary["corrected"], interval_protocol=summary.INTERVAL_PROTOCOL, per_window=records)
    years = sorted(set(m // 100 for m in months))
    for key in records[0]:
        values = [np.asarray(r[key]) for r in records]
        row[key] = {"month_balanced": np.asarray(manual_mean(values, months, years)).tolist(),
                    "mean": np.asarray(sum(values) / len(values)).tolist()}
    return row


@pytest.fixture
def diagnostic_pair():
    primary = {
        "schema_version": 2, "n": 4,
        "ids": ["2005-08-a-s52", "2005-08-b-s52", "2005-09-a-s71", "2006-08-a-s71"],
        "months": [200508, 200508, 200509, 200608], "protocol_sha256": "a" * 64,
        "procedure": "diffusion", "checkpoint_sha256": "b" * 64, "control_fit_sha256": None,
        "arrays": {"sha256": "c" * 64}, "ensemble_commit_sha256": "d" * 64,
        "corrected": False, "scores": [1., 3., 10., 20.],
    }
    return primary, make_secondary(primary)


def test_all_secondary_reductions_including_arrays_match_independent_reference(diagnostic_pair):
    primary, secondary = diagnostic_pair
    result = summary.validate_secondary(secondary, primary, primary["scores"], primary["months"])
    assert set(result) == set(summary.SCALARS + summary.POSITION_CHANNEL + ("extra_vector",))
    assert result["es5"]["month_balanced"] == pytest.approx(32 / 3)
    assert result["es5"]["mean"] == 8.5
    assert np.asarray(result["bias_by_position_channel"]["mean"]).shape == (12, 5)


@pytest.mark.parametrize("damage", ["checkpoint_sha256", "control_fit_sha256", "protocol_sha256",
                                    "ensemble_commit_sha256", "procedure", "arrays", "ordered_ids", "months",
                                    "schema_version", "n", "correct_int", "corrected_instead", "interval_protocol",
                                    "pooled_interval", "primary_data", "primary_file", "per_window_es5",
                                    "scalar_mean", "tensor_month_balanced", "vector_mean", "tensor_shape",
                                    "missing_metric", "missing_record", "nonfinite_scalar", "nonfinite_tensor",
                                    "nonfinite_report", "string_metric", "rtol"])
def test_secondary_mismatches_fail_closed(diagnostic_pair, damage):
    primary, row = diagnostic_pair
    scores = list(primary["scores"])
    if damage in ("checkpoint_sha256", "control_fit_sha256", "protocol_sha256", "ensemble_commit_sha256", "procedure"):
        row[damage] = "wrong"
    elif damage == "arrays":
        row[damage] = {}
    elif damage == "ordered_ids":
        row["ids"] = row["ids"][::-1]
    elif damage == "months":
        row["months"][0] = 200509
    elif damage in ("schema_version", "n"):
        row[damage] = 1
    elif damage == "correct_int":
        row["correct"] = 0
    elif damage == "corrected_instead":
        row["corrected"] = row.pop("correct")
    elif damage == "interval_protocol":
        row[damage] = "seedwise_39_draws_inclusive_order_statistics_1_39_and_10_30"
    elif damage == "pooled_interval":
        row["interval_protocol"] = "pooled_117_draws"
    elif damage == "primary_data":
        scores[0] += 1
    elif damage == "primary_file":
        primary["scores"][0] += 1
    elif damage == "per_window_es5":
        row["per_window"][0]["es5"] += 1
    elif damage == "scalar_mean":
        row["es6"]["mean"] += 1
    elif damage == "tensor_month_balanced":
        row["bias_by_position_channel"]["month_balanced"][0][0] += 1
    elif damage == "vector_mean":
        row["extra_vector"]["mean"][1] += 1
    elif damage == "tensor_shape":
        row["per_window"][0]["width90_by_position_channel"] = [0.] * 60
    elif damage == "missing_metric":
        del row["per_window"][0]["es6"]
    elif damage == "missing_record":
        row["per_window"].pop()
    elif damage == "nonfinite_scalar":
        row["per_window"][0]["r_obs_mean"] = float("nan")
    elif damage == "nonfinite_tensor":
        row["per_window"][0]["cov90_by_position_channel"][0][0] = float("inf")
    elif damage == "nonfinite_report":
        row["truth_seam_l"]["mean"] = float("nan")
    elif damage == "string_metric":
        row["per_window"][0]["es6"] = "1.0"
    else:
        row["es6"]["mean"] *= 1 + 2e-12
    with pytest.raises(ContractMismatch):
        summary.validate_secondary(row, primary, scores, primary["months"])


def test_current_secondary_generator_contract_is_supported():
    from prigm.normalize import AffineNormalizer
    from prigm.run_secondaries import summarize_diagnostics, window_diagnostics

    normalizer = AffineNormalizer(np.zeros(6), np.ones(6), 1., 48)
    truth = np.ones((48, 6))
    draws = np.ones((39, 12, 6)) * 1.1
    records = [window_diagnostics(draws, truth, normalizer)]
    primary = {"schema_version": 2, "n": 1, "ids": ["2005-08-a-s52"], "months": [200508],
               "protocol_sha256": "a" * 64, "procedure": "diffusion", "checkpoint_sha256": "b" * 64,
               "control_fit_sha256": None, "arrays": {}, "ensemble_commit_sha256": "c" * 64,
               "corrected": False, "scores": [records[0]["es5"]]}
    row = {k: primary[k] for k in summary.SECONDARY_IDENTITY}
    row.update(correct=False, interval_protocol=summary.INTERVAL_PROTOCOL, per_window=records,
               **summarize_diagnostics(records, primary["months"]))
    assert set(summary.validate_secondary(row, primary, primary["scores"], primary["months"])) == set(records[0])


@pytest.fixture
def study(tmp_path, monkeypatch):
    """Mock only the complete primary gate; secondary and array readers stay real."""
    inventory = tmp_path / "artifacts/inventory"
    inventory.mkdir(parents=True)
    ids = ["2005-08-a-s52", "2005-08-b-s52", "2005-09-a-s71", "2010-08-a-s71",
           "2016-09-a-s50", "2024-08-a-s60"]
    months = np.array([200508, 200508, 200509, 201008, 201609, 202408])
    z = np.ones((6, 48, 6))
    z[:, 18:30, 3] = np.array([0, 0, 0, 0, 1, 2])[:, None]
    z[:, 18:30, 0] = np.arange(6)[:, None] * np.arange(12)[None, :]
    identities, hashes = {}, {}
    for split in ("TRAIN", "VAL", "TEST"):
        path = inventory / f"{split.lower()}.npz"
        np.savez(path, z=z, ids=np.asarray(ids), split=split, mean=np.zeros(6), scale=np.ones(6), b_ref_sq=1., n_bins=48)
        identities[split] = load_arrays(path, split)[3]
        if split != "VAL":
            hashes[str(path)] = sha256_file(path)
    protocol = {
        "array_root": str(inventory), "arrays": identities,
        "training_data_hashes": {"arrays_sha256": identities["TRAIN"]["sha256"]},
        "inference_source_hashes": {f"prigm/{p.name}": sha256_file(p)
                                    for p in Path(summary.bootstrap.__file__).parent.glob("*.py")},
        "private_execution_path": "/home/private/training",
    }
    (tmp_path / "PROTOCOL.json").write_bytes(json_bytes(protocol))
    digest = sha256_file(tmp_path / "PROTOCOL.json")
    for name in ("PROTOCOL.json", "SELECTION.json", "TEST_STARTED.json"):
        path = tmp_path / name
        if name != "PROTOCOL.json":
            path.write_bytes(json_bytes({"fixture": True}))
        hashes[str(path)] = sha256_file(path)
    (inventory / "summary.json").write_bytes(json_bytes({
        "train_starts": 6, "train_windows_saved": 6, "train_bin_union": 48,
        "val_selected": 6, "test_selected": 6, "hashes": {str(y): "e" * 64 for y in summary.YEARS},
        "private_path": "/home/private/raw",
    }))
    data = matrices(len(ids))
    methods = {}
    for j, (label, values) in enumerate(data.items()):
        for i in range(len(values)):
            values[i] += 1 + 3 * i + j / 10
        if len(values) == 3:
            arm, variant = label.split("_", 1)
            selection_kind = "raw" if variant == "corrected_at_raw_checkpoint" else variant
            kind = "raw" if variant == "raw" else "corrected"
            generation = 10 if selection_kind == "raw" else 20
            keys = list(map(str, summary.FINAL_SEEDS))
            paths = {k: tmp_path / f"artifacts/test/{arm}_s{k}-g{generation}/score-{kind}.json" for k in keys}
            methods[label] = {"procedure": "diffusion", "selection_kind": selection_kind,
                              "generations": {k: generation for k in keys}, "scoring_kind": kind}
        else:
            name, kind = label.split("_", 1)
            keys = ["shared"]
            paths = {"shared": tmp_path / f"artifacts/controls_test/{name}/score-{kind}.json"}
            methods[label] = {"procedure": name, "scoring_kind": kind}
        methods[label]["score_files"] = {k: str(p) for k, p in paths.items()}
        for i, (k, path) in enumerate(paths.items()):
            path.parent.mkdir(parents=True, exist_ok=True)
            primary = {"schema_version": 2, "n": len(ids), "ids": ids, "months": months.tolist(),
                       "protocol_sha256": digest, "procedure": methods[label]["procedure"],
                       "checkpoint_sha256": sha256_bytes(path.parent.name.encode()) if len(values) == 3 else None,
                       "ensemble_commit_sha256": sha256_bytes(path.parent.name.encode()),
                       "control_fit_sha256": "f" * 64 if label.startswith("gaussian_") else None,
                       "arrays": identities["TEST"], "corrected": kind == "corrected", "scores": values[i].tolist()}
            path.write_bytes(json_bytes(primary))
            hashes[str(path)] = sha256_file(path)
            path.with_name(f"secondaries-{kind}.json").write_bytes(json_bytes(make_secondary(primary)))
    provenance = {"ids": ids, "months": months.tolist(), "arrays": identities["TEST"], "methods": methods,
                  "input_sha256": hashes, "selection_sha256": hashes[str(tmp_path / "SELECTION.json")],
                  "test_started_sha256": hashes[str(tmp_path / "TEST_STARTED.json")], "study_root": str(tmp_path),
                  "selected": {"private_execution_path": "/home/private/checkpoint"}}

    def load(root, protocol_hash):
        assert root == tmp_path and protocol_hash == digest
        return data, months, provenance

    monkeypatch.setattr(summary.bootstrap, "load_study", load)
    monkeypatch.setattr(summary, "BOOTSTRAP_REPLICATES", 31)
    return tmp_path, digest, data, months, provenance, z


def test_complete_cli_is_anonymous_deterministic_fresh_and_readonly(study, capsys):
    root, digest, data, months, provenance, z = study
    before = {p: sha256_file(p) for p in root.rglob("*") if p.is_file()}
    output = root / "synthetic-summary.json"
    argv = ["--study-root", str(root), "--protocol-sha256", digest, "--output", str(output)]
    assert summary.main(argv) == 0
    rec = json.loads(output.read_text())
    assert len(rec["methods"]) == 18
    assert rec["analysis_source_sha256"] == sha256_file(Path(summary.__file__))
    assert rec["protocol_sha256"] == digest
    assert rec["selection_sha256"] == provenance["selection_sha256"]
    assert str(root) not in output.read_text() and "/home/" not in output.read_text()
    assert str(root) not in capsys.readouterr().out
    for path, sha in rec["input_sha256"].items():
        assert not Path(path).is_absolute() and ".." not in Path(path).parts
        assert sha256_file(root / path) == sha
    for label, method in rec["methods"].items():
        seedmeans = [manual_mean(v, months, sorted(set(months // 100))) for v in data[label]]
        assert method["metrics"]["es5"]["month_balanced"] == pytest.approx(sum(seedmeans) / len(seedmeans))
        assert method["metrics"]["es5"]["mean"] == pytest.approx(data[label].mean())
        assert len(method["seedwise"]) == len(data[label])
        for metric in summary.POSITION_CHANNEL:
            expected = np.arange(60).reshape(12, 5) / 10 + sum(seedmeans) / len(seedmeans)
            np.testing.assert_allclose(method["metrics"][metric]["month_balanced"], expected, rtol=1e-13)
            np.testing.assert_allclose(method["metrics"][metric]["mean"],
                                       np.arange(60).reshape(12, 5) / 10 + data[label].mean(), rtol=1e-13)
    raw = rec["methods"]["D_raw"]
    same = rec["methods"]["D_corrected_at_raw_checkpoint"]
    selected = rec["methods"]["D_corrected"]
    assert same["selection_kind"] == "raw" and same["scoring_kind"] == "corrected"
    assert raw["checkpoint_sha256"] == same["checkpoint_sha256"] != selected["checkpoint_sha256"]
    assert set(rec["methods"]["gaussian_raw"]["seedwise"]) == {"shared"}
    assert "not full realizability" in rec["support_scope"]
    assert "R=0" in rec["support_scope"]
    assert "Never pool 117 draws" in rec["diagnostic_scope"]
    assert rec["bootstrap"]["replicates"] == 31
    assert rec["tertiles"]["hidden_mean_m"]["counts"] == [0, 4, 2]
    np.testing.assert_allclose(rec["tertiles"]["hidden_mean_vector_step"]["values_in_test_id_order"], np.arange(6))
    empty = rec["strata"]["hidden_mean_m_tertile_1"]
    assert empty["status"] == "empty_stratum" and empty["methods"] == {} and empty["comparisons"] == {}
    for name in ("years_2005_2009", "years_2010_2015", "years_2016_2019", "years_2020_2024"):
        assert name in rec["strata"]
    for name, stratum in rec["strata"].items():
        if not stratum["n_windows"]:
            continue
        mask = np.array([wid in stratum["ids"] for wid in provenance["ids"]])
        assert stratum["ids_sha256"] == summary.id_hash(stratum["ids"])
        assert stratum["months"] == sorted(set(months[mask]))
        assert sum(stratum["source_counts"].values()) == stratum["n_windows"]
        assert sum(stratum["year_counts"].values()) == stratum["n_windows"]
        for label, metrics in stratum["methods"].items():
            expected = sum(manual_mean(v[mask], months[mask], sorted(set(months[mask] // 100))) for v in data[label]) / len(data[label])
            assert metrics["month_balanced"] == pytest.approx(expected)
        if name.startswith("year_"):
            assert all(c["year_95"] is None for c in stratum["comparisons"].values())
    assert {p: sha256_file(p) for p in before} == before
    with pytest.raises(FileExistsError):
        summary.main(argv)
    second = root / "synthetic-summary-2.json"
    assert summary.main(argv[:-1] + [str(second)]) == 0
    assert output.read_bytes() == second.read_bytes()


@pytest.mark.parametrize("kind", ["raw", "corrected"])
def test_missing_secondaries_cannot_fall_back_to_primary(study, kind):
    root, digest, *_ = study
    (root / f"artifacts/test/D_s7-g10/secondaries-{kind}.json").unlink()
    with pytest.raises(FileNotFoundError):
        summary.summarize_study(root, digest)


def test_primary_gate_failure_stops_before_any_report(study, monkeypatch):
    root, digest, *_ = study

    def fail(*args):
        raise ContractMismatch("incomplete primary study")

    monkeypatch.setattr(summary.bootstrap, "load_study", fail)
    output = root / "not-written.json"
    with pytest.raises(ContractMismatch, match="incomplete primary"):
        summary.main(["--study-root", str(root), "--protocol-sha256", digest, "--output", str(output)])
    assert not output.exists()


@pytest.mark.parametrize("relative,timing", [
    ("artifacts/test/D_s7-g10/score-raw.json", "after_gate"),
    ("artifacts/test/D_s7-g10/secondaries-raw.json", "at_finish"),
    ("artifacts/inventory/train.npz", "at_finish"),
    ("artifacts/inventory/test.npz", "at_finish"),
    ("artifacts/inventory/summary.json", "at_finish"),
    ("PROTOCOL.json", "at_finish"), ("SELECTION.json", "at_finish"), ("TEST_STARTED.json", "at_finish"),
])
def test_hashes_bind_primary_reads_and_final_inputs(study, monkeypatch, relative, timing):
    root, digest, _, _, _, _ = study
    path = root / relative
    if timing == "after_gate":
        path.write_bytes(path.read_bytes() + b" ")
    else:
        original = summary.energy_strata

        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            path.write_bytes(path.read_bytes() + b" ")
            return result

        monkeypatch.setattr(summary, "energy_strata", mutate)
    with pytest.raises(ContractMismatch, match="input changed"):
        summary.summarize_study(root, digest)


def test_external_or_symlinked_input_paths_are_rejected(study):
    root, digest, _, _, provenance, _ = study
    provenance["input_sha256"]["/home/private/outside.npz"] = "a" * 64
    with pytest.raises(ContractMismatch, match="escapes study"):
        summary.summarize_study(root, digest)
    link = root / "escape"
    link.symlink_to(root.parent, target_is_directory=True)
    with pytest.raises(ContractMismatch, match="escapes study"):
        summary._relative(link / "anything", root)


@pytest.mark.parametrize("argv", [[], ["--study-root", "absent", "--output", "unused"]])
def test_cli_requires_pinned_protocol(argv):
    with pytest.raises(SystemExit):
        summary.main(argv)
