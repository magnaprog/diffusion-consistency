"""CPU deadline-gate regressions; never modify the public evidence or producers.

One read-only replay uses the released bundle. Adversarial cases use small pure
fixtures, except where a scientific completeness contract requires all 480 IDs.
"""

import copy
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from analysis import prigm_full_flank_gaussian as fg
from analysis import prigm_operator_mechanism as op
from experiments import verify_prigm_paper as gate
from prigm.artifacts import id_hash
from prigm.recovery import ContractMismatch
# Reuse invented old-paper evidence/PDF fixtures, not today's manuscript.
from tests.test_prigm_paper import bundle, bundle_template

ERRORS = (ValueError, KeyError, ContractMismatch)
PUBLIC = gate.PAPER / "evidence/deadline"


def save(path, record):
    path.write_text(json.dumps(record, allow_nan=False))
    return gate.digest(path)


def fake_hash(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def snapshot(root):
    return {p.relative_to(root).as_posix(): gate.digest(p)
            for p in root.rglob("*") if p.is_file()}


@pytest.fixture(scope="module")
def public_originals():
    if not (PUBLIC / "MANIFEST.json").is_file():
        pytest.skip("released public deadline evidence is not installed")
    real = gate.validate_real(gate.PAPER)
    observations = gate.validate_observations(gate.PAPER, real)
    protocol = gate.read_json(gate.PAPER / "evidence/real/protocol-summary.json")
    inputs = gate.read_json(gate.PAPER / "evidence/real/MANIFEST.json")["input_sha256"]
    core = {p.relative_to(gate.REPO).as_posix(): gate.execution_source_hash(p.relative_to(gate.REPO).as_posix())
            for p in (gate.REPO / "prigm").glob("*.py")}
    return real, observations, protocol, inputs, core


def test_released_deadline_cpu_evidence_replays_read_only(public_originals):
    real, observations, *_ = public_originals
    before = snapshot(PUBLIC)
    before_figure = snapshot(gate.PAPER / "figures/operator")
    try:
        result = gate.validate_deadline(gate.PAPER, real, observations)
        assert result is not None
        assert result["operator"]["n_windows"] == 480
        assert set(result["operator"]["methods"]) == set(gate.bootstrap.ARMS + gate.bootstrap.CONTROLS)
        assert {"raw", "corrected", "raw_r_obs_mean"} <= set(result["gaussian"]["metrics"])
        assert len(result["plotted_rows"]) == 5
        if "fixed-final.json" in gate.read_json(PUBLIC / "MANIFEST.json")["files"]:
            assert len(result["fixed_final"]["rows"]) == 12
    finally:
        assert snapshot(PUBLIC) == before
        assert snapshot(gate.PAPER / "figures/operator") == before_figure


@pytest.mark.parametrize("field", ["original_sha256", "files"])
def test_audited_fixed_final_pins_cannot_be_rebound(public_originals, monkeypatch, field):
    manifest_path = PUBLIC / "MANIFEST.json"
    original_reader, original_digest = gate.read_json, gate.digest
    manifest = original_reader(manifest_path)
    if "fixed-final.json" not in manifest["files"]:
        pytest.skip("complete fixed-final record is not installed")
    replacement = fake_hash("coherently rewritten payload")
    manifest[field]["fixed-final.json"] = replacement
    monkeypatch.setattr(gate, "read_json", lambda path: copy.deepcopy(manifest)
                        if Path(path) == manifest_path else original_reader(path))
    if field == "files":
        monkeypatch.setattr(gate, "digest", lambda path: replacement
                            if Path(path) == PUBLIC / "fixed-final.json" else original_digest(path))
    real, observations, *_ = public_originals
    with pytest.raises(ContractMismatch, match="audited fixed-final"):
        gate.validate_deadline(gate.PAPER, real, observations)


@pytest.mark.parametrize("exponent,expected", [("2", "bref2"), ("3", "bref")])
def test_pdf_squared_normalizer_is_not_a_page_number(exponent, expected):
    text = "               " + exponent + "\nx / Bref       are 0.125."
    assert expected in gate.prose_text(text)
    if exponent == "3":
        assert "bref2" not in gate.prose_text(text)


def test_math_commands_and_discretionary_hyphens_keep_their_meaning():
    source = r"Selecting corrected weights gives $\max(r-m,0)/B_{\mathrm{ref}}$ at $\lambda=0.1$, $\rho=0$."
    rendered = "Select-\ning corrected weights gives max(r - m, 0)/Bref at λ = 0.1, ρ = 0."
    anchor = gate.clause_pattern(gate.prose_text("Selecting corrected weights", source=True))
    gate.bind_clause((source, rendered), anchor, source)
    with pytest.raises(ValueError):
        gate.bind_clause((source, rendered.replace("max", "min")), anchor, source)


@pytest.fixture(scope="module")
def public_claims(public_originals):
    if shutil.which("pdftotext") is None:
        pytest.skip("PDF clause regression requires pdftotext")
    real, observations, *_ = public_originals
    evidence = gate.validate_deadline(gate.PAPER, real, observations)
    tex = (gate.PAPER / "main.tex").read_text()
    rendered = subprocess.check_output(["pdftotext", "-layout", str(gate.PAPER / "main.pdf"), "-"]).decode()
    return tex, rendered, real, evidence


def test_current_deadline_tables_and_prose_match_source_and_pdf(public_claims):
    gate.deadline_claims(*public_claims)


@pytest.mark.parametrize("scope", ["source", "PDF"])
def test_false_score_role_cannot_hide_before_valid_duplicate(public_claims, scope):
    tex, rendered, real, evidence = public_claims
    source = tex if scope == "source" else rendered
    valid = "The energy score compares predictive distributions:"
    match = re.search(r"\s+".join(re.escape(word) for word in valid.split()), source)
    assert match is not None
    changed = source[:match.start()] + "The energy score compares mean predictions. " + source[match.start():]
    with pytest.raises((ValueError, ContractMismatch)):
        gate.deadline_claims(changed if scope == "source" else tex,
                             changed if scope == "PDF" else rendered, real, evidence)


@pytest.mark.parametrize("scope", ["source", "PDF"])
@pytest.mark.parametrize("before,after", [
    ('prediction intervals cover fewer withheld values than intended', 'prediction intervals cover as many withheld values as intended'),
    ('with little change in energy score', 'with a large improvement in energy score'),
    ('The larger decrease in the subtracted term', 'The smaller decrease in the subtracted term'),
    ('0.1254', '0.2254'),
    ('-0.00287', '-0.01287'),
    ('1.9787', '1.9788'),
    ('0.7964', '0.8964'),
    ('1.7733', '1.7734'),
    ('1.5728', '1.6728'),
    ('1.0497', '1.1497'),
    ('2.1873', '2.2873'),
    ('0.01648', '0.02648'),
    ('0.34407', '0.44407'),
    ('the original coefficient gives the best mean score', 'the largest coefficient gives the best mean score'),
    ('1.9515', '1.9525'),
    ('exploratory', 'confirmatory'),
    ('involve different amounts of computation', 'involve equal amounts of computation'),
    ('score differences may reflect convergence or overfitting', 'score differences establish convergence and rule out overfitting'),
    ('slightly smaller mean distance', 'slightly larger mean distance'),
    ('a larger spread term', 'a smaller spread term'),
    ('specified in advance', 'chosen after observing TEST scores'),
    ('without choosing a checkpoint or coefficient from their scores', 'choosing a checkpoint or coefficient from their scores'),
    ('whose increase accounts for most of the mean improvement', 'whose decrease accounts for most of the mean improvement'),
    ('Further fits would be needed', 'Further fits would not be needed'),
    ('Minimum displacement is not established', 'Minimum displacement is established'),
    ('Its score difference cannot isolate the benefit of the additional context alone', 'Its score difference isolates the benefit of the additional context alone'),
])
def test_new_claims_reject_changed_numbers_and_scope(public_claims, scope, before, after):
    tex, rendered, real, evidence = public_claims
    source = tex if scope == "source" else rendered
    if scope == "PDF" and before not in source:
        before, after = before.replace("-", "−"), after.replace("-", "−")
        # PDF text can wrap a prose clause across lines.
        match = re.search(r"\s+".join(re.escape(word) for word in before.split()), source)
        if match is not None:
            before = match.group()
    assert before in source
    changed = source.replace(before, after)
    with pytest.raises((ValueError, ContractMismatch)):
        gate.deadline_claims(changed if scope == "source" else tex,
                             changed if scope == "PDF" else rendered, real, evidence)


@pytest.mark.parametrize("scope", ["source", "PDF"])
def test_score_decomposition_figure_reference_is_bound(public_claims, scope):
    tex, rendered, real, evidence = public_claims
    if scope == "source":
        before = r"improvement in Figure~\ref{fig:calibration}(a)"
        after = r"improvement in Figure~\ref{fig:checkpoints}(a)"
        assert tex.count(before) == 1
        tex = tex.replace(before, after, 1)
    else:
        labels = re.findall(r"\\label\{(fig:[^}]+)\}", gate.strip_comments(tex))
        number = labels.index("fig:calibration") + 1
        pattern = rf"(improvement\s+in\s+Figure\s+){number}(\s*\(a\))"
        rendered, count = re.subn(pattern, rf"\g<1>{number + 1}\g<2>", rendered)
        assert count == 1
    with pytest.raises(ValueError, match=scope + ": clause differs"):
        gate.deadline_claims(tex, rendered, real, evidence)


@pytest.mark.parametrize("scope", ["source", "PDF"])
def test_bad_minimum_displacement_claim_cannot_hide_behind_valid_duplicate(public_claims, scope):
    tex, rendered, real, evidence = public_claims
    original = tex if scope == "source" else rendered
    clause = "Minimum displacement is not established for this map."
    if scope == "PDF":
        matches = list(re.finditer(r"\s+".join(re.escape(word) for word in clause.split()), original))
        assert len(matches) == 1
        clause = matches[0][0]
    assert original.count(clause) == 1
    damaged, count = re.subn(r"is\s+not\s+established", "is established", clause)
    assert count == 1
    changed = original.replace(clause, damaged, 1) + "\n" + clause
    with pytest.raises(ValueError, match=scope + ": missing or duplicate clause"):
        gate.deadline_claims(changed if scope == "source" else tex,
                             changed if scope == "PDF" else rendered, real, evidence)


@pytest.mark.parametrize("reported,actual", [
    ("1", 1.), (True, 1.), (1, True), (1., 1), ([1.], 1.),
    ([[1., 2.]], [1., 2.]), ([1.], [1., 1.]),
    ([True, .5], [1., .5]), ([[True, .5]], [[1., .5]]),
    ([np.array([True]), np.array([.5])], [[1.], [.5]]),
    (["1", "2"], [1., 2.]), ([float("nan")], [1.]),
    ([float("inf")], [1.]), ({"x": 1., "extra": 0}, {"x": 1.}),
    ({"x": 1.}, {"x": 1., "y": 0}), ([{"x": 1.}], [{"x": 1.}, {"x": 2.}]),
])
def test_deadline_close_refuses_types_shapes_and_nonfinite(reported, actual):
    with pytest.raises(ERRORS):
        gate.deadline_close(reported, actual)


def test_deadline_close_accepts_numeric_roundoff_but_not_proxy_scale_loss():
    gate.deadline_close({"x": [1. + 1e-11, 2.], "flag": True}, {"x": [1., 2.], "flag": True})
    expected = {"nested": [{"resolved_energy_proxy_crps": {"mean": 1e-12, "month_balanced": 2e-12}}]}
    gate.deadline_close(copy.deepcopy(expected), expected)
    for value in (0., 1.5e-12, -1e-12):
        changed = copy.deepcopy(expected)
        changed["nested"][0]["resolved_energy_proxy_crps"]["mean"] = value
        with pytest.raises(ERRORS, match="numerical"):
            gate.deadline_close(changed, expected)


@pytest.mark.parametrize("damage", [None, "missing", "extra", "empty_directory", "lock", "latest",
                                    "partial", "file_symlink", "directory_symlink", "root_symlink"])
def test_deadline_inventory_exact_tree_without_aliases(tmp_path, damage):
    root = tmp_path / "deadline"
    child = root / "gaussian/test"
    child.mkdir(parents=True)
    names = {"MANIFEST.json", "gaussian/FIT.json", "gaussian/test/SCORES.json"}
    for name in names:
        (root / name).write_bytes(b"{}")
    gate.deadline_inventory(root, names)
    if damage == "missing":
        (child / "SCORES.json").unlink()
    elif damage == "extra":
        (root / "other.json").write_bytes(b"{}")
    elif damage == "empty_directory":
        (child / "empty").mkdir()
    elif damage in ("lock", "latest", "partial"):
        (child / {"lock": ".lock", "latest": "LATEST", "partial": "SCORES.json.partial"}[damage]).touch()
    elif damage == "file_symlink":
        path = child / "SCORES.json"
        path.unlink()
        path.symlink_to(root / "MANIFEST.json")
    elif damage == "directory_symlink":
        moved = tmp_path / "moved-test"
        child.rename(moved)
        child.symlink_to(moved, target_is_directory=True)
    elif damage == "root_symlink":
        moved = tmp_path / "moved-deadline"
        root.rename(moved)
        root.symlink_to(moved, target_is_directory=True)
    if damage is None:
        gate.deadline_inventory(root, names)
    else:
        with pytest.raises(ERRORS, match="inventory|symlink|directory"):
            gate.deadline_inventory(root, names)


def reduced(rows, months):
    """Independent explicit month-first mean, also for vector/tensor metrics."""
    groups = sorted(set(months))
    result = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows])
        balanced = sum(values[np.asarray(months) == month].mean(axis=0) for month in groups) / len(groups)
        result[key] = {"mean": values.mean(axis=0).tolist(), "month_balanced": balanced.tolist()}
    return result


@pytest.fixture
def reduction_case():
    months = [200508, 200508, 200508, 200509]
    rows = [{"es5": x, "cov90_mean": .5, "vector": [x, x * 2],
             "proxy_crps": x * 1e-12} for x in (1., 1., 1., 9.)]
    shapes = {"es5": (), "cov90_mean": (), "vector": (2,), "proxy_crps": ()}
    return {"per_window": rows, "summary": reduced(rows, months)}, months, shapes


def test_deadline_reductions_weight_months_not_windows(reduction_case):
    record, months, shapes = reduction_case
    values = gate.deadline_reductions(record, months, shapes)
    np.testing.assert_array_equal(values["vector"], [[1., 2.], [1., 2.], [1., 2.], [9., 18.]])
    assert record["summary"]["es5"] == {"mean": 3., "month_balanced": 5.}
    assert record["summary"]["vector"]["month_balanced"] == [5., 10.]


@pytest.mark.parametrize("damage", ["pooled_mean", "window", "proxy", "missing_window", "extra_metric",
                                    "missing_metric", "shape", "string", "bool", "fraction_low", "fraction_high"])
def test_deadline_reductions_reject_rebound_means_and_bad_domains(reduction_case, damage):
    record, months, shapes = reduction_case
    row = record["per_window"][0]
    if damage == "pooled_mean":
        record["summary"]["es5"]["month_balanced"] = 3.
    elif damage == "window":
        row["es5"] += .01
    elif damage == "proxy":
        record["summary"]["proxy_crps"]["mean"] = 0.
    elif damage == "missing_window":
        record["per_window"].pop()
    elif damage == "extra_metric":
        row["unexpected"] = 0.
    elif damage == "missing_metric":
        row.pop("vector")
    else:
        if damage == "shape":
            row["vector"] = [[1., 2.]]
        elif damage == "string":
            row["es5"] = "1.0"
        elif damage == "bool":
            row["es5"] = True
        else:
            row["cov90_mean"] = -.01 if damage == "fraction_low" else 1.01
        if damage != "shape":
            # Valid-looking aggregate numbers must not rescue a bad window type/domain.
            if damage != "string":
                record["summary"] = reduced(record["per_window"], months)
    with pytest.raises(ERRORS):
        gate.deadline_reductions(record, months, shapes)


@pytest.fixture
def operator_case():
    from prigm.normalize import AffineNormalizer
    from prigm.observation import derive_vv
    nrm = AffineNormalizer(np.zeros(6), np.ones(6), 1., 400)
    truth = derive_vv(np.tile([1., 0., 0., 2., .5], (12, 1)))
    samples = np.repeat(truth[None], 4, axis=0)
    samples[0, :, 3] = .5
    samples[1, :, 4] = -.25
    samples[..., 5] += .125
    months = [200508, 200508, 200509]
    rows = [op.window_mechanism(samples, truth, nrm, np.float64) for _ in months]
    return {"per_window": rows, "metrics": reduced(rows, months)}, months


def test_operator_completion_is_a_noop_for_primary_score(operator_case):
    record, months = operator_case
    values = gate.validate_operator_windows(record, months)
    for term in ("target_term", "dispersion_term", "energy_score"):
        np.testing.assert_array_equal(values["raw_" + term], values["completed_" + term])
    np.testing.assert_allclose(values["delta_energy_score"],
                               values["delta_target_term"] - values["delta_dispersion_term"], atol=1e-14)
    assert (values["displacement_bound"] >= np.abs(values["delta_energy_score"])).all()


@pytest.mark.parametrize("damage", ["completion", "energy_identity", "delta_sign", "delta_component",
                                    "bound", "slack", "movement", "unordered_quantile", "severity",
                                    "infeasible", "completion_residual", "union", "rounding", "negative_magnitude"])
def test_operator_algebra_domains_and_quantiles_survive_rebound_reductions(operator_case, damage):
    record, months = operator_case
    gate.validate_operator_windows(record, months)
    row = record["per_window"][0]
    if damage == "completion":
        row["completed_target_term"] += .1
        row["completed_energy_score"] += .1
    elif damage == "energy_identity":
        row["corrected_energy_score"] += .01
        row["delta_energy_score"] += .01
    elif damage == "delta_sign":
        row["delta_energy_score"] *= -1
    elif damage == "delta_component":
        row["delta_dispersion_term"] += .1
    elif damage == "bound":
        row["mean_movement"] = 0.
        row["movement_quantiles"] = [0.] * 5
        row["displacement_bound"] = 0.
        row["bound_slack"] = -abs(row["delta_energy_score"])
    elif damage == "slack":
        row["bound_slack"] += .01
    elif damage == "movement":
        row["movement_quantiles"] = [row["mean_movement"] + 1.] * 5
    elif damage == "unordered_quantile":
        row["raw_m_minus_r_signed_quantiles"][1] = row["raw_m_minus_r_signed_quantiles"][-1] + 1.
    elif damage == "severity":
        row["raw_m_minus_r_severity_quantiles"][-1] += .25
    elif damage == "infeasible":
        row["corrected_support_beyond_rounding_fraction"] = .01
    elif damage == "completion_residual":
        row["completed_absolute_equality_residual_mean"] = 1e-15
    elif damage == "union":
        row["raw_support_violation_fraction"] = 1.
    elif damage == "rounding":
        row["raw_m_minus_r_beyond_rounding_fraction"] = 1.
    else:
        row["raw_target_term"] = -1.
    record["metrics"] = reduced(record["per_window"], months)
    with pytest.raises(ERRORS):
        gate.validate_operator_windows(record, months)


def test_operator_disjoint_union_accepts_one_ulp_arithmetic():
    from prigm.normalize import AffineNormalizer
    from prigm.observation import derive_vv
    nrm = AffineNormalizer(np.zeros(6), np.ones(6), 1., 400)
    truth = derive_vv(np.tile([1., 0., 0., 2., .5], (12, 1)))
    samples = np.repeat(truth[None], 39, axis=0)
    # Produce 30 and 2 disjoint failures among 39*12 draw/bin pairs.
    samples.reshape(-1, 6)[:30, 3] = .5
    samples.reshape(-1, 6)[30:32, 4] = -.25
    row = op.window_mechanism(samples, truth, nrm, np.float64)
    fractions = [row[f"raw_{margin}_violation_fraction"] for margin in op.MARGINS]
    combined = row["raw_support_violation_fraction"]
    assert fractions == [30 / 468, 2 / 468, 0.]
    assert combined == 32 / 468 > sum(fractions)
    assert combined == np.nextafter(sum(fractions), np.inf)
    months = [200508]
    gate.validate_operator_windows({"per_window": [row], "metrics": reduced([row], months)}, months)


def test_operator_signed_changes_are_not_nonnegative_magnitudes(operator_case):
    record, months = operator_case
    values = gate.validate_operator_windows(record, months)
    assert (values["delta_energy_score"] < 0).all()
    assert (values["delta_target_term"] < 0).all()
    assert (values["delta_dispersion_term"] < 0).all()


def full_test_ids():
    return [f"{year}-{month:02d}-{year}-{month:02d}-{day:02d}T00:00-s52"
            for year in range(2005, 2025) for month in (8, 9) for day in range(1, 13)]


@pytest.fixture
def gaussian_fit_case(tmp_path):
    """Full-dimensional invented moments and complete VAL grid, no data fitting."""
    core = {p.relative_to(gate.REPO).as_posix(): gate.digest(p)
            for p in (gate.REPO / "prigm").glob("*.py")}
    val_ids = [str(np.datetime64("2005-11-01T00:00") + np.timedelta64(i, "m")) for i in range(318)]
    val_ids += ["2005-12-01T00:00"]
    identities = {split: {"sha256": fake_hash(split), "split": split, "n_windows": n,
                         "ids_sha256": id_hash(val_ids if split == "VAL" else full_test_ids()) if split != "TRAIN" else fake_hash("TRAIN ids"),
                         "normalization_sha256": fake_hash("normalization")}
                  for split, n in fg.EXPECTED_WINDOWS.items()}
    source = {"schema_version": 2, "exposure": "historical_test_exposed", "arrays": identities,
              "inference_source_hashes": core}
    ph = save(tmp_path / "PROTOCOL.json", source)
    contract = fg.source_contract(tmp_path, ph)
    # This fixture verifies the historical producer after a documentation edit.
    contract["producer_source_hashes"] = {
        name: gate.documentation.source_hash(gate.REPO, name)
        for name in contract["producer_source_hashes"]}
    root = tmp_path / "gaussian"
    root.mkdir()
    save(root / "EXTENSION.json", contract)
    x_mean, y_mean = np.linspace(-.5, .5, 216), np.linspace(.2, .8, 60)
    cxy = np.zeros((216, 60))
    cxy[:60] = np.eye(60) * .25
    weights = np.stack([cxy / (2 + lam) for lam in (.001, .1, 1.)])
    fit = {"n_train": np.array(identities["TRAIN"]["n_windows"], dtype=np.int64),
           "x_mean": x_mean, "y_mean": y_mean, "cxx": 2 * np.eye(216), "cxy": cxy, "cyy": np.eye(60),
           "weights": weights, "intercepts": np.stack([y_mean - x_mean @ w for w in weights]),
           "residual_covariances": np.stack([np.eye(60) - 2 * cxy.T @ w + 2 * w.T @ w for w in weights])}
    np.savez(root / "FIT.npz", **fit)
    save(root / "FIT.json", {"schema_version": 1, "training_arrays": identities["TRAIN"], "fit_payload": "FIT.npz",
                             "n_mean_fits": 3, "n_covariance_configurations": 9})
    candidates = [{"lambda": lam, "rho": rho, "scores": {"raw": [3.] * 319, "corrected": [1.] * 319}}
                  for lam, rho in reversed(fg.GRID)]
    by_pair = {(r["lambda"], r["rho"]): r for r in candidates}
    by_pair[.001, 0.]["scores"]["raw"] = [0.] * 318 + [4.]
    by_pair[.1, .5]["scores"]["raw"] = [2.2] * 318 + [0.]
    validation = {"schema_version": 1, "arrays": identities["VAL"], "ids": val_ids,
                  "months": [200511] * 318 + [200512], "candidates": candidates}
    save(root / "VALIDATION.json", validation)
    selected = fg.select_grid(validation)
    save(root / "SELECTION.json", {"schema_version": 1, "selected": selected,
                                   "selected_union": [fg.candidate_name(*p) for p in fg.selected_pairs({"selected": selected})]})
    protocol = {"source_protocol_sha256": ph, "arrays": identities}
    receipt = repin_gaussian(root, protocol)
    return root, protocol, core, receipt


def repin_gaussian(root, protocol):
    extension, fit = gate.digest(root / "EXTENSION.json"), gate.digest(root / "FIT.npz")
    for name in ("FIT.json", "VALIDATION.json", "SELECTION.json"):
        record = gate.read_json(root / name)
        record.update(extension_protocol_sha256=extension, fit_payload_sha256=fit)
        if name == "SELECTION.json":
            record["validation_sha256"] = gate.digest(root / "VALIDATION.json")
        save(root / name, record)
    return save(root / "FIT_SELECT.json", {"schema_version": 1, "phase": "FIT_SELECTED",
                "source_protocol_sha256": protocol["source_protocol_sha256"],
                "files": {name: gate.digest(root / name) for name in fg.FIT_FILES}})


def test_gaussian_stored_moments_complete_grid_and_independent_val_selection(gaussian_fit_case):
    contract, fit, selection, _ = gate.validate_gaussian_fit(*gaussian_fit_case)
    assert fit["weights"].shape == (3, 216, 60)
    assert contract["noise"]["n_draws"] == 39
    assert selection["selected"]["raw"]["lambda"] == .1
    assert selection["selected"]["raw"]["rho"] == .5
    assert selection["selected"]["raw"]["month_balanced_primary_es"] == pytest.approx(1.1)
    assert selection["selected"]["corrected"] == {"lambda": .001, "rho": 0., "month_balanced_primary_es": 1.}


@pytest.mark.parametrize("damage", ["weights", "intercept", "residual_covariance", "n_train", "dtype", "shape",
                                    "non_psd", "asymmetric", "grid_missing", "grid_duplicate", "val_ids", "val_months",
                                    "val_score_shape", "val_score_type", "selected", "pooled_selection", "noise", "fit_metadata", "hash"])
def test_gaussian_fit_and_val_mutations_rejected_after_repinning(gaussian_fit_case, damage):
    root, protocol, core, receipt = gaussian_fit_case
    gate.validate_gaussian_fit(root, protocol, core, receipt)
    if damage in {"weights", "intercept", "residual_covariance", "n_train", "dtype", "shape", "non_psd", "asymmetric"}:
        with np.load(root / "FIT.npz", allow_pickle=False) as blob:
            fit = {key: blob[key] for key in blob.files}
        if damage == "weights":
            fit["weights"][0, 0, 0] += .01
        elif damage == "intercept":
            fit["intercepts"][0, 0] += .01
        elif damage == "residual_covariance":
            fit["residual_covariances"][0, 0, 0] += .01
        elif damage == "n_train":
            fit["n_train"] += 1
        elif damage == "dtype":
            fit["cxx"] = fit["cxx"].astype(np.float32)
        elif damage == "shape":
            fit["x_mean"] = fit["x_mean"][None]
        elif damage == "non_psd":
            fit["cyy"][0, 0] = -1.
        else:
            fit["cxx"][0, 1] = .1
        np.savez(root / "FIT.npz", **fit)
    else:
        filename = ("VALIDATION.json" if damage.startswith(("grid", "val_")) else
                    "SELECTION.json" if damage in ("selected", "pooled_selection") else
                    "EXTENSION.json" if damage == "noise" else "FIT.json")
        record = gate.read_json(root / filename)
        if damage == "grid_missing":
            record["candidates"].pop()
        elif damage == "grid_duplicate":
            record["candidates"][0] = copy.deepcopy(record["candidates"][1])
        elif damage == "val_ids":
            record["ids"][0], record["ids"][1] = record["ids"][1], record["ids"][0]
        elif damage == "val_months":
            record["months"][-1] = 200511
        elif damage == "val_score_shape":
            record["candidates"][0]["scores"]["raw"].pop()
        elif damage == "val_score_type":
            record["candidates"][0]["scores"]["raw"][0] = "3.0"
        elif damage == "selected":
            record["selected"]["raw"]["lambda"] = 1.
        elif damage == "pooled_selection":
            record["selected"]["raw"] = {"lambda": .001, "rho": 0., "month_balanced_primary_es": 4 / 319}
            record["selected_union"] = [fg.candidate_name(.001, 0.)]
        elif damage == "noise":
            record["noise"]["seed"] += 1
        else:
            record["n_covariance_configurations"] = 3
        save(root / filename, record)
    if damage != "hash":
        receipt = repin_gaussian(root, protocol)
    with pytest.raises(ERRORS):
        gate.validate_gaussian_fit(root, protocol, core, receipt)


@pytest.fixture
def gaussian_test_case(tmp_path, monkeypatch):
    """Isolate the TEST interface from fit IO and score computation.

    All 480 IDs and 39 draws are retained. Identical invented observations and
    zero innovations allow a broadcast view instead of another 43 MB payload.
    The released replay above supplies real RNG, payload, and score coverage.
    """
    from prigm.normalize import AffineNormalizer
    from prigm.observation import derive_vv
    nrm = AffineNormalizer(np.zeros(6), np.ones(6), 1., 400)
    ids = full_test_ids()
    truth = derive_vv(np.tile([1., 0., 0., 2., .5], (48, 1)))
    z = np.broadcast_to(truth, (480, 48, 6))
    identity = {"sha256": fake_hash("TEST"), "ids_sha256": id_hash(ids), "split": "TEST",
                "n_windows": 480, "normalization_sha256": fake_hash("normalization")}
    protocol = {"source_protocol_sha256": gate.DEADLINE_PROTOCOL, "arrays": {"TEST": identity, "TRAIN": {}}}
    contract = {"source_protocol_sha256": gate.DEADLINE_PROTOCOL, "arrays": protocol["arrays"],
                "producer_source_hashes": {}, "core_source_hashes": {}, "interval": "synthetic inclusive interval"}
    hashes = {name: fake_hash(name) for name in fg.FIT_FILES}
    receipt = fake_hash("fit selection")
    selected = {kind: {"lambda": .1, "rho": 0., "month_balanced_primary_es": .25}
                for kind in ("raw", "corrected")}
    name = fg.candidate_name(.1, 0.)
    selection = {"selected": selected, "selected_union": [name]}
    fit = {"weights": np.zeros((3, 216, 60)), "intercepts": np.zeros((3, 60)),
           "residual_covariances": np.zeros((3, 60, 60))}
    monkeypatch.setattr(gate, "validate_gaussian_fit", lambda *args: (contract, fit, selection, hashes))
    monkeypatch.setattr(fg, "standard_noise", lambda split, wid: np.zeros((39, 60)))
    single = fg.draw_window(truth, nrm, fit["weights"][1], fit["intercepts"][1],
                            fg.candidate_factors(fit)[.1, 0.], np.zeros((39, 60)))
    meta = {**fg.ensemble_metadata(contract, hashes, selection, receipt, (.1, 0.)),
            "schema_version": 2, "ids_sha256": id_hash(ids), "generation": 0,
            "files": {"ensemble.npz": fake_hash("synthetic ensemble")}}
    state = {"single": single, "meta": meta, "ids": ids.copy()}
    monkeypatch.setattr(fg, "read_ensemble", lambda path:
                        (np.broadcast_to(state["single"], (480, 39, 6, 48)), state["ids"], state["meta"]))
    root = tmp_path / "gaussian"
    payload = root / f"test/{name}/gen_00000000"
    payload.mkdir(parents=True)
    # Only the gate's NPZ field inventory is read here; read_ensemble is the mock IO boundary.
    np.savez(payload / "ensemble.npz", ids=np.array([], dtype="U1"), ensemble=np.zeros(0))
    save(payload / "COMMIT.json", meta)
    phase = {"schema_version": 1, "phase": "TEST_STARTED", "previous_test_exposure": True,
             "source_protocol_sha256": gate.DEADLINE_PROTOCOL, "extension_protocol_sha256": hashes["EXTENSION.json"],
             "fit_select_sha256": receipt, "fit_payload_sha256": hashes["FIT.npz"],
             "selection_sha256": hashes["SELECTION.json"], "arrays": identity, "selected_union": [name]}
    save(root / "test/TEST_STARTED.json", phase)
    months = fg.months_for(ids).tolist()
    metrics = {"primary_es": .25, "coverage90": .5, "width90": .2, "vector_energy": .1,
               "resolved_energy_proxy_crps": 1e-12, "support_violation_fraction": 0.,
               "support_violation_beyond_rounding_fraction": 0.}
    rows = [dict(metrics) for _ in ids]
    calculated = {kind: {"per_window": copy.deepcopy(rows), "summary": reduced(rows, months)}
                  for kind in ("raw", "corrected")}
    movement = [{"mean_primary_l2": 0.} for _ in ids]
    calculated["correction_displacement"] = {"per_window": movement, "summary": reduced(movement, months)}
    monkeypatch.setattr(fg, "test_scores", lambda *args: copy.deepcopy(calculated))
    commit_name = f"test/{name}/gen_00000000/COMMIT.json"
    candidate = {"lambda": .1, "rho": 0., "selected_for": ["raw", "corrected"], "ensemble": f"test/{name}",
                 "ensemble_commit_sha256": gate.digest(root / commit_name), **copy.deepcopy(calculated)}
    scores = {"schema_version": 1, "scope": fg.SCOPE, "fit_select_sha256": receipt,
              "source_protocol_sha256": gate.DEADLINE_PROTOCOL, "extension_protocol_sha256": hashes["EXTENSION.json"],
              "selection_sha256": hashes["SELECTION.json"], "fit_payload_sha256": hashes["FIT.npz"], "arrays": identity,
              "ids": ids, "months": months, "n": 480, "n_draws": 39, "complete_case": True,
              "selected": selected, "interval": contract["interval"], "candidates": {name: candidate}}
    save(root / "test/SCORES.json", scores)
    save(root / "test/COMPLETED.json", {"schema_version": 1, "phase": "TEST_COMPLETED", "fit_select_sha256": receipt,
                "n_windows": 480, "n_draws": 39, "selected_union": [name],
                "files": {p: gate.digest(root / p) for p in ("test/TEST_STARTED.json", "test/SCORES.json", commit_name)}})
    real = ({}, {"methods": {f"procedure_{i}": {"metrics": {k: {"month_balanced": 2.}
                                for k in ("es5", "es_vec", "proxy_crps")}} for i in range(18)}})
    args = (root, protocol, {}, (z, ids, nrm, identity), real, receipt)
    return args, state, name


def test_gaussian_test_interface_preserves_population_and_ranks(gaussian_test_case):
    args, state, _ = gaussian_test_case
    result = gate.validate_gaussian(*args)
    assert len(state["ids"]) == 480
    assert state["single"].shape == (39, 6, 48)
    assert result["metrics"]["raw_r_obs_mean"]["mean"] == 0.
    assert result["ranks"]["raw"]["primary_es"]["rank_among_original_18_plus_selected"] == 1
    assert len(result["ranks"]["raw"]["primary_es"]["original_18"]) == 18


@pytest.mark.parametrize("damage,match", [
    ("draw_metadata", "metadata"), ("sampler_seed", "metadata"), ("fit_identity", "metadata"),
    ("draw", "regenerated"), ("noise", "regenerated"), ("flank", "flanks"),
    ("derived_vv", "derived v_v"), ("ids", "ordering"),
    ("stored_moment", "all TEST metrics"), ("proxy", "all TEST metrics"),
    ("test_draws", "TEST metadata"), ("complete_case", "TEST metadata"),
])
def test_gaussian_test_draw_metadata_and_stored_metrics(gaussian_test_case, monkeypatch, damage, match):
    args, state, name = gaussian_test_case
    if damage == "draw_metadata":
        state["meta"]["noise"] = {**state["meta"]["noise"], "n_draws": 38}
    elif damage == "sampler_seed":
        state["meta"]["noise"] = {**state["meta"]["noise"], "seed": 7}
    elif damage == "fit_identity":
        state["meta"]["control_fit_sha256"] = fake_hash("wrong fit")
    elif damage == "noise":
        monkeypatch.setattr(fg, "standard_noise", lambda split, wid: np.full((39, 60), .1))
    elif damage in ("draw", "flank", "derived_vv"):
        state["single"] = state["single"].copy()
        if damage == "flank":
            state["single"][0, 0, 0] += .01
        elif damage == "derived_vv":
            state["single"][0, 5, 18] += .01
        else:
            # Keep v_v consistent so regeneration, not the equality guard, catches the change.
            from prigm.observation import derive_vv
            state["single"][0, 0, 18] += .01
            state["single"][0, :, 18] = derive_vv(state["single"][0, :5, 18])
    elif damage == "ids":
        state["ids"][0], state["ids"][1] = state["ids"][1], state["ids"][0]
    else:
        path = args[0] / "test/SCORES.json"
        record = gate.read_json(path)
        if damage == "test_draws":
            record["n_draws"] = 38
        elif damage == "complete_case":
            record["complete_case"] = 1
        else:
            metric = "primary_es" if damage == "stored_moment" else "resolved_energy_proxy_crps"
            record["candidates"][name]["raw"]["summary"][metric]["mean"] = 0.
        save(path, record)
    with pytest.raises(ERRORS, match=match):
        gate.validate_gaussian(*args)


@pytest.mark.parametrize("flag", [False, 0, 1, "true", None])
def test_fixed_final_complete_flag_is_strict_even_before_rows(flag):
    ids = full_test_ids()
    identity = {"split": "TEST", "n_windows": 480}
    record = {"schema_version": 2, "protocol_sha256": gate.FIXED_FINAL_PROTOCOL,
              "selection": gate.FIXED_SELECTION, "arrays": identity, "ids": ids,
              "months": fg.months_for(ids).tolist(), "complete_grid": flag}
    with pytest.raises(ERRORS, match="complete_grid"):
        gate.validate_fixed_final(record, None, (None, ids, None, identity), None, None, None)


@pytest.mark.parametrize("damage", ["missing_seed", "duplicate_seed", "reordered_rows"])
def test_fixed_final_requires_twelve_ordered_rows_before_aggregating(damage):
    ids = full_test_ids()
    identity = {"split": "TEST", "n_windows": 480}
    rows = [{"id": f"c{c}_s{s}"} for c in ("0", "0.00005", "0.0005", "0.005") for s in (7, 13, 42)]
    if damage == "missing_seed":
        rows.pop()
    elif damage == "duplicate_seed":
        rows[-1] = dict(rows[0])
    else:
        rows[0], rows[1] = rows[1], rows[0]
    record = {"schema_version": 2, "protocol_sha256": gate.FIXED_FINAL_PROTOCOL,
              "selection": gate.FIXED_SELECTION, "arrays": identity, "ids": ids,
              "months": fg.months_for(ids).tolist(), "complete_grid": True, "rows": rows,
              "selection_sha256": fake_hash("selection"),
              "producers": {name: gate.documentation.source_hash(gate.REPO, name) for name in
                            ("experiments/run_prigm_fixed_final_sensitivity.py", "experiments/run_prigm_repair.py")}}
    with pytest.raises(ERRORS, match="ordered complete coefficient/seed grid"):
        gate.validate_fixed_final(record, None, (None, ids, None, identity), None, None, None)


@pytest.fixture(scope="module")
def public_fixed_final(public_originals):
    # Check once, never wait for another worker or use a private fallback.
    path = PUBLIC / "fixed-final.json"
    if not path.is_file():
        pytest.skip("public fixed-final evidence has not been published")
    record = gate.read_json(path)
    gate.validate_fixed_final(record, *public_originals)
    return record


def test_fixed_final_public_complete_grid_and_final_endpoint(public_fixed_final, public_originals):
    record = public_fixed_final
    assert len(record["rows"]) == 12
    assert [r["id"] for r in record["rows"]] == [f"c{c}_s{s}" for c in ("0", "0.00005", "0.0005", "0.005") for s in (7, 13, 42)]
    assert all(r["generation"] == 29636 for r in record["rows"])
    assert {r["id"] for r in record["rows"] if r["reuse"]} == {"c0_s7", "c0.0005_s7", "c0.0005_s13"}
    gate.validate_fixed_final(record, *public_originals)


@pytest.mark.parametrize("damage,match", [
    ("missing_seed", "complete coefficient/seed"), ("duplicate_row", "complete coefficient/seed"),
    ("reordered_rows", "complete coefficient/seed"), ("endpoint", "endpoint inventory"),
    ("reuse", "job reuse"), ("reuse_flag_type", "job reuse"), ("new_fit", "job new_fit"),
    ("reuse_protocol", "identity"), ("generation", "job generation"), ("metadata_generation", "generation_updates"),
    ("missing_chunk", "chunk inventory"), ("summary", "reductions"), ("row_metric", "reductions"),
    ("infeasible", "feasibility"), ("width", "domains"),
])
def test_fixed_final_complete_reuse_and_per_window_bindings(public_fixed_final, public_originals, damage, match):
    record = copy.deepcopy(public_fixed_final)
    row = record["rows"][0]
    if damage == "missing_seed":
        record["rows"].pop()
    elif damage == "duplicate_row":
        record["rows"][-1] = copy.deepcopy(row)
    elif damage == "reordered_rows":
        record["rows"][0], record["rows"][1] = record["rows"][1], record["rows"][0]
    elif damage == "endpoint":
        record["endpoints"].pop(record["rows"][-1]["id"])
    elif damage == "reuse":
        row["reuse"] = False
    elif damage == "reuse_flag_type":
        row["reuse"] = 1
    elif damage == "new_fit":
        row["new_fit"] = True
    elif damage == "reuse_protocol":
        row["protocol_sha256"] = gate.FIXED_FINAL_PROTOCOL
    elif damage == "generation":
        row["generation"] = 22944
    elif damage == "metadata_generation":
        row["metadata"]["generation_updates"] = 22944
    elif damage == "missing_chunk":
        row["metadata"]["chunks"].pop(next(iter(row["metadata"]["chunks"])))
    else:
        variant = row["results"]["corrected" if damage == "infeasible" else "raw"]
        if damage == "summary":
            variant["summary"]["es5"]["month_balanced"] += .01
        elif damage == "row_metric":
            variant["per_window"][0]["es5"] += .01
        else:
            key = "frac_support_violation_beyond_rounding" if damage == "infeasible" else "width90_mean"
            variant["per_window"][0][key] = .01 if damage == "infeasible" else -1.
            if damage == "infeasible":
                variant["per_window"][0]["frac_support_inequality_violation"] = .01
            variant["summary"] = reduced(variant["per_window"], record["months"])
    with pytest.raises(ERRORS, match=match):
        gate.validate_fixed_final(record, *public_originals)


@pytest.fixture(scope="module")
def public_operator(public_originals):
    return gate.read_json(PUBLIC / "operator-mechanism.json")


@pytest.mark.parametrize("damage", [None, "last_window", "sign", "seed_mean", "missing_seed", "generation"])
def test_operator_selection_decomposition_binds_all_rows(public_operator, public_originals, monkeypatch, damage):
    real, observations, protocol, inputs, core = public_originals
    scores = gate.read_json(gate.PAPER / "evidence/real/scores.json")
    record = {**public_operator, "D_selection_decomposition": copy.deepcopy(public_operator["D_selection_decomposition"])}
    # Window algebra has its own tests; isolate the outer selection/source bindings.
    monkeypatch.setattr(gate, "validate_operator_windows", lambda row, months:
                        {kind + "_energy_score": np.asarray([r[kind + "_energy_score"] for r in row["per_window"]])
                         for kind in ("raw", "completed", "corrected")})
    decomposition = record["D_selection_decomposition"]
    seed = decomposition["seedwise"]["7" if damage == "sign" else "42"]
    if damage == "last_window":
        seed["per_window"][-1]["total_change"] += .01
        seed["metrics"] = reduced(seed["per_window"], real[1]["months"])
        decomposition["metrics"] = op._seed_average(decomposition["seedwise"])
    elif damage == "sign":
        assert any(row["selection_effect"] != 0 for row in seed["per_window"])
        for row in seed["per_window"]:
            row["selection_effect"] *= -1
            row["total_change"] = row["fixed_checkpoint_correction"] + row["selection_effect"]
        seed["metrics"] = reduced(seed["per_window"], real[1]["months"])
        decomposition["metrics"] = op._seed_average(decomposition["seedwise"])
    elif damage == "seed_mean":
        decomposition["metrics"]["total_change"]["month_balanced"] += .01
    elif damage == "missing_seed":
        decomposition["seedwise"].pop("13")
    elif damage == "generation":
        seed["corrected_generation"] = 22944 if seed["corrected_generation"] != 22944 else 2868
    args = (record, real, observations, protocol, inputs, core, scores)
    if damage is None:
        gate.validate_operator(*args)
    else:
        with pytest.raises(ERRORS, match="D selection"):
            gate.validate_operator(*args)


@pytest.fixture
def operator_figure_case(tmp_path, operator_case):
    from paper.workshops.prigm import fig_operator
    record, _ = operator_case
    metric = record["metrics"]
    methods = {}
    for name in (*gate.bootstrap.ARMS, *gate.bootstrap.CONTROLS):
        seeds = ("7", "13", "42") if name in gate.bootstrap.ARMS else ("shared",)
        methods[name] = {"metrics": metric, "n_realizations": len(seeds),
                         "seedwise": {s: {"metrics": metric} for s in seeds}}
    fixed = metric["delta_energy_score"]["month_balanced"]
    selection = {key: {"month_balanced": value, "mean": value} for key, value in
                 (("fixed_checkpoint_correction", fixed), ("selection_effect", .1), ("total_change", fixed + .1))}
    # Figure-only summary fixture: the full numeric evidence is validated elsewhere.
    evidence = {"schema_version": 1, "analysis": "paired_operator_mechanism", "protocol_sha256": fig_operator.PROTOCOL,
                "n_windows": 480, "ids": full_test_ids(), "methods": methods,
                "D_selection_decomposition": {"metrics": selection, "seedwise": {s: {} for s in ("7", "13", "42")}}}
    root = tmp_path / "paper"
    directory = root / "evidence/deadline"
    directory.mkdir(parents=True)
    operator_hash = save(directory / "operator-mechanism.json", evidence)
    rows, effects = fig_operator.load_rows(directory / "operator-mechanism.json", operator_hash)
    figure = root / "figures/operator"
    figure.mkdir(parents=True)
    (figure / "fig_operator.pdf").write_bytes(b"%PDF-1.4\nsynthetic figure header\n")
    (figure / "fig_operator.png").write_bytes(b"\x89PNG\r\n\x1a\nsynthetic figure header\n")
    manifest = {"schema_version": 1, "figure": "fig_operator", "protocol_sha256": fig_operator.PROTOCOL,
                "source": {"path": "fig_operator.py", "sha256": gate.digest(gate.PAPER / "fig_operator.py")},
                "inputs": {"operator": {"path": "operator-mechanism.json", "sha256": operator_hash}},
                "plotted_rows": rows, "D_selection": effects,
                "output_sha256": {name: gate.digest(figure / name) for name in ("fig_operator.pdf", "fig_operator.png")}}
    save(figure / "fig_operator.manifest.json", manifest)
    return root, figure, operator_hash


@pytest.mark.parametrize("damage", [None, "coordinate", "selection", "input", "source", "pdf", "png", "header", "output_inventory"])
def test_operator_figure_hashes_and_exact_coordinates(operator_figure_case, damage):
    root, figure, operator_hash = operator_figure_case
    before = snapshot(root)
    gate.validate_operator_figure(root, operator_hash)
    assert snapshot(root) == before
    if damage is None:
        # A supplied bundle's Python is data, never trusted executable code.
        (root / "fig_operator.py").write_text("raise AssertionError('untrusted figure executed')\n")
        gate.validate_operator_figure(root, operator_hash)
        return
    path = figure / "fig_operator.manifest.json"
    manifest = gate.read_json(path)
    if damage == "coordinate":
        manifest["plotted_rows"][0]["movement"] += 1e-14
    elif damage == "selection":
        manifest["D_selection"]["total_change"] += 1e-14
    elif damage == "input":
        manifest["inputs"]["operator"]["sha256"] = fake_hash("other operator")
    elif damage == "source":
        manifest["source"]["sha256"] = fake_hash("other source")
    elif damage in ("pdf", "png"):
        target = figure / ("fig_operator." + damage)
        target.write_bytes(target.read_bytes() + b"changed")
    elif damage == "header":
        target = figure / "fig_operator.pdf"
        target.write_bytes(b"not a PDF")
        manifest["output_sha256"][target.name] = gate.digest(target)
    else:
        manifest["output_sha256"].pop("fig_operator.png")
    save(path, manifest)
    with pytest.raises(ERRORS, match="figure"):
        gate.validate_operator_figure(root, operator_hash)


@pytest.mark.parametrize("claim", ["", "% Full-flank Gaussian\n", "Full-flank Gaussian",
                                  "% Gaussian regression\n", "Gaussian regression", "Gaussian\nregression",
                                  r"\includegraphics{fig_operator.pdf}", r"\label{tab:operators}", r"\label{tab:fixedfinal}"])
def test_deadline_claim_detection_ignores_comments_but_not_visible_claims(tmp_path, claim):
    (tmp_path / "main.tex").write_text(claim)
    claimed = bool(claim) and not claim.startswith("%")
    assert gate.deadline_claimed(tmp_path) is claimed
    if claimed:
        with pytest.raises(FileNotFoundError):
            gate.validate_deadline(tmp_path, None, None)
    else:
        assert gate.validate_deadline(tmp_path, None, None) is None


@pytest.mark.skipif(not shutil.which("pdftotext") or not shutil.which("pdfinfo"), reason="PDF tools unavailable")
@pytest.mark.parametrize("claim", ["directory", "Full-flank Gaussian", "Gaussian regression", r"\label{tab:fixedfinal}"])
@pytest.mark.parametrize("entry", ["verify", "main"])
def test_claimed_extension_is_mandatory_through_verify_report_and_cli(bundle, monkeypatch, capsys, claim, entry):
    root = bundle["root"]
    if claim == "directory":
        (root / "evidence/deadline").mkdir()
    else:
        path = root / "main.tex"
        path.write_text(path.read_text() + "\n" + claim + "\n")
    calls = []

    def refuse(paper_dir, real, observations):
        assert Path(paper_dir) == root
        assert real is not None and observations is not None
        calls.append(paper_dir)
        raise ValueError("claimed deadline extension sentinel")

    monkeypatch.setattr(gate, "validate_deadline", refuse)
    before = snapshot(root)
    if entry == "verify":
        report = gate.verify(root)
        assert not report.ok()
        assert any(not ok and "claimed deadline extension sentinel" in detail for _, ok, detail in report.rows)
    else:
        assert gate.main(["--paper-dir", str(root)]) == 1
        output = capsys.readouterr().out
        assert "OVERALL: FAIL" in output and "claimed deadline extension sentinel" in output
    assert len(calls) == 1, "verify/CLI must invoke the deadline gate, not merely reject an unrelated table"
    assert snapshot(root) == before


@pytest.mark.skipif(not shutil.which("pdftotext") or not shutil.which("pdfinfo"), reason="PDF tools unavailable")
def test_unclaimed_old_bundle_remains_valid_through_cli(bundle, capsys):
    assert not gate.deadline_claimed(bundle["root"])
    before = snapshot(bundle["root"])
    assert gate.main(["--paper-dir", str(bundle["root"])]) == 0
    assert "OVERALL: PASS" in capsys.readouterr().out
    assert snapshot(bundle["root"]) == before
