"""Independent checkpoint reductions and public provenance perturbations."""

import copy
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from experiments import _prigm_checkpoint_evidence as c
from experiments import _prigm_camera_ready as camera
from experiments import verify_prigm_paper as g
from prigm.recovery import ContractMismatch


PAPER = Path(__file__).resolve().parents[1] / "paper/workshops/prigm"
PUBLIC = Path("evidence/followup/checkpoints")
ERRORS = (ValueError, ContractMismatch)


def population(split):
    ids, months = [], []
    for i in range(40):
        year, month = 2005 + i // 2, ((11, 12) if split == "VAL" else (8, 9))[i % 2]
        count = (7 if i == 39 else 8) if split == "VAL" else 12
        for j in range(count):
            ids.append(f"{year}-{month:02d}-{year}-{month:02d}-{j + 1:02d}T00:00-s71")
            months.append(year * 100 + month)
    return ids, months


def constant_diagnostics(means):
    # All twenty paired years have the same five scores in this fixture.
    order = sorted(range(5), key=lambda i: (means[i], i))
    return {"means": means, "selected_generation": c.UPDATES[order[0]],
            "runner_up_gap": means[order[1]] - means[order[0]],
            "bootstrap_selection_counts": {str(u): 10000 if i == order[0] else 0 for i, u in enumerate(c.UPDATES)},
            "leave_year_out_counts": {str(u): 20 if i == order[0] else 0 for i, u in enumerate(c.UPDATES)}}


@pytest.fixture
def synthetic():
    grids, curves = {}, {}
    for budget, split in (("VAL25", "VAL"), ("VAL50", "VAL"), ("TEST50", "TEST")):
        ids, months = population(split)
        grids[budget] = {"ids": ids, "months": months, "data": {}}
        curves[budget] = {}
        for n, name in enumerate(c.NAMES):
            base = np.roll([6, 4, 4, 7, 8], (n + (budget == "VAL50")) % 5).astype(float)
            if split == "TEST":
                base = np.roll([11, 4, 13, 7, 2], n % 5).astype(float)
            base += n / 8
            curves[budget][name] = (base + 1).tolist()
            for i, update in enumerate(c.UPDATES):
                grids[budget]["data"][f"{name}-g{update}"] = [float(base[i] + 2 * (m % 100 in (9, 12))) for m in months]
    policies = {n: {"P25": c.UPDATES[curves["VAL25"][n].index(min(curves["VAL25"][n]))],
                    "P50": c.UPDATES[curves["VAL50"][n].index(min(curves["VAL50"][n]))], "F": c.UPDATES[-1]} for n in c.NAMES}
    original = {n: {"generation": policies[n]["P25"], "score": min(curves["VAL25"][n])} for n in c.NAMES}
    saved = {"ids": grids["VAL25"]["ids"], "months": grids["VAL25"]["months"], "scores": grids["VAL25"]["data"],
             "curves": {n: constant_diagnostics(curves["VAL25"][n]) for n in c.NAMES}, "bootstrap": c.BOOTSTRAP.copy()}
    report = {"protocol_sha256": c.PROTOCOL, "selection_sha256": c.SELECTION, "policies": copy.deepcopy(policies),
              "scores": {s: grids[s + "50"] for s in ("VAL", "TEST")},
              "curves": {s: curves[s + "50"] for s in ("VAL", "TEST")}, "input_sha256": {}, "scope": "fixture",
              "VAL50_selection_diagnostics": {n: constant_diagnostics(curves["VAL50"][n]) for n in c.NAMES},
              "validation_budget_comparison": {}, "policy_means": {}, "common_update_TEST_contrasts": {},
              "contrast_changes": {}, "final_minus_selected_TEST": {}}
    for n in c.NAMES:
        old, new = curves["VAL25"][n], curves["VAL50"][n]
        report["validation_budget_comparison"][n] = {
            "VAL25": old, "VAL50": new, "VAL50_minus_VAL25": [b - a for a, b in zip(old, new)],
            "VAL25_ranks_earlier_ties": [sorted((v, i) for i, v in enumerate(old)).index((v, i)) + 1 for i, v in enumerate(old)],
            "VAL50_ranks_earlier_ties": [sorted((v, i) for i, v in enumerate(new)).index((v, i)) + 1 for i, v in enumerate(new)],
            "P25": policies[n]["P25"], "P50": policies[n]["P50"]}
    for policy in ("P25", "P50", "F"):
        row = {}
        for arm in c.ARMS:
            values = [curves["TEST50"][f"{arm}_s{s}"][c.UPDATES.index(policies[f"{arm}_s{s}"][policy])] for s in c.SEEDS]
            row[arm] = {"seedwise": values, "mean": sum(values) / 3}
        row["O-S_minus_D"] = row["O-S"]["mean"] - row["D"]["mean"]
        report["policy_means"][policy] = row
    for policy in ("P50", "F"):
        report["contrast_changes"][policy + "_minus_P25"] = (report["policy_means"][policy]["O-S_minus_D"]
                                                                - report["policy_means"]["P25"]["O-S_minus_D"])
    for i, update in enumerate(c.UPDATES):
        values = [curves["TEST50"][f"O-S_s{s}"][i] - curves["TEST50"][f"D_s{s}"][i] for s in c.SEEDS]
        report["common_update_TEST_contrasts"][str(update)] = {"seedwise": values, "mean": sum(values) / 3}
    for policy in ("P25", "P50"):
        report["final_minus_selected_TEST"][policy] = {
            arm: [a - b for a, b in zip(report["policy_means"]["F"][arm]["seedwise"],
                                        report["policy_means"][policy][arm]["seedwise"])] for arm in c.ARMS}
    return report, saved, {"policies": policies}, original


def test_complete_synthetic_reductions_exact_ties_and_unequal_months(synthetic):
    report, saved, selection, original = synthetic
    checked = c.validate_reductions(*synthetic)
    assert checked["VAL25"]["D_s7"] == [7, 5, 5, 8, 9]
    assert original["D_s7"]["generation"] == c.UPDATES[1]
    assert selection["policies"]["D_s7"]["P50"] == c.UPDATES[2]
    assert np.mean(saved["scores"]["D_s7-g9560"]) != 5
    assert report["common_update_TEST_contrasts"]["2868"]["seedwise"] != [0, 0, 0]


@pytest.mark.parametrize("change,message", [
    ("missing_update", "thirty-checkpoint grid"), ("missing_contrast", "common-update contrasts"),
    ("month_weight", "month-balanced curves"), ("rank", "validation budgets and ranks"),
    ("late_tie", "policies selected only"), ("per_seed_policy", "policies selected only"),
    ("test_selected", "policies selected only"), ("test_diagnostics", "TEST selection diagnostics"),
    ("contrast_sign", "fixed-policy TEST contrasts"), ("contrast_change", "policy contrast changes"),
    ("final_delta", "final-minus-selected"), ("old_curve", "VAL25 selection stability"),
    ("bootstrap25", "VAL25 selection stability"), ("bootstrap50", "VAL50 selection stability"),
    ("loyo25", "VAL25 selection stability"), ("loyo50", "VAL50 selection stability"),
    ("bootstrap_seed", "validation bootstrap"), ("old_selection", "original P25 generation"),
    ("month_order", "month/ID order"), ("duplicate_id", "window population"),
    ("nan_score", "finite per-window scores"), ("bool_score", "finite per-window scores"),
])
def test_semantic_tampering_is_rejected(synthetic, change, message):
    report, saved, selected, original = synthetic
    if change == "missing_update":
        del report["scores"]["TEST"]["data"][c.JOBS[0]]
    elif change == "missing_contrast":
        del report["common_update_TEST_contrasts"]["16252"]
    elif change == "month_weight":
        report["curves"]["VAL"]["D_s7"] = [float(np.mean(report["scores"]["VAL"]["data"][f"D_s7-g{u}"])) for u in c.UPDATES]
    elif change == "rank":
        report["validation_budget_comparison"]["D_s7"]["VAL25_ranks_earlier_ties"][1:3] = [2, 1]
    elif change in ("late_tie", "per_seed_policy", "test_selected"):
        value = (c.UPDATES[3] if change == "late_tie" else selected["policies"]["D_s13"]["P50"]
                 if change == "per_seed_policy" else c.UPDATES[4])
        selected["policies"]["D_s7"]["P50"] = value
        report["policies"]["D_s7"]["P50"] = value
    elif change == "test_diagnostics":
        report["TEST50_selection_diagnostics"] = report["VAL50_selection_diagnostics"]
    elif change == "contrast_sign":
        report["policy_means"]["P25"]["O-S_minus_D"] *= -1
    elif change == "contrast_change":
        report["contrast_changes"]["F_minus_P25"] += .125
    elif change == "final_delta":
        report["final_minus_selected_TEST"]["P25"]["D"][0] += .125
    elif change == "old_curve":
        saved["curves"]["D_s7"]["means"][0] += .125
    elif change.startswith(("bootstrap", "loyo")) and change != "bootstrap_seed":
        row = saved["curves"]["D_s7"] if change.endswith("25") else report["VAL50_selection_diagnostics"]["D_s7"]
        field = "bootstrap_selection_counts" if change.startswith("bootstrap") else "leave_year_out_counts"
        winner = str(row["selected_generation"])
        row[field][winner] -= 1
        row[field]["2868"] += 1
    elif change == "bootstrap_seed":
        saved["bootstrap"]["seed"] += 1
    elif change == "old_selection":
        original["D_s7"]["generation"] = c.UPDATES[2]
    elif change == "month_order":
        report["scores"]["TEST"]["months"][0] += 1
    elif change == "duplicate_id":
        report["scores"]["TEST"]["ids"][0] = report["scores"]["TEST"]["ids"][1]
    elif change == "nan_score":
        report["scores"]["VAL"]["data"][c.JOBS[0]][0] = float("nan")
    elif change == "bool_score":
        for key in c.JOBS[:5]:
            report["scores"]["VAL"]["data"][key] = [True] * 319
    with pytest.raises(ERRORS, match=message):
        c.validate_reductions(*synthetic)


def test_paired_year_bootstrap_and_loyo_use_the_whole_year():
    years = np.full((20, 5), 100.)
    years[:, :2] = [1, 0]
    years[0, :2] = [0, 10]
    monthly = np.repeat(years.T, 2, axis=1)
    row = c.selection_diagnostics(monthly)
    rng = np.random.Generator(np.random.PCG64(20261001))
    multiplicity = np.count_nonzero(rng.integers(20, size=(10000, 20)) == 0, axis=1)
    first = int(np.count_nonzero(20 - multiplicity < 10 * multiplicity))
    assert 0 < first < 10000
    assert list(row["bootstrap_selection_counts"].values()) == [first, 10000 - first, 0, 0, 0]
    assert list(row["leave_year_out_counts"].values()) == [0, 20, 0, 0, 0]
    # Opposing month shifts preserve paired years; month-wise sampling would
    # introduce uncertainty absent from the specified year bootstrap.
    monthly[0, ::2] += 8
    monthly[0, 1::2] -= 8
    assert c.selection_diagnostics(monthly) == row


def test_sub_tolerance_difference_is_not_an_exact_tie():
    monthly = np.repeat(np.array([[1.], [1. - 2 ** -45], [2.], [3.], [4.]]), 40, axis=1)
    row = c.selection_diagnostics(monthly)
    assert row["selected_generation"] == c.UPDATES[1]
    assert list(row["bootstrap_selection_counts"].values()) == [0, 10000, 0, 0, 0]
    assert c.ranks([1, 1 - 2 ** -45, 2, 3, 4]) == [2, 1, 3, 4, 5]


@pytest.fixture(scope="module")
def public_records():
    records = {name: g.read_json(PAPER / PUBLIC / name) for name in c.FILES}
    originals = [g.read_json(PAPER / "evidence" / name) for name in
                 ("real/comparison.json", "real/protocol-summary.json", "real/MANIFEST.json", "deadline/fixed-final.json", "real/scores.json")]
    return records, originals


def test_complete_public_checkpoint_evidence(public_records):
    _, originals = public_records
    checked = c.validate_checkpoints(PAPER, [originals[0]])
    assert set(checked) == {"manifest", "protocol", "scope", "selection", "saved", "summary", "curves"}
    assert len(checked["summary"]["scores"]["TEST"]["data"]) == 30


@pytest.mark.parametrize("change,message", [
    ("source", "training_source_hashes"), ("source_omission", "inference_source_hashes"),
    ("arrays", "arrays"), ("checkpoint", "selected weights"), ("reuse_payload", "authenticated original payload"),
    ("reuse_commit", "authenticated original payload"), ("final_weights", "final checkpoint weights"),
    ("p25", "P25"), ("test_ids", "original ordered TEST population"),
    ("reuse_receipt", "reuse authentication"), ("test_authorization", "predetermined TEST grid"),
    ("input_alias", "complete checkpoint input inventory"), ("input_omission", "complete checkpoint input inventory"),
    ("val_pin", "pre-TEST VAL50 score pins"), ("private_path", "unsafe private path"),
    ("saved_pin", "original VAL25 score pins"), ("test_selection", "VAL-only selection"),
])
def test_public_semantic_provenance_after_byte_pins(public_records, change, message):
    original_records, originals = public_records
    records = copy.deepcopy(original_records)
    p, selected, saved, summary = (records[n + ".json"] for n in ("PROTOCOL", "SELECTION", "SAVED_VAL25", "SUMMARY"))
    if change == "source":
        p["training_source_hashes"]["prigm/scoring.py"] = "0" * 64
    elif change == "source_omission":
        del p["inference_source_hashes"]["prigm/scoring.py"]
    elif change == "arrays":
        p["arrays"]["TRAIN"]["sha256"] = "0" * 64
    elif change == "checkpoint":
        next(j for j in p["jobs"] if j["id"] == "D_s7-g9560")["checkpoint_sha256"] = "0" * 64
    elif change in ("reuse_payload", "reuse_commit"):
        job = next(j for j in p["jobs"] if j["id"] == "D_s7-g9560")
        job["reuse"]["payload_sha256" if change == "reuse_payload" else "commit_sha256"] = "0" * 64
    elif change == "final_weights":
        next(j for j in p["jobs"] if j["id"] == "D_s13-g29636")["checkpoint_sha256"] = "0" * 64
    elif change == "p25":
        p["P25"]["D_s7"]["generation"] = c.UPDATES[0]
    elif change == "test_ids":
        summary["scores"]["TEST"]["ids"][:2] = summary["scores"]["TEST"]["ids"][1::-1]
    elif change == "reuse_receipt":
        selected["reuse_authentication_sha256"] = "0" * 64
    elif change == "test_authorization":
        selected["test_jobs"].pop()
    elif change in ("input_alias", "input_omission", "private_path"):
        key = "checkpoint-study/artifacts/test/D_s7-g2868/score-raw.json"
        value = summary["input_sha256"].pop(key)
        if change != "input_omission":
            summary["input_sha256"][key.replace("checkpoint-study", "original-study" if change == "input_alias" else "/home/private/study")] = value
    elif change == "val_pin":
        selected["validation_score_sha256"]["checkpoint-study/artifacts/val/D_s7-g2868/score-raw.json"] = "0" * 64
    elif change == "saved_pin":
        saved["input_sha256"]["original-study/artifacts/val/D_s7-g2868/score-raw.json"] = "0" * 64
    elif change == "test_selection":
        selected["selection"] = "Select minimum TEST score"
    with pytest.raises(ERRORS, match=message):
        c.validate_provenance(records, *originals)


def test_public_payload_rehash_cannot_replace_frozen_manifest(public_records, tmp_path):
    root = tmp_path / PUBLIC
    shutil.copytree(PAPER / PUBLIC, root)
    path = root / "SUMMARY.json"
    summary = g.read_json(path)
    summary["scores"]["VAL"]["data"][c.JOBS[0]][0] += .01
    path.write_text(json.dumps(summary))
    manifest = g.read_json(root / "MANIFEST.json")
    manifest["files"]["SUMMARY.json"] = g.digest(path)
    (root / "MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ERRORS, match="independent checkpoint public manifest"):
        c.validate_checkpoints(tmp_path, [public_records[1][0]])


@pytest.fixture(scope="module")
def checked_public(public_records):
    return c.validate_checkpoints(PAPER, [public_records[1][0]])


def test_public_figure_matches_independent_reductions(checked_public):
    figure = c.validate_checkpoint_figure(PAPER, checked_public)
    assert figure["curves"] == checked_public["curves"]


def test_checkpoint_axis_label_keeps_thousands_units(checked_public):
    spec = importlib.util.spec_from_file_location("checkpoint_figure", PAPER / "fig_checkpoints.py")
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    fig = renderer.plot({"curves": checked_public["curves"],
                         "policies": checked_public["selection"]["policies"]})
    try:
        assert [text.get_text() for text in fig.texts] == ["Training steps (thousands)"]
        for ax in fig.axes:
            np.testing.assert_array_equal(ax.lines[0].get_xdata(), np.asarray(c.UPDATES) / 1000)
            np.testing.assert_array_equal(ax.get_xticks(), np.asarray(c.UPDATES) / 1000)
    finally:
        renderer.plt.close(fig)


@pytest.mark.parametrize("damage,message", [
    (None, None), ("missing", "checkpoint figure fields"),
    ("incorrect", "checkpoint figure historical exporter"),
])
def test_checkpoint_figure_separates_exporter_from_renderer(checked_public, monkeypatch, damage, message):
    figure = g.read_json(PAPER / "figures/checkpoints/fig_checkpoints.manifest.json")
    exporter = checked_public["manifest"]["source"]
    renderer = {"path": "fig_checkpoints.py", "sha256": g.digest(PAPER / "fig_checkpoints.py")}
    assert renderer != exporter
    assert figure["evidence_source"] == exporter
    figure["source"] = renderer
    if damage == "missing":
        del figure["evidence_source"]
    elif damage == "incorrect":
        figure["evidence_source"] = renderer.copy()
    # Isolate semantic binding after the separate figure byte checks.
    monkeypatch.setattr(camera, "figure_record", lambda *args: figure)
    if damage:
        with pytest.raises(ERRORS, match=message):
            c.validate_checkpoint_figure(PAPER, checked_public)
    else:
        result = c.validate_checkpoint_figure(PAPER, checked_public)
        assert result["source"] == renderer
        assert result["evidence_source"] == exporter
        assert result["curves"] == checked_public["curves"]
        assert result["policies"] == checked_public["selection"]["policies"]


@pytest.mark.parametrize("change,message", [
    ("sign", "figure coordinates"), ("budget", "figure coordinates"),
    ("policy", "policy markers"), ("protocol", "figure protocol"),
])
def test_plotted_curves_and_policy_markers_after_byte_pin(checked_public, monkeypatch, change, message):
    figure = g.read_json(PAPER / "figures/checkpoints/fig_checkpoints.manifest.json")
    if change == "sign":
        figure["curves"]["TEST50"]["O-S_s13"][0] *= -1
    elif change == "budget":
        figure["curves"]["VAL50"] = figure["curves"]["VAL25"]
    elif change == "policy":
        figure["policies"]["D_s7"]["P50"] = c.UPDATES[-1]
    else:
        figure["protocol_sha256"] = c.ORIGINAL
    monkeypatch.setattr(camera, "figure_record", lambda *args: figure)
    with pytest.raises(ERRORS, match=message):
        c.validate_checkpoint_figure(PAPER, checked_public)


@pytest.mark.parametrize("target,message", [
    ("input", "figure inputs"), ("source", "figure source"), ("output", "figure output"),
    ("style", "figure style"),
])
def test_changed_figure_artifact_is_rejected(checked_public, monkeypatch, target, message):
    paths = {"input": PAPER / PUBLIC / "SUMMARY.json", "source": PAPER / "fig_checkpoints.py",
             "output": PAPER / "figures/checkpoints/fig_checkpoints.pdf",
             "style": PAPER / "prigm.mplstyle"}
    digest = g.digest
    monkeypatch.setattr(g, "digest", lambda path: "0" * 64 if Path(path) == paths[target] else digest(path))
    with pytest.raises(ERRORS, match=message):
        c.validate_checkpoint_figure(PAPER, checked_public)


def test_checkpoint_claims_match_checked_public_evidence(checked_public):
    source = (PAPER / "main.tex").read_text()
    camera.checkpoint_claims(source, g.prose_text(source, source=True), checked_public)


def test_checkpoint_claims_accept_rendered_thousands_counts(checked_public):
    source = (PAPER / "main.tex").read_text()
    rendered = g.prose_text(source, source=True)
    assert "10000 resamples" in rendered and "update 9560 in" in rendered
    rendered = rendered.replace("10000 resamples", "10,000 resamples")
    rendered = rendered.replace("update 9560 in", "update 9,560 in")
    camera.checkpoint_claims(source, rendered, checked_public)


@pytest.mark.parametrize("change", ["contrast", "bootstrap_frequency", "image_swap", "scope_removed"])
def test_checkpoint_claim_mutations_are_rejected(checked_public, change):
    source = (PAPER / "main.tex").read_text()
    if change == "contrast":
        source = source.replace("changing their difference from $-0.0028$", "changing their difference from $+0.0028$", 1)
    elif change == "bootstrap_frequency":
        source = source.replace(r"61.0\% of resamples with 25 evaluations", r"71.0\% of resamples with 25 evaluations", 1)
    elif change == "image_swap":
        # Preserve both included images, but attach each to the wrong caption.
        source = source.replace("figures/checkpoints/fig_checkpoints.pdf", "CHECKPOINT_IMAGE")
        source = source.replace("figures/calibration/fig_calibration.pdf", "figures/checkpoints/fig_checkpoints.pdf")
        source = source.replace("CHECKPOINT_IMAGE", "figures/calibration/fig_calibration.pdf")
    else:
        source = source.replace("both these choices and the evaluation count are fixed before additional TEST samples are generated.", "", 1)
    assert source != (PAPER / "main.tex").read_text()
    with pytest.raises(ERRORS):
        camera.checkpoint_claims(source, g.prose_text(source, source=True), checked_public)
