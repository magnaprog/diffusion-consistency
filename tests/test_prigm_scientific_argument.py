"""Independent arithmetic and false-claim controls for the new scientific claims."""
import copy
import re
import shutil
import subprocess

import numpy as np
import pytest

from experiments import _prigm_camera_ready as camera
from experiments import _prigm_checkpoint_evidence as checkpoints
from experiments import verify_prigm_paper as g
from prigm.recovery import ContractMismatch


@pytest.fixture(scope="module")
def checked():
    real = [g.read_json(g.PAPER / "evidence/real/comparison.json")]
    return checkpoints.validate_checkpoints(g.PAPER, real)


@pytest.fixture(scope="module")
def reselection(checked):
    return camera.validate_reselection(g.PAPER, checked)


@pytest.fixture(scope="module")
def calibration():
    real = [g.read_json(g.PAPER / "evidence/real/comparison.json")]
    return camera.validate_calibration(g.PAPER, real)


def override_result(monkeypatch, record):
    original = g.read_json
    monkeypatch.setattr(g, "read_json", lambda path: record if path.name == "checkpoint_reselection.json" else original(path))


def test_reselection_reproduces_original_choices_and_all_counts(reselection):
    assert reselection["original_choices"] == {
        "D_s7": 9560, "D_s13": 9560, "D_s42": 16252,
        "O-S_s7": 29636, "O-S_s13": 29636, "O-S_s42": 16252}
    assert all(sum(row.values()) == 10000 for row in reselection["selection_counts"].values())
    assert reselection["selection_counts"]["D_s7"]["9560"] == 7168


@pytest.mark.parametrize("path,value", [
    (("procedure", "replicates"), 9999), (("procedure", "seed"), 7004),
    (("procedure", "ensemble_members"), 3), (("procedure", "validation_evaluations"), 25),
    (("original_choices", "D_s7"), 29636), (("selection_counts", "D_s7", "9560"), 7169),
    (("selection_counts", "D_s7", "9560"), 7168.0),
    (("observed_TEST", "O-S_minus_D"), .0027851875589770536),
    (("fixed_selection", "percentile_2_5_97_5"), [-.0336, .0246]),
    (("reselected", "percentile_2_5_97_5"), [-.0346, .0242]),
    (("reselected", "relative_percentile_2_5_97_5"), [-1.629, 1.412]),
    (("reselected", "mean"), float("nan")),
    (("input", "sha256"), "0" * 64), (("source", "sha256"), "0" * 64),
    (("helper_sha256", "prigm/select.py"), "0" * 64),
    (("limits",), ["New-fit uncertainty included"]), (("schema_version",), True),
])
def test_reselection_rejects_changed_population_statistics_or_provenance(checked, reselection, monkeypatch, path, value):
    record = copy.deepcopy(reselection)
    node = record
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    override_result(monkeypatch, record)
    with pytest.raises((ValueError, ContractMismatch), match="checkpoint reselection"):
        camera.validate_reselection(g.PAPER, checked)


@pytest.mark.parametrize("mistake", ["TEST_selection", "unpaired_years", "drop_repeated_years", "window_weighting"])
def test_reselection_rejects_wrong_resampling_calculations(checked, reselection, monkeypatch, mistake):
    years = {}
    for split in ("VAL", "TEST"):
        data = checked["summary"]["scores"][split]
        months = np.array(data["months"])
        values = np.array([data["data"][job] for job in checkpoints.JOBS])
        if mistake == "window_weighting":
            means = [values[:, months // 100 == year].mean(axis=1) for year in range(2005, 2025)]
            years[split] = np.stack(means, axis=-1).reshape(6, 5, 20)
        else:
            _, monthly = checkpoints.reduce_grid(data, split)
            years[split] = np.array([monthly[name] for name in checkpoints.NAMES]).reshape(6, 5, 20, 2).mean(axis=-1)
    rng = np.random.default_rng(7003)
    draws = rng.integers(0, 20, (10000, 20))
    weights = np.array([np.bincount(row, minlength=20) for row in draws], dtype=float)
    if mistake == "drop_repeated_years":
        weights = (weights > 0).astype(float)
    weights /= weights.sum(axis=1, keepdims=True)
    val = np.einsum("ry,guy->rgu", weights, years["VAL"])
    test_weights = weights
    if mistake == "unpaired_years":
        test_weights = np.array([np.bincount(row, minlength=20) / 20 for row in rng.integers(0, 20, (10000, 20))])
    test = np.einsum("ry,guy->rgu", test_weights, years["TEST"])
    winner = (test if mistake == "TEST_selection" else val).argmin(axis=-1)
    chosen = np.take_along_axis(test, winner[..., None], axis=-1)[..., 0]
    difference = chosen[:, 3:].mean(axis=1) - chosen[:, :3].mean(axis=1)
    record = copy.deepcopy(reselection)
    wrong = np.quantile(difference, [.025, .975]).tolist()
    assert not np.allclose(wrong, record["reselected"]["percentile_2_5_97_5"], rtol=0, atol=1e-10)
    record["reselected"]["percentile_2_5_97_5"] = wrong
    override_result(monkeypatch, record)
    with pytest.raises(ContractMismatch, match="reselected"):
        camera.validate_reselection(g.PAPER, checked)


@pytest.mark.parametrize("scope", ["source", "PDF"])
@pytest.mark.parametrize("old,new", [
    ("10{,}000", "-10{,}000"), ("7003", "NaN"), ("50 evaluations", "25 evaluations"),
    ("2.5th", "1st"), ("97.5th", "99th"), ("-0.0336", "+0.0336"),
    ("0.0246", "0.0247"), ("-0.0346", "-0.0347"),
    ("same year multiplicities", "different year multiplicities"),
    ("earlier update", "later update"), ("remain outside", "are included in"),
])
@pytest.mark.parametrize("duplicate", [False, True])
def test_reselection_complete_claim_rejects_false_values_and_correct_duplicates(reselection, scope, old, new, duplicate):
    tex = (g.PAPER / "main.tex").read_text()
    rendered = g.prose_text(tex, source=True)
    camera.reselection_claims(tex, rendered, reselection)
    clause = next(p for p in tex.split("\n\n") if p.startswith("Repeating checkpoint choice"))
    if scope == "PDF":
        clause, old, new = [g.prose_text(x, source=True) for x in (clause, old, new)]
    original = tex if scope == "source" else rendered
    assert original.count(clause) == 1 and clause.count(old) == 1
    altered = original.replace(clause, clause.replace(old, new), 1)
    if duplicate:
        altered += "\n" + clause
    with pytest.raises(ValueError, match=scope + ": (clause differs|missing or duplicate clause)"):
        camera.reselection_claims(altered if scope == "source" else tex,
                                  altered if scope == "PDF" else rendered, reselection)


@pytest.mark.parametrize("scope", ["source", "PDF"])
@pytest.mark.parametrize("sentence,old,new", [
    ("In 10{,}000 resamples at seed 7003, the same year multiplicities weight validation and TEST.", "7003", "9999"),
    ("Validation with 50 evaluations chooses each fit's checkpoint anew, with ties favoring the earlier update.", "50", "999"),
    ("The 2.5th and 97.5th percentiles of O-S minus D are $[-0.0336,0.0246]$, close to $[-0.0346,0.0242]$ when the original choices are retained in the same resamples.", "2.5th", "NaN"),
])
def test_reselection_false_numeric_sentence_cannot_hide_outside_correct_paragraph(reselection, scope, sentence, old, new):
    tex = (g.PAPER / "main.tex").read_text()
    rendered = g.prose_text(tex, source=True)
    wrong = sentence.replace(old, new)
    if scope == "source":
        tex += "\n" + wrong
    else:
        rendered += "\n" + g.prose_text(wrong, source=True)
    with pytest.raises(ValueError, match=scope + ": missing or duplicate clause"):
        camera.reselection_claims(tex, rendered, reselection)


def test_overview_reductions_bind_released_inputs_and_outputs(checked, calibration):
    figure = camera.validate_overview(g.PAPER, checked, calibration)
    assert figure["panel_b"]["full_flank_gaussian"]["fixed_model_seeds"] == [None]
    assert figure["panel_b"]["full_flank_gaussian"]["individual_y"] == []


@pytest.mark.parametrize("path", [
    "fig_overview.py", "fig_checkpoints.py", "fig_calibration.py", "prigm.mplstyle",
    "figures/overview/fig_overview.manifest.json", "figures/overview/fig_overview.pdf",
    "figures/overview/fig_overview.png", "evidence/followup/checkpoints/SUMMARY.json",
    "evidence/followup/calibration/summary.json",
])
def test_overview_rejects_changed_source_input_or_output(checked, calibration, tmp_path, path):
    manifest = g.read_json(g.PAPER / "figures/overview/fig_overview.manifest.json")
    names = {row["path"] for row in manifest["inputs"].values()}
    names.update({"fig_overview.py", "fig_checkpoints.py", "fig_calibration.py", "prigm.mplstyle"})
    names.update("figures/overview/" + name for name in [*manifest["output_sha256"], "fig_overview.manifest.json"])
    for name in names:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(g.PAPER / name, destination)
    camera.validate_overview(tmp_path, checked, calibration)
    target = tmp_path / path
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises((ValueError, ContractMismatch), match="figure|overview"):
        camera.validate_overview(tmp_path, checked, calibration)


def test_overview_first_and_full_reviewed_figure_order_are_required():
    tex = (g.PAPER / "main.tex").read_text()
    g.source_checks(g.PAPER, tex)
    first = "figures/overview/fig_overview.pdf"
    second = "figures/checkpoints/fig_checkpoints.pdf"
    wrong = tex.replace(first, "SWAPPED_FIGURE").replace(second, first).replace("SWAPPED_FIGURE", second)
    assert sorted(re.findall(r"\\includegraphics[^\n]+", tex)) == sorted(re.findall(r"\\includegraphics[^\n]+", wrong))
    with pytest.raises(ValueError, match="reviewed figure order"):
        g.source_checks(g.PAPER, wrong)


@pytest.mark.parametrize("mistake", ["contrast_sign", "mean", "fit_mean", "gaussian_three_fits",
    "gaussian_three_dots", "coverage_units", "position", "seed_order", "draw_count", "ranks", "confidence"])
def test_overview_independently_checks_coordinates_and_populations(checked, calibration, monkeypatch, mistake):
    figure = g.read_json(g.PAPER / "figures/overview/fig_overview.manifest.json")
    if mistake == "contrast_sign":
        figure["panel_a"]["seedwise"][1][1] *= -1
    elif mistake == "mean":
        figure["panel_a"]["mean"][0] += .001
    elif mistake == "fit_mean":
        figure["panel_b"]["G5"]["mean_percent"] = figure["panel_b"]["G5"]["model_means_percent"][0]
    elif mistake == "gaussian_three_fits":
        figure["panel_b"]["full_flank_gaussian"]["fixed_model_seeds"] = [7, 13, 42]
    elif mistake == "gaussian_three_dots":
        row = figure["panel_b"]["full_flank_gaussian"]
        row["model_means_percent"] *= 3
        row["individual_y"] = [row["y"]] * 3
    elif mistake == "coverage_units":
        figure["panel_b"]["D"]["mean_percent"] = (np.array(figure["panel_b"]["D"]["mean_percent"]) / 100).tolist()
    elif mistake == "position":
        figure["panel_b"]["G5"]["individual_y"][0][0] += .01
    elif mistake == "seed_order":
        figure["panel_a"]["seeds"] = [42, 13, 7]
    elif mistake == "draw_count":
        figure["n_draws"] = 117
    elif mistake == "ranks":
        figure["interval_order_statistics"] = [1, 39]
    else:
        figure["pointwise_intervals_plotted"] = True
    monkeypatch.setattr(camera, "figure_record", lambda *args: figure)
    with pytest.raises((ValueError, ContractMismatch), match="overview"):
        camera.validate_overview(g.PAPER, checked, calibration)


@pytest.mark.parametrize("scope", ["source", "PDF"])
@pytest.mark.parametrize("old,new", [
    ("raw validation score", "raw TEST score"), ("three training seeds", "thirty training seeds"),
    ("Negative values favor O-S", "Positive values favor O-S"),
    ("25 or 50", "25 or 100"), ("same six checkpoints", "same five checkpoints"),
    ("one ensemble", "three independent fits"), ("twelve hidden positions", "sixty hidden positions"),
])
def test_overview_caption_rejects_false_claim_with_correct_duplicate(scope, old, new):
    tex = (g.PAPER / "main.tex").read_text()
    floats = tex.replace(r"\begin{figure}", r"\begin{table}").replace(r"\end{figure}", r"\end{table}")
    caption = g.float_caption(floats, "fig:overview")
    rendered = g.prose_text(tex, source=True)
    camera.overview_claims(tex, rendered)
    if scope == "source":
        tex = tex.replace(caption, caption.replace(old, new), 1) + "\n" + caption
    else:
        caption, old, new = [g.prose_text(x, source=True) for x in (caption, old, new)]
        rendered = rendered.replace(caption, caption.replace(old, new), 1) + "\n" + caption
    with pytest.raises(ValueError, match="caption differs|missing or duplicate clause"):
        camera.overview_claims(tex, rendered)


@pytest.mark.parametrize("scope", ["source", "PDF"])
@pytest.mark.parametrize("old,new", [("25", "-25"), ("50", "NaN"), ("six", "five")])
def test_overview_false_validation_count_cannot_hide_beside_correct_caption(scope, old, new):
    tex = (g.PAPER / "main.tex").read_text()
    rendered = g.prose_text(tex, source=True)
    camera.overview_claims(tex, rendered)
    sentence = "Validation with 25 or 50 evaluations chooses the same six checkpoints."
    wrong = sentence.replace(old, new)
    if scope == "source":
        tex += "\n" + wrong
    else:
        rendered += "\n" + wrong
    with pytest.raises(ValueError, match=scope + ": missing or duplicate clause"):
        camera.overview_claims(tex, rendered)


@pytest.fixture(scope="module")
def manuscript_pdf_text():
    return subprocess.check_output(["pdftotext", "-layout", str(g.PAPER / "main.pdf"), "-"], text=True)


@pytest.mark.parametrize("scope", ["source", "PDF"])
@pytest.mark.parametrize("position", ["before", "after"])
@pytest.mark.parametrize("claim,wrong", [
    ("reselection", "In 10{,}000 resamples at seed 7003, different year multiplicities weight validation and TEST."),
    ("overview", "Validation with 25 or 50 evaluations chooses different checkpoints."),
])
def test_false_semantic_condition_cannot_hide_beside_valid_claim(
        reselection, manuscript_pdf_text, scope, position, claim, wrong):
    tex = (g.PAPER / "main.tex").read_text()
    rendered = manuscript_pdf_text
    if claim == "reselection":
        def check(source, pdf):
            camera.reselection_claims(source, pdf, reselection)
    else:
        check = camera.overview_claims
    check(tex, rendered)
    original = tex if scope == "source" else rendered
    if scope == "PDF":
        wrong = wrong.replace("10{,}000", "10,000")
    changed = wrong + "\n" + original if position == "before" else original + "\n" + wrong
    with pytest.raises(ValueError, match=scope + ": missing or duplicate clause"):
        check(changed if scope == "source" else tex, changed if scope == "PDF" else rendered)
