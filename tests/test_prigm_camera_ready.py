"""Semantic calibration checks using only the release's public evidence."""

import copy
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pytest

from experiments import _prigm_camera_ready as camera
from experiments import verify_prigm_paper as verifier
from prigm.recovery import ContractMismatch


PAPER = Path(__file__).resolve().parents[1] / "paper/workshops/prigm"
CALIBRATION = Path("evidence/followup/calibration")


@pytest.fixture(scope="module")
def public_records():
    report = verifier.read_json(PAPER / CALIBRATION / "summary.json")
    real = verifier.read_json(PAPER / "evidence/real/comparison.json")
    with np.load(PAPER / CALIBRATION / "per_window.npz", allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    figure = verifier.read_json(PAPER / "figures/calibration/fig_calibration.manifest.json")
    return report, real, arrays, figure


def test_public_calibration_validates_without_private_inputs(public_records):
    report, real, _, _ = public_records
    assert camera.validate_calibration(PAPER, [real]) == report


def test_public_calibration_figure_validates_without_private_inputs(public_records):
    report, _, _, figure = public_records
    assert camera.validate_calibration_figure(PAPER, report) == figure


@pytest.mark.parametrize("change,message", [
    ("fraction", "twelve-position counts"), ("partition", "tail partition"),
    ("months", "tail month labels"), ("ids", "tail window order"),
    ("shape", "tail fraction population"), ("missing_method", "complete tail inventory"),
])
def test_per_window_population_and_integer_counts(public_records, change, message):
    report, real, original, _ = public_records
    arrays = {name: values.copy() for name, values in original.items()}
    if change == "fraction":
        # Preserve nonnegativity and the partition while introducing half a bin.
        window, channel = np.argwhere(arrays["D"][0, :, :, 1] >= 1 / 12)[0]
        arrays["D"][0, window, channel, 0] += 1 / 24
        arrays["D"][0, window, channel, 1] -= 1 / 24
    elif change == "partition":
        arrays["D"][0, 0, 0] = 0
    elif change == "months":
        arrays["months"][[0, -1]] = arrays["months"][[ -1, 0]]
    elif change == "ids":
        arrays["ids"][[0, -1]] = arrays["ids"][[ -1, 0]]
    elif change == "shape":
        arrays["D"] = arrays["D"][:, :, :4]
    else:
        del arrays["I-S"]
    with pytest.raises(ValueError, match=message):
        camera.tail_reductions(report, arrays, real["ids"])


@pytest.mark.parametrize("field,message", [
    ("model_means", "model tail means"), ("mean", "mean tails"),
    ("year_95", "year tail intervals"), ("twoyear_95", "twoyear tail intervals"),
    ("below_minus_above", "tail imbalance"),
    ("year_below_minus_above_95", "year paired imbalance intervals"),
    ("twoyear_below_minus_above_95", "twoyear paired imbalance intervals"),
])
def test_summary_reductions_are_recomputed(public_records, field, message):
    original, real, arrays, _ = public_records
    report = copy.deepcopy(original)
    values = np.asarray(report["methods"]["D"][field])
    values.flat[0] += .001
    report["methods"]["D"][field] = values.tolist()
    with pytest.raises(ValueError, match=message):
        camera.tail_reductions(report, arrays, real["ids"])


def test_unpaired_subtraction_of_tail_interval_bounds_is_rejected(public_records):
    original, real, arrays, _ = public_records
    report = copy.deepcopy(original)
    row = report["methods"]["D"]
    marginal = np.asarray(row["year_95"])
    unpaired = np.stack([marginal[0, :, 0] - marginal[1, :, 2],
                         marginal[1, :, 0] - marginal[0, :, 2]])
    assert not np.allclose(unpaired, row["year_below_minus_above_95"])
    row["year_below_minus_above_95"] = unpaired.tolist()
    with pytest.raises(ValueError, match="year paired imbalance intervals"):
        camera.tail_reductions(report, arrays, real["ids"])


@pytest.fixture
def public_copy(tmp_path):
    for relative in (
        *(CALIBRATION / name for name in ("summary.json", "per_window.npz", "PROTOCOL.json", "MANIFEST.json")),
        Path("evidence/real/MANIFEST.json"), Path("evidence/deadline/MANIFEST.json"),
        Path("evidence/real/diagnostics.json"),
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PAPER / relative, destination)
    return tmp_path


@pytest.mark.parametrize("change,message", [
    ("private_path", "unsafe private path"),
    ("input_alias", "complete calibration input inventory"),
    ("input_hash", "complete calibration input inventory"),
    ("input_omission", "complete calibration input inventory"),
    ("source_hash", "complete calibration source inventory"),
    ("source_omission", "complete calibration source inventory"),
])
def test_rehashed_public_summary_still_requires_input_and_source_binding(
        public_copy, public_records, monkeypatch, change, message):
    report = copy.deepcopy(public_records[0])
    inputs, sources = report["input_sha256"], report["source_sha256"]
    key = "original-study/artifacts/inventory/test.npz"
    if change == "private_path":
        inputs["/home/private/study/test.npz"] = inputs.pop(key)
    elif change == "input_alias":
        inputs["original-study/artifacts/inventory/val.npz"] = inputs.pop(key)
    elif change == "input_hash":
        inputs[key] = "0" * 64
    elif change == "input_omission":
        del inputs[key]
    elif change == "source_hash":
        sources["prigm/loop.py"] = "0" * 64
    else:
        del sources["prigm/loop.py"]
    path = public_copy / CALIBRATION / "summary.json"
    path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    digest = verifier.digest(path)
    manifest_path = public_copy / CALIBRATION / "MANIFEST.json"
    manifest = verifier.read_json(manifest_path)
    manifest["files"]["summary.json"] = digest
    manifest_path.write_text(json.dumps(manifest))
    # Admit this test revision through the independent byte pin so the deeper
    # identity/privacy guard, rather than a stale checksum, must reject it.
    monkeypatch.setitem(camera.CALIBRATION_PINS, "summary.json", digest)
    with pytest.raises(ContractMismatch, match=message):
        camera.validate_calibration(public_copy, [public_records[1]])


@pytest.mark.parametrize("field,source,message", [
    ("source", "analysis/export_prigm_calibration.py", "calibration exporter"),
    ("producer_sha256", "analysis/prigm_tail_calibration.py", "calibration producer"),
])
def test_calibration_requires_historical_source_identity(public_copy, public_records, field, source, message):
    path = public_copy / CALIBRATION / "MANIFEST.json"
    manifest = verifier.read_json(path)
    current = verifier.digest(verifier.REPO / source)
    historical = verifier.documentation.source_hash(verifier.REPO, source)
    assert current != historical
    if field == "source":
        assert manifest[field]["sha256"] == historical
        manifest[field]["sha256"] = current
    else:
        assert manifest[field] == historical
        manifest[field] = current
    path.write_text(json.dumps(manifest))
    with pytest.raises(ContractMismatch, match=message):
        camera.validate_calibration(public_copy, [public_records[1]])


@pytest.mark.parametrize("change,message", [
    ("sign", "plotted score contributions"), ("mean", "plotted score contributions"),
    ("seed_order", "plotted decomposition order"), ("tail_units", "plotted tail percentages"),
    ("seed_dots", "plotted tail percentages"), ("interval_dots", "cannot claim confidence intervals"),
])
def test_figure_coordinates_after_provenance_gate(public_records, monkeypatch, change, message):
    report, _, _, original = public_records
    figure = copy.deepcopy(original)
    if change == "sign":
        figure["decomposition"][1]["negative_delta_dispersion"] *= -1
    elif change == "mean":
        figure["decomposition"][-1]["delta_target"] += .001
    elif change == "seed_order":
        figure["decomposition"][:2] = figure["decomposition"][1::-1]
    elif change == "tail_units":
        figure["tails"]["G5"]["mean_percent"] = report["methods"]["G5"]["mean"]
    elif change == "seed_dots":
        row = figure["tails"]["G5"]
        row["model_means_percent"] = [row["mean_percent"]] * 3
    else:
        figure["pointwise_intervals_plotted"] = True
    monkeypatch.setattr(camera, "figure_record", lambda *args: figure)
    with pytest.raises(ValueError, match=message):
        camera.validate_calibration_figure(PAPER, report)


def manuscript():
    tex = (PAPER / "main.tex").read_text()
    # A text-only rendering surrogate isolates clause logic; the real PDF gate
    # is exercised by the integration verifier, not claimed by these unit tests.
    return tex, verifier.prose_text(tex, source=True)


def test_legitimate_repeated_seed_mentions_do_not_duplicate_claim_anchors(public_records):
    report, _, _, figure = public_records
    tex, _ = manuscript()
    tex += "\nAt seed 13, the tail diagnostic is separate. At seed 42, the tail diagnostic is separate.\n"
    camera.calibration_claims(tex, verifier.prose_text(tex, source=True), report, figure)


@pytest.mark.parametrize("scope", ["source", "PDF"])
@pytest.mark.parametrize("before,after", [
    ("49.0\\% of observations lie below G5's", "49.1\\% of observations lie below G5's"),
    ('G5 decreases both distance', 'G5 increases both distance'),
    ('the larger increase in spread', 'the larger increase in distance'),
    ('errors lie predominantly below the intervals at seeds 7 and 13', 'errors lie predominantly above the intervals at seeds 7 and 13'),
    ('37.7\\% below', '37.8\\% below'),
    ('by 38.4 percentage points', 'by 38.5 percentage points'),
    ('the fraction below exceeds the fraction above by', 'the fraction above exceeds the fraction below by'),
    ('It cannot identify a training mechanism', 'It identifies a training mechanism'),
    ('$[36.2,40.6]$', '$[36.2,40.7]$'),
    ('covers 98.1\\%', 'covers 98.2\\%'),
    ('Each of 10{,}000 resamples', 'Each of 1{,}000 resamples'),
    ('on the same sampled years', 'on different sampled years'),
    ('retains every occurrence of a repeated year', 'discards repeated years'),
    ('it excludes variation from new training seeds or a new checkpoint choice', 'it includes variation from new training seeds or a new checkpoint choice'),
    ('half the mean Euclidean distance', 'one quarter of the mean Euclidean distance'),
    ('after the original results were available', 'before the original results were available'),
    ('using diffusion checkpoints selected by raw validation score', 'using diffusion checkpoints selected by raw TEST score'),
])
def test_changed_calibration_claim_rejected_in_either_scope(public_records, scope, before, after):
    report, _, _, figure = public_records
    tex, rendered = manuscript()
    if scope == "source":
        assert before in tex
        tex = tex.replace(before, after)
    else:
        before, after = (verifier.prose_text(value, source=True) for value in (before, after))
        assert before in rendered
        rendered = rendered.replace(before, after)
    with pytest.raises(ValueError, match="caption differs|" + scope + ": (clause differs|missing or duplicate clause)"):
        camera.calibration_claims(tex, rendered, report, figure)


@pytest.mark.parametrize("scope", ["source", "PDF"])
@pytest.mark.parametrize("replacement", ["-49.0", "+49.1", "NaN"])
@pytest.mark.parametrize("clause,value", [
    (r"Averaged over the three fits, 49.0\% of observations lie below G5's $v_s$ intervals and 10.5\% above them.", "49.0"),
    (r"Gaussian regression covers 98.1\% of $v_s$ observations, with 0.6\% below and 1.2\% above its intervals.", "98.1"),
])
def test_bad_tail_percentage_cannot_hide_behind_valid_duplicate(
        public_records, scope, replacement, clause, value):
    report, _, _, figure = public_records
    tex, rendered = manuscript()
    if scope == "PDF":
        clause = verifier.prose_text(clause, source=True)
    original = tex if scope == "source" else rendered
    assert original.count(clause) == 1 and clause.count(value) == 1
    damaged = clause.replace(value, replacement, 1)
    changed = original.replace(clause, damaged, 1) + "\n" + clause
    if scope == "source":
        tex = changed
    else:
        rendered = changed
    with pytest.raises(ValueError, match=scope + ": missing or duplicate clause"):
        camera.calibration_claims(tex, rendered, report, figure)


def test_rendered_resample_count_accepts_thousands_separator(public_records):
    report, _, _, figure = public_records
    tex, rendered = manuscript()
    assert "10000 resamples" in rendered
    rendered = rendered.replace("10000 resamples", "10,000 resamples")
    camera.calibration_claims(tex, rendered, report, figure)


def test_swapped_images_cannot_keep_calibration_caption(public_records):
    report, _, _, figure = public_records
    tex, _ = manuscript()
    calibration = "figures/calibration/fig_calibration.pdf"
    operator = "figures/operator/fig_operator.pdf"
    before = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)
    tex = tex.replace(calibration, "SWAP_IMAGE").replace(operator, calibration).replace("SWAP_IMAGE", operator)
    assert sorted(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)) == sorted(before)
    with pytest.raises(ValueError, match="figure|caption|image|calibration"):
        camera.calibration_claims(tex, verifier.prose_text(tex, source=True), report, figure)


@pytest.mark.parametrize("scope", ["source", "PDF"])
def test_reversed_seed_dot_order_is_rejected(public_records, scope):
    report, _, _, figure = public_records
    tex, rendered = manuscript()
    before = "give seeds 7, 13 and 42"
    after = "give seeds 42, 13 and 7"
    if scope == "source":
        assert before in tex
        tex = tex.replace(before, after)
    else:
        # The rendering surrogate normalizes prose before checking the clause.
        before, after = verifier.prose_text(before), verifier.prose_text(after)
        assert before in rendered
        rendered = rendered.replace(before, after)
    with pytest.raises(ValueError, match="caption|clause differs"):
        camera.calibration_claims(tex, rendered, report, figure)


def test_caption_scope_cannot_be_rescued_by_identical_prose_elsewhere(public_records):
    report, _, _, figure = public_records
    tex, _ = manuscript()
    blocks = re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", tex, re.S)
    block = next(block for block in blocks if r"\label{fig:calibration}" in block)
    caption = re.search(r"\\caption\{(.*?)\}\s*\\label\{fig:calibration\}", block, re.S)
    assert caption is not None
    changed = block.replace(caption.group(1), "Placeholder caption.", 1)
    tex = tex.replace(block, changed + "\n" + caption.group(1))
    with pytest.raises(ValueError, match="figure|caption|calibration"):
        camera.calibration_claims(tex, verifier.prose_text(tex, source=True), report, figure)


@pytest.fixture(scope="module")
def checkpoint_claim_records():
    # Clause tests reuse the public records. Their independent numerical and
    # provenance verification belongs to test_prigm_checkpoint_evidence.py.
    root = PAPER / "evidence/followup/checkpoints"
    return {"summary": verifier.read_json(root / "SUMMARY.json"),
            "saved": verifier.read_json(root / "SAVED_VAL25.json")}


def test_checkpoint_clauses_and_rendered_thousands_counts(checkpoint_claim_records):
    tex, rendered = manuscript()
    camera.checkpoint_claims(tex, rendered, checkpoint_claim_records)
    assert "10000 resamples" in rendered and "update 9560 in" in rendered
    rendered = rendered.replace("10000 resamples", "10,000 resamples")
    rendered = rendered.replace("update 9560 in", "update 9,560 in")
    camera.checkpoint_claims(tex, rendered, checkpoint_claim_records)


@pytest.mark.parametrize("scope", ["source", "PDF"])
@pytest.mark.parametrize("before,after", [
    ('All six validation choices agree between 25 and 50 evaluations', 'All six validation choices agree between 25 and 100 evaluations'),
    ('$-0.1238$, $0.0490$', '$-0.1238$, $-0.0490$'),
    ('all three seeds at the last two counts', 'all three seeds at the first two counts'),
    ('score by 0.0945', 'score by 0.0946'),
    ("O-S's by 0.0419", "O-S's by 0.0420"),
    ('changing their difference from $-0.0028$', 'changing their difference from $+0.0028$'),
    ('61.0\\% of resamples', '61.1\\% of resamples'),
    ('every resample under either evaluation count', 'some resamples under either evaluation count'),
    ('both the choices and evaluation count are fixed before any additional TEST samples are generated.', 'both the choices and evaluation count are fixed after any additional TEST samples are generated.'),
    ('fixed before any additional TEST samples are generated', 'fixed after any additional TEST samples are generated'),
    ('This calculation leaves the roles of regularization and overfitting unresolved', 'This calculation establishes the roles of regularization and overfitting'),
    ('contributes both validation months', 'contributes one validation month'),
    ('Each occurrence of a sampled year', 'One occurrence of each sampled year'),
    ('every checkpoint uses the same resampled years', 'every checkpoint uses different resampled years'),
    ('The model weights and draws remain constant', 'The model weights and draws change'),
    ('Variation under fresh sampling noise and performance on a new population remain undetermined', 'Variation under fresh sampling noise and performance on a new population are determined'),
    ('crosses mark P50', 'crosses mark P25'),
    ('evaluates choices already made on VAL', 'evaluates choices made using TEST'),
])
def test_revised_checkpoint_clauses_reject_changes(checkpoint_claim_records, scope, before, after):
    tex, rendered = manuscript()
    if scope == "source":
        assert before in tex
        tex = tex.replace(before, after)
    else:
        before, after = (verifier.prose_text(value, source=True) for value in (before, after))
        assert before in rendered
        rendered = rendered.replace(before, after)
    with pytest.raises(ValueError, match="caption differs|" + scope + ": (clause differs|missing or duplicate clause)"):
        camera.checkpoint_claims(tex, rendered, checkpoint_claim_records)
