"""Check illustration clauses in TeX and a normalized-text surrogate."""
import re

import pytest

from experiments import verify_prigm_paper as gate


@pytest.fixture
def prose():
    source = (gate.PAPER / "main.tex").read_text()
    rendered = source
    for number, label in enumerate(re.findall(r"\\label\{(fig:[^}]+)\}", source), 1):
        rendered = rendered.replace(r"\ref{" + label + "}", str(number))
    return source, gate.clean_cell(rendered)


def test_illustration_clauses_bind_complete_source(prose):
    source, rendered = prose
    figure_note = "D, O-S and G5 share each row's scale; Gaussian uses the right axis."
    gate.illustration_claims(source, figure_note + "\n" + rendered)


MUTATIONS = (
    ("Colored lines show marginal medians", "Colored lines show joint medians"),
    ("marginal predictive intervals from all 39 samples", "simultaneous intervals from 117 samples"),
    ("the right gives mean absolute", "the right gives signed"),
    ("The joined medians can be irregular", "Connecting the medians must produce a smooth curve"),
    ("is a separate property", "is established by these bands"),
    ("residual covariance is common to every context", "different residual covariances for different contexts"),
    ("The frequency of these particular error patterns across TEST", "These particular errors always occur throughout TEST"),
    ('Windows 120 and 360', 'Windows 121 and 360'),
    ('Diffusion uses seed 7, the first', 'Diffusion uses seed 13, the first'),
    ('training step 9560 for D', 'training step 29636 for D'),
    ('noise realization from seed 13', 'noise realization from seed 7'),
    ('although TEST had already been examined', 'before TEST was examined'),
    ('These examples illustrate individual reconstructions, with uncertain representativeness across TEST', 'These two hours are representative of all TEST windows'),
    ('The colored curve joins the marginal medians of 39 samples at each time and coordinate', 'The colored curve gives the joint-trajectory coordinate from 39 samples'),
    ('medians of 39 samples', 'medians of 117 samples pooled across seeds'),
    ('values 2 and 38', 'the third to the thirty-seventh values'),
    ('differ from every sampled trajectory', 'always reproduces a sampled trajectory'),
    ('These curves and intervals concern each time and quantity separately', 'The curves and intervals give simultaneous coverage of a trajectory'),
    ('different sequences of minute averages can produce the same statistics', 'which uniquely determine the sequence of minute averages'),
    ('ensemble members 0, 1 and 2', 'ensemble members 1, 2 and 3'),
    ('the starts of bins of five minutes', 'the centers of five-minute bins'),
    ('show predictions before clipping', 'Predictions are shown after clipping'),
    ('small measured variances', 'large measured variances'),
    ('equally spaced categories', 'logarithmically spaced on the horizontal axis'),
    ('equal weight to the months present', 'equal weight to individual windows'),
    ('solid black diamonds show the mean across three seeds', 'solid black diamonds mark the median across three seeds'),
    ('solid black diamonds show the mean across three seeds', 'hollow black diamonds mark the mean across three seeds'),
    ('Reconstruction from ACE measurements on 3 August 2010, TEST window 120', 'Reconstruction of TEST window 121'),
    ('linear within', 'linear throughout within'),
    ('0.05', '0.5'),
    ('equal vertical distances can represent different absolute errors', 'equal vertical distances represent the same absolute error'),
    ('Reconstruction from Wind measurements on 4 August 2020, TEST window 360', 'Reconstruction of TEST window 361'),
    ('D, O-S and G5 share a linear scale', 'D, O-S and G5 share a logarithmic scale'),
    ('Gaussian regression uses a separate linear scale on the right', 'Gaussian regression uses the same linear scale on the left'),
    ('comparable on a common scale', 'comparable across all scales'),
    ('inclusive nominal', 'excluding both endpoints'),
    ('Every subplot includes all measurements, medians and interval ends in its axis limits', 'The axis limits exclude some measurements, medians and interval ends'),
    ('predictions below it are physically impossible because variance is nonnegative', 'predictions below it satisfy the nonnegativity of variance'),
    ('diffusion generally places its median above the withheld variance', 'diffusion medians generally fall below the withheld variance'),
    ("Gaussian regression's much wider intervals", 'with much narrower intervals'),
    ('zero of the twelve hidden bins', 'one of the twelve hidden bins'),
    ('cover zero, one and two bins', 'cover one, two and three bins'),
    ('contain all twelve withheld values in both windows', 'covers eleven bins in both windows'),
    ('its distribution places probability on negative', 'its distribution assigns zero probability to negative'),
    ('The aggregate coverage results show the same shortfall from nominal coverage for this quantity', 'The aggregate results also show that diffusion intervals cover the variance as often as nominal'),
    ('and their origin during training, remain uncertain', 'but their training causes are established'),
    ('Increasing the penalty lowers the mean absolute residual', 'increasing the penalty increases the mean residual'),
    ('the mean energy score first falls and then rises', 'the mean energy score first rises and then falls'),
    ('Available measurements retain their values during sampling', 'changes the available measurements'),
    ('with short line breaks', 'joining the transitions'),
)


@pytest.mark.parametrize("side", [0, 1], ids=["source", "rendered"])
@pytest.mark.parametrize("operation", ["replace", "omit"])
@pytest.mark.parametrize("old,new", MUTATIONS)
def test_illustration_convention_mutations_fail(prose, side, operation, old, new):
    values = list(prose)
    assert old in values[side]
    values[side] = values[side].replace(old, new if operation == "replace" else "")
    location = "source" if side == 0 else "PDF"
    with pytest.raises(ValueError, match=location + ": (clause differs|missing or duplicate clause)"):
        gate.illustration_claims(*values)


@pytest.mark.parametrize("side", [0, 1], ids=["source", "rendered"])
def test_bad_time_axis_claim_cannot_hide_behind_valid_duplicate(prose, side):
    values = list(prose)
    clause = "The horizontal axes mark the starts of bins of five minutes, relative to the missing hour."
    assert values[side].count(clause) == 1
    values[side] = values[side].replace(clause, clause.replace("starts", "centers"), 1) + "\n" + clause
    location = "source" if side == 0 else "PDF"
    with pytest.raises(ValueError, match=location + ": missing or duplicate clause"):
        gate.illustration_claims(*values)


@pytest.mark.parametrize("side", [0, 1], ids=["source", "rendered"])
def test_bad_variance_claim_cannot_hide_behind_valid_duplicate(prose, side):
    values = list(prose)
    clause = "predictions below it are physically impossible because variance is nonnegative."
    assert values[side].count(clause) == 1
    values[side] = values[side].replace(clause, clause.replace("impossible", "possible"), 1) + "\n" + clause
    location = "source" if side == 0 else "PDF"
    with pytest.raises(ValueError, match=location + ": missing or duplicate clause"):
        gate.illustration_claims(*values)


def test_illustration_image_cannot_swap_case_label(prose):
    source, rendered = prose
    source = source.replace("case_120/fig_imputation.pdf", "case_swap/fig_imputation.pdf")
    source = source.replace("case_360/fig_imputation.pdf", "case_120/fig_imputation.pdf")
    source = source.replace("case_swap/fig_imputation.pdf", "case_360/fig_imputation.pdf")
    with pytest.raises(ValueError, match="label and image differ"):
        gate.illustration_claims(source, rendered)


@pytest.mark.parametrize("marker", ["fig_fixed_final", "fig_imputation", "app:imputation", "fig:fixedfinal"])
def test_illustrations_declaration_is_detected(tmp_path, marker):
    assert gate.illustrations_claimed(tmp_path, marker)


def test_illustrations_directory_is_detected(tmp_path):
    (tmp_path / "evidence/imputation").mkdir(parents=True)
    assert gate.illustrations_claimed(tmp_path, "")


def test_unclaimed_historical_bundle_stays_unclaimed(tmp_path):
    assert not gate.illustrations_claimed(tmp_path, "Historical completed paper.")


@pytest.mark.parametrize("side", [0, 1], ids=["source", "rendered"])
def test_bad_setting_spacing_cannot_hide_behind_valid_duplicate(prose, side):
    values = list(prose)
    clause = "The horizontal axis treats the coefficient settings as equally spaced categories."
    assert values[side].count(clause) == 1
    values[side] = values[side].replace(clause, clause.replace("equally", "logarithmically"), 1) + "\n" + clause
    location = "source" if side == 0 else "PDF"
    with pytest.raises(ValueError, match=location + ": missing or duplicate clause"):
        gate.illustration_claims(*values)


@pytest.mark.parametrize("side", [0, 1], ids=["source", "rendered"])
def test_bad_interval_endpoints_cannot_hide_behind_valid_duplicate(prose, side):
    values = list(prose)
    clause = r"After sorting those samples in increasing order, values 2 and 38 form the inclusive nominal 90\% interval."
    if side == 1:
        clause = gate.clean_cell(clause)
    assert values[side].count(clause) == 1
    values[side] = values[side].replace(clause, clause.replace("inclusive", "exclusive"), 1) + "\n" + clause
    location = "source" if side == 0 else "PDF"
    with pytest.raises(ValueError, match=location + ": (clause differs|missing or duplicate clause)"):
        gate.illustration_claims(*values)



@pytest.mark.parametrize("side", [0, 1], ids=["source", "PDF"])
@pytest.mark.parametrize("value", ["999", "-999", "NaN"])
@pytest.mark.parametrize("valid_duplicate", [False, True])
@pytest.mark.parametrize("clause,slot", [
    (r"Figures~\ref{fig:imputation120} and \ref{fig:imputation360} show predictions before clipping.", r"\ref{fig:imputation120}"),
    (r"Figures~\ref{fig:imputation120} and \ref{fig:imputation360} show predictions before clipping.", r"\ref{fig:imputation360}"),
    ("Windows 120 and 360, counting from zero, are the midpoints of the two halves of the ordered list of 480 windows.", "120"),
    ("Windows 120 and 360, counting from zero, are the midpoints of the two halves of the ordered list of 480 windows.", "360"),
    ("Diffusion uses seed 7, the first declared seed.", "7"),
    (r"In TEST window 120, every diffusion method covers $v_s$ in zero of the twelve hidden bins.", "120"),
    ("In window 360, D, O-S and G5 cover zero, one and two bins, respectively.", "360"),
])
def test_illustration_numeric_slots_reject_malformed_values_and_valid_duplicates(
        prose, side, value, valid_duplicate, clause, slot):
    gate.illustration_claims(*prose)
    values = list(prose)
    if side == 1:
        labels = re.findall(r"\\label\{(fig:[^}]+)\}", prose[0])
        for number, label in enumerate(labels, 1):
            token = r"\ref{" + label + "}"
            clause, slot = clause.replace(token, str(number)), slot.replace(token, str(number))
        clause, slot = gate.clean_cell(clause), gate.clean_cell(slot)
    assert values[side].count(clause) == 1 and clause.count(slot) == 1
    wrong = clause.replace(slot, value, 1)
    values[side] = values[side].replace(clause, wrong + (" " + clause if valid_duplicate else ""), 1)
    message = "missing or duplicate clause" if valid_duplicate else "clause differs"
    location = "source" if side == 0 else "PDF"
    with pytest.raises(ValueError, match=location + ": " + message):
        gate.illustration_claims(*values)
