"""Check illustration clauses in TeX and a normalized-text surrogate."""
import pytest

from experiments import verify_prigm_paper as gate


@pytest.fixture
def prose():
    source = (gate.PAPER / "main.tex").read_text()
    return source, gate.clean_cell(source)


def test_illustration_clauses_bind_complete_source(prose):
    source, rendered = prose
    figure_note = "D, O-S and G5 share each row's scale; Gaussian uses the right axis."
    gate.illustration_claims(source, figure_note + "\n" + rendered)


MUTATIONS = (
    ("Connecting the medians can produce an irregular curve", "Connecting the medians must produce a smooth curve"),
    ("requires separate assessment", "is established by these bands"),
    ("same residual covariance for every context", "different residual covariances for different contexts"),
    ("How often the particular errors in these examples occur across TEST", "These particular errors always occur throughout TEST"),
    ('Windows 120 and 360', 'Windows 121 and 360'),
    ('Diffusion uses seed 7, the first', 'Diffusion uses seed 13, the first'),
    ('training step 9560 for D', 'training step 29636 for D'),
    ('noise realization from seed 13', 'noise realization from seed 7'),
    ('after TEST had already been examined', 'before TEST was examined'),
    ('How representative these two missing hours are across TEST remains uncertain', 'These two hours are representative of all TEST windows'),
    ('At each time and coordinate, the colored curve is the marginal median of the 39 samples', 'The colored curve gives the joint-trajectory median of the 39 samples'),
    ('median of the 39 samples', 'median of 117 samples pooled across seeds'),
    ('the second and thirty-eighth values', 'the third and thirty-seventh values'),
    ('differs from every sampled trajectory', 'always reproduces a sampled trajectory'),
    ('The curves and intervals describe each coordinate at each time separately', 'The curves and intervals give simultaneous coverage of a trajectory'),
    ('different sequences of minute averages can produce the same statistics', 'the statistics uniquely determine the sequence of minute averages'),
    ('ensemble members 0, 1 and 2', 'ensemble members 1, 2 and 3'),
    ('the starts of five-minute bins', 'the centers of five-minute bins'),
    ('Predictions are shown before clipping', 'Predictions are shown after clipping'),
    ('small measured variances', 'large measured variances'),
    ('equally spaced on the horizontal axis', 'logarithmically spaced on the horizontal axis'),
    ('equal weight to the months present', 'equal weight to individual windows'),
    ('solid black diamonds mark the mean across three seeds', 'solid black diamonds mark the median across three seeds'),
    ('solid black diamonds mark the mean across three seeds', 'hollow black diamonds mark the mean across three seeds'),
    ('Reconstruction of TEST window 120', 'Reconstruction of TEST window 121'),
    ('linear within', 'linear throughout within'),
    ('0.05', '0.5'),
    ('equal vertical distances can represent different absolute errors', 'equal vertical distances represent the same absolute error'),
    ('Reconstruction of TEST window 360', 'Reconstruction of TEST window 361'),
    ('D, O-S and G5 share a linear scale', 'D, O-S and G5 share a logarithmic scale'),
    ('Gaussian regression uses a separate linear scale on the right', 'Gaussian regression uses the same linear scale on the left'),
    ('comparable only on the same scale', 'comparable across all scales'),
    ('includes the endpoints', 'excludes the endpoints'),
    ('The axis limits include all measurements, medians and interval ends in each subplot', 'The axis limits exclude some measurements, medians and interval ends'),
    ('predictions below it violate the nonnegativity of variance', 'predictions below it satisfy the nonnegativity of variance'),
    ('diffusion medians generally exceed the withheld variance', 'diffusion medians generally fall below the withheld variance'),
    ('with much wider intervals', 'with much narrower intervals'),
    ('zero of the twelve hidden bins', 'one of the twelve hidden bins'),
    ('cover zero, one and two bins', 'cover one, two and three bins'),
    ('covers all twelve bins in both windows', 'covers eleven bins in both windows'),
    ('its distribution assigns probability to negative', 'its distribution assigns zero probability to negative'),
    ('The aggregate results also show that diffusion intervals cover the variance less often than nominal', 'The aggregate results also show that diffusion intervals cover the variance as often as nominal'),
    ('and how they arise during training, remain uncertain', 'and how they arise during training, are established'),
    ('the mean residual decreases as the penalty grows', 'the mean residual increases as the penalty grows'),
    ('the mean energy score first improves and then worsens', 'the mean energy score first worsens and then improves'),
    ('keeps the available measurements unchanged', 'changes the available measurements'),
    ('leaving short breaks', 'joining the transitions'),
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
    clause = "Times mark the starts of five-minute bins relative to the missing hour."
    assert values[side].count(clause) == 1
    values[side] = values[side].replace(clause, clause.replace("starts", "centers"), 1) + "\n" + clause
    location = "source" if side == 0 else "PDF"
    with pytest.raises(ValueError, match=location + ": missing or duplicate clause"):
        gate.illustration_claims(*values)


@pytest.mark.parametrize("side", [0, 1], ids=["source", "rendered"])
def test_bad_variance_claim_cannot_hide_behind_valid_duplicate(prose, side):
    values = list(prose)
    clause = "predictions below it violate the nonnegativity of variance."
    assert values[side].count(clause) == 1
    values[side] = values[side].replace(clause, clause.replace("violate", "satisfy"), 1) + "\n" + clause
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
    clause = "The settings are equally spaced on the horizontal axis."
    assert values[side].count(clause) == 1
    values[side] = values[side].replace(clause, clause.replace("equally", "logarithmically"), 1) + "\n" + clause
    location = "source" if side == 0 else "PDF"
    with pytest.raises(ValueError, match=location + ": missing or duplicate clause"):
        gate.illustration_claims(*values)


@pytest.mark.parametrize("side", [0, 1], ids=["source", "rendered"])
def test_bad_interval_endpoints_cannot_hide_behind_valid_duplicate(prose, side):
    values = list(prose)
    clause = r"The nominal 90\% interval includes the endpoints given by the second and thirty-eighth values in increasing order."
    if side == 1:
        clause = gate.clean_cell(clause)
    assert values[side].count(clause) == 1
    values[side] = values[side].replace(clause, clause.replace("includes", "excludes"), 1) + "\n" + clause
    location = "source" if side == 0 else "PDF"
    with pytest.raises(ValueError, match=location + ": (clause differs|missing or duplicate clause)"):
        gate.illustration_claims(*values)
