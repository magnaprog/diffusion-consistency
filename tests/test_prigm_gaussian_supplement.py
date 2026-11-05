"""Focused supplementary table arithmetic and exact source/PDF row bindings."""
import numpy as np
import pytest

from experiments import verify_prigm_paper as gate
from prigm.normalize import AffineNormalizer
from prigm.observation import derive_vv
from tests.test_prigm_deadline_paper import ERRORS, public_originals
from tests.test_prigm_paper import table


@pytest.fixture(scope="module")
def supplementary_case():
    normalizer = AffineNormalizer(np.zeros(6), np.ones(6), 1., 144)
    truth = np.broadcast_to(derive_vv(np.tile([0., 0., 0., 2., 0.], (48, 1))), (3, 48, 6))
    physical = np.broadcast_to(truth[:, None], (3, 39, 48, 6)).copy()
    draws = np.arange(39) / 19
    physical[:, :, 18:30, 0] = np.array([1., 3., 9.])[:, None, None] * draws[None, :, None]
    physical[:, :, 18:30, 4] = -1.
    physical[:, :, 18:30] = derive_vv(physical[:, :, 18:30, :5])
    ensemble = physical.transpose(0, 1, 3, 2)
    months = np.array([202508, 202508, 202509])
    return ensemble, truth, normalizer, months


def test_supplement_preserves_draw_rmse_month_weights_and_scoring_kind(supplementary_case):
    ensemble, truth, nrm, months = supplementary_case
    before = ensemble.copy()
    result = gate.gaussian_supplemental_metrics(ensemble, truth, nrm, months, ("raw", "corrected"))
    raw, corrected = result["raw"], result["corrected"]
    assert raw["seam_l_mean"]["month_balanced"] == pytest.approx(5.5)
    assert raw["seam_l_mean"]["mean"] == pytest.approx(13 / 3)
    assert raw["seam_r_mean"] == raw["seam_l_mean"]
    assert raw["internal_mean"]["month_balanced"] == 0.
    expected_rmse = 5.5 * np.sqrt(np.mean((np.arange(39) / 19)**2))
    assert raw["seam_l_rmse"]["month_balanced"] == pytest.approx(expected_rmse)
    assert raw["seam_r_rmse"] == raw["seam_l_rmse"]
    assert raw["cov90_mean"]["month_balanced"] == pytest.approx(.6)
    assert corrected["cov90_mean"]["month_balanced"] == pytest.approx(.8)
    assert np.asarray(raw["cov90_by_position_channel"]["month_balanced"]).shape == (12, 5)
    assert raw["es_vec"] == corrected["es_vec"]
    assert np.array_equal(ensemble, before)
    only = gate.gaussian_supplemental_metrics(ensemble, truth, nrm, months, ("corrected",))
    assert set(only) == {"corrected"}
    assert only["corrected"] == corrected


@pytest.fixture(scope="module")
def supplementary_tables(public_originals, supplementary_case):
    real, observations, *_ = public_originals
    ensemble, truth, nrm, months = supplementary_case
    supplemental = gate.gaussian_supplemental_metrics(ensemble, truth, nrm, months, ("raw", "corrected"))
    deadline = {"gaussian": {"supplemental_metrics": supplemental}}
    return gate.detailed_table_values(real, observations, deadline), deadline


def test_supplement_is_optional_and_formats_all_four_tables(public_originals, supplementary_tables):
    real, observations, *_ = public_originals
    current, deadline = supplementary_tables
    old = gate.detailed_table_values(real, observations)
    assert sum(map(len, current.values())) - sum(map(len, old.values())) == 6
    for label in ("tab:calibration", "tab:channelcoverage", "tab:alternative", "tab:geometry"):
        assert any(name.startswith("Gaussian regression") for name in current[label])
        assert not any(name.startswith("Gaussian regression") for name in old[label])
        names = list(current[label])
        first = "Gaussian regression raw" if label in ("tab:calibration", "tab:alternative") else "Gaussian regression"
        previous = "Gaussian clipped" if label in ("tab:calibration", "tab:alternative") else "Gaussian"
        assert names.index(first) == names.index(previous) + 1
    assert current["tab:channelcoverage"]["Gaussian regression"] == ["0.0", "100.0", "100.0", "100.0", "0.0"]
    assert current["tab:geometry"]["Gaussian regression"][1:4] == ["5.5000", "5.5000", "0.0000"]
    # Both kinds must come from their respective supplement, not a raw-row alias.
    assert current["tab:calibration"]["Gaussian regression raw"][3] == "60.0"
    assert current["tab:calibration"]["Gaussian regression clipped"][3] == "80.0"
    for label in set(current) - {"tab:calibration", "tab:channelcoverage", "tab:alternative", "tab:geometry"}:
        assert current[label] == old[label]


@pytest.mark.parametrize("label", ["tab:calibration", "tab:channelcoverage", "tab:alternative", "tab:geometry"])
@pytest.mark.parametrize("scope", ["source", "PDF"])
@pytest.mark.parametrize("damage", ["wrong", "missing", "wrong_then_valid"])
def test_supplement_rows_cannot_be_wrong_missing_or_rescued_by_duplicate(supplementary_tables, label, scope, damage):
    rows = supplementary_tables[0][label]
    caption = "Synthetic supplementary diagnostics."
    tex = table(label, rows, caption)
    lines = [name + "  " + "  ".join(values) for name, values in rows.items()]
    rendered = "Table 1: " + caption + "\nProcedure  Values\n" + "\n".join(lines) + "\n"
    gate.bind_table(tex, rendered, label, rows, ("Procedure",))
    name = next(name for name in rows if name.startswith("Gaussian regression"))
    values = rows[name]
    wrong = ["999.9999", *values[1:]]
    if scope == "source":
        original = name + " & " + " & ".join(values) + r" \\"
        false = name + " & " + " & ".join(wrong) + r" \\"
        replacement = "" if damage == "missing" else false
        if damage == "wrong_then_valid": replacement += "\n" + original
        assert tex.count(original) == 1
        tex = tex.replace(original, replacement)
    else:
        original = name + "  " + "  ".join(values)
        false = name + "  " + "  ".join(wrong)
        replacement = "" if damage == "missing" else false
        if damage == "wrong_then_valid": replacement += "\n" + original
        assert rendered.count(original) == 1
        rendered = rendered.replace(original, replacement)
    with pytest.raises(ERRORS):
        gate.bind_table(tex, rendered, label, rows, ("Procedure",))


def test_detailed_tables_forwards_deadline_without_changing_other_rows(public_originals, supplementary_tables, monkeypatch):
    real, observations, *_ = public_originals
    expected, deadline = supplementary_tables
    seen = {}
    monkeypatch.setattr(gate, "bind_table", lambda tex, rendered, label, values, headers: seen.update({label: values}))
    assert gate.detailed_tables("source", "PDF", real, observations, deadline) == expected
    assert seen == expected
