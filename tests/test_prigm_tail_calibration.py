"""Check nominal interval ranks, tail frequencies and their aggregation.

For 39 draws, use the second and thirty-eighth ordered values as inclusive
interval endpoints. Cycling a held-out observation through forty distinct
ranks gives the 5% below, 90% inside and 5% above proportions under
exchangeability. Separate tied-value cases check that endpoint equality
stays inside; the rank example is not a calibration claim about fitted models.

Further cases check month and model weighting and compute below-minus-above
differences within each paired resample. They also require valid input
identities before private data can be loaded.
"""

import numpy as np
import pytest

from analysis.prigm_tail_calibration import interval_fractions, summarize, analyze


def test_endpoints_are_inclusive_and_tails_have_exchangeable_reference():
    # Cycle a target through all forty ranks of an exchangeable finite sample.
    samples, targets = [], []
    for held_out in range(40):
        samples.append(np.broadcast_to(np.delete(np.arange(40.), held_out)[:, None, None], (39, 12, 5)))
        targets.append(np.full((12, 5), held_out, dtype=float))
    rates = interval_fractions(samples, targets)
    np.testing.assert_allclose(rates.mean(axis=(0, 1)), [.05, .9, .05], atol=1e-15)
    draws = np.broadcast_to(np.arange(39.)[None, :, None, None], (2, 39, 12, 5))
    truth = np.stack([np.full((12, 5), 1.), np.full((12, 5), 37.)])
    np.testing.assert_array_equal(interval_fractions(draws, truth)[..., 1], 1)


def test_ties_stay_inside_and_each_coordinate_has_exact_partition():
    rates = interval_fractions(np.ones((2, 39, 12, 5)), np.ones((2, 12, 5)))
    np.testing.assert_array_equal(rates[..., 1], 1)
    np.testing.assert_array_equal(rates.sum(axis=-1), 1)


@pytest.mark.parametrize("shape", [(39, 12, 5), (1, 38, 12, 5), (1, 39, 11, 5), (1, 39, 12, 6), (0, 39, 12, 5)])
def test_invalid_draw_shape_rejected(shape):
    with pytest.raises(ValueError):
        interval_fractions(np.zeros(shape), np.zeros((1, 12, 5)))


def test_target_broadcasting_and_nonfinite_inputs_rejected():
    samples = np.zeros((1, 39, 12, 5))
    with pytest.raises(ValueError):
        interval_fractions(samples, np.zeros((12, 5)))
    samples[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        interval_fractions(samples, np.zeros((1, 12, 5)))


def test_month_and_model_weights_and_repeated_month_bootstrap():
    values = np.zeros((2, 3, 5, 3))
    values[0, :2, :, 0] = 1
    values[0, 2, :, 2] = 1
    values[1, :, :, 1] = 1
    result = summarize(values, [200508, 200508, 200509], {"year": np.array([[2 / 3, 1 / 3]])})
    np.testing.assert_allclose(result["mean"], np.tile([.25, .5, .25], (5, 1)))
    np.testing.assert_allclose(result["year_95"][0], np.tile([1 / 3, .5, 1 / 6], (5, 1)))


def test_tail_asymmetry_uses_paired_replicates_not_separate_interval_bounds():
    values = np.tile(np.array([[[.1, .9, 0.], [.4, .3, .3]]])[:, :, None, :], (1, 1, 5, 1))
    result = summarize(values, [200508, 200608], {"year": np.eye(2)})
    np.testing.assert_allclose(result["below_minus_above"], .1, atol=1e-15)
    np.testing.assert_allclose(result["year_below_minus_above_95"], .1, atol=1e-15)


def test_existing_output_and_wrong_manifest_fail_before_loading_private_data(tmp_path):
    with pytest.raises(FileExistsError):
        analyze(tmp_path / "absent", tmp_path / "absent", tmp_path)
    evidence = tmp_path / "evidence/real"
    evidence.mkdir(parents=True)
    (evidence / "MANIFEST.json").write_text("{}")
    from prigm.recovery import ContractMismatch
    with pytest.raises(ContractMismatch, match="input hash"):
        analyze(tmp_path / "absent", evidence.parent, tmp_path / "new")
    assert not (tmp_path / "new").exists()
