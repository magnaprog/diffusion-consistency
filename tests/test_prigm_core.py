"""CPU tests for isolated PriGM modules. No OMNI download, no GPU jobs."""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prigm.calendar import role_for_date, select_quota_indices
from prigm.config import (
    N_TRAIN_STARTS,
    ROLE_BUFFER,
    ROLE_EXCLUDED,
    ROLE_TEST,
    ROLE_TRAIN,
    ROLE_VAL,
    TOY_B_REF_SQ,
)
from prigm.controls import interpolation_weights, linear_interpolate
from prigm.keys import digest_u64, noise_key
from prigm.normalize import fit_normalizer, mask_hidden
from prigm.observation import (
    closure_correct,
    derive_vv,
    is_realizable,
    r_from_z,
    residual_inst,
    residual_obs,
    summaries_from_vectors,
    vs_upper_bound,
)
from prigm.precision import arithmetic_tolerance, store_standardized_f32, ulp32
from prigm.checkpoint import weights_only_is_not_resume
from prigm.recovery import (
    PublicationError,
    RunLock,
    load_committed,
    previous_generation,
    publish_generation,
    publish_phase,
    read_phase,
)
from prigm.schedule import build_schedule, ddim_timesteps, pidm_index
from prigm.scoring import (
    coverage_inclusive,
    crps_univariate,
    energy_density_proxy,
    energy_score,
    flatten_primary,
    month_balanced_mean,
    superiority_observed,
    variogram_score,
    primary_variogram_pairs,
)
from prigm.toy import (
    exact_conditional_mean_residual_coeff,
    exact_mean_from_gamma,
    gamma_from_enumeration,
    hmm_forward_backward,
    kappa_alpha,
    path_probability,
    pooled_toy_observations,
    state_summary,
    vectors_state,
)
from prigm.train import milestone_epochs


def test_identity_on_compatible_block():
    rng = np.random.default_rng(0)
    B = rng.normal(size=(5, 3))
    z = summaries_from_vectors(B)
    assert abs(residual_obs(z)) < 1e-12
    assert residual_inst(z) >= -1e-12
    assert is_realizable(z, atol=1e-10)


def test_weak_vs_bound_counterexample():
    z = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    assert abs(residual_obs(z)) < 1e-12
    assert z[4] <= 4.0 * z[3] ** 2
    assert z[4] > vs_upper_bound(z[3], 0.0)
    assert not is_realizable(z, atol=1e-12)


def test_closure_idempotent_and_retains_mu():
    z = np.array([0.2, -0.1, 0.3, 0.1, -0.5, 9.0])
    c1 = closure_correct(z)
    c2 = closure_correct(c1)
    np.testing.assert_allclose(c1, c2)
    np.testing.assert_allclose(c1[:3], z[:3])
    assert is_realizable(c1, atol=1e-12)


def test_g5_derive_vv_kills_equality():
    z5 = np.array([1.0, 0.0, 0.0, 2.0, 0.5])
    z6 = derive_vv(z5)
    assert abs(residual_obs(z6)) < 1e-12


def test_q15_loses_mean_vv():
    from prigm.q15 import q15_from_three

    rng = np.random.default_rng(1)
    z3 = np.stack([summaries_from_vectors(rng.normal(size=(5, 3))) for _ in range(3)])
    q_keep, lost = q15_from_three(z3)
    np.testing.assert_allclose(lost, np.mean(z3[:, 5]))
    np.testing.assert_allclose(q_keep, np.mean(z3[:, 0] ** 2 + z3[:, 1] ** 2 + z3[:, 2] ** 2 + z3[:, 5]))


def test_roles_and_guards():
    assert role_for_date(2012, 1, 14) == ROLE_EXCLUDED
    assert role_for_date(2012, 1, 15) == ROLE_TRAIN
    assert role_for_date(2012, 7, 1) == ROLE_BUFFER
    assert role_for_date(2012, 8, 1) == ROLE_TEST
    assert role_for_date(2012, 11, 1) == ROLE_VAL
    assert role_for_date(2012, 12, 17) == ROLE_VAL
    assert role_for_date(2012, 12, 18) == ROLE_EXCLUDED
    assert role_for_date(2005, 1, 1) == ROLE_EXCLUDED
    assert role_for_date(2024, 12, 31) == ROLE_EXCLUDED


def test_quota_empty_and_spread():
    assert select_quota_indices(0, 8).size == 0
    idx = select_quota_indices(16, 8)
    assert len(idx) == 8
    assert idx[0] == 1  # floor(0.5*16/8)=1
    assert idx[-1] == floor_check()


def floor_check():
    return int(np.floor((7 + 0.5) * 16 / 8))


def test_normalizer_population_and_zero_std():
    z = np.array(
        [
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    n = fit_normalizer(z)
    assert n.n_bins == 2
    assert n.scale[0] > 0  # fallback B_ref because std 0
    np.testing.assert_allclose(n.inverse(n.transform(z)), z)


def test_schedule_sigma1_and_pidm_map():
    sch = build_schedule()
    assert sch["alpha_bar"][0] == 1.0
    assert sch["sigma"][1] == sch["sigma"][2]
    raw1 = sch["betas"][1] * (1 - sch["alpha_bar"][0]) / (1 - sch["alpha_bar"][1])
    assert abs(raw1) < 1e-15
    assert pidm_index(1) == 0
    assert pidm_index(100) == 99
    grid = ddim_timesteps(50)
    assert grid[0] == 100
    assert grid[-1] == 1
    assert len(np.unique(grid)) == len(grid)
    assert np.all(np.diff(grid) < 0)
    # coefficient ratio diagnostic at t=100, c_R=0.005
    w = sch["w"][100]
    sig = sch["sigma"][100]
    ratio = (0.005 / (2 * sig)) / w
    assert abs(ratio - 10307) / 10307 < 0.02


def test_energy_score_closed_form():
    # Identical atoms: ES=0.
    X = np.zeros((3, 1))
    assert energy_score(X, np.zeros(1)) == 0.0
    # All five draws equal 1, so every between-draw distance is zero.
    # Against the observed value 0, only the distance-to-observation term remains: ES=1.
    X = np.ones((5, 1))
    assert energy_score(X, np.zeros(1)) == 1.0
    # Bernoulli(p) law vs independent Bernoulli(p) target: E[ES]=p(1-p).
    # For p=0.1 that is 0.09 (plan enumeration check).
    p = 0.1
    # Two-atom exact: P(X=1)=p, K->inf limit vs Y=1 is (1-p) - p(1-p) = (1-p)^2
    # Finite closed form for K=2 samples {0,1} vs y=0:
    # term1=(0+1)/2=0.5; offdiag pair distances 1+1, mean=1; ES=0.5-0.5=0.
    X = np.array([[0.0], [1.0]])
    assert energy_score(X, np.zeros(1)) == 0.0
    # Expected ES under Y~Bern(p) for the true law equals p(1-p).
    # Compute by enumerating Y in {0,1} with a large iid ensemble.
    rng = np.random.default_rng(1)
    K = 4000
    X = rng.binomial(1, p, size=(K, 1)).astype(np.float64)
    es0 = energy_score(X, np.array([0.0]))
    es1 = energy_score(X, np.array([1.0]))
    expected = (1 - p) * es0 + p * es1
    assert abs(expected - p * (1 - p)) < 0.01


def test_month_balance_and_empty_month_not_filled():
    vals = np.array([1.0, 1.0, 3.0])
    months = np.array([8, 8, 9])
    assert month_balanced_mean(vals, months) == 2.0


@pytest.mark.parametrize("score", [energy_score, crps_univariate])
@pytest.mark.parametrize("sample_dim,target_dim", [(60, 1), (1, 60)])
def test_scores_reject_broadcasting_target_coordinates(score, sample_dim, target_dim):
    with pytest.raises(ValueError, match="coordinates"):
        score(np.ones((39, sample_dim)), np.zeros(target_dim))


@pytest.mark.parametrize("n_draws", [0, 1])
def test_crps_requires_two_draws_for_offdiagonal_estimator(n_draws):
    with pytest.raises(ValueError, match="K>=2"):
        crps_univariate(np.ones((n_draws, 1)), np.zeros(1))


def test_scores_match_three_atom_distances():
    samples = np.array([[0., 0.], [3., 0.], [0., 4.]])
    target = np.array([1., 1.])
    # Unordered inter-sample distances are 3, 4, 5; their score term is 2.
    expected = (np.sqrt(2) + np.sqrt(5) + np.sqrt(10)) / 3 - 2
    assert energy_score(samples, target) == pytest.approx(expected)
    np.testing.assert_allclose(crps_univariate(samples, target), [1 / 3, 1 / 3])


@pytest.mark.parametrize("shape", [(12, 5), (12, 7), (60,)])
def test_flatten_primary_requires_twelve_six_coordinate_bins(shape):
    with pytest.raises(ValueError, match="12, 6"):
        flatten_primary(np.zeros(shape))


def test_flatten_primary_keeps_bin_order_and_excludes_redundant_coordinate():
    z = np.arange(2 * 12 * 6).reshape(2, 12, 6)
    expected = np.array([[v for row in sample for v in row[:5]] for sample in z])
    np.testing.assert_array_equal(flatten_primary(z), expected)
    z[..., 5] = -1000
    np.testing.assert_array_equal(flatten_primary(z[0]), expected[0])


def test_coverage_order_stats():
    X = np.arange(39, dtype=np.float64)[:, None]
    y = np.array([1.0])
    cov = coverage_inclusive(X, y, 2, 38)
    assert cov[0]


def test_interpolation_weight():
    w = interpolation_weights()
    assert abs(w[0] - (18 - 17) / 13) < 1e-15
    assert abs(w[-1] - (29 - 17) / 13) < 1e-15
    z = np.zeros((48, 6))
    z[17] = 0.0
    z[30] = 13.0
    pred = linear_interpolate(z)
    np.testing.assert_allclose(pred[18, 0], 1.0)
    np.testing.assert_allclose(pred[29, 0], 12.0)


def test_toy_energy_match_and_b_ref():
    z = pooled_toy_observations()
    q = z[:, 3] ** 2 + z[:, 4]
    np.testing.assert_allclose(q[0::2], 1.0, atol=1e-12)
    np.testing.assert_allclose(q[1::2], 4.0, atol=1e-12)
    assert abs(q.mean() - TOY_B_REF_SQ) < 1e-12
    for a in (0, 1, 2):
        for s in (0, 1):
            assert abs(residual_obs(state_summary(a, s))) < 1e-12


def test_toy_residual_coefficients():
    c0 = exact_conditional_mean_residual_coeff(0)
    c1 = exact_conditional_mean_residual_coeff(1)
    c2 = exact_conditional_mean_residual_coeff(2)
    np.testing.assert_allclose(c0, 11.0 / 3.0 - kappa_alpha(0) ** 2, atol=1e-12)
    assert abs(c0 - 2.666667) < 1e-5
    assert abs(c1 - 0.303564) < 1e-5
    assert abs(c2 + 0.429006) < 1e-5


@pytest.mark.parametrize("alpha", [0, 1, 2])
@pytest.mark.parametrize("q", [0., .1, .5, .9, 1.])
def test_toy_posterior_mean_residual(alpha, q):
    z0, z1 = (state_summary(alpha, state) for state in (0, 1))
    posterior_mean = (1 - q) * z0 + q * z1
    coefficient = (11.0 / 3.0 - kappa_alpha(alpha) ** 2) / (1 + 2 * alpha ** 2)
    np.testing.assert_allclose(
        residual_obs(posterior_mean), q * (1 - q) * coefficient, rtol=0, atol=1e-12,
    )


def test_hmm_matches_enumeration():
    gamma_e = gamma_from_enumeration(0, 0, 0.1)
    gamma_h = hmm_forward_backward(0, 0, 0.1)
    np.testing.assert_allclose(gamma_e, gamma_h, atol=1e-12)
    # all-zero path probability at alpha=1 p=0.1 boundaries 00
    p0 = path_probability(np.zeros(12, dtype=int), 0, 0, 0.1)
    assert abs(p0 - 0.481881) < 1e-5
    assert abs(gamma_h[5, 1] - 0.276365) < 1e-6


def test_hmm_diagnostics_counts():
    from prigm.hmm_diagnostics import N_CONTEXT_PER_REGIME, N_REGIMES, TIMESTEPS

    assert N_REGIMES * N_CONTEXT_PER_REGIME * len(TIMESTEPS) == 144


def test_stationary_boundaries():
    from prigm.toy_windows import sample_boundaries
    from prigm.toy import transition_matrix

    rng = np.random.default_rng(0)
    p = 0.1
    T13 = np.linalg.matrix_power(transition_matrix(p), 13)
    n = 20000
    aa = 0
    for _ in range(n):
        a, b = sample_boundaries(p, rng)
        if a == 0 and b == 0:
            aa += 1
    # P(aa)=0.5 * T13[0,0]
    expect = 0.5 * T13[0, 0]
    assert abs(aa / n - expect) < 0.02


def test_keys_stable_not_python_hash():
    a = noise_key("eval", "VAL", "w1", 0, 7)
    b = noise_key("eval", "VAL", "w1", 0, 7)
    assert a == b
    assert a != noise_key("eval", "VAL", "w1", 1, 7)
    assert digest_u64("x") == digest_u64("x")


def test_milestones():
    assert milestone_epochs(3, 3) == [3]
    q = milestone_epochs(31, 3)
    assert q[0] == 3
    assert q[-1] == 31
    assert len(q) <= 5


def test_float32_roundtrip_can_break_strict_tol():
    # recorded: alpha=0 state 1 after f32 storage
    z = state_summary(0.0, 1)
    n = fit_normalizer(pooled_toy_observations())
    u = n.transform(z)
    u32 = store_standardized_f32(u)
    z2 = n.inverse(u32.astype(np.float64))
    gap = z2[3] - r_from_z(z2)
    tol = arithmetic_tolerance(z2[3], r_from_z(z2), np.sqrt(n.b_ref_sq))
    # documented to exceed arithmetic tolerance
    assert abs(gap) > float(tol)


def test_recovery_publish_and_hash(tmp_path):
    root = tmp_path / "run"
    payload = {"model.bin": b"abc", "opt.bin": b"def"}
    publish_generation(root, 0, payload, {"step": 0})
    meta = load_committed(root, 0)
    assert meta["step"] == 0
    publish_generation(root, 100, {"model.bin": b"xyz"}, {"step": 100})
    assert previous_generation(root, 100) == 0
    with pytest.raises(PublicationError):
        publish_generation(root, 100, {"model.bin": b"no"}, {"step": 100})
    publish_phase(root, "TEST_STARTED", {"E_star": 3})
    ph = read_phase(root)
    assert ph["phase"] == "TEST_STARTED"


def test_lock_exclusive(tmp_path):
    root = tmp_path / "run2"
    with RunLock(root):
        with pytest.raises(PublicationError):
            with RunLock(root):
                pass


def test_weights_only_rejected():
    with pytest.raises(PublicationError):
        weights_only_is_not_resume({"model": {}})


def test_inventory_constants_match_plan():
    assert N_TRAIN_STARTS == 122309
    from prigm.config import N_VAL_WINDOWS, N_TEST_WINDOWS, N_EPOCH_UPDATES, FINAL_BATCH

    assert N_VAL_WINDOWS == 319
    assert N_TEST_WINDOWS == 480
    assert N_EPOCH_UPDATES == 956
    assert FINAL_BATCH == 69


def test_energy_proxy_formula():
    z = np.zeros((12, 6))
    z[:, 3] = 1.0
    u = energy_density_proxy(z)
    from prigm.config import ENERGY_PREFACTOR, MU0

    expect = ENERGY_PREFACTOR * 12.0 / (24.0 * MU0)
    np.testing.assert_allclose(u, expect)


def test_variogram_zero_score_and_primary_pairs():
    X = np.ones((5, 4))
    y = np.zeros(4)
    pairs = [(0, 1)]
    s = variogram_score(X, y, pairs)
    # a=0, b=0, score 0
    assert s == 0.0
    pairs = primary_variogram_pairs()
    assert (0, 5) in pairs or (0, 1) in pairs
