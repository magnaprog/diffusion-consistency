"""Check exact conditional calculations for the toy hidden Markov model.

Independently enumerate hidden binary paths and mixture regimes, including
the declared probability law for the two observed endpoint states. Compare
path probabilities, regime weights and entropies with the production
calculations. Knowing a hidden path need not identify the regime that
generated it.

Separate tests retain the distinction between an exact conditional
distribution and finite scores estimated from its draws. Neither finite
sampler scores nor Monte Carlo self-scores are an exact Bayes floor.
"""

import itertools
import json

import numpy as np
import pytest

from prigm import hmm_diagnostics as diagnostics
from prigm import oracle
from prigm.config import HIDDEN_END, HIDDEN_START, N_BINS, N_HIDDEN
from prigm.normalize import fit_normalizer
from prigm.observation import residual_obs
from prigm.schedule import build_schedule, ddim_timesteps
from prigm.toy import (
    PS, hmm_forward_backward, hmm_posterior, mixture_hmm_posterior,
    pooled_toy_observations, sample_mixture_paths, state_summary,
)
from prigm.toy_windows import window_from_path


PATHS = np.array(list(itertools.product((0, 1), repeat=N_HIDDEN)))
assert PATHS.shape == (4096, 12)


def _entropy(prob):
    positive = prob > 0
    return float(-np.sum(prob[positive] * np.log(prob[positive])))


def _enumerate(a, b, log_emit, law="iid", ps=PS, prior=None):
    """Products over every path, without using any production HMM helper."""
    full = np.column_stack((np.full(len(PATHS), a), PATHS, np.full(len(PATHS), b)))
    changes = full[:, 1:] != full[:, :-1]
    path_mass = np.array([np.prod(np.where(changes, p, 1 - p), axis=1) for p in ps])
    boundary_mass = path_mass.sum(axis=1)
    prior = np.ones(len(ps)) / len(ps) if prior is None else np.asarray(prior) / np.sum(prior)
    context = prior * (boundary_mass if law == "stationary" else 1)
    context /= context.sum()
    row_max = log_emit.max(axis=1)
    le = log_emit - row_max[:, None]
    likelihood = le[np.arange(N_HIDDEN), PATHS].sum(axis=1)
    joint = np.log(context)[:, None] + np.log(path_mass / boundary_mass[:, None]) + likelihood
    log_z = np.logaddexp.reduce(joint.ravel())
    joint = np.exp(joint - log_z)
    probs = joint.sum(axis=0)
    weights = joint.sum(axis=1)
    gamma = np.stack([probs @ (PATHS == s) for s in (0, 1)], axis=1)
    return {
        "gamma": gamma, "weights": weights, "context_weights": context,
        "log_evidence": log_z + row_max.sum(),
        "path_entropy": _entropy(probs), "regime_entropy": _entropy(weights),
        "joint_regime_path_entropy": _entropy(joint),
        "path_entropy_given_regime": _entropy(joint) - _entropy(weights),
        "boundary_mass": boundary_mass, "path_probs": probs,
    }


@pytest.fixture
def nrm():
    return fit_normalizer(pooled_toy_observations())


def _emissions(kind):
    rng = np.random.default_rng(937)
    if kind == "none":
        return np.zeros((N_HIDDEN, 2))
    if kind == "noisy":
        return rng.normal(size=(N_HIDDEN, 2)) * 3 - 9
    if kind == "underflow":
        return rng.integers(-8, 0, size=(N_HIDDEN, 2)).astype(float) - 10000
    if kind == "concentrated":
        return -3000 * (np.arange(N_HIDDEN)[:, None] % 2 != np.arange(2))
    raise AssertionError(kind)


@pytest.mark.parametrize("p", PS)
@pytest.mark.parametrize("a,b", itertools.product((0, 1), repeat=2))
@pytest.mark.parametrize("kind", ("none", "noisy", "underflow", "concentrated"))
def test_hmm_all_4096_paths_each_regime(a, b, p, kind):
    le = _emissions(kind)
    expected = _enumerate(a, b, le, ps=(p,))
    actual = hmm_posterior(a, b, p, log_emit=le)
    np.testing.assert_allclose(actual["gamma"], expected["gamma"], atol=2e-12)
    np.testing.assert_allclose(actual["log_evidence"], expected["log_evidence"], atol=3e-10)
    np.testing.assert_allclose(actual["path_entropy"], expected["path_entropy"], atol=3e-12)
    np.testing.assert_allclose(np.exp(actual["log_boundary_probability"]), expected["boundary_mass"][0], atol=1e-14)
    np.testing.assert_allclose(hmm_forward_backward(a, b, p, log_emit=le), actual["gamma"])


@pytest.mark.parametrize("law", ("iid", "stationary"))
@pytest.mark.parametrize("a,b", itertools.product((0, 1), repeat=2))
@pytest.mark.parametrize("kind", ("none", "noisy", "underflow", "concentrated"))
def test_mixture_all_4096_paths_times_three_p(a, b, law, kind):
    le = _emissions(kind)
    prior = (0.2, 0.5, 0.3)
    expected = _enumerate(a, b, le, law, prior=prior)
    actual = mixture_hmm_posterior(a, b, boundary_law=law, log_emit=le, prior=prior, entropies=True)
    for key in ("gamma", "weights", "context_weights", "log_evidence", "path_entropy", "regime_entropy", "path_entropy_given_regime", "joint_regime_path_entropy"):
        np.testing.assert_allclose(actual[key], expected[key], atol=3e-10, err_msg=key)
    assert actual["path_entropy"] <= actual["joint_regime_path_entropy"] + 1e-12
    assert actual["path_entropy"] >= actual["path_entropy_given_regime"] - 1e-12


def test_boundary_laws_and_joint_not_marginal_entropy():
    iid = mixture_hmm_posterior(0, 0, boundary_law="iid", entropies=True)
    stationary = mixture_hmm_posterior(0, 0, boundary_law="stationary")
    np.testing.assert_allclose(iid["weights"], np.ones(3) / 3, atol=1e-14)
    expected = np.array([(1 + (1 - 2 * p) ** 13) / 2 for p in PS])
    np.testing.assert_allclose(stationary["weights"], expected / expected.sum(), atol=1e-14)
    different = mixture_hmm_posterior(0, 1, boundary_law="stationary")
    assert stationary["weights"][0] > 1 / 3 > different["weights"][0]
    assert iid["path_entropy"] < sum(_entropy(row) for row in iid["gamma"]) - 0.1
    fair = hmm_posterior(0, 1, 0.5)
    np.testing.assert_allclose(fair["path_entropy"], 12 * np.log(2), atol=1e-12)


def test_row_log_offsets_preserve_p_weights_even_when_likelihoods_underflow():
    le = _emissions("underflow") + 10000
    offsets = -1e12 - np.arange(N_HIDDEN)[:, None] * 1e10
    assert np.all(np.exp(le + offsets) == 0)
    for law in ("iid", "stationary"):
        base = mixture_hmm_posterior(0, 1, boundary_law=law, log_emit=le, entropies=True)
        shifted = mixture_hmm_posterior(0, 1, boundary_law=law, log_emit=le + offsets, entropies=True)
        for key in ("gamma", "weights", "path_entropy", "joint_regime_path_entropy"):
            np.testing.assert_allclose(shifted[key], base[key], atol=1e-12)
        np.testing.assert_allclose(shifted["log_evidence"], base["log_evidence"] + offsets.sum(), rtol=0, atol=0.005)


def test_hard_emissions_and_invalid_inputs():
    le = np.full((12, 2), -np.inf)
    le[np.arange(12), PATHS[731]] = 0
    post = mixture_hmm_posterior(1, 0, boundary_law="iid", log_emit=le, entropies=True)
    np.testing.assert_allclose(post["gamma"][:, 1], PATHS[731], atol=1e-14)
    assert abs(post["path_entropy"]) < 1e-12
    np.testing.assert_array_equal(sample_mixture_paths(post, np.random.default_rng(42), 5), np.tile(PATHS[731], (5, 1)))
    np.testing.assert_allclose(hmm_forward_backward(0, 1, 0.1, np.ones((12, 2))), hmm_forward_backward(0, 1, 0.1))
    for kwargs in ({"log_emit": np.full((12, 2), -np.inf)}, {"emit": np.zeros((12, 2))},
                   {"emit": np.ones((12, 2)), "log_emit": le}, {"log_emit": np.zeros((11, 2))}):
        with pytest.raises(ValueError):
            hmm_posterior(0, 1, 0.1, **kwargs)
    with pytest.raises(ValueError, match="boundary_law"):
        mixture_hmm_posterior(0, 0, boundary_law=None)


@pytest.mark.parametrize("law", ("iid", "stationary"))
@pytest.mark.parametrize("noisy", (False, True))
def test_ffbs_samples_mixture_not_an_average_regime_score(law, noisy):
    le = _emissions("noisy" if noisy else "none")
    expected = _enumerate(0, 0, le, law)
    post = mixture_hmm_posterior(0, 0, boundary_law=law, log_emit=le)
    paths = sample_mixture_paths(post, np.random.default_rng(721), 20000)
    np.testing.assert_allclose(paths.mean(axis=0), expected["gamma"][:, 1], atol=0.015)
    np.testing.assert_allclose(np.mean(np.all(paths == 0, axis=1)), expected["path_probs"][0], atol=0.015)


@pytest.mark.parametrize("alpha", (0, 1, 2))
@pytest.mark.parametrize("t", (1, 25, 50, 100))
@pytest.mark.parametrize("law", ("iid", "stationary"))
def test_noisy_posterior_means_and_density_evidence_by_enumeration(alpha, t, law, nrm):
    sched = build_schedule()
    states = nrm.transform(np.array([state_summary(alpha, s) for s in (0, 1)]))
    rng = np.random.default_rng(41)
    u_t = sched["a"][t] * states[PATHS[129]] + sched["s"][t] * rng.standard_normal((12, 6))
    le = np.stack([-0.5 * np.sum(((u_t - sched["a"][t] * state) / sched["s"][t]) ** 2, axis=1)
                   - 3 * np.log(2 * np.pi * sched["s"][t] ** 2) for state in states], axis=1)
    np.testing.assert_allclose(oracle.gaussian_log_emissions(u_t, alpha, t, nrm, sched), le, atol=1e-10)
    expected = _enumerate(0, 1, le, law)
    post = oracle.posterior_hidden(alpha, None, 0, 1, nrm, u_t, t, sched, conditioning="pooled", boundary_law=law, entropies=True)
    for key in ("gamma", "weights", "log_evidence", "path_entropy"):
        np.testing.assert_allclose(post[key], expected[key], atol=2e-10)
    z_hat, gamma = oracle.exact_mean_hidden(alpha, None, 0, 1, nrm, u_t, t, sched, conditioning="pooled", boundary_law=law)
    expected_z = expected["gamma"] @ np.array([state_summary(alpha, s) for s in (0, 1)])
    np.testing.assert_allclose(z_hat, expected_z, atol=1e-12)
    np.testing.assert_allclose(gamma, expected["gamma"], atol=1e-12)
    if t == 1:
        assert post["path_entropy"] < 1e-6  # legitimate fixed-noisy concentration
    if t == 100:
        assert post["path_entropy"] > 5


def test_observed_context_only_and_backward_mean_api(nrm):
    for alpha, a, b in itertools.product((0, 1, 2), (0, 1), (0, 1)):
        z = window_from_path(alpha, 0.1, a, b, PATHS[63])
        z[HIDDEN_START:HIDDEN_END] = np.nan
        assert oracle.infer_context(z, nrm) == (alpha, a, b)
    z_hat, gamma = oracle.exact_mean_hidden(0, 0.1, 0, 0, nrm)
    np.testing.assert_allclose(gamma, hmm_forward_backward(0, 0, 0.1))
    assert np.max(np.abs(residual_obs(z_hat))) > 0.1  # no support projection of a mean
    with pytest.raises(ValueError, match="held-out p"):
        oracle.exact_mean_hidden(0, 0.1, 0, 0, nrm, conditioning="pooled", boundary_law="iid")
    with pytest.raises(ValueError, match="boundary_law"):
        oracle.exact_mean_hidden(0, None, 0, 0, nrm, conditioning="pooled")


def _enumerated_mean(u, t, nrm, sched, law):
    le = oracle.gaussian_log_emissions(u, 1, t, nrm, sched)
    post = _enumerate(0, 0, le, law)
    return post["gamma"] @ np.array([state_summary(1, s) for s in (0, 1)])


@pytest.mark.parametrize("law", ("iid", "stationary"))
def test_two_step_updates_noisy_weights_at_both_calls(nrm, monkeypatch, law):
    sched = build_schedule()
    u = np.random.default_rng(219).normal(size=(12, 6))
    calls = []
    original = oracle.posterior_hidden

    def spy(*args, **kwargs):
        post = original(*args, **kwargs)
        calls.append((args[6], post["weights"].copy()))
        return post

    monkeypatch.setattr(oracle, "posterior_hidden", spy)
    first, endpoint, _ = oracle.two_step_exact(u, 100, 1, None, 0, 0, nrm, sched, conditioning="pooled", boundary_law=law)
    expected_first = _enumerated_mean(u, 100, nrm, sched, law)
    eps = (u - sched["a"][100] * nrm.transform(expected_first)) / sched["s"][100]
    u1 = sched["a"][1] * nrm.transform(expected_first) + sched["s"][1] * eps
    expected_end = _enumerated_mean(u1, 1, nrm, sched, law)
    np.testing.assert_allclose(first, expected_first, atol=1e-12)
    np.testing.assert_allclose(endpoint, expected_end, atol=1e-12)
    assert [c[0] for c in calls] == [100, 1]
    assert np.max(np.abs(calls[0][1] - calls[1][1])) > 0.01
    calls.clear()
    oracle.two_step_exact(u1, 1, 1, None, 0, 0, nrm, sched, conditioning="pooled", boundary_law=law)
    assert len(calls) == 1


@pytest.mark.parametrize("n_eval", (2, 25, 50, 100))
def test_ddim_reconditions_at_each_evaluation_and_ignores_hidden_target(nrm, monkeypatch, n_eval):
    sched = build_schedule()
    noise = np.random.default_rng(12).standard_normal((6, 48))
    z = window_from_path(1, 0.1, 0, 0, PATHS[92])
    z[HIDDEN_START:HIDDEN_END] = np.nan
    calls = []
    original = oracle.exact_mean_hidden

    def spy(alpha, p, a, b, nrm, u, t, sched, **kwargs):
        assert p is None
        actual = original(alpha, p, a, b, nrm, u, t, sched, **kwargs)
        expected = _enumerated_mean(u, t, nrm, sched, "stationary")
        np.testing.assert_allclose(actual[0], expected, atol=3e-11)
        calls.append(t)
        return actual

    monkeypatch.setattr(oracle, "exact_mean_hidden", spy)
    actual = oracle.ddim_exact(z, 1, None, 0, 0, nrm, noise, n_eval, sched, conditioning="pooled", boundary_law="stationary")
    assert calls == list(ddim_timesteps(n_eval))
    assert np.all(np.isfinite(actual))
    if n_eval == 2:
        _, endpoint, _ = oracle.two_step_exact(noise.T[18:30], 100, 1, None, 0, 0, nrm, sched, conditioning="pooled", boundary_law="stationary")
        np.testing.assert_allclose(actual, endpoint, atol=1e-12)


def test_replicated_context_draws_fresh_paths_before_each_corruption(nrm, monkeypatch):
    clean_inputs = []
    original = diagnostics._forward_corrupt

    def spy(u0, t, sched, rng):
        clean_inputs.append(u0.copy())
        return original(u0, t, sched, rng)

    monkeypatch.setattr(diagnostics, "_forward_corrupt", spy)
    ends = diagnostics._replicated_context_endpoints(1, None, 0, 0, nrm, build_schedule(), 1,
                                                      np.random.default_rng(211), 64, conditioning="pooled", boundary_law="iid")
    assert len(clean_inputs) == 64
    assert len(np.unique(np.array(clean_inputs).reshape(64, -1), axis=0)) > 20
    # At low noise, the endpoint recovers each newly sampled clean path.
    np.testing.assert_allclose(nrm.transform(ends), clean_inputs, atol=1e-10)
    context = oracle.posterior_hidden(1, None, 0, 0, nrm, conditioning="pooled", boundary_law="iid")
    ref1 = sample_mixture_paths(context, np.random.default_rng(881), 64)
    diagnostics._replicated_context_endpoints(1, None, 0, 0, nrm, build_schedule(), 25,
                                              np.random.default_rng(123), 2, conditioning="pooled", boundary_law="iid")
    ref2 = sample_mixture_paths(context, np.random.default_rng(881), 64)
    np.testing.assert_array_equal(ref1, ref2)


def _toy_file(path, nrm, p=0.1, boundary_law=None):
    payload = dict(z=window_from_path(1, p, 0, 0, PATHS[91])[None],
                   alpha=np.array([999]), p=np.array([p]), mean=nrm.mean, scale=nrm.scale,
                   b_ref_sq=nrm.b_ref_sq, n_bins=nrm.n_bins)
    if boundary_law is not None:
        payload["boundary_law"] = boundary_law
    np.savez(path, **payload)


@pytest.mark.parametrize("module", (oracle, diagnostics))
def test_cli_requires_information_and_boundary_law(module, tmp_path):
    with pytest.raises(SystemExit):
        module.main(["--toy-eval", str(tmp_path / "absent.npz"), "--output", str(tmp_path / "out.json"), "--conditioning", "pooled"])
    with pytest.raises(SystemExit):
        module.main(["--toy-eval", str(tmp_path / "absent.npz"), "--output", str(tmp_path / "out.json"), "--boundary-law", "iid"])


def test_oracle_cli_budgets_p_strata_no_target_p_leak_and_freshness(tmp_path, nrm):
    records = []
    for p in (0.1, 0.5):
        source, dest = tmp_path / f"toy{p}.npz", tmp_path / f"oracle{p}.json"
        _toy_file(source, nrm, p)
        original = source.read_bytes()
        argv = ["--toy-eval", str(source), "--output", str(dest), "--conditioning", "pooled",
                "--boundary-law", "iid", "--n-windows", "1", "--n-draws", "2"]
        assert oracle.main(argv) == 0
        assert source.read_bytes() == original
        with pytest.raises(FileExistsError):
            oracle.main(argv)
        rec = json.loads(dest.read_text())
        assert rec["boundary_law"] == "iid" and rec["conditioning"] == "pooled"
        assert "no boundary_law field" in rec["boundary_law_source"]
        assert rec["contexts"][0]["alpha"] == 1  # ignored bogus metadata alpha
        assert rec["contexts"][0]["generating_p"] == p
        assert {v["denoiser_evaluations_per_draw"] for v in rec["methods"].values()} == {0, 2, 25, 50, 100}
        assert "Neither finite sampler scores nor Monte Carlo self-scores are an exact Bayes floor" in rec["note"]
        records.append(rec)
    for method in records[0]["methods"]:
        assert records[0]["methods"][method]["scores"] == records[1]["methods"][method]["scores"]
    bad = tmp_path / "law.npz"
    _toy_file(bad, nrm, boundary_law="stationary")
    with pytest.raises(ValueError, match="disagrees"):
        oracle.main(["--toy-eval", str(bad), "--output", str(tmp_path / "bad.json"), "--conditioning", "pooled", "--boundary-law", "iid"])


def test_regime_known_entropy_and_explicit_stationary_cli(tmp_path, nrm):
    for law in ("iid", "stationary"):
        post = oracle.posterior_hidden(1, 0.25, 0, 1, nrm, boundary_law=law, entropies=True)
        np.testing.assert_allclose(post["gamma"], hmm_forward_backward(0, 1, 0.25))
        assert post["regime_entropy"] == 0
        np.testing.assert_allclose(post["path_entropy"], post["joint_regime_path_entropy"], atol=1e-12)
    source, dest = tmp_path / "toy.npz", tmp_path / "known.json"
    _toy_file(source, nrm, p=0.25, boundary_law="stationary")
    assert oracle.main(["--toy-eval", str(source), "--output", str(dest), "--conditioning", "regime-known",
                        "--boundary-law", "stationary", "--n-windows", "1", "--n-draws", "2"]) == 0
    rec = json.loads(dest.read_text())
    assert rec["boundary_law_source"] == "input field and caller"
    assert rec["conditioning"] == "regime-known" and "extra information" in rec["information"]


def test_diagnostics_small_cli_joint_entropy_and_independent_context(tmp_path, nrm, monkeypatch):
    source, dest = tmp_path / "toy.npz", tmp_path / "diagnostics.json"
    _toy_file(source, nrm)
    monkeypatch.setattr(diagnostics, "TIMESTEPS", (1, 100))
    assert diagnostics.main(["--toy-eval", str(source), "--output", str(dest), "--conditioning", "pooled",
                             "--boundary-law", "iid", "--n-paths", "4", "--n-context-per-regime", "1"]) == 0
    rec = json.loads(dest.read_text())
    assert rec["n_noisy"] == rec["n_context"] == 2
    assert "independent stream" in rec["context_conditioning"]
    assert "legitimate posterior concentration" in rec["noisy_conditioning"]
    assert "bin_entropy" not in rec["noisy"][0]
    assert rec["noisy"][0]["path_entropy"] < 1e-6
    assert rec["noisy"][0]["regime_entropy"] > 0  # exact path need not identify p
    assert rec["context"][0]["path_entropy"] > 5
    assert rec["context"][0]["path_entropy"] == rec["context"][1]["path_entropy"]
    assert [r["denoiser_evaluations_per_endpoint"] for r in rec["noisy"]] == [1, 2]
