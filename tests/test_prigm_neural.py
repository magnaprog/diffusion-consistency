"""Neural/loss/sampler CPU tests. No CUDA jobs."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prigm.arrays import physical_to_model
from prigm.config import HIDDEN_END, HIDDEN_START, N_BINS, N_CHANNELS, PARAM_CEILING, TOY_WIDTHS
from prigm.loss import ESTIMATOR_MEAN, ESTIMATOR_SAMPLE, LossConfig, prigm_loss, two_step_u0
from prigm.model import ConditionalUNet, count_parameters, hidden_mask
from prigm.normalize import fit_normalizer
from prigm.sampler import ddim_generate, ddim_timesteps, initial_noise
from prigm.schedule import build_schedule
from prigm.toy import pooled_toy_observations


def _batch(n_gen=N_CHANNELS, B=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    u0 = torch.randn(B, n_gen, N_BINS, generator=g)
    obs = u0.clone()
    obs[:, :, HIDDEN_START:HIDDEN_END] = 0.0
    mask = torch.ones(B, 1, N_BINS)
    mask[:, :, HIDDEN_START:HIDDEN_END] = 0.0
    rel = torch.linspace(-1, 1, N_BINS)[None, None, :].expand(B, 1, N_BINS)
    return u0, obs, mask, rel


def test_param_count_under_ceiling():
    m = ConditionalUNet(n_generated=6)
    n = count_parameters(m)
    assert n < PARAM_CEILING
    m5 = ConditionalUNet(n_generated=5)
    assert count_parameters(m5) != n


def test_forward_shape_and_gap_depends_on_flanks():
    m = ConditionalUNet(n_generated=6)
    m.eval()
    u0, obs, mask, rel = _batch()
    t = torch.ones(u0.shape[0], dtype=torch.long)
    noisy = u0.clone()
    y = m(noisy, obs, mask, rel, t)
    assert y.shape == u0.shape
    obs2 = obs.clone()
    obs2[:, :, 0] += 3.0
    y2 = m(noisy, obs2, mask, rel, t)
    # gap output should change if receptive field reaches left flank
    gap = slice(HIDDEN_START, HIDDEN_END)
    assert not torch.allclose(y[:, :, gap], y2[:, :, gap])


def test_zero_coefficient_matches_data_only_grad():
    torch.manual_seed(1)
    m = ConditionalUNet(n_generated=6, widths=TOY_WIDTHS)
    u0, obs, mask, rel = _batch(B=3, seed=1)
    t = torch.tensor([1, 10, 100])
    noise = torch.randn_like(u0)
    sch = build_schedule()
    a = torch.as_tensor(sch["a"], dtype=u0.dtype)[t][:, None, None]
    s = torch.as_tensor(sch["s"], dtype=u0.dtype)[t][:, None, None]
    u_t = u0.clone()
    hid = slice(HIDDEN_START, HIDDEN_END)
    u_t[:, :, hid] = a * u0[:, :, hid] + s * noise[:, :, hid]
    nrm = fit_normalizer(pooled_toy_observations())
    cfg0 = LossConfig(
        c_r=0.0,
        residual="obs",
        estimator=ESTIMATOR_SAMPLE,
        mean=nrm.mean,
        scale=nrm.scale,
        b_ref_sq=nrm.b_ref_sq,
    )
    m.zero_grad(set_to_none=True)
    out0 = prigm_loss(m, u0, u_t, t, obs, mask, rel, cfg0, sch)
    out0["batch"].backward()
    g0 = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
    cfg_none = LossConfig(
        c_r=0.0,
        residual=None,
        estimator=ESTIMATOR_SAMPLE,
        mean=nrm.mean,
        scale=nrm.scale,
        b_ref_sq=nrm.b_ref_sq,
    )
    m.zero_grad(set_to_none=True)
    out1 = prigm_loss(m, u0, u_t, t, obs, mask, rel, cfg_none, sch)
    out1["batch"].backward()
    assert torch.allclose(out0["batch"].detach(), out1["batch"].detach())
    for n, p in m.named_parameters():
        if n in g0:
            assert torch.allclose(g0[n], p.grad), n


def test_two_step_reuses_at_t1_and_has_grad():
    torch.manual_seed(2)
    m = ConditionalUNet(n_generated=6, widths=TOY_WIDTHS)
    u0, obs, mask, rel = _batch(B=2, seed=2)
    t = torch.ones(2, dtype=torch.long)
    sch = build_schedule()
    u_t = u0.clone()
    u0_hat, u0_star = two_step_u0(m, u_t, t, obs, mask, rel, sch)
    assert torch.equal(u0_hat, u0_star)
    loss = (u0_star ** 2).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_two_step_at_t100_second_call():
    torch.manual_seed(3)
    m = ConditionalUNet(n_generated=6, widths=TOY_WIDTHS)
    u0, obs, mask, rel = _batch(B=1, seed=3)
    t = torch.tensor([100])
    sch = build_schedule()
    u0_hat, u0_star = two_step_u0(m, u0, t, obs, mask, rel, sch)
    assert not torch.equal(u0_hat, u0_star)


def test_ddim_grid_and_generate_finite():
    torch.manual_seed(4)
    m = ConditionalUNet(n_generated=6, widths=TOY_WIDTHS)
    m.eval()
    u0, obs, mask, rel = _batch(B=1, seed=4)
    noise = torch.randn_like(u0)
    with torch.no_grad():
        out = ddim_generate(m, obs, mask, rel, noise, n_eval=10)
    assert torch.isfinite(out).all()
    assert out.shape == u0.shape
    grid = ddim_timesteps(25)
    assert grid[0] == 100 and grid[-1] == 1


@pytest.mark.parametrize("n_generated", [6, 5])
@pytest.mark.parametrize("n_eval", [2, 10])
def test_ddim_preserves_observed_flanks_at_every_call(n_generated, n_eval):
    _, obs, mask, rel = _batch(B=2, seed=14)
    noise = torch.randn(2, n_generated, N_BINS, generator=torch.Generator().manual_seed(16))
    noise_before = noise.clone()
    obs_before = obs.clone()
    hid = slice(HIDDEN_START, HIDDEN_END)
    ctx = list(range(HIDDEN_START)) + list(range(HIDDEN_END, N_BINS))
    calls = []

    def denoiser(noisy, observed, mask, relpos, t):
        calls.append((noisy.detach().clone(), observed.detach().clone(), t.clone()))
        flank_sum = (
            noisy[:, :, HIDDEN_START - 1:HIDDEN_START]
            + noisy[:, :, HIDDEN_END:HIDDEN_END + 1]
        )
        return flank_sum.expand_as(noisy)

    sch = build_schedule()
    out = ddim_generate(
        denoiser, obs, mask, rel, noise, n_eval=n_eval,
        sched=sch, n_generated=n_generated,
    )
    steps = ddim_timesteps(n_eval, sch["n_steps"])
    expected = (
        obs_before[:, :n_generated, HIDDEN_START - 1:HIDDEN_START]
        + obs_before[:, :n_generated, HIDDEN_END:HIDDEN_END + 1]
    ).expand_as(noise)
    a = torch.as_tensor(sch["a"], dtype=noise.dtype)
    s = torch.as_tensor(sch["s"], dtype=noise.dtype)
    eps = (noise_before[:, :, hid] - a[steps[0]] * expected[:, :, hid]) / s[steps[0]]
    assert len(calls) == n_eval
    assert torch.equal(calls[0][0][:, :, hid], noise_before[:, :, hid])
    for (state, conditioning, t), step in zip(calls, steps):
        assert state.shape == noise.shape
        assert torch.equal(state[:, :, ctx], obs_before[:, :n_generated, ctx])
        assert torch.equal(conditioning, obs_before)
        assert torch.equal(t, torch.full_like(t, int(step)))
        torch.testing.assert_close(
            state[:, :, hid], a[step] * expected[:, :, hid] + s[step] * eps,
            rtol=1e-5, atol=1e-5,
        )
    assert torch.equal(obs, obs_before)
    assert torch.equal(noise, noise_before)
    assert out.shape == noise.shape
    assert torch.equal(out[:, :, hid], expected[:, :, hid])
    assert torch.count_nonzero(out[:, :, ctx]) == 0


@pytest.mark.parametrize("length", [1, N_BINS])
@pytest.mark.parametrize("key", [0, 7012])
def test_initial_noise_g5_is_canonical_six_channel_prefix(length, key):
    expected = np.random.Generator(np.random.PCG64(key)).standard_normal((length, N_CHANNELS))
    noise6 = initial_noise(6, length, key)
    noise5 = initial_noise(5, length, key)
    np.testing.assert_array_equal(noise6, expected)
    np.testing.assert_array_equal(noise5, noise6[:, :5])
    assert noise6.dtype == noise5.dtype == np.float64


@pytest.mark.parametrize("n_generated, source_channels", [(5, 5), (5, 6), (6, 6)])
def test_initial_noise_explicit_pair(n_generated, source_channels):
    source = np.arange(N_BINS * source_channels, dtype=np.float32).reshape(N_BINS, source_channels)
    source_before = source.copy()
    noise = initial_noise(n_generated, N_BINS, 7012, pair_from=source)
    np.testing.assert_array_equal(noise, source_before[:, :n_generated])
    np.testing.assert_array_equal(source, source_before)
    assert noise.dtype == np.float64


@pytest.mark.parametrize("shape", [
    (), (N_BINS,), (1, N_BINS, N_CHANNELS),
    (N_CHANNELS, N_BINS), (N_BINS - 1, N_CHANNELS), (N_BINS, 4),
])
def test_initial_noise_rejects_incompatible_pair_shape(shape):
    with pytest.raises(ValueError, match="pair_from must have shape"):
        initial_noise(5, N_BINS, 7012, pair_from=np.zeros(shape))


@pytest.mark.parametrize("n_generated", [0, N_CHANNELS + 1])
def test_initial_noise_rejects_noncanonical_channel_count(n_generated):
    with pytest.raises(ValueError, match="n_generated must be between"):
        initial_noise(n_generated, N_BINS, 7012)


@pytest.mark.parametrize("n_generated", [6, 5])
def test_physical_hidden_targets_do_not_affect_fixed_noise_predictions(n_generated):
    torch.manual_seed(15)
    pooled = pooled_toy_observations()
    nrm = fit_normalizer(pooled)
    z = pooled[np.arange(2 * N_BINS).reshape(2, N_BINS) % len(pooled)]
    hid = slice(HIDDEN_START, HIDDEN_END)
    changed = z.copy()
    changed[:, hid] = np.roll(z[:, hid], 1, axis=1)
    u0, obs_array = physical_to_model(z, nrm, n_generated=n_generated)
    changed_u0, changed_obs_array = physical_to_model(changed, nrm, n_generated=n_generated)
    assert not np.array_equal(u0[:, :, hid], changed_u0[:, :, hid])
    np.testing.assert_array_equal(obs_array, changed_obs_array)
    assert np.count_nonzero(obs_array[:, :, hid]) == 0
    obs = torch.from_numpy(obs_array)
    changed_obs = torch.from_numpy(changed_obs_array)
    obs_before = obs.clone()
    _, _, mask, rel = _batch(B=2)
    noise = torch.as_tensor(
        np.stack([initial_noise(n_generated, N_BINS, key).T for key in (17, 18)]),
        dtype=obs.dtype,
    )
    noise_before = noise.clone()
    model = ConditionalUNet(n_generated=n_generated, widths=TOY_WIDTHS).eval()
    with torch.no_grad():
        out = ddim_generate(model, obs, mask, rel, noise, n_eval=3, n_generated=n_generated)
        changed_out = ddim_generate(
            model, changed_obs, mask, rel, noise, n_eval=3, n_generated=n_generated,
        )
    assert torch.isfinite(out).all()
    assert torch.equal(out, changed_out)
    assert torch.equal(obs, obs_before)
    assert torch.equal(changed_obs, obs_before)
    assert torch.equal(noise, noise_before)


def test_g5_five_channel_forward():
    m = ConditionalUNet(n_generated=5, widths=TOY_WIDTHS)
    noisy, _, mask, rel = _batch(n_gen=5, B=1)
    # observed conditioning stays six channels
    obs6 = torch.randn(1, 6, N_BINS)
    obs6[:, :, HIDDEN_START:HIDDEN_END] = 0.0
    t = torch.tensor([50])
    y = m(noisy, obs6, mask, rel, t)
    assert y.shape[1] == 5
    assert m.n_observed == 6


def test_two_step_preserves_context():
    torch.manual_seed(6)
    m = ConditionalUNet(n_generated=6, widths=TOY_WIDTHS)
    captured = {}

    orig = m.forward

    def wrapped(noisy, observed, mask, relpos, t):
        captured.setdefault("noisy", []).append(noisy.detach().clone())
        return orig(noisy, observed, mask, relpos, t)

    m.forward = wrapped
    u0, obs, mask, rel = _batch(B=2, seed=6)
    t = torch.tensor([50, 50])
    sch = build_schedule()
    u_t = u0.clone()
    two_step_u0(m, u_t, t, obs, mask, rel, sch)
    # second call is at t=1 on u1*; context bins must match u_t
    u1 = captured["noisy"][1]
    ctx = list(range(0, HIDDEN_START)) + list(range(HIDDEN_END, N_BINS))
    assert torch.allclose(u1[:, :, ctx], u_t[:, :, ctx])
    m.forward = orig


def test_positive_residual_changes_loss():
    torch.manual_seed(5)
    m = ConditionalUNet(n_generated=6, widths=TOY_WIDTHS)
    u0, obs, mask, rel = _batch(B=2, seed=5)
    t = torch.tensor([50, 50])
    noise = torch.randn_like(u0)
    sch = build_schedule()
    a = torch.as_tensor(sch["a"], dtype=u0.dtype)[t][:, None, None]
    s = torch.as_tensor(sch["s"], dtype=u0.dtype)[t][:, None, None]
    u_t = u0.clone()
    hid = slice(HIDDEN_START, HIDDEN_END)
    u_t[:, :, hid] = a * u0[:, :, hid] + s * noise[:, :, hid]
    nrm = fit_normalizer(pooled_toy_observations())
    kw = dict(mean=nrm.mean, scale=nrm.scale, b_ref_sq=nrm.b_ref_sq)
    out0 = prigm_loss(
        m, u0, u_t, t, obs, mask, rel,
        LossConfig(c_r=0.0, residual=None, estimator=ESTIMATOR_SAMPLE, **kw),
        sch,
    )
    out1 = prigm_loss(
        m, u0, u_t, t, obs, mask, rel,
        LossConfig(c_r=0.005, residual="obs", estimator=ESTIMATOR_SAMPLE, **kw),
        sch,
    )
    assert out1["phys"].item() >= 0.0
    assert not torch.allclose(out0["batch"].detach(), out1["batch"].detach())
