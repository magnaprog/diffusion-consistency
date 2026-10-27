"""Check reported loss gradients on CPU with small differentiable models.

Use the production loss to compare automatic differentiation with directional
finite differences and to recompute selected recorded gradient statistics.
The probe reports the weighted data and residual terms separately, including
their gradient norms and relative direction.

No fitted scientific checkpoint is used. These tests establish local
calculation and serialization checks; a gradient measured at fixed inputs
does not establish why a trained model performed as it did.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from prigm.artifacts import load_arrays
from prigm.arrays import physical_to_model
from prigm.checkpoint import array_hash, save_torch_payload
from prigm.config import HIDDEN_END, HIDDEN_START, N_BINS, N_CHANNELS, N_HIDDEN
from prigm.loss import LossConfig, prigm_loss
from prigm.model import ConditionalUNet, count_parameters
from prigm.recovery import ContractMismatch, PublicationError, publish_generation, sha256_file
from prigm.run_gradient_probe import (
    TIMESTEPS, corrupt_gap, fixed_window_indices, gradient_statistics,
    main, parse_args, probe_case, schedule_record,
)
from prigm.schedule import build_schedule


class DifferentiableToy(torch.nn.Module):
    """Nonlinear six-channel denoiser with a differentiable second call."""

    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.linspace(0.15, 0.55, 6, dtype=torch.float64))
        self.bias = torch.nn.Parameter(torch.linspace(-0.25, 0.35, 6, dtype=torch.float64))
        self.unused = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
        self.calls = []

    def forward(self, noisy, observed, mask, relpos, t):
        self.calls.append((noisy.detach().clone(), observed.detach().clone(), t.clone()))
        return torch.tanh(self.gain[None, :, None] * noisy + self.bias[None, :, None]
                          + 0.03 * observed + 0.07 * relpos + t[:, None, None] / 400.0)


@pytest.fixture
def toy_case():
    generator = torch.Generator(device="cpu").manual_seed(71)
    u0 = torch.randn(2, N_CHANNELS, N_BINS, generator=generator, dtype=torch.float64) * 0.4
    noise = torch.randn(u0.shape, generator=generator, dtype=u0.dtype)
    observed = u0.clone()
    observed[:, :, HIDDEN_START:HIDDEN_END] = 0.0
    mask = torch.ones(2, 1, N_BINS, dtype=u0.dtype)
    mask[:, :, HIDDEN_START:HIDDEN_END] = 0.0
    relpos = torch.linspace(-1, 1, N_BINS, dtype=u0.dtype)[None, None].expand(2, -1, -1)
    cfg = LossConfig(c_r=0.037, residual="obs", estimator="S", b_ref_sq=3.4,
                     mean=np.array([0.3, -0.5, 0.2, 1.8, 0.6, 0.9]),
                     scale=np.array([1.2, 0.8, 1.6, 0.5, 0.7, 1.3]))
    return DifferentiableToy(), u0, noise, observed, mask, relpos, cfg, build_schedule()


def flat_grad(value, model, retain_graph=False):
    params = tuple(model.parameters())
    grads = torch.autograd.grad(value, params, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                      for g, p in zip(grads, params)])


@pytest.mark.parametrize("estimator", ["M", "S"])
@pytest.mark.parametrize("residual", ["obs", "inst"])
@pytest.mark.parametrize("step", [1, 50, 100])
def test_weighted_gradients_add_and_probe_statistics_match_direct_loss(toy_case, estimator, residual, step):
    model, u0, noise, observed, mask, relpos, cfg, sched = toy_case
    cfg = replace(cfg, estimator=estimator, residual=residual)
    u_t = corrupt_gap(u0, noise, step, sched)
    t = torch.full((len(u0),), step, dtype=torch.long)
    out = prigm_loss(model, u0, u_t, t, observed, mask, relpos, cfg, sched)
    data = flat_grad(out["data"], model, retain_graph=True)
    phys = flat_grad(out["phys"], model, retain_graph=True)
    total = flat_grad(out["batch"], model)
    torch.testing.assert_close(total, data + phys, rtol=2e-13, atol=2e-13)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    report = probe_case(model, u0, u_t, t, observed, mask, relpos, cfg, sched)
    dn, pn = np.linalg.norm(data.numpy()), np.linalg.norm(phys.numpy())
    cosine = np.dot(data.numpy(), phys.numpy()) / (dn * pn)
    expected = {
        "weighted_data_mean": out["data"].item(),
        "weighted_physics_mean": out["phys"].item(), "total_mean": out["batch"].item(),
        "data_gradient_norm": dn, "physics_gradient_norm": pn,
        "physics_to_data_norm_ratio": pn / dn, "gradient_cosine": cosine,
        "gradient_angle_degrees": math.degrees(math.acos(cosine)),
    }
    assert report == pytest.approx(expected, rel=2e-14, abs=2e-14)
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("estimator", ["M", "S"])
@pytest.mark.parametrize("residual", ["obs", "inst"])
@pytest.mark.parametrize("component", ["data", "phys", "batch"])
def test_actual_loss_directional_finite_difference_matches_autograd(toy_case, estimator, residual, component):
    model, u0, noise, observed, mask, relpos, cfg, sched = toy_case
    cfg = replace(cfg, estimator=estimator, residual=residual)
    t = torch.full((len(u0),), 50, dtype=torch.long)
    u_t = corrupt_gap(u0, noise, 50, sched)
    args = (u0, u_t, t, observed, mask, relpos, cfg, sched)
    loss = prigm_loss(model, *args)[component]
    grad = flat_grad(loss, model)
    # Gradient direction gives a nonzero directional derivative; do not detach
    # either denoiser call in the sample estimator.
    direction = grad / torch.linalg.vector_norm(grad)
    offset = 0
    directions = {}
    for name, param in model.named_parameters():
        directions[name] = direction[offset:offset + param.numel()].reshape_as(param)
        offset += param.numel()
    values = []
    h = 1e-5
    for sign in (-1, 1):
        state = {name: param.detach() + sign * h * directions[name]
                 for name, param in model.named_parameters()}

        def displaced(*inputs):
            return torch.func.functional_call(model, state, inputs)

        values.append(prigm_loss(displaced, *args)[component].item())
    finite_difference = (values[1] - values[0]) / (2 * h)
    direct = torch.dot(grad, direction).item()
    assert finite_difference == pytest.approx(direct, rel=3e-8, abs=1e-10)
    assert finite_difference > 0


@pytest.mark.parametrize("residual", ["obs", "inst"])
def test_nonzero_coefficient_uses_hidden_data_and_physical_normalizations(toy_case, residual):
    model, u0, noise, observed, mask, relpos, cfg, sched = toy_case
    cfg = replace(cfg, estimator="M", residual=residual)
    t = torch.tensor([25, 25])
    u_t = corrupt_gap(u0, noise, 25, sched)
    out = prigm_loss(model, u0, u_t, t, observed, mask, relpos, cfg, sched)
    hidden = slice(HIDDEN_START, HIDDEN_END)
    expected_data = sched["w"][25] * (
        (out["u0_hat"][:, :, hidden] - u0[:, :, hidden]).square().sum((1, 2))
        / (N_CHANNELS * N_HIDDEN)
    ).mean()
    z = (out["u0_hat"] * torch.from_numpy(cfg.scale)[None, :, None]
         + torch.from_numpy(cfg.mean)[None, :, None])[:, :, hidden]
    residual_value = z[:, 3].square() - z[:, :3].square().sum(1)
    if residual == "obs":
        residual_value = residual_value + z[:, 4] - z[:, 5]
    expected_physics = (cfg.c_r / (2 * sched["sigma"][25])) * (residual_value / cfg.b_ref_sq).square().mean()
    torch.testing.assert_close(out["data"], expected_data, rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(out["phys"], expected_physics, rtol=2e-14, atol=2e-14)
    original = probe_case(model, u0, u_t, t, observed, mask, relpos, cfg, sched)
    doubled = probe_case(model, u0, u_t, t, observed, mask, relpos, replace(cfg, c_r=2 * cfg.c_r), sched)
    rescaled = probe_case(model, u0, u_t, t, observed, mask, relpos, replace(cfg, b_ref_sq=2 * cfg.b_ref_sq), sched)
    for key in ("weighted_physics_mean", "physics_gradient_norm", "physics_to_data_norm_ratio"):
        assert doubled[key] == pytest.approx(2 * original[key], rel=2e-14)
        assert rescaled[key] == pytest.approx(original[key] / 4, rel=2e-14)
    assert doubled["weighted_data_mean"] == rescaled["weighted_data_mean"] == original["weighted_data_mean"]
    assert doubled["data_gradient_norm"] == rescaled["data_gradient_norm"] == original["data_gradient_norm"]


@pytest.mark.parametrize("step,expected_calls", [(1, 1), (25, 2), (50, 2), (100, 2)])
def test_sample_reuses_t1_call_and_keeps_all_flanks_clean(toy_case, step, expected_calls):
    model, u0, noise, observed, mask, relpos, cfg, sched = toy_case
    before_u0, before_noise = u0.clone(), noise.clone()
    u_t = corrupt_gap(u0, noise, step, sched)
    hidden = slice(HIDDEN_START, HIDDEN_END)
    torch.testing.assert_close(u_t[:, :, hidden],
                               sched["a"][step] * u0[:, :, hidden] + sched["s"][step] * noise[:, :, hidden],
                               rtol=0, atol=0)
    t = torch.full((len(u0),), step, dtype=torch.long)
    report = probe_case(model, u0, u_t, t, observed, mask, relpos, cfg, sched)
    assert len(model.calls) == expected_calls
    context = list(range(HIDDEN_START)) + list(range(HIDDEN_END, N_BINS))
    for state, obs, _ in model.calls:
        assert torch.equal(state[:, :, context], u0[:, :, context])
        assert torch.equal(obs, observed)
    assert torch.equal(model.calls[0][0], u_t)
    assert torch.equal(u0, before_u0) and torch.equal(noise, before_noise)
    if step == 1:
        mean = probe_case(model, u0, u_t, t, observed, mask, relpos, replace(cfg, estimator="M"), sched)
        assert report == mean
    else:
        assert torch.equal(model.calls[1][2], torch.ones_like(t))


def test_zero_coefficient_reports_zero_not_epsilon_angles(toy_case):
    model, u0, noise, observed, mask, relpos, cfg, sched = toy_case
    cfg = replace(cfg, c_r=0.0)
    report = probe_case(model, u0, corrupt_gap(u0, noise, 50, sched), torch.tensor([50, 50]),
                        observed, mask, relpos, cfg, sched)
    assert report["weighted_physics_mean"] == report["physics_gradient_norm"] == 0.0
    assert report["physics_to_data_norm_ratio"] == 0.0
    assert report["gradient_cosine"] is report["gradient_angle_degrees"] is None
    assert len(model.calls) == 1


@pytest.mark.parametrize("data,phys,ratio,cosine,angle", [
    ([3., 0.], [0., 4.], 4 / 3, 0., 90.),
    ([3., 0.], [-4., 0.], 4 / 3, -1., 180.),
    ([0., 0.], [0., 4.], None, None, None),
    ([3., 0.], [0., 0.], 0., None, None),
    ([0., 0.], [0., 0.], None, None, None),
    ([1e-24, 0.], [1e-24, 0.], 1., 1., 0.),
])
def test_exact_gradient_geometry_and_undefined_zero_norms(data, phys, ratio, cosine, angle):
    result = gradient_statistics(torch.tensor(data, dtype=torch.float64), torch.tensor(phys, dtype=torch.float64))
    assert result == {
        "data_gradient_norm": np.linalg.norm(data), "physics_gradient_norm": np.linalg.norm(phys),
        "physics_to_data_norm_ratio": ratio, "gradient_cosine": cosine, "gradient_angle_degrees": angle,
    }
    json.dumps(result, allow_nan=False)


def test_nonfinite_gradients_fail_closed():
    with pytest.raises(PublicationError, match="nonfinite"):
        gradient_statistics(torch.tensor([float("nan")]), torch.ones(1))


def test_fixed_indices_cover_full_ordered_split_without_score_selection():
    np.testing.assert_array_equal(fixed_window_indices(100, 8), [0, 14, 28, 42, 56, 70, 84, 99])
    np.testing.assert_array_equal(fixed_window_indices(3, 8), [0, 1, 2])
    np.testing.assert_array_equal(fixed_window_indices(100, 1), [0])
    with pytest.raises(ValueError, match="positive"):
        fixed_window_indices(100, 0)


@pytest.fixture
def pinned_fixture(tmp_path):
    """New protocol-v2 generation owned by this test, not legacy migration."""
    arrays = tmp_path / "arrays.npz"
    rng = np.random.default_rng(71)
    mean = np.array([0.3, -0.5, 0.2, 1.8, 0.6, 0.9])
    scale = np.array([1.2, 0.8, 1.6, 0.5, 0.7, 1.3])
    np.savez(arrays, z=rng.normal(size=(7, N_BINS, 6)) * scale + mean,
             ids=np.array([f"fixture-val-{i}" for i in range(7)]), split=np.array("VAL"),
             mean=mean, scale=scale, b_ref_sq=np.array(3.4), n_bins=np.array(300))
    identity = load_arrays(arrays, "VAL")[3]
    contract = {
        "contract_version": 2, "arm": "D", "seed": 71, "n_generated": 6, "n_aux": 0,
        "widths": [8, 8, 8], "c_r": 0.0, "residual": None, "estimator": "S",
        "source_hashes": {"training_before_probe.py": "a" * 64},
        "data_hashes": {"fixture": "owned_synthetic_training_identity"},
        "normalization_hash": identity["normalization_sha256"],
        "settings": {"diffusion": schedule_record(build_schedule())},
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        model = ConditionalUNet(n_generated=6, widths=(8, 8, 8)).cpu().eval()
    normalization = {"mean": torch.from_numpy(mean), "scale": torch.from_numpy(scale),
                     "b_ref_sq": 3.4, "n_bins": 300}
    payload = {"model": model.state_dict(), "contract": contract, "updates": 0,
               "normalization": normalization}
    run = tmp_path / "run"
    publish_generation(run, 0, {"state.pt": save_torch_payload(payload)}, {"contract": contract})
    # A later committed generation must never replace the explicitly pinned one.
    publish_generation(run, 4, {"state.pt": save_torch_payload({**payload, "updates": 4})}, {"contract": contract})
    output = tmp_path / "probe.json"
    argv = ["--run-root", str(run), "--generation", "0", "--checkpoint-sha256",
            sha256_file(run / "gen_00000000/state.pt"), "--arrays", str(arrays), "--split", "VAL",
            "--c-r", "0.037", "--n-windows", "3", "--output", str(output)]
    return argv, output, arrays, run, model, payload


def test_full_cli_fixed_pairing_provenance_and_exact_output(pinned_fixture, monkeypatch):
    argv, output, arrays, run, model, payload = pinned_fixture
    original_files = {str(p): sha256_file(p) for p in run.rglob("*") if p.is_file()}
    seen = []
    forward = ConditionalUNet.forward

    def tracked(self, noisy, observed, mask, relpos, t):
        seen.append((noisy.detach().clone(), observed.detach().clone(), t.clone()))
        return forward(self, noisy, observed, mask, relpos, t)

    monkeypatch.setattr(ConditionalUNet, "forward", tracked)
    monkeypatch.setenv("PRIGM_REQUIRE_CUDA", "1")  # probe must remain CPU-only
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert main(argv) == 0
        result = json.loads(output.read_text())
        assert len(result["records"]) == 16
        assert result["selection"]["indices"] == [0, 3, 6]
        assert result["selection"]["ids"] == ["fixture-val-0", "fixture-val-3", "fixture-val-6"]
        assert result["arrays"] == load_arrays(arrays, "VAL")[3]
        assert result["generation_updates"] == 0
        assert result["training_source_hashes"] == payload["contract"]["source_hashes"]
        assert "prigm/run_gradient_probe.py" in result["current_source_hashes"]
        assert result["training_source_hashes"] != result["current_source_hashes"]
        assert result["model"]["n_parameters"] == count_parameters(model)
        assert result["environment"]["device"] == "cpu"
        assert result["normalization"]["b_ref_sq"] == 3.4
        assert result["normalization"]["sha256"] == result["arrays"]["normalization_sha256"]
        assert result["c_r"] == 0.037 and result["training_contract"]["c_r"] == 0.0
        assert result["noise"]["seed"] == 7003
        assert result["schedule"]["probe_timesteps"] == [1, 25, 50, 100]
        assert "not causal evidence" in result["scope"]
        z, _, nrm, _ = load_arrays(arrays, "VAL")
        u0_array, obs_array = physical_to_model(z[[0, 3, 6]], nrm)
        u0, obs = torch.from_numpy(u0_array), torch.from_numpy(obs_array)
        noise = torch.randn(u0.shape, generator=torch.Generator().manual_seed(7003), dtype=u0.dtype)
        assert result["noise"]["sha256"] == array_hash(noise.numpy())
        assert result["inputs"]["u0_sha256"] == array_hash(u0_array)
        sched = build_schedule()
        assert result["schedule"]["array_hashes"] == schedule_record(sched)["array_hashes"]
        calls = iter(seen.copy())
        for step in TIMESTEPS:
            expected = corrupt_gap(u0, noise, step, sched)
            for estimator in ("M", "S"):
                for _ in ("obs", "inst"):
                    state, conditioning, t = next(calls)
                    assert torch.equal(state, expected)
                    assert torch.equal(conditioning, obs)
                    assert torch.equal(t, torch.full((3,), step, dtype=torch.long))
                    if estimator == "S" and step != 1:
                        state, conditioning, t = next(calls)
                        assert torch.equal(state[:, :, :HIDDEN_START], u0[:, :, :HIDDEN_START])
                        assert torch.equal(state[:, :, HIDDEN_END:], u0[:, :, HIDDEN_END:])
                        assert torch.equal(conditioning, obs)
                        assert torch.equal(t, torch.ones(3, dtype=torch.long))
        assert next(calls, None) is None
        # Independently recompute one serialized row using the real model/loss.
        mask = torch.ones(3, 1, N_BINS)
        mask[:, :, HIDDEN_START:HIDDEN_END] = 0
        center = 0.5 * (N_BINS - 1)
        rel = torch.from_numpy(((np.arange(N_BINS, dtype=np.float32) - center) / center)[None, None]).expand(3, -1, -1)
        cfg = LossConfig(c_r=0.037, residual="inst", estimator="S", mean=nrm.mean,
                         scale=nrm.scale, b_ref_sq=nrm.b_ref_sq)
        out = prigm_loss(model, u0, corrupt_gap(u0, noise, 100, sched), torch.full((3,), 100),
                         obs, mask, rel, cfg, sched)
        gd = flat_grad(out["data"], model, retain_graph=True).double().numpy()
        gp = flat_grad(out["phys"], model).double().numpy()
        dn, pn = np.linalg.norm(gd), np.linalg.norm(gp)
        cosine = np.dot(gd / dn, gp / pn)
        expected_row = {"t": 100, "estimator": "S", "residual": "inst",
                        "weighted_data_mean": out["data"].item(), "weighted_physics_mean": out["phys"].item(),
                        "total_mean": out["batch"].item(), "data_gradient_norm": dn,
                        "physics_gradient_norm": pn, "physics_to_data_norm_ratio": pn / dn,
                        "gradient_cosine": cosine, "gradient_angle_degrees": math.degrees(math.acos(cosine))}
        for key, value in expected_row.items():
            if isinstance(value, (float, np.floating)):
                assert result["records"][-1][key] == pytest.approx(value, rel=2e-14, abs=2e-14)
            else:
                assert result["records"][-1][key] == value
        repeat = output.with_name("repeat.json")
        assert main([*argv[:-1], str(repeat)]) == 0
        assert repeat.read_bytes() == output.read_bytes()
    finally:
        torch.set_num_threads(threads)
    assert original_files == {str(p): sha256_file(p) for p in run.rglob("*") if p.is_file()}


@pytest.mark.parametrize("bad,message", [
    ("hash", "checkpoint hash"), ("normalization", "array normalization"),
    ("legacy", "contract version 2"), ("g5", "G5"),
    ("payload", "committed identity"), ("updates", "committed identity"),
    ("schedule", "diffusion schedule"), ("weights", "model state"),
    ("payload_normalization", "normalization payload"),
])
def test_cli_rejects_mismatched_checkpoint_without_output(pinned_fixture, bad, message):
    argv, output, arrays, run, model, payload = pinned_fixture
    contract = {**payload["contract"]}
    changed = {**payload, "contract": contract, "updates": 9}
    if bad == "normalization":
        contract["normalization_hash"] = "0" * 64
    elif bad == "legacy":
        contract["contract_version"] = 1
    elif bad == "g5":
        contract.update(n_generated=5, arm="G5")
    elif bad == "payload":
        changed["contract"] = {**contract, "seed": 123}
    elif bad == "updates":
        changed["updates"] = 8
    elif bad == "schedule":
        contract["settings"] = {"diffusion": {**schedule_record(build_schedule()), "snr_clip": 999}}
    elif bad == "weights":
        changed["model"] = {**payload["model"]}
        changed["model"].pop(next(iter(changed["model"])))
    elif bad == "payload_normalization":
        changed["normalization"] = {**payload["normalization"], "b_ref_sq": 1.0}
    publish_generation(run, 9, {"state.pt": save_torch_payload(changed)}, {"contract": contract})
    args = argv.copy()
    args[args.index("--generation") + 1] = "9"
    args[args.index("--checkpoint-sha256") + 1] = "0" * 64 if bad == "hash" else sha256_file(run / "gen_00000009/state.pt")
    with pytest.raises(ContractMismatch, match=message):
        main(args)
    assert not output.exists()


@pytest.mark.parametrize("missing", ["ids", "split"])
def test_cli_requires_embedded_ids_and_split(pinned_fixture, missing):
    argv, output, arrays, *_ = pinned_fixture
    with np.load(arrays) as blob:
        values = {k: blob[k] for k in blob.files if k != missing}
    np.savez(arrays, **values)
    with pytest.raises(ContractMismatch, match="IDs"):
        main(argv)
    assert not output.exists()


def test_cli_rejects_test_arrays_disguised_as_val(pinned_fixture):
    argv, output, arrays, *_ = pinned_fixture
    with np.load(arrays) as blob:
        values = dict(blob)
    values["split"] = np.array("TEST")
    np.savez(arrays, **values)
    with pytest.raises(ContractMismatch, match="split"):
        main(argv)
    assert not output.exists()


def test_corrupt_pinned_generation_never_falls_back(pinned_fixture):
    argv, output, _, run, *_ = pinned_fixture
    (run / "gen_00000000/state.pt").write_bytes(b"invalid checkpoint")
    with pytest.raises(PublicationError, match="hash mismatch"):
        main(argv)
    assert not output.exists()


@pytest.mark.parametrize("flag,value", [("--split", "TEST"), ("--c-r", "nan"), ("--c-r", "-1"),
                                        ("--n-windows", "0"), ("--generation", "-1")])
def test_cli_rejects_invalid_arguments_before_loading(pinned_fixture, flag, value):
    argv, output, *_ = pinned_fixture
    args = argv.copy()
    args[args.index(flag) + 1] = value
    with pytest.raises(SystemExit):
        main(args)
    assert not output.exists()


def test_cli_requires_coefficient_and_defaults_to_eight_windows(pinned_fixture):
    argv, *_ = pinned_fixture
    args = argv.copy()
    idx = args.index("--n-windows")
    del args[idx:idx + 2]
    parsed = parse_args(args)
    assert parsed.n_windows == 8 and parsed.seed == 7003
    idx = args.index("--c-r")
    del args[idx:idx + 2]
    with pytest.raises(SystemExit):
        parse_args(args)


def test_existing_output_refused_before_reading_inputs(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("preserve")
    args = ["--run-root", "missing", "--generation", "0", "--checkpoint-sha256", "0" * 64,
            "--arrays", "missing", "--split", "TRAIN", "--c-r", "0", "--output", str(output)]
    with pytest.raises(PublicationError, match="exists"):
        main(args)
    assert output.read_text() == "preserve"
