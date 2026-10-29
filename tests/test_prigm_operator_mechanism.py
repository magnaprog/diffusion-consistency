"""Check the clipping map and its score changes with synthetic predictions.

Independently calculate the energy score as mean distance to the observation
minus half the mean distance between distinct draws. Apply the clipping map
to the same draws and check feasibility, displacement and the corresponding
score-component differences. Feasible observations do not guarantee that
clipping improves the energy score; explicit cases exercise both signs.

Local study fixtures use real array readers, hashes and scoring routines,
but their selected-checkpoint identities are synthetic and do not contain
historical fitted weights. These tests make no minimum-displacement or
causal training claim.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from analysis import prigm_operator_mechanism as mechanism
from prigm.artifacts import (
    NOISE_CONVENTION, STATE_CONVENTION, id_hash, json_bytes, load_arrays,
    publish_ensemble,
)
from prigm.normalize import AffineNormalizer
from prigm.observation import closure_correct, derive_vv, support_gaps
from prigm.recovery import ContractMismatch, PublicationError, sha256_bytes, sha256_file
from prigm.run_score import decode_ensemble, hidden_primary_std
from prigm.run_secondaries import summarize_diagnostics, window_diagnostics
from prigm.scoring import energy_score, month_balanced_mean


@pytest.fixture
def normalizer():
    return AffineNormalizer(np.array([1., -2., .5, 2., .2, 1.]),
                            np.array([2., 3., 4., .7, 1.2, 2.]), 9., 96)


@pytest.mark.parametrize("k", [20, 39])
def test_energy_terms_match_independent_offdiagonal_sum(k):
    rng = np.random.default_rng(k)
    x, y = rng.normal(size=(k, 60)), rng.normal(size=60)
    rec = mechanism.energy_terms(x, y)
    target = sum(np.sqrt(sum((v - y) ** 2)) for v in x) / k
    dispersion = sum(np.sqrt(sum((x[i] - x[j]) ** 2))
                     for i in range(k) for j in range(k) if i != j) / (2 * k * (k - 1))
    assert rec["target_term"] == pytest.approx(target, rel=1e-14)
    assert rec["dispersion_term"] == pytest.approx(dispersion, rel=1e-14)
    assert rec["energy_score"] == energy_score(x, y)
    assert rec["dispersion_term"] != pytest.approx(dispersion * (k - 1) / k)


@pytest.mark.parametrize("k", [20, 39])
@pytest.mark.parametrize("stored_dtype", [np.float32, np.float64])
def test_same_draw_completion_feasibility_components_and_bound(k, stored_dtype, normalizer):
    rng = np.random.default_rng(k)
    z = rng.normal(size=(k, 12, 6))
    z[..., 0] += 2
    z[..., 4] *= 30
    z[0, 0, 1] = -0.0
    before = z.tobytes()
    target = closure_correct(rng.normal(size=(12, 6)))
    rec = mechanism.window_mechanism(z, target, normalizer, stored_dtype)
    assert z.tobytes() == before
    completed = derive_vv(z[..., :5])
    assert completed.shape == z.shape
    assert completed[..., :5].tobytes() == z[..., :5].tobytes()
    for key in ("energy_score", "target_term", "dispersion_term"):
        assert rec[f"completed_{key}"] == rec[f"raw_{key}"]
    corrected = closure_correct(z)
    np.testing.assert_array_equal(closure_correct(corrected), corrected)
    for key in mechanism.MARGINS:
        assert np.min(support_gaps(corrected)[key]) >= -1e-12
    np.testing.assert_allclose(support_gaps(corrected)["r_obs"], 0, atol=1e-12)
    xs = hidden_primary_std(z, normalizer)
    cs = hidden_primary_std(corrected, normalizer)
    movement = np.sqrt(np.sum((cs - xs) ** 2, axis=1))
    assert rec["mean_movement"] == pytest.approx(movement.mean())
    np.testing.assert_allclose(rec["movement_quantiles"], np.quantile(movement, mechanism.QUANTILES))
    assert abs(rec["delta_energy_score"]) <= rec["displacement_bound"] + rec["bound_arithmetic_tolerance"]
    assert rec["delta_energy_score"] == pytest.approx(rec["delta_target_term"] - rec["delta_dispersion_term"], abs=1e-12)
    assert rec["raw_support_violation_fraction"] > 0
    assert rec["corrected_support_beyond_rounding_fraction"] == 0
    assert rec["raw_dispersion_term"] > 0


@pytest.mark.parametrize("k", [20, 39])
@pytest.mark.parametrize("sign", [-1, 1])
def test_correction_can_improve_or_worsen_es_even_for_feasible_truth(k, sign):
    nrm = AffineNormalizer(np.zeros(6), np.ones(6), 1., 48)
    z = np.zeros((k, 12, 6))
    if sign < 0:
        z[..., 3] = -2  # correcting negative m moves to the true all-zero summary
    else:
        z[..., 0] = 2  # retained wrong mu forces m away from the same feasible truth
    rec = mechanism.window_mechanism(z, np.zeros((12, 6)), nrm)
    assert sign * rec["delta_energy_score"] > 0
    assert rec["raw_dispersion_term"] == rec["corrected_dispersion_term"] == 0


@pytest.mark.parametrize("k", [20, 39])
def test_feasible_draws_have_zero_primary_movement_and_finite_zero_severity(k, normalizer):
    rng = np.random.default_rng(10)
    z = closure_correct(rng.normal(size=(k, 12, 6)))
    rec = mechanism.window_mechanism(z, z[0], normalizer)
    assert rec["mean_movement"] == rec["delta_energy_score"] == 0
    assert rec["displacement_bound"] == rec["bound_slack"] == 0
    for key in mechanism.MARGINS:
        assert rec[f"raw_{key}_severity_quantiles"] == [0.] * len(mechanism.QUANTILES)
    json_bytes(rec)


def test_signed_margins_and_severity_use_train_reference_units(normalizer):
    z = np.zeros((20, 12, 6))
    z[..., 0], z[..., 3], z[..., 4] = 2, 1, -9
    stats = mechanism.support_statistics(z, normalizer, np.float64)
    assert stats["m_minus_r_signed_quantiles"] == pytest.approx([-1 / 3] * 5)
    assert stats["vs_signed_quantiles"] == [-1.] * 5
    assert stats["vs_upper_minus_vs_signed_quantiles"] == pytest.approx([20.5 / 9] * 5)
    assert stats["m_minus_r_severity_quantiles"] == pytest.approx([1 / 3] * 5)
    assert stats["vs_upper_minus_vs_severity_quantiles"] == [0.] * 5
    assert stats["m_minus_r_beyond_rounding_fraction"] == 1


@pytest.mark.parametrize("stored_dtype", [np.float32, np.float64])
def test_support_roundoff_branches_match_existing_secondaries(stored_dtype):
    nrm = AffineNormalizer(np.zeros(6), np.ones(6), 1., 48)
    z = np.zeros((39, 12, 6))
    z[..., 0], z[..., 3] = 1., 1. - 2e-8
    result = mechanism.support_statistics(z, nrm, stored_dtype)
    old = window_diagnostics(z, np.ones((48, 6)), nrm, stored_dtype)
    assert result["support_violation_fraction"] == old["frac_support_inequality_violation"] == 1
    assert result["support_beyond_rounding_fraction"] == old["frac_support_violation_beyond_rounding"]
    assert result["support_beyond_rounding_fraction"] == (0 if stored_dtype == np.float32 else 1)


def test_bound_rejects_false_claim_and_accepts_rounding_slack():
    raw = {"target_term": 2., "dispersion_term": 1., "energy_score": 1.}
    corrected = {"target_term": 5., "dispersion_term": 1., "energy_score": 4.}
    with pytest.raises(ContractMismatch, match="bound"):
        mechanism.check_displacement_bound(raw, corrected, 1.)
    corrected = {**raw, "energy_score": 1. + 1e-15}
    assert mechanism.check_displacement_bound(raw, corrected, 0.)["bound_slack"] < 0
    with pytest.raises(ContractMismatch, match="negative"):
        mechanism.check_displacement_bound(raw, raw, -1.)


def test_unequal_month_weighting_and_equal_seed_mean():
    months = [200508, 200508, 200509, 200608]
    rows = [{"value": v, "quantiles": [v, 2 * v]} for v in (1., 3., 10., 20.)]
    result = mechanism.summarize_windows(rows, months)
    assert result["value"]["month_balanced"] == pytest.approx(32 / 3)
    assert result["value"]["mean"] == 8.5
    np.testing.assert_allclose(result["quantiles"]["month_balanced"], [32 / 3, 64 / 3])
    second = mechanism.summarize_windows([{k: np.asarray(v) * 2 for k, v in r.items()} for r in rows], months)
    means = mechanism._seed_average({"7": {"metrics": result}, "13": {"metrics": second}})
    assert means["value"]["month_balanced"] == pytest.approx(16.)
    assert means["value"]["mean"] == 12.75


def test_selection_decomposition_separates_signed_effects():
    raw = np.array([1., 3., 10., 20.])
    same = raw + [-.5, -.5, 2., 3.]
    chosen = same + [3., -2., -4., 1.]
    rec = mechanism.selection_decomposition(raw, same, chosen, [200508, 200508, 200509, 200608])
    for row, r, c, s in zip(rec["per_window"], raw, same, chosen):
        assert row["fixed_checkpoint_correction"] == c - r
        assert row["selection_effect"] == s - c
        assert row["total_change"] == s - r
    metrics = rec["metrics"]
    for weighting in ("mean", "month_balanced"):
        assert metrics["total_change"][weighting] == pytest.approx(
            metrics["fixed_checkpoint_correction"][weighting] + metrics["selection_effect"][weighting])
    same_checkpoint = mechanism.selection_decomposition(raw, same, same, [200508] * 4)
    assert same_checkpoint["metrics"]["selection_effect"]["month_balanced"] == 0


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_fail_instead_of_nan_summaries(value, normalizer):
    z = np.zeros((20, 12, 6))
    z[0, 0, 5] = value
    with pytest.raises(ContractMismatch, match="finite"):
        mechanism.window_mechanism(z, np.zeros((12, 6)), normalizer)
    with pytest.raises(ContractMismatch, match="finite"):
        mechanism.summarize_windows([{"x": value}], [200508])


def test_completion_regression_cannot_silently_add_seventh_coordinate(monkeypatch, normalizer):
    def wrong(z5):
        out = derive_vv(z5)
        out[..., 0] += 1
        return out

    monkeypatch.setattr(mechanism, "derive_vv", wrong)
    with pytest.raises(ContractMismatch, match="primary coordinate bits"):
        mechanism.window_mechanism(np.ones((20, 12, 6)), np.ones((12, 6)), normalizer)


@pytest.fixture
def study(normalizer, tmp_path):
    """Real loader, array/commit hashes and primary/secondary producers, tiny N.

    Checkpoint digests are synthetic selected identities, not model weight files.
    Files live in the temporary directory pytest assigns to this test.
    """
    root = tmp_path
    arrays = root / "artifacts/inventory"
    arrays.mkdir(parents=True)
    ids = ["2005-08-01T00:00-s52", "2005-08-02T00:00-s52", "2005-09-01T00:00-s71"]
    months = np.array([200508, 200508, 200509])
    rng = np.random.default_rng(902)
    truth = closure_correct(rng.normal(size=(len(ids), 48, 6)))
    identity = {}
    for split in ("TRAIN", "VAL", "TEST"):
        path = arrays / f"{split.lower()}.npz"
        np.savez(path, z=truth, ids=np.asarray(ids), split=split, mean=normalizer.mean,
                 scale=normalizer.scale, b_ref_sq=normalizer.b_ref_sq, n_bins=normalizer.n_bins)
        identity[split] = load_arrays(path, split)[3]
    protocol = {
        "schema_version": 2, "exposure": "historical_test_exposed", "array_root": str(arrays),
        "arrays": identity, "arms": list(mechanism.bootstrap.ARMS), "seeds": list(mechanism.FINAL_SEEDS),
        "milestones": [9560, 29636], "samplers": {"TEST": {"n_eval": 50, "n_draws": 39, "eval_seed": 7}},
        "controls": {"interpolation": {"conditioning": "two_boundaries"},
                     "persistence": {"conditioning": "left_boundary"},
                     "gaussian": {"rho": 0., "jitter": 1e-6, "seed": 13}},
        "inference_source_hashes": {f"prigm/{p.name}": sha256_file(p)
                                    for p in Path(mechanism.bootstrap.__file__).parent.glob("*.py")},
        "private_execution_path": "/home/private/do-not-export",
    }
    (root / "PROTOCOL.json").write_bytes(json_bytes(protocol))
    digest = sha256_file(root / "PROTOCOL.json")
    selection = {"schema_version": 2, "protocol_sha256": digest, "exposure": "historical_test_exposed",
                 "selected": {}, "test_jobs": [], "test_checkpoint_hashes": []}

    def publish(parent, ensemble_root, meta):
        channels = meta["n_generated"]
        dtype = np.float32 if meta.get("procedure", "diffusion") == "diffusion" else np.float64
        draws = rng.normal(size=(len(ids), 39, channels, 48)).astype(dtype)
        publish_ensemble(ensemble_root, draws, ids, meta)
        commit = sha256_file(ensemble_root / "gen_00000000/COMMIT.json")
        for kind in ("raw", "corrected"):
            records, scores = [], []
            for samples, y in zip(draws, truth):
                z = decode_ensemble(samples, normalizer)
                if kind == "corrected":
                    z = closure_correct(z)
                scores.append(energy_score(hidden_primary_std(z, normalizer),
                                           hidden_primary_std(y[None, 18:30], normalizer)[0]))
                records.append(window_diagnostics(z, y, normalizer, dtype))
            primary = {
                "schema_version": 2, "protocol_sha256": digest, "ensemble_commit_sha256": commit,
                "arrays": identity["TEST"], "ids": ids, "months": months.tolist(), "n": len(ids),
                "procedure": meta.get("procedure", "diffusion"),
                "checkpoint_sha256": meta.get("checkpoint_sha256"),
                "control_fit_sha256": meta.get("control_fit_sha256"), "corrected": kind == "corrected",
                "scores": scores, "mean": float(np.mean(scores)),
                "month_balanced": month_balanced_mean(scores, months),
            }
            (parent / f"score-{kind}.json").write_bytes(json_bytes(primary))
            secondary = {k: primary[k] for k in mechanism.followup.SECONDARY_IDENTITY}
            secondary.update(correct=kind == "corrected", interval_protocol=mechanism.followup.INTERVAL_PROTOCOL,
                             per_window=records, **summarize_diagnostics(records, months))
            (parent / f"secondaries-{kind}.json").write_bytes(json_bytes(secondary))

    for arm in mechanism.bootstrap.ARMS:
        for seed in mechanism.FINAL_SEEDS:
            name = f"{arm}_s{seed}"
            raw_gen, corrected_gen = 9560, 29636 if (arm, seed) == ("D", 7) else 9560
            selection["selected"][name] = {
                kind: {"generation": gen, "checkpoint_sha256": sha256_bytes(f"{name}:{gen}".encode()), "score": 1.}
                for kind, gen in (("raw", raw_gen), ("corrected", corrected_gen))}
            for gen in sorted({raw_gen, corrected_gen}):
                checkpoint = sha256_bytes(f"{name}:{gen}".encode())
                selection["test_jobs"].append([name, gen])
                selection["test_checkpoint_hashes"].append(checkpoint)
                parent = root / f"artifacts/test/{name}-g{gen}"
                parent.mkdir(parents=True)
                publish(parent, parent / "ensemble", {
                    "protocol_sha256": digest, "arrays": identity["TEST"], "checkpoint_sha256": checkpoint,
                    "arm": arm, "seed": seed, "generation_updates": gen, "n_generated": 5 if arm == "G5" else 6,
                    "sampler": protocol["samplers"]["TEST"], "state_convention": STATE_CONVENTION,
                    "noise_convention": NOISE_CONVENTION})
    controls = root / "artifacts/controls_test"
    controls.mkdir()
    fit = controls / "gaussian_fit.npz"
    np.savez(fit, residual_mean=np.zeros(72), covariance=np.eye(72))
    for name in mechanism.bootstrap.CONTROLS:
        publish(controls / name, controls / name, {
            "protocol_sha256": digest, "arrays": identity["TEST"], "training_arrays": identity["TRAIN"],
            "procedure": name, "control": protocol["controls"][name], "n_generated": 6,
            "state_convention": "physical_control_standardized_float64",
            "control_fit_sha256": sha256_file(fit) if name == "gaussian" else None})
    (root / "SELECTION.json").write_bytes(json_bytes(selection))
    (root / "TEST_STARTED.json").write_bytes(json_bytes({
        "phase": "TEST_STARTED", "previous_test_exposure": True, "protocol_sha256": digest,
        "selection_sha256": sha256_file(root / "SELECTION.json")}))
    yield root, digest


def cli(study, output):
    root, digest = study
    return ["--study-root", str(root), "--protocol-sha256", digest, "--output", str(output)]


def test_complete_original_gate_cli_anonymous_fresh_and_no_bootstrap(study, monkeypatch, capsys):
    root, digest = study
    original_load, original_read = mechanism.bootstrap.load_study, mechanism.read_ensemble
    calls, ensemble_reads = [], []

    def load(*args):
        calls.append(args)
        return original_load(*args)

    def read(path):
        ensemble_reads.append(path)
        return original_read(path)

    def forbidden(*args, **kwargs):
        raise AssertionError("mechanism must not bootstrap or call the full summary")

    monkeypatch.setattr(mechanism.bootstrap, "load_study", load)
    monkeypatch.setattr(mechanism, "read_ensemble", read)
    monkeypatch.setattr(mechanism.bootstrap, "compare", forbidden)
    monkeypatch.setattr(mechanism.bootstrap, "bootstrap_weights", forbidden)
    monkeypatch.setattr(mechanism.followup, "summarize_study", forbidden)
    before = {p: sha256_file(p) for p in root.rglob("*") if p.is_file()}
    output = root / "mechanism.json"
    assert mechanism.main(cli(study, output)) == 0
    assert len(calls) == 1
    assert len(ensemble_reads) == len(set(ensemble_reads)) == 16
    rec = json.loads(output.read_text())
    assert rec["protocol_sha256"] == digest
    assert rec["producer_source_sha256"]["analysis/prigm_operator_mechanism.py"] == sha256_file(Path(mechanism.__file__))
    assert len(rec["core_source_sha256"]) > 10
    assert rec["n_windows"] == 3 and rec["month_counts"] == {"200508": 2, "200509": 1}
    assert set(rec["methods"]) == set(mechanism.bootstrap.ARMS + mechanism.bootstrap.CONTROLS)
    assert set(rec["D_additional_corrected_selected"]) == {"7"}
    for name, method in rec["methods"].items():
        keys = set(map(str, mechanism.FINAL_SEEDS)) if name in mechanism.bootstrap.ARMS else {"shared"}
        assert set(method["seedwise"]) == keys
        assert method["n_realizations"] == len(keys)
        points = []
        for key, row in method["seedwise"].items():
            records = row["per_window"]
            assert len(records) == 3 and row["n_draws"] == 39
            assert row["verified_primary_scoring_kinds"] == ["raw", "corrected"]
            scores = [r["raw_energy_score"] for r in records]
            point = ((scores[0] + scores[1]) / 2 + scores[2]) / 2
            assert row["metrics"]["raw_energy_score"]["month_balanced"] == pytest.approx(point)
            points.append(point)
        assert method["metrics"]["raw_energy_score"]["month_balanced"] == pytest.approx(sum(points) / len(points))
    decomp = rec["D_selection_decomposition"]
    assert decomp["seedwise"]["7"]["raw_generation"] == 9560
    assert decomp["seedwise"]["7"]["corrected_generation"] == 29636
    assert decomp["seedwise"]["7"]["selection_changed"] is True
    for seed in ("13", "42"):
        assert not decomp["seedwise"][seed]["selection_changed"]
        assert decomp["seedwise"][seed]["metrics"]["selection_effect"]["month_balanced"] == 0
    for seed, item in decomp["seedwise"].items():
        original = json.loads((root / f"artifacts/test/D_s{seed}-g{item['corrected_generation']}/score-corrected.json").read_text())
        assert item["metrics"]["corrected_selected_energy_score"]["month_balanced"] == pytest.approx(original["month_balanced"])
    assert str(root) not in output.read_text() and "/home/" not in output.read_text()
    assert str(root) not in capsys.readouterr().out
    for relative, digest in rec["input_sha256"].items():
        assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
        assert sha256_file(root / relative) == digest
    assert any("secondaries-" in p for p in rec["input_sha256"])
    assert {p: sha256_file(p) for p in before} == before
    with pytest.raises(FileExistsError):
        mechanism.main(cli(study, output))
    second = root / "second.json"
    assert mechanism.main(cli(study, second)) == 0
    assert second.read_bytes() == output.read_bytes()


@pytest.mark.parametrize("damage", ["ids", "checkpoint", "source", "protocol", "payload", "secondary", "missing_secondary"])
def test_original_gate_rejects_wrong_identities_and_missing_payloads(study, damage):
    root, digest = study
    if damage == "protocol":
        digest = "0" * 64
    elif damage == "source":
        path = root / "PROTOCOL.json"
        row = json.loads(path.read_text())
        row["inference_source_hashes"]["prigm/run_score.py"] = "0" * 64
        path.write_bytes(json_bytes(row))
        digest = sha256_file(path)
    elif damage == "payload":
        path = root / "artifacts/test/D_s7-g9560/ensemble/gen_00000000/ensemble.npz"
        path.write_bytes(path.read_bytes() + b"changed")
    elif damage in ("secondary", "missing_secondary"):
        path = root / "artifacts/test/G5_s42-g9560/secondaries-corrected.json"
        if damage == "missing_secondary":
            path.unlink()
        else:
            row = json.loads(path.read_text())
            row["checkpoint_sha256"] = "0" * 64
            path.write_bytes(json_bytes(row))
    else:
        path = root / "artifacts/test/D_s7-g9560/score-raw.json"
        row = json.loads(path.read_text())
        if damage == "ids":
            row["ids"] = row["ids"][::-1]
        else:
            row["checkpoint_sha256"] = "0" * 64
        path.write_bytes(json_bytes(row))
    output = root / "must-not-exist.json"
    with pytest.raises((ContractMismatch, PublicationError, FileNotFoundError)):
        mechanism.main(cli((root, digest), output))
    assert not output.exists()


@pytest.mark.parametrize("field", ["ids", "arrays", "checkpoint_sha256", "generation_updates", "arm", "seed", "protocol_sha256", "sampler"])
def test_reread_payload_binding_is_checked(field):
    ids = ["2005-08-a", "2005-09-b"]
    identity = {"ids_sha256": id_hash(ids)}
    selected = {"checkpoint_sha256": "a" * 64, "generation": 9560}
    protocol = {"samplers": {"TEST": {"n_eval": 50, "n_draws": 39}}}
    selection = {"test_checkpoint_hashes": [selected["checkpoint_sha256"]]}
    meta = {"arrays": identity, "ids_sha256": identity["ids_sha256"], "protocol_sha256": "b" * 64,
            "checkpoint_sha256": selected["checkpoint_sha256"], "generation_updates": 9560,
            "arm": "D", "seed": 7, "n_generated": 6, "sampler": protocol["samplers"]["TEST"]}
    mechanism.validate_payload(meta, ids, ids, identity, protocol, selection, "b" * 64, "diffusion", selected, "D", 7)
    meta = copy.deepcopy(meta)
    sampled = ids[::-1] if field == "ids" else ids
    if field != "ids":
        meta[field] = "wrong"
    with pytest.raises(ContractMismatch):
        mechanism.validate_payload(meta, sampled, ids, identity, protocol, selection, "b" * 64, "diffusion", selected, "D", 7)


def test_coherent_wrong_primary_and_secondary_scores_cannot_replace_actual_draws(study):
    root, digest = study
    parent = root / "artifacts/test/D_s7-g9560"
    path = parent / "score-raw.json"
    primary = json.loads(path.read_text())
    primary["scores"] = (np.asarray(primary["scores"]) + 1.).tolist()
    primary["mean"] += 1.
    primary["month_balanced"] += 1.
    path.write_bytes(json_bytes(primary))
    path = parent / "secondaries-raw.json"
    secondary = json.loads(path.read_text())
    for row in secondary["per_window"]:
        row["es5"] += 1.
    for key in ("mean", "month_balanced"):
        secondary["es5"][key] += 1.
    path.write_bytes(json_bytes(secondary))
    with pytest.raises(ContractMismatch, match="recomputed D_raw"):
        mechanism.analyze_study(root, digest)


def test_final_rehash_rejects_input_change_after_operator_analysis(study, monkeypatch):
    root, digest = study
    original = mechanism.selection_decomposition

    def change(*args):
        result = original(*args)
        path = root / "artifacts/test/I-S_s13-g9560/secondaries-raw.json"
        path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(mechanism, "selection_decomposition", change)
    with pytest.raises(ContractMismatch, match="input changed during mechanism"):
        mechanism.analyze_study(root, digest)


def test_output_dangling_symlink_is_not_followed(tmp_path):
    root = tmp_path
    output = root / "output.json"
    output.symlink_to(root / "absent")
    with pytest.raises(FileExistsError):
        mechanism.main(["--study-root", "unused", "--protocol-sha256", "unused", "--output", str(output)])
    assert not (root / "absent").exists()


@pytest.mark.parametrize("args", [[], ["--study-root", "absent", "--output", "unused"]])
def test_cli_requires_explicit_protocol(args):
    with pytest.raises(SystemExit):
        mechanism.main(args)
