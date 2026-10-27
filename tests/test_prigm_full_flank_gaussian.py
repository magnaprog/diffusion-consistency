"""Check Gaussian regression using synthetic arrays and independently computed fits.

The historical full_flank name denotes all 216 observed coordinates in the
two intervals surrounding the gap. The regression predicts 60 hidden
coordinates relative to interpolation between the adjacent observed bins.
Tests compare ridge fits and residual covariance with direct calculations,
keep the intercept unpenalized, and ensure fitting uses TRAIN values only.

Prediction checks change the hidden truth and the observed context separately:
hidden truth must have no effect, whereas an observed value may affect the
prediction. These are checks of the implementation, not evidence that extra
context alone explains a difference from Gaussian interpolation.
"""

import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from analysis import prigm_full_flank_gaussian as fg
from prigm.artifacts import id_hash, json_bytes, load_arrays, read_ensemble
from prigm.normalize import AffineNormalizer, fit_normalizer
from prigm.observation import derive_vv, summaries_from_vectors
from prigm.recovery import ContractMismatch, PublicationError, sha256_file
from prigm.run_score import decode_ensemble
from prigm.run_secondaries import window_diagnostics


@pytest.fixture
def study(tmp_path, monkeypatch):
    counts = {"TRAIN": 31, "VAL": 3, "TEST": 4}
    monkeypatch.setattr(fg, "EXPECTED_WINDOWS", counts)
    monkeypatch.setattr(fg, "CHUNK_WINDOWS", 11)
    rng = np.random.Generator(np.random.PCG64(302))
    arrays = {split: summaries_from_vectors(rng.normal(size=(n, 48, 5, 3)))
              for split, n in counts.items()}
    nrm = fit_normalizer(arrays["TRAIN"].reshape(-1, 6))
    identities, paths = {}, {}
    for split, z in arrays.items():
        # VAL intentionally has unequal month counts.
        month = {"TRAIN": "01", "VAL": "11", "TEST": "08"}[split]
        ids = [f"2005-{month}-{i + 1:02d}T00:00:00" for i in range(len(z))]
        if split == "VAL":
            ids[-1] = "2005-12-01T00:00:00"
        path = tmp_path / f"{split.lower()}.npz"
        np.savez(path, z=z, ids=np.asarray(ids), split=np.asarray(split),
                 mean=nrm.mean, scale=nrm.scale, b_ref_sq=np.asarray(nrm.b_ref_sq),
                 n_bins=np.asarray(nrm.n_bins))
        identities[split] = load_arrays(path, split)[3]
        paths[split] = path
    protocol = {
        "schema_version": 2, "exposure": "historical_test_exposed", "arrays": identities,
        "array_root": "/private/host/not-for-publication",  # never copied to extension JSON
        "inference_source_hashes": {
            f"prigm/{p.name}": sha256_file(p)
            for p in (Path(fg.__file__).resolve().parents[1] / "prigm").glob("*.py")
        },
    }
    protocol_path = tmp_path / "PROTOCOL.json"
    protocol_path.write_bytes(json_bytes(protocol))
    output = tmp_path / "extension"
    return {"root": tmp_path, "paths": paths, "output": output, "protocol": protocol,
            "hash": sha256_file(protocol_path), "arrays": arrays, "nrm": nrm}


def cli(study, stage, receipt=None):
    args = ["--stage", stage, "--train-arrays", str(study["paths"]["TRAIN"]),
            "--val-arrays", str(study["paths"]["VAL"]),
            "--test-arrays", str(study["paths"]["TEST"]),
            "--source-study-root", str(study["root"]), "--source-protocol-sha256", study["hash"],
            "--output-root", str(study["output"])]
    if receipt is not None:
        args += ["--fit-select-sha256", receipt]
    return fg.main(args)


def fit_select(study):
    assert cli(study, "fit-select") == 0
    return sha256_file(study["output"] / "FIT_SELECT.json")


def read(path):
    return json.loads(path.read_text())


def rewrite(path, value):
    path.write_bytes(json_bytes(value))


def repin(study):
    """Deliberately repin tampering to test semantic checks beyond byte hashes."""
    out = study["output"]
    fit_hash = sha256_file(out / "FIT.npz")
    for filename in ("FIT.json", "VALIDATION.json", "SELECTION.json"):
        value = read(out / filename)
        value["fit_payload_sha256"] = fit_hash
        if filename == "SELECTION.json":
            value["validation_sha256"] = sha256_file(out / "VALIDATION.json")
        rewrite(out / filename, value)
    receipt = read(out / "FIT_SELECT.json")
    receipt["files"] = {name: sha256_file(out / name) for name in fg.FIT_FILES}
    rewrite(out / "FIT_SELECT.json", receipt)
    return sha256_file(out / "FIT_SELECT.json")


def deny_test_access(study, monkeypatch):
    real_open, real_load = Path.open, np.load
    test_path = study["paths"]["TEST"]

    def guarded(path, *args, **kwargs):
        if path == test_path:
            pytest.fail("TEST payload accessed before fit/selection verification")
        return real_open(path, *args, **kwargs)

    def guarded_load(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path) == test_path:
            pytest.fail("TEST payload loaded before fit/selection verification")
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    monkeypatch.setattr(np, "load", guarded_load)


def test_train_centering_ridge_and_population_covariance_against_lstsq(study, monkeypatch):
    z, nrm = study["arrays"]["TRAIN"], study["nrm"]
    x, y = fg.design(z, nrm)
    xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
    calls = {"stats": 0, "solve": 0}
    original_gaussian, original_solve = fg.gaussian_fit, np.linalg.solve

    def gaussian_spy(batch, rho, jitter):
        calls["stats"] += 1
        assert len(batch) <= fg.CHUNK_WINDOWS
        assert batch.shape[1] == 276 and rho == jitter == 0
        return original_gaussian(batch, rho, jitter)

    def solve_spy(a, b):
        calls["solve"] += 1
        return original_solve(a, b)

    monkeypatch.setattr(fg, "gaussian_fit", gaussian_spy)
    monkeypatch.setattr(np.linalg, "solve", solve_spy)
    stats = fg.sufficient_statistics(z, nrm)
    fit = fg.fit_from_statistics(stats)
    assert calls == {"stats": math.ceil(len(z) / fg.CHUNK_WINDOWS), "solve": 3}
    np.testing.assert_allclose(stats["x_mean"], x.mean(axis=0), atol=1e-14)
    np.testing.assert_allclose(stats["y_mean"], y.mean(axis=0), atol=1e-14)
    np.testing.assert_allclose(stats["cxx"], xc.T @ xc / len(x), atol=1e-14)
    for i, lam in enumerate(fg.LAMBDAS):
        # Independent augmented least squares implements the declared /N objective.
        a = np.vstack((xc, np.sqrt(len(x) * lam) * np.eye(216)))
        b = np.vstack((yc, np.zeros((216, 60))))
        w = np.linalg.lstsq(a, b, rcond=None)[0]
        intercept = y.mean(axis=0) - x.mean(axis=0) @ w
        np.testing.assert_allclose(fit["weights"][i], w, rtol=1e-8, atol=2e-11)
        np.testing.assert_allclose(fit["intercepts"][i], intercept, rtol=1e-8, atol=2e-11)
        errors = y - (x @ w + intercept)
        centered_errors = errors - errors.mean(axis=0)
        population = centered_errors.T @ centered_errors / len(x)
        np.testing.assert_allclose(fit["residual_covariances"][i], population, atol=3e-12)
        np.testing.assert_allclose(errors.mean(axis=0), 0, atol=1e-13)
        for rho in fg.RHOS:
            chol = fg.candidate_factors(fit)[lam, rho]
            expected = (1 - rho) * population + rho * np.diag(np.diag(population)) + 1e-6 * np.eye(60)
            np.testing.assert_allclose(chol @ chol.T, expected, atol=3e-12)
            assert np.all(np.diag(chol) > 0)


def test_unpenalized_intercept_and_train_only_fit(study):
    z, nrm = study["arrays"]["TRAIN"], study["nrm"]
    initial = fg.fit_from_statistics(fg.sufficient_statistics(z, nrm))
    # No VAL or TEST values are arguments to either fitting routine.
    study["arrays"]["VAL"][:] += 999
    study["arrays"]["TEST"][:] *= -9
    again = fg.fit_from_statistics(fg.sufficient_statistics(z, nrm))
    for key in initial:
        np.testing.assert_array_equal(initial[key], again[key])
    shift = np.linspace(-2, 2, 60)
    shifted_stats = fg.sufficient_statistics(z, nrm)
    shifted_stats["y_mean"] = shifted_stats["y_mean"] + shift
    shifted = fg.fit_from_statistics(shifted_stats)
    np.testing.assert_array_equal(shifted["weights"], initial["weights"])
    np.testing.assert_allclose(shifted["intercepts"] - initial["intercepts"], np.broadcast_to(shift, (3, 60)), atol=1e-14)


def test_observation_design_and_predictions_ignore_hidden_truth(study):
    z, nrm = study["arrays"]["TRAIN"], study["nrm"]
    x, y = fg.design(z, nrm)
    assert x.shape == (31, 216) and y.shape == (31, 60)
    observed = np.concatenate((z[:, :18], z[:, 30:48]), axis=1)
    np.testing.assert_array_equal(x, ((observed - nrm.mean) / nrm.scale).reshape(31, 216))
    weights = (np.arange(18, 30) - 17) / 13
    reference = (1 - weights)[None, :, None] * z[:, 17:18, :5] + weights[None, :, None] * z[:, 30:31, :5]
    np.testing.assert_array_equal(y, ((z[:, 18:30, :5] - reference) / nrm.scale[:5]).reshape(31, 60))
    truth = z[0]
    w = np.zeros((216, 60))
    w[0, :] = 0.2  # a nonboundary observed coordinate must affect prediction
    params = (w, np.zeros(60), np.eye(60) * 0.1, fg.standard_noise("VAL", "2005-11-01"))
    draws = fg.draw_window(truth, nrm, *params)
    assert draws.shape == (39, 6, 48) and draws.dtype == np.float64
    physical = decode_ensemble(draws, nrm)
    np.testing.assert_allclose(physical, derive_vv(physical[..., :5]), atol=1e-12)
    flank = np.r_[0:18, 30:48]
    expected = np.broadcast_to(nrm.transform(truth).T[:, flank], (39, 6, 36))
    np.testing.assert_array_equal(draws[:, :, flank], expected)
    hidden_changed = truth.copy()
    hidden_changed[18:30] += 1000
    np.testing.assert_array_equal(draws, fg.draw_window(hidden_changed, nrm, *params))
    context_changed = truth.copy()
    context_changed[0, 0] += 1
    changed = fg.draw_window(context_changed, nrm, *params)
    assert not np.array_equal(draws[:, :5, 18:30], changed[:, :5, 18:30])


def test_common_noise_hash_contract_split_window_and_draw():
    wid = "2005-11-01T00:00:00"
    noise = fg.standard_noise("VAL", wid)
    np.testing.assert_array_equal(noise, fg.standard_noise("VAL", wid))
    assert not np.array_equal(noise, fg.standard_noise("TEST", wid))
    assert not np.array_equal(noise, fg.standard_noise("VAL", wid + "different"))
    assert not np.array_equal(noise[0], noise[1])
    # Independent serialization, not a call through the producer's key helper.
    parts = [b"prigm-noise", b"full-flank-gaussian-v1", b"VAL", wid.encode(),
             struct.pack("<q", 0), struct.pack("<q", 13)]
    seed = int.from_bytes(hashlib.sha256(b"|".join(parts) + b"|").digest()[:8], "little")
    expected = np.random.Generator(np.random.PCG64(seed)).standard_normal(60)
    np.testing.assert_array_equal(noise[0], expected)
    with pytest.raises(ContractMismatch):
        fg.standard_noise("TRAIN", wid)


def test_innovations_common_across_nine_candidates(study, monkeypatch):
    nrm = study["nrm"]
    fit = fg.fit_from_statistics(fg.sufficient_statistics(study["arrays"]["TRAIN"], nrm))
    z = study["arrays"]["VAL"][:1]
    ids = ["2005-11-01"]
    innovations = []
    original = fg.draw_window

    def spy(truth, nrm, w, intercept, chol, noise):
        innovations.append(noise.copy())
        result = original(truth, nrm, w, intercept, chol, noise)
        u = result[:, :5, 18:30].transpose(0, 2, 1).reshape(39, 60)
        x = nrm.transform(np.concatenate((truth[:18], truth[30:]))).reshape(216)
        reference = fg.linear_interpolate(truth)[18:30, :5]
        mean = ((reference - nrm.mean[:5]) / nrm.scale[:5]).reshape(60) + x @ w + intercept
        recovered = np.linalg.solve(chol, (u - mean).T).T
        np.testing.assert_allclose(recovered, noise, atol=1e-10)
        return result

    monkeypatch.setattr(fg, "draw_window", spy)
    result = fg.validation_scores(fit, z, ids, nrm)
    assert len(result["candidates"]) == len(innovations) == 9
    for noise in innovations:
        np.testing.assert_array_equal(noise, innovations[0])


def grid_fixture():
    return {"ids": ["2005-11-a", "2005-11-b", "2005-12-c"], "months": [200511, 200511, 200512],
            "candidates": [{"lambda": lam, "rho": rho, "scores": {"raw": [3., 3., 3.], "corrected": [1., 1., 1.]}}
                           for lam, rho in reversed(fg.GRID)]}


def test_grid_exact_ties_independent_selection_and_month_balance():
    grid = grid_fixture()
    selection = fg.select_grid(grid)
    assert selection["raw"] == {"lambda": .001, "rho": 0., "month_balanced_primary_es": 3.}
    assert selection["corrected"]["lambda"] == .001 and selection["corrected"]["rho"] == 0
    by_pair = {(r["lambda"], r["rho"]): r for r in grid["candidates"]}
    by_pair[.001, 0.]["scores"]["raw"] = [0., 0., 4.]  # pooled 1.33, month-balanced 2
    by_pair[.1, .5]["scores"]["raw"] = [2.2, 2.2, 0.]  # pooled 1.47, month-balanced 1.1
    selection = fg.select_grid(grid)
    assert selection["raw"] == {"lambda": .1, "rho": .5, "month_balanced_primary_es": 1.1}
    assert selection["corrected"]["lambda"] == .001
    assert fg.selected_pairs({"selected": selection}) == [(.001, 0.), (.1, .5)]
    # Near ties are not exact ties; no rounded or tolerance-based tie rule.
    by_pair[1., 1.]["scores"]["corrected"] = [np.nextafter(1., 0.)] * 3
    assert fg.select_grid(grid)["corrected"]["lambda"] == 1.


@pytest.mark.parametrize("defect", ["missing", "duplicate", "nonfinite", "partial", "months"])
def test_grid_refuses_incomplete_or_changed_population(defect):
    grid = grid_fixture()
    if defect == "missing":
        grid["candidates"].pop()
    elif defect == "duplicate":
        grid["candidates"][0] = grid["candidates"][1]
    elif defect == "nonfinite":
        grid["candidates"][0]["scores"]["raw"][0] = float("nan")
    elif defect == "partial":
        grid["candidates"][0]["scores"]["raw"].pop()
    else:
        grid["months"][0] += 1
    with pytest.raises(ContractMismatch):
        fg.select_grid(grid)


def test_fixed_jitter_never_adapts():
    bad = {"residual_covariances": np.repeat((-2e-6 * np.eye(60))[None], 3, axis=0)}
    with pytest.raises(ContractMismatch, match="fixed jitter"):
        fg.candidate_factors(bad)
    zero = {"residual_covariances": np.zeros((3, 60, 60))}
    for factor in fg.candidate_factors(zero).values():
        np.testing.assert_array_equal(factor, .001 * np.eye(60))


def test_fit_select_never_reads_test_even_if_supplied(study, monkeypatch):
    deny_test_access(study, monkeypatch)
    calls = []
    original = fg.load_arrays

    def track(path, split, expected):
        calls.append(split)
        return original(path, split, expected)

    monkeypatch.setattr(fg, "load_arrays", track)
    receipt = fit_select(study)
    assert calls == ["TRAIN", "VAL"]
    assert len(receipt) == 64 and not (study["output"] / "test").exists()
    out = study["output"]
    selection = read(out / "SELECTION.json")
    assert selection["fit_payload_sha256"] == sha256_file(out / "FIT.npz")
    assert selection["validation_sha256"] == sha256_file(out / "VALIDATION.json")
    contract = read(out / "EXTENSION.json")
    assert contract["arrays"]["TEST"] == study["protocol"]["arrays"]["TEST"]
    assert contract["noise"] == fg.NOISE_CONTRACT
    for path in out.glob("*.json"):
        text = path.read_text()
        assert str(study["root"]) not in text and "/private/host" not in text


def test_test_access_only_after_verified_selection_and_phase(study, monkeypatch):
    receipt = fit_select(study)
    original = fg.load_arrays
    order = []

    def tracked(path, split, expected):
        order.append(split)
        if split == "TEST":
            out = study["output"]
            assert sha256_file(out / "FIT_SELECT.json") == receipt
            phase = read(out / "test/TEST_STARTED.json")
            assert phase["selection_sha256"] == sha256_file(out / "SELECTION.json")
            assert phase["fit_payload_sha256"] == sha256_file(out / "FIT.npz")
        return original(path, split, expected)

    monkeypatch.setattr(fg, "load_arrays", tracked)
    assert cli(study, "test", receipt) == 0
    assert order == ["TRAIN", "VAL", "TEST"]
    completed = read(study["output"] / "test/COMPLETED.json")
    assert completed["n_windows"] == 4 and completed["n_draws"] == 39
    assert 1 <= len(completed["selected_union"]) <= 2
    before = {str(p.relative_to(study["output"])): sha256_file(p)
              for p in study["output"].rglob("*") if p.is_file()}
    assert cli(study, "verify", receipt) == 0
    after = {str(p.relative_to(study["output"])): sha256_file(p)
             for p in study["output"].rglob("*") if p.is_file()}
    assert before == after
    scores = read(study["output"] / "test/SCORES.json")
    assert scores["n"] == 4 and scores["complete_case"] is True
    for name in completed["selected_union"]:
        ens, ids, meta = read_ensemble(study["output"] / "test" / name)
        assert ens.shape == (4, 39, 6, 48) and ens.dtype == np.float64
        assert meta["protocol_sha256"] != study["hash"]
        assert meta["source_protocol_sha256"] == study["hash"]
        assert meta["selected_for"] and meta["complete_case"] is True
        assert meta["arrays"] == study["protocol"]["arrays"]["TEST"]
        assert ids == scores["ids"]
        raw = scores["candidates"][name]["raw"]
        corrected = scores["candidates"][name]["corrected"]
        for row in corrected["per_window"]:
            assert row["support_violation_beyond_rounding_fraction"] == 0
        # Correction preserves mu; the vector-energy endpoint is unchanged.
        assert raw["summary"]["vector_energy"] == corrected["summary"]["vector_energy"]
        physical = decode_ensemble(ens[0], study["nrm"])
        secondary = window_diagnostics(physical, study["arrays"]["TEST"][0], study["nrm"], np.float64)
        mapping = {"primary_es": "es5", "coverage90": "cov90_mean", "width90": "width90_mean",
                   "vector_energy": "es_vec", "resolved_energy_proxy_crps": "proxy_crps",
                   "support_violation_beyond_rounding_fraction": "frac_support_violation_beyond_rounding"}
        for key, existing in mapping.items():
            assert raw["per_window"][0][key] == secondary[existing]


@pytest.mark.parametrize("separate", [False, True])
def test_publishes_only_one_or_two_selected_ensembles(study, monkeypatch, separate):
    original = fg.validation_scores

    def controlled_validation(fit, z, ids, nrm):
        # Exercise both selection-union paths without choosing fixtures by outcomes.
        result = original(fit, z, ids, nrm)
        for row in result["candidates"]:
            pair = (row["lambda"], row["rho"])
            for kind in ("raw", "corrected"):
                winner = (1., 1.) if separate and kind == "corrected" else (.001, 0.)
                row["scores"][kind] = [1. if pair == winner else 2.] * len(ids)
        return result

    monkeypatch.setattr(fg, "validation_scores", controlled_validation)
    receipt = fit_select(study)
    cli(study, "test", receipt)
    completed = read(study["output"] / "test/COMPLETED.json")
    assert len(completed["selected_union"]) == (2 if separate else 1)
    for name in completed["selected_union"]:
        _, _, meta = read_ensemble(study["output"] / "test" / name)
        assert len(meta["selected_for"]) == (1 if separate else 2)
    cli(study, "verify", receipt)


@pytest.mark.parametrize("file", ["FIT.npz", "FIT.json", "VALIDATION.json", "SELECTION.json", "FIT_SELECT.json"])
def test_byte_tampering_refused_before_test_access(study, monkeypatch, file):
    receipt = fit_select(study)
    path = study["output"] / file
    path.write_bytes(path.read_bytes() + b" ")
    deny_test_access(study, monkeypatch)
    with pytest.raises(ContractMismatch, match="hash changed"):
        cli(study, "test", receipt)
    assert not (study["output"] / "test").exists()


@pytest.mark.parametrize("parameter", ["weights", "intercepts", "residual_covariances", "x_mean", "n_train"])
def test_rehashed_fit_parameter_tampering_fails_train_recomputation(study, monkeypatch, parameter):
    fit_select(study)
    path = study["output"] / "FIT.npz"
    with np.load(path, allow_pickle=False) as blob:
        payload = {key: blob[key] for key in blob.files}
    payload[parameter].flat[0] += 1
    np.savez(path, **payload)
    receipt = repin(study)
    deny_test_access(study, monkeypatch)
    with pytest.raises(ContractMismatch, match="TRAIN fit .* failed recomputation"):
        cli(study, "test", receipt)


@pytest.mark.parametrize("file", ["SELECTION.json", "VALIDATION.json", "EXTENSION.json"])
def test_rehashed_selection_scores_or_producer_tampering_refused(study, monkeypatch, file):
    fit_select(study)
    path = study["output"] / file
    value = read(path)
    if file == "SELECTION.json":
        value["selected"]["raw"]["month_balanced_primary_es"] += 1
    elif file == "VALIDATION.json":
        value["candidates"][0]["scores"]["raw"][0] += 1
    else:
        value["producer_source_hashes"]["analysis/prigm_full_flank_gaussian.py"] = "0" * 64
    rewrite(path, value)
    receipt = repin(study)
    deny_test_access(study, monkeypatch)
    with pytest.raises(ContractMismatch):
        cli(study, "test", receipt)


@pytest.mark.parametrize("defect", ["metadata", "partial", "flanks", "vv", "scores", "phase", "payload", "extra"])
def test_verify_rejects_changed_selected_complete_case_evidence(study, defect):
    receipt = fit_select(study)
    cli(study, "test", receipt)
    out = study["output"]
    completed = read(out / "test/COMPLETED.json")
    name = completed["selected_union"][0]
    root = out / "test" / name / "gen_00000000"
    commit = read(root / "COMMIT.json")
    if defect == "metadata":
        commit["selected_for"] = []
        rewrite(root / "COMMIT.json", commit)
    elif defect in ("partial", "flanks", "vv"):
        with np.load(root / "ensemble.npz") as blob:
            ens, ids = blob["ensemble"], blob["ids"]
        if defect == "partial":
            ens, ids = ens[:-1], ids[:-1]
        elif defect == "flanks":
            ens[0, 0, 0, 0] += 1
        else:
            ens[0, 0, 5, 20] += 1
        np.savez(root / "ensemble.npz", ensemble=ens, ids=ids)
        commit["files"]["ensemble.npz"] = sha256_file(root / "ensemble.npz")
        commit["ids_sha256"] = id_hash(ids.tolist())
        rewrite(root / "COMMIT.json", commit)
    elif defect == "scores":
        path = out / "test/SCORES.json"
        scores = read(path)
        scores["candidates"][name]["raw"]["per_window"][0]["coverage90"] += .1
        rewrite(path, scores)
    elif defect == "phase":
        path = out / "test/TEST_STARTED.json"
        phase = read(path)
        phase["selection_sha256"] = "0" * 64
        rewrite(path, phase)
    elif defect == "extra":
        (out / "test/unselected-candidate").mkdir()
    else:
        (root / "ensemble.npz").write_bytes(b"corrupted")
    with pytest.raises((ContractMismatch, PublicationError)):
        cli(study, "verify", receipt)


@pytest.mark.parametrize("split", ["TRAIN", "VAL", "TEST"])
def test_original_array_payload_pin_enforced(study, split):
    receipt = fit_select(study)
    with study["paths"][split].open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ContractMismatch, match="array hash changed"):
        cli(study, "test", receipt)


def test_original_protocol_and_core_hashes_enforced(study):
    study["hash"] = "0" * 64
    with pytest.raises(PublicationError, match="protocol"):
        cli(study, "fit-select")
    protocol_path = study["root"] / "PROTOCOL.json"
    protocol = read(protocol_path)
    protocol["inference_source_hashes"]["prigm/controls.py"] = "0" * 64
    rewrite(protocol_path, protocol)
    study["hash"] = sha256_file(protocol_path)
    with pytest.raises(ContractMismatch, match="source changed"):
        cli(study, "fit-select")
    assert not study["output"].exists()


def test_fresh_outputs_and_external_receipt_pin_required(study, monkeypatch):
    receipt = fit_select(study)
    deny_test_access(study, monkeypatch)
    with pytest.raises(FileExistsError):
        cli(study, "fit-select")
    with pytest.raises(SystemExit):
        cli(study, "test")
    with pytest.raises(ContractMismatch, match="receipt hash changed"):
        cli(study, "test", "0" * 64)
    (study["output"] / "test").mkdir()
    (study["output"] / "test/partial").write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        cli(study, "test", receipt)
    assert (study["output"] / "test/partial").read_bytes() == b"preserve"


def test_inclusive_2nd_38th_intervals_are_not_interpolated_quantiles():
    nrm = AffineNormalizer(np.zeros(6), np.ones(6), 1., 10)
    z5 = np.broadcast_to(np.arange(39.)[:, None, None], (39, 12, 5)).copy()
    samples = derive_vv(z5)
    truth = np.zeros((48, 6))
    truth[18:30] = derive_vv(np.ones((12, 5)))  # exactly second order statistic
    metrics = fg.window_metrics(samples, truth, nrm)
    assert metrics["coverage90"] == 1. and metrics["width90"] == 36.
    truth[18:30] = derive_vv(np.full((12, 5), 37.))
    assert fg.window_metrics(samples, truth, nrm)["coverage90"] == 1.
    truth[18:30] = derive_vv(np.full((12, 5), np.nextafter(37., 38.)))
    assert fg.window_metrics(samples, truth, nrm)["coverage90"] == 0.
