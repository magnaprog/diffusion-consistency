"""Fit and evaluate the Gaussian regression added after the PriGM TEST results were known.

The mean uses all 216 observed coordinates: six channels in each of the 18
bins before and 18 bins after the missing interval. It predicts 60 residual
coordinates, comprising five primary channels in each of the twelve hidden
bins, relative to interpolation across the gap. Gaussian noise is joint in
those 60 coordinates; v_v is then derived in physical units. The covariance
does not vary with the observed context. Every saved TRAIN window, including
overlapping windows, receives equal weight. This is not blind confirmation.

Run fit-select with TRAIN/VAL and a fresh output root. Then run test with that
root, TEST, and the printed --fit-select-sha256. Before opening TEST, this
stage recomputes the TRAIN fit and VAL selection. The read-only verify stage
also regenerates the selected TEST draws and checks scores and complete-case
IDs. This runner has access to more inputs than the public paper verifier.

JSON paths are relative to the output root or repository. Interrupted roots
are retained and refused; they are not resumed or overwritten.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prigm.artifacts import (
    id_hash, json_bytes, load_arrays, publish_ensemble, read_ensemble,
    require_source_match,
)
from prigm.barrier import load_protocol
from prigm.config import (
    GAUSSIAN_JITTER, HIDDEN_END, HIDDEN_START, N_BINS, N_CHANNELS, N_DRAWS,
    N_TEST_WINDOWS, N_TRAIN_STARTS, N_VAL_WINDOWS, PRIMARY_CHANNELS,
)
from prigm.controls import gaussian_fit, linear_interpolate
from prigm.keys import noise_key, pcg64
from prigm.observation import closure_correct, derive_vv, support_gaps
from prigm.precision import arithmetic_tolerance
from prigm.recovery import (
    ContractMismatch, atomic_write_bytes, read_json, sha256_file,
)
from prigm.run_score import decode_ensemble, hidden_primary_std
from prigm.run_secondaries import summarize_diagnostics
from prigm.scoring import (
    coverage_inclusive, crps_univariate, energy_density_proxy, energy_score,
    interval_width, month_balanced_mean,
)

LAMBDAS = (0.001, 0.1, 1.0)
RHOS = (0.0, 0.5, 1.0)
GRID = tuple((lam, rho) for lam in LAMBDAS for rho in RHOS)
X_DIM, Y_DIM = 216, 60
CHUNK_WINDOWS = 4096
EXPECTED_WINDOWS = {"TRAIN": N_TRAIN_STARTS, "VAL": N_VAL_WINDOWS, "TEST": N_TEST_WINDOWS}
NOISE_STREAM = "full-flank-gaussian-v1"
NOISE_SEED = 13
NOISE_CONTRACT = {
    "stream": NOISE_STREAM, "seed": NOISE_SEED, "n_draws": N_DRAWS,
    "key": "prigm.keys.noise_key(stream, split, window_id, draw_index_0_to_38, seed)",
    "hash": "SHA256 first 8 bytes little-endian; prigm.keys.digest_u64 encoding",
    "generator": "PCG64 per draw; standard_normal(60), float64, bin-major primary-channel order",
    "coupling": "same noise for all nine candidates and raw/corrected; split and window keyed",
}
SCOPE = "selected complete-case windows; artificial hidden gap; historical TEST exposed"
FIT_FILES = ("EXTENSION.json", "FIT.npz", "FIT.json", "VALIDATION.json", "SELECTION.json")


def design(z, nrm):
    """Build regression inputs and interpolation residuals from physical windows.

    For z with shape (N, 48, 6), X has shape (N, 216): standardized values
    from all six channels in the 36 observed bins. Y has shape (N, 60): the
    five primary channels in the twelve hidden bins, minus their interpolated
    values, divided by the corresponding TRAIN scales. Both arrays flatten
    positions before channels.
    """
    x = nrm.transform(np.concatenate((z[:, :HIDDEN_START], z[:, HIDDEN_END:]), axis=1))
    reference = linear_interpolate(z)[:, HIDDEN_START:HIDDEN_END, :PRIMARY_CHANNELS]
    y = (z[:, HIDDEN_START:HIDDEN_END, :PRIMARY_CHANNELS] - reference) / nrm.scale[:PRIMARY_CHANNELS]
    return x.reshape(len(z), X_DIM), y.reshape(len(z), Y_DIM)


def sufficient_statistics(z_train, nrm):
    """One bounded pass; stable centered batch merging, population divisor N."""
    count = 0
    mean = np.zeros(X_DIM + Y_DIM)
    scatter = np.zeros((X_DIM + Y_DIM, X_DIM + Y_DIM))
    for start in range(0, len(z_train), CHUNK_WINDOWS):
        x, y = design(z_train[start:start + CHUNK_WINDOWS], nrm)
        batch = np.concatenate((x, y), axis=1)
        batch_mean, batch_cov = gaussian_fit(batch, rho=0.0, jitter=0.0)
        n = len(batch)
        delta = batch_mean - mean
        total = count + n
        scatter += n * batch_cov + (count * n / total) * np.outer(delta, delta)
        mean += (n / total) * delta
        count = total
    if count < 2:
        raise ContractMismatch("at least two TRAIN windows required")
    covariance = scatter / count
    result = {
        "n_train": np.asarray(count, dtype=np.int64),
        "x_mean": mean[:X_DIM], "y_mean": mean[X_DIM:],
        "cxx": covariance[:X_DIM, :X_DIM], "cxy": covariance[:X_DIM, X_DIM:],
        "cyy": covariance[X_DIM:, X_DIM:],
    }
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ContractMismatch("nonfinite TRAIN sufficient statistics")
    return result


def fit_from_statistics(stats):
    """Fit one linear mean for each of the three ridge coefficients.

    Centering leaves the intercept unpenalized. For each mean fit, calculate
    the residual covariance from the stored population moments. The later
    three covariance-shrinkage settings reuse that mean, producing nine
    candidates in total. Positive definiteness must hold with the fixed jitter;
    the function does not tune the jitter after a failure.
    """
    cxx, cxy, cyy = (stats[k] for k in ("cxx", "cxy", "cyy"))
    weights, intercepts, residual_covariances = [], [], []
    for lam in LAMBDAS:
        w = np.linalg.solve(cxx + lam * np.eye(X_DIM), cxy)
        intercept = stats["y_mean"] - stats["x_mean"] @ w
        residual_cov = cyy - cxy.T @ w - w.T @ cxy + w.T @ cxx @ w
        residual_cov = (residual_cov + residual_cov.T) / 2.0
        weights.append(w)
        intercepts.append(intercept)
        residual_covariances.append(residual_cov)
    fit = {**stats, "weights": np.stack(weights), "intercepts": np.stack(intercepts),
           "residual_covariances": np.stack(residual_covariances)}
    if any(not np.isfinite(v).all() for v in fit.values()):
        raise ContractMismatch("nonfinite Gaussian fit")
    candidate_factors(fit)  # Fail closed with the fixed jitter, never tune it.
    return fit


def candidate_factors(fit):
    factors = {}
    for i, lam in enumerate(LAMBDAS):
        cov = fit["residual_covariances"][i]
        for rho in RHOS:
            shrunk = (1.0 - rho) * cov + rho * np.diag(np.diag(cov))
            try:
                factors[lam, rho] = np.linalg.cholesky(shrunk + GAUSSIAN_JITTER * np.eye(Y_DIM))
            except np.linalg.LinAlgError as exc:
                raise ContractMismatch("covariance is not positive definite at the fixed jitter") from exc
    return factors


def standard_noise(split, window_id):
    """Candidate-independent 39x60 innovations, keyed separately for each draw."""
    if split not in ("VAL", "TEST") or not isinstance(window_id, str) or not window_id:
        raise ContractMismatch("noise requires VAL/TEST and a nonempty window ID")
    return np.stack([
        pcg64(noise_key(NOISE_STREAM, split, window_id, draw, NOISE_SEED)).standard_normal(Y_DIM)
        for draw in range(N_DRAWS)
    ])


def draw_window(z, nrm, weight, intercept, factor, noise):
    """39x6x48 float64 stored control; no hidden truth in the predictive mean."""
    if noise.shape != (N_DRAWS, Y_DIM):
        raise ContractMismatch("innovations must be 39 by 60")
    context = nrm.transform(np.concatenate((z[:HIDDEN_START], z[HIDDEN_END:]), axis=0)).reshape(X_DIM)
    reference = linear_interpolate(z)[HIDDEN_START:HIDDEN_END, :PRIMARY_CHANNELS]
    loc = (reference - nrm.mean[:PRIMARY_CHANNELS]) / nrm.scale[:PRIMARY_CHANNELS]
    u5 = (loc.reshape(Y_DIM) + context @ weight + intercept + noise @ factor.T)
    z5 = u5.reshape(N_DRAWS, 12, PRIMARY_CHANNELS) * nrm.scale[:PRIMARY_CHANNELS] + nrm.mean[:PRIMARY_CHANNELS]
    draws = np.repeat(nrm.transform(z)[None], N_DRAWS, axis=0)
    draws[:, HIDDEN_START:HIDDEN_END] = nrm.transform(derive_vv(z5))
    return draws.transpose(0, 2, 1)


def candidate_name(lam, rho):
    if (lam, rho) not in GRID:
        raise ContractMismatch("candidate outside fixed grid")
    return f"lambda_{lam:g}_rho_{rho:g}"


def months_for(ids):
    try:
        months = np.asarray([int(w[:4]) * 100 + int(w[5:7]) for w in ids], dtype=np.int64)
        if any(w[4:5] != "-" for w in ids) or np.any((months % 100 < 1) | (months % 100 > 12)):
            raise ValueError("invalid calendar month")
    except (ValueError, TypeError) as exc:
        raise ContractMismatch("window IDs must identify calendar months") from exc
    return months


def validation_scores(fit, z, ids, nrm):
    factors = candidate_factors(fit)
    scores = {pair: {kind: [] for kind in ("raw", "corrected")} for pair in GRID}
    for truth, wid in zip(z, ids):
        noise = standard_noise("VAL", wid)
        target = hidden_primary_std(truth[None, HIDDEN_START:HIDDEN_END], nrm)[0]
        for lam, rho in GRID:
            i = LAMBDAS.index(lam)
            stored = draw_window(truth, nrm, fit["weights"][i], fit["intercepts"][i], factors[lam, rho], noise)
            raw = decode_ensemble(stored, nrm)
            for kind, samples in (("raw", raw), ("corrected", closure_correct(raw))):
                scores[lam, rho][kind].append(energy_score(hidden_primary_std(samples, nrm), target))
    return {"ids": ids, "months": months_for(ids).tolist(), "candidates": [
        {"lambda": lam, "rho": rho, "scores": scores[lam, rho]} for lam, rho in GRID
    ]}


def select_grid(validation):
    """Independent minima using full precision and exact lexicographic ties."""
    ids = validation["ids"]
    months = months_for(ids)
    rows = validation["candidates"]
    if (not ids or len(set(ids)) != len(ids) or validation["months"] != months.tolist()
            or len(rows) != len(GRID)
            or {(r["lambda"], r["rho"]) for r in rows} != set(GRID)):
        raise ContractMismatch("selection requires the complete nine-candidate VAL grid")
    selected = {}
    for kind in ("raw", "corrected"):
        choices = []
        for row in rows:
            values = np.asarray(row["scores"][kind], dtype=np.float64)
            if values.shape != (len(ids),) or not np.isfinite(values).all():
                raise ContractMismatch("selection requires complete finite per-window VAL scores")
            choices.append((month_balanced_mean(values, months), row["lambda"], row["rho"]))
        score, lam, rho = min(choices)
        selected[kind] = {"lambda": lam, "rho": rho, "month_balanced_primary_es": score}
    return selected


def selected_pairs(selection):
    return sorted({(row["lambda"], row["rho"]) for row in selection["selected"].values()})


def source_contract(study_root, protocol_hash):
    protocol = load_protocol(study_root, protocol_hash)
    require_source_match(protocol)
    identities = {}
    keys = {"sha256", "split", "n_windows", "ids_sha256", "normalization_sha256"}
    for split, count in EXPECTED_WINDOWS.items():
        identity = protocol["arrays"][split]
        if (set(identity) != keys or identity["split"] != split
                or type(identity["n_windows"]) is not int or identity["n_windows"] != count):
            raise ContractMismatch("source protocol must pin the original complete-case array identities")
        for key in ("sha256", "ids_sha256", "normalization_sha256"):
            value = identity[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ContractMismatch("source protocol has an invalid array digest")
        identities[split] = identity
    if len({v["normalization_sha256"] for v in identities.values()}) != 1:
        raise ContractMismatch("all arrays must share the TRAIN normalization")
    return {
        "schema_version": 1, "study": "full_flank_linear_gaussian_extension", "scope": SCOPE,
        "source_protocol_sha256": protocol_hash, "arrays": identities,
        "producer_source_hashes": {"analysis/prigm_full_flank_gaussian.py": sha256_file(Path(__file__))},
        "core_source_hashes": protocol["inference_source_hashes"],
        "numpy_version": np.__version__,
        "model": {
            "conditioning": "216 standardized observed coordinates: bins [0,18) + [30,48), all six channels",
            "target": "60 standardized primary residuals about interpolation from bins 17 and 30",
            "objective": "||Y-XW-intercept||_F^2/N + lambda*||W||_F^2; unpenalized intercept",
            "training": "TRAIN-only centering and population residual covariance; equal weight per saved overlapping window",
            "lambdas": list(LAMBDAS), "rhos": list(RHOS), "jitter": GAUSSIAN_JITTER,
            "covariance": "(1-rho)*population_cov + rho*diag(population_cov) + jitter*I; fixed Cholesky",
            "representation": "joint Gaussian in first five coordinates only; derive physical v_v; preserve observed flanks",
            "chunk_windows": CHUNK_WINDOWS,
        },
        "noise": NOISE_CONTRACT,
        "selection": "independent raw/corrected minimum month-balanced VAL primary energy; exact ties lexicographic (lambda,rho)",
        "interval": "39 draws; inclusive 1-based order statistics 2 and 38; mean over 60 standardized primary coordinates",
        "recomputation_tolerance": {"rtol": 1e-10, "atol": 1e-12},
    }


def pinned_arrays(path, split, contract):
    z, ids, nrm, identity = load_arrays(Path(path), split, contract["arrays"][split]["sha256"])
    if identity != contract["arrays"][split]:
        raise ContractMismatch("array identity differs from original protocol")
    return z, ids, nrm


def fresh_json(path, value):
    if path.exists():
        raise FileExistsError(path)
    atomic_write_bytes(path, json_bytes(value))


def fit_select(output, contract, train_path, val_path):
    output.mkdir()  # exclusive claim; partial output is never reusable
    fresh_json(output / "EXTENSION.json", contract)
    extension_hash = sha256_file(output / "EXTENSION.json")
    train, _, nrm = pinned_arrays(train_path, "TRAIN", contract)
    fit = fit_from_statistics(sufficient_statistics(train, nrm))
    del train
    buffer = io.BytesIO()
    np.savez(buffer, **fit)
    atomic_write_bytes(output / "FIT.npz", buffer.getvalue())
    fit_hash = sha256_file(output / "FIT.npz")
    fresh_json(output / "FIT.json", {
        "schema_version": 1, "extension_protocol_sha256": extension_hash,
        "training_arrays": contract["arrays"]["TRAIN"], "fit_payload": "FIT.npz",
        "fit_payload_sha256": fit_hash, "n_mean_fits": 3, "n_covariance_configurations": 9,
    })
    val, ids, val_nrm = pinned_arrays(val_path, "VAL", contract)
    validation = {
        "schema_version": 1, "extension_protocol_sha256": extension_hash, "fit_payload_sha256": fit_hash,
        "arrays": contract["arrays"]["VAL"], **validation_scores(fit, val, ids, val_nrm),
    }
    fresh_json(output / "VALIDATION.json", validation)
    selected = select_grid(validation)
    selection = {
        "schema_version": 1, "extension_protocol_sha256": extension_hash, "fit_payload_sha256": fit_hash,
        "validation_sha256": sha256_file(output / "VALIDATION.json"), "selected": selected,
    }
    selection["selected_union"] = [candidate_name(*pair) for pair in selected_pairs(selection)]
    fresh_json(output / "SELECTION.json", selection)
    fresh_json(output / "FIT_SELECT.json", {
        "schema_version": 1, "phase": "FIT_SELECTED", "source_protocol_sha256": contract["source_protocol_sha256"],
        "files": {name: sha256_file(output / name) for name in FIT_FILES},
    })
    return sha256_file(output / "FIT_SELECT.json")


def require_close(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (actual.shape != expected.shape or actual.dtype != expected.dtype
            or not np.isfinite(actual).all()
            or not np.allclose(actual, expected, rtol=1e-10, atol=1e-12)):
        raise ContractMismatch(f"{label} failed recomputation")


def verify_fit_selection(output, contract, expected_receipt_hash, train_path, val_path):
    """No TEST path argument: every fit, score and selection is recomputed first."""
    receipt_path = output / "FIT_SELECT.json"
    if sha256_file(receipt_path) != expected_receipt_hash:
        raise ContractMismatch("fit/selection receipt hash changed")
    receipt = read_json(receipt_path)
    if (receipt.get("schema_version") != 1 or receipt.get("phase") != "FIT_SELECTED"
            or receipt.get("source_protocol_sha256") != contract["source_protocol_sha256"]
            or set(receipt.get("files", {})) != set(FIT_FILES)):
        raise ContractMismatch("invalid fit/selection receipt")
    for name in FIT_FILES:
        if sha256_file(output / name) != receipt["files"][name]:
            raise ContractMismatch(f"fit/selection payload hash changed: {name}")
    if read_json(output / "EXTENSION.json") != contract:
        raise ContractMismatch("extension producer, core, arrays or scientific contract changed")
    hashes = receipt["files"]
    fit_meta = read_json(output / "FIT.json")
    if fit_meta != {
        "schema_version": 1, "extension_protocol_sha256": hashes["EXTENSION.json"],
        "training_arrays": contract["arrays"]["TRAIN"], "fit_payload": "FIT.npz",
        "fit_payload_sha256": hashes["FIT.npz"], "n_mean_fits": 3, "n_covariance_configurations": 9,
    }:
        raise ContractMismatch("fit metadata changed")
    train, _, nrm = pinned_arrays(train_path, "TRAIN", contract)
    recomputed = fit_from_statistics(sufficient_statistics(train, nrm))
    del train
    with np.load(output / "FIT.npz", allow_pickle=False) as blob:
        if set(blob.files) != set(recomputed):
            raise ContractMismatch("fit parameter inventory changed")
        fit = {key: blob[key] for key in blob.files}
    for key in recomputed:
        require_close(fit[key], recomputed[key], f"TRAIN fit {key}")
    val, ids, nrm = pinned_arrays(val_path, "VAL", contract)
    validation = read_json(output / "VALIDATION.json")
    actual = validation_scores(fit, val, ids, nrm)
    if (validation.get("schema_version") != 1 or validation.get("arrays") != contract["arrays"]["VAL"]
            or validation.get("extension_protocol_sha256") != hashes["EXTENSION.json"]
            or validation.get("fit_payload_sha256") != hashes["FIT.npz"]
            or validation.get("ids") != ids or validation.get("months") != actual["months"]):
        raise ContractMismatch("VAL metadata changed")
    selected = select_grid(validation)
    by_pair = {(r["lambda"], r["rho"]): r for r in validation["candidates"]}
    for row in actual["candidates"]:
        saved = by_pair[row["lambda"], row["rho"]]
        for kind in ("raw", "corrected"):
            require_close(saved["scores"][kind], row["scores"][kind], f"VAL {kind}")
    if selected != select_grid(actual):
        raise ContractMismatch("recomputed VAL selection changed")
    selection = read_json(output / "SELECTION.json")
    expected = {
        "schema_version": 1, "extension_protocol_sha256": hashes["EXTENSION.json"],
        "fit_payload_sha256": hashes["FIT.npz"], "validation_sha256": hashes["VALIDATION.json"],
        "selected": selected,
        "selected_union": [candidate_name(*pair) for pair in selected_pairs({"selected": selected})],
    }
    if selection != expected:
        raise ContractMismatch("selection differs from verified complete VAL grid")
    return fit, selection, hashes


def window_metrics(samples, truth, nrm):
    """Only the requested endpoints plus support-correction behavior."""
    target = truth[HIDDEN_START:HIDDEN_END]
    x, y = hidden_primary_std(samples, nrm), hidden_primary_std(target[None], nrm)[0]
    u, v = nrm.transform(samples), nrm.transform(target)
    proxy = energy_density_proxy(samples)
    proxy_y = energy_density_proxy(target[None])
    gaps = support_gaps(samples)
    r = np.linalg.norm(samples[..., :3], axis=-1)
    m, vs = samples[..., 3], samples[..., 4]
    bands = {
        "m_minus_r": arithmetic_tolerance(m, r, np.sqrt(nrm.b_ref_sq)),
        "vs": arithmetic_tolerance(vs, 0, nrm.b_ref_sq),
        "vs_upper_minus_vs": arithmetic_tolerance(1.5 * m ** 2 + 2.5 * r ** 2, vs, nrm.b_ref_sq),
    }
    return {
        "primary_es": energy_score(x, y),
        "coverage90": float(coverage_inclusive(x, y, 2, 38).mean()),
        "width90": float(interval_width(x, 2, 38).mean()),
        "vector_energy": energy_score(u[..., :3].reshape(N_DRAWS, 36), v[..., :3].reshape(36)),
        "resolved_energy_proxy_crps": float(crps_univariate(proxy[:, None], proxy_y)[0]),
        "support_violation_fraction": float(np.logical_or.reduce([gaps[k] < 0 for k in bands]).mean()),
        "support_violation_beyond_rounding_fraction": float(np.logical_or.reduce([gaps[k] < -bands[k] for k in bands]).mean()),
    }


def test_scores(ensemble, z, ids, nrm):
    records = {"raw": [], "corrected": []}
    displacement = []
    for stored, truth in zip(ensemble, z):
        raw = decode_ensemble(stored, nrm)
        corrected = closure_correct(raw)
        for kind, samples in (("raw", raw), ("corrected", corrected)):
            records[kind].append(window_metrics(samples, truth, nrm))
        delta = hidden_primary_std(corrected, nrm) - hidden_primary_std(raw, nrm)
        displacement.append({"mean_primary_l2": float(np.linalg.norm(delta, axis=1).mean())})
    months = months_for(ids)
    return {
        **{kind: {"summary": summarize_diagnostics(rows, months), "per_window": rows}
           for kind, rows in records.items()},
        "correction_displacement": {"summary": summarize_diagnostics(displacement, months), "per_window": displacement},
    }


def ensemble_metadata(contract, hashes, selection, receipt_hash, pair):
    lam, rho = pair
    return {
        "protocol_sha256": hashes["EXTENSION.json"],  # NOT the frozen original protocol
        "source_protocol_sha256": contract["source_protocol_sha256"],
        "producer_source_hashes": contract["producer_source_hashes"], "core_source_hashes": contract["core_source_hashes"],
        "fit_select_sha256": receipt_hash, "selection_sha256": hashes["SELECTION.json"],
        "procedure": "gaussian", "control": {"family": "full_flank_linear_gaussian", "lambda": lam, "rho": rho,
                                               "jitter": GAUSSIAN_JITTER, "joint_gaussian_channels": 5},
        "n_generated": 6, "state_convention": "physical_control_standardized_float64",
        "noise_convention": NOISE_STREAM, "noise": NOISE_CONTRACT,
        "arrays": contract["arrays"]["TEST"], "training_arrays": contract["arrays"]["TRAIN"],
        "control_fit_sha256": hashes["FIT.npz"], "scope": SCOPE,
        "selected_for": [kind for kind, row in selection["selected"].items() if (row["lambda"], row["rho"]) == pair],
        "complete_case": True,
    }


def verify_selected_ensemble(root, expected_meta, z, ids, nrm):
    ensemble, actual_ids, meta = read_ensemble(root)
    expected = {**expected_meta, "schema_version": 2, "ids_sha256": id_hash(ids)}
    if actual_ids != ids or {k: v for k, v in meta.items() if k not in ("files", "generation")} != expected:
        raise ContractMismatch("selected ensemble metadata or complete-case ordering changed")
    observed = np.r_[0:HIDDEN_START, HIDDEN_END:N_BINS]
    context = nrm.transform(z).transpose(0, 2, 1)[:, None, :, observed]
    if not np.array_equal(ensemble[:, :, :, observed], np.broadcast_to(context, (len(ids), N_DRAWS, N_CHANNELS, len(observed)))):
        raise ContractMismatch("selected ensemble changed observed flanks")
    for stored in ensemble:
        physical = decode_ensemble(stored, nrm)
        require_close(physical, derive_vv(physical[..., :PRIMARY_CHANNELS]), "derived v_v")
    return ensemble


def test_stage(output, contract, train_path, val_path, test_path, receipt_hash, *, verify=False):
    test_root = output / "test"
    if not verify and test_root.exists():
        raise FileExistsError(test_root)
    fit, selection, hashes = verify_fit_selection(output, contract, receipt_hash, train_path, val_path)
    phase = {
        "schema_version": 1, "phase": "TEST_STARTED", "previous_test_exposure": True,
        "source_protocol_sha256": contract["source_protocol_sha256"],
        "extension_protocol_sha256": hashes["EXTENSION.json"], "fit_select_sha256": receipt_hash,
        "fit_payload_sha256": hashes["FIT.npz"], "selection_sha256": hashes["SELECTION.json"],
        "arrays": contract["arrays"]["TEST"], "selected_union": selection["selected_union"],
    }
    if verify:
        expected_files = {"TEST_STARTED.json", "SCORES.json", "COMPLETED.json", *selection["selected_union"]}
        if {p.name for p in test_root.iterdir()} != expected_files:
            raise ContractMismatch("TEST inventory must contain exactly the complete selected union")
        if read_json(test_root / "TEST_STARTED.json") != phase:
            raise ContractMismatch("TEST exposure receipt changed")
    else:
        test_root.mkdir()
        fresh_json(test_root / "TEST_STARTED.json", phase)
    # First TEST input access, after externally pinned fit/selection verification.
    z, ids, nrm = pinned_arrays(test_path, "TEST", contract)
    factors = candidate_factors(fit)
    results = {}
    ensemble_paths = []
    for pair in selected_pairs(selection):
        lam, rho = pair
        name = candidate_name(*pair)
        root = test_root / name
        i = LAMBDAS.index(lam)
        generated = np.stack([
            draw_window(truth, nrm, fit["weights"][i], fit["intercepts"][i], factors[pair], standard_noise("TEST", wid))
            for truth, wid in zip(z, ids)
        ])
        meta = ensemble_metadata(contract, hashes, selection, receipt_hash, pair)
        if not verify:
            publish_ensemble(root, generated, ids, meta)
        stored = verify_selected_ensemble(root, meta, z, ids, nrm)
        require_close(stored, generated, "selected TEST draws")
        commit_path = f"test/{name}/gen_00000000/COMMIT.json"
        ensemble_paths.append(commit_path)
        results[name] = {
            "lambda": lam, "rho": rho, "selected_for": meta["selected_for"],
            "ensemble": f"test/{name}", "ensemble_commit_sha256": sha256_file(output / commit_path),
            **test_scores(stored, z, ids, nrm),
        }
    scores = {
        "schema_version": 1, "scope": SCOPE, "fit_select_sha256": receipt_hash,
        "source_protocol_sha256": contract["source_protocol_sha256"],
        "extension_protocol_sha256": hashes["EXTENSION.json"], "selection_sha256": hashes["SELECTION.json"],
        "fit_payload_sha256": hashes["FIT.npz"], "arrays": contract["arrays"]["TEST"],
        "ids": ids, "months": months_for(ids).tolist(), "n": len(ids), "n_draws": N_DRAWS,
        "complete_case": True, "selected": selection["selected"], "interval": contract["interval"],
        "candidates": results,
    }
    if verify:
        if read_json(test_root / "SCORES.json") != scores:
            raise ContractMismatch("TEST scores failed recomputation")
    else:
        fresh_json(test_root / "SCORES.json", scores)
    completed = {
        "schema_version": 1, "phase": "TEST_COMPLETED", "fit_select_sha256": receipt_hash,
        "n_windows": len(ids), "n_draws": N_DRAWS, "selected_union": selection["selected_union"],
        "files": {name: sha256_file(output / name) for name in
                  ["test/TEST_STARTED.json", "test/SCORES.json", *ensemble_paths]},
    }
    if verify:
        if read_json(test_root / "COMPLETED.json") != completed:
            raise ContractMismatch("TEST completion hashes changed")
    else:
        fresh_json(test_root / "COMPLETED.json", completed)
    return completed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fit-select", "test", "verify"), required=True)
    parser.add_argument("--train-arrays", required=True)
    parser.add_argument("--val-arrays", required=True)
    parser.add_argument("--test-arrays")
    parser.add_argument("--source-study-root", required=True)
    parser.add_argument("--source-protocol-sha256", required=True)
    parser.add_argument("--output-root", required=True, help="fresh for fit-select; existing frozen root for test/verify")
    parser.add_argument("--fit-select-sha256", help="external pin printed by fit-select; required before TEST access")
    args = parser.parse_args(argv)
    output = Path(args.output_root)
    if args.stage == "fit-select":
        if output.exists():
            raise FileExistsError(output)
        if args.fit_select_sha256:
            parser.error("fit-select does not accept an existing receipt")
    elif not args.test_arrays or not args.fit_select_sha256:
        parser.error("test/verify require --test-arrays and --fit-select-sha256")
    contract = source_contract(Path(args.source_study_root), args.source_protocol_sha256)
    if args.stage == "fit-select":
        digest = fit_select(output, contract, args.train_arrays, args.val_arrays)
        print(json.dumps({"phase": "FIT_SELECTED", "fit_select_sha256": digest}))
    else:
        result = test_stage(output, contract, args.train_arrays, args.val_arrays,
                            args.test_arrays, args.fit_select_sha256, verify=args.stage == "verify")
        print(json.dumps({"phase": "VERIFIED" if args.stage == "verify" else result["phase"],
                          "n_windows": result["n_windows"], "n_draws": result["n_draws"],
                          "selected_union": result["selected_union"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
