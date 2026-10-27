"""Describe prediction errors, intervals, and constraints for one saved ensemble.

The command keeps draws from each trained model separate. It reports energy
scores and CRPS, inclusive intervals from the 39 ordered draws, the magnetic
observation-identity residual, and violations of the three support
inequalities. Rounding bands qualify the inequality counts; the equality
residual is reported separately.

Boundary measurements compare generated mean vectors with the adjacent
observed bins on both sides of the hidden hour. Results include individual
time positions and channels, plus averages with equal weight per represented
month. --correct applies the support clipping map before these calculations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prigm.artifacts import json_bytes, load_arrays, read_ensemble, require_array_match, require_procedure_match, require_source_match
from prigm.barrier import load_protocol, require_test_started
from prigm.config import HIDDEN_END, HIDDEN_START
from prigm.observation import closure_correct, residual_obs, support_gaps
from prigm.precision import arithmetic_tolerance, representation_bands
from prigm.recovery import ContractMismatch, atomic_write_bytes, sha256_file
from prigm.run_score import decode_ensemble, hidden_primary_std
from prigm.scoring import (
    coverage_inclusive, crps_univariate, energy_density_proxy, energy_score,
    interval_width, primary_variogram_pairs, variogram_score,
)


def window_diagnostics(samples, truth_window, nrm, stored_dtype=np.float32):
    """Physical samples (39,12,6), full physical truth (48,6). No resampling."""
    z_s = np.asarray(samples, dtype=np.float64)
    y = np.asarray(truth_window[HIDDEN_START:HIDDEN_END], dtype=np.float64)
    xs5 = hidden_primary_std(z_s, nrm)
    ys5 = hidden_primary_std(y[None], nrm)[0]
    u_s = (z_s - nrm.mean) / nrm.scale
    u_y = (y - nrm.mean) / nrm.scale
    proxy_s = energy_density_proxy(z_s)
    proxy_y = float(energy_density_proxy(y[None])[0])
    gaps = support_gaps(z_s)
    if np.dtype(stored_dtype) == np.dtype("float32"):
        bands = representation_bands(z_s, u_s, nrm.scale, np.sqrt(nrm.b_ref_sq), nrm.b_ref_sq)
    else:
        r = np.linalg.norm(z_s[..., :3], axis=-1)
        m, vs = z_s[..., 3], z_s[..., 4]
        bands = {
            "m_minus_r": arithmetic_tolerance(m, r, np.sqrt(nrm.b_ref_sq)),
            "vs": arithmetic_tolerance(vs, 0, nrm.b_ref_sq),
            "vs_upper_minus_vs": arithmetic_tolerance(1.5 * m ** 2 + 2.5 * r ** 2, vs, nrm.b_ref_sq),
        }
    cov90 = coverage_inclusive(xs5, ys5, 2, 38)
    width90 = interval_width(xs5, 2, 38)
    crps = crps_univariate(xs5, ys5)
    left = truth_window[HIDDEN_START - 1, :3]
    right = truth_window[HIDDEN_END, :3]
    seam_l = z_s[:, 0, :3] - left
    seam_r = right - z_s[:, -1, :3]
    internal = np.diff(z_s[..., :3], axis=1)
    truth_seam_l = y[0, :3] - left
    truth_seam_r = right - y[-1, :3]
    bias = u_s[..., :5].mean(axis=0) - u_y[..., :5]
    strict = np.logical_or.reduce([gaps[k] < 0 for k in bands])
    beyond_rounding = np.logical_or.reduce([gaps[k] < -bands[k] for k in bands])
    return {
        "es5": energy_score(xs5, ys5),
        "es6": energy_score(u_s.reshape(len(z_s), 72), u_y.reshape(72)),
        "es_vec": energy_score(u_s[..., :3].reshape(len(z_s), 36), u_y[..., :3].reshape(36)),
        "crps5_mean": float(crps.mean()),
        "cov90_mean": float(cov90.mean()),
        "cov50_mean": float(coverage_inclusive(xs5, ys5, 10, 30).mean()),
        "width90_mean": float(width90.mean()),
        "width50_mean": float(interval_width(xs5, 10, 30).mean()),
        "proxy_crps": float(crps_univariate(proxy_s[:, None], np.array([proxy_y]))[0]),
        "proxy_cov90": float(coverage_inclusive(proxy_s[:, None], np.array([proxy_y]), 2, 38)[0]),
        "proxy_width90": float(interval_width(proxy_s[:, None], 2, 38)[0]),
        "r_obs_mean": float(np.abs(residual_obs(z_s)).mean()),
        "frac_m_lt_r": float((gaps["m_minus_r"] < 0).mean()),
        "frac_vs_lt0": float((gaps["vs"] < 0).mean()),
        "frac_vs_gt_hi": float((gaps["vs_upper_minus_vs"] < 0).mean()),
        "frac_support_inequality_violation": float(strict.mean()),
        # This legacy key counts violations of the three support inequalities
        # beyond rounding bands. It excludes the observation-identity equality;
        # r_obs_mean reports the mean absolute equality residual separately.
        "frac_support_violation_beyond_rounding": float(beyond_rounding.mean()),
        "variogram": variogram_score(xs5, ys5, primary_variogram_pairs()),
        "seam_l_mean": float(np.linalg.norm(seam_l, axis=-1).mean()),
        "seam_r_mean": float(np.linalg.norm(seam_r, axis=-1).mean()),
        "internal_mean": float(np.linalg.norm(internal, axis=-1).mean()),
        "truth_seam_l": float(np.linalg.norm(truth_seam_l)),
        "truth_seam_r": float(np.linalg.norm(truth_seam_r)),
        "truth_internal": float(np.linalg.norm(np.diff(y[:, :3], axis=0), axis=-1).mean()),
        "seam_l_rmse": float(np.sqrt(np.mean(np.sum((seam_l - truth_seam_l) ** 2, axis=-1)))),
        "seam_r_rmse": float(np.sqrt(np.mean(np.sum((seam_r - truth_seam_r) ** 2, axis=-1)))),
        "neg_proxy": float((proxy_s < 0).mean()),
        "crps_by_position_channel": crps.reshape(12, 5).tolist(),
        "cov90_by_position_channel": cov90.reshape(12, 5).astype(float).tolist(),
        "width90_by_position_channel": width90.reshape(12, 5).tolist(),
        "bias_by_position_channel": bias.tolist(),
    }


def summarize_diagnostics(records, months):
    """Equal month weights, retaining per-position/channel axes."""
    summary = {}
    months = np.asarray(months)
    for key in records[0]:
        values = np.asarray([row[key] for row in records], dtype=np.float64)
        mb = np.mean([values[months == m].mean(axis=0) for m in np.unique(months)], axis=0)
        summary[key] = {"month_balanced": mb.tolist(), "mean": values.mean(axis=0).tolist()}
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ensemble", required=True)
    p.add_argument("--arrays", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--correct", action="store_true")
    p.add_argument("--split", required=True, choices=("VAL", "TEST", "TOY"))
    p.add_argument("--study-root", required=True)
    p.add_argument("--protocol-sha256", required=True)
    args = p.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    protocol = load_protocol(Path(args.study_root), args.protocol_sha256)
    require_source_match(protocol)
    selection = require_test_started(Path(args.study_root), args.split, args.protocol_sha256)
    ensemble, ids, meta = read_ensemble(Path(args.ensemble))
    if meta["protocol_sha256"] != args.protocol_sha256:
        raise ContractMismatch("ensemble belongs to another protocol")
    require_procedure_match(meta, protocol, selection, split=args.split)
    z, target_ids, nrm, identity = load_arrays(
        Path(args.arrays), args.split, protocol["arrays"][args.split]["sha256"]
    )
    require_array_match(meta, identity)
    if ids != target_ids:
        raise ContractMismatch("diagnostics require complete aligned ensembles")
    months = (np.zeros(len(ids), dtype=np.int64) if args.split == "TOY" else
              np.array([int(w[:4]) * 100 + int(w[5:7]) for w in ids], dtype=np.int64))
    records = []
    for draws, truth in zip(ensemble, z):
        samples = decode_ensemble(draws, nrm)
        if args.correct:
            samples = closure_correct(samples)
        records.append(window_diagnostics(samples, truth, nrm, ensemble.dtype))
    result = {
        "schema_version": 2, "n": len(ids), "ids": ids, "months": months.tolist(),
        "correct": bool(args.correct), "protocol_sha256": args.protocol_sha256,
        "procedure": meta.get("procedure", "diffusion"),
        "checkpoint_sha256": meta.get("checkpoint_sha256"),
        "control_fit_sha256": meta.get("control_fit_sha256"), "arrays": identity,
        "ensemble_commit_sha256": sha256_file(Path(args.ensemble) / "gen_00000000/COMMIT.json"),
        "interval_protocol": "seedwise_39_draws_inclusive_order_statistics_2_38_and_10_30",
        **summarize_diagnostics(records, months), "per_window": records,
    }
    atomic_write_bytes(out, json_bytes(result))
    print(json.dumps({k: result[k] for k in ("n", "es5", "cov90_mean", "r_obs_mean")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
