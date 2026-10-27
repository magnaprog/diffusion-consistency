"""Describe the completed PriGM follow-up using its saved scores and metric records.

Require the complete primary and secondary records, validate their identities,
and recompute their averages. Each neural model contributes its own 39-draw
statistics; the three fixed training seeds receive equal weight. Each
non-neural comparison method contributes one saved realization.

The exploratory groups describe calendar periods, data sources and properties
of the withheld observations. Their comparisons resample represented years
while keeping models and predictions fixed. They describe an already examined
TEST population and do not account for retraining or checkpoint-selection
uncertainty, or establish causal explanations.

Run with --study-root, --protocol-sha256 and --output (a fresh JSON path).
Input records are read without modification; predictions are not generated.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prigm import run_bootstrap as bootstrap
from prigm.artifacts import id_hash, json_bytes, load_arrays, require_source_match
from prigm.config import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, FINAL_SEEDS, YEARS
from prigm.recovery import ContractMismatch, sha256_bytes, sha256_file

INTERVAL_PROTOCOL = "seedwise_39_draws_inclusive_order_statistics_2_38_and_10_30"
SCALARS = (
    "es5", "es6", "es_vec", "crps5_mean", "cov90_mean", "cov50_mean",
    "width90_mean", "width50_mean", "proxy_crps", "proxy_cov90", "proxy_width90",
    "r_obs_mean", "frac_m_lt_r", "frac_vs_lt0", "frac_vs_gt_hi",
    "frac_support_inequality_violation", "frac_support_violation_beyond_rounding",
    "variogram", "seam_l_mean", "seam_r_mean", "internal_mean", "truth_seam_l",
    "truth_seam_r", "truth_internal", "seam_l_rmse", "seam_r_rmse", "neg_proxy",
)
POSITION_CHANNEL = (
    "crps_by_position_channel", "cov90_by_position_channel",
    "width90_by_position_channel", "bias_by_position_channel",
)
SECONDARY_IDENTITY = (
    "schema_version", "n", "ids", "months", "protocol_sha256", "procedure",
    "checkpoint_sha256", "control_fit_sha256", "arrays", "ensemble_commit_sha256",
)
ARRAY_FIELDS = ("sha256", "split", "n_windows", "ids_sha256", "normalization_sha256")
YEAR_GROUPS = ((2005, 2009), (2010, 2015), (2016, 2019), (2020, 2024))
# HRO modified format, footnote D; prigm/data.py documents 50 and 60 too.
SOURCE_LABELS = {50: "IMF source 50", 52: "Wind", 60: "IMF source 60", 71: "ACE"}
PAIRS = (("O-S_raw", "D_raw"), ("O-S_raw", "G5_raw"),
         ("O-S_raw", "gaussian_raw"), ("D_raw", "interpolation_raw"))


def _finite(values):
    try:
        result = np.asarray(values)
        if result.dtype.kind not in "iuf" or not np.isfinite(result).all():
            raise ValueError("not finite numeric values")
        return result.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ContractMismatch("expected finite numeric metric values") from exc


def reductions(values, months):
    """Average arrays whose first axis indexes windows.

    Return both the mean over all windows and the mean of the represented
    months' means. In the latter, each month receives equal weight regardless
    of its window count. Any remaining axes, such as position and channel,
    are retained.
    """
    values = _finite(values)
    months = np.asarray(months)
    if months.ndim != 1 or not months.size or values.ndim < 1 or len(values) != len(months):
        raise ContractMismatch("metric window/month shape mismatch or empty population")
    return {
        "month_balanced": np.stack([values[months == m].mean(axis=0)
                                     for m in np.unique(months)]).mean(axis=0),
        "mean": values.mean(axis=0),
    }


def _agree(reported, actual, label):
    reported, actual = _finite(reported), _finite(actual)
    if reported.shape != actual.shape or not np.allclose(reported, actual, rtol=1e-12, atol=0):
        raise ContractMismatch(f"{label} does not match recomputed values")


def validate_secondary(row, primary, scores, months):
    """Bind to loader-verified primary scores/ensemble and validate metric records.

    Recompute each mean and month-balanced summary from stored per-window
    metrics, rather than recreating those metrics from ensemble samples.
    """
    if (any(key not in row or key not in primary or row[key] != primary[key]
            for key in SECONDARY_IDENTITY)
            or type(row.get("schema_version")) is not int or row["schema_version"] != 2
            or type(row.get("n")) is not int or row["n"] != len(months)
            or type(row.get("correct")) is not bool or row["correct"] is not primary["corrected"]
            or row.get("interval_protocol") != INTERVAL_PROTOCOL
            or row["months"] != np.asarray(months).tolist()):
        raise ContractMismatch("secondary identity, ordered IDs, correction or interval protocol mismatch")
    records = row.get("per_window")
    metadata = {*SECONDARY_IDENTITY, "correct", "interval_protocol", "per_window"}
    metrics = set(row) - metadata
    if (not isinstance(records, list) or len(records) != len(months)
            or not set(SCALARS + POSITION_CHANNEL).issubset(metrics)
            or any(not isinstance(r, dict) or set(r) != metrics for r in records)):
        raise ContractMismatch("secondary per-window metric inventory/count mismatch")
    result = {}
    for key in sorted(metrics):
        values = _finite([r[key] for r in records])
        expected_shape = (12, 5) if key in POSITION_CHANNEL else () if key in SCALARS else None
        if expected_shape is not None and values.shape[1:] != expected_shape:
            raise ContractMismatch(f"secondary {key} shape mismatch")
        if not isinstance(row[key], dict) or set(row[key]) != {"month_balanced", "mean"}:
            raise ContractMismatch(f"secondary {key} summary fields mismatch")
        actual = reductions(values, months)
        for weighting, value in actual.items():
            _agree(row[key][weighting], value, f"secondary {key} {weighting}")
        if key == "es5":
            _agree(values, scores, "secondary es5 versus primary scores")
            _agree(values, primary["scores"], "secondary es5 versus primary file")
        result[key] = {k: v.tolist() for k, v in actual.items()}
    return result


def validate_matrices(data, months, ids):
    """The loader owns the 480-window protocol; retain its 3-model/1-control axes."""
    expected = {f"{arm}_{kind}": 3 for arm in bootstrap.ARMS
                for kind in (*bootstrap.KINDS, "corrected_at_raw_checkpoint")}
    expected.update({f"{name}_{kind}": 1 for name in bootstrap.CONTROLS for kind in bootstrap.KINDS})
    months = np.asarray(months)
    if (set(data) != set(expected) or months.shape != (len(ids),) or not len(ids)
            or len(set(ids)) != len(ids)):
        raise ContractMismatch("incomplete methods or window population")
    for label, n_models in expected.items():
        if _finite(data[label]).shape != (n_models, len(ids)):
            raise ContractMismatch(f"{label} requires {n_models} fixed model rows and all windows")


def source_codes(ids):
    codes = []
    for wid in ids:
        match = re.search(r"-s(50|52|60|71)\Z", wid)
        if match is None or int(match[1]) not in SOURCE_LABELS:
            raise ContractMismatch("window ID lacks a documented terminal IMF source")
        codes.append(int(match[1]))
    return np.asarray(codes, dtype=np.int64)


def tertiles(values):
    values = _finite(values)
    if values.ndim != 1 or not len(values):
        raise ContractMismatch("tertiles require a nonempty target vector")
    edges = np.quantile(values, [1 / 3, 2 / 3], method="linear")
    bins = np.searchsorted(edges, values, side="right")
    _, repeats = np.unique(values, return_counts=True)
    return bins, {
        "edges": edges.tolist(), "quantiles": [1 / 3, 2 / 3], "quantile_method": "linear",
        "assignment": "searchsorted_side_right", "edge_tie_counts": [int((values == e).sum()) for e in edges],
        "windows_in_tied_values": int(repeats[repeats > 1].sum()),
        "counts": [int((bins == i).sum()) for i in range(3)],
        "population": "full frozen test targets; counts are not forced equal",
    }


def _counts(values):
    keys, counts = np.unique(values, return_counts=True)
    return {str(k): int(n) for k, n in zip(keys, counts)}


def paired_year_summary(data, months, *, seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES):
    """Compare fixed models using the same resampled years on both sides.

    Within the supplied subset, average scores by represented month. Resample
    its represented calendar years with replacement and use the same year
    occurrences for every model and comparison. The three neural training
    seeds stay fixed; a non-neural comparison contributes one shared row.

    Return mean and seedwise differences plus separate percentile intervals
    for each comparison. With only one represented year, report the interval
    as unavailable. This resampling does not measure uncertainty from training,
    new prediction draws, or checkpoint selection.
    The intervals are not adjusted for multiple comparisons.
    """
    if type(replicates) is not int or replicates < 1:
        raise ValueError("bootstrap replicates must be a positive integer")
    months = np.asarray(months)
    per_month = {k: bootstrap.monthly_means(v, months) for k, v in data.items()}
    years = np.unique(months // 100)
    weights = None
    if len(years) > 1:
        rng = np.random.Generator(np.random.PCG64(seed))
        draws = years[rng.integers(len(years), size=(replicates, len(years)))]
        weights = bootstrap.month_occurrence_weights(months, draws)
    result = {}
    for method, baseline in PAIRS:
        left, right = per_month[method], per_month[baseline]
        if left.shape[0] != 3 or right.shape[0] != (1 if baseline.endswith(("gaussian_raw", "interpolation_raw")) else 3):
            raise ContractMismatch("paired comparison needs three fixed seeds or one shared control")
        differences = left.mean(axis=1) - right.mean(axis=1)
        boot = None if weights is None else weights @ (left.mean(axis=0) - right.mean(axis=0))
        if boot is not None and not np.isfinite(boot).all():
            raise ContractMismatch("nonfinite bootstrap replicate; cannot drop it")
        result[f"{method}_minus_{baseline}"] = {
            "method": method, "baseline": baseline, "mean_diff": float(differences.mean()),
            "seed_diffs": dict(zip(map(str, FINAL_SEEDS), differences.tolist())),
            "method_fixed_models": len(left), "baseline_realizations": len(right),
            "year_95": None if boot is None else np.quantile(boot, [.025, .975]).tolist(),
            "ci_status": "unavailable_one_represented_year" if boot is None else "available",
            "replicates_used": 0 if boot is None else replicates,
        }
    return result


def energy_strata(data, months, ids, z, *, seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES):
    validate_matrices(data, months, ids)
    months, z = np.asarray(months), _finite(z)
    if z.shape != (len(ids), 48, 6):
        raise ContractMismatch("target strata require aligned physical windows")
    years, sources = months // 100, source_codes(ids)
    if (not np.isin(years, YEARS).all()
            or months.tolist() != [int(w[:4]) * 100 + int(w[5:7]) for w in ids]):
        raise ContractMismatch("stratum calendar differs from ordered IDs")
    target_values = {
        "hidden_mean_m": z[:, 18:30, 3].mean(axis=1),
        "hidden_mean_vector_step": np.linalg.norm(np.diff(z[:, 18:30, :3], axis=1), axis=2).mean(axis=1),
    }
    masks = {"full_test": np.ones(len(ids), dtype=bool)}
    masks.update({f"year_{y}": years == y for y in np.unique(years)})
    masks.update({f"years_{lo}_{hi}": (years >= lo) & (years <= hi) for lo, hi in YEAR_GROUPS})
    masks.update({f"imf_source_{s}": sources == s for s in np.unique(sources)})
    definitions = {}
    for name, values in target_values.items():
        bins, definitions[name] = tertiles(values)
        definitions[name]["values_in_test_id_order"] = values.tolist()
        masks.update({f"{name}_tertile_{i + 1}": bins == i for i in range(3)})
    strata = {}
    for name, mask in masks.items():
        subset_months = months[mask]
        subset_ids = [w for w, keep in zip(ids, mask) if keep]
        row = {
            "n_windows": len(subset_ids), "ids": subset_ids, "ids_sha256": id_hash(subset_ids),
            "months": np.unique(subset_months).tolist(), "month_counts": _counts(subset_months),
            "year_counts": _counts(years[mask]), "source_counts": _counts(sources[mask]),
            "year_source_counts": {str(y): _counts(sources[mask & (years == y)]) for y in np.unique(years[mask])},
            "status": "represented" if subset_ids else "empty_stratum", "methods": {}, "comparisons": {},
        }
        if subset_ids:
            subset = {k: np.asarray(v)[:, mask] for k, v in data.items()}
            for label, values in subset.items():
                points = bootstrap.monthly_means(values, subset_months).mean(axis=1)
                window_points = values.mean(axis=1)
                keys = ["shared"] if len(points) == 1 else list(map(str, FINAL_SEEDS))
                row["methods"][label] = {
                    "month_balanced": float(points.mean()), "mean": float(window_points.mean()),
                    "seedwise": {k: {"month_balanced": float(p), "mean": float(w)}
                                 for k, p, w in zip(keys, points, window_points)},
                }
            row["comparisons"] = paired_year_summary(subset, subset_months, seed=seed, replicates=replicates)
        strata[name] = row
    return {"tertiles": definitions, "strata": strata}


def _relative(path, study):
    path = Path(path)
    if not path.is_absolute():
        path = study / path
    try:
        return path.resolve().relative_to(study).as_posix()
    except ValueError as exc:
        raise ContractMismatch("input path escapes study root") from exc


def _read_json(path, study, hashes):
    relative = _relative(path, study)
    payload = (study / relative).read_bytes()
    digest = sha256_bytes(payload)
    if relative in hashes and hashes[relative] != digest:
        raise ContractMismatch(f"input changed after primary gate: {relative}")
    hashes[relative] = digest
    return json.loads(payload, object_pairs_hook=bootstrap._unique_object)


def _array_identity(row):
    return {key: row[key] for key in ARRAY_FIELDS}


def summarize_study(study_root, protocol_sha256):
    """Return a verified report in memory. Rehash all bound inputs before return."""
    study = Path(study_root).resolve()
    source_hash = sha256_file(Path(__file__))
    data, months, provenance = bootstrap.load_study(study, protocol_sha256)
    ids = provenance["ids"]
    validate_matrices(data, months, ids)
    hashes = {_relative(p, study): h for p, h in provenance["input_sha256"].items()}
    protocol = _read_json(study / "PROTOCOL.json", study, hashes)
    if hashes["PROTOCOL.json"] != protocol_sha256:
        raise ContractMismatch("protocol hash differs from requested protocol")
    inventory = study / "artifacts/inventory"
    if _relative(protocol["array_root"], study) != "artifacts/inventory":
        raise ContractMismatch("follow-up arrays must be study/artifacts/inventory")
    z, target_ids, _, identity = load_arrays(inventory / "test.npz", "TEST", provenance["arrays"]["sha256"])
    if target_ids != ids or identity != provenance["arrays"]:
        raise ContractMismatch("stratum targets differ from primary array identity or ordered IDs")
    inventory_summary = _read_json(inventory / "summary.json", study, hashes)
    raw_hashes = inventory_summary["hashes"]
    inventory_counts = {k: inventory_summary[k] for k in
                        ("train_starts", "train_windows_saved", "train_bin_union", "val_selected", "test_selected")}
    if (set(raw_hashes) != set(map(str, YEARS))
            or any(not isinstance(h, str) or re.fullmatch(r"[0-9a-f]{64}", h) is None for h in raw_hashes.values())
            or any(type(n) is not int or n < 0 for n in inventory_counts.values())
            or inventory_counts["test_selected"] != len(ids)
            or inventory_counts["train_windows_saved"] != protocol["arrays"]["TRAIN"]["n_windows"]
            or inventory_counts["val_selected"] != protocol["arrays"]["VAL"]["n_windows"]
            or protocol["training_data_hashes"]["arrays_sha256"] != protocol["arrays"]["TRAIN"]["sha256"]):
        raise ContractMismatch("inventory raw hashes or population counts mismatch")
    methods = {}
    for label, values in data.items():
        meta = provenance["methods"][label]
        keys = ["shared"] if len(values) == 1 else list(map(str, FINAL_SEEDS))
        if set(meta["score_files"]) != set(keys):
            raise ContractMismatch("method score paths differ from fixed seed rows")
        seedwise, files, checkpoints = {}, {}, {}
        for index, key in enumerate(keys):
            score_path = study / _relative(meta["score_files"][key], study)
            if _relative(score_path, study) not in hashes:
                raise ContractMismatch("primary score is not bound by load_study")
            primary = _read_json(score_path, study, hashes)
            secondary_path = score_path.with_name(f"secondaries-{meta['scoring_kind']}.json")
            secondary = _read_json(secondary_path, study, hashes)
            seedwise[key] = validate_secondary(secondary, primary, values[index], months)
            files[key] = {"primary": _relative(score_path, study), "secondaries": _relative(secondary_path, study),
                          "ensemble_commit_sha256": primary["ensemble_commit_sha256"],
                          "control_fit_sha256": primary["control_fit_sha256"]}
            checkpoints[key] = primary["checkpoint_sha256"]
        if any(set(s) != set(seedwise[keys[0]]) for s in seedwise.values()):
            raise ContractMismatch("seedwise diagnostic metric inventories differ")
        metrics = {metric: {weight: np.mean([seedwise[k][metric][weight] for k in keys], axis=0).tolist()
                            for weight in ("month_balanced", "mean")} for metric in seedwise[keys[0]]}
        methods[label] = {
            "procedure": meta["procedure"], "scoring_kind": meta["scoring_kind"],
            "selection_kind": meta.get("selection_kind"), "generations": meta.get("generations", {}),
            "fixed_model_seeds": list(FINAL_SEEDS) if len(keys) == 3 else [],
            "n_realizations": len(keys), "checkpoint_sha256": checkpoints,
            "files": files, "metrics": metrics, "seedwise": seedwise,
        }
    result = {
        "schema_version": 1, "analysis_source_sha256": source_hash,
        "protocol_sha256": protocol_sha256, "selection_sha256": provenance["selection_sha256"],
        "test_started_sha256": provenance["test_started_sha256"], "arrays": _array_identity(identity),
        "ids": ids, "months": np.asarray(months).tolist(), "input_sha256": hashes,
        "input_manifest": {
            "array_identities": {k: _array_identity(protocol["arrays"][k]) for k in ("TRAIN", "VAL", "TEST")},
            "inventory_counts": inventory_counts,
            "recorded_raw_sha256_by_year": raw_hashes,
            "raw_hash_scope": "Copied from artifacts/inventory/summary.json; raw OMNI files are not reread or independently reverified here.",
            "training_arrays_sha256": protocol["training_data_hashes"]["arrays_sha256"],
        },
        "methods": methods, "interval_protocol": INTERVAL_PROTOCOL,
        "weighting": {"month_balanced": "Equal represented-month means per fixed seed, then equal seed mean.",
                      "mean": "Equal window means per fixed seed, then equal seed mean; not month balanced."},
        "diagnostic_scope": "Each diffusion diagnostic uses its own 39-draw ensemble, then equal means over three fixed seeds. Never pool 117 draws. Each control is one realization, broadcast only for paired contrasts.",
        "support_scope": "Support violation fractions check inequalities only, not full realizability: they do not test R=0. r_obs_mean separately reports mean absolute R.",
        "position_channel_axes": {"positions": list(range(18, 30)), "channels": ["mu_x", "mu_y", "mu_z", "m", "v_s"]},
        "strata_scope": "Descriptive, target-defined exploratory diagnostics on frozen primary energy scores. No redraw, refit, changed target, selection or forecast. Calendar groups are numeric, not pure solar maximum/minimum regimes; source and year mixes are descriptive, not causal.",
        "strata_definitions": {"year_groups": [list(g) for g in YEAR_GROUPS],
                               "sources": {str(k): v for k, v in SOURCE_LABELS.items()},
                               "hidden_mean_m": "z[:,18:30,3].mean(axis=1), physical nT",
                               "hidden_mean_vector_step": "norm(diff(z[:,18:30,:3],axis=1),axis=2).mean(axis=1), physical nT per five-minute step"},
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "generator": "PCG64",
                      "scheme": "Sample represented calendar years with replacement, n_draws=n_represented_years. Every year occurrence retains its distinct represented months. Common draws for all methods within each stratum; RNG restarts at the configured seed for each stratum.",
                      "scope": "Marginal paired test-year percentile 95% intervals, not multiplicity-adjusted, training-seed, refitting or selection uncertainty. Fixed models and one shared realization per control. One-year intervals unavailable. No two-year stratum intervals or p-values. Historically exposed test, not blind confirmation."},
        **energy_strata(data, months, ids, z, seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES),
    }
    require_source_match(protocol)
    for relative, digest in hashes.items():
        if _relative(study / relative, study) != relative or sha256_file(study / relative) != digest:
            raise ContractMismatch(f"input changed during summary: {relative}")
    if sha256_file(Path(__file__)) != source_hash:
        raise ContractMismatch("summary source changed during analysis")
    # Also refuses nonfinite values produced by any final reduction.
    json_bytes(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", required=True, help="fresh JSON output path")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("summary output must be fresh")
    result = summarize_study(args.study_root, args.protocol_sha256)
    with output.open("xb") as handle:
        handle.write(json_bytes(result))
    print(json.dumps({"n_methods": len(result["methods"]), "n_strata": len(result["strata"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
