"""Training-derived PriGM sensitivities on existing, protocol-bound primary scores.

Implements docs/prigm_plan_2026-09-05.md, paragraphs 467-469, after inspection
of the original results. This is not a new prospective registration. The original
summary, gradient diagnostics and outputs are neither modified nor regenerated.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import prigm_followup_summary as summary
from prigm import run_bootstrap as bootstrap
from prigm.artifacts import id_hash, json_bytes, load_arrays, require_source_match
from prigm.config import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, FINAL_SEEDS, TEST_MONTHS, YEARS
from prigm.recovery import ContractMismatch, fsync_dir, sha256_bytes, sha256_file

FEATURES = ("target_energy", "target_cancellation", "target_absolute_cancellation",
            "context_energy", "context_cancellation")
CUTOFF_QUANTILES = {
    "target_energy": [.25, .5, .75], "context_energy": [.25, .5, .75],
    "target_absolute_cancellation": [.75], "context_cancellation": [.75],
}
SCOPE = {
    "timing": "Implemented after inspection of the original results, using the original documented training-derived rules. Not a new prospective registration or restored blind confirmation.",
    "rules": "docs/prigm_plan_2026-09-05.md, original paragraphs at lines 467-469.",
    "target": "Outcome-selected descriptive associations. Conditioning on realized targets can reverse a proper-score ranking even for a correct forecast. Subgroup rankings do not establish conditional-distribution superiority or calibration.",
    "context": "Groups use only the actual 36 observed conditioning bins, not hidden targets. Still exploratory secondary analyses, not causal interventions or confirmatory superiority tests.",
    "fixed": "Existing 18 primary procedures, fixed checkpoints and ensembles only. Three individual diffusion seeds, one shared realization per control; never pool 117 draws. No refit, redraw, selection, gradients or change to primary case selection.",
    "claims": "The full-test month-balanced primary endpoint is unchanged. No all-Gaussian or neural universality claims; source and calendar mixes are descriptive, not causal.",
    "validation": "Loader checks score, checkpoint-selection and ensemble identities and ensemble payload integrity. Scores are aggregated, not recomputed from draws; checkpoint training and selection are not rerun. Original completion-linked outputs are hash-checked, not recalculated. Raw OMNI and the normalizer fit are not independently reconstructed. Full training feature arrays are not published.",
}


def physical_features(z, b_ref_sq):
    """Per-window means of physical bin formulas, with no clipping or refitting."""
    try:
        z = np.asarray(z)
        if (z.dtype.kind not in "iuf" or z.ndim != 3 or z.shape[1:] != (48, 6)
                or not len(z) or not np.isfinite(z).all()
                or not np.isscalar(b_ref_sq) or not np.isfinite(b_ref_sq) or b_ref_sq <= 0):
            raise ValueError("invalid physical windows or B_ref_sq")
        z = np.asarray(z, dtype=np.float64)
        result = {}
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for population, bins in (("target", z[:, 18:30]),
                                     ("context", np.concatenate((z[:, :18], z[:, 30:48]), axis=1))):
                m2 = bins[:, :, 3] ** 2
                cancellation = m2 - np.sum(bins[:, :, :3] ** 2, axis=2)
                ratio = np.divide(cancellation, m2, out=np.zeros_like(m2), where=m2 != 0)
                result[f"{population}_energy"] = ((m2 + bins[:, :, 4]) / b_ref_sq).mean(axis=1)
                result[f"{population}_cancellation"] = ratio.mean(axis=1)
                if population == "target":
                    result["target_absolute_cancellation"] = (cancellation / b_ref_sq).mean(axis=1)
        if any(not np.isfinite(v).all() for v in result.values()):
            raise ValueError("nonfinite derived features")
        return result
    except (TypeError, ValueError, FloatingPointError) as exc:
        raise ContractMismatch("expected finite physical features and positive finite B_ref_sq") from exc


def _validate_features(features):
    if set(features) != set(FEATURES):
        raise ContractMismatch("physical feature inventory differs")
    arrays = {k: summary._finite(v) for k, v in features.items()}
    n = arrays[FEATURES[0]].size
    if not n or any(v.shape != (n,) for v in arrays.values()):
        raise ContractMismatch("features require aligned nonempty window vectors")
    return arrays


def fit_cutoffs(train_features):
    """Equal weight to every saved TRAIN window, including overlapping windows."""
    train = _validate_features(train_features)
    return {
        name: {
            "values": (edges := np.quantile(train[name], quantiles, method="linear")).tolist(),
            "quantiles": quantiles, "quantile_method": "linear",
            "n_training_windows": len(train[name]),
            "training_edge_tie_counts": [int(np.count_nonzero(train[name] == edge)) for edge in edges],
            "assignment": "searchsorted_side_right_1_based" if name.endswith("energy") else "strictly_greater_than",
        }
        for name, quantiles in CUTOFF_QUANTILES.items()
    }


def group_masks(test_features, cutoffs):
    """Return all 21 masks and membership arrays without fitting on TEST."""
    features = _validate_features(test_features)
    if set(cutoffs) != set(CUTOFF_QUANTILES):
        raise ContractMismatch("cutoff inventory differs")
    for name, quantiles in CUTOFF_QUANTILES.items():
        edges = summary._finite(cutoffs[name]["values"])
        if edges.shape != (len(quantiles),) or np.any(np.diff(edges) < 0):
            raise ContractMismatch("invalid ordered training cutoffs")
    memberships, masks = {}, {"full_test": np.ones(len(features[FEATURES[0]]), dtype=bool)}
    for population, cancellation in (("target", "target_absolute_cancellation"),
                                     ("context", "context_cancellation")):
        quartiles = np.searchsorted(cutoffs[f"{population}_energy"]["values"],
                                    features[f"{population}_energy"], side="right") + 1
        high = features[cancellation] > cutoffs[cancellation]["values"][0]
        memberships[f"{population}_energy_quartile"] = quartiles
        memberships[f"{population}_high_cancellation"] = high
        for label, partition in (("high", high), ("rest", ~high)):
            masks[f"{population}_{label}"] = partition
            for quartile in range(1, 5):
                masks[f"{population}_q{quartile}_{label}"] = partition & (quartiles == quartile)
    return masks, memberships


def summarize_groups(data, months, ids, test_features, cutoffs, *,
                     seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES):
    """Fixed-model month means and existing four paired year contrasts per mask.

    Reduced replicate counts are supported here for bounded synthetic tests only;
    the study API and CLI always request the production constants.
    """
    summary.validate_matrices(data, months, ids)
    features = _validate_features(test_features)
    months = np.asarray(months)
    if (len(features[FEATURES[0]]) != len(ids) or months.dtype.kind not in "iu"
            or not np.isin(months // 100, YEARS).all() or not np.isin(months % 100, TEST_MONTHS).all()
            or any(re.fullmatch(r"\d{4}-\d{2}-.+-s(50|52|60|71)", wid) is None for wid in ids)
            or months.tolist() != [int(w[:4]) * 100 + int(w[5:7]) for w in ids]):
        raise ContractMismatch("feature population or calendar differs from ordered IDs")
    masks, memberships = group_masks(features, cutoffs)
    years, sources = months // 100, summary.source_codes(ids)
    groups = {}
    for name, mask in masks.items():
        subset_ids = [wid for wid, keep in zip(ids, mask) if keep]
        subset_months = months[mask]
        n_years = len(np.unique(years[mask]))
        row = {
            "ids": subset_ids, "ids_sha256": id_hash(subset_ids), "n_windows": len(subset_ids),
            "months": subset_months.tolist(), "month_counts": summary._counts(subset_months),
            "year_counts": summary._counts(years[mask]), "source_counts": summary._counts(sources[mask]),
            "year_source_counts": {str(y): summary._counts(sources[mask & (years == y)])
                                   for y in np.unique(years[mask])},
            "n_months": len(np.unique(subset_months)), "n_years": n_years,
            "status": "empty_stratum" if not subset_ids else "single_year" if n_years == 1 else "represented",
            "membership_scope": "full_test" if name == "full_test" else (
                "outcome_selected_descriptive" if name.startswith("target_") else "conditioning_inputs_exploratory"),
            "realized_feature_means": {k: {"mean": None, "month_balanced": None} for k in FEATURES},
            "methods": {}, "comparisons": {},
        }
        subset = {k: np.asarray(v)[:, mask] for k, v in data.items()}
        for label, values in subset.items():
            keys = ["shared"] if len(values) == 1 else list(map(str, FINAL_SEEDS))
            points = bootstrap.monthly_means(values, subset_months).mean(axis=1) if subset_ids else [None] * len(keys)
            window_points = values.mean(axis=1) if subset_ids else [None] * len(keys)
            row["methods"][label] = {
                "month_balanced": float(np.mean(points)) if subset_ids else None,
                "mean": float(np.mean(window_points)) if subset_ids else None,
                "seedwise": {key: {"month_balanced": float(p) if p is not None else None,
                                   "mean": float(w) if w is not None else None}
                             for key, p, w in zip(keys, points, window_points)},
            }
        if subset_ids:
            row["realized_feature_means"] = {
                key: {weight: float(v) for weight, v in summary.reductions(values[mask], subset_months).items()}
                for key, values in features.items()
            }
            row["comparisons"] = summary.paired_year_summary(subset, subset_months, seed=seed, replicates=replicates)
        groups[name] = row
    return {"groups": groups, "membership_in_test_id_order": {k: v.tolist() for k, v in memberships.items()}}


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ContractMismatch("expected lowercase SHA256")
    return value


def _input_path(path, study):
    # Reject links instead of losing their original spelling and missing retargets.
    path = Path(path)
    path = path if path.is_absolute() else study / path
    relative = summary._relative(path, study)
    if path.absolute() != study / relative:
        raise ContractMismatch("input paths must be canonical study-relative files without symlinks")
    return relative


def _recheck(study, result):
    require_source_match({"inference_source_hashes": result["inference_source_sha256"]})
    root = Path(__file__).resolve().parents[1]
    for relative, digest in result["producer_source_sha256"].items():
        if sha256_file(root / relative) != digest:
            raise ContractMismatch("producer source changed during analysis")
    for relative, digest in result["input_sha256"].items():
        if _input_path(relative, study) != relative or sha256_file(study / relative) != digest:
            raise ContractMismatch(f"input changed during analysis: {relative}")


def summarize_study(study_root, protocol_sha256):
    """Load the completed real-study contract; return a rechecked schema-1 report."""
    study = Path(study_root).resolve()
    _digest(protocol_sha256)
    producer = {"analysis/prigm_training_strata.py": sha256_file(Path(__file__)),
                "analysis/prigm_followup_summary.py": sha256_file(Path(summary.__file__))}
    hashes = {}

    def bind(path, digest):
        relative = _input_path(path, study)
        digest = _digest(digest)
        if relative in hashes and hashes[relative] != digest:
            raise ContractMismatch("input hash bindings disagree")
        hashes[relative] = digest
        return relative

    def read(relative):
        relative = _input_path(relative, study)
        payload = (study / relative).read_bytes()
        bind(relative, sha256_bytes(payload))
        return json.loads(payload, object_pairs_hook=bootstrap._unique_object)

    def check(relative, digest):
        relative = bind(relative, digest)
        if sha256_file(study / relative) != digest:
            raise ContractMismatch("completion or input hash mismatch")

    protocol = read("PROTOCOL.json")
    check("PROTOCOL.json", protocol_sha256)
    require_source_match(protocol)
    producer.update(protocol["inference_source_hashes"])
    if _input_path(protocol["array_root"], study) != "artifacts/inventory":
        raise ContractMismatch("follow-up arrays must be study/artifacts/inventory")
    pipeline, cpu = read("PIPELINE_COMPLETED.json"), read("CPU_FOLLOWUP_COMPLETED.json")
    if (pipeline["protocol_sha256"] != protocol_sha256 or cpu["protocol_sha256"] != protocol_sha256
            or cpu["analysis_source_sha256"] != producer["analysis/prigm_followup_summary.py"]):
        raise ContractMismatch("completion protocol or original summary source mismatch")
    for relative, digest in (
        ("artifacts/comparison.json", pipeline["comparison_sha256"]),
        ("SELECTION.json", pipeline["selection_sha256"]),
        ("artifacts/diagnostic-summary-01.json", cpu["summary_sha256"]),
        ("run_fixed_followup.py", pipeline["launcher_sha256"]),
        ("run_cpu_after_real.py", cpu["launcher_sha256"]),
    ):
        check(relative, digest)

    data, months, provenance = bootstrap.load_study(study, protocol_sha256)
    for path, digest in provenance["input_sha256"].items():
        bind(path, digest)
    ids = provenance["ids"]
    summary.validate_matrices(data, months, ids)
    if (len(ids) != 480 or provenance["protocol_sha256"] != protocol_sha256
            or provenance["selection_sha256"] != pipeline["selection_sha256"]):
        raise ContractMismatch("requires the complete 480-ID protocol-bound test study")
    arrays, normalizers, features = {}, {}, {}
    for split in ("TRAIN", "TEST"):
        relative = f"artifacts/inventory/{split.lower()}.npz"
        check(relative, protocol["arrays"][split]["sha256"])
        z, array_ids, normalizer, identity = load_arrays(study / relative, split, hashes[relative])
        if identity != protocol["arrays"][split]:
            raise ContractMismatch("actual array identity differs from protocol")
        if split == "TEST" and (identity != provenance["arrays"] or array_ids != ids):
            raise ContractMismatch("test array identity or ordered IDs differ from primary scores")
        arrays[split], normalizers[split] = identity, normalizer
        features[split] = physical_features(z, normalizer.b_ref_sq)
        del z
    train, test = normalizers["TRAIN"], normalizers["TEST"]
    if (arrays["TRAIN"]["normalization_sha256"] != arrays["TEST"]["normalization_sha256"]
            or not np.array_equal(train.mean, test.mean) or not np.array_equal(train.scale, test.scale)
            or train.b_ref_sq != test.b_ref_sq or train.n_bins != test.n_bins
            or protocol["training_data_hashes"]["arrays_sha256"] != arrays["TRAIN"]["sha256"]):
        raise ContractMismatch("TRAIN and TEST transforms or training hash differ")
    inventory = read("artifacts/inventory/summary.json")
    counts = {k: inventory[k] for k in ("train_starts", "train_windows_saved", "train_bin_union", "val_selected", "test_selected")}
    if (any(type(v) is not int or v <= 0 for v in counts.values())
            or counts["train_starts"] != counts["train_windows_saved"]
            or counts["train_windows_saved"] != arrays["TRAIN"]["n_windows"]
            or counts["train_bin_union"] != train.n_bins or counts["test_selected"] != 480
            or counts["val_selected"] != protocol["arrays"]["VAL"]["n_windows"]
            or set(inventory["hashes"]) != set(map(str, YEARS))):
        raise ContractMismatch("inventory does not identify the complete training-window pool")
    raw_hashes = {key: _digest(value) for key, value in inventory["hashes"].items()}
    methods = {}
    for label, values in data.items():
        meta = provenance["methods"][label]
        keys = ["shared"] if len(values) == 1 else list(map(str, FINAL_SEEDS))
        if set(meta["score_files"]) != set(keys):
            raise ContractMismatch("primary paths differ from fixed model rows")
        files = {}
        for key in keys:
            relative = _input_path(meta["score_files"][key], study)
            if relative not in hashes:
                raise ContractMismatch("primary score not bound by loader")
            primary = read(relative)
            # The loader returns resolved inventory paths. Also check its original
            # fixed locations, so internal ensemble/fit symlinks cannot retarget
            # while leaving the old canonical payload's hash unchanged.
            ensemble = Path(relative).parent
            if len(values) == 3:
                ensemble /= "ensemble"
            for filename in ("COMMIT.json", "ensemble.npz"):
                location = (ensemble / "gen_00000000" / filename).as_posix()
                if location not in hashes:
                    raise ContractMismatch("ensemble path not bound by loader")
                expected = primary["ensemble_commit_sha256"] if filename == "COMMIT.json" else hashes[location]
                bind(location, expected)
            if primary["control_fit_sha256"] is not None:
                location = "artifacts/controls_test/gaussian_fit.npz"
                if location not in hashes:
                    raise ContractMismatch("control fit not bound by loader")
                check(location, primary["control_fit_sha256"])
            files[key] = {"path": relative, "sha256": hashes[relative],
                          **{k: _digest(primary[k]) if primary[k] is not None else None for k in
                             ("checkpoint_sha256", "control_fit_sha256", "ensemble_commit_sha256")}}
        methods[label] = {"procedure": meta["procedure"], "scoring_kind": meta["scoring_kind"],
                          "selection_kind": meta.get("selection_kind"), "generations": meta.get("generations", {}),
                          "fixed_model_seeds": list(FINAL_SEEDS) if len(values) == 3 else [],
                          "n_realizations": len(values), "files": files}
    cutoffs = fit_cutoffs(features["TRAIN"])
    result = {
        "schema_version": 1, "analysis_source_sha256": producer["analysis/prigm_training_strata.py"],
        "producer_source_sha256": producer, "inference_source_sha256": protocol["inference_source_hashes"],
        "source_inventory_scope": "Producer source keys are repository-relative; input inventory keys are study-relative. Hashes identify actual source bytes, not rewritten historical summary or gradient producers.",
        "protocol_sha256": protocol_sha256, "selection_sha256": provenance["selection_sha256"],
        "test_started_sha256": provenance["test_started_sha256"], "input_sha256": hashes,
        "arrays": arrays, "ids": ids, "ids_sha256": id_hash(ids), "months": np.asarray(months).tolist(),
        "inventory_counts": counts, "recorded_raw_sha256_by_year": raw_hashes,
        "normalization": {"mean": train.mean.tolist(), "scale": train.scale.tolist(),
                          "b_ref_sq": train.b_ref_sq, "n_bins": train.n_bins,
                          "sha256": arrays["TRAIN"]["normalization_sha256"]},
        "cutoffs": cutoffs,
        "cutoff_definitions": {
            "population": "All saved TRAIN windows, equal window weight, including overlapping windows. Not deduplicated bins and not month balanced; TEST never fits cutoffs.",
            "target_indices": list(range(18, 30)), "context_indices": list(range(18)) + list(range(30, 48)),
            "energy": "Window mean of Q=(m^2+v_s)/B_ref_sq over the stated bins.",
            "dimensionless_cancellation": "Window mean of C=(m^2-sum(mu^2))/m^2; bin ratio is zero when m^2=0.",
            "absolute_cancellation": "Target window mean of A=(m^2-sum(mu^2))/B_ref_sq.",
            "tie_policy": "High strictly exceeds the 75th-percentile cutoff; equality is rest. Energy quartiles use searchsorted(side='right')+1, so equality goes above every tied edge. Empty quartiles are retained; counts are not forced equal. No clipping.",
            "test_edge_tie_counts": {k: [int(np.count_nonzero(features["TEST"][k] == edge)) for edge in v["values"]]
                                     for k, v in cutoffs.items()},
        },
        "features_in_test_id_order": {k: v.tolist() for k, v in features["TEST"].items()},
        "methods": methods, "scope": SCOPE,
        "weighting": {"month_balanced": "Equal represented-month means per fixed seed, then equal seed mean.",
                      "mean": "Equal windows per fixed seed, then equal seed mean. Realized feature means are descriptive under both weightings."},
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "generator": "PCG64",
                      "scheme": "Resample represented years with replacement, retaining each occurrence's distinct months. Common draws across methods within a group; restart RNG per group. Existing four raw contrasts only.",
                      "scope": "Marginal paired test-year percentile 95% intervals; no familywise correction, training-threshold uncertainty, seed-population, refitting or selection uncertainty. Empty groups have no contrasts; single-year intervals are unavailable. No two-year intervals or p-values."},
        **summarize_groups(data, months, ids, features["TEST"], cutoffs,
                           seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES),
    }
    _recheck(study, result)
    json_bytes(result)  # Refuse nonfinite final reductions before publication.
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("study-root", "protocol-sha256", "output"):
        parser.add_argument(f"--{flag}", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).absolute()
    if not output.parent.is_dir():
        raise FileNotFoundError("output parent must already exist")
    output = output.parent.resolve() / output.name
    partial = output.with_name(f".{output.name}.partial")
    if any(p.exists() or p.is_symlink() for p in (output, partial)):
        raise FileExistsError("output and partial must be fresh")
    result = summarize_study(args.study_root, args.protocol_sha256)
    payload = json_bytes(result)
    # Same atomic, exclusive publication pattern as the existing toy evidence
    # exporter. A failed attempt retains its partial, never a success JSON.
    with partial.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    fsync_dir(output.parent)
    _recheck(Path(args.study_root).resolve(), result)
    if partial.is_symlink() or sha256_file(partial) != sha256_bytes(payload):
        raise ContractMismatch("partial output changed during publication")
    os.link(partial, output)
    partial.unlink()
    fsync_dir(output.parent)
    print(json.dumps({"n_methods": len(result["methods"]), "n_groups": len(result["groups"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
