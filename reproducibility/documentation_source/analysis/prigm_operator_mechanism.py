"""Paired correction diagnostics from a complete, protocol-bound PriGM study.

Read original ensembles, never aggregate-only evidence. No fitting, selection,
bootstrap or prediction generation. The output path must be fresh.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import prigm_followup_summary as followup
from prigm import run_bootstrap as bootstrap
from prigm.artifacts import (
    json_bytes, load_arrays, read_ensemble, require_array_match,
    require_procedure_match, require_source_match,
)
from prigm.config import FINAL_SEEDS, HIDDEN_END, HIDDEN_START, N_DRAWS
from prigm.observation import closure_correct, derive_vv, support_gaps
from prigm.precision import arithmetic_tolerance, representation_bands
from prigm.recovery import ContractMismatch, sha256_file
from prigm.run_score import decode_ensemble, hidden_primary_std
from prigm.scoring import pairwise_l2_offdiag

QUANTILES = (0.0, 0.5, 0.9, 0.99, 1.0)
MARGINS = ("m_minus_r", "vs", "vs_upper_minus_vs")


def energy_terms(samples, target):
    """Off-diagonal energy estimator, retaining its two positive contributions."""
    x, y = followup._finite(samples), followup._finite(target)
    if x.ndim != 2 or len(x) < 2 or y.shape != (x.shape[1],):
        raise ContractMismatch("energy terms require K>=2 draws and one aligned target")
    target_term = float(np.linalg.norm(x - y[None], axis=-1).mean())
    dispersion_term = pairwise_l2_offdiag(x) / 2.0
    terms = {"target_term": target_term, "dispersion_term": dispersion_term,
             "energy_score": target_term - dispersion_term}
    followup._finite(list(terms.values()))
    return terms


def check_displacement_bound(raw, corrected, mean_movement):
    """Check the paired bound with a float64 score-arithmetic tolerance."""
    values = followup._finite([*raw.values(), *corrected.values(), mean_movement])
    if values[-1] < 0:
        raise ContractMismatch("mean movement cannot be negative")
    delta = corrected["energy_score"] - raw["energy_score"]
    tolerance = float(arithmetic_tolerance(
        raw["target_term"] + raw["dispersion_term"],
        corrected["target_term"] + corrected["dispersion_term"], 1.0))
    bound = 2.0 * mean_movement
    if abs(delta) > bound + tolerance:
        raise ContractMismatch("paired energy change exceeds displacement bound")
    return {"displacement_bound": bound, "bound_slack": bound - abs(delta),
            "bound_arithmetic_tolerance": tolerance}


def _quantiles(values):
    return np.quantile(followup._finite(values), QUANTILES, method="linear").tolist()


def support_statistics(samples, nrm, stored_dtype):
    """Per-window margins over draw x hidden-bin pairs; positive is inside.

    Bands match run_secondaries.window_diagnostics, including its float32
    representation branch. Severity is max(-margin, 0), including zeros.
    """
    z = followup._finite(samples)
    gaps = support_gaps(z)
    b_ref = np.sqrt(nrm.b_ref_sq)
    if np.dtype(stored_dtype) == np.dtype("float32"):
        bands = representation_bands(z, (z - nrm.mean) / nrm.scale,
                                     nrm.scale, b_ref, nrm.b_ref_sq)
    elif np.dtype(stored_dtype) == np.dtype("float64"):
        r = np.linalg.norm(z[..., :3], axis=-1)
        m, vs = z[..., 3], z[..., 4]
        bands = {
            "m_minus_r": arithmetic_tolerance(m, r, b_ref),
            "vs": arithmetic_tolerance(vs, 0, nrm.b_ref_sq),
            "vs_upper_minus_vs": arithmetic_tolerance(1.5 * m ** 2 + 2.5 * r ** 2, vs, nrm.b_ref_sq),
        }
    else:
        raise ContractMismatch("support bands require original float32 or float64 storage")
    result = {}
    for key in MARGINS:
        unit = b_ref if key == "m_minus_r" else nrm.b_ref_sq
        margin = gaps[key] / unit
        result[f"{key}_signed_quantiles"] = _quantiles(margin)
        result[f"{key}_severity_quantiles"] = _quantiles(np.maximum(-margin, 0))
        result[f"{key}_violation_fraction"] = float((gaps[key] < 0).mean())
        result[f"{key}_beyond_rounding_fraction"] = float((gaps[key] < -bands[key]).mean())
    result["support_violation_fraction"] = float(np.logical_or.reduce([gaps[k] < 0 for k in MARGINS]).mean())
    result["support_beyond_rounding_fraction"] = float(np.logical_or.reduce([gaps[k] < -bands[k] for k in MARGINS]).mean())
    result["absolute_equality_residual_mean"] = float(np.abs(gaps["r_obs"]).mean() / nrm.b_ref_sq)
    return result


def window_mechanism(samples, truth_hidden, nrm, stored_dtype=np.float32):
    """Same-draw no-op, equality completion and full physical-unit correction.

    Pure helper accepts arbitrary K>=2, including synthetic 20-draw tests. The
    original-study reader separately requires the frozen 39-draw payloads.
    """
    z, y = followup._finite(samples), followup._finite(truth_hidden)
    if z.ndim != 3 or z.shape[1:] != (12, 6) or len(z) < 2 or y.shape != (12, 6):
        raise ContractMismatch("mechanism requires (K,12,6) samples and (12,6) truth")
    completed = derive_vv(z[..., :5])  # derive_vv APPENDS a sixth coordinate.
    if completed[..., :5].tobytes() != z[..., :5].tobytes():
        raise ContractMismatch("equality completion changed primary coordinate bits")
    corrected = closure_correct(z)
    if not np.array_equal(closure_correct(corrected), corrected):
        raise ContractMismatch("support correction is not idempotent")
    gaps = support_gaps(corrected)
    r = np.linalg.norm(corrected[..., :3], axis=-1)
    m, vs, vv = corrected[..., 3], corrected[..., 4], corrected[..., 5]
    tolerances = {
        "m_minus_r": arithmetic_tolerance(m, r, np.sqrt(nrm.b_ref_sq)),
        "vs": arithmetic_tolerance(vs, 0, nrm.b_ref_sq),
        "vs_upper_minus_vs": arithmetic_tolerance(1.5 * m ** 2 + 2.5 * r ** 2, vs, nrm.b_ref_sq),
    }
    if (any(np.any(gaps[k] < -tolerances[k]) for k in MARGINS)
            or np.any(np.abs(gaps["r_obs"]) > arithmetic_tolerance(m ** 2 + vs, r ** 2 + vv, nrm.b_ref_sq))):
        raise ContractMismatch("corrected samples are not feasible within arithmetic tolerance")
    xs, cs = hidden_primary_std(z, nrm), hidden_primary_std(corrected, nrm)
    completed_xs = hidden_primary_std(completed, nrm)
    if completed_xs.tobytes() != xs.tobytes():
        raise ContractMismatch("equality completion changed standardized primary bits")
    ys = hidden_primary_std(y[None], nrm)[0]
    raw, complete, repair = (energy_terms(x, ys) for x in (xs, completed_xs, cs))
    if raw != complete:
        raise ContractMismatch("equality completion changed primary energy terms")
    movement = np.linalg.norm(cs - xs, axis=-1)
    result = {f"{kind}_{key}": value for kind, terms in
              (("raw", raw), ("completed", complete), ("corrected", repair))
              for key, value in terms.items()}
    result.update({f"delta_{key}": repair[key] - raw[key] for key in raw})
    term_delta = result["delta_target_term"] - result["delta_dispersion_term"]
    # Subtraction of separately rounded terms can differ at machine precision.
    bound = check_displacement_bound(raw, repair, float(movement.mean()))
    if abs(term_delta - result["delta_energy_score"]) > bound["bound_arithmetic_tolerance"]:
        raise ContractMismatch("energy component changes do not reproduce paired change")
    result.update(bound)
    result["mean_movement"] = float(movement.mean())
    result["movement_quantiles"] = _quantiles(movement)
    for kind, values in (("raw", z), ("corrected", corrected)):
        result.update({f"{kind}_{key}": value for key, value in
                       support_statistics(values, nrm, stored_dtype).items()})
    result["completed_absolute_equality_residual_mean"] = float(
        np.abs(support_gaps(completed)["r_obs"]).mean() / nrm.b_ref_sq)
    # Refuse overflow as well as nonfinite input. Never emit NaN placeholders.
    for value in result.values():
        followup._finite(value)
    return result


def summarize_windows(records, months):
    """Average per-window quantities, including within-window quantiles."""
    if not records or any(set(r) != set(records[0]) for r in records):
        raise ContractMismatch("incomplete per-window mechanism records")
    return {key: {weight: value.tolist() for weight, value in
                  followup.reductions([r[key] for r in records], months).items()}
            for key in records[0]}


def selection_decomposition(raw, corrected_at_raw, corrected_selected, months):
    """Selected corrected minus raw = fixed-weight repair + selection effect."""
    raw, same, selected = map(followup._finite, (raw, corrected_at_raw, corrected_selected))
    if raw.ndim != 1 or raw.shape != same.shape or raw.shape != selected.shape:
        raise ContractMismatch("selection decomposition requires aligned window vectors")
    fixed, selection, total = same - raw, selected - same, selected - raw
    tol = arithmetic_tolerance(np.abs(raw) + np.abs(same), np.abs(selected) + np.abs(same), 1.0)
    if np.any(np.abs(total - fixed - selection) > tol):
        raise ContractMismatch("selection decomposition does not add up")
    rows = [{"raw_energy_score": float(r), "corrected_at_raw_energy_score": float(c),
             "corrected_selected_energy_score": float(s), "fixed_checkpoint_correction": float(f),
             "selection_effect": float(e), "total_change": float(t)}
            for r, c, s, f, e, t in zip(raw, same, selected, fixed, selection, total)]
    return {"per_window": rows, "metrics": summarize_windows(rows, months)}


def _seed_average(seedwise):
    first = next(iter(seedwise.values()))["metrics"]
    return {key: {weight: np.mean([row["metrics"][key][weight] for row in seedwise.values()], axis=0).tolist()
                  for weight in ("month_balanced", "mean")} for key in first}


def validate_payload(meta, sampled_ids, ids, identity, protocol, selection, protocol_hash,
                     procedure, selected=None, arm=None, seed=None):
    """Bind a reread, hash-verified payload to its original selection and arrays."""
    require_array_match(meta, identity)
    require_procedure_match(meta, protocol, selection)
    if (sampled_ids != ids or meta.get("ids_sha256") != identity["ids_sha256"]
            or meta.get("protocol_sha256") != protocol_hash
            or meta.get("procedure", "diffusion") != procedure):
        raise ContractMismatch("mechanism ensemble population/protocol/procedure mismatch")
    if procedure == "diffusion":
        if (selected is None or meta.get("checkpoint_sha256") != selected["checkpoint_sha256"]
                or meta.get("generation_updates") != selected["generation"]
                or meta.get("arm") != arm or meta.get("seed") != seed
                or meta.get("n_generated") != (5 if arm == "G5" else 6)
                or meta.get("sampler") != protocol["samplers"]["TEST"]):
            raise ContractMismatch("mechanism ensemble selected checkpoint/sampler mismatch")
    elif meta.get("checkpoint_sha256") is not None or meta.get("n_generated") != 6:
        raise ContractMismatch("mechanism control claims a neural checkpoint or wrong channels")


def analyze_study(study_root, protocol_sha256):
    """Gate the complete original study once, then stream diagnostic ensembles."""
    study = Path(study_root).resolve()
    producer_paths = {"analysis/prigm_operator_mechanism.py": Path(__file__),
                      "analysis/prigm_followup_summary.py": Path(followup.__file__)}
    producer_hashes = {name: sha256_file(path) for name, path in producer_paths.items()}
    data, months, provenance = bootstrap.load_study(study, protocol_sha256)
    ids = provenance["ids"]
    followup.validate_matrices(data, months, ids)
    hashes = {followup._relative(p, study): h for p, h in provenance["input_sha256"].items()}
    protocol = followup._read_json(study / "PROTOCOL.json", study, hashes)
    selection = followup._read_json(study / "SELECTION.json", study, hashes)
    if hashes["PROTOCOL.json"] != protocol_sha256 or selection["selected"] != provenance["selected"]:
        raise ContractMismatch("protocol or selection changed after original gate")
    array_root = study / followup._relative(protocol["array_root"], study)
    z, target_ids, nrm, identity = load_arrays(array_root / "test.npz", "TEST", provenance["arrays"]["sha256"])
    if target_ids != ids or identity != provenance["arrays"]:
        raise ContractMismatch("mechanism targets differ from original ordered arrays")
    if protocol["samplers"]["TEST"]["n_draws"] != N_DRAWS:
        raise ContractMismatch("mechanism requires the original 39-draw protocol")

    # Complete secondary gate without calling summarize_study (which bootstraps
    # strata and reloads the entire primary study). Validate repeated files once.
    primary_records, secondary_files = {}, {}
    for label, values in data.items():
        method = provenance["methods"][label]
        keys = list(map(str, FINAL_SEEDS)) if len(values) == 3 else ["shared"]
        for index, key in enumerate(keys):
            path = study / followup._relative(method["score_files"][key], study)
            relative = followup._relative(path, study)
            if relative not in hashes:
                raise ContractMismatch("primary score is not bound by original loader")
            if relative not in primary_records:
                primary = followup._read_json(path, study, hashes)
                secondary_path = path.with_name(f"secondaries-{method['scoring_kind']}.json")
                secondary = followup._read_json(secondary_path, study, hashes)
                followup.validate_secondary(secondary, primary, values[index], months)
                primary_records[relative] = primary
                secondary_files[relative] = followup._relative(secondary_path, study)
            followup._agree(primary_records[relative]["scores"], values[index], "aliased primary score")

    def analyze(label, key, index, selection_kind):
        method = provenance["methods"][label]
        primary_path = study / followup._relative(method["score_files"][key], study)
        relative = followup._relative(primary_path, study)
        primary = primary_records[relative]
        procedure = method["procedure"]
        arm = label.split("_", 1)[0] if procedure == "diffusion" else None
        selected = selection["selected"][f"{arm}_s{key}"][selection_kind] if arm else None
        ensemble_root = primary_path.parent / "ensemble" if arm else primary_path.parent
        commit = followup._relative(ensemble_root / "gen_00000000/COMMIT.json", study)
        payload = followup._relative(ensemble_root / "gen_00000000/ensemble.npz", study)
        if commit not in hashes or payload not in hashes or sha256_file(study / commit) != hashes[commit]:
            raise ContractMismatch("ensemble commit changed after original gate")
        draws, sampled_ids, meta = read_ensemble(ensemble_root)
        validate_payload(meta, sampled_ids, ids, identity, protocol, selection, protocol_sha256,
                         procedure, selected, arm, int(key) if arm else None)
        if (meta["files"]["ensemble.npz"] != hashes[payload]
                or hashes[commit] != primary["ensemble_commit_sha256"]
                or meta.get("control_fit_sha256") != primary["control_fit_sha256"]):
            raise ContractMismatch("ensemble payload or control fit changed after original gate")
        rows = [window_mechanism(decode_ensemble(d, nrm), truth[HIDDEN_START:HIDDEN_END], nrm, draws.dtype)
                for d, truth in zip(draws, z)]
        if len(rows) != len(ids) or draws.shape[1] != N_DRAWS:
            raise ContractMismatch("incomplete mechanism ensemble")
        del draws
        expected = {"corrected": data[label][index]} if selection_kind == "corrected" else {
            "raw": data[label][index],
            "corrected": data[f"{arm}_corrected_at_raw_checkpoint" if arm else f"{procedure}_corrected"][index],
        }
        for kind, scores in expected.items():
            followup._agree(scores, [r[f"{kind}_energy_score"] for r in rows], f"recomputed {label} {key} {kind}")
        return {"procedure": procedure, "selection_kind": selection_kind,
                "checkpoint_sha256": primary["checkpoint_sha256"],
                "control_fit_sha256": primary["control_fit_sha256"],
                "generation": selected["generation"] if selected else None,
                "stored_dtype": str(np.dtype("float32" if arm else "float64")),
                "files": {"primary": relative, "secondary": secondary_files[relative],
                          "ensemble_commit": commit, "ensemble_payload": payload},
                "verified_primary_scoring_kinds": list(expected),
                "n_draws": N_DRAWS, "metrics": summarize_windows(rows, months), "per_window": rows}

    methods = {}
    for name in (*bootstrap.ARMS, *bootstrap.CONTROLS):
        keys = list(map(str, FINAL_SEEDS)) if name in bootstrap.ARMS else ["shared"]
        seedwise = {key: analyze(f"{name}_raw", key, i, "raw") for i, key in enumerate(keys)}
        methods[name] = {"seedwise": seedwise, "metrics": _seed_average(seedwise),
                         "n_realizations": len(keys)}
    extra, decomp = {}, {}
    for i, seed in enumerate(FINAL_SEEDS):
        key = str(seed)
        raw = methods["D"]["seedwise"][key]
        selected = selection["selected"][f"D_s{seed}"]["corrected"]
        changed = (selected["checkpoint_sha256"], selected["generation"]) != (raw["checkpoint_sha256"], raw["generation"])
        if changed:
            extra[key] = analyze("D_corrected", key, i, "corrected")
        chosen = extra[key] if changed else raw
        followup._agree(data["D_corrected"][i], [r["corrected_energy_score"] for r in chosen["per_window"]],
                        "D corrected-selected energy")
        rec = selection_decomposition(
            [r["raw_energy_score"] for r in raw["per_window"]],
            [r["corrected_energy_score"] for r in raw["per_window"]],
            [r["corrected_energy_score"] for r in chosen["per_window"]], months)
        rec.update(raw_generation=raw["generation"], corrected_generation=selected["generation"],
                   raw_checkpoint_sha256=raw["checkpoint_sha256"],
                   corrected_checkpoint_sha256=selected["checkpoint_sha256"], selection_changed=changed)
        decomp[key] = rec
    result = {
        "schema_version": 1, "analysis": "paired_operator_mechanism", "protocol_sha256": protocol_sha256,
        "selection_sha256": provenance["selection_sha256"], "test_started_sha256": provenance["test_started_sha256"],
        "producer_source_sha256": producer_hashes, "core_source_sha256": protocol["inference_source_hashes"],
        "input_sha256": hashes, "arrays": followup._array_identity(identity),
        "input_manifest": {"array_root": followup._relative(array_root, study),
                           "array_identities": {k: followup._array_identity(v) for k, v in protocol["arrays"].items()},
                           "array_identity_scope": "TEST payload reloaded. TRAIN file hash and normalization agreement checked by original loader. VAL identity copied from original protocol; VAL payload not reread."},
        "normalization": {"mean": nrm.mean.tolist(), "scale": nrm.scale.tolist(),
                          "b_ref_sq": nrm.b_ref_sq, "n_bins": nrm.n_bins},
        "ids": ids, "months": months.tolist(), "n_windows": len(ids),
        "month_counts": {str(m): int((months == m).sum()) for m in np.unique(months)},
        "methods": methods, "D_additional_corrected_selected": extra,
        "D_selection_decomposition": {"seedwise": decomp, "metrics": _seed_average(decomp)},
        "quantiles": list(QUANTILES), "quantile_method": "linear",
        "quantile_scope": "Within each window: support over draw x 12 hidden bins; movement over draws. Severity max(-signed_margin,0) includes feasible zeros. Summary quantiles are means of within-window quantiles, not pooled quantiles.",
        "units": {"primary_score_and_movement": "Euclidean norm in 60 TRAIN-standardized coordinates, not coordinate-mean or RMS.",
                  "m_minus_r": "physical nT divided by TRAIN B_ref",
                  "vs_and_vs_upper_minus_vs": "physical nT^2 divided by TRAIN B_ref^2",
                  "absolute_equality_residual_mean": "physical nT^2 divided by TRAIN B_ref^2"},
        "weighting": {"month_balanced": "Equal represented-month means per fixed seed, then equal seed mean.",
                      "mean": "Equal windows per fixed seed, then equal seed mean; not month balanced."},
        "operator_scope": "Equality completion changes only v_v. Full closure_correct retains mu, raises m to r, clips v_s using the repaired m, then derives v_v. Movement is correction displacement, not minimum distance to the feasible set. Margins check inequalities; equality residual is separate.",
        "score_scope": "ES = target_term - dispersion_term; dispersion_term uses ordered off-diagonal pairs divided by 2 K (K-1). D selection_effect compares corrected predictions at corrected-selected versus raw-selected weights; it is not same-draw postprocessing.",
        "roundoff_scope": "Inequality frequencies use exactly the stored-dtype branches in run_secondaries.window_diagnostics. Correction feasibility and the displacement bound use arithmetic_tolerance; equality completion preserves primary bits and energy terms exactly.",
        "population_scope": "Historically test-exposed, complete-case artificial gaps in the original selected TEST population. Three fixed seeds per neural arm, one shared realization per frozen control. Never pool 117 draws. Descriptive paired postprocessing, not why training produced a distribution; no bootstrap, refitting, new selection or causal claim.",
        "verification_scope": "Complete original primary/secondary gate, then raw and fixed-weight corrected primary scores recomputed from original payloads; additional D corrected-selected scores also recomputed. Checkpoint hashes are bound to original selection and ensemble metadata; checkpoint weight files are not reread. Per-window rows follow ids/months order. Input names are relative to the original study root, which is not published.",
    }
    require_source_match(protocol)
    for relative, digest in hashes.items():
        if followup._relative(study / relative, study) != relative or sha256_file(study / relative) != digest:
            raise ContractMismatch(f"input changed during mechanism analysis: {relative}")
    if any(sha256_file(producer_paths[name]) != digest for name, digest in producer_hashes.items()):
        raise ContractMismatch("mechanism producer source changed during analysis")
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
        raise FileExistsError("mechanism output must be fresh")
    result = analyze_study(args.study_root, args.protocol_sha256)
    with output.open("xb") as handle:
        handle.write(json_bytes(result))
    print(json.dumps({"n_windows": result["n_windows"], "n_methods": len(result["methods"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
