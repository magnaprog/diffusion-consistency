"""Paired test-year uncertainty for the protocol-bound, exposed PriGM repair.

Only existing scores are aggregated. Ensemble payloads are read for integrity,
not rescored. The three trained models are fixed, not bootstrap sampling units.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prigm.artifacts import (
    id_hash, json_bytes, load_arrays, read_ensemble, require_array_match,
    require_procedure_match, require_source_match,
)
from prigm.barrier import load_protocol, require_test_started
from prigm.config import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, FINAL_SEEDS, TEST_MONTHS, YEARS
from prigm.recovery import ContractMismatch, sha256_bytes, sha256_file
from prigm.scoring import month_balanced_mean, percentile_interval, two_year_blocks, year_blocks

ARMS = ("D", "I-S", "O-S", "G5")
KINDS = ("raw", "corrected")
CONTROLS = ("interpolation", "persistence", "gaussian")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractMismatch(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path, hashes):
    path = Path(path).resolve()
    payload = path.read_bytes()
    hashes[str(path)] = sha256_bytes(payload)
    return json.loads(payload, object_pairs_hook=_unique_object)


def _selection_contract(selection, protocol, protocol_hash):
    names = {f"{arm}_s{seed}" for arm in ARMS for seed in FINAL_SEEDS}
    if (protocol["arms"] != list(ARMS) or protocol["seeds"] != list(FINAL_SEEDS)
            or selection.get("schema_version") != 2
            or selection.get("protocol_sha256") != protocol_hash
            or selection.get("exposure") != "historical_test_exposed"
            or set(selection.get("selected", {})) != names):
        raise ContractMismatch("selection requires the complete four-arm, three-seed study")
    jobs, checkpoints = set(), set()
    for name, variants in selection["selected"].items():
        if set(variants) != set(KINDS):
            raise ContractMismatch(f"missing raw/corrected selection: {name}")
        for row in variants.values():
            generation = row.get("generation")
            digest = row.get("checkpoint_sha256")
            if (type(generation) is not int or generation not in protocol["milestones"]
                    or not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                    or not np.isfinite(row["score"])):
                raise ContractMismatch(f"invalid selected checkpoint: {name}")
            jobs.add((name, generation))
            checkpoints.add(digest)
        if (variants["raw"]["generation"] == variants["corrected"]["generation"]
                and variants["raw"]["checkpoint_sha256"] != variants["corrected"]["checkpoint_sha256"]):
            raise ContractMismatch("one generation cannot identify two checkpoints")
    actual_jobs = [tuple(job) for job in selection["test_jobs"]]
    actual_hashes = selection["test_checkpoint_hashes"]
    if (len(actual_jobs) != len(set(actual_jobs)) or set(actual_jobs) != jobs
            or len(actual_hashes) != len(set(actual_hashes)) or set(actual_hashes) != checkpoints):
        raise ContractMismatch("test jobs/checkpoint hashes differ from selected union or contain duplicates")


def load_study(study_root: Path, protocol_hash: str):
    """Load every selected variant, raw-weight correction and non-neural control."""
    study = Path(study_root).resolve()
    protocol = load_protocol(study, protocol_hash)
    require_source_match(protocol)
    gated_selection = require_test_started(study, "TEST", protocol_hash)
    hashes = {}
    if _read_json(study / "PROTOCOL.json", hashes) != protocol:
        raise ContractMismatch("protocol changed while reading")
    selection = _read_json(study / "SELECTION.json", hashes)
    phase = _read_json(study / "TEST_STARTED.json", hashes)
    if (selection != gated_selection or phase["selection_sha256"] != hashes[str(study / "SELECTION.json")]
            or phase["protocol_sha256"] != protocol_hash or phase["phase"] != "TEST_STARTED"
            or phase["previous_test_exposure"] is not True):
        raise ContractMismatch("test receipt changed while reading")
    _selection_contract(selection, protocol, protocol_hash)

    array_path = Path(protocol["array_root"]) / "test.npz"
    targets, ids, _, identity = load_arrays(array_path, "TEST", protocol["arrays"]["TEST"]["sha256"])
    del targets
    if identity != protocol["arrays"]["TEST"]:
        raise ContractMismatch("actual test array identity differs from protocol")
    hashes[str(array_path.resolve())] = identity["sha256"]
    months = np.array([int(w[:4]) * 100 + int(w[5:7]) for w in ids], dtype=np.int64)
    if (any(len(w) < 7 or w[4] != "-" for w in ids)
            or not np.isin(months // 100, YEARS).all()
            or not np.isin(months % 100, TEST_MONTHS).all()):
        raise ContractMismatch("test IDs are outside the declared test calendar")
    train_path = Path(protocol["array_root"]) / "train.npz"
    train_hash = sha256_file(train_path)
    if train_hash != protocol["arrays"]["TRAIN"]["sha256"]:
        raise ContractMismatch("control training array hash changed")
    hashes[str(train_path.resolve())] = train_hash
    if identity["normalization_sha256"] != protocol["arrays"]["TRAIN"]["normalization_sha256"]:
        raise ContractMismatch("test and training normalization differ")

    ensembles = {}

    def load_score(path, ensemble_root, kind, *, arm=None, seed=None, selected=None, procedure="diffusion"):
        row = _read_json(path, hashes)
        if ensemble_root not in ensembles:
            commit_path = ensemble_root / "gen_00000000/COMMIT.json"
            hashes[str(commit_path.resolve())] = sha256_file(commit_path)
            draws, sampled_ids, meta = read_ensemble(ensemble_root)
            del draws
            hashes[str((ensemble_root / "gen_00000000/ensemble.npz").resolve())] = meta["files"]["ensemble.npz"]
            ensembles[ensemble_root] = sampled_ids, meta
        sampled_ids, meta = ensembles[ensemble_root]
        require_array_match(meta, identity)
        require_procedure_match(meta, protocol, selection)
        if (sampled_ids != ids or meta["ids_sha256"] != identity["ids_sha256"]
                or meta.get("protocol_sha256") != protocol_hash
                or meta.get("procedure", "diffusion") != procedure):
            raise ContractMismatch(f"ensemble population/protocol/procedure mismatch: {ensemble_root}")
        checkpoint = selected["checkpoint_sha256"] if selected is not None else None
        if procedure == "diffusion":
            if (meta.get("checkpoint_sha256") != checkpoint or meta.get("arm") != arm
                    or meta.get("seed") != seed or meta.get("generation_updates") != selected["generation"]
                    or meta.get("n_generated") != (5 if arm == "G5" else 6)
                    or meta.get("sampler") != protocol["samplers"]["TEST"]):
                raise ContractMismatch(f"ensemble does not match selected checkpoint/sampler: {ensemble_root}")
        elif meta.get("checkpoint_sha256") is not None or meta.get("n_generated") != 6:
            raise ContractMismatch("control must not claim a neural checkpoint")
        fit_hash = meta.get("control_fit_sha256")
        if procedure == "gaussian":
            fit_path = study / "artifacts/controls_test/gaussian_fit.npz"
            actual_fit_hash = sha256_file(fit_path)
            if fit_hash != actual_fit_hash:
                raise ContractMismatch("Gaussian fit payload hash mismatch")
            hashes[str(fit_path.resolve())] = actual_fit_hash
        elif fit_hash is not None:
            raise ContractMismatch("unexpected control fit hash")
        if (row.get("schema_version") != 2 or row.get("protocol_sha256") != protocol_hash
                or row.get("arrays") != identity or row.get("ids") != ids
                or id_hash(row["ids"]) != identity["ids_sha256"]
                or type(row.get("n")) is not int or row["n"] != len(ids)
                or row.get("corrected") is not (kind == "corrected")
                or row.get("checkpoint_sha256") != checkpoint
                or row.get("procedure") != procedure or row.get("control_fit_sha256") != fit_hash
                or row.get("ensemble_commit_sha256") != hashes[str((ensemble_root / "gen_00000000/COMMIT.json").resolve())]):
            raise ContractMismatch(f"score identity, correction or ensemble mismatch: {path}")
        values = np.asarray(row["scores"], dtype=np.float64)
        if (values.shape != (len(ids),) or not np.isfinite(values).all()
                or row.get("months") != months.tolist()):
            raise ContractMismatch(f"score values/count/months mismatch: {path}")
        for key, actual in (("month_balanced", month_balanced_mean(values, months)), ("mean", values.mean())):
            if (not np.isfinite(row[key]) or not np.isclose(row[key], actual, rtol=1e-12, atol=0.0)):
                raise ContractMismatch(f"score {key} does not match window scores: {path}")
        return values

    data, methods = {}, {}
    for arm in ARMS:
        for kind in (*KINDS, "corrected_at_raw_checkpoint"):
            label = f"{arm}_{kind}"
            selection_kind = "raw" if kind == "corrected_at_raw_checkpoint" else kind
            scoring_kind = "raw" if kind == "raw" else "corrected"
            rows, paths, generations = [], {}, {}
            for seed in FINAL_SEEDS:
                name = f"{arm}_s{seed}"
                selected = selection["selected"][name][selection_kind]
                parent = study / "artifacts/test" / f"{name}-g{selected['generation']}"
                path = parent / f"score-{scoring_kind}.json"
                rows.append(load_score(path, parent / "ensemble", scoring_kind,
                                       arm=arm, seed=seed, selected=selected))
                paths[str(seed)] = str(path)
                generations[str(seed)] = selected["generation"]
            data[label] = np.stack(rows)
            methods[label] = {"procedure": "diffusion", "fixed_model_seeds": list(FINAL_SEEDS),
                              "selection_kind": selection_kind, "scoring_kind": scoring_kind,
                              "generations": generations, "score_files": paths}
    for name in CONTROLS:
        for kind in KINDS:
            label = f"{name}_{kind}"
            parent = study / "artifacts/controls_test" / name
            path = parent / f"score-{kind}.json"
            data[label] = load_score(path, parent, kind, procedure=name)[None, :]
            methods[label] = {"procedure": name, "fixed_model_seeds": [], "n_baseline_realizations": 1,
                              "scoring_kind": kind, "score_files": {"shared": str(path)}}
    provenance = {
        "study_root": str(study), "protocol_sha256": protocol_hash,
        "selection_sha256": hashes[str(study / "SELECTION.json")],
        "test_started_sha256": hashes[str(study / "TEST_STARTED.json")],
        "arrays": identity, "ids": ids, "months": months.tolist(),
        "input_sha256": hashes, "source_sha256": protocol["inference_source_hashes"],
        "methods": methods, "selected": selection["selected"],
    }
    return data, months, provenance


def monthly_means(values, months):
    """One row per fixed model, one column per represented month."""
    values = np.asarray(values, dtype=np.float64)
    months = np.asarray(months, dtype=np.int64)
    if (values.ndim != 2 or values.shape[1] != len(months) or not len(months)
            or not np.isfinite(values).all()):
        raise ContractMismatch("expected complete finite model-by-window scores")
    return np.stack([values[:, months == m].mean(axis=1) for m in np.unique(months)], axis=1)


def month_occurrence_weights(months, years_draw):
    """Same estimand as month_balanced_mean_replicated, without window copying.

    Each represented month receives its year's occurrence count. A year with
    only August contributes one month per occurrence, not the weight of two.
    """
    month_years = np.unique(months) // 100
    draws = np.asarray(years_draw, dtype=np.int64)
    if draws.ndim != 2:
        raise ValueError("year draws must be replicate-by-occurrence")
    weights = (draws[:, :, None] == month_years[None, None, :]).sum(axis=1)
    if np.any(weights.sum(axis=1) == 0):
        raise ContractMismatch("bootstrap replicate has no represented months; cannot drop it")
    return weights / weights.sum(axis=1, keepdims=True)


def bootstrap_weights(months):
    years = year_blocks()
    blocks = np.asarray(two_year_blocks(), dtype=np.int64)
    if tuple(years) != tuple(YEARS) or tuple(blocks.ravel()) != tuple(YEARS):
        raise ContractMismatch("bootstrap blocks differ from configured years")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    year_draws = years[rng.integers(len(years), size=(BOOTSTRAP_REPLICATES, len(years)))]
    block_draws = rng.integers(len(blocks), size=(BOOTSTRAP_REPLICATES, len(blocks)))
    return {
        "year": month_occurrence_weights(months, year_draws),
        "twoyear": month_occurrence_weights(months, blocks[block_draws].reshape(BOOTSTRAP_REPLICATES, -1)),
    }


def paired_summary(method, baseline):
    """Paired differences, broadcasting a single shared non-neural baseline."""
    differences = method["seed_means"] - baseline["seed_means"]
    base_mean = float(baseline["seed_means"].mean())
    difference = float(differences.mean())
    relative = difference / base_mean if base_mean > 0 else None
    rec = {
        "seed_diffs": differences.tolist(), "mean_diff": difference,
        "method_mean": float(method["seed_means"].mean()), "baseline_mean": base_mean,
        "method_fixed_models": len(method["seed_means"]),
        "baseline_fixed_models": len(baseline["seed_means"]),
        "seeds_agree_neg": bool(np.all(differences < 0)), "rel_mean_vs_base": relative,
        "relative_point_status": "defined" if relative is not None else "undefined_nonpositive_baseline",
    }
    for scheme in ("year", "twoyear"):
        base_boot = baseline[scheme]
        diff_boot = method[scheme] - base_boot
        if not np.isfinite(diff_boot).all() or not np.isfinite(base_boot).all():
            raise ContractMismatch("nonfinite bootstrap statistic")
        rec[f"{scheme}_95"] = list(percentile_interval(diff_boot))
        undefined = int(np.count_nonzero(base_boot <= 0))
        rec[f"rel_{scheme}_undefined_replicates"] = undefined
        # Never discard invalid denominators or replace them with epsilon.
        rec[f"rel_{scheme}_95"] = (None if undefined else
                                     list(percentile_interval(diff_boot / base_boot)))
    relative_intervals = [rec[f"rel_{scheme}_95"] for scheme in ("year", "twoyear")]
    rec["descriptive_gates"] = {
        "observed_improvement_at_least_2_percent": None if relative is None else relative <= -0.02,
        "both_difference_intervals_below_zero": all(rec[f"{s}_95"][1] < 0 for s in ("year", "twoyear")),
        "strict_relative_equivalence_within_2_percent_both_intervals": (
            None if any(ci is None for ci in relative_intervals)
            else all(-0.02 < ci[0] and ci[1] < 0.02 for ci in relative_intervals)
        ),
    }
    return rec


def compare(data, months):
    weights = bootstrap_weights(months)
    stats, seed_means, arm_means = {}, {}, {}
    for label, values in data.items():
        per_month = monthly_means(values, months)
        points = per_month.mean(axis=1)
        stats[label] = {"seed_means": points,
                        **{scheme: weight @ per_month.mean(axis=0) for scheme, weight in weights.items()}}
        control = label in {f"{name}_{kind}" for name in CONTROLS for kind in KINDS}
        keys = ["shared"] if control else list(map(str, FINAL_SEEDS))
        if len(points) != len(keys):
            raise ContractMismatch("missing fixed-seed models or duplicated control realizations")
        seed_means[label] = dict(zip(keys, points.tolist()))
        arm_means[label] = float(points.mean())

    comparisons, seen = {}, set()

    def add(method, baseline, family, estimand):
        pair = frozenset((method, baseline))
        if method == baseline or pair in seen:
            return
        seen.add(pair)
        comparisons[f"{method}_minus_{baseline}"] = {
            "method": method, "baseline": baseline, "family": family, "estimand": estimand,
            **paired_summary(stats[method], stats[baseline]),
        }

    baselines = ["D_raw", "D_corrected", "gaussian_raw", "gaussian_corrected", "G5_raw", "G5_corrected"]
    # Keep primary O-S directions explicit before deduplicating reverse pairs.
    for baseline in [*baselines, "I-S_raw", "interpolation_raw", "interpolation_corrected",
                     "persistence_raw", "persistence_corrected"]:
        if baseline in {"D_raw", "D_corrected", "G5_raw", "G5_corrected"}:
            family = "primary_O-S_repair"
        elif baseline == "I-S_raw":
            family = "preplanned_secondary_repair"
        else:
            family = "exploratory_control_comparison"
        add("O-S_raw", baseline, family, "validation_selected_procedures")
    add("I-S_raw", "D_raw", "preplanned_secondary_repair", "validation_selected_procedures")
    # Corrected selection can choose different weights. It is not postprocessing alone.
    for arm in ARMS:
        add(f"{arm}_corrected", f"{arm}_raw", "exploratory_selected_correction",
            "independently_validation_selected_corrected_minus_raw")
        add(f"{arm}_corrected_at_raw_checkpoint", f"{arm}_raw", "exploratory_same_checkpoint_correction",
            "support_correction_minus_raw_at_raw_selected_weights")
    for arm in ARMS:
        for kind in KINDS:
            for baseline in baselines:
                add(f"{arm}_{kind}", baseline, "exploratory_multiple_comparisons", "validation_selected_procedures")
    for name in CONTROLS:
        add(f"{name}_corrected", f"{name}_raw", "exploratory_control_correction",
            "support_correction_minus_raw_same_control_ensemble")
    return {"arm_means": arm_means, "seed_means": seed_means, "comparisons": comparisons}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", required=True, help="fresh JSON output path")
    args = parser.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    data, months, provenance = load_study(Path(args.study_root), args.protocol_sha256)
    rec = {
        "schema_version": 2, **provenance, **compare(data, months),
        "exposure": "historical_test_exposed", "n_replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED, "fixed_training_seeds": list(FINAL_SEEDS),
        "multiplicity": "with_replacement_year_blocks",
        "year_blocks": year_blocks().tolist(), "two_year_blocks": two_year_blocks(),
        "month_counts": {str(y * 100 + m): int(np.count_nonzero(months == y * 100 + m))
                         for y in YEARS for m in TEST_MONTHS},
        "uncertainty_scope": "Paired test-year uncertainty averaged over three fixed trained models; identical year draws for every model and comparison. Not training-seed, refitting or selection uncertainty. Each non-neural control is one shared realization, not three independent baselines.",
        "comparison_scope": "Primary O-S comparisons are distinguished for the preplanned repair of a historically exposed test study, not a restored blind confirmation. Expanded and correction comparisons are exploratory multiple comparisons. Percentile intervals are marginal, not familywise corrected.",
        "interpretation": "Lower primary energy score is better; contrasts are method minus baseline. Relative intervals use each replicate's baseline mean. Two-percent gates are descriptive, not causal verdicts or universal negative claims. Failure to show superiority is not equivalence; the optional equivalence criterion requires both relative intervals strictly inside [-0.02, 0.02]. No p-value establishes equivalence. Block lengths are temporal-dependence sensitivity checks, not a guarantee under solar-cycle nonstationarity.",
    }
    require_source_match(load_protocol(Path(args.study_root), args.protocol_sha256))
    for path, digest in provenance["input_sha256"].items():
        if sha256_file(Path(path)) != digest:
            raise ContractMismatch(f"input changed during comparison: {path}")
    payload = json_bytes(rec)
    with out.open("xb") as handle:
        handle.write(payload)
    print(json.dumps({"output": str(out.resolve()), "n_comparisons": len(rec["comparisons"]),
                      "arm_means": rec["arm_means"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
