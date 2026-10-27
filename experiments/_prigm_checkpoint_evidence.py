"""Verify public results for the retrospective comparison of saved checkpoints.

The grid contains D and O-S at three training seeds and five saved updates,
giving thirty checkpoints. Recompute averages from the published per-window
scores for validation at 25 and 50 denoiser evaluations and TEST at 50.
Check the corresponding validation-selected and fixed-final update choices,
their score differences, and the reported stability of validation selection
under year resampling.

File identities connect these results to the original study and its later
fixed-final comparison. The checks use public scores and recorded metadata;
they do not import the checkpoint producer, retrain models or regenerate
neural predictions. TEST scores do not select the checkpoint.
"""

import hashlib
import json
from pathlib import Path
import re

import numpy as np

from experiments import verify_prigm_paper as g


ARMS = ("D", "O-S")
SEEDS = (7, 13, 42)
UPDATES = (2868, 9560, 16252, 22944, 29636)
NAMES = tuple(f"{arm}_s{seed}" for arm in ARMS for seed in SEEDS)
JOBS = tuple(f"{name}-g{update}" for name in NAMES for update in UPDATES)
PROTOCOL = "eeca7c2e14d982f0b3e972256cc9acf7a5aaa8dc03da6a9299a708c78a7b5946"
SCOPE = "5210fbe5432aec70433ed9d7c10a1ba61503e5c7c15e43cc2fa37bfac478674a"
SAVED_VAL25 = "3d8c4b835fd3156490d75db877c675ec8f6a85c61c2bfb9fb0e6dc4a6d27f0b4"
ORIGINAL = "25d0cdefdd5dbb160c32c7b5f4a6053bab99e4a9f64832d9eb6502232bf24e6b"
ORIGINAL_SELECTION = "b2fa1d9233b94c9db22472e644de1fc84f217baaa3d7385faaab2208e0a9c351"
EXTENSION = "1408cd52f306af0095f808dfe1d34aa8493fb26db9a4e75dae422839157bc3b4"
EXTENSION_SUMMARY = "be795841a48d9ef1864c7adcbeb52e499e25a3bd6cc8a60909c5fbd557884b34"
SELECTION = "7381e01b15b5a9f6965fcc1d5083183301ca166d2607abc2e13eea276bf137e7"
REUSE = "776458f389a464340dd19c7bb73d38be854af4e676a569931288efb3f9cda11e"
RUNNER = "d8f1a882d14d3c308862a57fcb5ed77629daa95e304197165f7eb1e833b07182"
REAL_MANIFEST = "17ac806ec1d3af99764fd4ce23bce0d0502cd49fd89719504f0e2926389967c0"
DEADLINE_MANIFEST = "e0599acfbdf3541cd7d3004e364ef43ae11c395b776c0fc519ddc7e839ea1756"
PUBLIC_MANIFEST = "385b7048217cddfaed2c6bb48a1706928ddca1c4b35fd71e533065a518c5d036"
# Historical exporter at the accepted checkpoint follow-up. Later figure styling
# does not regenerate the public evidence or change its producer identity.
PUBLIC_EXPORTER = "ec574515f3738141a6efeaa84cb63ccaeccc72fbdf07770ffea091741cf8fc9d"
SUMMARY = "f835ec62257ba143c707cf89c43e705180db5eee29b0ef1b05a041596fbd3d0c"
FILES = {"PROTOCOL.json", "SCOPE.json", "SELECTION.json", "SAVED_VAL25.json", "SUMMARY.json"}
COUNTS = {"VAL": 319, "TEST": 480}
BOOTSTRAP = {"replicates": 10000, "seed": 20261001, "generator": "PCG64",
             "scope": "Paired validation years, fixed weights/draws; not MC or refitting uncertainty."}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def window_months(ids, months, split):
    require(isinstance(ids, list) and len(ids) == COUNTS[split]
            and all(isinstance(w, str) and re.match(r"^\d{4}-\d{2}-", w) for w in ids)
            and len(set(ids)) == len(ids), split + " window population")
    labels = [int(w[:4]) * 100 + int(w[5:7]) for w in ids]
    g.agree(months, labels, split + " month/ID order")
    expected = [year * 100 + month for year in range(2005, 2025)
                for month in ((11, 12) if split == "VAL" else (8, 9))]
    require(np.unique(labels).tolist() == expected, split + " complete twenty-year calendar")
    return np.asarray(labels)


def reduce_grid(record, split):
    require(set(record) == {"ids", "months", "data"}, split + " score fields")
    months = window_months(record["ids"], record["months"], split)
    require(set(record["data"]) == set(JOBS), split + " complete thirty-checkpoint grid")
    monthly, means = {}, {}
    for name in NAMES:
        scores = np.asarray([record["data"][f"{name}-g{u}"] for u in UPDATES])
        require(scores.dtype.kind in "fi" and scores.shape == (5, COUNTS[split])
                and np.isfinite(scores).all(), split + " finite per-window scores: " + name)
        # Each represented month has equal weight, including VAL's shorter month.
        monthly[name] = np.column_stack([scores[:, months == m].mean(axis=1) for m in np.unique(months)])
        means[name] = monthly[name].mean(axis=1).tolist()
    return means, monthly


def ordered_indices(values):
    return sorted(range(5), key=lambda i: (values[i], UPDATES[i]))


def ranks(values):
    result = [0] * 5
    for rank, index in enumerate(ordered_indices(values), 1):
        result[index] = rank
    return result


def selection_diagnostics(monthly):
    """Measure validation-selection stability for five saved updates.

    The input contains five score curves over forty months: two represented
    months in each of twenty years. Average the two months within each year,
    then draw twenty years with replacement in each of 10,000 replicates.
    Use the same sampled years for all five updates. An exact tie selects
    the earlier update. Also repeat the choice with each year omitted once.

    These counts describe choices among fixed weights and fixed prediction
    draws. They do not measure variation from new training runs or new draws.
    """
    monthly = np.asarray(monthly)
    require(monthly.shape == (5, 40) and monthly.dtype.kind in "fi"
            and np.isfinite(monthly).all(), "five complete monthly validation curves")
    means = monthly.mean(axis=1)
    order = ordered_indices(means)
    years = monthly.reshape(5, 20, 2).mean(axis=2).T
    sampled = np.random.Generator(np.random.PCG64(20261001)).integers(0, 20, size=(10000, 20))
    winners = np.argmin(years[sampled].mean(axis=1), axis=1)
    omitted = [np.argmin(years[np.arange(20) != leave].mean(axis=0)) for leave in range(20)]
    return {"means": means.tolist(), "selected_generation": UPDATES[order[0]],
            "runner_up_gap": float(means[order[1]] - means[order[0]]),
            "bootstrap_selection_counts": {str(u): int(np.count_nonzero(winners == i)) for i, u in enumerate(UPDATES)},
            "leave_year_out_counts": {str(u): int(np.count_nonzero(np.asarray(omitted) == i)) for i, u in enumerate(UPDATES)}}


def validate_reductions(summary, saved, selection, original_p25):
    require(set(summary) == {"protocol_sha256", "selection_sha256", "policies", "scores", "curves",
            "input_sha256", "scope", "policy_means", "contrast_changes", "common_update_TEST_contrasts",
            "VAL50_selection_diagnostics", "validation_budget_comparison", "final_minus_selected_TEST"},
            "checkpoint summary fields; TEST selection diagnostics are not authorized")
    require(set(summary["scores"]) == set(summary["curves"]) == {"VAL", "TEST"}, "both complete score grids")
    g.agree(saved["bootstrap"], BOOTSTRAP, "validation bootstrap")
    old, monthly25 = reduce_grid({"ids": saved["ids"], "months": saved["months"], "data": saved["scores"]}, "VAL")
    val, monthly50 = reduce_grid(summary["scores"]["VAL"], "VAL")
    test, _ = reduce_grid(summary["scores"]["TEST"], "TEST")
    g.agree(saved["ids"], summary["scores"]["VAL"]["ids"], "paired VAL IDs")
    g.agree(saved["months"], summary["scores"]["VAL"]["months"], "paired VAL months")
    g.agree(summary["curves"], {"VAL": val, "TEST": test}, "checkpoint month-balanced curves")
    diagnostics25 = {n: selection_diagnostics(monthly25[n]) for n in NAMES}
    diagnostics50 = {n: selection_diagnostics(monthly50[n]) for n in NAMES}
    g.agree(saved["curves"], diagnostics25, "VAL25 selection stability")
    g.agree(summary["VAL50_selection_diagnostics"], diagnostics50, "VAL50 selection stability")
    require(set(original_p25) == set(NAMES), "six original raw selections")
    policies = {name: {"P25": UPDATES[ordered_indices(old[name])[0]],
                      "P50": UPDATES[ordered_indices(val[name])[0]], "F": UPDATES[-1]} for name in NAMES}
    for name in NAMES:
        g.agree(policies[name]["P25"], original_p25[name]["generation"], "original P25 generation " + name)
        g.agree(old[name][UPDATES.index(policies[name]["P25"])], original_p25[name]["score"], "original P25 score " + name)
    g.agree(selection["policies"], policies, "policies selected only from complete VAL")
    g.agree(summary["policies"], policies, "reported fixed policies")
    budget = {}
    for name in NAMES:
        budget[name] = {"VAL25": old[name], "VAL50": val[name],
                       "VAL50_minus_VAL25": (np.asarray(val[name]) - old[name]).tolist(),
                       "VAL25_ranks_earlier_ties": ranks(old[name]), "VAL50_ranks_earlier_ties": ranks(val[name]),
                       "P25": policies[name]["P25"], "P50": policies[name]["P50"]}
    g.agree(summary["validation_budget_comparison"], budget, "complete paired validation budgets and ranks")
    policy_means = {}
    for policy in ("P25", "P50", "F"):
        row = {}
        for arm in ARMS:
            values = [test[f"{arm}_s{s}"][UPDATES.index(policies[f"{arm}_s{s}"][policy])] for s in SEEDS]
            row[arm] = {"seedwise": values, "mean": float(np.mean(values))}
        row["O-S_minus_D"] = row["O-S"]["mean"] - row["D"]["mean"]
        policy_means[policy] = row
    g.agree(summary["policy_means"], policy_means, "fixed-policy TEST contrasts")
    changes = {f"{policy}_minus_P25": policy_means[policy]["O-S_minus_D"] - policy_means["P25"]["O-S_minus_D"]
               for policy in ("P50", "F")}
    g.agree(summary["contrast_changes"], changes, "policy contrast changes")
    common = {}
    for i, update in enumerate(UPDATES):
        values = [test[f"O-S_s{s}"][i] - test[f"D_s{s}"][i] for s in SEEDS]
        common[str(update)] = {"seedwise": values, "mean": float(np.mean(values))}
    g.agree(summary["common_update_TEST_contrasts"], common, "all five common-update contrasts")
    final = {policy: {arm: [test[f"{arm}_s{s}"][-1] - test[f"{arm}_s{s}"][UPDATES.index(policies[f"{arm}_s{s}"][policy])]
                           for s in SEEDS] for arm in ARMS} for policy in ("P25", "P50")}
    g.agree(summary["final_minus_selected_TEST"], final, "per-seed final-minus-selected contrasts")
    return {"VAL25": old, "VAL50": val, "TEST50": test}


def validate_provenance(records, comparison, original, real_manifest, final, original_scores):
    """Bind the new grid to the released original arrays, selections and payloads."""
    p, scope, selected, saved, summary = (records[n + ".json"] for n in
                                       ("PROTOCOL", "SCOPE", "SELECTION", "SAVED_VAL25", "SUMMARY"))
    expected = {"schema_version": 2, "exposure": "historical_test_exposed", "study": "checkpoint_followup",
                "scope_sha256": SCOPE, "runner_sha256": RUNNER,
                "original_root": "original-study", "extension_root": "fixed-final-study",
                "source_root": "original-study/verification-source-02",
                "array_root": "original-study/artifacts/inventory", "original_protocol_sha256": ORIGINAL,
                "original_selection_sha256": ORIGINAL_SELECTION, "extension_protocol_sha256": EXTENSION,
                "extension_summary_sha256": EXTENSION_SUMMARY, "accepted_real_manifest_sha256": REAL_MANIFEST,
                "checkpoint_verifier_sha256": g.execution_source_hash("experiments/run_prigm_repair.py"),
                "execution_limits": {"child_seconds": 120, "stage_seconds": 3600},
                "arrays": original["arrays"], "training_data_hashes": original["training_data_hashes"],
                "controls": original["controls"],
                "samplers": {s: {"n_eval": 50, "n_draws": 39, "eval_seed": 7} for s in COUNTS},
                "P25": {n: comparison["selected"][n]["raw"] for n in NAMES}}
    core = {f"prigm/{path.name}": g.execution_source_hash(f"prigm/{path.name}")
            for path in sorted((g.REPO / "prigm").glob("*.py"))}
    require(len(core) == 43, "complete original source inventory")
    expected.update(training_source_hashes=core, inference_source_hashes=core)
    require(set(p) == {*expected, "environment", "jobs"}, "checkpoint protocol fields")
    g.agree({k: p[k] for k in expected}, expected, "original checkpoint protocol identities")
    g.agree(p["runner_sha256"], g.documentation.source_hash(g.REPO, "scripts/prigm_checkpoint_followup.py"), "frozen checkpoint runner")
    g.agree(original["source_protocol_sha256"], ORIGINAL, "public original protocol")
    g.agree(comparison["selection_sha256"], ORIGINAL_SELECTION, "public original selection")
    g.agree(final["protocol_sha256"], EXTENSION, "public fixed-final protocol")
    g.agree(final["ids"], comparison["ids"], "fixed-final TEST order")
    g.agree(final["arrays"], original["arrays"]["TEST"], "fixed-final TEST identity")
    require(p["environment"]["executable"] == "environment-python"
            and p["environment"]["hostname"] == "execution-host", "public environment aliases")
    # 'hostname' is an explicitly declared schema key, never an exported host.
    g.private_check({**p, "environment": {k: v for k, v in p["environment"].items() if k != "hostname"}})
    for record in (scope, selected, saved, summary):
        g.private_check(record)
    expected_scope = {
        "schema_version": 1, "study": "retrospective_checkpoint_dependence", "exposure": "historical_test_exposed",
        "arms": list(ARMS), "training_seeds": list(SEEDS), "milestones": list(UPDATES),
        "coefficients": {"D": 0., "O-S": .0005}, "no_training": True, "accepted_Table1_unchanged": True,
        "original_root": "original-study", "extension_root": "fixed-final-study",
        "original_protocol_sha256": ORIGINAL, "original_selection_sha256": ORIGINAL_SELECTION,
        "extension_summary_sha256": EXTENSION_SUMMARY,
        "policies": {"P25": "original_raw_VAL25_selection",
                     "P50": "minimum_complete_VAL50_month_balanced_ES_earlier_exact_tie", "F": "update_29636"},
        "scoring": "raw_only_60_primary_coordinates_month_then_seed_equal_weight",
        "report": "all_checkpoints_all_seeds_all_three_policies_no_TEST_selection",
        "test_gate": "all_30_hashes_and_P25_P50_F_frozen_after_complete_VAL50_before_new_TEST_sampling_or_score_inspection",
        "primary_contrasts": ["O-S_minus_D_P25", "O-S_minus_D_P50", "O-S_minus_D_F", "P50_minus_P25_contrast", "F_minus_P25_contrast"],
        "secondary_contrasts": ["O-S_minus_D_at_each_common_update", "final_minus_P25_each_arm_seed", "final_minus_P50_each_arm_seed"],
        "bootstrap": {"replicates": 10000, "seed": 20261001, "generator": "PCG64",
                      "unit": "paired_calendar_year_occurrences_with_month_multiplicity",
                      "scope": "fixed_models_noise_and_policies_not_training_or_selection_uncertainty"}}
    for split, count in COUNTS.items():
        expected_scope[split] = {"chunk_windows": 8, "jobs": 30, "n_windows": count,
                                 **expected["samplers"][split]}
    expected_scope["TEST"].update(authenticated_reuse_jobs=10, new_jobs=20)
    require(set(scope) == {*expected_scope, "created_utc", "gpu_admission"}, "checkpoint scope fields")
    g.agree({k: scope[k] for k in expected_scope}, expected_scope, "checkpoint scientific scope")
    g.agree([j["id"] for j in p["jobs"]], list(JOBS), "complete ordered checkpoint grid")
    original_jobs = {f"{name}-g{row['generation']}": row["checkpoint_sha256"]
                     for name in NAMES for row in comparison["selected"][name].values()}
    finals = {(row["arm"], row["seed"]): row for row in final["rows"]
              if row["arm"] in ARMS and row["c_r"] == {"D": 0., "O-S": .0005}[row["arm"]]}
    require(set(finals) == {(a, s) for a in ARMS for s in SEEDS}, "six original final checkpoints")
    for job, (name, update) in zip(p["jobs"], ((n, u) for n in NAMES for u in UPDATES)):
        arm, seed = name.rsplit("_s", 1)
        key, reuse = job["id"], None
        expected_job = {"id": f"{name}-g{update}", "arm": arm, "seed": int(seed), "generation": update,
                        "run_root": "original-study/runs/" + name}
        g.agree({k: job[k] for k in expected_job}, expected_job, "checkpoint job identity " + key)
        for field in ("checkpoint_sha256", "checkpoint_commit_sha256", "source_snapshot_sha256"):
            g.hash_map({field: job[field]})
        g.agree(job["source_snapshot_sha256"], final["original_inputs"]["checkpoints"][f"c{0 if arm == 'D' else 0.0005}_s{seed}"]["source_snapshot_sha256"],
                "original training archive " + key)
        require(set(job["val25"]) == {"path", "sha256"}, "original VAL25 score identity")
        g.agree(job["val25"]["path"], f"original-study/artifacts/val/{key}/score-raw.json", "original VAL25 path")
        g.hash_map({job["val25"]["path"]: job["val25"]["sha256"]})
        if key in original_jobs:
            g.agree(job["checkpoint_sha256"], original_jobs[key], "original selected weights " + key)
            prefix = f"artifacts/test/{key}/ensemble"
            reuse = {"root": "original-study/" + prefix, "study_root": "original-study", "protocol_sha256": ORIGINAL,
                     "commit_sha256": real_manifest["input_sha256"][prefix + "/gen_00000000/COMMIT.json"],
                     "payload_sha256": real_manifest["input_sha256"][prefix + "/gen_00000000/ensemble.npz"]}
        if update == UPDATES[-1]:
            row = finals[arm, int(seed)]
            bridge = final["original_inputs"]["checkpoints"][row["id"]]
            g.agree(job["checkpoint_sha256"], row["metadata"]["checkpoint_sha256"], "final checkpoint weights")
            g.agree(job["checkpoint_commit_sha256"], bridge["commit_sha256"], "final checkpoint commit")
            final_reuse = {"root": row["ensemble_root"].replace("extension-study/", "fixed-final-study/", 1),
                           "study_root": "original-study" if row["reuse"] else "fixed-final-study",
                           "protocol_sha256": row["protocol_sha256"], "commit_sha256": row["ensemble_commit_sha256"],
                           "payload_sha256": row["metadata"]["files"]["ensemble.npz"]}
            if reuse is not None:
                g.agree(reuse, final_reuse, "original/final reuse overlap")
            reuse = final_reuse
            g.agree(summary["scores"]["TEST"]["data"][key], [r["es5"] for r in row["results"]["raw"]["per_window"]],
                    "unchanged original final scores " + key)
        require(set(job) == {*expected_job, "checkpoint_sha256", "checkpoint_commit_sha256", "source_snapshot_sha256",
                             "val25", *(["reuse"] if reuse is not None else [])}, "checkpoint job fields " + key)
        if reuse is not None:
            g.agree(job["reuse"], reuse, "authenticated original payload " + key)
    require(sum("reuse" in j for j in p["jobs"]) == 10, "exact ten reused ensembles")
    for name in NAMES:
        arm, seed = name.rsplit("_s", 1)
        key = f"{name}-g{p['P25'][name]['generation']}"
        g.agree(summary["scores"]["TEST"]["data"][key], original_scores["data"][arm + "_raw"][SEEDS.index(int(seed))],
                "unchanged original P25 scores " + name)
    for split in COUNTS:
        ids = summary["scores"][split]["ids"]
        actual = hashlib.sha256(json.dumps(ids, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
        g.agree(actual, original["arrays"][split]["ids_sha256"], "original ordered " + split + " population")
    g.agree(summary["scores"]["TEST"]["ids"], comparison["ids"], "original TEST IDs")
    g.agree(summary["scores"]["TEST"]["months"], comparison["months"], "original TEST month order")
    require(set(saved) == {"protocol_sha256", "scope_sha256", "input_sha256", "ids", "months", "scores", "curves", "bootstrap"},
            "saved VAL25 fields")
    g.agree(saved["input_sha256"], {j["val25"]["path"]: j["val25"]["sha256"] for j in p["jobs"]}, "original VAL25 score pins")
    g.agree(summary["scope"], "Raw complete retrospective grid. No TEST selection or training-cause inference.", "summary scope")
    require(set(selected) == {"schema_version", "protocol_sha256", "scope_sha256", "policies", "validation_score_sha256",
                             "test_jobs", "test_checkpoint_hashes", "reuse_authentication_sha256", "selection"}, "selection fields")
    g.agree(selected["schema_version"], 2, "selection schema")
    for record in (saved, selected, summary):
        g.agree(record["protocol_sha256"], PROTOCOL, "checkpoint protocol binding")
    for record in (saved, selected):
        g.agree(record["scope_sha256"], SCOPE, "checkpoint scope binding")
    g.agree(summary["selection_sha256"], SELECTION, "frozen selection binding")
    g.agree(selected["reuse_authentication_sha256"], REUSE, "pre-TEST reuse authentication")
    g.agree(selected["selection"], "P50 solely complete raw VAL50; TEST covers complete predetermined grid", "VAL-only selection")
    g.agree(selected["test_jobs"], [[n, u] for n in NAMES for u in UPDATES], "complete predetermined TEST grid")
    hashes = [j["checkpoint_sha256"] for j in p["jobs"]]
    require(len(set(hashes)) == 30, "thirty distinct checkpoint weights")
    g.agree(selected["test_checkpoint_hashes"], sorted(hashes), "authorized checkpoint weights")
    inputs = summary["input_sha256"]
    g.hash_map(inputs)
    require(set(inputs) == {f"checkpoint-study/artifacts/{s.lower()}/{j}/score-raw.json" for s in COUNTS for j in JOBS}
            | {"checkpoint-study/SAVED_VAL25.json"}, "complete checkpoint input inventory")
    g.agree(inputs["checkpoint-study/SAVED_VAL25.json"], SAVED_VAL25, "saved VAL25 input")
    g.agree(selected["validation_score_sha256"], {k: v for k, v in inputs.items() if "/val/" in k}, "pre-TEST VAL50 score pins")


def validate_checkpoints(paper, real):
    """Validate public provenance and all reductions; return checked records."""
    paper = Path(paper)
    root = g.local(paper, "evidence/followup/checkpoints")
    g.agree(g.digest(root / "MANIFEST.json"), PUBLIC_MANIFEST, "independent checkpoint public manifest")
    manifest = g.read_json(root / "MANIFEST.json")
    g.private_check(manifest)
    require(set(manifest) == {"schema_version", "analysis", "source_protocol_sha256", "source_scope_sha256",
                             "source", "files", "original_sha256", "transformations"}, "checkpoint manifest fields")
    g.agree(manifest["schema_version"], 1)
    g.agree(manifest["analysis"], "checkpoint_followup_public_export")
    g.agree(manifest["source_protocol_sha256"], PROTOCOL)
    g.agree(manifest["source_scope_sha256"], SCOPE)
    g.agree(manifest["source"], {"path": "fig_checkpoints.py", "sha256": PUBLIC_EXPORTER}, "historical checkpoint exporter")
    require(set(manifest["files"]) == FILES, "five checkpoint public payloads")
    g.deadline_inventory(root, FILES | {"MANIFEST.json"})
    g.agree(manifest["original_sha256"], {"PROTOCOL.json": PROTOCOL, "SCOPE.json": SCOPE, "SELECTION.json": SELECTION,
                                         "SAVED_VAL25.json": SAVED_VAL25, "SUMMARY.json": SUMMARY}, "immutable checkpoint originals")
    transformations = manifest["transformations"]
    require(set(transformations) == {"root_replacements", "field_replacements"}, "declared checkpoint transformations")
    roots = transformations["root_replacements"]
    require(len(roots) == 3 and {r["public_root"] for r in roots} == {"checkpoint-study", "original-study", "fixed-final-study"},
            "three known public roots")
    for row in roots:
        require(set(row) == {"original_root_sha256", "public_root"}, "root replacement fields")
        g.hash_map({row["public_root"]: row["original_root_sha256"]})
    fields = transformations["field_replacements"]
    require(len(fields) == 2 and {r["field"] for r in fields} == {"environment.executable", "environment.hostname"},
            "two declared environment fields")
    for row in fields:
        require(set(row) == {"file", "field", "original_value_sha256", "public_value"}
                and row["file"] == "PROTOCOL.json", "environment replacement fields")
        g.agree(row["public_value"], {"environment.executable": "environment-python", "environment.hostname": "execution-host"}[row["field"]])
        g.hash_map({row["field"]: row["original_value_sha256"]})
    records = {}
    for name, pin in manifest["files"].items():
        path = g.local(root, name)
        g.agree(g.digest(path), pin, "checkpoint public payload " + name)
        records[name] = g.read_json(path)
    real_root, final_root = g.local(paper, "evidence/real"), g.local(paper, "evidence/deadline")
    g.agree(g.digest(real_root / "MANIFEST.json"), REAL_MANIFEST, "original public manifest")
    g.agree(g.digest(final_root / "MANIFEST.json"), DEADLINE_MANIFEST, "fixed-final public manifest")
    real_manifest, deadline = g.read_json(real_root / "MANIFEST.json"), g.read_json(final_root / "MANIFEST.json")
    original = {}
    for name in ("protocol-summary.json", "scores.json", "comparison.json"):
        g.agree(g.digest(real_root / name), real_manifest["output_sha256"][name], "original public " + name)
        original[name] = g.read_json(real_root / name)
    g.agree(real[0], original["comparison.json"], "supplied original comparison")
    g.agree(g.digest(final_root / "fixed-final.json"), deadline["files"]["fixed-final.json"], "fixed-final public payload")
    final = g.read_json(final_root / "fixed-final.json")
    validate_provenance(records, real[0], original["protocol-summary.json"], real_manifest, final, original["scores.json"])
    result = {name[:-5].lower(): record for name, record in records.items()}
    result["saved"] = result.pop("saved_val25")
    result["curves"] = validate_reductions(result["summary"], result["saved"], result["selection"], result["protocol"]["P25"])
    result["manifest"] = manifest
    return result


def validate_checkpoint_figure(paper, checkpoints):
    """Bind the plotted coordinates and policy markers to independent reductions."""
    from experiments._prigm_camera_ready import figure_record

    figure = figure_record(paper, "fig_checkpoints", {
        name: "evidence/followup/checkpoints/" + name for name in FILES | {"MANIFEST.json"}})
    require(set(figure) == {"schema_version", "figure", "protocol_sha256", "source", "inputs", "curves",
                            "policies", "caption", "matplotlib_version", "output_sha256",
                            "style", "evidence_source"}, "checkpoint figure fields")
    g.agree(figure["evidence_source"], checkpoints["manifest"]["source"],
            "checkpoint figure historical exporter")
    g.agree(figure["schema_version"], 1, "checkpoint figure schema")
    g.agree(figure["figure"], "fig_checkpoints", "checkpoint figure identity")
    g.agree(figure["protocol_sha256"], PROTOCOL, "checkpoint figure protocol")
    g.agree(figure["curves"], checkpoints["curves"], "independent checkpoint figure coordinates")
    g.agree(figure["policies"], checkpoints["selection"]["policies"], "independent checkpoint policy markers")
    return figure
