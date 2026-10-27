"""Read-only final gate for the PriGM paper and its numeric evidence (anonymous or camera-ready).

Usage: python experiments/verify_prigm_paper.py [--paper-dir DIRECTORY]
No incomplete mode. Checks the numerical tables in source and rendered PDF, selected
learned-toy, residual-weight and cancellation-group clauses, stationary and learned
toy evidence, public primary statistics, observation round trips, TEST strata,
TRAIN-cutoff sensitivity bindings, gradients and data audits. Does not certify
arbitrary prose, full TRAIN cutoff rederivation, gradients or ensemble regeneration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from zipfile import BadZipFile
from pathlib import Path

# Including trusted validators must not create bytecode in a supplied bundle.
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np

from analysis import export_prigm_paper_evidence as exporter
from analysis import prigm_followup_summary as diagnostics
from analysis import prigm_training_strata as sensitivity
from experiments import _prigm_documentation as documentation
from experiments._paper_tables import (
    assert_rendered, clean_cell, float_block, float_caption, strip_comments,
)
from prigm import run_bootstrap as bootstrap
from prigm.artifacts import id_hash, load_arrays
from prigm.recovery import ContractMismatch, PublicationError

PAPER = REPO / "paper/workshops/prigm"
STYLE_SHA256 = "67a960f35da6fecb5a0ce0202fa400459f4c8bb6c294f1a3fdd82fbbd39244c2"
# Fixed release review: original producer bytes and the eight accepted revisions.
ACCEPTED_SOURCE_MANIFEST_SHA256 = "a7bddc1dddf5a09ac1741c56dbad3f669a9c0afb73e1bf8b962764733d7d20bc"
NOTICE_ANONYMOUS = "Under review at the 2nd Workshop on Principles of Generative Modeling, NeurIPS 2026."
NOTICE_CAMERA_READY = ("40th Conference on Neural Information Processing Systems (NeurIPS 2026) "
                       "Workshop: 2nd Workshop on Principles of Generative Modeling (PriGM).")
NOTICE = NOTICE_ANONYMOUS  # back-compat for anonymous fixtures/tests
MILESTONES = [2868, 9560, 16252, 22944, 29636]
# Evidence and anonymous manuscripts forbid author identity; camera-ready has explicit exceptions.
FORBIDDEN_EVIDENCE = ("PISCES", "magnaprog", "Kevin Lee", "kevinlee", "skykev1928",
                      "Alison March", "colorado.edu", "ucla.edu", "Jet Propulsion", "orcid")
FORBIDDEN_ANONYMOUS = FORBIDDEN_EVIDENCE
FORBIDDEN_CAMERA_READY = ("PISCES", "magnaprog", "skykev1928", "Alison March", "colorado.edu")
FORBIDDEN = FORBIDDEN_ANONYMOUS  # default for evidence helpers
AUTHOR_CAMERA_READY = "Kevin Lee"
ORCID_CAMERA_READY = "0009-0004-0388-9260"
EMAIL_CAMERA_READY = "kevinlee69720@g.ucla.edu"
PUBLIC_COMPANION_URL = "https://github.com/magnaprog/diffusion-consistency"
EMAIL_ADDRESS = re.compile(r"[\w.+-]+@(?:[\w-]+\.)+[a-z]{2,}\b", re.I)
AFFILIATION_MARKERS = (
    "University of California at Los Angeles",
    "NASA Jet Propulsion Laboratory",
)
STYLE_ANONYMOUS = r"\usepackage[dblblindworkshop]{neurips_2025}"
STYLE_CAMERA_READY = r"\usepackage[final,sglblindworkshop]{neurips_2025}"
DRAFT = re.compile(r"\b(?:in\s+progress|not\s+ready|TODO|TBD)\b", re.I)
MAIN_ROWS = {"D": "D", "I-S": "I-S", "O-S": "O-S", "G5": "G5",
             "Interpolation": "interpolation", "Persistence": "persistence", "Gaussian": "gaussian"}
ORACLE_ROWS = {"Exact conditional draws": "exact_reference",
               "Two evaluations of exact denoiser": "two_step",
               **{f"Exact denoiser, {n} evaluations": f"ddim_{n}" for n in (25, 50, 100)}}
PAYLOADS = {"comparison.json", "diagnostics.json", "scores.json", "protocol-summary.json", "gradients.json"}
DETAIL_LABELS = tuple("tab:" + name for name in (
    "seeds", "contrasts", "calibration", "channelcoverage", "alternative",
    "geometry", "roundtrip", "strata", "years"))
TABLE_LABELS = {"tab:main", "tab:oracle", "tab:learnedtoy", *DETAIL_LABELS}
DATA_RECORDS = {"quality.json": "quality-01.json", "q15.json": "q15-raw-01.json",
                "inventory.json": "reconstructed_summary.json"}
DATA_DECLARATIONS = {
    "scope": "Anonymous copies of the complete reconstruction, data-quality and raw-vector aggregation records. No numerical values, populations or raw-year hashes were changed. The original records remain preserved separately.",
    "transformation": "For quality.json and q15.json only, replace each of the twenty raw_inputs[year].path values with its file basename. All other parsed content is identical. inventory.json is byte-identical to its source record.",
    "validation": "All twenty raw-year hashes agree across the three records; summed year counts reproduce eligible starts, candidate starts and all eligible Q15 blocks. This packaging check does not reread raw OMNI files or recreate their original execution provenance.",
}


class Report:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail=""):
        self.rows.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

    def ok(self):
        return bool(self.rows) and all(ok for _, ok, _ in self.rows)

    def check(self, name, function):
        try:
            value = function()
        except (ContractMismatch, PublicationError, OSError, ValueError, KeyError, TypeError, AttributeError, IndexError,
                ArithmeticError, BadZipFile, subprocess.SubprocessError) as exc:
            self.add(name, False, str(exc))
            return None
        self.add(name, True)
        return value


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local(root, name):
    """Never follow manuscript-controlled references outside the supplied paper."""
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts, "nonlocal publication path")
    target = (root / path).resolve()
    require(target.is_relative_to(root.resolve()), "publication path escapes paper directory")
    return target


def parse_json(payload):
    value = json.loads(payload, object_pairs_hook=bootstrap._unique_object)
    # Reject NaN/Infinity anywhere, including fields not currently used as claims.
    json.dumps(value, allow_nan=False)
    return value


def read_json(path):
    return parse_json(Path(path).read_bytes())


def agree(reported, actual, location="record"):
    exporter._agree(reported, actual, location)


def execution_source_hash(name):
    """Resolve frozen provenance only for the exact reviewed release revisions."""
    root = REPO / "reproducibility/accepted_source"
    manifest = local(root, "MANIFEST.json")
    payload = manifest.read_bytes()
    agree(hashlib.sha256(payload).hexdigest(), ACCEPTED_SOURCE_MANIFEST_SHA256, "accepted source manifest")
    current = documentation.source_hash(REPO, name)
    revision = parse_json(payload)["files"].get(name)
    if revision is None:
        return current
    agree(current, revision["accepted_sha256"], "reviewed current source " + name)
    agree(digest(local(root, name)), revision["execution_sha256"], "archived execution source " + name)
    return revision["execution_sha256"]


def private_check(value):
    # Per-window evidence repeats metric keys thousands of times. Scan each exact
    # string once, retaining every key/value and the same private-text rules.
    exporter._safe(list(set(exporter._strings(value))), {name.casefold() for name in FORBIDDEN})


def hash_map(record):
    require(isinstance(record, dict) and bool(record), "missing hash inventory")
    for name, value in record.items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts,
                "hash inventory is not study-relative")
        exporter._digest(value)


def validate_real(paper_dir):
    """Return validated public comparison/diagnostics, without private inputs."""
    root = local(paper_dir, "evidence/real")
    manifest = read_json(local(root, "MANIFEST.json"))
    private_check(manifest)
    agree(manifest["schema_version"], 1)
    agree(manifest["exporter_source_sha256"], documentation.source_hash(REPO, "analysis/export_prigm_paper_evidence.py"))
    analysis_hash = documentation.source_hash(REPO, "analysis/prigm_followup_summary.py")
    agree(manifest["analysis_source_sha256"], analysis_hash)
    require(set(manifest["output_sha256"]) == PAYLOADS, "requires five public payloads")
    records = {}
    for name in sorted(PAYLOADS):
        path = local(root, name)
        agree(digest(path), manifest["output_sha256"][name], name + " hash")
        records[name] = read_json(path)
        private_check(records[name])
    inputs, originals = manifest["input_sha256"], manifest["original_sha256"]
    hash_map(inputs)
    hash_map(originals)
    require(set(originals) == {"PROTOCOL.json", "PIPELINE_COMPLETED.json", "CPU_FOLLOWUP_COMPLETED.json",
                              exporter.COMPARISON, exporter.SUMMARY}, "incomplete original receipts")
    for name, sha in originals.items():
        agree(inputs[name], sha, name)
    comparison, summary, scores, protocol = [records[n] for n in
        ("comparison.json", "diagnostics.json", "scores.json", "protocol-summary.json")]
    for obj, version in ((comparison, 2), (summary, 1), (scores, 1), (protocol, 1)):
        agree(obj["schema_version"], version)
    ph = originals["PROTOCOL.json"]
    agree(protocol["source_protocol_sha256"], ph)
    for obj in (comparison, summary, scores):
        agree(obj["protocol_sha256"], ph)
    for key, filename in (("selection_sha256", "SELECTION.json"), ("test_started_sha256", "TEST_STARTED.json")):
        agree(comparison[key], inputs[filename], key)
        agree(summary[key], comparison[key], key)
    agree(comparison["source_comparison_sha256"], originals[exporter.COMPARISON])
    agree(manifest["output_sha256"]["diagnostics.json"], originals[exporter.SUMMARY])
    agree(summary["analysis_source_sha256"], analysis_hash)
    hash_map(summary["input_sha256"])
    for name, sha in summary["input_sha256"].items():
        agree(inputs[name], sha, name)
    for key, expected in {
        "arms": list(bootstrap.ARMS), "seeds": [7, 13, 42], "milestones": MILESTONES,
        "exposure": "historical_test_exposed", "study": "fixed_recipe_corrected_sampling_followup",
        "batch_size": 128, "budget_updates": 29636,
        "c_r": {"D": 0., "I-S": .0005, "O-S": .0005, "G5": 0.},
        "samplers": {"VAL": {"n_eval": 25, "n_draws": 39, "eval_seed": 7},
                     "TEST": {"n_eval": 50, "n_draws": 39, "eval_seed": 7}},
    }.items():
        agree(protocol[key], expected, "protocol." + key)
    require("selected independently" in protocol["selection"], "independent selection not declared")
    agree(protocol["controls"]["gaussian"]["rho"], 0.)
    identities = protocol["arrays"]
    require(set(identities) == {"TRAIN", "VAL", "TEST"}, "missing array identities")
    for split, identity in identities.items():
        require(set(identity) == set(diagnostics.ARRAY_FIELDS), "array identity fields differ")
        agree(identity["split"], split)
        for key in ("sha256", "ids_sha256", "normalization_sha256"):
            exporter._digest(identity[key])
        agree(identity["normalization_sha256"], identities["TRAIN"]["normalization_sha256"])
    agree(identities["TEST"]["n_windows"], 480)
    agree(identities["VAL"]["n_windows"], 319)
    agree(protocol["training_data_hashes"]["arrays_sha256"], identities["TRAIN"]["sha256"])
    agree(summary["arrays"], identities["TEST"])
    agree({k: comparison["arrays"][k] for k in diagnostics.ARRAY_FIELDS}, identities["TEST"])
    agree(summary["input_manifest"]["array_identities"], identities)
    ids, months = scores["ids"], np.asarray(scores["months"])
    require(len(ids) == 480 and len(set(ids)) == 480 and ids == sorted(ids), "requires 480 unique ordered IDs")
    diagnostics.source_codes(ids)
    require(all(re.fullmatch(r"(\d{4}-\d{2})-\1-\d{2}T\d{2}:\d{2}-s(?:50|52|60|71)", wid) for wid in ids),
            "invalid test window ID")
    timestamps = np.asarray([wid[8:].rsplit("-s", 1)[0] for wid in ids], dtype="datetime64[m]")
    require(np.all(np.diff(timestamps) > np.timedelta64(0, "m")), "unordered timestamps")
    agree(scores["months"], [int(w[:4]) * 100 + int(w[5:7]) for w in ids])
    require(np.isin(months // 100, bootstrap.YEARS).all() and np.isin(months % 100, (8, 9)).all(),
            "test calendar mismatch")
    agree(identities["TEST"]["ids_sha256"], id_hash(ids))
    for obj in (comparison, summary):
        agree(obj["ids"], ids)
        agree(obj["months"], scores["months"])
    agree(scores["fixed_training_seeds"], [7, 13, 42])
    data = {k: diagnostics._finite(v) for k, v in scores["data"].items()}
    diagnostics.validate_matrices(data, months, ids)
    require(set(comparison["methods"]) == set(summary["methods"]) == set(scores["row_ids"]) == set(data),
            "procedure inventories differ")
    require(set(comparison["selected"]) == {f"{a}_s{s}" for a in bootstrap.ARMS for s in (7, 13, 42)},
            "requires all twelve selected fits")
    for name, selected in comparison["selected"].items():
        require(set(selected) == {"raw", "corrected"}, "selection variants differ")
        for row in selected.values():
            require(type(row["generation"]) is int and row["generation"] in MILESTONES, "unknown milestone")
            exporter._digest(row["checkpoint_sha256"])
            diagnostics._finite(row["score"])
        if selected["raw"]["generation"] == selected["corrected"]["generation"]:
            agree(selected["raw"]["checkpoint_sha256"], selected["corrected"]["checkpoint_sha256"])
    for label, values in data.items():
        meta, diag = comparison["methods"][label], summary["methods"][label]
        arm, kind = label.split("_", 1)
        neural = arm in bootstrap.ARMS
        keys = ["7", "13", "42"] if neural else ["shared"]
        scoring_kind = "raw" if kind == "raw" else "corrected"
        selection_kind = "raw" if kind == "corrected_at_raw_checkpoint" else kind
        agree(scores["row_ids"][label], keys)
        agree(diag["n_realizations"], len(keys))
        require(set(diag["seedwise"]) == set(diag["files"]) == set(diag["checkpoint_sha256"]) == set(keys),
                "diagnostic seed inventory mismatch")
        require(set(meta["score_files"]) == set(keys), "score file seed inventory mismatch")
        for obj in (meta, diag):
            agree(obj["procedure"], "diffusion" if neural else arm)
            agree(obj["fixed_model_seeds"], [7, 13, 42] if neural else [])
            agree(obj["scoring_kind"], scoring_kind)
        agree(diag["selection_kind"], selection_kind if neural else None)
        if neural:
            agree(meta["selection_kind"], selection_kind)
            require(set(meta["generations"]) == set(keys), "generation seed inventory mismatch")
            agree(diag["generations"], meta["generations"])
        else:
            agree(meta["n_baseline_realizations"], 1)
        for i, key in enumerate(keys):
            file = diag["files"][key]
            agree(file["primary"], meta["score_files"][key])
            require(file["primary"] in inputs and file["secondaries"] in inputs, "unbound primary/secondary file")
            exporter._digest(file["ensemble_commit_sha256"])
            if neural:
                selected = comparison["selected"][f"{arm}_s{key}"][selection_kind]
                agree(meta["generations"][key], selected["generation"])
                agree(diag["checkpoint_sha256"][key], selected["checkpoint_sha256"])
            else:
                agree(diag["checkpoint_sha256"][key], None)
            actual = {k: v.tolist() for k, v in diagnostics.reductions(values[i], months).items()}
            agree(diag["seedwise"][key]["es5"], actual, label + ".seedwise.es5")
        metrics = diag["metrics"]
        require(set(diagnostics.SCALARS + diagnostics.POSITION_CHANNEL).issubset(metrics), "incomplete diagnostic metrics")
        for key in keys:
            require(set(diag["seedwise"][key]) == set(metrics), "seedwise metric inventory mismatch")
        for metric, reduction in metrics.items():
            shape = (12, 5) if metric in diagnostics.POSITION_CHANNEL else () if metric in diagnostics.SCALARS else None
            for key in keys:
                require(set(diag["seedwise"][key][metric]) == {"mean", "month_balanced"}, "secondary reductions differ")
                for value in diag["seedwise"][key][metric].values():
                    values_array = diagnostics._finite(value)
                    if shape is not None:
                        require(values_array.shape == shape, "secondary scalar/array shape differs")
            expected = {w: np.mean([diagnostics._finite(diag["seedwise"][k][metric][w]) for k in keys], axis=0).tolist()
                        for w in ("mean", "month_balanced")}
            agree(reduction, expected, label + "." + metric)
        agree(metrics["es5"], {k: v.tolist() for k, v in diagnostics.reductions(values.mean(axis=0), months).items()})
    agree(summary["interval_protocol"], diagnostics.INTERVAL_PROTOCOL)
    expected = {"exposure": "historical_test_exposed", "n_replicates": bootstrap.BOOTSTRAP_REPLICATES,
                "seed": bootstrap.BOOTSTRAP_SEED, "fixed_training_seeds": [7, 13, 42],
                "multiplicity": "with_replacement_year_blocks", "year_blocks": bootstrap.year_blocks().tolist(),
                "two_year_blocks": bootstrap.two_year_blocks(),
                "month_counts": {str(y * 100 + m): int(np.count_nonzero(months == y * 100 + m))
                                 for y in bootstrap.YEARS for m in bootstrap.TEST_MONTHS}, **exporter.COMPARISON_TEXT}
    for key, actual in expected.items():
        agree(comparison[key], actual, key)
    for key, actual in bootstrap.compare(data, months).items():
        agree(comparison[key], actual, "recomputed " + key)
    gradients = records["gradients.json"]
    validate_gradients(gradients, inputs, protocol)
    for seed in (7, 13, 42):
        for selected in comparison["selected"][f"O-S_s{seed}"].values():
            generation = str(selected["generation"])
            if generation in gradients["grids"]:
                agree(selected["checkpoint_sha256"],
                      gradients["grids"][generation]["request"]["checkpoint_sha256"][str(seed)],
                      "selected/probed checkpoint identity")
    return comparison, summary


# External pins identify the frozen deadline producers and original public bytes.
# They do not replace any original real-evidence manifest or its source hashes.
DEADLINE_PROTOCOL = "25d0cdefdd5dbb160c32c7b5f4a6053bab99e4a9f64832d9eb6502232bf24e6b"
FIXED_FINAL_PROTOCOL = "1408cd52f306af0095f808dfe1d34aa8493fb26db9a4e75dae422839157bc3b4"
DEADLINE_PRODUCERS = {
    "analysis/prigm_operator_mechanism.py": "009c809d5be8486baa191c08a32f67f9af1c77148e4c79662a9d05fe303588aa",
    "analysis/prigm_full_flank_gaussian.py": "8df68e33b9f0cc3d797be4531195a1a6c192c6f00cb3bf9c5edc5c77c67a333b",
    "experiments/run_prigm_fixed_final_sensitivity.py": "b731d1a5ccf962e048b15255eab17554ee8cebcf30b677cff8655a5defe1618a",
}
DEADLINE_ORIGINALS = {
    "operator-mechanism.json": "9b68a8fa5a61a24bdd537080ac62b44c6b56e47b63119430eb7f7dfa30b45008",
    "gaussian/FIT_SELECT.json": "5ce6ea010bcf77b02663b2ef3ce573d35e5ccbf09b19a0b143dd3e859f0b156c",
}
DEADLINE_TRANSFORMATIONS = {
    "operator-mechanism.json": "byte-identical",
    "gaussian": "byte-identical selected evidence; execution locks and LATEST omitted",
    "fixed-final.json": "host paths replaced by original-study/ or extension-study/ prefixes; no numerical changes",
}
FIXED_COEFFICIENTS = ("0", "0.00005", "0.0005", "0.005")
FIXED_METRICS = ("es5", "r_obs_mean", "frac_support_inequality_violation",
                 "frac_support_violation_beyond_rounding", "cov90_mean", "width90_mean")
FIXED_SELECTION = "predetermined_final_update_29636_not_VAL_selected"
FIXED_FINAL_ORIGINAL_SHA256 = "be795841a48d9ef1864c7adcbeb52e499e25a3bd6cc8a60909c5fbd557884b34"
FIXED_FINAL_PUBLIC_SHA256 = "c740c4cacf19b38623d565e34ae0c7a7aa2111e870d82d4a7911b4029c94c078"


def deadline_claimed(paper_dir, tex=None):
    if tex is None:
        path = local(paper_dir, "main.tex")
        tex = path.read_text() if path.is_file() else ""
    return ((paper_dir / "evidence/deadline").exists() or (paper_dir / "evidence/deadline").is_symlink()
            or bool(re.search(r"fig_operator|Full[ -]flank Gaussian|Gaussian\s+regression|tab:(?:operators|fixedfinal)", strip_comments(tex), re.I)))


def deadline_numeric(values):
    """Reject Boolean leaves before NumPy can coerce mixed lists to floats."""
    pending = [values]
    while pending:
        value = pending.pop()
        if isinstance(value, (list, tuple)):
            pending.extend(value)
        else:
            require(not isinstance(value, (bool, np.bool_)), "Boolean value is not a numerical metric")
            if isinstance(value, np.ndarray):
                require(value.dtype.kind in "iuf", "nonnumeric metric array")
    return diagnostics._finite(values)


def deadline_close(reported, actual, label="deadline", *, atol=1e-12):
    """Shape/type strict, with scale-aware tolerances for SI energy proxies."""
    if isinstance(actual, dict):
        require(isinstance(reported, dict) and set(reported) == set(actual), label + ": fields differ")
        for key in actual:
            deadline_close(reported[key], actual[key], label + "." + key,
                           atol=1e-24 if "proxy" in key else atol)
    elif isinstance(actual, (list, tuple)) and not all(type(v) in (int, float) for v in actual):
        require(isinstance(reported, (list, tuple)) and len(reported) == len(actual), label + ": sequence differs")
        for i, (left, right) in enumerate(zip(reported, actual)):
            deadline_close(left, right, f"{label}[{i}]", atol=atol)
    elif isinstance(actual, (np.ndarray, list, tuple)) or type(actual) is float:
        left, right = deadline_numeric(reported), deadline_numeric(actual)
        require(left.shape == right.shape and np.allclose(left, right, rtol=1e-10, atol=atol),
                label + ": numerical recomputation differs")
    else:
        agree(reported, actual, label)


def deadline_inventory(root, names):
    """Exact files AND directories, no aliases, symlinks, locks or partial outputs."""
    names = set(names)
    expected_dirs = {p.as_posix() for name in names for p in Path(name).parents if p != Path(".")}
    require(root.is_dir() and not root.is_symlink(), "missing deadline evidence directory")
    paths = list(root.rglob("*"))
    require(not any(p.is_symlink() for p in paths), "deadline symlink is not a public payload")
    require({p.relative_to(root).as_posix() for p in paths if p.is_file()} == names
            and {p.relative_to(root).as_posix() for p in paths if p.is_dir()} == expected_dirs,
            "deadline inventory contains missing, extra or partial evidence")


def deadline_sources(sources, expected_names):
    hash_map(sources)
    require(set(sources) == set(expected_names), "deadline producer inventory differs")
    for name, sha in sources.items():
        agree(sha, documentation.source_hash(REPO, name), "recorded deadline source " + name)
        if name in DEADLINE_PRODUCERS:
            agree(sha, DEADLINE_PRODUCERS[name], "frozen deadline source " + name)


def deadline_reductions(record, months, shapes, *, summary_key="summary"):
    """Validate per-window metrics and recompute their summary reductions."""
    rows = record["per_window"]
    require(isinstance(rows, list) and len(rows) == len(months) and bool(rows)
            and all(isinstance(row, dict) and set(row) == set(shapes) for row in rows),
            "deadline per-window metric inventory differs")
    values = {}
    for key, shape in shapes.items():
        values[key] = deadline_numeric([row[key] for row in rows])
        require(values[key].shape == (len(months), *shape), "deadline metric shape differs: " + key)
        if "fraction" in key or key.startswith(("frac_", "cov")):
            require(np.all((values[key] >= 0) & (values[key] <= 1)), "fraction outside [0,1]: " + key)
    reduced = {key: {w: v.tolist() for w, v in diagnostics.reductions(value, months).items()}
               for key, value in values.items()}
    deadline_close(record[summary_key], reduced, "deadline reductions", atol=0.)
    return values


def validate_operator_windows(record, months):
    """Verify algebra, domains and reductions, not unshipped draw-level margins."""
    from analysis import prigm_operator_mechanism as op
    shapes = {f"{kind}_{term}": () for kind in ("raw", "completed", "corrected")
              for term in ("target_term", "dispersion_term", "energy_score")}
    shapes.update({"delta_" + key: () for key in ("target_term", "dispersion_term", "energy_score")})
    shapes.update({key: () for key in ("displacement_bound", "bound_slack", "bound_arithmetic_tolerance",
                                     "mean_movement", "completed_absolute_equality_residual_mean")})
    shapes["movement_quantiles"] = (5,)
    for kind in ("raw", "corrected"):
        for margin in op.MARGINS:
            shapes.update({f"{kind}_{margin}_{key}": (5,) for key in ("signed_quantiles", "severity_quantiles")})
            shapes.update({f"{kind}_{margin}_{key}": () for key in ("violation_fraction", "beyond_rounding_fraction")})
        shapes.update({f"{kind}_{key}": () for key in ("support_violation_fraction", "support_beyond_rounding_fraction",
                                                     "absolute_equality_residual_mean")})
    values = deadline_reductions(record, months, shapes, summary_key="metrics")
    for key, array in values.items():
        if key.endswith("quantiles"):
            require(np.all(np.diff(array, axis=1) >= 0), "unordered mechanism quantiles")
        if any(token in key for token in ("severity", "movement", "absolute_equality", "target_term", "dispersion_term")) and not key.startswith("delta_"):
            require(np.all(array >= 0), "negative mechanism magnitude: " + key)
    require(np.all(values["mean_movement"] >= values["movement_quantiles"][:, 0])
            and np.all(values["mean_movement"] <= values["movement_quantiles"][:, -1]), "movement mean outside extrema")
    for row in record["per_window"]:
        terms = {kind: {term: row[f"{kind}_{term}"] for term in ("target_term", "dispersion_term", "energy_score")}
                 for kind in ("raw", "completed", "corrected")}
        require(terms["raw"] == terms["completed"], "equality completion changed primary energy terms")
        bound = op.check_displacement_bound(terms["raw"], terms["corrected"], row["mean_movement"])
        deadline_close({key: row[key] for key in bound}, bound, "operator displacement bound", atol=0.)
        tol = bound["bound_arithmetic_tolerance"]
        for kind, term in terms.items():
            require(abs(term["energy_score"] - (term["target_term"] - term["dispersion_term"])) <= tol,
                    "operator energy-component identity differs")
        for term in terms["raw"]:
            require(abs(row["delta_" + term] - (terms["corrected"][term] - terms["raw"][term])) <= tol,
                    "operator paired delta differs")
        require(abs(row["delta_energy_score"] - row["delta_target_term"] + row["delta_dispersion_term"]) <= tol,
                "operator delta decomposition differs")
        require(row["completed_absolute_equality_residual_mean"] == 0
                and row["corrected_support_beyond_rounding_fraction"] == 0, "recorded completion or feasibility differs")
        for kind in ("raw", "corrected"):
            for margin in op.MARGINS:
                require(row[f"{kind}_{margin}_beyond_rounding_fraction"] <= row[f"{kind}_{margin}_violation_fraction"],
                        "rounding failures exceed strict failures")
                signed, severity = (row[f"{kind}_{margin}_{k}_quantiles"] for k in ("signed", "severity"))
                deadline_close([severity[0], severity[-1]], [max(-signed[-1], 0.), max(-signed[0], 0.)],
                               "margin/severity extrema", atol=0.)
            for suffix, combined in (("violation_fraction", "support_violation_fraction"),
                                     ("beyond_rounding_fraction", "support_beyond_rounding_fraction")):
                fractions = [row[f"{kind}_{m}_{suffix}"] for m in op.MARGINS]
                union = row[f"{kind}_{combined}"]
                rounding = 4 * np.finfo(np.float64).eps
                require(max(fractions) - rounding <= union <= min(1., sum(fractions)) + rounding,
                        "support union fraction differs")
    return values


def validate_operator(record, real, observations, protocol, inputs, core, scores):
    from analysis import prigm_operator_mechanism as op
    comparison, summary = real
    _, ids, nrm, identity = observations
    months = np.asarray(summary["months"])
    private_check(record)
    json.dumps(record, allow_nan=False)
    for key, value in {"schema_version": 1, "analysis": "paired_operator_mechanism",
                       "protocol_sha256": protocol["source_protocol_sha256"], "ids": ids,
                       "months": months.tolist(), "n_windows": 480, "arrays": identity,
                       "selection_sha256": summary["selection_sha256"], "test_started_sha256": summary["test_started_sha256"],
                       "core_source_sha256": core, "quantiles": list(op.QUANTILES), "quantile_method": "linear",
                       "month_counts": {str(m): int((months == m).sum()) for m in np.unique(months)},
                       "normalization": {"mean": nrm.mean.tolist(), "scale": nrm.scale.tolist(),
                                         "b_ref_sq": nrm.b_ref_sq, "n_bins": nrm.n_bins}}.items():
        agree(record[key], value, "operator." + key)
    deadline_sources(record["producer_source_sha256"], ("analysis/prigm_operator_mechanism.py", "analysis/prigm_followup_summary.py"))
    agree(record["input_manifest"]["array_identities"], protocol["arrays"])
    agree(record["input_manifest"]["array_root"], "artifacts/inventory")
    hashes = record["input_sha256"]
    hash_map(hashes)
    for name, sha in hashes.items():
        agree(sha, inputs[name], "original operator input " + name)
    required = {"PROTOCOL.json", "SELECTION.json", "TEST_STARTED.json",
                "artifacts/inventory/test.npz", "artifacts/inventory/train.npz", "artifacts/controls_test/gaussian_fit.npz"}
    for name in ("PROTOCOL.json", "SELECTION.json", "TEST_STARTED.json"):
        agree(hashes[name], inputs[name])
    require(set(record["methods"]) == set(bootstrap.ARMS + bootstrap.CONTROLS), "operator method inventory differs")

    def realization(row, name, seed, kind):
        label = name + "_" + kind
        diag = summary["methods"][label]
        neural = name in bootstrap.ARMS
        index = ("7", "13", "42").index(seed) if neural else 0
        selected = comparison["selected"][f"{name}_s{seed}"][kind] if neural else None
        primary = diag["files"][seed]
        parent = Path(primary["primary"]).parent
        ensemble = parent / "ensemble" if neural else parent
        files = {"primary": primary["primary"], "secondary": primary["secondaries"],
                 "ensemble_commit": (ensemble / "gen_00000000/COMMIT.json").as_posix(),
                 "ensemble_payload": (ensemble / "gen_00000000/ensemble.npz").as_posix()}
        for key, value in {"procedure": "diffusion" if neural else name, "selection_kind": kind,
                           "checkpoint_sha256": selected["checkpoint_sha256"] if selected else None,
                           "generation": selected["generation"] if selected else None,
                           "control_fit_sha256": primary["control_fit_sha256"], "files": files,
                           "stored_dtype": "float32" if neural else "float64", "n_draws": 39,
                           "verified_primary_scoring_kinds": ["corrected"] if kind == "corrected" else ["raw", "corrected"]}.items():
            agree(row[key], value, "operator realization " + key)
        required.update(files.values())
        agree(hashes[files["ensemble_commit"]], primary["ensemble_commit_sha256"])
        values = validate_operator_windows(row, months)
        kinds = ["corrected"] if kind == "corrected" else ["raw", "corrected"]
        for scoring in kinds:
            original = label if kind == "corrected" or scoring == "raw" else (
                name + "_corrected_at_raw_checkpoint" if neural else name + "_corrected")
            diagnostics._agree(values[scoring + "_energy_score"], scores["data"][original][index], "original per-window energy")
            # These secondary reductions were already verified against the original record.
            for new, old in (("support_violation_fraction", "frac_support_inequality_violation"),
                             ("support_beyond_rounding_fraction", "frac_support_violation_beyond_rounding"),
                             ("m_minus_r_violation_fraction", "frac_m_lt_r"), ("vs_violation_fraction", "frac_vs_lt0"),
                             ("vs_upper_minus_vs_violation_fraction", "frac_vs_gt_hi"),
                             ("absolute_equality_residual_mean", "r_obs_mean")):
                for weighting in ("mean", "month_balanced"):
                    value = row["metrics"][scoring + "_" + new][weighting]
                    if old == "r_obs_mean":
                        value *= nrm.b_ref_sq
                    deadline_close(value, summary["methods"][original]["seedwise"][seed][old][weighting], "original support reduction", atol=0.)
        return values

    for name, method in record["methods"].items():
        seeds = ("7", "13", "42") if name in bootstrap.ARMS else ("shared",)
        require(set(method) == {"metrics", "seedwise", "n_realizations"} and set(method["seedwise"]) == set(seeds),
                "operator seed inventory differs")
        agree(method["n_realizations"], len(seeds))
        for seed in seeds:
            realization(method["seedwise"][seed], name, seed, "raw")
        deadline_close(method["metrics"], op._seed_average(method["seedwise"]), "operator seed mean", atol=0.)
    changed = {s for s in ("7", "13", "42") if any(
        comparison["selected"][f"D_s{s}"]["raw"][k] != comparison["selected"][f"D_s{s}"]["corrected"][k]
        for k in ("generation", "checkpoint_sha256"))}
    require(set(record["D_additional_corrected_selected"]) == changed, "D additional selection inventory differs")
    decomposition = record["D_selection_decomposition"]
    require(set(decomposition["seedwise"]) == {"7", "13", "42"}, "D selection requires three seeds")
    for i, seed in enumerate(("7", "13", "42")):
        if seed in changed:
            realization(record["D_additional_corrected_selected"][seed], "D", seed, "corrected")
        chosen = comparison["selected"][f"D_s{seed}"]
        expected = op.selection_decomposition(*[scores["data"][label][i] for label in
                    ("D_raw", "D_corrected_at_raw_checkpoint", "D_corrected")], months)
        expected.update(raw_generation=chosen["raw"]["generation"], corrected_generation=chosen["corrected"]["generation"],
                        raw_checkpoint_sha256=chosen["raw"]["checkpoint_sha256"],
                        corrected_checkpoint_sha256=chosen["corrected"]["checkpoint_sha256"], selection_changed=seed in changed)
        deadline_close(decomposition["seedwise"][seed], expected, "D selection decomposition")
    deadline_close(decomposition["metrics"], op._seed_average(decomposition["seedwise"]), "D selection seed mean", atol=0.)
    # Include every original raw/corrected score and secondary, including aliases.
    for method in summary["methods"].values():
        for row in method["files"].values():
            required.update((row["primary"], row["secondaries"]))
    require(set(hashes) == required, "operator original input inventory differs")
    return record


def validate_gaussian_fit(root, protocol, core, receipt_hash):
    """Check stored TRAIN moments/coefficients and VAL selection, without refitting."""
    from analysis import prigm_full_flank_gaussian as fg
    receipt = read_json(root / "FIT_SELECT.json")
    agree(digest(root / "FIT_SELECT.json"), receipt_hash, "Gaussian fit-select hash")
    hashes = {name: digest(root / name) for name in fg.FIT_FILES}
    agree(receipt, {"schema_version": 1, "phase": "FIT_SELECTED",
                    "source_protocol_sha256": protocol["source_protocol_sha256"], "files": hashes}, "FIT_SELECT receipt")
    contract = read_json(root / "EXTENSION.json")
    expected = {
        "schema_version": 1, "study": "full_flank_linear_gaussian_extension", "scope": fg.SCOPE,
        "source_protocol_sha256": protocol["source_protocol_sha256"], "arrays": protocol["arrays"],
        "producer_source_hashes": {"analysis/prigm_full_flank_gaussian.py": DEADLINE_PRODUCERS["analysis/prigm_full_flank_gaussian.py"]},
        "core_source_hashes": core, "numpy_version": contract["numpy_version"],
        "model": {
            "conditioning": "216 standardized observed coordinates: bins [0,18) + [30,48), all six channels",
            "target": "60 standardized primary residuals about interpolation from bins 17 and 30",
            "objective": "||Y-XW-intercept||_F^2/N + lambda*||W||_F^2; unpenalized intercept",
            "training": "TRAIN-only centering and population residual covariance; equal weight per saved overlapping window",
            "lambdas": [0.001, 0.1, 1.0], "rhos": [0.0, 0.5, 1.0], "jitter": 1e-6,
            "covariance": "(1-rho)*population_cov + rho*diag(population_cov) + jitter*I; fixed Cholesky",
            "representation": "joint Gaussian in first five coordinates only; derive physical v_v; preserve observed flanks",
            "chunk_windows": 4096,
        },
        "noise": fg.NOISE_CONTRACT,
        "selection": "independent raw/corrected minimum month-balanced VAL primary energy; exact ties lexicographic (lambda,rho)",
        "interval": "39 draws; inclusive 1-based order statistics 2 and 38; mean over 60 standardized primary coordinates",
        "recomputation_tolerance": {"rtol": 1e-10, "atol": 1e-12},
    }
    require(isinstance(contract["numpy_version"], str) and bool(contract["numpy_version"]), "missing producer NumPy version")
    agree(contract, expected, "Gaussian scientific contract")
    deadline_sources(contract["producer_source_hashes"], ("analysis/prigm_full_flank_gaussian.py",))
    agree(contract["noise"]["n_draws"], 39)
    agree(read_json(root / "FIT.json"), {
        "schema_version": 1, "extension_protocol_sha256": hashes["EXTENSION.json"],
        "training_arrays": protocol["arrays"]["TRAIN"], "fit_payload": "FIT.npz",
        "fit_payload_sha256": hashes["FIT.npz"], "n_mean_fits": 3, "n_covariance_configurations": 9,
    }, "Gaussian FIT metadata")
    shapes = {"n_train": (), "x_mean": (216,), "y_mean": (60,), "cxx": (216, 216), "cxy": (216, 60),
              "cyy": (60, 60), "weights": (3, 216, 60), "intercepts": (3, 60), "residual_covariances": (3, 60, 60)}
    with np.load(root / "FIT.npz", allow_pickle=False) as blob:
        require(len(blob.files) == len(shapes) and set(blob.files) == set(shapes), "Gaussian fit array inventory differs")
        fit = {key: blob[key] for key in shapes}
    for key, array in fit.items():
        require(array.shape == shapes[key] and array.dtype == (np.int64 if key == "n_train" else np.float64)
                and np.isfinite(array).all(), "Gaussian coefficient shape/dtype/finite mismatch: " + key)
    agree(fit["n_train"].item(), protocol["arrays"]["TRAIN"]["n_windows"], "Gaussian TRAIN count")
    for key in ("cxx", "cyy"):
        deadline_close(fit[key], fit[key].T, "symmetric population moment")
    joint = np.block([[fit["cxx"], fit["cxy"]], [fit["cxy"].T, fit["cyy"]]])
    require(np.linalg.eigvalsh(joint).min() >= -1e-10 * max(1., np.linalg.norm(joint, ord=2)),
            "stored population covariance is not positive semidefinite")
    stats = {key: fit[key] for key in ("n_train", "x_mean", "y_mean", "cxx", "cxy", "cyy")}
    recomputed = fg.fit_from_statistics(stats)
    for key in ("weights", "intercepts", "residual_covariances"):
        deadline_close(fit[key], recomputed[key], "Gaussian sufficient-statistic formula " + key)
    validation = read_json(root / "VALIDATION.json")
    for key, value in {"schema_version": 1, "extension_protocol_sha256": hashes["EXTENSION.json"],
                       "fit_payload_sha256": hashes["FIT.npz"], "arrays": protocol["arrays"]["VAL"]}.items():
        agree(validation[key], value, "Gaussian VAL " + key)
    val_ids = validation["ids"]
    require(len(val_ids) == protocol["arrays"]["VAL"]["n_windows"] and val_ids == sorted(val_ids), "incomplete ordered VAL IDs")
    agree(id_hash(val_ids), protocol["arrays"]["VAL"]["ids_sha256"], "VAL ID hash")
    for row in validation["candidates"]:
        require(set(row) == {"lambda", "rho", "scores"} and set(row["scores"]) == {"raw", "corrected"}, "VAL row fields differ")
        for value in row["scores"].values():
            require(diagnostics._finite(value).shape == (len(val_ids),), "VAL score shape differs")
    selected = fg.select_grid(validation)
    selection = read_json(root / "SELECTION.json")
    agree(selection, {"schema_version": 1, "extension_protocol_sha256": hashes["EXTENSION.json"],
                      "fit_payload_sha256": hashes["FIT.npz"], "validation_sha256": hashes["VALIDATION.json"],
                      "selected": selected, "selected_union": [fg.candidate_name(*p) for p in fg.selected_pairs({"selected": selected})]},
          "Gaussian month-balanced VAL selection")
    return contract, fit, selection, hashes


def validate_gaussian(root, protocol, core, observations, real, receipt_hash):
    from analysis import prigm_full_flank_gaussian as fg
    from prigm.observation import residual_obs
    contract, fit, selection, hashes = validate_gaussian_fit(root, protocol, core, receipt_hash)
    z, ids, nrm, identity = observations
    require(len(ids) == 480 and len(set(ids)) == 480, "Gaussian TEST requires complete 480 IDs")
    agree(identity, protocol["arrays"]["TEST"])
    months = fg.months_for(ids).tolist()
    phase = {"schema_version": 1, "phase": "TEST_STARTED", "previous_test_exposure": True,
             "source_protocol_sha256": protocol["source_protocol_sha256"], "extension_protocol_sha256": hashes["EXTENSION.json"],
             "fit_select_sha256": receipt_hash, "fit_payload_sha256": hashes["FIT.npz"],
             "selection_sha256": hashes["SELECTION.json"], "arrays": identity, "selected_union": selection["selected_union"]}
    agree(read_json(root / "test/TEST_STARTED.json"), phase, "Gaussian TEST phase")
    record = read_json(root / "test/SCORES.json")
    header = {"schema_version": 1, "scope": fg.SCOPE, "fit_select_sha256": receipt_hash,
              "source_protocol_sha256": protocol["source_protocol_sha256"], "extension_protocol_sha256": hashes["EXTENSION.json"],
              "selection_sha256": hashes["SELECTION.json"], "fit_payload_sha256": hashes["FIT.npz"], "arrays": identity,
              "ids": ids, "months": months, "n": 480, "n_draws": 39, "complete_case": True,
              "selected": selection["selected"], "interval": contract["interval"]}
    agree({k: v for k, v in record.items() if k != "candidates"}, header, "Gaussian TEST metadata")
    require(set(record["candidates"]) == set(selection["selected_union"]), "Gaussian selected TEST union differs")
    factors = fg.candidate_factors(fit)
    completions = ["test/TEST_STARTED.json", "test/SCORES.json"]
    selected_metrics, selected_scores = {}, {}
    for pair in fg.selected_pairs(selection):
        name = fg.candidate_name(*pair)
        path = local(root, f"test/{name}")
        payload = path / "gen_00000000/ensemble.npz"
        with np.load(payload, allow_pickle=False) as blob:
            require(len(blob.files) == 2 and set(blob.files) == {"ids", "ensemble"}, "Gaussian ensemble NPZ inventory differs")
        meta = fg.ensemble_metadata(contract, hashes, selection, receipt_hash, pair)
        # read_ensemble pins the COMMIT payload digest; this helper checks flanks and v_v.
        ensemble = fg.verify_selected_ensemble(path, meta, z, ids, nrm)
        i = fg.LAMBDAS.index(pair[0])
        for stored, truth, wid in zip(ensemble, z, ids):
            generated = fg.draw_window(truth, nrm, fit["weights"][i], fit["intercepts"][i], factors[pair], fg.standard_noise("TEST", wid))
            deadline_close(stored, generated, "regenerated Gaussian TEST draw")
        calculated = fg.test_scores(ensemble, z, ids, nrm)
        commit = f"test/{name}/gen_00000000/COMMIT.json"
        completions.append(commit)
        expected = {"lambda": pair[0], "rho": pair[1], "selected_for": meta["selected_for"],
                    "ensemble": f"test/{name}", "ensemble_commit_sha256": digest(root / commit), **calculated}
        deadline_close(record["candidates"][name], expected, "Gaussian all TEST metrics")
        for kind in ("raw", "corrected"):
            values = deadline_reductions(record["candidates"][name][kind], months,
                                        {key: () for key in calculated[kind]["summary"]})
            require(np.all(values["width90"] >= 0), "negative Gaussian interval width")
            if kind == "corrected":
                require(np.all(values["support_violation_beyond_rounding_fraction"] == 0), "Gaussian correction is infeasible")
        if "raw" in meta["selected_for"]:
            selected_metrics["correction_displacement"] = calculated["correction_displacement"]["summary"]["mean_primary_l2"]
            residuals = [float(np.abs(residual_obs(fg.decode_ensemble(stored, nrm))).mean()) for stored in ensemble]
            selected_metrics["raw_r_obs_mean"] = {w: v.tolist() for w, v in diagnostics.reductions(residuals, months).items()}
        for kind in meta["selected_for"]:
            selected_metrics[kind] = calculated[kind]["summary"]
            selected_scores[kind] = [row["primary_es"] for row in calculated[kind]["per_window"]]
        del ensemble
    agree(read_json(root / "test/COMPLETED.json"), {
        "schema_version": 1, "phase": "TEST_COMPLETED", "fit_select_sha256": receipt_hash,
        "n_windows": 480, "n_draws": 39, "selected_union": selection["selected_union"],
        "files": {name: digest(root / name) for name in completions}}, "Gaussian completion receipt")
    ranks = {}
    for kind in ("raw", "corrected"):
        ranks[kind] = {}
        for new, old in (("primary_es", "es5"), ("vector_energy", "es_vec"), ("resolved_energy_proxy_crps", "proxy_crps")):
            original = {label: row["metrics"][old]["month_balanced"] for label, row in real[1]["methods"].items()}
            require(len(original) == 18, "Gaussian ranking requires all 18 original procedures")
            value = selected_metrics[kind][new]["month_balanced"]
            ranks[kind][new] = {"rank_among_original_18_plus_selected": 1 + sum(v < value for v in original.values()),
                                "original_18": original, "selected_value": value}
    return {"selection": selection, "metrics": selected_metrics, "ranks": ranks, "scores": selected_scores}


def validate_fixed_final(record, real, observations, protocol, inputs, core, operator=None):
    """Full twelve-row reduction and identity gate; never load neural weights."""
    from prigm.artifacts import NOISE_CONVENTION, STATE_CONVENTION
    private_check(record)
    json.dumps(record, allow_nan=False)
    _, ids, nrm, identity = observations
    months = [int(w[:4]) * 100 + int(w[5:7]) for w in ids]
    for key, value in {"schema_version": 2, "protocol_sha256": FIXED_FINAL_PROTOCOL, "selection": FIXED_SELECTION,
                       "arrays": identity, "ids": ids, "months": months, "complete_grid": True}.items():
        agree(record[key], value, "fixed-final " + key)
    require(len(ids) == 480, "fixed-final needs all 480 TEST IDs")
    exporter._digest(record["selection_sha256"])
    deadline_sources(record["producers"], ("experiments/run_prigm_fixed_final_sensitivity.py", "experiments/run_prigm_repair.py"))
    jobs = {}
    for coefficient in FIXED_COEFFICIENTS:
        for seed in (7, 13, 42):
            name = f"c{coefficient}_s{seed}"
            arm = "D" if coefficient == "0" else "O-S"
            new = coefficient in ("0.00005", "0.005")
            reuse = (coefficient, seed) in (("0", 7), ("0.0005", 7), ("0.0005", 13))
            run = f"extension-study/runs/{name}" if new else f"original-study/runs/{arm}_s{seed}"
            parent = f"original-study/artifacts/test/{arm}_s{seed}-g29636" if reuse else f"extension-study/artifacts/test/{name}"
            jobs[name] = {"id": name, "arm": arm, "seed": seed, "c_r": float(coefficient), "generation": 29636,
                          "new_fit": new, "reuse": reuse, "run_root": run, "ensemble_root": parent + "/ensemble"}
    rows = record["rows"]
    require(isinstance(rows, list) and len(rows) == 12 and [row["id"] for row in rows] == list(jobs),
            "fixed-final requires ordered complete coefficient/seed grid")
    require(set(record["endpoints"]) == set(jobs), "fixed-final endpoint inventory differs")
    bridge = record["original_inputs"]
    agree(bridge["receipts"], {name: inputs[name] for name in ("PROTOCOL.json", "SELECTION.json", "TEST_STARTED.json")})
    require(set(bridge) == {"receipts", "checkpoints", "ensembles"}
            and set(bridge["checkpoints"]) == {name for name, job in jobs.items() if not job["new_fit"]}
            and set(bridge["ensembles"]) == {name for name, job in jobs.items() if job["reuse"]}, "fixed-final original bridge incomplete")
    snapshots = set()
    for row in rows:
        job = jobs[row["id"]]
        for key, value in job.items():
            agree(row[key], value, "fixed-final job " + key)
        endpoint = record["endpoints"][row["id"]]
        require(set(endpoint) == {"run_root", "generation", "checkpoint_sha256", "commit_sha256", "source_snapshot_sha256"},
                "fixed-final checkpoint binding inventory differs")
        agree(endpoint["run_root"], job["run_root"])
        agree(endpoint["generation"], 29636)
        for key in ("checkpoint_sha256", "commit_sha256", "source_snapshot_sha256"):
            exporter._digest(endpoint[key])
        snapshots.add(endpoint["source_snapshot_sha256"])
        if not job["new_fit"]:
            agree(bridge["checkpoints"][job["id"]], endpoint, "fixed-final original endpoint")
            old_run = job["run_root"].removeprefix("original-study/")
            for name, key in (("gen_00029636/state.pt", "checkpoint_sha256"), ("gen_00029636/COMMIT.json", "commit_sha256"),
                              ("source_snapshot.zip", "source_snapshot_sha256")):
                if old_run + "/" + name in inputs:
                    agree(endpoint[key], inputs[old_run + "/" + name], "original fixed-final checkpoint hash")
        own_protocol = protocol["source_protocol_sha256"] if job["reuse"] else FIXED_FINAL_PROTOCOL
        agree(row["protocol_sha256"], own_protocol)
        exporter._digest(row["ensemble_commit_sha256"])
        meta = row["metadata"]
        expected_meta = {"schema_version": 2, "generation": 0, "protocol_sha256": own_protocol,
                         "checkpoint_sha256": endpoint["checkpoint_sha256"], "generation_updates": 29636,
                         "arm": job["arm"], "seed": job["seed"], "n_generated": 6, "n_params": 1603398,
                         "arrays": identity, "ids_sha256": id_hash(ids), "sampler": protocol["samplers"]["TEST"],
                         "state_convention": STATE_CONVENTION, "noise_convention": NOISE_CONVENTION,
                         "inference_source_sha256": core["prigm/run_eval.py"]}
        for key, value in expected_meta.items():
            agree(meta[key], value, "fixed-final ensemble " + key)
        require(set(meta) == {*expected_meta, "device", "torch_version", "files", "chunks"}, "fixed-final ensemble metadata inventory differs")
        require(all(isinstance(meta[k], str) and bool(meta[k]) for k in ("device", "torch_version")), "missing sampler environment")
        require(set(meta["files"]) == {"ensemble.npz"}, "fixed-final ensemble payload inventory differs")
        hash_map(meta["files"])
        chunks = meta["chunks"]
        hash_map(chunks)
        parent = Path(job["ensemble_root"]).parent.as_posix()
        require(set(chunks) == {f"{parent}/chunk-{offset:04d}" for offset in range(0, 480, 8)}, "fixed-final complete chunk inventory differs")
        if job["reuse"]:
            agree(bridge["ensembles"][job["id"]], {"ensemble_root": job["ensemble_root"], "commit_sha256": row["ensemble_commit_sha256"],
                                                  "metadata": meta, "protocol_sha256": own_protocol}, "reused ensemble bridge")
            original_parent = job["ensemble_root"].removeprefix("original-study/")
            agree(row["ensemble_commit_sha256"], inputs[original_parent + "/gen_00000000/COMMIT.json"])
            agree(meta["files"]["ensemble.npz"], inputs[original_parent + "/gen_00000000/ensemble.npz"])
            for name, sha in chunks.items():
                # Original exported evidence may bind the merged payload rather than individual chunks.
                key = name.removeprefix("original-study/") + "/gen_00000000/COMMIT.json"
                if key in inputs:
                    agree(sha, inputs[key], "original reused chunk hash")
        require(set(row) == {*job, "protocol_sha256", "metadata", "ensemble_commit_sha256", "results"}
                and set(row["results"]) == {"raw", "corrected"}, "fixed-final row fields differ")
        for kind in ("raw", "corrected"):
            variant = row["results"][kind]
            require(set(variant) == {"summary", "per_window"}, "fixed-final metric fields differ")
            values = deadline_reductions(variant, months, {key: () for key in FIXED_METRICS})
            require(np.all(values["width90_mean"] >= 0) and np.all(values["r_obs_mean"] >= 0)
                    and np.all(values["frac_support_violation_beyond_rounding"] <= values["frac_support_inequality_violation"]),
                    "fixed-final metric domains differ")
            if kind == "corrected":
                require(np.all(values["frac_support_violation_beyond_rounding"] == 0), "fixed-final corrected feasibility differs")
            selected = real[0]["selected"][f"{job['arm']}_s{job['seed']}"][kind]
            if selected["generation"] == 29636 and not job["new_fit"]:
                agree(endpoint["checkpoint_sha256"], selected["checkpoint_sha256"])
                original = real[1]["methods"][job["arm"] + "_" + kind]["seedwise"][str(job["seed"])]
                deadline_close(variant["summary"], {k: original[k] for k in FIXED_METRICS}, "original fixed-final reductions", atol=0.)
            if job["reuse"] and operator is not None:
                seed = str(job["seed"])
                source = (operator["D_additional_corrected_selected"].get(seed, operator["methods"]["D"]["seedwise"][seed])
                          if job["arm"] == "D" else operator["methods"]["O-S"]["seedwise"][seed])
                for key, metric, factor in (("es5", "energy_score", 1.), ("r_obs_mean", "absolute_equality_residual_mean", nrm.b_ref_sq),
                                           ("frac_support_inequality_violation", "support_violation_fraction", 1.),
                                           ("frac_support_violation_beyond_rounding", "support_beyond_rounding_fraction", 1.)):
                    expected = np.asarray([r[kind + "_" + metric] for r in source["per_window"]]) * factor
                    deadline_close(values[key], expected, "reused per-window fixed-final metric", atol=0.)
    require(len(snapshots) == 1, "fixed-final source snapshots differ across the fixed recipe")
    require(set(record) == {"schema_version", "protocol_sha256", "selection", "selection_sha256", "endpoints",
                            "original_inputs", "producers", "arrays", "ids", "months", "complete_grid", "rows"}, "fixed-final summary fields differ")
    return record


def validate_operator_figure(paper_dir, operator_hash):
    # Never import Python from the supplied anonymous bundle.
    from paper.workshops.prigm import fig_operator
    path = local(paper_dir, "evidence/deadline/operator-mechanism.json")
    rows, selection = fig_operator.load_rows(path, operator_hash)
    root = local(paper_dir, "figures/operator")
    manifest = read_json(root / "fig_operator.manifest.json")
    # These literal plot labels are not host paths or private control text.
    private_check([s.replace("Boundary\nGaussian", "Boundary Gaussian")
                   .replace("Gaussian\ninterpolation", "Gaussian interpolation")
                   .replace("within +/-0.005", "within plus or minus 0.005")
                   for s in exporter._strings(manifest)])
    for key, value in {"schema_version": 1, "figure": "fig_operator", "protocol_sha256": DEADLINE_PROTOCOL,
                       "source": {"path": "fig_operator.py", "sha256": digest(PAPER / "fig_operator.py")},
                       "inputs": {"operator": {"path": "operator-mechanism.json", "sha256": operator_hash}},
                       "plotted_rows": rows, "D_selection": selection}.items():
        require(manifest[key] == value, "operator figure coordinate/source/input differs: " + key)
    require(set(manifest["output_sha256"]) == {"fig_operator.pdf", "fig_operator.png"}, "operator figure outputs incomplete")
    deadline_inventory(root, {"fig_operator.manifest.json", *manifest["output_sha256"]})
    for name, sha in manifest["output_sha256"].items():
        file = local(root, name)
        agree(digest(file), sha, "operator figure output hash")
        require(file.read_bytes().startswith(b"%PDF-" if name.endswith(".pdf") else b"\x89PNG\r\n\x1a\n"), "operator figure header differs")
    return rows, selection


def validate_deadline(paper_dir, real, observations):
    """Validate the public extension; absence is allowed only without any claim.

    Operator/neural quantities: hashes, algebra and reductions, not sample replay.
    Gaussian: stored-moment formulas, VAL reduction/selection and all TEST draws and
    scores. TRAIN and VAL arrays are not released, so this is not a TRAIN refit or
    an independent proof that execution obeyed the recorded TRAIN-only fit.
    """
    paper_dir = Path(paper_dir)
    if not deadline_claimed(paper_dir):
        return None
    root = local(paper_dir, "evidence/deadline")
    manifest = read_json(root / "MANIFEST.json")
    private_check(manifest)
    require(set(manifest) == {"schema_version", "source_protocol_sha256", "files", "original_sha256", "producers", "transformations"},
            "deadline manifest fields differ")
    agree(manifest["schema_version"], 1)
    agree(manifest["source_protocol_sha256"], DEADLINE_PROTOCOL)
    protocol = read_json(local(paper_dir, "evidence/real/protocol-summary.json"))
    original_manifest = read_json(local(paper_dir, "evidence/real/MANIFEST.json"))
    inputs = original_manifest["input_sha256"]
    agree(protocol["source_protocol_sha256"], DEADLINE_PROTOCOL, "original deadline protocol")
    for name in ("protocol-summary.json", "scores.json", "gradients.json"):
        agree(digest(local(paper_dir, "evidence/real/" + name)), original_manifest["output_sha256"][name], "original public hash")
    core = {p.relative_to(REPO).as_posix(): execution_source_hash(p.relative_to(REPO).as_posix())
            for p in (REPO / "prigm").glob("*.py")}
    require(len(core) == 43, "frozen core source inventory differs")
    gradients = read_json(local(paper_dir, "evidence/real/gradients.json"))
    for grid in gradients["grids"].values():
        for report in grid["reports"].values():
            agree(report["current_source_hashes"], core, "original inference sources")
            agree(report["training_source_hashes"], core, "original training sources")
    files = manifest["files"]
    hash_map(files)
    fixed = "fixed-final.json" in files
    originals = {*DEADLINE_ORIGINALS, *(("fixed-final.json",) if fixed else ())}
    require(set(manifest["original_sha256"]) == originals, "deadline original inventory differs")
    for name, expected in DEADLINE_ORIGINALS.items():
        agree(manifest["original_sha256"][name], expected, "frozen deadline original " + name)
        agree(files[name], expected, "byte-identical deadline payload " + name)
    if fixed:
        agree(manifest["original_sha256"]["fixed-final.json"], FIXED_FINAL_ORIGINAL_SHA256, "audited fixed-final original")
        agree(files["fixed-final.json"], FIXED_FINAL_PUBLIC_SHA256, "audited fixed-final public payload")
    expected_producers = {"analysis/prigm_operator_mechanism.py", "analysis/prigm_full_flank_gaussian.py"}
    if fixed:
        expected_producers.add("experiments/run_prigm_fixed_final_sensitivity.py")
    optional_producers = {"analysis/prigm_followup_summary.py", "experiments/run_prigm_repair.py"}
    require(expected_producers <= set(manifest["producers"]) <= expected_producers | optional_producers, "deadline producer inventory differs")
    deadline_sources(manifest["producers"], manifest["producers"])
    agree(manifest["transformations"], {key: value for key, value in DEADLINE_TRANSFORMATIONS.items() if fixed or key != "fixed-final.json"})
    from analysis import prigm_full_flank_gaussian as fg
    selection = read_json(root / "gaussian/SELECTION.json")
    selected_names = [fg.candidate_name(*pair) for pair in fg.selected_pairs(selection)]
    expected_files = {"operator-mechanism.json", *("gaussian/" + name for name in (*fg.FIT_FILES, "FIT_SELECT.json")),
                      *("gaussian/test/" + name for name in ("TEST_STARTED.json", "SCORES.json", "COMPLETED.json")),
                      *(f"gaussian/test/{name}/gen_00000000/{file}" for name in selected_names for file in ("COMMIT.json", "ensemble.npz"))}
    if fixed:
        expected_files.add("fixed-final.json")
    require(set(files) == expected_files, "deadline required payload inventory differs")
    deadline_inventory(root, expected_files | {"MANIFEST.json"})
    for name, sha in files.items():
        agree(digest(local(root, name)), sha, "deadline payload hash " + name)
        if name.endswith(".json"):
            record = read_json(root / name)
            # These literal algebra operators are not execution paths. Do not
            # exempt arbitrary slashes or alter the original evidence bytes.
            strings = [s.replace(" + noise @ factor.T", " + noise times factor.T")
                       .replace("/N + lambda", " divided by N + lambda")
                       for s in exporter._strings(record)]
            private_check(strings)
    scores = read_json(local(paper_dir, "evidence/real/scores.json"))
    operator = validate_operator(read_json(root / "operator-mechanism.json"), real, observations, protocol, inputs, core, scores)
    gaussian = validate_gaussian(root / "gaussian", protocol, core, observations, real, DEADLINE_ORIGINALS["gaussian/FIT_SELECT.json"])
    months = np.asarray(real[1]["months"])
    weights = bootstrap.bootstrap_weights(months)
    statistics = {}
    for label, values in {**{arm: scores["data"][arm + "_raw"] for arm in bootstrap.ARMS},
                          "full_flank_gaussian": [gaussian["scores"]["raw"]]}.items():
        monthly = bootstrap.monthly_means(values, months)
        statistics[label] = {"seed_means": monthly.mean(axis=1),
                             **{scheme: weight @ monthly.mean(axis=0) for scheme, weight in weights.items()}}
    gaussian["comparisons"] = {arm: bootstrap.paired_summary(statistics[arm], statistics["full_flank_gaussian"])
                               for arm in bootstrap.ARMS}
    fixed_record = validate_fixed_final(read_json(root / "fixed-final.json"), real, observations, protocol, inputs, core, operator) if fixed else None
    plotted, selection_effects = validate_operator_figure(paper_dir, files["operator-mechanism.json"])
    # Detect changes during numerical verification as well as before it.
    for name, sha in files.items():
        agree(digest(root / name), sha, "deadline input changed during verification")
    return {"operator": operator, "gaussian": gaussian, "fixed_final": fixed_record,
            "plotted_rows": plotted, "D_selection": selection_effects}


def validate_gradients(gradients, inputs, protocol):
    """Public identity/geometry checks only; no checkpoints, raw arrays or gradients loaded."""
    private_check(gradients)
    require(set(gradients) == {"schema_version", "protocol_sha256", "scope", "grids"}, "gradient payload fields differ")
    agree(gradients["schema_version"], 1)
    agree(gradients["protocol_sha256"], protocol["source_protocol_sha256"])
    agree(gradients["scope"], exporter.GRADIENT_SCOPE)
    require(set(gradients["grids"]) == {"2868", "29636"}, "requires early/final gradient grids")
    expected_reports = {f"O-S_s{s}-{split}.json" for s in (7, 13, 42) for split in ("train", "val")}
    cases = {(t, e, r) for t in (1, 25, 50, 100) for e in ("M", "S") for r in ("inst", "obs")}
    shared_source, shared_selection = None, {}

    def number(value, low=0, high=float("inf")):
        require(type(value) in (int, float) and np.isfinite(value) and low <= value <= high,
                "gradient number outside its finite domain")
        return value

    for generation, grid in gradients["grids"].items():
        require(set(grid) == {"request", "source_request_sha256", "source_completion_sha256", "reports"},
                "gradient grid fields differ")
        parent = f"artifacts/gradient-g{generation}"
        for kind in ("request", "completion"):
            filename = "REQUEST.json" if kind == "request" else "COMPLETED.json"
            exporter._digest(grid[f"source_{kind}_sha256"])
            agree(grid[f"source_{kind}_sha256"], inputs[f"{parent}/{filename}"], "original gradient " + kind)
        request = grid["request"]
        for key, value in {"protocol_sha256": gradients["protocol_sha256"], "generation": int(generation),
                           "arm": "O-S", "seeds": [7, 13, 42], "splits": ["TRAIN", "VAL"],
                           "n_windows_per_split": 8, "noise_seed": 7003, "c_r": .0005}.items():
            agree(request[key], value, "gradient.request." + key)
        agree(request["launcher_sha256"], inputs["run_probe_grid.py"])
        require(set(request["checkpoint_sha256"]) == {"7", "13", "42"}, "gradient checkpoint inventory differs")
        require(set(grid["reports"]) == expected_reports, "requires all six gradient reports")
        for filename, report in grid["reports"].items():
            # This hash identifies the ORIGINAL report. The output payload hash
            # protects the reduced copy; path reduction intentionally changes bytes.
            exporter._digest(inputs[f"{parent}/{filename}"])
            seed, split_lower = re.fullmatch(r"O-S_s(7|13|42)-(train|val)\.json", filename).groups()
            split = split_lower.upper()
            checkpoint_sha = request["checkpoint_sha256"][seed]
            exporter._digest(checkpoint_sha)
            agree(checkpoint_sha, inputs[f"runs/O-S_s{seed}/gen_{int(generation):08d}/state.pt"])
            for key, value in {"schema_version": 1, "diagnostic": "weighted_physics_data_parameter_gradients",
                               "generation_updates": int(generation), "checkpoint_sha256": checkpoint_sha,
                               "c_r": .0005, "arrays": protocol["arrays"][split],
                               "arrays_path": f"artifacts/inventory/{split_lower}.npz",
                               "run_root": f"runs/O-S_s{seed}"}.items():
                agree(report[key], value, "gradient.report." + key)
            agree(inputs[report["arrays_path"]], report["arrays"]["sha256"])
            contract = report["training_contract"]
            for key, value in {"arm": "O-S", "seed": int(seed), "c_r": .0005, "contract_version": 2,
                               "estimator": "S", "residual": "obs", "n_generated": 6,
                               "batch_size": 128, "max_updates": 29636,
                               "n_examples": protocol["arrays"]["TRAIN"]["n_windows"],
                               "data_hashes": protocol["training_data_hashes"],
                               "normalization_hash": report["arrays"]["normalization_sha256"]}.items():
                agree(contract[key], value, "gradient.training." + key)
            for key, value in {"class": "prigm.model.ConditionalUNet", "widths": contract["widths"],
                               "n_generated": 6, "mode": "eval"}.items():
                agree(report["model"][key], value, "gradient.model." + key)
            require(type(report["model"]["n_parameters"]) is int and report["model"]["n_parameters"] > 0,
                    "invalid gradient parameter count")
            hash_map(report["current_source_hashes"])
            hash_map(report["training_source_hashes"])
            agree(report["training_source_hashes"], contract["source_hashes"])
            if shared_source is None:
                shared_source = report["current_source_hashes"]
            agree(report["current_source_hashes"], shared_source)
            nrm = report["normalization"]
            agree(nrm["sha256"], report["arrays"]["normalization_sha256"])
            require(len(nrm["mean"]) == len(nrm["scale"]) == 6, "gradient normalization shape differs")
            for value in nrm["mean"]:
                number(value, -float("inf"))
            require(all(number(v) > 0 for v in nrm["scale"]) and number(nrm["b_ref_sq"]) > 0
                    and type(nrm["n_bins"]) is int and nrm["n_bins"] > 0, "invalid gradient normalization")
            selected = report["selection"]
            for key, value in {"n_requested": 8, "n_selected": 8,
                               "rule": "np.linspace(0, n_available - 1, min(n_available, n_windows), dtype=int64); first index if count is one",
                               "indices": np.linspace(0, report["arrays"]["n_windows"] - 1, 8, dtype=np.int64).tolist(),
                               "ids_sha256": id_hash(selected["ids"])}.items():
                agree(selected[key], value, "gradient.selection." + key)
            require(len(selected["ids"]) == len(set(selected["ids"])) == 8, "gradient selected IDs differ")
            if split not in shared_selection:
                shared_selection[split] = selected
            agree(selected, shared_selection[split], "shared fixed gradient windows")
            for key, value in {"seed": 7003, "shape": [8, 6, 48], "device": "cpu",
                               "dtype": "torch.float32", "generator": "torch.Generator"}.items():
                agree(report["noise"][key], value)
            exporter._digest(report["noise"]["sha256"])
            for key in ("u0_sha256", "observed_sha256"):
                exporter._digest(report["inputs"][key])
            agree(report["inputs"]["hidden_slice"], [18, 30])
            agree(report["inputs"]["state_convention"], "clean_observed_flanks_hidden_diffusion")
            for key, value in contract["settings"]["diffusion"].items():
                agree(report["schedule"][key], value)
            hash_map(report["schedule"]["array_hashes"])
            agree(report["schedule"]["probe_timesteps"], [1, 25, 50, 100])
            require(set(report["schedule"]["probe_values"]) == {"a", "s", "w", "sigma"}, "gradient schedule fields differ")
            for values in report["schedule"]["probe_values"].values():
                require(len(values) == 4, "gradient schedule shape differs")
                for value in values:
                    number(value)
            rows = report["records"]
            require(len(rows) == 16, "requires all sixteen gradient cases")
            seen = set()
            for row in rows:
                case = (row["t"], row["estimator"], row["residual"])
                require(type(row["t"]) is int and case in cases and case not in seen, "gradient Cartesian cases differ")
                seen.add(case)
                for key in ("weighted_data_mean", "weighted_physics_mean", "total_mean",
                            "data_gradient_norm", "physics_gradient_norm"):
                    number(row[key])
                data_norm, phys_norm = row["data_gradient_norm"], row["physics_gradient_norm"]
                for key, undefined, low, high in (
                    ("physics_to_data_norm_ratio", data_norm == 0, 0, float("inf")),
                    ("gradient_cosine", data_norm == 0 or phys_norm == 0, -1, 1),
                    ("gradient_angle_degrees", data_norm == 0 or phys_norm == 0, 0, 180),
                ):
                    if undefined:
                        agree(row[key], None, "undefined gradient geometry")
                    else:
                        number(row[key], low, high)
                if data_norm:
                    agree(row["physics_to_data_norm_ratio"], phys_norm / data_norm)
                if data_norm and phys_norm:
                    agree(row["gradient_angle_degrees"], float(np.degrees(np.arccos(row["gradient_cosine"]))))
                require(np.isclose(row["total_mean"], row["weighted_data_mean"] + row["weighted_physics_mean"],
                                   rtol=2e-7, atol=1e-12), "gradient weighted losses do not sum at float32 precision")


def validate_toy(paper_dir):
    # This is ALWAYS repository code, never paper_dir/fig_hmm.py.
    from paper.workshops.prigm import fig_hmm
    from prigm.normalize import AffineNormalizer
    from prigm.oracle import infer_context, oracle_metadata

    root = local(paper_dir, "evidence/toy")
    manifest = read_json(local(paper_dir, "figures/hmm/fig_hmm.manifest.json"))
    private_check(manifest)
    agree(manifest["schema_version"], 1)
    agree(manifest["figure"], "fig_hmm")
    agree(manifest["source"], {"path": "fig_hmm.py", "sha256": digest(PAPER / "fig_hmm.py")})
    records = {}
    for name in ("hmm", "oracle"):
        filename = f"{name}-stationary-01.json"
        spec = manifest["inputs"][name]
        agree(spec["path"], filename)
        records[name], _ = fig_hmm.load_evidence(local(root, filename), spec["sha256"])
        private_check(records[name])
        agree(spec["source_hashes"], records[name]["source_hashes"])
    hmm, oracle = records["hmm"], records["oracle"]
    toy_path = local(root, "toy.npz")
    with np.load(toy_path, allow_pickle=False) as blob:
        require(len(blob.files) == len(set(blob.files)), "duplicate NPZ arrays")
        agree(str(blob["boundary_law"].item()), "stationary")
        agree(str(blob["split"].item()), "TOY")
        expected = oracle_metadata(toy_path, "pooled", "stationary", blob)
        expected["source_hashes"] = {name: execution_source_hash("prigm/" + name)
                                     for name in expected["source_hashes"]}
        for record in (hmm, oracle):
            for key, value in expected.items():
                agree(record[key], value, "toy." + key)
        z, p, alpha = blob["z"], blob["p"], blob["alpha"]
        require(z.shape == (288, 48, 6) and p.shape == alpha.shape == (288,), "toy population shape mismatch")
        require(np.isfinite(z).all() and np.isfinite(p).all(), "nonfinite toy arrays")
        agree(blob["ids"].tolist(), [f"TOY:{i}" for i in range(288)])
        require(blob["mean"].shape == blob["scale"].shape == (6,) and np.isfinite(blob["mean"]).all()
                and np.isfinite(blob["scale"]).all() and (blob["scale"] > 0).all(), "invalid toy transform")
        b_ref, n_bins = blob["b_ref_sq"].item(), blob["n_bins"].item()
        require(type(n_bins) is int and n_bins > 0 and np.isfinite(b_ref) and b_ref > 0,
                "invalid toy normalization population")
        nrm = AffineNormalizer(blob["mean"], blob["scale"], float(b_ref), n_bins)
        contexts = []
        for i, window in enumerate(z):
            a_value, a, b = infer_context(window, nrm)
            require(a_value == alpha[i], "toy alpha array differs from observed flanks")
            contexts.append({"window": i, "alpha": a_value, "generating_p": float(p[i]), "a": a, "b": b})
        agree(oracle["contexts"], contexts)
        from prigm.toy import ALPHAS, PS
        require(all(np.count_nonzero((alpha == a) & (p == prob)) == 32 for a in ALPHAS for prob in PS),
                "toy requires 32 windows in each of nine regimes")
    context, samplers, reference, shared, residuals = fig_hmm.validate_evidence(hmm, oracle)
    regimes = {(float(prob), float(a)): i for i, (prob, a) in
               enumerate((prob, a) for prob in PS for a in ALPHAS)}
    for row in hmm["context"]:
        agree(row["regime"], regimes[(row["generating_p"], row["alpha"])], "HMM regime identity")
    agree(hmm["n_paths"], 128)
    require(set(oracle["methods"]) == set(ORACLE_ROWS.values()), "unknown oracle method")
    for key, value in {
        "input_schema_version": 2, "toy_eval_sha256": digest(toy_path), "boundary_law": "stationary",
        "conditioning": "pooled", "hmm_n_context": 144, "hmm_n_paths": 128,
        "verified_common_source_hashes": {name: hmm["source_hashes"][name] for name in shared},
        "plotted_rows": {"panel_a_context": context, "panel_b_samplers": samplers,
                         "panel_b_horizontal_reference": reference},
        "unplotted_context_mean_abs_r_star": residuals,
    }.items():
        # Figure point values are copied exactly, not rounded within a tolerance.
        require(manifest[key] == value, "figure manifest differs: " + key)
    require(set(manifest["output_sha256"]) == {"fig_hmm.pdf", "fig_hmm.png"}, "figure outputs incomplete")
    for name, sha in manifest["output_sha256"].items():
        agree(digest(local(paper_dir, "figures/hmm/" + name)), sha, name)
    means = {key: float(np.mean(row["scores"])) for key, row in oracle["methods"].items()}
    return means, hmm["context_by_t"]["100"]


def table_text(text):
    """Normalize typography, never rounding, signs, digits or interval punctuation."""
    for old, new in (("\u2013", "-"), ("\u2212", "-"), ("--", "-"),
                     ("\u00d7", "x"), ("\ufb00", "ff"), ("\ufb01", "fi"),
                     ("\ufb02", "fl"), ("^", ""), ("_", "")):
        text = text.replace(old, new)
    return text.lower()


def visible(text):
    return re.sub(r"\s+", "", table_text(text))


def source_table_rows(tex, label):
    """Ordered cells, including composite arm/seed keys and nested column specs."""
    block = float_block(tex, label)
    require(block.count(r"\begin{tabular}") == block.count(r"\end{tabular}") == 1,
            label + ": requires one tabular")
    start = block.index(r"\begin{tabular}") + len(r"\begin{tabular}")
    while block[start].isspace():
        start += 1
    require(block[start] == "{", "missing column specification")
    depth, end = 1, start + 1
    while depth:
        depth += (block[end] == "{") - (block[end] == "}")
        end += 1
    body = block[end:block.index(r"\end{tabular}")]
    body = re.sub(r"\\(?:toprule|midrule|bottomrule)\b|\\cmidrule(?:\([lr]+\))?\{[^}]*\}", "", body)
    return [[clean_cell(cell) for cell in row.split("&")]
            for row in body.split(r"\\") if row.strip()]


def rendered_table(tex, rendered, label):
    """Use the unique numbered caption, not incidental row labels elsewhere."""
    labels = re.findall(r"\\label\{(tab:[^}]+)\}", strip_comments(tex))
    require(labels.count(label) == 1, "ambiguous table label")
    number = labels.index(label) + 1
    text = pdf_without_line_numbers(rendered)
    captions = list(re.finditer(r"(?m)^[ \t]*Table[ \t]+(\d+):", text))
    matches = [m for m in captions if int(m[1]) == number]
    require(len(matches) == 1, label + ": PDF numbered caption missing or duplicated")
    start = matches[0].end()
    end = re.search(r"(?m)^[ \t]*(?:Table|Figure)[ \t]+\d+:", text[start:])
    return text[start:start + end.start()] if end else text[start:]


def join_pdf_exponents(lines):
    """pdftotext can place a superscript above its row. Keep its column binding."""
    lines = list(lines)
    for i in range(len(lines)):
        if not re.fullmatch(r"\s*[-\u2212]\d+(?:\s+[-\u2212]\d+)*\s*", lines[i]):
            continue
        require(i + 1 < len(lines), "orphan PDF scientific exponent")
        below = lines[i + 1]
        insertions = []
        for token in re.finditer(r"[-\u2212]\d+", lines[i]):
            bases = [m for m in re.finditer(r"(?:x|\u00d7)\s*10(?![\d\u2212-])", below)
                     if abs(m.end() - token.start()) <= 8]
            require(len(bases) == 1, "ambiguous PDF scientific exponent")
            pos = bases[0].end()
            require(not below[min(pos, token.start()):max(pos, token.start())].strip(),
                    "PDF scientific exponent crosses non-whitespace")
            insertions.append((pos, token[0]))
        require(len({p for p, _ in insertions}) == len(insertions), "duplicate PDF exponent")
        for pos, exponent in sorted(insertions, reverse=True):
            below = below[:pos] + exponent + below[pos:]
        lines[i], lines[i + 1] = None, below
    return [line for line in lines if line is not None]


def bind_table(tex, rendered, label, expected, headers):
    """Bind every ordered source cell and contiguous PDF row inside one table.

    Tuple keys denote multiple identifying columns, specifically (arm, seed).
    The full rendered row is compared, retaining intervals and seed arrays.
    """
    expected_rows = [[*(key if isinstance(key, tuple) else (key,)), *cells]
                     for key, cells in expected.items()]
    rows = source_table_rows(tex, label)
    data = []
    for row in rows:
        if row[0] in (*headers, ""):
            require(not any(re.search(r"[-+]?\d+\.\d+|(?:^|[\[(,])[-+]?\d+(?=[,\])]|x10)|^[-+]?\d+%?$", c)
                            for c in row[1:]), label + ": numeric data disguised as header")
        else:
            data.append(row)
    agree([[visible(c) for c in row] for row in data],
          [[visible(c) for c in row] for row in expected_rows], label + ": source cells")
    lines = join_pdf_exponents(rendered_table(tex, rendered, label).splitlines())
    # A data block is contiguous. A paragraph or another table cannot rescue a row.
    first = visible(expected_rows[0][0])
    candidates = [i for i, line in enumerate(lines)
                  if visible(line).startswith(first) and re.search(r"\d", line)]
    valid = []
    # PDF extractors vary in column spacing. Numeric tokens must stay whole;
    # require a separator between cells, but allow spacing inside bracketed arrays
    # and between a scientific mantissa, multiplication sign and exponent.
    def cell_pattern(cell):
        tokens = re.findall(r"[-+]?\d+(?:\.\d+)?|[a-z]+|[^\s]", table_text(cell))
        return r"\s*".join(re.escape(token) for token in tokens)
    wanted = [r"\s*" + r"\s+".join(cell_pattern(c) for c in row) + r"\s*" for row in expected_rows]
    for i in candidates:
        actual = lines[i:i + len(wanted)]
        normalized = [table_text(line) for line in actual]
        if len(actual) == len(wanted) and all(re.fullmatch(pattern, line) for pattern, line in zip(wanted, normalized)):
            valid.append(i)
    require(len(valid) == 1, label + ": PDF row block differs, missing or duplicated")
    # Reject additional data-looking rows next to the bound block, including duplicates.
    i = valid[0]
    for neighbor in (i - 1, i + len(wanted)):
        if 0 <= neighbor < len(lines) and lines[neighbor].strip():
            require(not re.search(r"(?:\s{2,}|^)[-+]?\d+\.\d+", lines[neighbor]),
                    label + ": extra PDF row")


def main_table(tex, rendered, evidence, deadline=None):
    comparison, summary = evidence
    expected = {}
    for row, method in MAIN_ROWS.items():
        raw, corrected = method + "_raw", method + "_corrected"
        metrics = summary["methods"][raw]["metrics"]
        expected[row] = [f"{comparison['arm_means'][raw]:.3f}", f"{comparison['arm_means'][corrected]:.3f}",
                         f"{100 * metrics['cov90_mean']['month_balanced']:.1f}",
                         f"{100 * summary['methods'][corrected]['metrics']['cov90_mean']['month_balanced']:.1f}",
                         f"{metrics['r_obs_mean']['month_balanced']:.3f}",
                         f"{100 * metrics['frac_support_violation_beyond_rounding']['month_balanced']:.1f}"]
    if deadline is not None:
        expected["Gaussian interpolation"] = expected.pop("Gaussian")
        metrics = deadline["gaussian"]["metrics"]
        raw, corrected = metrics["raw"], metrics["corrected"]
        expected["Gaussian regression"] = [f"{raw['primary_es']['month_balanced']:.3f}",
            f"{corrected['primary_es']['month_balanced']:.3f}", f"{100 * raw['coverage90']['month_balanced']:.1f}",
            f"{100 * corrected['coverage90']['month_balanced']:.1f}", f"{metrics['raw_r_obs_mean']['month_balanced']:.3f}",
            f"{100 * raw['support_violation_beyond_rounding_fraction']['month_balanced']:.1f}"]
    main_rendered = re.split(r"(?m)^[ \t]*References[ \t]*$", pdf_without_line_numbers(rendered))[0]
    bind_table(tex, main_rendered, "tab:main", expected, ("Procedure", "Method", "Arm"))
    cap = clean_cell(float_caption(tex, "tab:main")).lower()
    require("inequalit" in cap and "rounding" in cap and re.search(
        r"excluding (?:the )?equality|do(?:es)? not (?:test|check) (?:the )?equality|not (?:a (?:test|measure) of )?full realizability", cap),
        "main caption must distinguish inequality-only failures from equality")
    require(re.search(r"independently[ -]selected|selected independently|raw and clipped diffusion predictions use separate validation checkpoint choices"
                      r"|validation selects checkpoints separately for raw and clipped diffusion predictions", cap),
            "main caption must declare independent checkpoint selection")
    require(visible(cap).replace("-", "") in visible(rendered_table(tex, main_rendered, "tab:main")).replace("-", ""),
            "main caption differs from rendered PDF")
    require(tex.index("\\label{tab:main}") < tex.index("\\bibliography{"), "main table must precede references")


def toy_claims(tex, rendered, evidence):
    means, context = evidence
    bind_table(tex, rendered, "tab:oracle", {row: [f"{means[key]:.4f}"] for row, key in ORACLE_ROWS.items()}, ("Procedure",))
    # The sampler scores moved to the bound table; prose still states their order.
    gap = means["two_step"] - means["exact_reference"]
    require(gap > 0 and all(abs(means[f"ddim_{n}"] - means["exact_reference"]) < gap for n in (25, 50, 100)),
            "toy sampler comparison directions changed")
    source = tex[:tex.index("\\bibliography{")]
    paragraphs = [p for p in re.split(r"\n\s*\n", source) if "A stationary hidden Markov model isolates" in p]
    require(len(paragraphs) == 1, "missing unique main-text toy diagnostic paragraph")
    main_rendered = re.split(r"(?m)^[ \t]*References[ \t]*$", pdf_without_line_numbers(rendered))[0]
    # A page-top operator figure can separate "DDIM with" from the evaluation
    # counts. Rejoin only that page break and its source-matching figure caption.
    interrupted = list(re.finditer(
        r"DDIM with\s+\d+\s*\f(?P<figure>[\s\S]*?)\n[ \t]*(?=25,\s*50,\s*and\s*100 evaluations gives)",
        main_rendered))
    if interrupted:
        require(len(interrupted) == 1, "ambiguous float in toy comparison")
        match = interrupted[0]
        figure_labels = re.findall(r"\\label\{(fig:[^}]+)\}", strip_comments(tex))
        require(figure_labels.count("fig:operator") == 1, "toy interruption requires operator figure")
        number = figure_labels.index("fig:operator") + 1
        captions = list(re.finditer(r"(?m)^[ \t]*Figure[ \t]+(\d+):", match["figure"]))
        require(len(captions) == 1 and int(captions[0][1]) == number,
                "toy interruption has wrong figure")
        table_form = tex.replace(r"\begin{figure}", r"\begin{table}").replace(r"\end{figure}", r"\end{table}")
        caption = float_caption(table_form, "fig:operator")
        require(re.fullmatch(clause_pattern(prose_text(caption, source=True)),
                             prose_text(match["figure"][captions[0].end():]).replace("±", "+-")) is not None,
                "toy interruption caption differs")
        main_rendered = main_rendered[:match.start()] + "DDIM with " + main_rendered[match.end():]
    mantissa, exponent = f"{context['mean_abs_r_star']:.2e}".split("e")
    residual = mantissa + r"\times10^{" + str(int(exponent)) + "}"
    clauses = (
        (r"For \d+ replicate contexts", "For 36 replicate contexts at $t=100$, two denoiser evaluations give mean absolute $R_{\\mathrm{obs}}="
         + residual + f"$ in HMM units and energy score {context['mean_es_vs_ref']:.3f}."),
        ("Independent draws from the exact", f"Independent draws from the exact conditional distribution give an energy score of {context['mean_reference_self_energy_mc']:.3f}."),
        ("In a separate evaluation", "In a separate evaluation on 288 HMM windows, two denoiser evaluations also score worse than exact conditional draws, while DDIM with 25, 50, and 100 evaluations gives scores much closer to the score of exact draws on those windows"),
        ("These Monte Carlo comparisons", "These Monte Carlo comparisons have finite sampling error."),
    )
    for anchor, expected in clauses:
        bind_clause((paragraphs[0], main_rendered), anchor.lower(), expected)


def original_payload(paper_dir, directory, filename, source, protocol_hash):
    """A byte-identical public original, with protocol and relative-name bindings."""
    root = local(paper_dir, "evidence/" + directory)
    manifest = read_json(local(root, "MANIFEST.json"))
    private_check(manifest)
    agree(manifest["schema_version"], 1)
    agree(manifest["source_protocol_sha256"], protocol_hash)
    require(set(manifest["files"]) == set(manifest["records"]) == {filename}
            and {p.name for p in root.iterdir()} == {filename, "MANIFEST.json"},
            directory + ": incomplete or extra files")
    for key in ("scope", "transformation", "validation"):
        require(isinstance(manifest.get(key), str) and bool(manifest[key].strip()), "missing declared " + key)
    sha = digest(local(root, filename))
    exporter._digest(manifest["files"][filename])
    agree(manifest["files"][filename], sha, directory + " public hash")
    agree(manifest["records"][filename], {"source_record": source, "source_sha256": sha},
          directory + " byte-identical original hash")
    return local(root, filename)


def validate_observations(paper_dir, real):
    protocol = read_json(local(paper_dir, "evidence/real/protocol-summary.json"))
    path = original_payload(paper_dir, "observations", "test.npz", "artifacts/inventory/test.npz",
                            protocol["source_protocol_sha256"])
    with np.load(path, allow_pickle=False) as blob:
        fields = {"z", "ids", "split", "mean", "scale", "b_ref_sq", "n_bins"}
        require(len(blob.files) == 7 and set(blob.files) == fields, "observation field inventory differs")
        require(all(not blob[k].dtype.hasobject for k in fields), "object observation array")
        require(all(blob[k].dtype == np.float64 for k in ("z", "mean", "scale", "b_ref_sq")),
                "observations and transform must retain float64")
        require(blob["split"].shape == blob["b_ref_sq"].shape == blob["n_bins"].shape == ()
                and blob["n_bins"].dtype.kind in "iu" and blob["split"].item() == "TEST",
                "invalid observation scalar fields")
    z, ids, nrm, identity = load_arrays(path, "TEST", protocol["arrays"]["TEST"]["sha256"])
    agree(identity, protocol["arrays"]["TEST"], "observation protocol identity")
    agree(identity, real[1]["arrays"])
    agree(ids, real[1]["ids"])
    agree(identity["normalization_sha256"], protocol["arrays"]["TRAIN"]["normalization_sha256"])
    agree(nrm.n_bins, real[1]["input_manifest"]["inventory_counts"]["train_bin_union"])
    return z, ids, nrm, identity


def observation_roundtrips(observations):
    """Use the scoring decoder; do not replace the original residual by zero."""
    from prigm.observation import residual_obs, support_gaps
    from prigm.run_score import decode_ensemble
    z, _, nrm, _ = observations
    target = z[:, 18:30]
    u = nrm.transform(z)
    routes = {"Original float64": target}
    for label, dtype, channels in (("Float64 normalization and inverse", np.float64, 6),
                                   ("Float32, six coordinates", np.float32, 6),
                                   ("Float32, derived v_v", np.float32, 5)):
        routes[label] = decode_ensemble(u[..., :channels].astype(dtype).transpose(0, 2, 1), nrm)
    rows = {}
    for label, values in routes.items():
        residual = np.abs(residual_obs(values))
        gaps = support_gaps(values)
        require(all(np.all(gaps[k] >= 0) for k in ("m_minus_r", "vs", "vs_upper_minus_vs")),
                "observation route violates a support inequality")
        rows[label] = [float(np.max(np.abs((values[..., :5] - target[..., :5]) / nrm.scale[:5]))),
                       float(residual.mean()), float(residual.max())]
    return rows


def validate_test_strata(paper_dir, real, observations):
    scores = read_json(local(paper_dir, "evidence/real/scores.json"))
    actual = diagnostics.energy_strata({k: np.asarray(v) for k, v in scores["data"].items()},
                                      np.asarray(scores["months"]), scores["ids"], observations[0])
    for key in ("tertiles", "strata"):
        agree(real[1][key], actual[key], "recomputed TEST " + key)
    for key, expected in (("seed", bootstrap.BOOTSTRAP_SEED), ("replicates", bootstrap.BOOTSTRAP_REPLICATES),
                          ("generator", "PCG64")):
        agree(real[1]["bootstrap"][key], expected)
    return actual


def validate_sensitivity(paper_dir, real, observations):
    """Recompute TEST features/groups, not the unpublished full TRAIN population."""
    comparison, summary = real
    protocol = read_json(local(paper_dir, "evidence/real/protocol-summary.json"))
    path = original_payload(paper_dir, "sensitivity", "training-strata.json", "artifacts/training-strata-01.json",
                            protocol["source_protocol_sha256"])
    record = read_json(path)
    # Formula division is not an execution path. Only these two literal physical
    # denominators are exempted in a lexical copy; original bytes stay untouched.
    private_check([text.replace(")/B_ref_sq", ") divided by B_ref_sq").replace(")/m^2", ") divided by m^2")
                   for text in exporter._strings(record)])
    agree(record["schema_version"], 1)
    for key in ("protocol_sha256", "selection_sha256", "test_started_sha256", "ids", "months"):
        agree(record[key], summary[key], "sensitivity." + key)
    agree(record["ids_sha256"], id_hash(summary["ids"]))
    sources = record["producer_source_sha256"]
    hash_map(sources)
    hash_map(record["inference_source_sha256"])
    require(set(sources) == {"analysis/prigm_training_strata.py", "analysis/prigm_followup_summary.py",
                             *record["inference_source_sha256"]}, "sensitivity producer inventory differs")
    for name, sha in sources.items():
        agree(sha, execution_source_hash(name), "actual producer source " + name)
    agree(record["analysis_source_sha256"], sources["analysis/prigm_training_strata.py"])
    for name, sha in record["inference_source_sha256"].items():
        agree(sources[name], sha)
    inputs = record["input_sha256"]
    hash_map(inputs)
    common = read_json(local(paper_dir, "evidence/real/MANIFEST.json"))["input_sha256"]
    for name in inputs.keys() & common.keys():
        agree(inputs[name], common[name], "common sensitivity input " + name)
    for name in ("PROTOCOL.json", "SELECTION.json", "TEST_STARTED.json", "PIPELINE_COMPLETED.json",
                 "CPU_FOLLOWUP_COMPLETED.json", exporter.SUMMARY, exporter.COMPARISON):
        agree(inputs[name], common[name], name)
    agree(record["arrays"], {s: protocol["arrays"][s] for s in ("TRAIN", "TEST")})
    for split, identity in record["arrays"].items():
        agree(inputs[f"artifacts/inventory/{split.lower()}.npz"], identity["sha256"])
    z, ids, nrm, identity = observations
    agree(record["normalization"], {"mean": nrm.mean.tolist(), "scale": nrm.scale.tolist(),
          "b_ref_sq": nrm.b_ref_sq, "n_bins": nrm.n_bins, "sha256": identity["normalization_sha256"]})
    agree(record["inventory_counts"], summary["input_manifest"]["inventory_counts"])
    agree(record["recorded_raw_sha256_by_year"], summary["input_manifest"]["recorded_raw_sha256_by_year"])
    require(set(record["methods"]) == set(summary["methods"]), "sensitivity procedure inventory differs")
    for label, method in record["methods"].items():
        diag = summary["methods"][label]
        for key in ("procedure", "scoring_kind", "selection_kind", "generations", "fixed_model_seeds", "n_realizations"):
            agree(method[key], diag[key], "sensitivity procedure " + key)
        require(set(method["files"]) == set(diag["files"]), "sensitivity seed inventory differs")
        for seed, file in method["files"].items():
            original = diag["files"][seed]
            agree(file, {"path": original["primary"], "sha256": inputs[original["primary"]],
                         "checkpoint_sha256": diag["checkpoint_sha256"][seed],
                         "control_fit_sha256": original["control_fit_sha256"],
                         "ensemble_commit_sha256": original["ensemble_commit_sha256"]})
            agree(file["sha256"], common[file["path"]])
    features = sensitivity.physical_features(z, nrm.b_ref_sq)
    agree(record["features_in_test_id_order"], {k: v.tolist() for k, v in features.items()})
    cutoffs = record["cutoffs"]
    require(set(cutoffs) == set(sensitivity.CUTOFF_QUANTILES), "TRAIN cutoff inventory differs")
    for name, quantiles in sensitivity.CUTOFF_QUANTILES.items():
        cutoff = cutoffs[name]
        for key, value in {"quantiles": quantiles, "quantile_method": "linear",
                           "n_training_windows": protocol["arrays"]["TRAIN"]["n_windows"],
                           "assignment": "searchsorted_side_right_1_based" if name.endswith("energy")
                                         else "strictly_greater_than"}.items():
            agree(cutoff[key], value, "TRAIN cutoff " + key)
        ties = cutoff["training_edge_tie_counts"]
        require(len(ties) == len(quantiles) and all(type(n) is int and 0 <= n <= cutoff["n_training_windows"] for n in ties),
                "invalid TRAIN edge tie counts")
    definitions = record["cutoff_definitions"]
    agree(definitions["target_indices"], list(range(18, 30)))
    agree(definitions["context_indices"], list(range(18)) + list(range(30, 48)))
    agree(definitions["test_edge_tie_counts"], {k: [int(np.count_nonzero(features[k] == e)) for e in v["values"]]
                                               for k, v in cutoffs.items()})
    for key, expected in (("seed", bootstrap.BOOTSTRAP_SEED), ("replicates", bootstrap.BOOTSTRAP_REPLICATES),
                          ("generator", "PCG64")):
        agree(record["bootstrap"][key], expected)
    scores = read_json(local(paper_dir, "evidence/real/scores.json"))
    actual = sensitivity.summarize_groups({k: np.asarray(v) for k, v in scores["data"].items()},
                                         np.asarray(summary["months"]), ids, features, cutoffs)
    for key, value in actual.items():
        agree(record[key], value, "recomputed sensitivity " + key)
    require(len(record["groups"]) == 21, "requires all 21 sensitivity groups")
    for key in ("target", "context", "validation"):
        agree(record["scope"][key], sensitivity.SCOPE[key], "sensitivity limitation " + key)
    for label, method in record["groups"]["full_test"]["methods"].items():
        agree(method["month_balanced"], comparison["arm_means"][label])
    return record


def validate_numeric_record(record, scalar_metrics, array_metrics, *, corrected):
    """Validate ordered per-window values and reductions without loading ensembles."""
    from analysis import export_prigm_toy_evidence as toy_export
    agree(record["corrected"], corrected)
    rows = record["per_window"]
    metrics = set(scalar_metrics) | set(array_metrics)
    require(isinstance(rows, list) and len(rows) == 288
            and all(isinstance(r, dict) and set(r) == metrics for r in rows),
            "incomplete toy per-window metric inventory")
    require(set(record["secondary"]) == metrics, "toy secondary metric inventory differs")
    values = {}
    for metric in metrics:
        array = toy_export._numeric([r[metric] for r in rows])
        require(array.shape == ((288, 12, 5) if metric in array_metrics else (288,)),
                "toy metric shape differs: " + metric)
        if metric.startswith(("cov", "frac_")) or metric == "proxy_cov90":
            require(np.all((array >= 0) & (array <= 1)), "coverage/fraction outside [0,1]")
        reduced = record["secondary"][metric]
        require(set(reduced) == {"mean", "month_balanced"}, "toy reduction fields differ")
        for value in reduced.values():
            toy_export._close(value, array.mean(axis=0), "toy reduction " + metric)
        values[metric] = array
    for scalar, matrix in (("crps5_mean", "crps_by_position_channel"),
                           ("cov90_mean", "cov90_by_position_channel"),
                           ("cov50_mean", "cov50_by_position_channel"),
                           ("width90_mean", "width90_by_position_channel"),
                           ("width50_mean", "width50_by_position_channel")):
        if matrix in values:
            toy_export._close(values[scalar].tolist(), values[matrix].mean(axis=(1, 2)), "toy coordinate reduction")
    scores = toy_export._numeric(record["scores"])
    require(scores.shape == (288,), "requires all 288 toy scores")
    toy_export._close(record["scores"], values["es5"], "ordered toy es5")
    for key in ("mean_energy", "month_balanced_energy"):
        toy_export._close(record[key], scores.mean(), key)


def replay_frozen_toy_reference(array_path, oracle, oracle_hash):
    """Use the unchanged exporter after checking the frozen/current source pair."""
    from analysis import export_prigm_toy_evidence as toy_export
    for name, sha in oracle["source_hashes"].items():
        agree(sha, execution_source_hash("prigm/" + name), "frozen oracle source " + name)
    # The exporter is pinned historical code and requires current runtime hashes.
    # Adapt only this in-memory replay input; retain all numerical checks and
    # restore historical metadata for comparison with the untouched public record.
    replay = {**oracle, "source_hashes": {name: digest(local(REPO, "prigm/" + name))
                                        for name in oracle["source_hashes"]}}
    expected = toy_export._reference_calibration(array_path, replay, oracle_hash)
    expected["oracle_metadata"]["source_hashes"] = dict(oracle["source_hashes"])
    return expected


def validate_learned_toy(paper_dir):
    """All nine public fits; verify bindings and reductions, never run a network."""
    from analysis import export_prigm_toy_evidence as toy_export
    from experiments import run_prigm_toy_repair as toy
    from prigm.checkpoint import normalization_hash
    from prigm.config import ORACLE_REF_SEED
    root = local(paper_dir, "evidence/toy/learned")
    manifest = read_json(local(root, "MANIFEST.json"))
    private_check(manifest)
    agree(manifest["schema_version"], 1)
    agree(manifest["exporter_source_sha256"], documentation.source_hash(REPO, "analysis/export_prigm_toy_evidence.py"), "toy exporter source hash")
    names = {"learned.json", "reference-calibration.json"}
    require(set(manifest["output_sha256"]) == names
            and {p.name for p in root.iterdir()} == names | {"MANIFEST.json"}, "incomplete learned toy files")
    records = {}
    for name in names:
        path = local(root, name)
        agree(digest(path), manifest["output_sha256"][name], name + " public hash")
        records[name] = read_json(path)
        private_check(records[name])
    learned, reference = records["learned.json"], records["reference-calibration.json"]
    originals, inputs, sources = manifest["original_sha256"], manifest["input_sha256"], manifest["sources"]
    hash_map(originals)
    hash_map(inputs)
    require(set(originals) == {"protocol", "array", "trained", "evaluated", "oracle"}, "toy original inventory differs")
    for key, filename in (("protocol", "PROTOCOL.json"), ("array", "artifacts/toy.npz"),
                          ("trained", "TRAINED.json"), ("evaluated", "EVALUATED.json")):
        agree(originals[key], inputs[filename], "original toy " + key)
    array_path = local(paper_dir, "evidence/toy/toy.npz")
    oracle_path = local(paper_dir, "evidence/toy/oracle-stationary-01.json")
    agree(originals["oracle"], digest(oracle_path), "same stationary original oracle")
    _, ids, nrm, identity = load_arrays(array_path, "TOY", originals["array"])
    agree(ids, [f"TOY:{i}" for i in range(288)])
    agree(learned["original_sha256"], originals)
    agree(learned["sources"], sources)
    require(set(sources) == {"training", "inference", "coordinator_sha256"}, "toy source inventory differs")
    for kind in ("training", "inference"):
        hash_map(sources[kind])
        agree(sources[kind], {name: execution_source_hash(name) for name in toy.source_identity()},
              "actual toy " + kind + " sources")
    agree(sources["coordinator_sha256"], documentation.source_hash(REPO, "experiments/run_prigm_toy_repair.py"))
    for record in (learned, reference):
        for key, value in {"schema_version": 1, "n": 288, "n_draws": 39, "ids": ids, "months": [0] * 288,
                           "arrays": identity, "interval_protocol": toy_export.INTERVAL_PROTOCOL}.items():
            agree(record[key], value, "toy." + key)
    recipe = toy.fixed_recipe()
    protocol = learned["protocol"]
    for key, value in {"study": "controlled_stationary_toy_repair", "exposure": "historical_test_exposed",
                       "boundary_law": "stationary", "recipe": recipe,
                       "window_law": "stationary_boundary_markov_bridge_with_repeated_endpoint_flanks; not a full 48-bin Markov trajectory",
                       "samplers": {"TOY": {"n_eval": 50, "n_draws": 39, "eval_seed": 7}},
                       "sampling_chunk_windows": 8, "state_convention": toy.STATE_CONVENTION,
                       "noise_convention": toy.NOISE_CONVENTION,
                       "selection": "fixed final checkpoint at update 5000 only; no validation or coefficient selection; all nine fits retained"}.items():
        agree(protocol[key], value, "learned toy protocol " + key)
    training = learned["training_identity"]
    # Deterministic finite-state data generation only, not fitting or denoising.
    # This is separate from the unpublished real TRAIN cutoff population.
    toy_export._exact(training, toy.training_identity()[0], "toy training data/source identity")
    agree(training["normalization"], {"mean": nrm.mean.tolist(), "scale": nrm.scale.tolist(),
          "b_ref_sq": nrm.b_ref_sq, "n_bins": nrm.n_bins})
    agree(training["normalization_sha256"], normalization_hash(nrm.mean, nrm.scale, nrm.b_ref_sq, nrm.n_bins))
    settings = learned["training_settings"]
    exporter._digest(settings["source_snapshot_sha256"])
    expected_settings = toy.scientific_settings(settings["source_snapshot_sha256"])
    # Older PyTorch AdamW defaults omit this key; decoupled decay was already
    # intrinsic to AdamW. Compare that semantic default, not local-version keys.
    normalized_settings = {**settings, "optimizer": {"decoupled_weight_decay": True, **settings["optimizer"]}}
    expected_settings["optimizer"] = {"decoupled_weight_decay": True, **expected_settings["optimizer"]}
    agree(normalized_settings, expected_settings, "toy training settings")
    order = [fit["name"] for fit in recipe["fits"]]
    agree(learned["fit_order"], order)
    require(set(learned["fits"]) == set(order), "requires all nine toy fits")
    required_inputs = {"PROTOCOL.json", "PROTOCOL_RECEIPT.json", "artifacts/toy.npz", "TRAINED.json", "EVALUATED.json"}
    for fit in recipe["fits"]:
        name = fit["name"]
        row = learned["fits"][name]
        run, parent = f"runs/{name}", f"artifacts/toy/{name}-g5000"
        config = {k: v for k, v in recipe.items() if k not in ("fits", "milestones")}
        config.update({k: v for k, v in fit.items() if k != "name"})
        config.update(contract_version=2, source_hashes=sources["training"], data_hashes=training["data_hashes"],
                      normalization_hash=training["normalization_sha256"], settings=settings)
        agree(row["config"], config, "toy fit config")
        agree(row["generation_updates"], 5000)
        for key, filename in (("checkpoint_sha256", f"{run}/gen_00005000/state.pt"),
                              ("checkpoint_commit_sha256", f"{run}/gen_00005000/COMMIT.json"),
                              ("ensemble_commit_sha256", f"{parent}/ensemble/gen_00000000/COMMIT.json"),
                              ("completed_sha256", f"{parent}/COMPLETED.json")):
            agree(row[key], inputs[filename], "toy " + key)
            required_inputs.add(filename)
        required_inputs.update({f"{run}/source_snapshot.zip", f"{run}/gen_00005000/meta.json",
                                f"artifacts/training/{name}-request.json", f"artifacts/training/{name}-completed.json",
                                f"{parent}/REQUEST.json", f"{parent}/ensemble/gen_00000000/ensemble.npz"})
        agree(inputs[f"{run}/source_snapshot.zip"], settings["source_snapshot_sha256"])
        for offset in range(0, 288, 8):
            required_inputs.update(f"{parent}/chunk-{offset:04d}/gen_00000000/{f}" for f in ("COMMIT.json", "ensemble.npz"))
        for kind in ("raw", "corrected"):
            variant = row[kind]
            require(set(variant["source_sha256"]) == {"score", "secondaries"}, "toy score source inventory differs")
            for key in ("score", "secondaries"):
                filename = f"{parent}/{key}-{kind}.json"
                required_inputs.add(filename)
                agree(variant["source_sha256"][key], inputs[filename], "toy original score hash")
            validate_numeric_record(variant, toy_export.SCALAR_METRICS, toy_export.ARRAY_METRICS,
                                    corrected=kind == "corrected")
    require(set(inputs) == required_inputs, "toy input inventory differs from complete nine-fit/chunk union")
    agree(reference["source_protocol_sha256"], originals["protocol"])
    agree(reference["source_oracle_sha256"], originals["oracle"])
    agree(reference["reference_seed"], ORACLE_REF_SEED)
    oracle = read_json(oracle_path)
    # Only replay the frozen direct finite-state reference RNG, never a denoiser.
    expected = replay_frozen_toy_reference(array_path, oracle, originals["oracle"])
    scalar = ("es5", "cov90_mean", "cov50_mean", "width90_mean", "width50_mean", "crps5_mean")
    arrays = ("cov90_by_position_channel", "cov50_by_position_channel", "width90_by_position_channel",
              "width50_by_position_channel", "crps_by_position_channel")
    validate_numeric_record(reference, scalar, arrays, corrected=False)
    for key in ("oracle_metadata", "contexts"):
        agree(reference[key], expected[key])
    for metric in (*scalar, *arrays):
        toy_export._close([r[metric] for r in reference["per_window"]],
                          [r[metric] for r in expected["per_window"]], "same-stream reference " + metric)
    # Coverage of the atomic law is checked against that law, never against 0.90/0.50.
    return learned, reference


def scientific_cell(value):
    if value == 0:
        return "0"
    m, e = f"{value:.3e}".split("e")
    return m + "x10^" + str(int(e))


def prose_text(text, *, source=False):
    """Typography only: retain numeric tokens, signs, exponents and percentages."""
    if source:
        for command, symbol in ((r"\lambda", "λ"), (r"\Sigma", "Σ"), (r"\rho", "ρ"), (r"\max", "max")):
            text = text.replace(command, symbol)
        text = clean_cell(strip_comments(text))
    else:
        # Restore the isolated squared normalization unit before removing page numbers.
        lines = text.splitlines()
        for i, line in enumerate(lines[:-1]):
            if re.fullmatch(r"[ \t]*2[ \t]*", line):
                column, below = line.index("2"), lines[i + 1]
                bases = [m for m in re.finditer(r"\bBref\b", below)
                         if 0 <= column - m.end() <= 8 and not below[m.end():column].strip()]
                if len(bases) == 1:
                    end = bases[0].end()
                    lines[i], lines[i + 1] = "", below[:end] + "2" + below[end:]
        text = pdf_without_line_numbers("\n".join(lines))
        text = re.sub(r"(?m)^[ \t]*\d+[ \t]*$", "", text)  # Page numbers, not prose counts.
        text = re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", "\\1\u00ad\\2", text)
    text = table_text(text).replace("’", "'").replace("~", " ").replace("λ", "lambda").replace("σ", "sigma").replace("ρ", "rho")
    # Only count contexts, never a comma-delimited timestep or interval vector.
    text = re.sub(r"\b\d{1,3}(?:,\d{3})+(?=-update|\s+(?:overlapping|windows|training updates|optimizer updates))",
                  lambda m: m[0].replace(",", ""), text)
    return " ".join(text.split())


def clause_pattern(text):
    # Whole numeric tokens cannot be rescued by concatenating digits across spaces.
    tokens = re.findall(r"\d+(?:\.\d+)?|[^\W\d_]+|[^\s]", text)
    def token_pattern(token):
        if token.isalpha():
            return "\u00ad?".join(re.escape(c) for c in token)
        return "[-\u00ad]" if token == "-" else re.escape(token)
    return r"\s*".join(token_pattern(token) for token in tokens)


def claim_subsection(tex, rendered, title):
    """One named source subsection and its PDF heading, ending at the next heading."""
    tex = strip_comments(tex)
    marker = r"\subsection{" + title + "}"
    require(tex.count(marker) == 1, "missing or duplicate source subsection: " + title)
    source = re.split(r"\\(?:sub)*section\*?\{", tex.split(marker, 1)[1], maxsplit=1)[0]
    pdf = table_text(pdf_without_line_numbers(rendered))
    titles = re.findall(r"\\(?:sub)*section\*?\{([^{}]+)\}", tex)
    headings = list(re.finditer(r"(?m)^[ \t]*(?:[a-z]|\d+)(?:\.\d+)*[ \t]+(?:"
                               + "|".join(re.escape(table_text(t)) for t in titles) + r")[ \t]*$", pdf))
    matches = [i for i, h in enumerate(headings) if re.fullmatch(
        r"[ \t]*(?:[a-z]|\d+)(?:\.\d+)*[ \t]+" + re.escape(title.lower()) + r"[ \t]*", h[0])]
    require(len(matches) == 1, "missing or duplicate PDF subsection: " + title)
    i = matches[0]
    end = headings[i + 1].start() if i + 1 < len(headings) else len(pdf)
    return source, pdf[headings[i].end():end]


def bind_clause(scopes, anchor, expected):
    """Bind a unique clause, never a number elsewhere in the subsection or PDF.

    Anchors are fixed lexical prefixes (regex for displayed counts). Matching the
    entire expected clause from that prefix also checks its units and qualifiers.
    """
    for source, scope in zip((True, False), scopes):
        text = prose_text(scope, source=source)
        matches = list(re.finditer(r"(?<!\w)" + anchor.lower(), text))
        location = "source" if source else "PDF"
        require(len(matches) == 1, location + ": missing or duplicate clause " + anchor)
        pattern = clause_pattern(prose_text(expected, source=True))
        require(re.match(pattern + r"(?![\w\d])", text[matches[0].start():]) is not None,
                location + ": clause differs: " + anchor)


def deadline_claims(tex, rendered, real, evidence):
    """Bind extension tables and selected explanatory clauses to checked values."""
    operator, gaussian = evidence["operator"], evidence["gaussian"]
    expected = {}
    for row in evidence["plotted_rows"]:
        expected[row["label"].replace("\n", " ")] = [f"{row['movement']:.4f}", f"{row['delta_target']:.5f}",
            f"{row['negative_delta_dispersion']:.5f}", f"{row['delta_energy']:.5f}"]
    bind_table(tex, rendered, "tab:operators", expected, ("Procedure",))
    def bind(anchor, expected_clause):
        bind_clause((tex, rendered), clause_pattern(prose_text(anchor, source=True)), expected_clause)
    selection = evidence["D_selection"]
    bind("For D, clipping the ensemble", f"For D, clipping the ensemble from the checkpoint selected using raw draws changes the energy score by ${selection['fixed_checkpoint_correction']:.5f}$")
    bind("Selecting weights independently", f"Selecting weights independently on VAL after clipping adds ${selection['selection_effect']:.5f}$; only seed~7 changes checkpoint.")
    for anchor, expected_clause in (
        ("Each percentile is calculated", "Each percentile is calculated from the draws and hidden bins of one window, then averaged over months and seeds. This differs from a percentile of all samples pooled together."),
        ("Minimum displacement", "Minimum displacement is not established for this map."),
        ("Relative to Gaussian interpolation, regression", "Relative to Gaussian interpolation, regression changes the conditional mean fit and the finite noise realization as well as using more available measurements. Its score difference cannot isolate the benefit of the additional context alone."),
        ("They hold the three diffusion fits", "They hold the three diffusion fits and one Gaussian ensemble fixed and exclude variation from new fits, seed resampling and the validation choice. The later comparison uses a TEST period examined earlier."),
    ):
        bind(anchor, expected_clause)
    plotted = {row["method"]: row for row in evidence["plotted_rows"]}
    require(plotted["G5"]["movement"] > 0
            and plotted["gaussian"]["negative_delta_dispersion"] > -plotted["gaussian"]["delta_target"] > 0,
            "clipping displacement and spread directions changed")
    bind("Clipping moves G5 draws", "Clipping moves G5 draws in the standardized primary coordinates with little change in energy score")
    bind("For Gaussian interpolation, both", "For Gaussian interpolation, both the mean distance to withheld values and half the mean distance between draws decrease. The larger decrease in the subtracted term makes the energy score worse.")
    # Tables retain the numerical results omitted from the conceptual abstract.
    difference = real[0]["arm_means"]["D_raw"] - real[0]["arm_means"]["O-S_raw"]
    require(difference > 0, "selected penalty improvement direction changed")
    bind("The energy score compares", "The energy score compares predictive distributions: a penalty on the identity residual gives little mean improvement over diffusion alone when validation chooses the model.")
    require(real[1]['methods']['G5_raw']['metrics']['cov90_mean']['month_balanced'] < .9,
            "G5 coverage no longer falls below nominal")
    bind("Generating the quantities needed", "Generating the quantities needed to derive the remaining statistic gives the lowest mean score on those quantities. Both representation and training loss change, and prediction intervals cover fewer withheld values than intended.")
    g5, boundary = (operator["methods"][name]["metrics"] for name in ("G5", "gaussian"))
    data_only = operator["methods"]["D"]["metrics"]
    target_reduction = data_only["raw_target_term"]["month_balanced"] - g5["raw_target_term"]["month_balanced"]
    dispersion_increase = g5["raw_dispersion_term"]["month_balanced"] - data_only["raw_dispersion_term"]["month_balanced"]
    energy_reduction = data_only["raw_energy_score"]["month_balanced"] - g5["raw_energy_score"]["month_balanced"]
    require(min(target_reduction, dispersion_increase, energy_reduction) > 0
            and dispersion_increase > target_reduction, "G5 raw decomposition directions changed")
    figure_labels = re.findall(r"\\label\{(fig:[^}]+)\}", strip_comments(tex))
    require(figure_labels.count("fig:calibration") == 1, "decomposition requires calibration figure")
    number = figure_labels.index("fig:calibration") + 1
    bind_clause((tex.replace(r"\ref{fig:calibration}", str(number)), rendered),
                clause_pattern("at the checkpoints chosen using raw validation draws"),
                "At the checkpoints chosen using raw validation draws, G5 has a slightly smaller mean distance to the withheld measurements than D and a larger spread term. "
                f"The score subtracts the spread term, whose increase accounts for most of the mean improvement in Figure~{number}(a).")
    bind("Both terms use the same", "Both terms use the same 60 standardized coordinates and equal month and seed weights.")
    bind("For G5, the 90th percentile", "For G5, the 90th percentile of $\\max(r-m,0)/B_{\\mathrm{ref}}$ is calculated within each window and averages to "
         f"{g5['raw_m_minus_r_severity_quantiles']['month_balanced'][2]:.5f}, compared with "
         f"{boundary['raw_m_minus_r_severity_quantiles']['month_balanced'][2]:.5f} for Gaussian interpolation.")
    bind("For $\\max(-v_s,0)", "For $\\max(-v_s,0)/B_{\\mathrm{ref}}^2$, the corresponding values are "
         f"{g5['raw_vs_severity_quantiles']['month_balanced'][2]:.6f} and {boundary['raw_vs_severity_quantiles']['month_balanced'][2]:.6f}.")
    bind("Clipping changes D's energy score", f"Clipping changes D's energy score by ${selection['fixed_checkpoint_correction']:.4f}$; choosing a separate checkpoint adds ${selection['selection_effect']:.4f}$.")
    metrics = gaussian["metrics"]
    raw, corrected = metrics["raw"], metrics["corrected"]
    selection = gaussian["selection"]["selected"]
    require(all(selection[k]["lambda"] == .1 and selection[k]["rho"] == 0 for k in ("raw", "corrected")),
            "published Gaussian selection differs")
    bind("Both procedures select", "Both procedures select $\\lambda=0.1$ and $\\rho=0$, with VAL scores "
         f"{selection['raw']['month_balanced_primary_es']:.5f} and {selection['corrected']['month_balanced_primary_es']:.5f}.")
    for arm in ("G5", "O-S"):
        pair = gaussian["comparisons"][arm]
        if arm == "G5":
            prefix = "G5 minus Gaussian regression has a mean difference in energy score"
            clause = (f"{prefix} of ${pair['mean_diff']:.4f}$, with year interval "
                f"$[{pair['year_95'][0]:.4f},{pair['year_95'][1]:.4f}]$ and interval from resampling groups of two years "
                f"$[{pair['twoyear_95'][0]:.4f},{pair['twoyear_95'][1]:.4f}]$.")
        else:
            prefix = "For O-S, the mean difference"
            clause = (f"{prefix} is ${pair['mean_diff']:.4f}$, with intervals "
                f"$[{pair['year_95'][0]:.4f},{pair['year_95'][1]:.4f}]$ and "
                f"$[{pair['twoyear_95'][0]:.4f},{pair['twoyear_95'][1]:.4f}]$, respectively.")
        bind(prefix, clause)
    require(gaussian["ranks"]["raw"]["resolved_energy_proxy_crps"]["rank_among_original_18_plus_selected"] == 1,
            "full-flank raw proxy CRPS ranking changed")
    primary = {k: v for k, v in real[0]["arm_means"].items() if k.endswith(("_raw", "_corrected"))}
    primary["full_flank_gaussian_raw"] = raw["primary_es"]["month_balanced"]
    primary["full_flank_gaussian_corrected"] = corrected["primary_es"]["month_balanced"]
    require(min(primary, key=primary.get).startswith("G5_"), "main-table G5 primary ranking changed")
    require(raw["primary_es"]["month_balanced"] < real[0]["arm_means"]["gaussian_raw"]
            and raw["coverage90"]["month_balanced"] > .9
            and corrected["primary_es"]["month_balanced"] > raw["primary_es"]["month_balanced"], "Gaussian score/coverage directions changed")
    bind("Mean displacement over", f"Mean displacement over the 60 standardized primary coordinates is {metrics['correction_displacement']['month_balanced']:.4f}.")
    bind("Before clipping, Gaussian regression's", "Before clipping, Gaussian regression's vector energy score is "
         f"{raw['vector_energy']['month_balanced']:.4f} in 36 standardized coordinates. Its CRPS for mean resolved magnetic energy density is "
         f"${raw['resolved_energy_proxy_crps']['month_balanced'] * 1e12:.4f}\\times10^{{-12}}$ J\\,m$^{{-3}}$.")
    bind("Clipping preserves the vector", "Clipping preserves the vector predictions but raises the CRPS for energy density to "
         f"${corrected['resolved_energy_proxy_crps']['month_balanced'] * 1e12:.4f}\\times10^{{-12}}$ J\\,m$^{{-3}}$.")
    coverage = real[1]["methods"]
    difference = 100 * (abs(coverage["O-S_raw"]["metrics"]["cov90_mean"]["month_balanced"] - .9)
                        - abs(coverage["D_raw"]["metrics"]["cov90_mean"]["month_balanced"] - .9))
    require(difference > 2, "original descriptive coverage flag changed")
    bind("O-S's absolute departure", f"O-S's absolute departure from nominal 90\\% coverage exceeds D's by {difference:.1f} percentage points, beyond the prespecified two-point threshold.")
    fixed = evidence["fixed_final"]
    if fixed is not None:
        expected = {}
        means = {coefficient: {kind: [] for kind in ("raw", "corrected")} for coefficient in FIXED_COEFFICIENTS}
        for row in fixed["rows"]:
            raw_row, corrected_row = (row["results"][k]["summary"] for k in ("raw", "corrected"))
            value = lambda key: raw_row[key]["month_balanced"]
            expected[row["id"][1:].replace("_s", " / ")] = [f"{value('es5'):.4f}", f"{corrected_row['es5']['month_balanced']:.4f}",
                f"{value('r_obs_mean'):.4f}", f"{100 * value('frac_support_violation_beyond_rounding'):.1f}",
                f"{100 * value('cov90_mean'):.1f}", f"{value('width90_mean'):.4f}"]
            coefficient = row["id"][1:].split("_s")[0]
            for kind in ("raw", "corrected"):
                means[coefficient][kind].append(row["results"][kind]["summary"]["es5"]["month_balanced"])
        bind_table(tex, rendered, "tab:fixedfinal", expected, ("Coefficient / seed",))
        require(np.all(np.asarray(means["0.0005"]["raw"]) < means["0"]["raw"]), "final original-coefficient seed signs changed")
        bind("At the original coefficient, O-S", "At the original coefficient, O-S has a lower energy score than D for every seed at the end of training.")
        bind("At the final training step, the original", "At the final training step, the original O-S coefficient gives a lower mean raw energy score than D and improves all three seeds.")
        bind("Further fits would be needed", "Further fits would be needed to establish whether this pattern persists beyond the current exploratory comparison.")
        bind("The final step was", "The final step was specified in advance. All twelve fits enter this exploratory comparison, without choosing a checkpoint or coefficient from their scores.")
        bind("Residual estimation uses extra", "Residual estimation uses extra denoiser evaluations, so equal update counts involve different amounts of computation.")
        bind("Computation can differ", "Computation can differ despite equal step counts, and score differences may reflect convergence or overfitting.")
        residuals = [float(np.mean([row["results"]["raw"]["summary"]["r_obs_mean"]["month_balanced"]
                                   for row in fixed["rows"] if row["id"].startswith("c" + coefficient + "_s")]))
                     for coefficient in FIXED_COEFFICIENTS]
        energies = [float(np.mean(means[coefficient]["raw"])) for coefficient in FIXED_COEFFICIENTS]
        require(np.all(np.diff(residuals) < 0) and np.all(np.diff(energies[:3]) < 0)
                and energies[-1] > energies[-2] and np.argmin(energies) == 2, "fixed-final mean trends changed")
        bind("the mean residual decreases", "the mean residual decreases as the penalty grows, while the mean energy score first improves and then worsens")
        bind("The strongest penalty gives", "The strongest penalty gives the smallest residual, while the original coefficient gives the best mean score.")
        signs = np.asarray(means["0.005"]["raw"]) - means["0"]["raw"]
        require(np.all(signs[:2] > 0) and signs[2] < 0, "largest-coefficient seed directions changed")
        bind("At $c_R=0.005$", "At $c_R=0.005$, the score improves over D for seed 42 and worsens for seeds 7 and 13.")


def learned_toy_claims(tex, rendered, evidence):
    """Nine ordered fits, same-checkpoint repair, and the atomic reference paragraph."""
    learned, reference = evidence
    rows = {}
    for name in learned["fit_order"]:
        fit = learned["fits"][name]
        key = (fit["config"]["arm"], format(fit["config"]["c_r"], "g"))
        require(key not in rows, "duplicate learned composite ID")
        raw = fit["raw"]["secondary"]
        rows[key] = [f"{fit[k]['mean_energy']:.4f}" for k in ("raw", "corrected")]
        rows[key] += [f"{raw['r_obs_mean']['mean']:.4f}",
                      f"{100 * raw['frac_support_violation_beyond_rounding']['mean']:.1f}",
                      f"{100 * raw['cov90_mean']['mean']:.1f}"]
    bind_table(tex, rendered, "tab:learnedtoy", rows, ("Procedure",))
    scopes = claim_subsection(tex, rendered, "Training diffusion models on the same HMM")
    bind_clause(scopes, "compares nine diffusion fits", "compares nine diffusion fits to the stationary chain, with different residual objectives and 50 DDIM evaluations for sampling.")
    caption = float_caption(tex, "tab:learnedtoy")
    pdf_caption = re.split(r"\n[ \t]*\n", rendered_table(tex, rendered, "tab:learnedtoy").strip(), maxsplit=1)[0]
    cap_scopes = (caption, pdf_caption)
    bind_clause(cap_scopes, "Diffusion models for the hidden Markov example", "Diffusion models for the hidden Markov example after 5{,}000 optimizer updates at seed 7, evaluated on 288 windows with 39 draws each.")
    for anchor, expected in (
        ("Clipping uses", "Clipping uses the same model state and draws."),
        (r"Raw \|R\|", "Raw $|R|$ is the mean absolute residual of the observation identity in HMM units."),
        ("The inequality column", "The inequality column is the percentage of generated bins failing a physical inequality beyond rounding tolerance; it excludes the equality test."),
        ("Coverage averages", "Coverage averages the five primary coordinates and twelve positions."),
        ("The separate exact conditional ensemble", "The separate exact conditional ensemble has "
         f"{100 * reference['secondary']['cov90_mean']['mean']:g}\\% coverage under the inclusive nominal 90\\% interval convention."),
    ):
        bind_clause(cap_scopes, anchor, expected)
    # Compare the complete rendered caption as well as the load-bearing meanings.
    cap_pdf = prose_text(cap_scopes[1])
    require(re.match(clause_pattern(prose_text(caption, source=True)), cap_pdf) is not None,
            "learned caption differs from rendered PDF")
    secondary = reference["secondary"]
    expected = (f"For exact conditional draws, inclusive intervals cover {100 * secondary['cov90_mean']['mean']:.1f}\\% "
                f"and {100 * secondary['cov50_mean']['mean']:.2f}\\% of the primary coordinates under the nominal 90\\% and 50\\% endpoint conventions. "
                f"Their mean widths are {secondary['width90_mean']['mean']:.4f} and {secondary['width50_mean']['mean']:.4f} in standardized coordinates. "
                "Ties in this discrete distribution can raise the coverage of inclusive intervals. Coverage expected for a continuous distribution is therefore insufficient to assess calibration here. "
                "The interval calculations use the original random stream and reproduce all 288 original energy scores before computing coverage and width.")
    paragraphs = [p for p in re.split(r"\n\s*\n", scopes[0]) if p.startswith("For exact conditional draws")]
    require(len(paragraphs) == 1, "missing unique adjacent reference paragraph")
    bind_clause((paragraphs[0], scopes[1]), "For exact conditional draws", expected)
    return rows


def residual_weight_claims(paper_dir, tex, rendered):
    """Read the already validated public payload with its hash; do not compute gradients."""
    root = local(paper_dir, "evidence/real")
    manifest = read_json(local(root, "MANIFEST.json"))
    path = local(root, "gradients.json")
    agree(digest(path), manifest["output_sha256"]["gradients.json"], "gradients.json hash")
    gradients = read_json(path)
    shared = None
    for grid in gradients["grids"].values():
        for report in grid["reports"].values():
            agree(report["schedule"]["probe_timesteps"], [1, 25, 50, 100])
            values = {k: report["schedule"]["probe_values"][k] for k in ("sigma", "w")}
            if shared is None:
                shared = values
            require(values == shared, "shared gradient schedule probes differ")
    sigma, w = (np.asarray(shared[k], dtype=float) for k in ("sigma", "w"))
    require(sigma.shape == w.shape == (4,) and np.all(sigma > 0) and np.all(w > 0), "invalid coefficient denominators")
    grid = gradients["grids"]["29636"]
    ratios = grid["request"]["c_r"] / (2 * sigma * w)
    # This scope statement moved before the subsection. Normalize its numeric
    # set notation locally while retaining every timestep token.
    gradient_scopes = (re.sub(r"\\in\\\{([0-9, ]+)\\\}", r"in (\1)", tex),
                       re.sub(r"∈\s*\{([0-9,\s]+)\}", r"in (\1)", rendered))
    for anchor, expected in (
        ("For each O-S seed", "For each O-S seed, the calculation uses the same eight TRAIN windows and eight VAL windows at the first and final common checkpoints, with noise seed 7003 and $c_R=0.0005$."),
        ("Both residuals and both estimators", "Both residuals and both estimators are evaluated at t in (1,25,50,100) while the model weights remain constant."),
    ):
        bind_clause(gradient_scopes, anchor, expected)
    scopes = claim_subsection(tex, rendered, "How the residual penalty is weighted")
    bind_clause(scopes, "The relative weight", "The relative weight of the residual penalty varies with the diffusion timestep, even at $c_R=0.0005$.")
    bind_clause(scopes, clause_pattern(prose_text(r"With $\lambda_t", source=True)), r"With $\lambda_t=c_R/(2\Sigma_t)$, the ratios $\lambda_t/w_t$ at $t=1,25,50,100$ are "
                + f"{ratios[0]:.4f}, {ratios[1]:.6f}, {ratios[2]:.6f}, and {ratios[3]:.1f}.")
    bind_clause(scopes, "At the final noise level", "At the final noise level, the Min-SNR data weight is only " + scientific_cell(w[-1]) + ".")
    bind_clause(scopes, "These ratios compare", "These ratios compare loss coefficients; the corresponding gradient ratios also depend on the derivatives of the two terms.")

    def bounds(role, timestep, metric, scientific=False):
        rows = [row for report in grid["reports"].values() if report["arrays"]["split"] == role
                for row in report["records"] if (row["residual"], row["estimator"], row["t"]) == ("obs", "S", timestep)]
        require(len(rows) == 3, "requires three final O-S batch gradients per role/timestep")
        values = [row[metric] for row in rows]
        require(all(type(v) in (int, float) and np.isfinite(v) and v >= 0 for v in values), "invalid displayed gradient bounds")
        if scientific:
            return " to ".join(scientific_cell(v) for v in (min(values), max(values)))
        precision = 0 if timestep == 100 else 4
        return " to ".join(f"{v:.{precision}f}" for v in (min(values), max(values)))

    ratio = "physics_to_data_norm_ratio"
    bind_clause(scopes, "At the final O-S checkpoints", "At the final O-S checkpoints, the norm of the residual gradient divided by the norm of the data gradient ranges from "
                + bounds("TRAIN", 100, ratio) + " on TRAIN and " + bounds("VAL", 100, ratio)
                + " on VAL at $t=100$.")
    bind_clause(scopes, clause_pattern(prose_text("At $t=25$", source=True)), "At $t=25$, the ranges are " + bounds("TRAIN", 25, ratio)
                + " and " + bounds("VAL", 25, ratio) + ".")
    bind_clause(scopes, "Each range spans", "Each range spans three seeds, with one calculation on eight windows per seed.")
    bind_clause(scopes, "The absolute TRAIN", "The absolute TRAIN residual-gradient norms at $t=100$ are "
                + bounds("TRAIN", 100, "physics_gradient_norm", scientific=True)
                + ", much smaller than the " + bounds("TRAIN", 1, "physics_gradient_norm") + " range at $t=1$.")
    bind_clause(scopes, "Thus the large ratio at high noise", "Thus the large ratio at high noise accompanies a small absolute gradient.")
    bind_clause(scopes, "These measurements", "These measurements precede clipping of the total gradient, and optimizer updates combine noise levels. The residual term's contribution to prediction quality remains unresolved.")


def cancellation_claims(tex, rendered, record):
    """Bind published TRAIN cutoffs and validated TEST reductions, not a TRAIN refit."""
    scopes = claim_subsection(tex, rendered, "Directional cancellation with thresholds from TRAIN")
    # A page-top annual table can interrupt the seed-sign sentence. Rejoin only
    # its complete source-matching caption, header and rows; the detailed-table
    # gate separately binds these cells to the numerical evidence.
    text = prose_text(scopes[1])
    interruptions = list(re.finditer(r"all three seed\s+(table \d+:[\s\S]*?)\s+(?=differences are)", text))
    if interruptions:
        require(len(interruptions) == 1, "ambiguous annual table in cancellation prose")
        labels = re.findall(r"\\label\{(tab:[^}]+)\}", strip_comments(tex))
        require(labels.count("tab:years") == 1, "cancellation interruption requires annual table")
        expected = (f"Table {labels.index('tab:years') + 1}: " + float_caption(tex, "tab:years")
                    + " " + " ".join(" ".join(row) for row in source_table_rows(tex, "tab:years")))
        match = interruptions[0]
        require(re.fullmatch(clause_pattern(prose_text(expected, source=True)), match[1]) is not None,
                "cancellation interruption table differs")
        scopes = (scopes[0], text[:match.start()] + "all three seed " + text[match.end():])
    cutoffs, groups = record["cutoffs"], record["groups"]
    n = record["arrays"]["TRAIN"]["n_windows"]
    for cutoff in cutoffs.values():
        agree(cutoff["n_training_windows"], n)
    for anchor, expected in (
        ("Group membership uses", "Group membership uses mean $C$ over the 36 available bins or mean $A$ over the twelve hidden bins."),
        ("High cancellation means", "High cancellation means strictly above the TRAIN 75th percentile: "
         f"{cutoffs['context_cancellation']['values'][0]:.6f} for the available intervals and {cutoffs['target_absolute_cancellation']['values'][0]:.6f} for the missing interval."),
        (r"All \d+ overlapping training windows", f"All {n} overlapping training windows contribute equally to these thresholds; normalization instead uses distinct bins."),
        ("Although the rules were specified", "Although the rules were specified earlier, they were implemented after inspection of the primary results, so the analysis remains exploratory."),
        ("TRAIN quartile thresholds", "TRAIN quartile thresholds are (" + ",".join(f"{v:.6f}" for v in cutoffs["context_energy"]["values"])
         + ") for the available intervals and (" + ",".join(f"{v:.6f}" for v in cutoffs["target_energy"]["values"]) + ") for the missing interval."),
        ("A tie enters", "A tie enters the higher energy group, whereas a tie at the cancellation threshold enters the rest group."),
        ("The analysis covers all", f"The analysis covers all {len(groups)} groups: the full test set, high cancellation and the rest, and combinations of cancellation with energy conditions. Group membership and scores for all {len(record['methods'])} procedures are recorded."),
        ("Each score gives equal weight", "Each score gives equal weight to the months present, then averages the three diffusion seeds."),
    ):
        bind_clause(scopes, anchor, expected)
    ch, cr, th, tr, sparse = [groups[k] for k in ("context_high", "context_rest", "target_high", "target_rest", "target_q1_high")]
    pair = ch["comparisons"]["O-S_raw_minus_D_raw"]
    require(set(pair["seed_diffs"]) == {"7", "13", "42"} and all(v < 0 for v in pair["seed_diffs"].values()),
            "context-high fixed-seed differences must all be negative")
    def interval(pair):
        require(pair["year_95"] is not None and len(pair["year_95"]) == 2, "displayed cancellation interval unavailable")
        return "[" + ",".join(f"{v:.4f}" for v in pair["year_95"]) + "]"
    # Preserve synthetic population counts rather than borrowing production data.
    def count(value):
        return {5: "five", 7: "seven", 20: "twenty"}.get(value, str(value))
    bind_clause(scopes, "High cancellation in the available measurements", f"High cancellation in the available measurements identifies {ch['n_windows']} windows across all {count(ch['n_years'])} years. "
                f"O-S minus D is {pair['mean_diff']:.4f}, with a 95\\% interval of {interval(pair)} from resampling years.")
    includes = pair['year_95'][0] <= 0 <= pair['year_95'][1]
    bind_clause(scopes, clause_pattern("all three seed differences"), "All three seed differences are negative, but the interval still "
                + ("includes" if includes else "excludes") + " zero.")
    g5 = ch["comparisons"]["O-S_raw_minus_G5_raw"]
    bind_clause(scopes, "O-S minus G5 is", f"O-S minus G5 is {g5['mean_diff']:.4f}, with interval {interval(g5)}, "
                f"while O-S minus D is {cr['comparisons']['O-S_raw_minus_D_raw']['mean_diff']:.4f} in the remaining {cr['n_windows']} windows.")
    target = th["comparisons"]["O-S_raw_minus_D_raw"]
    bind_clause(scopes, "High cancellation during the missing hour", f"High cancellation during the missing hour identifies {th['n_windows']} windows. "
                f"O-S minus D is {target['mean_diff']:.4f}, with interval {interval(target)}, "
                f"compared with {tr['comparisons']['O-S_raw_minus_D_raw']['mean_diff']:.4f} in the remaining {tr['n_windows']} windows.")
    for anchor, expected in (
        ("These results do not establish", "These results do not establish an O-S advantage over both methods that learn from data alone."),
        ("Grouping by the available context", "Grouping by the available context avoids selection on the hidden outcome, but it does not isolate cancellation from other correlated conditions."),
        ("Although TRAIN supplies the thresholds", "Although TRAIN supplies the thresholds, the withheld outcome determines which TEST windows enter each group, so the comparison still depends on outcome selection."),
        ("The group combining high cancellation", "The group combining high cancellation in the missing hour with the lowest energy quartile contains only "
         f"{count(sparse['n_windows'])} windows from {count(sparse['n_years'])} years."),
        ("Such comparisons describe", "Such comparisons describe losses on selected TEST cases. A claim of conditional predictive superiority would need an evaluation that accounts for the rule assigning cases to each group."),
    ):
        bind_clause(scopes, anchor, expected)


def detailed_table_values(real, observations):
    """98 rows, 494 data cells / 534 scalars, excluding the identifying seed column."""
    comparison, summary = real
    methods = summary["methods"]
    def metric(label, name):
        return methods[label]["metrics"][name]["month_balanced"]
    tables = {label: {} for label in DETAIL_LABELS}
    for arm in bootstrap.ARMS:
        for seed in (7, 13, 42):
            selected = comparison["selected"][f"{arm}_s{seed}"]
            tables["tab:seeds"][(arm, str(seed))] = [str(selected[k]["generation"]) for k in ("raw", "corrected")]
            tables["tab:seeds"][(arm, str(seed))] += [f"{comparison['seed_means'][arm + '_' + k][str(seed)]:.4f}" for k in ("raw", "corrected")]
            tables["tab:seeds"][(arm, str(seed))] += [f"{100 * methods[arm + '_raw']['seedwise'][str(seed)]['cov90_mean']['month_balanced']:.1f}"]
    contrasts = (("O-S", "D", "raw"), ("O-S", "D", "corrected"), ("O-S", "G5", "raw"),
                 ("O-S", "G5", "corrected"), ("O-S", "I-S", "raw"), ("I-S", "D", "raw"),
                 ("O-S", "Gaussian", "raw"))
    def vector(values, brackets="[]"):
        return brackets[0] + ",".join(f"{v:.4f}" for v in values) + brackets[1]
    for arm, baseline, kind in contrasts:
        row = arm + " - " + baseline + ("^c" if kind == "corrected" else "")
        pair = comparison["comparisons"][arm + "_raw_minus_" + baseline.replace("Gaussian", "gaussian") + "_" + kind]
        tables["tab:contrasts"][row] = [f"{pair['mean_diff']:.4f}", vector(pair["year_95"]),
                                      vector(pair["twoyear_95"]), vector(pair["seed_diffs"], "()")]
    names = ("D", "I-S", "O-S", "G5", "Gaussian", "Interpolation", "Persistence")
    for name in names:
        arm = MAIN_ROWS[name]
        for kind in ("raw", "corrected", *(('corrected_at_raw_checkpoint',) if name == "D" else ())):
            label = arm + "_" + kind
            row = name + " " + ({"corrected": "clipped", "corrected_at_raw_checkpoint": "clipped, raw weights"}.get(kind, kind))
            tables["tab:calibration"][row] = [f"{metric(label, 'crps5_mean'):.4f}",
                f"{100 * metric(label, 'cov50_mean'):.1f}", f"{metric(label, 'width50_mean'):.4f}",
                f"{100 * metric(label, 'cov90_mean'):.1f}", f"{metric(label, 'width90_mean'):.4f}"]
            if kind != "corrected_at_raw_checkpoint":
                tables["tab:alternative"][row] = [f"{metric(label, 'es_vec'):.4f}", f"{metric(label, 'es6'):.4f}",
                    f"{1e12 * metric(label, 'proxy_crps'):.4f}", f"{100 * metric(label, 'proxy_cov90'):.1f}",
                    f"{1e12 * metric(label, 'proxy_width90'):.4f}"]
        coverage = np.asarray(metric(arm + "_raw", "cov90_by_position_channel"))
        require(coverage.shape == (12, 5), "coverage position/channel axes differ")
        tables["tab:channelcoverage"][name] = [f"{100 * v:.1f}" for v in coverage.mean(axis=0)]
        tables["tab:geometry"][name] = [f"{metric(arm + '_raw', k):.4f}" for k in
            ("variogram", "seam_l_mean", "seam_r_mean", "internal_mean", "seam_l_rmse", "seam_r_rmse")]
    tables["tab:roundtrip"] = {k: [scientific_cell(v) for v in values]
                               for k, values in observation_roundtrips(observations).items()}
    strata = {"Wind (52)": "imf_source_52", "ACE (71)": "imf_source_71"}
    strata.update({f"{lo}--{hi}": f"years_{lo}_{hi}" for lo, hi in diagnostics.YEAR_GROUPS})
    for word, feature in (("Magnitude", "hidden_mean_m"), ("Increment", "hidden_mean_vector_step")):
        strata.update({f"{word}, {third} third": f"{feature}_tertile_{i}" for i, third in enumerate(("lower", "middle", "upper"), 1)})
    for name, key in strata.items():
        row = summary["strata"][key]
        pair = row["comparisons"]["O-S_raw_minus_D_raw"]
        require(pair["year_95"] is not None, "displayed stratum has no year interval")
        tables["tab:strata"][name] = [str(row["n_windows"])] + [f"{row['methods'][a + '_raw']['month_balanced']:.4f}" for a in ("D", "O-S", "G5")]
        tables["tab:strata"][name] += [f"{pair['mean_diff']:.4f}", vector(pair["year_95"])]
    for year in bootstrap.YEARS:
        row = summary["strata"][f"year_{year}"]
        tables["tab:years"][str(year)] = [f"{row['methods'][a + '_raw']['month_balanced']:.4f}" for a in ("D", "I-S", "O-S", "G5", "gaussian")]
    return tables


def detailed_tables(tex, rendered, real, observations):
    tables = detailed_table_values(real, observations)
    for label, values in tables.items():
        bind_table(tex, rendered, label, values, ("Arm", "Method", "Contrast", "Procedure", "Representation", "Stratum", "Year"))
    return tables


def validate_data(paper_dir):
    """Check published bytes and declared original-record identities, not execution.

    Original quality/Q15 records are private. Their hashes identify those records;
    they cannot authenticate a reconstruction of private paths or source code.
    """
    root = local(paper_dir, "evidence/data")
    manifest = read_json(local(root, "MANIFEST.json"))
    private_check(manifest)
    agree(manifest["schema_version"], 1)
    require(set(manifest) == {"schema_version", "files", "records", *DATA_DECLARATIONS},
            "unexpected data manifest fields")
    for key, value in DATA_DECLARATIONS.items():
        agree(manifest[key], value, "data." + key)
    names = set(DATA_RECORDS)
    require(set(manifest["files"]) == set(manifest["records"]) == names
            and {p.name for p in root.iterdir()} == names | {"MANIFEST.json"},
            "unfinished public data-audit coverage")
    hash_map(manifest["files"])
    records = {}
    for name, original in DATA_RECORDS.items():
        spec = manifest["records"][name]
        require(set(spec) == {"source_record", "source_sha256"}, "invalid original record binding")
        agree(spec["source_record"], original, name + " original name")
        exporter._digest(spec["source_sha256"])
        public_sha = manifest["files"][name]
        require((public_sha == spec["source_sha256"]) == (name == "inventory.json"),
                "original/public data hash distinction differs: " + name)
        path = local(root, name)
        agree(digest(path), public_sha, name + " public hash")
        records[name] = read_json(path)
        # Data definitions use a standalone division slash, not an absolute path.
        # Scan only a lexical copy; never alter the published record or its hash.
        for text in exporter._strings(records[name]):
            private_check(text.replace(" / ", " divided by "))
    return records


def data_audit(paper_dir, tex, rendered, real):
    records = validate_data(paper_dir)
    quality, q15, inventory = [records[n] for n in DATA_RECORDS]
    years = set(map(str, bootstrap.YEARS))
    require(set(inventory["hashes"]) == years, "data inventory requires all twenty raw years")
    hash_map(inventory["hashes"])
    for record in (quality, q15):
        require(set(record["raw_inputs"]) == years, "data audit requires all twenty raw years")
        for year, row in record["raw_inputs"].items():
            require(set(row) == {"path", "sha256", "n_minutes"}, "raw input fields differ")
            agree(row["path"], f"omni_min{year}.asc", "anonymous raw basename")
            require(type(row["n_minutes"]) is int and row["n_minutes"] > 0, "invalid raw minute count")
            exporter._digest(row["sha256"])
        agree({y: record["raw_inputs"][y]["sha256"] for y in years}, inventory["hashes"])
    agree(quality["raw_inputs"], q15["raw_inputs"], "raw input records")
    agree(real[1]["input_manifest"]["recorded_raw_sha256_by_year"], inventory["hashes"])
    for key in ("train_starts", "train_windows_saved", "train_bin_union", "val_selected", "test_selected"):
        require(type(inventory[key]) is int and inventory[key] > 0, "invalid inventory count")
        agree(real[1]["input_manifest"]["inventory_counts"][key], inventory[key])
    agree(inventory["train_windows_saved"], inventory["train_starts"])
    for key, total in (("train_starts_by_year", "eligible_train_starts"),
                       ("candidate_train_grid_starts_by_year", "candidate_train_grid_starts")):
        require(set(quality[key]) == years, "incomplete retention years")
        require(all(type(n) is int and n >= 0 for n in quality[key].values()), "invalid yearly count")
        agree(quality[total], sum(quality[key].values()))
    agree(quality["eligible_train_starts"], inventory["train_starts"])
    agree(quality["eligible_bins_scanned"], inventory["train_bin_union"])
    fraction = quality["eligible_train_starts"] / quality["candidate_train_grid_starts"]
    agree(quality["eligible_start_frac_of_candidate_grid"], fraction)
    agree(q15["schema_version"], 2)
    agree(q15["selection"]["method"], "all")
    agree(q15["selection"]["max_blocks"], None)
    agree(q15["selection"]["blocks_sha256"], q15["population_blocks_sha256"])
    agree(q15["n_blocks"], q15["population_n_blocks"])
    require(set(q15["population_by_year"]) == set(q15["selection"]["by_year"]) == years, "incomplete Q15 years")
    exporter._digest(q15["population_blocks_sha256"])
    source_counts = {}
    for year, row in q15["population_by_year"].items():
        require(all(type(row[k]) is int and row[k] >= 0 for k in
                    ("eligible_blocks", "eligible_train_starts", "eligible_train_bins")), "invalid Q15 year counts")
        agree(row["eligible_train_starts"], quality["train_starts_by_year"][year])
        require(row["eligible_train_starts"] <= quality["candidate_train_grid_starts_by_year"][year],
                "eligible starts exceed candidate grid")
        counts = row["source_block_counts"]
        require(bool(counts) and set(counts) <= {"50", "52", "60", "71"}
                and all(type(n) is int and n >= 0 for n in counts.values()), "invalid Q15 source counts")
        agree(row["eligible_blocks"], sum(counts.values()))
        agree(q15["selection"]["by_year"][year], {"n_blocks": row["eligible_blocks"], "source_block_counts": counts})
        for source, n in counts.items():
            source_counts[source] = source_counts.get(source, 0) + n
    agree(q15["population_source_block_counts"], source_counts)
    agree(inventory["train_bin_union"], sum(row["eligible_train_bins"] for row in q15["population_by_year"].values()))
    agree(q15["n_blocks"], sum(row["eligible_blocks"] for row in q15["population_by_year"].values()))
    require(type(q15["n_blocks"]) is int and q15["n_blocks"] > 0
            and type(q15["max_abs_err_vs_raw"]) in (int, float)
            and 0 <= q15["max_abs_err_vs_raw"] < 1e-8, "Q15 identity check failed")
    flat = " ".join(tex.split())
    counts = [quality[k] for k in ("eligible_train_starts", "candidate_train_grid_starts", "train_role_minutes",
              "train_valid_magnetic_minutes", "dbot_known_minutes", "dbot_negative_minutes",
              "bins_with_any_negative_dbot", "eligible_bins_scanned", "distinct_potential_target_bins")]
    counts += [q15["n_blocks"], quality["train_starts_by_year"]["2005"], quality["train_starts_by_year"]["2014"]]
    for value in counts:
        source = f"{value:,}".replace(",", "{,}")
        require(source in flat and not assert_rendered([str(value)], rendered), "data-audit count missing or changed: " + str(value))
    require(f"{100 * fraction:.2f}\\%" in flat, "retention percentage differs")
    m, e = f"{q15['max_abs_err_vs_raw']:.2e}".split("e")
    require(m + r"\times10^{" + str(int(e)) + "}" in flat, "Q15 error differs")
    require(f"{q15['mean_only_loss_mean_nT2']:.4f}" in flat, "Q15 mean-only loss differs")


ILLUSTRATION_FIGURES = {
    "figures/fixed_final/fig_fixed_final.pdf",
    "figures/imputation/case_120/fig_imputation.pdf",
    "figures/imputation/case_360/fig_imputation.pdf",
    "figures/imputation/detail/fig_imputation.pdf",
}


def illustrations_claimed(paper_dir, tex):
    return (local(paper_dir, "evidence/imputation").exists()
            or bool(re.search(r"fig_fixed_final|fig_imputation|app:imputation|fig:fixedfinal", strip_comments(tex))))


def illustration_claims(tex, rendered):
    table_form = tex.replace(r"\begin{figure}", r"\begin{table}").replace(r"\end{figure}", r"\end{table}")
    labels = {
        "fig:fixedfinal": "figures/fixed_final/fig_fixed_final.pdf",
        "fig:imputation120": "figures/imputation/case_120/fig_imputation.pdf",
        "fig:imputation360": "figures/imputation/case_360/fig_imputation.pdf",
        "fig:imputationdetail": "figures/imputation/detail/fig_imputation.pdf",
    }
    for label, path in labels.items():
        block = float_block(table_form, label)
        require(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", block) == [path],
                "illustration label and image differ: " + label)
        float_caption(table_form, label)
    # The prose and window-120 caption now share a prefix. Exclude only the
    # complete valid caption sentence; malformed or duplicate claims still count.
    timing_anchor = clause_pattern("times mark the") + r"(?!\s+" + clause_pattern(
        "starts of five-minute bins relative to the start of the missing hour.") + ")"
    bind_clause((tex, rendered), timing_anchor,
                "Times mark the starts of five-minute bins relative to the missing hour.")
    # A sentence-start prefix distinguishes this interval definition from
    # the learned-toy and undercoverage mentions without anchoring on its claim.
    bind_clause((tex, rendered), r"(?:(?<=\.\s)|^)" + clause_pattern("the nominal"),
                "The nominal 90\\% interval includes the endpoints given by the second and thirty-eighth values in increasing order.")
    clauses = (
        ('D, O-S and G5 share a linear scale', 'D, O-S and G5 share a linear scale within each row;'),
        ('Gaussian regression uses a separate', 'Gaussian regression uses a separate linear scale on the right.'),
        ('Vertical distances and band heights', 'Vertical distances and band heights are comparable only on the same scale.'),
        ('The axis limits include', 'The axis limits include all measurements, medians and interval ends in each subplot.'),
        ('The horizontal gray line', 'The horizontal gray line marks zero;'),
        ('predictions below it', 'predictions below it violate the nonnegativity of variance.'),
        ('In both examples, diffusion', 'In both examples, diffusion medians generally exceed the withheld variance and the nominal intervals often miss it.'),
        ('In TEST window 120, each diffusion method', 'In TEST window 120, each diffusion method covers $v_s$ in zero of the twelve hidden bins.'),
        ('In window 360, D, O-S', 'In window 360, D, O-S and G5 cover zero, one and two bins, respectively.'),
        ('Gaussian regression covers all', 'Gaussian regression covers all twelve bins in both windows with much wider intervals.'),
        ('It uses the same residual covariance', 'It uses the same residual covariance for every context, and its distribution assigns probability to negative $v_s$, which is physically impossible.'),
        ('Gaussian regression covers all twelve bins in both windows with much wider intervals', 'Gaussian regression covers all twelve bins in both windows with much wider intervals. It uses the same residual covariance for every context, and its distribution assigns probability to negative $v_s$, which is physically impossible.'),
        ('The aggregate results also', 'The aggregate results also show undercoverage of the variance. How often the particular errors in these examples occur across TEST, and how they arise during training, remain uncertain.'),
        ('Windows 120 and 360', 'Windows 120 and 360, counting from zero, are the midpoints of the two halves of the ordered list of 480 windows.'),
        ('The windows were chosen by position', 'The windows were chosen by position, independently of model error, after TEST had already been examined.'),
        ('Diffusion uses seed 7, the first', 'Diffusion uses seed 7, the first declared seed, with model weights chosen from raw validation samples: training step 9560 for D and 29636 for O-S and G5.'),
        ('Gaussian regression uses the fit', 'Gaussian regression uses the fit chosen by validation and the noise realization from seed 13.'),
        ('How representative these two missing hours', 'How representative these two missing hours are across TEST remains uncertain.'),
        ('At each time and coordinate', 'At each time and coordinate, the colored curve is the marginal median of the 39 samples.'),
        ('The curves and intervals', 'The curves and intervals describe each coordinate at each time separately; simultaneous coverage of a trajectory requires separate assessment.'),
        ('Connecting the medians', 'Connecting the medians can produce an irregular curve that differs from every sampled trajectory.'),
        ('Thin lines show three sampled', 'Thin lines show three sampled trajectories, ensemble members 0, 1 and 2.'),
        ('Predictions are shown', 'Predictions are shown before clipping.'),
        ('The symmetric logarithmic scale', 'The symmetric logarithmic scale for $v_s$ displays small measured variances and Gaussian predictions outside the physical range.'),
        ('The task reconstructs statistics', 'The task reconstructs statistics over five minutes; different sequences of minute averages can produce the same statistics.'),
        ('Lines connect', 'Lines connect the four coefficient settings for each seed; solid black diamonds mark the mean across three seeds.'),
        ('The settings', 'The settings are equally spaced on the horizontal axis.'),
        ('Each model value gives', 'Each model value gives equal weight to the months present.'),
        ('the mean residual decreases as the penalty grows, while', 'the mean residual decreases as the penalty grows, while the mean energy score first improves and then worsens.'),
        ('Reconstruction of TEST window 120', 'Reconstruction of TEST window 120 by D, O-S, G5 and Gaussian regression (columns), with model weights chosen by validation.'),
        ('The $v_s$ axis is linear', 'The $v_s$ axis is linear within $0.05$ nT$^2$ of zero and logarithmic outside, so equal vertical distances can represent different absolute errors in that row.'),
        ('Reconstruction of TEST window 360', 'Reconstruction of TEST window 360, using the same methods, seed choices, units and plotting conventions'),
        ('The sampler keeps', 'The sampler keeps the available measurements unchanged.'),
        ('The plots draw', 'The plots draw these measurements and the twelve predicted bins as separate segments, leaving short breaks from $-5$ to $0$ minutes and from $55$ to $60$ minutes.'),
    )
    for anchor, expected in clauses:
        bind_clause((tex, rendered), clause_pattern(prose_text(anchor, source=True)), expected)


def manuscript_mode(tex):
    """Return 'anonymous' or 'camera_ready' from the NeurIPS style option line."""
    has_anon = STYLE_ANONYMOUS in tex
    has_cr = STYLE_CAMERA_READY in tex
    require(has_anon ^ has_cr, "requires exactly one of dblblindworkshop or final,sglblindworkshop")
    return "camera_ready" if has_cr else "anonymous"


def mode_notice(mode):
    return NOTICE_CAMERA_READY if mode == "camera_ready" else NOTICE_ANONYMOUS


def mode_forbidden(mode):
    return FORBIDDEN_CAMERA_READY if mode == "camera_ready" else FORBIDDEN_ANONYMOUS



def source_checks(paper_dir, tex):
    mode = manuscript_mode(tex)
    notice = mode_notice(mode)
    forbidden = mode_forbidden(mode)
    require(not DRAFT.search(tex), "unresolved draft marker")
    identity_text = tex
    if mode == "camera_ready":
        identity_text = re.sub(r"(?<![^\s{])" + re.escape(PUBLIC_COMPANION_URL) + r"(?![^\s}])", "", tex)
    require(not any(s.lower() in identity_text.lower() for s in forbidden), "author-identifying source text")
    require(not re.search(r"/(?:home|Users|tmp)/|[A-Za-z]:\\Users\\", tex), "private source path")
    agree(digest(local(paper_dir, "neurips_2025.sty")), STYLE_SHA256, "canonical vendor style")
    workshop = "2nd Workshop on Principles of Generative Modeling"
    if mode == "camera_ready":
        workshop += " (PriGM)"
    require(r"\workshoptitle{" + workshop + "}" in tex, "wrong workshop title")
    require(r"\renewcommand{\@noticestring}{" + notice + "}" in tex, "requires document-level 2026 notice")
    if mode == "anonymous":
        require(STYLE_ANONYMOUS in tex, "requires dblblindworkshop vendor style")
        require(re.findall(r"\\author\{([^{}]*)\}", tex) == ["Anonymous"], "nonanonymous author declaration")
        require(not re.search(r"\bpdfauthor\s*=\s*\{(?!\s*\})", tex),
                "nonempty pdfauthor is forbidden when anonymous")
    else:
        require(STYLE_CAMERA_READY in tex, "requires final,sglblindworkshop vendor style")
        require(AUTHOR_CAMERA_READY in tex and ORCID_CAMERA_READY in tex, "camera-ready author/ORCID missing")
        require(EMAIL_CAMERA_READY in tex, "camera-ready corresponding email missing")
        require({email.casefold() for email in EMAIL_ADDRESS.findall(tex)} == {EMAIL_CAMERA_READY},
                "camera-ready source contains an undeclared email")
        require(all(marker in tex for marker in AFFILIATION_MARKERS), "camera-ready affiliations incomplete")
        require("Alison" not in tex, "PriGM camera-ready must not list Alison J. March")
        require(re.search(r"\bpdfauthor\s*=\s*\{" + re.escape(AUTHOR_CAMERA_READY) + r"\}", tex),
                "camera-ready pdfauthor must be Kevin Lee")
        require(r"\usepackage{orcidlink}" in tex, "camera-ready requires orcidlink")
    require(tex.count(r"\bibliography{references}") == 1 and tex.count(r"\appendix") == 1
            and tex.index(r"\bibliography{references}") < tex.index(r"\appendix"), "references must precede appendix")
    require(not re.search(r"\\(?:input|include|write18|IfFileExists)\b|\\begin\{comment\}|\\iffalse\b", tex),
            "unsupported hidden/external TeX; cannot bind source")
    extension = deadline_claimed(paper_dir, tex)
    if extension:
        require(local(paper_dir, "evidence/deadline/MANIFEST.json").is_file(), "claimed deadline evidence is missing")
    figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)
    expected_figures = {"figures/hmm/fig_hmm.pdf"} | ({"figures/operator/fig_operator.pdf"} if extension else set())
    if illustrations_claimed(paper_dir, tex):
        require(extension and local(paper_dir, "evidence/deadline/fixed-final.json").is_file(),
                "illustrations require complete fixed-final evidence")
        require(local(paper_dir, "evidence/imputation/MANIFEST.json").is_file(),
                "imputation illustration evidence is missing")
        expected_figures |= ILLUSTRATION_FIGURES
    from experiments import _prigm_camera_ready as camera_ready
    if camera_ready.claimed(paper_dir, tex):
        expected_figures |= camera_ready.FIGURES
    require(len(figures) == len(expected_figures) and set(figures) == expected_figures,
            "primary figure inventory contains missing, historical or unbound assets")
    for name in figures:
        local(paper_dir, name).read_bytes()
        manifest = read_json(local(paper_dir, str(Path(name).with_suffix(".manifest.json"))))
        agree(manifest.get("style"), {"path": "prigm.mplstyle",
                                      "sha256": digest(local(paper_dir, "prigm.mplstyle"))},
              "figure shared style: " + name)
    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    require(len(labels) == len(set(labels)), "duplicate source labels")
    refs = re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", tex)
    require(set(refs).issubset(labels), "undefined source reference")
    table_labels = set(re.findall(r"\\label\{(tab:[^}]+)\}", tex))
    expected_tables = TABLE_LABELS | ({"tab:operators"} if extension else set())
    if extension and local(paper_dir, "evidence/deadline/fixed-final.json").is_file():
        expected_tables = expected_tables | {"tab:fixedfinal"}
    require(table_labels == expected_tables, "mandatory tables absent or extra tables unbound")
    require(len(re.findall(r"\\begin\{table\*?\}", tex)) == len(expected_tables), "unlabelled or duplicate table")


def citation_checks(paper_dir, tex, rendered):
    bib = strip_comments(local(paper_dir, "references.bib").read_text())
    require(not any(s.lower() in bib.lower() for s in FORBIDDEN_EVIDENCE), "author-identifying bibliography")
    keys = re.findall(r"^\s*@(?!(?:comment|string|preamble)\b)\w+\s*\{\s*([^,]+),", bib, re.M | re.I)
    cited = {key.strip() for group in re.findall(r"\\cite[a-zA-Z]*\*?(?:\[[^]]*\])*\{([^}]*)\}", tex)
             for key in group.split(",")}
    require(bool(cited) and len(keys) == len(set(keys)) and set(keys) == cited, "undefined, duplicate or uncited bibliography entry")
    require(not re.search(r"\[\s*\?+\s*\]|\?\?", rendered), "unresolved PDF citation/reference")
    require("References" in rendered, "PDF bibliography absent")
    # Require actual rendered bibliography entries, not just a References heading.
    references = pdf_without_line_numbers(rendered).split("References", 1)[1].replace("-", "").replace("\u2019", "'")
    for key in cited:
        entry = re.search(r"@\w+\s*\{\s*" + re.escape(key) + r"\s*,([\s\S]*?)(?=\n\s*@|\Z)", bib)[1]
        title = re.search(r"\btitle\s*=\s*\{", entry, re.I)
        require(title is not None, "citation missing title: " + key)
        start, depth, end = title.end(), 1, title.end()
        while end < len(entry) and depth:
            depth += (entry[end] == "{") - (entry[end] == "}")
            end += 1
        require(depth == 0 and not assert_rendered([clean_cell(entry[start:end - 1]).replace("-", "")],
                                                  " ".join(references.split())),
                "cited title absent from PDF: " + key)


def pdf_without_line_numbers(text):
    """Remove the vendor review line-number column, retaining page separators."""
    return re.sub(r"(?m)^[ \t]{0,6}(?!20(?:0[5-9]|1\d|2[0-4])\s)\d{1,4}[ \t]{2,}(?=\S)",
                  "", text.replace("\f", "\f\n"))


def pdf_checks(paper_dir):
    pdf = local(paper_dir, "main.pdf")
    require(pdf.is_file() and shutil.which("pdftotext") and shutil.which("pdfinfo"), "requires main.pdf, pdftotext and pdfinfo")
    def run(command):
        return subprocess.run(command, capture_output=True, text=True, check=True, timeout=30).stdout
    rendered = run(["pdftotext", "-layout", str(pdf), "-"])
    info = run(["pdfinfo", str(pdf)])
    links = run(["pdfinfo", "-url", str(pdf)])
    tex = strip_comments(local(paper_dir, "main.tex").read_text())
    mode = manuscript_mode(tex)
    notice = mode_notice(mode)
    forbidden = mode_forbidden(mode)
    require(not DRAFT.search(rendered), "unresolved PDF draft marker")
    identity_text = rendered
    if mode == "camera_ready":
        # PDF text must contain the complete literal URL. Metadata stays unexempted.
        identity_text = re.sub(r"(?<!\S)" + re.escape(PUBLIC_COMPANION_URL) + r"(?!\S)", "", rendered)
    # URI output has no exception: even the exact clickable companion is refused.
    # pdfinfo prints embedded URI newlines verbatim, so rows cannot prove exact targets.
    require(not any(s.lower() in (identity_text + info + links).lower() for s in forbidden),
            "author-identifying PDF text/metadata/link")
    require(not re.search(r"/(?:home|Users|tmp)/|[A-Za-z]:\\Users\\", rendered + info + links), "private PDF path")
    author = re.search(r"^Author:[ \t]*(.*)$", info, re.M)
    if mode == "anonymous":
        require(author is None or author[1].strip() in ("", "Anonymous"), "nonanonymous PDF author metadata")
    else:
        require(author is not None and author[1].strip() == AUTHOR_CAMERA_READY,
                "camera-ready PDF author metadata must be Kevin Lee")
        require(AUTHOR_CAMERA_READY in rendered and EMAIL_CAMERA_READY in rendered,
                "camera-ready PDF missing author or correspondence email")
        require({email.casefold() for email in EMAIL_ADDRESS.findall(rendered + info + links)} == {EMAIL_CAMERA_READY},
                "camera-ready PDF contains an undeclared email")
        require(all(marker in rendered for marker in AFFILIATION_MARKERS),
                "camera-ready PDF missing affiliations")
    require(notice in " ".join(rendered.split()), "PDF workshop notice differs")
    titles = re.findall(r"\\title\{([^{}]+)\}", tex)
    title_info = re.search(r"^Title:[ \t]*(.*)$", info, re.M)
    require(len(titles) == 1 and title_info is not None
            and clean_cell(titles[0]) == title_info[1].strip()
            and not assert_rendered([clean_cell(titles[0])], " ".join(pdf_without_line_numbers(rendered).split())),
            "source/PDF title or title metadata differs")
    pages = pdf_without_line_numbers(rendered).split("\f")
    if not pages[-1].strip():
        pages.pop()
    count = re.search(r"^Pages:\s*(\d+)\s*$", info, re.M)
    require(count is not None and int(count[1]) == len(pages), "PDF page metadata disagrees with text")
    headings = [(i, j) for i, page in enumerate(pages) for j, line in enumerate(page.splitlines())
                if re.fullmatch(r"\s*References\s*", line)]
    require(len(headings) == 1, "missing or ambiguous PDF References heading")
    index, line = headings[0]
    before = [s.strip() for s in pages[index].splitlines()[:line] if s.strip() and not s.strip().isdigit()]
    require(index + bool(before) <= 4, "main text exceeds four pages")
    appendix_source = re.split(r"\\appendix\b", tex)
    require(len(appendix_source) == 2, "missing or ambiguous source appendix")
    section = re.search(r"\\section\s*\{", appendix_source[1])
    require(section is not None, "source appendix section absent")
    start, depth, end = section.end(), 1, section.end()
    while end < len(appendix_source[1]) and depth:
        depth += (appendix_source[1][end] == "{") - (appendix_source[1][end] == "}")
        end += 1
    require(depth == 0, "unclosed source appendix section")
    title = clean_cell(appendix_source[1][start:end - 1])
    require(bool(title.strip()), "empty source appendix section")
    pattern = r"[ \t]*A[ \t]+" + r"[ \t]+".join(re.escape(word) for word in title.split()) + r"[ \t]*"
    appendix = [(i, j) for i, page in enumerate(pages) for j, text in enumerate(page.splitlines())
                if re.fullmatch(pattern, text)]
    require(len(appendix) == 1, "PDF appendix heading absent or duplicated")
    appendix_page, appendix_line = appendix[0]
    require(appendix_page > headings[0][0], "PDF appendix must start on a new page after references")
    before = [s.strip() for s in pages[appendix_page].splitlines()[:appendix_line]
              if s.strip() and s.strip() != str(appendix_page + 1)]
    require(not before, "PDF appendix page contains earlier body content")
    return rendered


def verify(paper_dir=PAPER):
    """Return a Report; every missing mandatory component contributes a failure."""
    paper_dir = Path(paper_dir).resolve()
    report = Report()
    tex = report.check("read manuscript", lambda: strip_comments(local(paper_dir, "main.tex").read_text()))
    rendered = report.check("PDF, authorship mode and four-page boundary", lambda: pdf_checks(paper_dir))
    if tex is not None:
        report.check("source, template, draft refusal and mandatory tables", lambda: source_checks(paper_dir, tex))
    if tex is not None and rendered is not None:
        report.check("citations resolve in source and PDF", lambda: citation_checks(paper_dir, tex, rendered))
    real = report.check("public real evidence, identities and all primary comparisons", lambda: validate_real(paper_dir))
    toy = report.check("stationary pooled toy, raw/source hashes and figure", lambda: validate_toy(paper_dir))
    learned = report.check("all nine learned toy fits and same-stream atomic reference", lambda: validate_learned_toy(paper_dir))
    observations, training_strata, deadline = None, None, None
    extension = deadline_claimed(paper_dir, tex or "")
    if real is not None:
        observations = report.check("byte-identical TEST observations, IDs and TRAIN transform",
                                    lambda: validate_observations(paper_dir, real))
    else:
        report.add("TEST observations", False, "requires real evidence")
    if extension:
        if real is not None and observations is not None:
            deadline = report.check("deadline evidence: paired operators, Gaussian replay and complete fixed-final grid",
                                    lambda: validate_deadline(paper_dir, real, observations))
        else:
            report.add("deadline evidence", False, "requires original evidence and TEST observations")
    if illustrations_claimed(paper_dir, tex or ""):
        if deadline is not None and observations is not None:
            from experiments._prigm_illustrations import validate_illustrations
            report.check("audited imputation subset and all new figure coordinates",
                         lambda: validate_illustrations(paper_dir, observations))
        else:
            report.add("illustration evidence", False, "requires original observations and verified deadline evidence")
        if tex is not None and rendered is not None:
            report.check("illustration selection, intervals and figure-caption conventions",
                         lambda: illustration_claims(tex, rendered))
        else:
            report.add("illustration claims", False, "requires manuscript and PDF")
    from experiments import _prigm_camera_ready as camera_ready
    if camera_ready.claimed(paper_dir, tex or ""):
        from experiments import _prigm_checkpoint_evidence as checkpoints
        checkpoint_data, checkpoint_figure = None, None
        if real is not None:
            checkpoint_data = report.check("camera-ready checkpoint provenance and all ninety score vectors",
                lambda: checkpoints.validate_checkpoints(paper_dir, real))
        else:
            report.add("camera-ready checkpoint diagnostics", False, "requires original evidence")
        if checkpoint_data is not None:
            checkpoint_figure = report.check("camera-ready checkpoint figure: complete trajectories and VAL choices",
                lambda: checkpoints.validate_checkpoint_figure(paper_dir, checkpoint_data))
        if checkpoint_figure is not None and tex is not None and rendered is not None:
            report.check("camera-ready checkpoint claims and scope in source and PDF",
                lambda: camera_ready.checkpoint_claims(tex, rendered, checkpoint_data))
        calibration, calibration_figure = None, None
        if real is not None and deadline is not None:
            calibration = report.check("camera-ready tail diagnostics: all models, months and paired intervals",
                                       lambda: camera_ready.validate_calibration(paper_dir, real))
        else:
            report.add("camera-ready tail diagnostics", False, "requires original and deadline evidence")
        if calibration is not None:
            calibration_figure = report.check("camera-ready calibration figure: seedwise terms and tail coordinates",
                lambda: camera_ready.validate_calibration_figure(paper_dir, calibration))
        if calibration_figure is not None and tex is not None and rendered is not None:
            report.check("camera-ready calibration claims and scope in source and PDF",
                         lambda: camera_ready.calibration_claims(tex, rendered, calibration, calibration_figure))
    if real is not None and observations is not None:
        report.check("TEST tertiles, group IDs and frozen-score statistics",
                     lambda: validate_test_strata(paper_dir, real, observations))
        training_strata = report.check("TRAIN sensitivity provenance and all 21 TEST groups",
                                       lambda: validate_sensitivity(paper_dir, real, observations))
        if tex is not None and rendered is not None:
            report.check("nine detailed tables: 98 rows, 494 cells / 534 scalars",
                         lambda: detailed_tables(tex, rendered, real, observations))
        else:
            report.add("nine detailed tables", False, "requires manuscript and PDF")
    else:
        report.add("nine detailed tables, TEST strata and TRAIN sensitivity", False, "requires observations and real evidence")
    if tex is not None and rendered is not None and real is not None:
        report.check("main table: every procedure, cell and caption", lambda: main_table(tex, rendered, real, deadline))
        if extension:
            if deadline is not None:
                report.check("deadline tables, correction terms, Gaussian contrasts and explanatory clauses",
                             lambda: deadline_claims(tex, rendered, real, deadline))
            else:
                report.add("deadline tables and claims", False, "requires verified extension evidence")
        report.check("public data-audit coverage", lambda: data_audit(paper_dir, tex, rendered, real))
        report.check("relative residual weight: coefficients and final batch-gradient ranges",
                     lambda: residual_weight_claims(paper_dir, tex, rendered))
    else:
        report.add("main table and public data-audit coverage", False, "unfinished: requires manuscript, PDF and real evidence")
    if tex is not None and rendered is not None and toy is not None:
        report.check("toy table and main-text numerical claims", lambda: toy_claims(tex, rendered, toy))
    else:
        report.add("toy table and main-text numerical claims", False, "unfinished: requires manuscript, PDF and toy evidence")
    if tex is not None and rendered is not None and learned is not None:
        report.check("learned toy table: 45 cells, composite IDs, caption and atomic coverage paragraph",
                     lambda: learned_toy_claims(tex, rendered, learned))
    else:
        report.add("learned toy table and coverage paragraph", False, "requires manuscript, PDF and learned evidence")
    if tex is not None and rendered is not None and training_strata is not None:
        report.check("training-defined cancellation groups: cutoffs, contrasts and scope clauses",
                     lambda: cancellation_claims(tex, rendered, training_strata))
    else:
        report.add("training-defined cancellation group clauses", False, "requires manuscript, PDF and sensitivity evidence")
    if extension:
        print("Extension coverage: exact public inventory/hashes; operator window algebra/reductions and selection effects; "
              "Gaussian stored-moment fits, VAL reduction/selection, all TEST draws/metrics and fixed-model contrasts; "
              "complete fixed-final grid identities and per-window reductions when present; new tables, figure coordinates and selected clauses. "
              "No neural draw replay, TRAIN/VAL refit, new-seed uncertainty or raw-data reconstruction is claimed.")
    print("Original coverage: twelve tables in source and table-scoped PDF; toy diagnostic prose/figure; "
          "public primary statistics; observation round trips; TEST strata; TRAIN sensitivity bindings and TEST group recomputation; "
          "nine learned toy fits, their table/caption and atomic reference coverage paragraph; "
          "relative residual-weight coefficients and final batch-gradient ranges; cancellation cutoffs, contrasts and selected scope clauses; data-audit counts. "
          "Arbitrary prose, full TRAIN cutoff rederivation, gradient computation, learned ensembles and raw real-data execution are not revalidated.")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, default=PAPER)
    args = parser.parse_args(argv)
    report = verify(args.paper_dir)
    print("OVERALL: " + ("PASS" if report.ok() else "FAIL"))
    return 0 if report.ok() else 1


if __name__ == "__main__":
    raise SystemExit(main())
