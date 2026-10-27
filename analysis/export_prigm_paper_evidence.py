"""Export public numerical records from the completed PriGM follow-up.

Validate the saved study records, recompute comparisons and aggregate
statistics from stored per-window scores and metrics, and publish five files
with local execution paths removed or expressed relative to the study.
Gradient records are checked for identity and consistency; their derivatives
are not recomputed. The exporter does not retrain models, regenerate
predictions, rescore ensemble draws, or repeat checkpoint selection.

The study had already been examined on TEST. This export preserves that
qualification and leaves the original execution records unchanged. It cannot
be used to resume the study. A fresh output directory is complete only when
MANIFEST.json exists and all five output hashes match; interrupted exports
remain available for inspection.
"""

from __future__ import annotations

import argparse
import getpass
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import prigm_followup_summary as diagnostics
from experiments.run_prigm_repair import verify_checkpoint
from prigm import run_bootstrap as bootstrap
from prigm.artifacts import id_hash, json_bytes, require_source_match
from prigm.recovery import ContractMismatch, atomic_write_bytes, sha256_bytes, sha256_file

COMPARISON = "artifacts/comparison.json"
SUMMARY = "artifacts/diagnostic-summary-01.json"
PROTOCOL_VALUES = (
    "exposure", "study", "arms", "seeds", "milestones", "c_r", "batch_size",
    "budget_updates", "samplers", "sampling_chunk_windows", "controls",
    "selection", "claim_scope", "training_data_hashes",
)
# These are public interpretation fields of the frozen schema-2 producer, not
# substitute execution metadata. Keep exact text as well as numeric identities.
COMPARISON_TEXT = {
    "uncertainty_scope": "Paired test-year uncertainty averaged over three fixed trained models; identical year draws for every model and comparison. Not training-seed, refitting or selection uncertainty. Each non-neural control is one shared realization, not three independent baselines.",
    "comparison_scope": "Primary O-S comparisons are distinguished for the preplanned repair of a historically exposed test study, not a restored blind confirmation. Expanded and correction comparisons are exploratory multiple comparisons. Percentile intervals are marginal, not familywise corrected.",
    "interpretation": "Lower primary energy score is better; contrasts are method minus baseline. Relative intervals use each replicate's baseline mean. Two-percent gates are descriptive, not causal verdicts or universal negative claims. Failure to show superiority is not equivalence; the optional equivalence criterion requires both relative intervals strictly inside [-0.02, 0.02]. No p-value establishes equivalence. Block lengths are temporal-dependence sensitivity checks, not a guarantee under solar-cycle nonstationarity.",
}
SCOPE = (
    "Reduced anonymous numeric evidence from a historically test-exposed study. "
    "Not the original execution protocol, a resumable study, or restored blind confirmation. "
    "Primary statistics and full diagnostics are recomputed from loader-validated inputs. "
    "Gradient receipts and their payload hashes are checked, not gradient calculations. "
    "Raw OMNI, training, selection and ensemble generation are not rerun. "
    "Original records remain unchanged and are identified by their byte hashes."
)

GRADIENT_SCOPE = (
    "Path-reduced copies of fixed local diagnostics, not original execution records. "
    "Only run_root and arrays_path are reduced to study-relative paths. All other fields are unchanged. "
    "Weighted gradients of batch means on eight fixed TRAIN or VAL windows, not means "
    "of per-window gradient norms. No gradient computation, refitting or TEST gradients "
    "are performed by this export. These diagnostics do not causally explain energy differences."
)


def _agree(reported, actual, location="record"):
    """Recursive, shape/key-strict equality; booleans and integers are identities."""
    if isinstance(actual, dict):
        if not isinstance(reported, dict) or set(reported) != set(actual):
            raise ContractMismatch(f"{location}: fields differ")
        for key in actual:
            _agree(reported[key], actual[key], f"{location}.{key}")
    elif isinstance(actual, (list, tuple)):
        if not isinstance(reported, (list, tuple)) or len(reported) != len(actual):
            raise ContractMismatch(f"{location}: sequence differs")
        for i, (left, right) in enumerate(zip(reported, actual)):
            _agree(left, right, f"{location}[{i}]")
    elif type(actual) is float:
        if (type(reported) not in (int, float) or not math.isfinite(actual)
                or not math.isfinite(reported)
                or not np.isclose(reported, actual, rtol=1e-12, atol=1e-12)):
            raise ContractMismatch(f"{location}: numeric value differs")
    elif type(reported) is not type(actual) or reported != actual:
        raise ContractMismatch(f"{location}: identity differs")


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ContractMismatch("expected lowercase SHA256")
    return value


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _strings(child)


def _private_names(study, *records):
    # Names known from execution paths are forbidden even when detached from paths.
    names = {getpass.getuser(), Path.home().name, study.name}
    for text in _strings(records):
        names.update(re.findall(r"(?:/home/|/Users/|[A-Za-z]:\\Users\\)([^/\\\s]+)", text))
    return {name.casefold() for name in names if name}


def _safe(value, private_names):
    """Fail closed on path/name/credential strings; never rewrite scientific text."""
    for text in _strings(value):
        folded = text.casefold()
        if (any(re.search(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", folded) for name in private_names)
                or re.search(r"(?<![\w./])(?:/|~/|[A-Za-z]:[\\/])", text)
                or "\\" in text or re.search(r"(?<![\w.])\.\.(?:/|$)", text)
                or "://" in text or "@" in text
                or re.search(r"(?<![a-z0-9])(?:password|passwd|credential\w*|secret|api[_ -]?key|access[_ -]?token|username)(?![a-z0-9])", folded)
                or re.search(r"(?:^|[_ -])(?:user|token|author|owner|login|hostname)(?:$|[_ -])", folded)
                or any(ord(c) < 32 for c in text)):
            raise ContractMismatch("unsafe private path/name or credential string in public evidence")


def _finite_gradient(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ContractMismatch("gradient metric must be finite and numeric")
    return value


def _validate_gradient(report, study, protocol, checkpoint, name, generation, split, arrays):
    """Check recorded identities and geometry domains; never calculate gradients."""
    ids, nrm, identity = arrays
    contract = checkpoint["contract"]
    expected = {
        "schema_version": 1, "diagnostic": "weighted_physics_data_parameter_gradients",
        "generation_updates": generation, "checkpoint_sha256": checkpoint["files"]["state.pt"],
        "current_source_hashes": protocol["inference_source_hashes"],
        "training_contract": contract, "training_source_hashes": contract["source_hashes"],
        "arrays": identity, "c_r": 0.0005,
        "normalization": {"mean": nrm.mean.tolist(), "scale": nrm.scale.tolist(),
                          "b_ref_sq": nrm.b_ref_sq, "n_bins": nrm.n_bins,
                          "sha256": identity["normalization_sha256"]},
    }
    for key, value in expected.items():
        _agree(report[key], value, f"gradient.{key}")
    _agree(diagnostics._relative(report["run_root"], study), f"runs/{name}", "gradient.run_root")
    _agree(diagnostics._relative(report["arrays_path"], study),
           f"artifacts/inventory/{split.lower()}.npz", "gradient.arrays_path")
    indices = np.linspace(0, len(ids) - 1, min(len(ids), 8), dtype=np.int64).tolist()
    selected_ids = [ids[i] for i in indices]
    _agree(report["selection"], {
        "rule": "np.linspace(0, n_available - 1, min(n_available, n_windows), dtype=int64); first index if count is one",
        "n_requested": 8, "n_selected": 8, "indices": indices,
        "ids": selected_ids, "ids_sha256": id_hash(selected_ids),
    }, "gradient.selection")
    if len(selected_ids) != 8:
        raise ContractMismatch("gradient requires eight fixed windows")
    for key, value in {"seed": 7003, "shape": [8, 6, 48], "device": "cpu",
                       "dtype": "torch.float32", "generator": "torch.Generator"}.items():
        _agree(report["noise"][key], value, f"gradient.noise.{key}")
    _digest(report["noise"]["sha256"])
    for key in ("u0_sha256", "observed_sha256"):
        _digest(report["inputs"][key])
    _agree(report["inputs"]["state_convention"], "clean_observed_flanks_hidden_diffusion")
    _agree(report["inputs"]["hidden_slice"], [18, 30])
    for key in ("deterministic_algorithms", "mkldnn_enabled"):
        if type(report["environment"][key]) is not bool:
            raise ContractMismatch("gradient environment flags must be booleans")
    for key, value in {"class": "prigm.model.ConditionalUNet", "widths": contract["widths"],
                       "n_generated": 6, "mode": "eval"}.items():
        _agree(report["model"][key], value, f"gradient.model.{key}")
    if type(report["model"]["n_parameters"]) is not int or report["model"]["n_parameters"] <= 0:
        raise ContractMismatch("gradient parameter count must be a positive integer")
    for key, value in contract["settings"]["diffusion"].items():
        _agree(report["schedule"][key], value, f"gradient.schedule.{key}")
    _agree(report["schedule"]["probe_timesteps"], [1, 25, 50, 100])
    values = report["schedule"]["probe_values"]
    if set(values) != {"a", "s", "w", "sigma"}:
        raise ContractMismatch("gradient schedule arrays differ")
    for row in values.values():
        if not isinstance(row, list) or len(row) != 4:
            raise ContractMismatch("gradient schedule array shape differs")
        for value in row:
            _finite_gradient(value)
    records = report["records"]
    expected_cases = {(t, e, r) for t in (1, 25, 50, 100) for e in ("M", "S") for r in ("inst", "obs")}
    if not isinstance(records, list) or len(records) != len(expected_cases):
        raise ContractMismatch("gradient requires all sixteen Cartesian records")
    seen = set()
    for row in records:
        case = (row["t"], row["estimator"], row["residual"])
        if (type(case[0]) is not int or type(case[1]) is not str or type(case[2]) is not str
                or case not in expected_cases or case in seen):
            raise ContractMismatch("gradient Cartesian record identity differs")
        seen.add(case)
        for key in ("weighted_data_mean", "weighted_physics_mean", "total_mean",
                    "data_gradient_norm", "physics_gradient_norm"):
            if _finite_gradient(row[key]) < 0:
                raise ContractMismatch("gradient loss or norm is negative")
        data_zero, phys_zero = row["data_gradient_norm"] == 0, row["physics_gradient_norm"] == 0
        if phys_zero and not data_zero:
            _agree(row["physics_to_data_norm_ratio"], 0., "gradient.physics_to_data_norm_ratio")
        for key, undefined, low, high in (
            ("physics_to_data_norm_ratio", data_zero, 0, math.inf),
            ("gradient_cosine", data_zero or phys_zero, -1, 1),
            ("gradient_angle_degrees", data_zero or phys_zero, 0, 180),
        ):
            if undefined:
                _agree(row[key], None, f"gradient.{key}")
            elif not low <= _finite_gradient(row[key]) <= high:
                raise ContractMismatch("gradient geometry outside its domain")


def export_study(study_root, protocol_sha256, output_dir):
    """Validate read-only, exclusively create a fresh directory, return its manifest.

    The parent must already exist. Failures after mkdir preserve all partial files;
    an existing directory (complete or partial) is never reused or removed.
    """
    study = Path(study_root).resolve()
    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("output directory must be fresh")
    if not output.parent.is_dir():
        raise FileNotFoundError("output parent must exist")
    # Pin the parent, but do not resolve away a dangling output symlink.
    output = output.parent.resolve() / output.name
    _digest(protocol_sha256)
    source = Path(__file__).resolve()
    source_hash = sha256_file(source)
    analysis_source = Path(diagnostics.__file__).resolve()
    analysis_hash = sha256_file(analysis_source)
    hashes, reads = {}, {}

    def bind(path, digest):
        relative = diagnostics._relative(path, study)
        digest = _digest(digest)
        if relative in hashes and hashes[relative] != digest:
            raise ContractMismatch("input hash bindings disagree")
        hashes[relative] = digest
        # Retain original references too, so retargeted internal symlinks fail.
        reads[str(path)] = relative
        return relative

    def read(relative):
        path = study / relative
        diagnostics._relative(path, study)
        payload = path.read_bytes()
        bind(path, sha256_bytes(payload))
        return json.loads(payload, object_pairs_hook=bootstrap._unique_object), payload

    def check_file(relative, digest):
        path = study / relative
        bind(path, digest)
        if sha256_file(path) != digest:
            raise ContractMismatch("receipt/input hash mismatch")

    protocol, _ = read("PROTOCOL.json")
    check_file("PROTOCOL.json", protocol_sha256)
    require_source_match(protocol)
    pipeline, _ = read("PIPELINE_COMPLETED.json")
    cpu, _ = read("CPU_FOLLOWUP_COMPLETED.json")
    _agree(pipeline["protocol_sha256"], protocol_sha256)
    _agree(cpu["protocol_sha256"], protocol_sha256)
    _agree(cpu["analysis_source_sha256"], analysis_hash)
    for relative, digest in (
        (COMPARISON, pipeline["comparison_sha256"]),
        ("SELECTION.json", pipeline["selection_sha256"]),
        (SUMMARY, cpu["summary_sha256"]),
        ("run_fixed_followup.py", pipeline["launcher_sha256"]),
        ("run_cpu_after_real.py", cpu["launcher_sha256"]),
    ):
        check_file(relative, digest)
    if set(cpu["gradient_completion_sha256"]) != {"2868", "29636"}:
        raise ContractMismatch("requires fixed early/final gradient completions")
    gradients = {"schema_version": 1, "protocol_sha256": protocol_sha256,
                 "scope": GRADIENT_SCOPE, "grids": {}}
    gradient_arrays = {}
    _agree(diagnostics._relative(protocol["array_root"], study), "artifacts/inventory")
    for generation, digest in cpu["gradient_completion_sha256"].items():
        parent = f"artifacts/gradient-g{generation}"
        check_file(f"{parent}/COMPLETED.json", digest)
        receipt, _ = read(f"{parent}/COMPLETED.json")
        check_file(f"{parent}/REQUEST.json", receipt["request_sha256"])
        request, _ = read(f"{parent}/REQUEST.json")
        for key, value in {
            "protocol_sha256": protocol_sha256, "arm": "O-S", "generation": int(generation),
            "seeds": [7, 13, 42], "splits": ["TRAIN", "VAL"], "n_windows_per_split": 8,
            "noise_seed": 7003, "c_r": 0.0005,
        }.items():
            _agree(request[key], value, f"gradient.request.{key}")
        _agree(protocol["c_r"]["O-S"], request["c_r"])
        check_file("run_probe_grid.py", request["launcher_sha256"])
        if set(request["checkpoint_sha256"]) != {"7", "13", "42"}:
            raise ContractMismatch("gradient request requires three checkpoint hashes")
        expected = {f"O-S_s{s}-{split}.json" for s in bootstrap.FINAL_SEEDS for split in ("train", "val")}
        if set(receipt["outputs"]) != expected:
            raise ContractMismatch("gradient completion requires all six reports")
        grid = {"request": request, "source_request_sha256": hashes[f"{parent}/REQUEST.json"],
                "source_completion_sha256": hashes[f"{parent}/COMPLETED.json"], "reports": {}}
        for seed in bootstrap.FINAL_SEEDS:
            name = f"O-S_s{seed}"
            fixed = f"runs/{name}/gen_{int(generation):08d}"
            checkpoint, _ = read(f"{fixed}/COMMIT.json")
            _agree(verify_checkpoint(study, name, int(generation), protocol), checkpoint, "gradient.checkpoint")
            _agree(checkpoint["generation"], int(generation))
            for filename, value in checkpoint["files"].items():
                check_file(f"{fixed}/{filename}", value)
            _agree(request["checkpoint_sha256"][str(seed)], checkpoint["files"]["state.pt"])
            for split in ("TRAIN", "VAL"):
                if split not in gradient_arrays:
                    relative = f"artifacts/inventory/{split.lower()}.npz"
                    check_file(relative, protocol["arrays"][split]["sha256"])
                    gradient_arrays[split] = diagnostics.load_arrays(study / relative, split)[1:]
                    _agree(gradient_arrays[split][2], protocol["arrays"][split])
                filename = f"{name}-{split.lower()}.json"
                check_file(f"{parent}/{filename}", receipt["outputs"][filename])
                report, _ = read(f"{parent}/{filename}")
                _validate_gradient(report, study, protocol, checkpoint, name, int(generation),
                                   split, gradient_arrays[split])
                grid["reports"][filename] = {
                    **report, "arrays_path": diagnostics._relative(report["arrays_path"], study),
                    "run_root": diagnostics._relative(report["run_root"], study),
                }
        gradients["grids"][generation] = grid

    original, _ = read(COMPARISON)
    original_summary, summary_bytes = read(SUMMARY)
    data, months, provenance = bootstrap.load_study(study, protocol_sha256)
    diagnostics.validate_matrices(data, months, provenance["ids"])
    _agree(provenance["arrays"]["ids_sha256"], id_hash(provenance["ids"]))
    _agree(provenance["months"], np.asarray(months).tolist())
    _agree(provenance["months"], [int(w[:4]) * 100 + int(w[5:7]) for w in provenance["ids"]])
    _agree(provenance["protocol_sha256"], protocol_sha256)
    _agree(protocol["seeds"], list(bootstrap.FINAL_SEEDS))
    _agree(protocol["exposure"], "historical_test_exposed")
    _agree(protocol["schema_version"], 2)
    recomputed_summary = diagnostics.summarize_study(study, protocol_sha256)
    _agree(original_summary, recomputed_summary, "diagnostics")
    for key in ("ids", "months", "arrays", "protocol_sha256", "selection_sha256", "test_started_sha256"):
        _agree(original_summary[key], provenance[key], f"diagnostics.{key}")
    _agree(original_summary["analysis_source_sha256"], analysis_hash)
    expected = {
        "schema_version": 2, **provenance, **bootstrap.compare(data, months),
        "exposure": "historical_test_exposed", "n_replicates": bootstrap.BOOTSTRAP_REPLICATES,
        "seed": bootstrap.BOOTSTRAP_SEED, "fixed_training_seeds": list(bootstrap.FINAL_SEEDS),
        "multiplicity": "with_replacement_year_blocks",
        "year_blocks": bootstrap.year_blocks().tolist(), "two_year_blocks": bootstrap.two_year_blocks(),
        "month_counts": {str(y * 100 + m): int(np.count_nonzero(months == y * 100 + m))
                         for y in bootstrap.YEARS for m in bootstrap.TEST_MONTHS},
        **COMPARISON_TEXT,
    }
    _agree(original, expected, "comparison")
    for path, digest in provenance["input_sha256"].items():
        bind(path, digest)
    for path, digest in original_summary["input_sha256"].items():
        bind(study / path, digest)
    for relative, key in (("SELECTION.json", "selection_sha256"), ("TEST_STARTED.json", "test_started_sha256")):
        check_file(relative, provenance[key])

    public_comparison = {k: v for k, v in original.items() if k not in ("study_root", "input_sha256", "source_sha256", "methods")}
    public_comparison["methods"] = {
        label: {**meta, "score_files": {k: diagnostics._relative(p, study) for k, p in meta["score_files"].items()}}
        for label, meta in original["methods"].items()
    }
    public_comparison["source_comparison_sha256"] = hashes[COMPARISON]
    protocol_summary = {
        "schema_version": 1, "source_protocol_sha256": protocol_sha256,
        **{k: protocol[k] for k in PROTOCOL_VALUES},
        "arrays": {split: {k: protocol["arrays"][split][k] for k in diagnostics.ARRAY_FIELDS}
                   for split in ("TRAIN", "VAL", "TEST")},
        "scope": SCOPE,
    }
    scores = {
        "schema_version": 1, "ids": provenance["ids"], "months": np.asarray(months).tolist(),
        "data": {label: values.tolist() for label, values in data.items()},
        "protocol_sha256": protocol_sha256, "fixed_training_seeds": list(bootstrap.FINAL_SEEDS),
        "row_ids": {label: ["shared"] if len(values) == 1 else list(map(str, bootstrap.FINAL_SEEDS))
                    for label, values in data.items()},
        "control_scope": "Each non-neural control has one shared row, broadcast only for paired contrasts; no independent seed replicates.",
    }
    manifest = {
        "schema_version": 1, "exporter_source_sha256": source_hash,
        "analysis_source_sha256": analysis_hash, "scope": SCOPE,
        "original_sha256": {p: hashes[p] for p in ("PIPELINE_COMPLETED.json", "CPU_FOLLOWUP_COMPLETED.json",
                                                   "PROTOCOL.json", COMPARISON, SUMMARY)},
        "input_sha256": hashes, "output_sha256": {},
    }
    names = _private_names(study, str(study), str(source), protocol, original, gradients)
    records = {"comparison.json": public_comparison, "diagnostics.json": original_summary,
               "scores.json": scores, "protocol-summary.json": protocol_summary,
               "gradients.json": gradients}
    for rec in (*records.values(), manifest):
        _safe(rec, names)
        json_bytes(rec)  # Reject nonfinite values even in an unanticipated field.
    payloads = {name: summary_bytes if name == "diagnostics.json" else json_bytes(rec)
                for name, rec in records.items()}
    manifest["output_sha256"] = {name: sha256_bytes(payload) for name, payload in payloads.items()}

    def recheck():
        require_source_match(protocol)
        if sha256_file(source) != source_hash or sha256_file(analysis_source) != analysis_hash:
            raise ContractMismatch("exporter or analysis source changed during export")
        for path, relative in reads.items():
            if diagnostics._relative(path, study) != relative:
                raise ContractMismatch("input path changed during export")
        for relative, digest in hashes.items():
            if diagnostics._relative(study / relative, study) != relative or sha256_file(study / relative) != digest:
                raise ContractMismatch("input changed during export")

    recheck()
    output.mkdir()  # Exclusive race refusal, only after complete validation.
    owned = output.stat()

    def publish(name, payload):
        current = output.stat()
        if output.is_symlink() or (current.st_dev, current.st_ino) != (owned.st_dev, owned.st_ino):
            raise ContractMismatch("output directory changed during publication")
        target = output / name
        if target.exists() or target.is_symlink():
            raise FileExistsError("output file appeared during publication")
        atomic_write_bytes(target, payload)

    for name, payload in payloads.items():
        publish(name, payload)
    recheck()
    for name, digest in manifest["output_sha256"].items():
        if (output / name).is_symlink() or sha256_file(output / name) != digest:
            raise ContractMismatch("output changed before manifest publication")
    publish("MANIFEST.json", json_bytes(manifest))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", required=True, help="fresh directory; parent must exist")
    args = parser.parse_args(argv)
    export_study(args.study_root, args.protocol_sha256, args.output_dir)
    print(json.dumps({"manifest": "MANIFEST.json", "n_outputs": 5}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
