"""Curate complete frozen toy evidence; never train, evaluate, or select a fit.

Schema 1 layout:
* learned.json: identities, reduced protocol/training provenance, ordered ids and
  fit_order; fits[name] has config, checkpoint/generation identities, raw/corrected.
* Each scoring record has scores, mean_energy, month_balanced_energy, secondary
  (metric -> original mean/month_balanced), and ordered per_window metric records.
* reference-calibration.json uses that scoring layout for the SAME reference RNG
  realization, with oracle metadata and original oracle/array identities attached.
  Position/channel arrays have shape (12, 5); all reductions weight 288 windows
  equally. Learned metrics are retained, not rescored from the ensemble samples.
* MANIFEST.json is the completion marker, published last. input_sha256 inventories
  study-relative inputs; original_sha256 separately identifies the external oracle.

These are reduced public records, not rewritten original execution records. No
checkpoint, ensemble, private execution path or environment is redistributed.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from experiments import run_prigm_toy_repair as toy
from prigm import oracle as oracle_module
from prigm.artifacts import json_bytes, load_arrays, read_ensemble
from prigm.checkpoint import load_torch_payload
from prigm.config import ORACLE_REF_SEED
from prigm.keys import pcg64
from prigm.recovery import ContractMismatch, fsync_dir, sha256_bytes, sha256_file
from prigm.run_secondaries import summarize_diagnostics
from prigm.scoring import coverage_inclusive, crps_univariate, energy_score, interval_width
from prigm.toy import ALPHAS, PS, sample_mixture_paths, state_summary

IMPORTED_EXPORTER_SHA256 = sha256_file(Path(__file__).resolve())

SCOPE = (
    "Reduced public provenance and complete numeric evidence, not original execution records "
    "or a resumable study. All nine fixed update-5000 fits and both scoring variants are "
    "retained without score-based selection or seed pooling. Checkpoint payloads, source "
    "pins, chunk unions and score reductions are checked read-only. Learned samples are "
    "not rescored. No training, denoising or new reference realization is performed."
)
PATH_REMOVAL = (
    "Private absolute execution paths and training environments are omitted, not rewritten "
    "as original provenance. Source file identities use repository-relative names; the "
    "input inventory uses study-relative names. Original byte hashes remain authoritative."
)
INTERVAL_PROTOCOL = "seedwise_39_draws_inclusive_order_statistics_2_38_and_10_30"
SCALAR_METRICS = (
    "es5", "es6", "es_vec", "crps5_mean", "cov90_mean", "cov50_mean",
    "width90_mean", "width50_mean", "proxy_crps", "proxy_cov90", "proxy_width90",
    "r_obs_mean", "frac_m_lt_r", "frac_vs_lt0", "frac_vs_gt_hi",
    "frac_support_inequality_violation", "frac_support_violation_beyond_rounding",
    "variogram", "seam_l_mean", "seam_r_mean", "internal_mean", "truth_seam_l",
    "truth_seam_r", "truth_internal", "seam_l_rmse", "seam_r_rmse", "neg_proxy",
)
ARRAY_METRICS = (
    "crps_by_position_channel", "cov90_by_position_channel",
    "width90_by_position_channel", "bias_by_position_channel",
)


def _exact(actual, expected, label="identity"):
    """Identities include JSON types: True is not 1 and 5000.0 is not 5000."""
    if type(actual) is not type(expected):
        raise ContractMismatch(f"{label}: type differs")
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ContractMismatch(f"{label}: fields differ")
        for key in expected:
            _exact(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ContractMismatch(f"{label}: length differs")
        for a, b in zip(actual, expected):
            _exact(a, b, label)
    elif actual != expected:
        raise ContractMismatch(f"{label}: value differs")


def _numeric(value):
    if isinstance(value, list):
        for item in value:
            _numeric(item)
    elif type(value) not in (int, float) or not np.isfinite(value):
        raise ContractMismatch("numeric evidence must be finite numbers, not flags or strings")
    return np.asarray(value, dtype=np.float64)


def _close(actual, expected, label="reduction"):
    a, b = _numeric(actual), np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(b).all() or not np.allclose(a, b, rtol=0, atol=1e-12):
        raise ContractMismatch(f"{label}: differs at absolute tolerance 1e-12")


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


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractMismatch("duplicate JSON key")
        result[key] = value
    return result


def _read_json(path):
    value = json.loads(path.read_bytes(), object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ContractMismatch("JSON object required")
    return value


class _Inputs:
    """Hash before validation and again before the manifest; never follow symlinks."""

    def __init__(self, study):
        self.study = study
        self.hashes = {}
        self.private_names = {getpass.getuser(), Path.home().name, study.name}

    def bind(self, relative):
        path = self.study / relative
        if (Path(relative).is_absolute() or ".." in Path(relative).parts
                or path.resolve() != path or not path.is_file()):
            raise ContractMismatch("missing input or redirected study path")
        digest = sha256_file(path)
        if relative in self.hashes:
            _exact(digest, self.hashes[relative], "input changed")
        self.hashes[relative] = digest
        return path

    def read(self, relative):
        path = self.bind(relative)
        value = _read_json(path)
        for text in _strings(value):
            self.private_names.update(re.findall(r"(?:/home/|/Users/|[A-Za-z]:\\Users\\)([^/\\\s]+)", text))
        return value

    def generation(self, relative, generation, payloads):
        directory = f"{relative}/gen_{generation:08d}"
        meta = self.read(f"{directory}/COMMIT.json")
        _exact(meta["generation"], generation, "generation")
        if set(meta["files"]) != set(payloads):
            raise ContractMismatch("generation payload inventory differs")
        for name in payloads:
            self.bind(f"{directory}/{name}")
            _exact(self.hashes[f"{directory}/{name}"], _digest(meta["files"][name]), "payload hash")
        return meta

    def rehash(self):
        for relative in list(self.hashes):
            self.bind(relative)


def _safe(value, private_names):
    for text in _strings(value):
        if (any(re.search(r"(?<![a-z0-9])" + re.escape(name.casefold()) + r"(?![a-z0-9])",
                          text.casefold()) for name in private_names if name)
                or re.search(r"(?<![\w./])(?:/|~/|[A-Za-z]:[\\/])", text)
                or "\\" in text or re.search(r"(?<![\w.])\.\.(?:/|$)", text)
                or "://" in text or "@" in text or any(ord(c) < 32 for c in text)):
            raise ContractMismatch("private name or path in public evidence")


def _score_identity(row, ids, protocol, digest, checkpoint, commit, corrected, secondary):
    expected = {
        "schema_version": 2, "protocol_sha256": digest, "ids": ids, "n": len(ids),
        "arrays": protocol["arrays"]["TOY"], "checkpoint_sha256": checkpoint,
        "ensemble_commit_sha256": commit, "procedure": "diffusion",
        "correct" if secondary else "corrected": corrected, "months": [0] * len(ids),
        "control_fit_sha256": None,
    }
    if secondary:
        expected["interval_protocol"] = INTERVAL_PROTOCOL
    _exact({k: row.get(k) for k in expected}, expected, "score identity")
    return set(expected)


def _scoring_record(primary, secondary, ids, protocol, digest, checkpoint, commit, corrected):
    pkeys = _score_identity(primary, ids, protocol, digest, checkpoint, commit, corrected, False)
    skeys = _score_identity(secondary, ids, protocol, digest, checkpoint, commit, corrected, True)
    metrics = set(SCALAR_METRICS + ARRAY_METRICS)
    if set(primary) != pkeys | {"scores", "mean", "month_balanced"}:
        raise ContractMismatch("primary score fields differ")
    if set(secondary) != skeys | metrics | {"per_window"}:
        raise ContractMismatch("secondary score fields differ")
    scores = _numeric(primary["scores"])
    if scores.shape != (len(ids),):
        raise ContractMismatch("incomplete primary scores")
    _close(primary["mean"], scores.mean(), "primary mean")
    _close(primary["month_balanced"], scores.mean(), "primary month mean")
    records = secondary["per_window"]
    if not isinstance(records, list) or len(records) != len(ids):
        raise ContractMismatch("incomplete secondary windows")
    for record in records:
        if not isinstance(record, dict) or set(record) != metrics:
            raise ContractMismatch("per-window secondary fields differ")
        for key in metrics:
            shape = (12, 5) if key in ARRAY_METRICS else ()
            if _numeric(record[key]).shape != shape:
                raise ContractMismatch("secondary metric shape differs")
        for scalar, matrix in (("crps5_mean", "crps_by_position_channel"),
                               ("cov90_mean", "cov90_by_position_channel"),
                               ("width90_mean", "width90_by_position_channel")):
            _close(record[scalar], np.asarray(record[matrix]).mean(), "coordinate mean")
    for key in metrics:
        summary = secondary[key]
        if not isinstance(summary, dict) or set(summary) != {"mean", "month_balanced"}:
            raise ContractMismatch("secondary reduction fields differ")
        mean = np.asarray([record[key] for record in records], dtype=np.float64).mean(axis=0)
        for value in summary.values():
            _close(value, mean, f"secondary {key}")
    _close([r["es5"] for r in records], scores, "primary versus es5")
    for reduction in ("mean", "month_balanced"):
        _close(secondary["es5"][reduction], primary[reduction], "primary versus es5 mean")
    return {
        "corrected": corrected, "scores": primary["scores"], "mean_energy": primary["mean"],
        "month_balanced_energy": primary["month_balanced"],
        "secondary": {key: secondary[key] for key in sorted(metrics)}, "per_window": records,
    }


def _reference_calibration(array_path, original, oracle_hash):
    """Replay only oracle.main's direct-reference RNG, once, in original array order."""
    z, ids, nrm, identity = load_arrays(array_path, "TOY")
    _exact(ids, [f"TOY:{i}" for i in range(288)], "reference IDs")
    with np.load(array_path, allow_pickle=False) as blob:
        _exact(blob["boundary_law"].item(), "stationary", "array law")
        expected = oracle_module.oracle_metadata(array_path, "pooled", "stationary", blob)
        ps, alphas = blob["p"], blob["alpha"]
    _exact({k: original.get(k) for k in expected}, expected, "oracle metadata")
    _exact({k: original.get(k) for k in ("n", "n_draws", "reference_seed")},
           {"n": 288, "n_draws": 39, "reference_seed": ORACLE_REF_SEED}, "oracle realization")
    if (ps.shape != (288,) or alphas.shape != (288,)
            or any(((alphas == a) & (ps == p)).sum() != 32 for a in ALPHAS for p in PS)):
        raise ContractMismatch("reference regime population differs")
    method = original["methods"]["exact_reference"]
    _exact(method["denoiser_evaluations_per_draw"], 0, "direct reference")
    _exact(method["total_denoiser_evaluations"], 0, "direct reference")
    if _numeric(method["scores"]).shape != (288,):
        raise ContractMismatch("incomplete oracle scores")
    rng_ref = pcg64(ORACLE_REF_SEED)
    records, contexts, recomputed_scores = [], [], []
    for i, zw in enumerate(z):
        alpha, a, b = oracle_module.infer_context(zw, nrm)
        _exact(float(alphas[i]), alpha, "observed alpha")
        contexts.append({"window": i, "alpha": alpha, "generating_p": float(ps[i]), "a": a, "b": b})
        # The held-out generating p is NEVER an input to this posterior.
        post = oracle_module.posterior_hidden(alpha, None, a, b, nrm,
                                              conditioning="pooled", boundary_law="stationary")
        paths = sample_mixture_paths(post, rng_ref, 39)
        states = np.array([state_summary(alpha, s) for s in (0, 1)])
        ref = nrm.transform(states[paths])[..., :5].reshape(39, -1)
        target = nrm.transform(zw[18:30])[:, :5].reshape(-1)
        recomputed_scores.append(energy_score(ref, target))
        _close(method["scores"][i], recomputed_scores[-1], "frozen oracle score")
        record = {"es5": method["scores"][i]}
        for level, lo, hi in ((90, 2, 38), (50, 10, 30)):
            for name, values in ((f"cov{level}", coverage_inclusive(ref, target, lo, hi).astype(float)),
                                 (f"width{level}", interval_width(ref, lo, hi))):
                record[f"{name}_mean"] = float(values.mean())
                record[f"{name}_by_position_channel"] = values.reshape(12, 5).tolist()
        crps = crps_univariate(ref, target)
        record.update(crps5_mean=float(crps.mean()), crps_by_position_channel=crps.reshape(12, 5).tolist())
        records.append(record)
    _exact(original["contexts"], contexts, "oracle window order")
    values = np.asarray([r["es5"] for r in records])
    _close(method["mean_energy"], np.mean(recomputed_scores), "recomputed oracle mean")
    _close(method["mean_energy"], values.mean(), "frozen oracle mean")
    by_p = method["mean_energy_by_generating_p"]
    _exact(sorted(by_p), sorted(str(p) for p in PS), "oracle strata")
    for p in PS:
        _close(by_p[str(p)], values[ps == p].mean(), "oracle stratum mean")
    return {
        "schema_version": 1, "source_oracle_sha256": oracle_hash, "arrays": identity,
        "oracle_metadata": expected, "reference_seed": ORACLE_REF_SEED,
        "n": 288, "n_draws": 39, "ids": ids, "months": [0] * 288, "contexts": contexts,
        "corrected": False, "interval_protocol": INTERVAL_PROTOCOL,
        "scores": method["scores"], "mean_energy": method["mean_energy"],
        "month_balanced_energy": method["mean_energy"],
        "secondary": summarize_diagnostics(records, [0] * 288), "per_window": records,
        "scope": (
            "Diagnostics of the already frozen exact-reference realization, reconstructed with "
            "one PCG64 stream in array order. No new reference is selected on results. This is "
            "an atomic conditional law: inclusive coverage can exceed 90 or 50 percent without "
            "establishing miscalibration. Finite self-scores are not a Bayes floor. These "
            "diagnostics establish neither sampler convergence nor a mechanism for real OMNI scores."
        ),
        "path_removal": PATH_REMOVAL,
    }


def _ensemble(inputs, relative, ids, protocol, digest, fit, checkpoint, **kwargs):
    meta = inputs.generation(relative, 0, ("ensemble.npz",))
    root = inputs.study / relative
    toy.check_ensemble(root, ids, protocol, digest, fit, checkpoint, **kwargs)
    expected = {
        "schema_version": 2, "protocol_sha256": digest, "arrays": protocol["arrays"]["TOY"],
        "checkpoint_sha256": checkpoint, "generation_updates": 5000, "arm": fit["arm"],
        "seed": 7, "n_generated": 6, "sampler": protocol["samplers"]["TOY"],
        "state_convention": toy.STATE_CONVENTION, "noise_convention": toy.NOISE_CONVENTION,
        "inference_source_sha256": protocol["inference_source_hashes"]["prigm/run_eval.py"],
    }
    if "offset" in kwargs:
        expected.update(offset=kwargs["offset"], count=len(ids))
    _exact({k: meta.get(k) for k in expected}, expected, "ensemble identity")
    _exact(meta.get("procedure", "diffusion"), "diffusion", "ensemble procedure")
    return read_ensemble(root)[0]


def _write(path, payload):
    """Exclusive writes preserve incomplete bytes on failure, never replace files."""
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    fsync_dir(path.parent)


def export_study(study_root, protocol_sha256, oracle, oracle_sha256, output_dir):
    """Read-only validation; fresh output with existing parent; return schema-1 manifest."""
    study = Path(study_root).absolute()
    output = Path(output_dir).absolute()
    oracle_path = Path(oracle).absolute()
    if study.resolve() != study or not study.is_dir():
        raise ContractMismatch("study root must exist without symlink redirection")
    if output.exists() or output.is_symlink():
        raise FileExistsError("output directory must be fresh")
    if not output.parent.is_dir():
        raise FileNotFoundError("output parent must exist")
    output = output.parent.resolve() / output.name
    if oracle_path.resolve() != oracle_path or not oracle_path.is_file():
        raise ContractMismatch("oracle must exist without symlink redirection")
    _digest(protocol_sha256)
    _exact(sha256_file(oracle_path), _digest(oracle_sha256), "oracle hash")
    source = Path(__file__).resolve()
    source_hash = sha256_file(source)
    _exact(source_hash, IMPORTED_EXPORTER_SHA256, "exporter changed after import")
    inputs = _Inputs(study)
    protocol = inputs.read("PROTOCOL.json")
    _exact(inputs.hashes["PROTOCOL.json"], protocol_sha256, "protocol hash")
    # execution_protocol is read-only: it reconstructs deterministic training-data
    # identities, NOT training updates. train/evaluate must never be invoked here.
    toy.guard(study, protocol, protocol_sha256)
    toy.execution_protocol(study, protocol_sha256)
    _exact(protocol["training_identity"], toy.training_identity()[0], "training identity")
    _exact(protocol["training_settings"], toy.scientific_settings(
        protocol["training_settings"]["source_snapshot_sha256"]), "training settings")
    _exact(protocol["training_environment_required"], {"device_type": "cuda", "torch_threads": 2}, "required environment")
    _exact(protocol["recipe"], toy.fixed_recipe(), "recipe")
    _exact(protocol["schema_version"], 2, "protocol schema")
    _exact(protocol["boundary_law"], "stationary", "protocol law")
    _exact(protocol["window_law"], "stationary_boundary_markov_bridge_with_repeated_endpoint_flanks; not a full 48-bin Markov trajectory", "window law")
    _exact(protocol["state_convention"], toy.STATE_CONVENTION, "clean flanks")
    _exact(protocol["noise_convention"], toy.NOISE_CONVENTION, "canonical noise")
    _exact(protocol["sampling_chunk_windows"], 8, "chunk size")
    _exact(protocol["samplers"], {"TOY": {"n_eval": 50, "n_draws": 39, "eval_seed": 7}}, "sampler")
    _exact(protocol["array_path"], "artifacts/toy.npz", "array path")
    _exact(inputs.read("PROTOCOL_RECEIPT.json"), {"protocol_sha256": protocol_sha256,
           "published_before": ["train", "evaluate"]}, "protocol receipt")
    arrays = inputs.bind("artifacts/toy.npz")
    _, ids, _, identity = load_arrays(arrays, "TOY", protocol["arrays"]["TOY"]["sha256"])
    _exact(identity, protocol["arrays"]["TOY"], "array identity")
    _exact(ids, [f"TOY:{i}" for i in range(288)], "ordered IDs")
    trained, evaluated = inputs.read("TRAINED.json"), inputs.read("EVALUATED.json")
    training_rows, evaluation_rows, fits = {}, {}, {}
    for fit in protocol["recipe"]["fits"]:
        name = fit["name"]
        run = f"runs/{name}"
        _exact(inputs.read(f"artifacts/training/{name}-request.json"),
               toy.fit_request(study, fit, protocol_sha256), "pre-fit request")
        completion = inputs.read(f"artifacts/training/{name}-completed.json")
        inputs.bind(f"{run}/source_snapshot.zip")
        meta = inputs.generation(run, 5000, ("state.pt", "meta.json"))
        contract = inputs.read(f"{run}/gen_00005000/meta.json")
        expected = {k: v for k, v in protocol["recipe"].items() if k not in ("fits", "milestones")}
        expected.update({k: v for k, v in fit.items() if k != "name"})
        expected.update(contract_version=2, source_hashes=protocol["training_source_hashes"],
                        data_hashes=protocol["training_data_hashes"],
                        normalization_hash=protocol["training_identity"]["normalization_sha256"],
                        settings={**protocol["training_settings"], "environment": contract["settings"]["environment"]})
        _exact(contract, expected, "checkpoint config")
        _exact(meta["contract"], contract, "checkpoint contract")
        _exact({k: meta[k] for k in ("updates", "epochs", "cursor")},
               {"updates": 5000, "epochs": 156, "cursor": 1024}, "checkpoint counters")
        required_env = protocol["training_environment_required"]
        _exact({k: contract["settings"]["environment"].get(k) for k in required_env}, required_env, "training environment")
        row = toy.completed_fit(study, fit, protocol, protocol_sha256)
        payload = load_torch_payload((study / f"{run}/gen_00005000/state.pt").read_bytes(), map_location="cpu")
        _exact(payload["contract"], contract, "checkpoint payload contract")
        _exact(payload["data_hashes"], protocol["training_identity"]["normalized_data_hashes"], "checkpoint data identity")
        del payload
        _exact(completion, row, "training completion")
        training_rows[name] = row
        parent = f"artifacts/toy/{name}-g5000"
        _exact(inputs.read(f"{parent}/REQUEST.json"), row, "evaluation request")
        completed = inputs.read(f"{parent}/COMPLETED.json")
        chunks = [study / f"{parent}/chunk-{offset:04d}" for offset in range(0, 288, 8)]
        merged = _ensemble(inputs, f"{parent}/ensemble", ids, protocol, protocol_sha256,
                           fit, row["checkpoint_sha256"], chunks=chunks)
        for offset, chunk in zip(range(0, 288, 8), chunks):
            draws = _ensemble(inputs, chunk.relative_to(study).as_posix(), ids[offset:offset + 8],
                              protocol, protocol_sha256, fit, row["checkpoint_sha256"], offset=offset)
            if draws.tobytes() != merged[offset:offset + 8].tobytes():
                raise ContractMismatch("merged ensemble is not the exact 36-chunk union")
        # sampler.ddim_generate conditions on clean observed flanks, but stores
        # only hidden endpoints in a zero-filled (6,48) array.
        for flank in (slice(0, 18), slice(30, 48)):
            if np.count_nonzero(merged[..., flank]):
                raise ContractMismatch("stored ensemble has nonzero non-hidden coordinates")
        commit = inputs.hashes[f"{parent}/ensemble/gen_00000000/COMMIT.json"]
        outputs, variants = {}, {}
        for corrected in (False, True):
            kind = "corrected" if corrected else "raw"
            score_rows = []
            for secondary in (False, True):
                key = f"{'secondaries' if secondary else 'score'}-{kind}"
                relative = f"{parent}/{key}.json"
                score_rows.append(inputs.read(relative))
                outputs[key] = toy.check_score(study / relative, ids, protocol, protocol_sha256,
                                               row["checkpoint_sha256"], commit, corrected, secondary)
                _exact(outputs[key], {"path": relative, "sha256": inputs.hashes[relative]}, "score receipt")
            variants[kind] = _scoring_record(*score_rows, ids, protocol, protocol_sha256,
                                             row["checkpoint_sha256"], commit, corrected)
            variants[kind]["source_sha256"] = {key: outputs[f"{key}-{kind}"]["sha256"]
                                                for key in ("score", "secondaries")}
        evaluation_rows[name] = {**row, "ensemble_commit_sha256": commit, "outputs": outputs}
        _exact(completed, evaluation_rows[name], "evaluation completion")
        public_config = {**contract, "settings": {k: v for k, v in contract["settings"].items() if k != "environment"}}
        fits[name] = {"config": public_config, "generation_updates": 5000,
                      "checkpoint_sha256": row["checkpoint_sha256"],
                      "checkpoint_commit_sha256": row["checkpoint_commit_sha256"],
                      "ensemble_commit_sha256": commit,
                      "completed_sha256": inputs.hashes[f"{parent}/COMPLETED.json"], **variants}
    _exact(trained, {"protocol_sha256": protocol_sha256, "completed_runs": training_rows}, "TRAINED")
    _exact(evaluated, {"protocol_sha256": protocol_sha256, "trained_manifest_sha256": inputs.hashes["TRAINED.json"],
                       "completed_runs": evaluation_rows}, "EVALUATED")
    original = {"protocol": protocol_sha256, "array": inputs.hashes["artifacts/toy.npz"],
                "trained": inputs.hashes["TRAINED.json"], "evaluated": inputs.hashes["EVALUATED.json"],
                "oracle": oracle_sha256}
    sources = {"training": protocol["training_source_hashes"], "inference": protocol["inference_source_hashes"],
               "coordinator_sha256": protocol["coordinator_sha256"]}
    learned = {
        "schema_version": 1, "scope": SCOPE, "path_removal": PATH_REMOVAL,
        "original_sha256": original, "sources": sources, "arrays": identity,
        "protocol": {k: protocol[k] for k in ("study", "exposure", "boundary_law", "window_law",
                     "recipe", "samplers", "sampling_chunk_windows", "state_convention", "noise_convention", "selection")},
        "training_identity": protocol["training_identity"], "training_settings": protocol["training_settings"],
        "n": 288, "n_draws": 39, "ids": ids, "months": [0] * 288,
        "interval_protocol": INTERVAL_PROTOCOL, "fit_order": [fit["name"] for fit in protocol["recipe"]["fits"]],
        "fits": fits,
    }
    reference = _reference_calibration(arrays, _read_json(oracle_path), oracle_sha256)
    reference["source_protocol_sha256"] = protocol_sha256
    _exact(reference["arrays"], identity, "shared reference array")
    output_payloads = {"learned.json": learned, "reference-calibration.json": reference}
    manifest = {
        "schema_version": 1, "scope": SCOPE, "path_removal": PATH_REMOVAL,
        "exporter_source_sha256": source_hash, "sources": sources,
        "original_sha256": original, "input_sha256": inputs.hashes,
        "output_sha256": {name: sha256_bytes(json_bytes(value)) for name, value in output_payloads.items()},
    }
    for value in (*output_payloads.values(), manifest):
        _safe(value, inputs.private_names)
    output.mkdir()  # No exist_ok, no cleanup or reuse, including on failed publication.
    fsync_dir(output.parent)
    for name, value in output_payloads.items():
        _write(output / name, json_bytes(value))
    temporary = output / ".MANIFEST.json.partial"
    _write(temporary, json_bytes(manifest))
    inputs.rehash()
    _exact(sha256_file(oracle_path), oracle_sha256, "oracle changed during export")
    _exact(sha256_file(source), source_hash, "exporter changed during export")
    toy.guard(study, protocol, protocol_sha256)  # Rehash all current source modules too.
    for name, digest in manifest["output_sha256"].items():
        _exact(sha256_file(output / name), digest, "output changed during export")
    _exact(sha256_file(temporary), sha256_bytes(json_bytes(manifest)), "manifest changed during export")
    os.link(temporary, output / "MANIFEST.json")  # Atomic and exclusive; preserve partial on failure.
    temporary.unlink()
    fsync_dir(output)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("study-root", "protocol-sha256", "oracle", "oracle-sha256", "output-dir"):
        parser.add_argument(f"--{flag}", required=True)
    args = parser.parse_args(argv)
    export_study(args.study_root, args.protocol_sha256, args.oracle, args.oracle_sha256, args.output_dir)
    print("Published complete reduced toy evidence and same-stream reference calibration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
