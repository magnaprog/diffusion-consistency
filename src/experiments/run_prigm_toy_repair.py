"""Run the fresh nine-fit toy follow-up with explicitly recorded training data.

This experiment uses the stationary endpoint law for a Markov bridge across
the hidden interval. Observed bins repeat the endpoints; the 48-bin window
is not a full Markov trajectory. The fixed recipe trains all nine procedure
and coefficient combinations at seed 7 and evaluates only their update-5000
checkpoints. It does not choose a coefficient or checkpoint from validation
results.

Create an empty study directory owned by this run, publish protocol, and pass
its SHA256 back to train and evaluate. The records bind generated training
data, normalization, source files, model states and evaluation inputs.
These are fresh controlled fits; they do not establish the training history
of the earlier nine fits. No oracle execution, year resampling or scientific
conclusions are performed by this coordinator.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from prigm.arrays import physical_to_model
from prigm.artifacts import (
    NOISE_CONVENTION, STATE_CONVENTION, json_bytes, load_arrays,
    merge_ensembles, read_ensemble, require_source_match,
)
from prigm.barrier import load_protocol
from prigm.checkpoint import (
    array_hash, frozen_source_snapshot, load_torch_payload, normalization_hash,
    source_hashes, verify_source_snapshot,
)
from prigm.config import (
    BATCH_SIZE, C_R_GRID, N_DRAWS, PILOT_SEED, SAMPLE_CHUNK_WINDOWS,
    TOY_NOISE_SEED, TOY_PATH_SEED, TOY_UPDATES, TOY_WIDTHS,
)
from prigm.model import ConditionalUNet
from prigm.normalize import fit_normalizer
from prigm.recovery import (
    ContractMismatch, PublicationError, RunLock, atomic_write_bytes, fsync_dir, load_committed,
    read_json, sha256_bytes, sha256_file,
)
from prigm.run_fit import arm_cfg, training_settings
from prigm.toy import ALPHAS, PS, pooled_toy_observations
from prigm.toy_eval import build_eval_windows
from prigm.toy_windows import make_toy_training


def source_identity():
    return source_hashes(sorted((ROOT / "prigm").rglob("*.py")), relative_to=ROOT)


# Catch edits after import as well as edits between CLI invocations.
IMPORTED_SOURCES = source_identity()
IMPORTED_COORDINATOR = sha256_file(Path(__file__))


def require_sources(protocol=None):
    if source_identity() != IMPORTED_SOURCES or sha256_file(Path(__file__)) != IMPORTED_COORDINATOR:
        raise ContractMismatch("source changed after coordinator import")
    if protocol is not None:
        if (protocol["training_source_hashes"] != IMPORTED_SOURCES
                or protocol["inference_source_hashes"] != IMPORTED_SOURCES
                or protocol["coordinator_sha256"] != IMPORTED_COORDINATOR):
            raise ContractMismatch("source differs from frozen toy protocol")
        require_source_match(protocol)


def fresh_json(path, value):
    """Publish a complete receipt while the caller holds the coordinator lock."""
    path = Path(path)
    if path.exists():
        raise PublicationError(f"receipt exists: {path}")
    atomic_write_bytes(path, json_bytes(value))


def record_or_check(path, value):
    if Path(path).exists():
        if read_json(path) != value:
            raise ContractMismatch(f"existing receipt differs: {path}")
    else:
        fresh_json(path, value)


def owned_path(study, relative):
    path = study / relative
    if path.resolve() != path or not path.is_relative_to(study):
        raise PublicationError(f"study paths must not redirect through symlinks: {path}")
    return path


def fixed_recipe():
    if (PILOT_SEED, BATCH_SIZE, TOY_UPDATES, tuple(TOY_WIDTHS), tuple(C_R_GRID),
            N_DRAWS, SAMPLE_CHUNK_WINDOWS, TOY_PATH_SEED, TOY_NOISE_SEED) != (
            7, 128, 5000, (32, 64, 64), (0.0, 0.0005, 0.005), 39, 8, 7001, 7002):
        raise ContractMismatch("configured toy recipe differs from the requested follow-up")
    fits = []
    for arm, residual, estimator in (("D", None, "S"), ("I-M", "inst", "M"),
                                     ("I-S", "inst", "S"), ("O-M", "obs", "M"), ("O-S", "obs", "S")):
        for c_r in (C_R_GRID[:1] if arm == "D" else C_R_GRID[1:]):
            if arm_cfg(arm, c_r) != (6, c_r, residual, estimator):
                raise ContractMismatch("run_fit arm semantics differ from the frozen toy recipe")
            fits.append({"name": f"{arm}_c{str(c_r).replace('.', 'p')}_s7", "arm": arm,
                         "c_r": c_r, "residual": residual, "estimator": estimator})
    return {"seed": 7, "n_generated": 6, "n_aux": 0, "widths": list(TOY_WIDTHS),
            "n_examples": 4096, "batch_size": 128, "max_updates": 5000,
            "milestones": [5000], "fits": fits}


def training_identity():
    """Recreate the training inputs used by run_fit --toy and record their identity.

    Generate the 4096 windows with seed 106. Fit normalization from the six
    pooled family states, then record hashes for the physical windows and
    standardized model inputs and conditioning arrays. Evaluation windows
    are not used to estimate this training-data identity or the normalization.
    """
    z, alphas, ps = make_toy_training(4096, seed=106)
    if z.shape != (4096, 48, 6) or not np.isfinite(z).all():
        raise ContractMismatch("invalid generated training source")
    pooled = pooled_toy_observations()
    nrm = fit_normalizer(pooled)
    u0, observed = physical_to_model(z, nrm, n_generated=6)
    record = {
        "generator": "prigm.toy_windows.make_toy_training",
        "seed": 106, "n_examples": 4096, "shape": list(z.shape),
        "boundary_law": "stationary", "hash_encoding": "C-contiguous little-endian float64 z bytes only",
        "data_hashes": {"toy_arrays_sha256": sha256_bytes(np.ascontiguousarray(z, dtype="<f8").tobytes())},
        "normalized_data_hashes": {"u0_sha256": array_hash(u0), "observed_sha256": array_hash(observed)},
        "regime_counts": [{"alpha": a, "p": p, "n": int(((alphas == a) & (ps == p)).sum())}
                          for a in ALPHAS for p in PS],
        "normalizer_source": "fit_normalizer(pooled_toy_observations()); six family states, not evaluation windows",
        "pooled_observations_sha256": sha256_bytes(np.ascontiguousarray(pooled, dtype="<f8").tobytes()),
        "normalization": {"mean": nrm.mean.tolist(), "scale": nrm.scale.tolist(),
                          "b_ref_sq": nrm.b_ref_sq, "n_bins": nrm.n_bins},
        "normalization_sha256": normalization_hash(nrm.mean, nrm.scale, nrm.b_ref_sq, nrm.n_bins),
    }
    return record, nrm


def scientific_settings(snapshot_hash):
    # Optimizer/schedule defaults come from the same entry point as training.
    # Hardware and software versions remain descriptive in each saved contract.
    settings = training_settings(torch.device("cpu"), [5000])
    settings.pop("environment")
    settings["precision"].update({
        "default_dtype": "torch.float32", "deterministic_algorithms": True,
        "deterministic_warn_only": False, "cudnn_benchmark": False,
        "cudnn_tf32": False, "matmul_tf32": False, "cublas_workspace_config": ":4096:8",
    })
    settings["source_snapshot_sha256"] = snapshot_hash
    settings["data"] = {"kind": "toy", "ids_covered_by_arrays_hash": False, "seed": 106, "n_examples": 4096}
    return settings


def publish_protocol(study):
    # main already holds the coordinator lock. No adoption of an old study.
    if any(p.name != "coordinator" for p in study.iterdir()):
        raise PublicationError("protocol requires a fresh empty owned study root")
    require_sources()
    recipe = fixed_recipe()
    training, nrm = training_identity()  # Identify training before building TOY.
    hashes, snapshot = frozen_source_snapshot(ROOT)
    artifacts = owned_path(study, "artifacts")
    artifacts.mkdir()
    z, alpha, p = build_eval_windows(32)
    if z.shape != (288, 48, 6) or not np.isfinite(z).all():
        raise ContractMismatch("expected all 288 finite stationary toy windows")
    buffer = io.BytesIO()
    np.savez(buffer, z=z, alpha=alpha, p=p, ids=np.array([f"TOY:{i}" for i in range(288)]),
             split=np.array("TOY"), boundary_law=np.array("stationary"),
             mean=nrm.mean, scale=nrm.scale, b_ref_sq=nrm.b_ref_sq, n_bins=nrm.n_bins)
    arrays = artifacts / "toy.npz"
    with arrays.open("xb") as handle:
        handle.write(buffer.getvalue())
        handle.flush()
        os.fsync(handle.fileno())
    identity = load_arrays(arrays, "TOY")[3]
    protocol = {
        "schema_version": 2, "exposure": "historical_test_exposed",
        "study": "controlled_stationary_toy_repair", "study_root": str(study),
        "reason": "historical nine-fit training source identity is unprovable; fresh controlled follow-up, not historical lineage certification",
        "window_law": "stationary_boundary_markov_bridge_with_repeated_endpoint_flanks; not a full 48-bin Markov trajectory",
        "boundary_law": "stationary", "evaluation_generator": {"name": "prigm.toy_eval.build_eval_windows", "n_per_regime": 32, "path_seed": 7001},
        "training_source_hashes": hashes, "inference_source_hashes": hashes,
        "coordinator_sha256": IMPORTED_COORDINATOR,
        "training_data_hashes": training["data_hashes"], "training_identity": training,
        "recipe": recipe, "training_settings": scientific_settings(sha256_bytes(snapshot)),
        "training_environment_required": {"device_type": "cuda", "torch_threads": 2},
        "training_environment_policy": "other environment fields are recorded per checkpoint, not forced equal across hardware",
        "arrays": {"TOY": identity}, "array_path": "artifacts/toy.npz",
        "samplers": {"TOY": {"n_eval": 50, "n_draws": 39, "eval_seed": 7}},
        "sampling_chunk_windows": 8, "state_convention": STATE_CONVENTION,
        "noise_convention": NOISE_CONVENTION,
        "noise_followup_change": {
            "previous_proposal_eval_seed": 7002, "eval_seed": 7,
            "reason": "prespecified follow-up change to match existing prigm.oracle.main; not selected from scores",
            "key": "noise_key('eval', 'TOY', f'TOY:{i}', draw, 7)",
            "layout": "PCG64 float64 standard_normal((48,6)); neural sampler casts to float32",
        },
        "selection": "fixed final checkpoint at update 5000 only; no validation or coefficient selection; all nine fits retained",
        "outputs": "complete raw/corrected energy scores and per-window secondaries; no target filtering, bootstrap or conclusions",
    }
    require_sources(protocol)
    fresh_json(study / "PROTOCOL.json", protocol)
    digest = sha256_file(study / "PROTOCOL.json")
    fresh_json(study / "PROTOCOL_RECEIPT.json", {"protocol_sha256": digest, "published_before": ["train", "evaluate"]})
    return digest


def execution_protocol(study, digest):
    protocol = load_protocol(study, digest)
    require_sources(protocol)
    if (protocol["study"] != "controlled_stationary_toy_repair" or protocol["study_root"] != str(study)
            or protocol["recipe"] != fixed_recipe()
            or protocol["samplers"] != {"TOY": {"n_eval": 50, "n_draws": 39, "eval_seed": 7}}):
        raise ContractMismatch("study or frozen toy recipe mismatch")
    if read_json(study / "PROTOCOL_RECEIPT.json") != {"protocol_sha256": digest, "published_before": ["train", "evaluate"]}:
        raise ContractMismatch("missing matching protocol publication receipt")
    training, _ = training_identity()
    _, snapshot = frozen_source_snapshot(ROOT)
    if (training != protocol["training_identity"] or training["data_hashes"] != protocol["training_data_hashes"]
            or scientific_settings(sha256_bytes(snapshot)) != protocol["training_settings"]):
        raise ContractMismatch("training source data or settings differ from protocol")
    _, ids, _, identity = load_arrays(owned_path(study, "artifacts/toy.npz"), "TOY", protocol["arrays"]["TOY"]["sha256"])
    if (identity != protocol["arrays"]["TOY"] or ids != [f"TOY:{i}" for i in range(288)]
            or identity["normalization_sha256"] != training["normalization_sha256"]):
        raise ContractMismatch("TOY population or training normalization mismatch")
    return protocol


def guard(study, protocol, digest):
    load_protocol(study, digest)
    require_sources(protocol)


def verify_checkpoint(study, fit, protocol, generation=5000):
    """Strict explicit-generation verification, including payload rather than just its label."""
    root = owned_path(study, f"runs/{fit['name']}")
    meta = load_committed(root, generation=generation)
    contract = meta["contract"]
    expected = {k: v for k, v in protocol["recipe"].items() if k not in ("fits", "milestones")}
    expected.update({k: v for k, v in fit.items() if k != "name"})
    expected.update(contract_version=2, source_hashes=protocol["training_source_hashes"],
                    data_hashes=protocol["training_data_hashes"],
                    normalization_hash=protocol["training_identity"]["normalization_sha256"])
    if {k: v for k, v in contract.items() if k != "settings"} != expected:
        raise ContractMismatch(f"scientific recipe mismatch: {fit['name']}")
    settings = contract["settings"]
    if ({k: v for k, v in settings.items() if k != "environment"} != protocol["training_settings"]
            or any(settings.get("environment", {}).get(k) != v
                   for k, v in protocol["training_environment_required"].items())):
        raise ContractMismatch(f"training settings mismatch: {fit['name']}")
    directory = root / f"gen_{generation:08d}"
    if set(meta["files"]) != {"state.pt", "meta.json"} or read_json(directory / "meta.json") != contract:
        raise ContractMismatch("checkpoint payload inventory or contract mismatch")
    payload = load_torch_payload((directory / "state.pt").read_bytes(), map_location="cpu")
    if payload["contract"] != contract or payload["data_hashes"] != protocol["training_identity"]["normalized_data_hashes"]:
        raise ContractMismatch("saved contract or normalized training data mismatch")
    epochs, remainder = divmod(generation, 32)
    counters = {"updates": generation, "epochs": epochs, "cursor": remainder * 128}
    if (not 0 <= generation <= 5000 or any(type(payload[k]) is not int or payload[k] != v
            or meta[k] != v for k, v in counters.items())):
        raise ContractMismatch("checkpoint completed-update counters mismatch")
    nrm = payload["normalization"]
    if (any(not torch.is_tensor(nrm[k]) or nrm[k].dtype != torch.float64 or nrm[k].shape != (6,)
            for k in ("mean", "scale")) or normalization_hash(
                nrm["mean"], nrm["scale"], nrm["b_ref_sq"], nrm["n_bins"]) != expected["normalization_hash"]):
        raise ContractMismatch("saved normalization mismatch")
    verify_source_snapshot((root / "source_snapshot.zip").read_bytes(), contract["source_hashes"],
                           settings["source_snapshot_sha256"])
    with torch.random.fork_rng(devices=[]):
        model_state = ConditionalUNet(n_generated=6, widths=tuple(TOY_WIDTHS)).state_dict()
    if (payload["model"].keys() != model_state.keys() or any(
            not torch.is_tensor(t) or t.shape != model_state[k].shape or t.dtype != model_state[k].dtype
            or not torch.isfinite(t).all().item() for k, t in payload["model"].items())):
        raise ContractMismatch("checkpoint model weights mismatch or nonfinite")
    return meta


def preserve_uncommitted(root):
    """Only called for receipt-owned paths; retain failed attempts, never erase them."""
    if not root.exists() or any(root.glob("gen_*/COMMIT.json")):
        return
    with RunLock(root):
        # A writer may have committed between the first check and lock acquisition.
        if any(root.glob("gen_*/COMMIT.json")):
            return
        index = 1
        destination = root.with_name(f"{root.name}.incomplete-{index}")
        while destination.exists():
            index += 1
            destination = root.with_name(f"{root.name}.incomplete-{index}")
        root.rename(destination)
        fresh_json(destination / "PRESERVED.json", {"original": root.name, "reason": "no committed generation; retained before deterministic retry"})
        fsync_dir(root.parent)


def launch(command, log, gpu):
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(ROOT),
                "PRIGM_REQUIRE_CUDA": "1", "PISCES_REQUIRE_CUDA": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"})
    with Path(log).open("ab") as handle:
        handle.write(json_bytes({"command": command, "gpu": gpu}))
        handle.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def on_devices(fits, gpus, worker):
    def device_worker(index, gpu):
        return [worker(fit, gpu) for fit in fits[index::len(gpus)]]
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(device_worker, i, gpu) for i, gpu in enumerate(gpus)]
        return {row["name"]: row for future in futures for row in future.result()}


def fit_request(study, fit, digest):
    return {"protocol_sha256": digest, "fit": fit, "run_root": f"runs/{fit['name']}",
            "generation_updates": 5000, "training_cli": "prigm.run_fit --toy"}


def completed_fit(study, fit, protocol, digest):
    request = owned_path(study, f"artifacts/training/{fit['name']}-request.json")
    if read_json(request) != fit_request(study, fit, digest):
        raise ContractMismatch("run has no matching pre-fit request")
    meta = verify_checkpoint(study, fit, protocol)
    return {"name": fit["name"], "protocol_sha256": digest, "generation_updates": 5000,
            "run_root": f"runs/{fit['name']}", "request_sha256": sha256_file(request),
            "checkpoint_sha256": meta["files"]["state.pt"],
            "checkpoint_commit_sha256": sha256_file(study / f"runs/{fit['name']}/gen_00005000/COMMIT.json")}


def train(study, protocol, digest, gpus):
    requests = owned_path(study, "artifacts/training")
    requests.mkdir(exist_ok=True)

    def worker(fit, gpu):
        guard(study, protocol, digest)
        name = fit["name"]
        root = owned_path(study, f"runs/{name}")
        request = owned_path(study, f"artifacts/training/{name}-request.json")
        if root.exists() and not request.exists():
            raise PublicationError("refusing pre-existing run without a pre-fit request")
        record_or_check(request, fit_request(study, fit, digest))
        completion = owned_path(study, f"artifacts/training/{name}-completed.json")
        if completion.exists() or (root / "gen_00005000").exists():
            row = completed_fit(study, fit, protocol, digest)
        else:
            preserve_uncommitted(root)
            command = [sys.executable, "-B", "-u", "-m", "prigm.run_fit", "--run-root", str(root),
                       "--arm", fit["arm"], "--seed", "7", "--c-r", str(fit["c_r"]),
                       "--toy", "--batch-size", "128", "--updates", "5000",
                       "--budget-updates", "5000", "--milestones", "5000"]
            if root.exists():
                recovery = load_committed(root)
                verify_checkpoint(study, fit, protocol, generation=recovery["generation"])
                command.append("--resume")
            launch(command, owned_path(study, f"logs/train-{name}.log"), gpu)
            guard(study, protocol, digest)
            row = completed_fit(study, fit, protocol, digest)
        record_or_check(completion, row)
        return row

    rows = on_devices(protocol["recipe"]["fits"], gpus, worker)
    guard(study, protocol, digest)
    record_or_check(study / "TRAINED.json", {"protocol_sha256": digest, "completed_runs": rows})


def check_ensemble(root, ids, protocol, digest, fit, checkpoint_hash, *, offset=None, chunks=None):
    _, actual_ids, meta = read_ensemble(root)
    expected = {"protocol_sha256": digest, "arrays": protocol["arrays"]["TOY"],
                "checkpoint_sha256": checkpoint_hash, "generation_updates": 5000,
                "arm": fit["arm"], "seed": 7, "n_generated": 6,
                "sampler": protocol["samplers"]["TOY"], "state_convention": STATE_CONVENTION,
                "noise_convention": NOISE_CONVENTION,
                "inference_source_sha256": protocol["inference_source_hashes"]["prigm/run_eval.py"]}
    if offset is not None:
        expected.update(offset=offset, count=len(ids))
    if chunks is not None:
        expected["chunks"] = {str(p): sha256_file(p / "gen_00000000/COMMIT.json") for p in chunks}
    if actual_ids != ids or meta.get("procedure", "diffusion") != "diffusion" or any(meta.get(k) != v for k, v in expected.items()):
        raise ContractMismatch("existing ensemble differs from pinned toy procedure or exact IDs")


def check_score(path, ids, protocol, digest, checkpoint_hash, commit_hash, corrected, secondary):
    row = read_json(path)
    expected = {"schema_version": 2, "protocol_sha256": digest, "ids": ids, "n": len(ids),
                "arrays": protocol["arrays"]["TOY"], "checkpoint_sha256": checkpoint_hash,
                "ensemble_commit_sha256": commit_hash, "procedure": "diffusion",
                "correct" if secondary else "corrected": corrected, "months": [0] * len(ids)}
    if any(row.get(k) != v for k, v in expected.items()):
        raise ContractMismatch("stale score output provenance")
    if secondary:
        records = row.get("per_window", [])
        if len(records) != len(ids) or not records or not records[0]:
            raise ContractMismatch("secondaries must retain every window")
        for key in records[0]:
            values = np.asarray([r[key] for r in records], dtype=np.float64)
            summary = row[key]
            if (not np.isfinite(values).all() or not np.allclose(summary["mean"], values.mean(axis=0))
                    or not np.allclose(summary["month_balanced"], values.mean(axis=0))):
                raise ContractMismatch("nonfinite or inconsistent per-window secondaries")
    else:
        values = np.asarray(row.get("scores", []), dtype=np.float64)
        if (values.shape != (len(ids),) or not np.isfinite(values).all()
                or not np.isclose(row["mean"], values.mean()) or not np.isclose(row["month_balanced"], values.mean())):
            raise ContractMismatch("nonfinite or incomplete energy scores")
    return {"path": str(path.relative_to(Path(protocol["study_root"]))), "sha256": sha256_file(path)}


def evaluate(study, protocol, digest, gpus):
    # Verify ALL nine final states and their immutable pins before any sampling.
    rows = {fit["name"]: completed_fit(study, fit, protocol, digest) for fit in protocol["recipe"]["fits"]}
    if read_json(study / "TRAINED.json") != {"protocol_sha256": digest, "completed_runs": rows}:
        raise ContractMismatch("TRAINED manifest does not pin the nine final checkpoints")
    for name, row in rows.items():
        if read_json(study / f"artifacts/training/{name}-completed.json") != row:
            raise ContractMismatch("final checkpoint completion receipt changed")
    arrays = owned_path(study, "artifacts/toy.npz")
    _, ids, _, _ = load_arrays(arrays, "TOY", protocol["arrays"]["TOY"]["sha256"])

    def worker(fit, gpu):
        guard(study, protocol, digest)
        name = fit["name"]
        checkpoint_hash = rows[name]["checkpoint_sha256"]
        parent = owned_path(study, f"artifacts/toy/{name}-g5000")
        request = parent / "REQUEST.json"
        if parent.exists() and not request.exists() and any(parent.iterdir()):
            raise PublicationError("refusing pre-existing evaluation outputs without an owned request")
        parent.mkdir(parents=True, exist_ok=True)
        record_or_check(request, rows[name])
        chunks = []
        for offset in range(0, len(ids), 8):
            guard(study, protocol, digest)
            chunk = owned_path(study, str((parent / f"chunk-{offset:04d}").relative_to(study)))
            preserve_uncommitted(chunk)
            if not chunk.exists():
                launch([sys.executable, "-B", "-u", "-m", "prigm.run_eval",
                        "--run-root", str(study / rows[name]["run_root"]), "--generation", "5000",
                        "--checkpoint-sha256", checkpoint_hash, "--out", str(chunk),
                        "--arrays", str(arrays), "--split", "TOY", "--n-eval", "50", "--eval-seed", "7",
                        "--offset", str(offset), "--limit", "8", "--study-root", str(study),
                        "--protocol-sha256", digest], owned_path(study, f"logs/sample-{name}.log"), gpu)
            check_ensemble(chunk, ids[offset:offset + 8], protocol, digest, fit, checkpoint_hash, offset=offset)
            chunks.append(chunk)
        merged = owned_path(study, str((parent / "ensemble").relative_to(study)))
        preserve_uncommitted(merged)
        if not merged.exists():
            merge_ensembles(chunks, merged, ids)
        check_ensemble(merged, ids, protocol, digest, fit, checkpoint_hash, chunks=chunks)
        commit_hash = sha256_file(merged / "gen_00000000/COMMIT.json")
        outputs = {}
        for secondary in (False, True):
            for corrected in (False, True):
                guard(study, protocol, digest)
                key = f"{'secondaries' if secondary else 'score'}-{'corrected' if corrected else 'raw'}"
                output = owned_path(study, str((parent / f"{key}.json").relative_to(study)))
                if not output.exists():
                    command = [sys.executable, "-B", "-m", "prigm.run_secondaries" if secondary else "prigm.run_score",
                               "--ensemble", str(merged), "--arrays", str(arrays), "--output", str(output),
                               "--split", "TOY", "--study-root", str(study), "--protocol-sha256", digest]
                    if corrected:
                        command.append("--correct")
                    launch(command, owned_path(study, f"logs/score-{name}.log"), gpu)
                outputs[key] = check_score(output, ids, protocol, digest, checkpoint_hash, commit_hash, corrected, secondary)
        result = {**rows[name], "ensemble_commit_sha256": commit_hash, "outputs": outputs}
        record_or_check(parent / "COMPLETED.json", result)
        return result

    completed = on_devices(protocol["recipe"]["fits"], gpus, worker)
    guard(study, protocol, digest)
    record_or_check(study / "EVALUATED.json", {"protocol_sha256": digest,
                    "trained_manifest_sha256": sha256_file(study / "TRAINED.json"), "completed_runs": completed})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("protocol", "train", "evaluate"))
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--protocol-sha256")
    args = parser.parse_args(argv)
    study = Path(args.study_root).absolute()
    if study.resolve() != study or not study.is_dir() or study.stat().st_uid != os.getuid():
        raise PublicationError("create a fresh owned study directory without symlink redirection first")
    gpus = [int(v) for v in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus) or any(v < 0 for v in gpus):
        raise ValueError("GPU IDs must be distinct nonnegative integers")
    if args.stage == "protocol":
        if args.protocol_sha256:
            parser.error("protocol publishes a new hash; do not supply --protocol-sha256")
        if any(study.iterdir()):
            raise PublicationError("protocol requires a fresh empty owned study root")
    else:
        if not args.protocol_sha256:
            parser.error("train/evaluate require --protocol-sha256")
        # Refuse old/unpublished roots before creating even a lock or log there.
        protocol = execution_protocol(study, args.protocol_sha256)
    with RunLock(owned_path(study, "coordinator")):
        if args.stage == "protocol":
            print(publish_protocol(study))
        else:
            guard(study, protocol, args.protocol_sha256)
            owned_path(study, "logs").mkdir(exist_ok=True)
            if args.stage == "train":
                train(study, protocol, args.protocol_sha256, gpus)
            else:
                evaluate(study, protocol, args.protocol_sha256, gpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
