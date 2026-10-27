"""Export two fixed, post-exposure raw imputation illustrations, without sampling."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prigm.artifacts import load_arrays, read_ensemble, require_array_match, json_bytes
from prigm.recovery import ContractMismatch, sha256_file
from prigm.run_score import decode_ensemble, hidden_primary_std
from prigm.scoring import energy_score

OBS_SHA256 = "2cbe75399a190d1522379701cced5c74718aaced251bdf4179f246b72b46c1c7"
PROTOCOL = "25d0cdefdd5dbb160c32c7b5f4a6053bab99e4a9f64832d9eb6502232bf24e6b"
GAUSSIAN_PROTOCOL = "8456fdae42a086fa9cb11dbc4bc4da9d54d5a3fe233aff6ed6d4bb30d25d0090"
FIT_SELECT = "5ce6ea010bcf77b02663b2ef3ce573d35e5ccbf09b19a0b143dd3e859f0b156c"
METHODS = ("D", "O-S", "G5", "full_flank_gaussian")
INDICES = [120, 360]
STEPS = {"D": 9560, "O-S": 29636, "G5": 29636}
GAUSSIAN = "gaussian/test/lambda_0.1_rho_0"
REPO = Path(__file__).resolve().parents[1]


def _require(condition, message):
    if not condition:
        raise ContractMismatch(message)


def export_excerpt(paper_dir, ensemble_dir, output_dir):
    """Validate complete frozen inputs and publish a fresh two-file excerpt directory.

    ``paper_dir`` contains evidence/; ``ensemble_dir`` contains the three
    ARM_s7-gSTEP roots. No complete COMMIT metadata or local paths are exported.
    Returns the manifest dictionary; all energies use standardized primary coords.
    """
    paper, ensembles, out = map(Path, (paper_dir, ensemble_dir, output_dir))
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    if not out.parent.is_dir():
        raise FileNotFoundError(out.parent)
    consumed = {}
    inputs, sources = {}, {}

    def track(path, public, expected=None, source=False):
        digest = sha256_file(path)
        _require(expected is None or digest == expected, f"hash mismatch: {public}")
        if path in consumed:
            _require(consumed[path] == digest, f"source changed: {public}")
        consumed[path] = digest
        (sources if source else inputs)[public] = digest
        return digest

    def read(relative, expected=None):
        path = paper / relative
        track(path, relative, expected)
        return json.loads(path.read_text())

    # Include the exporter and the complete frozen core, including transitive imports.
    core_paths = sorted((REPO / "prigm").glob("*.py"))
    for path in [Path(__file__).resolve(), *core_paths]:
        track(path, path.relative_to(REPO).as_posix(), source=True)
    real = read("evidence/real/MANIFEST.json")
    deadline = read("evidence/deadline/MANIFEST.json")
    comparison = read("evidence/real/comparison.json", real["output_sha256"]["comparison.json"])
    scores = read("evidence/real/scores.json", real["output_sha256"]["scores.json"])
    gaussian_scores = read("evidence/deadline/gaussian/test/SCORES.json",
                           deadline["files"]["gaussian/test/SCORES.json"])
    obs = paper / "evidence/observations/test.npz"
    track(obs, "evidence/observations/test.npz", OBS_SHA256)
    truth, ids, nrm, identity = load_arrays(obs, "TEST", OBS_SHA256)
    _require(len(ids) == 480, "expected all 480 TEST windows")
    for record in (comparison, scores, gaussian_scores):
        _require(record["ids"] == ids, "full TEST IDs/order mismatch")
    for record in (comparison, gaussian_scores):
        require_array_match(record, identity)
    for record in (comparison, scores):
        _require(record["protocol_sha256"] == PROTOCOL, "neural score protocol mismatch")
    _require(gaussian_scores["extension_protocol_sha256"] == GAUSSIAN_PROTOCOL
             and gaussian_scores["source_protocol_sha256"] == PROTOCOL
             and gaussian_scores["fit_select_sha256"] == FIT_SELECT
             and gaussian_scores["n"] == 480 and gaussian_scores["n_draws"] == 39,
             "Gaussian score lineage mismatch")
    samples, methods, energies = [], {}, {}
    for method in METHODS:
        neural = method in STEPS
        if neural:
            step = STEPS[method]
            name = f"{method}_s7-g{step}"
            root = ensembles / name
            public = f"artifacts/test/{name}/ensemble"
            pins = real["input_sha256"]
        else:
            step = None
            root = paper / "evidence/deadline" / GAUSSIAN
            public = GAUSSIAN
            pins = deadline["files"]
        hashes = {}
        for filename in ("COMMIT.json", "ensemble.npz"):
            key = f"{public}/gen_00000000/{filename}"
            label = key if neural else f"evidence/deadline/{key}"
            hashes[filename] = track(root / "gen_00000000" / filename, label, pins[key])
        ens, ensemble_ids, meta = read_ensemble(root)
        require_array_match(meta, identity)
        _require(ensemble_ids == ids, "full ensemble IDs/order mismatch")
        channels = 5 if method == "G5" else 6
        _require(ens.shape == (480, 39, channels, 48) and meta["n_generated"] == channels,
                 "full ensemble shape mismatch")
        if neural:
            selected = comparison["selected"][f"{method}_s7"]["raw"]
            _require(meta.get("procedure", "diffusion") == "diffusion"
                     and meta["arm"] == method and meta["seed"] == 7
                     and meta["generation_updates"] == step == selected["generation"]
                     and meta["checkpoint_sha256"] == selected["checkpoint_sha256"]
                     and meta["protocol_sha256"] == PROTOCOL
                     and meta["sampler"] == {"eval_seed": 7, "n_draws": 39, "n_eval": 50},
                     f"{method} raw-selected checkpoint/sampler mismatch")
            key = f"{method}_raw"
            published = np.asarray(scores["data"][key], dtype=np.float64)
            _require(published.shape == (3, 480)
                     and scores["row_ids"][key] == ["7", "13", "42"], "neural score rows mismatch")
            _require(np.isfinite(published).all(), "nonfinite published scores")
            published = published[0]
            lineage = {"checkpoint_sha256": meta["checkpoint_sha256"], "seed": 7,
                       "step": step, "n_eval": 50, "selection": "raw"}
        else:
            _require(meta["procedure"] == "gaussian"
                     and meta["protocol_sha256"] == GAUSSIAN_PROTOCOL
                     and meta["source_protocol_sha256"] == PROTOCOL
                     and meta["fit_select_sha256"] == FIT_SELECT
                     and meta["control_fit_sha256"] == gaussian_scores["fit_payload_sha256"]
                     == deadline["files"]["gaussian/FIT.npz"]
                     and meta["control"] == {"family": "full_flank_linear_gaussian",
                         "jitter": 1e-6, "joint_gaussian_channels": 5, "lambda": 0.1, "rho": 0.0}
                     and meta["noise"]["seed"] == 13 and meta["noise"]["n_draws"] == 39,
                     "Gaussian fit/control lineage mismatch")
            # The public deadline schema calls this primary_es, not es5.
            rows = gaussian_scores["candidates"]["lambda_0.1_rho_0"]["raw"]["per_window"]
            published = np.asarray([row["primary_es"] for row in rows], dtype=np.float64)
            _require(published.shape == (480,) and np.isfinite(published).all(),
                     "Gaussian per-window scores mismatch")
            lineage = {"control_fit_sha256": meta["control_fit_sha256"],
                       "fit_select_sha256": FIT_SELECT, "seed": 13, "step": None,
                       "control": meta["control"], "selection": "raw"}
        decoded = np.stack([decode_ensemble(ens[i], nrm) for i in INDICES])
        _require(decoded.shape == (2, 39, 12, 6) and np.isfinite(decoded).all(),
                 "nonfinite or invalid decoded samples")
        samples.append(decoded[..., :5])
        energies[method] = []
        for j, i in enumerate(INDICES):
            value = energy_score(hidden_primary_std(decoded[j], nrm),
                                 hidden_primary_std(truth[i, 18:30][None], nrm)[0])
            _require(np.isfinite(value) and abs(value - published[i]) <= 1e-12,
                     f"{method} per-case primary energy mismatch")
            energies[method].append({"case_id": ids[i], "zero_based_index": i,
                                    "recomputed": value, "published": float(published[i])})
        methods[method] = {**lineage, "protocol_sha256": meta["protocol_sha256"],
                           "n_generated": channels, "n_draws": 39, "arrays": identity,
                           "ensemble_npz_sha256": hashes["ensemble.npz"],
                           "ensemble_commit_sha256": hashes["COMMIT.json"],
                           "source": public if neural else f"evidence/deadline/{public}"}
        del ens
    case_ids = [ids[i] for i in INDICES]
    buffer = io.BytesIO()
    np.savez(buffer, truth=np.asarray(truth[INDICES], dtype=np.float64),
             samples=np.asarray(samples, dtype=np.float64), ids=np.asarray(case_ids, dtype=str),
             methods=np.asarray(METHODS, dtype=str))
    from hashlib import sha256
    manifest = {
        "schema_version": 1, "analysis": "frozen_raw_imputation_examples",
        "selection": {"rule": "midpoints_of_two_equal_ordered_TEST_partitions",
                      "zero_based_indices": INDICES, "original_window_count": 480,
                      "neural_seed": 7, "displayed_draws": [0, 1, 2], "after_test_exposure": True},
        "case_ids": case_ids, "method_order": list(METHODS), "methods": methods,
        "channels": ["mu_x", "mu_y", "mu_z", "m", "v_s"],
        "units": ["nT", "nT", "nT", "nT", "nT^2"],
        "arrays": identity, "input_sha256": inputs, "source_sha256": sources,
        "per_case_primary_energy": energies, "energy_absolute_tolerance": 1e-12,
        "energy_coordinates": "training-standardized hidden primary channels; all 39 raw draws",
        "truth_source": "physical observations, not stored ensemble flanks",
        "files": {"sample-excerpt.npz": sha256(buffer.getvalue()).hexdigest()},
    }
    manifest_bytes = json_bytes(manifest)
    _require(core_paths == sorted((REPO / "prigm").glob("*.py")), "core source inventory changed")
    for path, digest in consumed.items():
        _require(sha256_file(path) == digest, "consumed input or source changed before publication")
    out.mkdir()  # Exclusive creation; never replace an existing directory.
    try:
        (out / "sample-excerpt.npz").write_bytes(buffer.getvalue())
        (out / "MANIFEST.json").write_bytes(manifest_bytes)
    except BaseException:
        for filename in ("sample-excerpt.npz", "MANIFEST.json"):
            (out / filename).unlink(missing_ok=True)
        out.rmdir()
        raise
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("paper-dir", "ensemble-dir", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    export_excerpt(args.paper_dir, args.ensemble_dir, args.output_dir)
    print(json.dumps({"manifest_sha256": sha256_file(args.output_dir / "MANIFEST.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
