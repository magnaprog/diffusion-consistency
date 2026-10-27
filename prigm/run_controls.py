"""Save linear interpolation, persistence, and Gaussian interpolation predictions.

Linear interpolation joins observed bins 17 and 30 across the hidden hour.
Persistence repeats bin 17. Gaussian interpolation adds a jointly sampled
72-coordinate residual, fitted to training deviations from interpolation;
it can retain covariance across the 12 hidden positions and six quantities.

The protocol records source and data hashes, normalization, and method
settings. Predictions are saved with that identity in a new output directory.
The deterministic methods repeat one prediction 39 times for the common
ensemble format; these copies do not represent predictive uncertainty.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np

from prigm.artifacts import json_bytes, load_arrays, publish_ensemble, require_source_match
from prigm.barrier import load_protocol, require_test_started
from prigm.config import GAUSSIAN_JITTER, HIDDEN_END, HIDDEN_START, N_DRAWS, PRIMARY_CHANNELS
from prigm.controls import gaussian_fit, gaussian_sample, linear_interpolate
from prigm.keys import pcg64
from prigm.normalize import AffineNormalizer
from prigm.recovery import ContractMismatch, atomic_write_bytes, sha256_bytes, sha256_file
from prigm.scoring import energy_score


def primary_std_hid(z_hid: np.ndarray, nrm: AffineNormalizer) -> np.ndarray:
    u = (z_hid[..., :PRIMARY_CHANNELS] - nrm.mean[:PRIMARY_CHANNELS]) / nrm.scale[:PRIMARY_CHANNELS]
    return np.reshape(u, u.shape[:-2] + (12 * PRIMARY_CHANNELS,))


def score_pointmass(z_hid_pred: np.ndarray, z_hid_true: np.ndarray, nrm: AffineNormalizer) -> float:
    xs = primary_std_hid(z_hid_pred[None, ...], nrm)
    y = primary_std_hid(z_hid_true[None, ...], nrm)[0]
    return energy_score(np.repeat(xs, N_DRAWS, axis=0), y)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train-arrays", required=True)
    p.add_argument("--eval-arrays", required=True)
    p.add_argument("--output", required=True, help="fresh control artifact directory")
    p.add_argument("--study-root", required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--split", required=True, choices=("VAL", "TEST"))
    args = p.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    protocol = load_protocol(Path(args.study_root), args.protocol_sha256)
    require_source_match(protocol)
    require_test_started(Path(args.study_root), args.split, args.protocol_sha256)
    z_tr, _, nrm, train_identity = load_arrays(
        Path(args.train_arrays), "TRAIN", protocol["arrays"]["TRAIN"]["sha256"]
    )
    z_ev, ids, _, eval_identity = load_arrays(
        Path(args.eval_arrays), args.split, protocol["arrays"][args.split]["sha256"]
    )
    if train_identity["normalization_sha256"] != eval_identity["normalization_sha256"]:
        raise ContractMismatch("controls require the same training normalization")
    controls = protocol["controls"]
    if set(controls) != {"interpolation", "persistence", "gaussian"}:
        raise ContractMismatch("unexpected control procedure set")
    gaussian = controls["gaussian"]
    if gaussian["rho"] not in (0.0, 0.5, 1.0) or gaussian["jitter"] != GAUSSIAN_JITTER:
        raise ContractMismatch("Gaussian parameters differ from declared control family")
    interpolation_train = linear_interpolate(z_tr)[:, HIDDEN_START:HIDDEN_END]
    residuals = ((z_tr[:, HIDDEN_START:HIDDEN_END] - interpolation_train) / nrm.scale).reshape(len(z_tr), 72)
    mu, covariance = gaussian_fit(residuals, gaussian["rho"], gaussian["jitter"])
    del z_tr, interpolation_train, residuals
    fit_buffer = io.BytesIO()
    np.savez(fit_buffer, residual_mean=mu, covariance=covariance)
    fit_bytes = fit_buffer.getvalue()
    out.mkdir(parents=True)
    atomic_write_bytes(out / "gaussian_fit.npz", fit_bytes)
    rng = pcg64(gaussian["seed"])
    fit_hash = sha256_bytes(fit_bytes)
    for name in ("interpolation", "persistence", "gaussian"):
        ensembles = []
        for z in z_ev:
            if name == "persistence":
                pred = z.copy()
                pred[HIDDEN_START:HIDDEN_END] = z[HIDDEN_START - 1]
            else:
                pred = linear_interpolate(z)
            draws = np.repeat(nrm.transform(pred)[None], N_DRAWS, axis=0)
            if name == "gaussian":
                reference = nrm.transform(pred[HIDDEN_START:HIDDEN_END]).reshape(72)
                draws[:, HIDDEN_START:HIDDEN_END] = gaussian_sample(
                    reference, mu, covariance, N_DRAWS, rng
                ).reshape(N_DRAWS, 12, 6)
            ensembles.append(draws.transpose(0, 2, 1))
        meta = {
            "protocol_sha256": args.protocol_sha256, "procedure": name,
            "control": controls[name], "n_generated": 6,
            "state_convention": "physical_control_standardized_float64",
            "arrays": eval_identity, "training_arrays": train_identity,
            "control_fit_sha256": fit_hash if name == "gaussian" else None,
            "noise_convention": "pcg64_ordered_windows_multivariate_normal" if name == "gaussian" else "point_mass",
            "source_sha256": sha256_file(Path(__file__)),
        }
        publish_ensemble(out / name, np.stack(ensembles), ids, meta)
    atomic_write_bytes(out / "COMPLETED.json", json_bytes({
        "protocol_sha256": args.protocol_sha256, "split": args.split,
        "arrays": eval_identity, "training_arrays": train_identity,
        "gaussian_fit_sha256": fit_hash,
        "controls": {name: sha256_file(out / name / "gen_00000000/COMMIT.json") for name in controls},
    }))
    print(json.dumps({"output": str(out), "n_windows": len(ids), "controls": list(controls)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
