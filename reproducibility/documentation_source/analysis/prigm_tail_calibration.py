"""Locate marginal interval misses in the accepted, frozen TEST ensembles.

No model fitting, resampling of predictions, or checkpoint selection occurs.
The original neural ensembles are required in addition to the public evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from prigm.artifacts import json_bytes, load_arrays, read_ensemble, require_array_match
from prigm.config import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, FINAL_SEEDS
from prigm.recovery import ContractMismatch, sha256_file
from prigm.run_bootstrap import bootstrap_weights
from prigm.run_score import decode_ensemble

MANIFESTS = {
    "real": "17ac806ec1d3af99764fd4ce23bce0d0502cd49fd89719504f0e2926389967c0",
    "deadline": "e0599acfbdf3541cd7d3004e364ef43ae11c395b776c0fc519ddc7e839ea1756",
}
CHANNELS = ["mu_x", "mu_y", "mu_z", "m", "v_s"]
OUTCOMES = ["below", "inside", "above"]


def interval_fractions(samples, truth):
    """Return window/channel/outcome fractions over twelve hidden positions.

    Inputs are physical first-five-coordinate arrays with shapes N,39,12,5
    and N,12,5. Inclusive endpoints keep ties inside the interval.
    """
    x, y = np.asarray(samples), np.asarray(truth)
    if (x.ndim != 4 or x.shape[1:] != (39, 12, 5) or not len(x)
            or y.shape != (len(x), 12, 5)):
        raise ValueError("expected N,39,12,5 draws and N,12,5 targets")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("interval inputs must be finite")
    ordered = np.sort(x, axis=1)
    below = y < ordered[:, 1]
    above = y > ordered[:, 37]
    inside = ~(below | above)
    return np.stack([below.mean(axis=1), inside.mean(axis=1), above.mean(axis=1)], axis=-1)


def summarize(values, months, weights):
    """Average fixed models equally after averaging within each month."""
    values, months = np.asarray(values), np.asarray(months)
    if (values.ndim != 4 or values.shape[1:] != (len(months), 5, 3)
            or not values.shape[0] or not len(months) or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 1)
            or not np.allclose(values.sum(axis=-1), 1, rtol=0, atol=1e-15)):
        raise ValueError("invalid model/window/channel/outcome fractions")
    monthly = np.stack([values[:, months == m].mean(axis=1) for m in np.unique(months)], axis=1)
    model_means = monthly.mean(axis=1)
    mean = model_means.mean(axis=0)
    result = {"model_means": model_means.tolist(), "mean": mean.tolist(),
              "below_minus_above": (mean[:, 0] - mean[:, 2]).tolist()}
    for scheme, w in weights.items():
        w = np.asarray(w)
        if (w.ndim != 2 or w.shape[1] != monthly.shape[1] or not len(w)
                or not np.isfinite(w).all() or np.any(w < 0)
                or not np.allclose(w.sum(axis=1), 1, rtol=0, atol=1e-12)):
            raise ValueError("invalid month resampling weights")
        draws = np.einsum("rm,mco->rco", w, monthly.mean(axis=0))
        result[scheme + "_95"] = np.quantile(draws, [.025, .975], axis=0).tolist()
        result[scheme + "_below_minus_above_95"] = np.quantile(
            draws[:, :, 0] - draws[:, :, 2], [.025, .975], axis=0).tolist()
    return result


def analyze(study, evidence, output):
    study, evidence, output = Path(study), Path(evidence), Path(output)
    if output.exists():
        raise FileExistsError(output)
    inputs = {}

    def pinned(path, expected):
        digest = sha256_file(path)
        if digest != expected:
            raise ContractMismatch(f"input hash mismatch: {path}")
        inputs[str(path.resolve())] = digest
        return path

    manifests = {
        name: json.loads(pinned(evidence / name / "MANIFEST.json", digest).read_text())
        for name, digest in MANIFESTS.items()
    }
    real, deadline = manifests["real"], manifests["deadline"]
    pins = real["input_sha256"]
    comparison = json.loads(pinned(evidence / "real/comparison.json", real["output_sha256"]["comparison.json"]).read_text())
    diagnostics = json.loads(pinned(evidence / "real/diagnostics.json", real["output_sha256"]["diagnostics.json"]).read_text())
    array_path = pinned(study / "artifacts/inventory/test.npz", pins["artifacts/inventory/test.npz"])
    truth, ids, nrm, identity = load_arrays(array_path, "TEST", pins["artifacts/inventory/test.npz"])
    if ids != comparison["ids"] or ids != diagnostics["ids"] or len(ids) != 480:
        raise ContractMismatch("TEST population differs from the accepted comparison")
    months = np.asarray(comparison["months"], dtype=np.int64)
    if months.tolist() != [int(w[:4]) * 100 + int(w[5:7]) for w in ids]:
        raise ContractMismatch("month labels differ from ordered window IDs")
    weights = bootstrap_weights(months)
    records, arrays = {}, {"ids": np.asarray(ids), "months": months}

    def read_draws(root, digest_map, prefix):
        for filename in ("ensemble.npz", "COMMIT.json"):
            relative = prefix + "/gen_00000000/" + filename
            pinned(root / "gen_00000000" / filename, digest_map[relative])
        ensemble, sampled_ids, meta = read_ensemble(root)
        require_array_match(meta, identity)
        if sampled_ids != ids:
            raise ContractMismatch("ensemble window order differs from TEST")
        physical = np.stack([decode_ensemble(row, nrm)[..., :5] for row in ensemble])
        return interval_fractions(physical, truth[:, 18:30, :5])

    for arm in ("D", "I-S", "O-S", "G5", "gaussian"):
        key = arm + "_raw"
        seeds = list(FINAL_SEEDS) if arm != "gaussian" else [None]
        fractions = []
        for seed in seeds:
            if seed is None:
                prefix = "artifacts/controls_test/gaussian"
            else:
                generation = comparison["methods"][key]["generations"][str(seed)]
                prefix = f"artifacts/test/{arm}_s{seed}-g{generation}/ensemble"
            fractions.append(read_draws(study / prefix, pins, prefix))
        values = np.stack(fractions)
        arrays[arm] = values
        summary = summarize(values, months, weights)
        expected = np.asarray(diagnostics["methods"][key]["metrics"]["cov90_by_position_channel"]["month_balanced"]).mean(axis=0)
        if not np.allclose(np.asarray(summary["mean"])[:, 1], expected, rtol=0, atol=1e-14):
            raise ContractMismatch(f"coverage does not reproduce the accepted diagnostics: {arm}")
        records[arm] = {"fixed_model_seeds": seeds, **summary}

    prefix = "gaussian/test/lambda_0.1_rho_0"
    values = read_draws(evidence / "deadline" / prefix, deadline["files"], prefix)[None]
    summary = summarize(values, months, weights)
    score_path = pinned(evidence / "deadline/gaussian/test/SCORES.json", deadline["files"]["gaussian/test/SCORES.json"])
    gaussian = json.loads(score_path.read_text())["candidates"]["lambda_0.1_rho_0"]["raw"]
    expected = np.array([r["coverage90"] for r in gaussian["per_window"]])
    if not np.allclose(values[0, :, :, 1].mean(axis=1), expected, rtol=0, atol=1e-14):
        raise ContractMismatch("full-flank Gaussian coverage differs from frozen records")
    records["full_flank_gaussian"] = {"fixed_model_seeds": [None], **summary}
    arrays["full_flank_gaussian"] = values
    for name, digest in inputs.items():
        if sha256_file(Path(name)) != digest:
            raise ContractMismatch("an input changed during analysis")
    output.mkdir()
    np.savez_compressed(output / "per_window.npz", **arrays)
    report = {
        "schema_version": 1, "scope": "Retrospective marginal interval diagnostics of frozen raw-selected TEST ensembles.",
        "channels": CHANNELS, "outcomes": OUTCOMES, "n_windows": len(ids), "n_draws": 39,
        "interval_order_statistics": [2, 38], "continuous_exchangeability_reference": [.05, .90, .05],
        "weighting": "Average twelve positions within each window, windows within month, then months and fixed models equally.",
        "uncertainty": "Pointwise percentile intervals conditional on fitted models, selected checkpoints and frozen ensembles; no multiplicity correction or training/selection uncertainty.",
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "schemes": ["year", "twoyear"]},
        "methods": records, "input_sha256": inputs,
        "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in [Path(__file__), *sorted((ROOT / "prigm").glob("*.py"))]},
        "per_window_sha256": sha256_file(output / "per_window.npz"),
    }
    (output / "summary.json").write_bytes(json_bytes(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, default=ROOT / "paper/workshops/prigm/evidence")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.study_root, args.evidence_root, args.output)


if __name__ == "__main__":
    main()
