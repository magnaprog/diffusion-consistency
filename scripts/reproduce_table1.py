"""Print the eight rows of Table 1 as tab-separated values from the saved TEST results."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "prigm").is_dir():
    sys.path.insert(0, str(ROOT))
from analysis.prigm_followup_summary import reductions
from analysis.prigm_full_flank_gaussian import candidate_name
from prigm.normalize import AffineNormalizer
from prigm.observation import residual_obs
from prigm.run_score import decode_ensemble

EVIDENCE = ROOT / "evidence"
if not EVIDENCE.is_dir():
    EVIDENCE = ROOT / "analysis/evidence"
if not EVIDENCE.is_dir():
    EVIDENCE = ROOT / "paper/workshops/prigm/evidence"
METHODS = {"D": "D", "I-S": "I-S", "O-S": "O-S", "G5": "G5",
           "Linear interpolation": "interpolation", "Persistence": "persistence",
           "Gaussian interpolation": "gaussian"}


def gaussian_regression(months, window_ids):
    gaussian_root = EVIDENCE / "deadline/gaussian"
    scores = json.loads((gaussian_root / "test/SCORES.json").read_bytes())
    np.testing.assert_array_equal(scores["months"], months)
    np.testing.assert_array_equal(scores["ids"], window_ids)
    # Raw and clipped forecasts use the parameters that minimized their respective validation scores.
    selected = {}
    for treatment, choice in scores["selected"].items():
        candidate = candidate_name(choice["lambda"], choice["rho"])
        selected[treatment] = scores["candidates"][candidate]
    with np.load(EVIDENCE / "observations/test.npz", allow_pickle=False) as observations:
        np.testing.assert_array_equal(observations["ids"], window_ids)
        normalizer = AffineNormalizer(observations["mean"], observations["scale"],
                                      float(observations["b_ref_sq"]), int(observations["n_bins"]))
    path = gaussian_root / selected["raw"]["ensemble"] / "gen_00000000/ensemble.npz"
    with np.load(path, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["ids"], window_ids)
        residuals = [np.abs(residual_obs(decode_ensemble(window, normalizer))).mean()
                     for window in arrays["ensemble"]]

    def metric(treatment, name):
        rows = selected[treatment][treatment]["per_window"]
        return reductions([row[name] for row in rows], months)["month_balanced"]

    return [metric("raw", "primary_es"), metric("corrected", "primary_es"),
            100 * metric("raw", "coverage90"), 100 * metric("corrected", "coverage90"),
            reductions(residuals, months)["month_balanced"],
            100 * metric("raw", "support_violation_beyond_rounding_fraction")]


def table_rows():
    scores = json.loads((EVIDENCE / "real/scores.json").read_bytes())
    diagnostics = json.loads((EVIDENCE / "real/diagnostics.json").read_bytes())
    months, window_ids = scores["months"], scores["ids"]
    np.testing.assert_array_equal(diagnostics["ids"], window_ids)
    np.testing.assert_array_equal(diagnostics["months"], months)

    def energy(key):
        return np.mean([reductions(fit, months)["month_balanced"] for fit in scores["data"][key]])

    def diagnostic(key, metric):
        by_fit = diagnostics["methods"][key]["seedwise"]
        return np.mean([fit[metric]["month_balanced"] for fit in by_fit.values()])

    for label, method in METHODS.items():
        raw, clipped = method + "_raw", method + "_corrected"
        yield label, [energy(raw), energy(clipped),
                      100 * diagnostic(raw, "cov90_mean"), 100 * diagnostic(clipped, "cov90_mean"),
                      diagnostic(raw, "r_obs_mean"), 100 * diagnostic(raw, "frac_support_violation_beyond_rounding")]
    yield "Gaussian regression", gaussian_regression(months, window_ids)


if __name__ == "__main__":
    print("Procedure\tRaw ES\tClipped ES\tRaw coverage (%)\tClipped coverage (%)\tMean |R|\tOutside range (%)")
    for label, values in table_rows():
        raw_score, clipped_score, raw_coverage, clipped_coverage, residual, outside = values
        print(f"{label}\t{raw_score:.3f}\t{clipped_score:.3f}\t{raw_coverage:.1f}\t"
              f"{clipped_coverage:.1f}\t{residual:.3f}\t{outside:.1f}")
