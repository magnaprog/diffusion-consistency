"""Score the 2025 forecasts at the epochs with the lowest validation scores and at epoch 31."""
import argparse
import json
from pathlib import Path

import numpy as np

from prigm.observation import derive_vv, support_gaps
from prigm.precision import representation_bands


METRICS = ("energy", "variance_crps", "variance_coverage", "variance_width",
           "mean_absolute_error_per_member", "half_pair_distance", "outside_bounds")
REGISTERED = METRICS[:4]
SEEDS = (71, 73, 79)


def score_window(draws, truth, mean, scale, b_ref_sq):
    """Score standardized member forecasts (member, quantity, bin) against physical observations (bin, quantity)."""
    standardized_members = draws.transpose(0, 2, 1).astype(np.float64)
    standardized_observation = (truth[:, :5] - mean[:5]) / scale[:5]
    member_vectors = standardized_members.reshape(len(draws), -1)
    first_member, second_member = np.triu_indices(len(draws), 1)
    # The energy score uses the joint vector of five quantities over twelve bins.
    energy = (np.linalg.norm(member_vectors - standardized_observation.ravel(), axis=1).mean()
              - np.linalg.norm(member_vectors[first_member] - member_vectors[second_member], axis=1).mean() / 2)

    physical_moments = derive_vv(standardized_members * scale[:5] + mean[:5])
    predicted_variance = physical_moments[:, :, 5]
    observed_variance = truth[:, 5]
    distance_to_observation = np.abs(predicted_variance - observed_variance).mean()
    half_member_distance = np.abs(predicted_variance[first_member] - predicted_variance[second_member]).mean() / 2
    # Ranks 2 and 38 of the 39 members define the central 90% interval in each bin.
    lower, upper = np.sort(predicted_variance, axis=0)[[1, -2]]
    coverage = ((lower <= observed_variance) & (observed_variance <= upper)).mean()

    bands = representation_bands(physical_moments, (physical_moments - mean) / scale,
                                 scale, np.sqrt(b_ref_sq), b_ref_sq)
    gaps = support_gaps(physical_moments)
    outside_bounds = np.logical_or.reduce([gaps[bound] < -tolerance for bound, tolerance in bands.items()])
    return [energy, distance_to_observation - half_member_distance, coverage, (upper - lower).mean(),
            distance_to_observation, half_member_distance, outside_bounds.mean()]


def replay(folder):
    registered = json.loads((folder / "results.json").read_text())
    by_fit = {}
    with np.load(folder / "forecasts_2025.npz", allow_pickle=False) as archive:
        window_ids, observations = archive["ids"], archive["truth"]
        window_months = np.array([identifier[:7] for identifier in window_ids])
        months, counts = np.unique(window_months, return_counts=True)
        if (len(set(window_ids)) != 144 or observations.shape != (144, 12, 6)
                or months.tolist() != [f"2025-{m:02}" for m in range(1, 13)]
                or np.any(counts != 12)):
            raise ValueError("unexpected 2025 window inventory")
        mean, scale, b_ref_sq = (archive[key] for key in ("mean", "scale", "b_ref_sq"))
        for name, fit in registered["fits"].items():
            variants = [name] if fit["selected_epoch"] == 31 else [name, name + "_final"]
            for key in variants:
                draws = archive[key]
                if draws.shape != (len(window_ids), 39, 5, 12):
                    raise ValueError(f"unexpected forecasts for {key}")
                window_scores = np.array([score_window(members, observed, mean, scale, b_ref_sq)
                                          for members, observed in zip(draws, observations)])
                monthly = np.array([window_scores[window_months == month].mean(axis=0) for month in months])
                if key == name:
                    expected = [[fit["by_month"][month][metric] for metric in REGISTERED] for month in months]
                    np.testing.assert_allclose(monthly[:, :4], expected, rtol=3e-12, atol=3e-12)
                by_fit[key] = dict(zip(METRICS, monthly.mean(axis=0).tolist()))

    comparisons = {}
    for policy in ("selected", "epoch_31"):
        differences = {}
        for seed in SEEDS:
            baseline, auxiliary = f"G5_{seed}", f"G5aux_{seed}"
            if policy == "epoch_31":
                if registered["fits"][baseline]["selected_epoch"] != 31:
                    baseline += "_final"
                if registered["fits"][auxiliary]["selected_epoch"] != 31:
                    auxiliary += "_final"
            differences[str(seed)] = {metric: by_fit[auxiliary][metric] - by_fit[baseline][metric] for metric in METRICS}
        average = {metric: float(np.mean([row[metric] for row in differences.values()])) for metric in METRICS}
        comparisons[policy] = {"by_seed": differences, "mean": average}
    return {"fits": by_fit, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print all per-fit diagnostics and differences.")
    module_dir = Path(__file__).resolve().parent
    source_root = module_dir.parents[1]
    # In the public src layout an editable install still points into this checkout.
    if source_root.name == "src" and not (module_dir / "results.json").is_file():
        evidence_dir = source_root.parent / "evidence" / "g5_auxiliary"
    else:
        evidence_dir = module_dir
    parser.add_argument("--evidence-dir", type=Path, default=evidence_dir,
                        help="Directory containing results.json and forecasts_2025.npz.")
    args = parser.parse_args()
    outcome = replay(args.evidence_dir)
    if args.json:
        print(json.dumps(outcome, indent=2, allow_nan=False))
    else:
        print("Auxiliary minus G5; negative score differences favor auxiliary supervision.")
        print("Policy       Seed       Joint energy    Vector-variance CRPS (nT^2)")
        for policy, result in outcome["comparisons"].items():
            for label, row in [*result["by_seed"].items(), ("Mean", result["mean"])]:
                print(f"{policy:<12} {label:<6} {row['energy']:>15.3f} {row['variance_crps']:>27.3f}")


if __name__ == "__main__":
    main()
