"""Explore checkpoint reselection using existing VAL50/TEST50 scores on CPU.

No models, samples or historical evidence are changed. The same calendar-year
resamples evaluate fixed selections and selections made anew from validation.
These percentile summaries do not establish nominal confidence coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prigm.scoring import bootstrap_indices, month_balanced_mean, percentile_interval
from prigm.select import select_milestone

INPUT = "paper/workshops/prigm/evidence/followup/checkpoints/SUMMARY.json"
OUTPUT = "paper/workshops/prigm/evidence/followup/checkpoint_reselection.json"
UPDATES = (2868, 9560, 16252, 22944, 29636)
GROUPS = tuple(f"{arm}_s{seed}" for arm in ("D", "O-S") for seed in (7, 13, 42))
YEARS = np.arange(2005, 2025)
REPLICATES, SEED = 10000, 7003


def year_scores(values, months, years):
    """Equal month means within each year, retaining unequal window counts."""
    values, months = np.asarray(values, dtype=float), np.asarray(months)
    if (values.ndim != 2 or values.shape[1] != len(months)
            or not np.isfinite(values).all()):
        raise ValueError("finite model-by-window scores required")
    if set(months // 100) != set(years):
        raise ValueError("score years differ from declared years")
    result = []
    for year in years:
        mask = months // 100 == year
        if len(np.unique(months[mask])) != 2:
            raise ValueError("each year must contain two represented months")
        result.append([month_balanced_mean(row[mask], months[mask]) for row in values])
    return np.asarray(result).T


def checkpoint_indices(scores):
    """Minimum validation score; exact ties select the earlier update."""
    scores = np.asarray(scores)
    if scores.shape[-1] != len(UPDATES) or not np.isfinite(scores).all():
        raise ValueError("five finite checkpoint scores required")
    selected = [select_milestone([
        {"epoch": update, "score": float(score)} for update, score in zip(UPDATES, row)
    ])["epoch"] for row in scores.reshape(-1, len(UPDATES))]
    return np.array([UPDATES.index(update) for update in selected]).reshape(scores.shape[:-1])


def contrast_summary(scores):
    d, o = scores[:, :3].mean(axis=1), scores[:, 3:].mean(axis=1)
    if np.any(d <= 0):
        raise ValueError("relative contrast requires positive D score")
    difference = o - d
    return {"mean": float(difference.mean()),
            "percentile_2_5_97_5": list(percentile_interval(difference)),
            "relative_percentile_2_5_97_5": list(percentile_interval(difference / d))}


def analyze(summary):
    keys = [f"{group}-g{update}" for group in GROUPS for update in UPDATES]
    arrays = {}
    for split, count, expected_months in (("VAL", 319, (11, 12)), ("TEST", 480, (8, 9))):
        data = summary["scores"][split]
        months = np.asarray(data["months"])
        if (set(data["data"]) != set(keys) or len(months) != count
                or len(data["ids"]) != count or len(set(data["ids"])) != count
                or set(months % 100) != set(expected_months)):
            raise ValueError("incomplete or different saved score population")
        arrays[split] = year_scores([data["data"][key] for key in keys], months, YEARS).reshape(6, 5, 20)
    val_means = arrays["VAL"].mean(axis=-1)
    original = checkpoint_indices(val_means)
    for i, group in enumerate(GROUPS):
        record = summary["validation_budget_comparison"][group]
        if (not np.allclose(val_means[i], record["VAL50"], rtol=0, atol=1e-12)
                or UPDATES[original[i]] != summary["policies"][group]["P50"]
                or summary["policies"][group]["P25"] != summary["policies"][group]["P50"]):
            raise ValueError("VAL50 means or original choices differ from saved records")
    draws = bootstrap_indices(len(YEARS), REPLICATES, SEED)
    weights = np.array([np.bincount(row, minlength=len(YEARS)) / len(YEARS) for row in draws])
    val = np.einsum("ry,guy->rgu", weights, arrays["VAL"])
    test = np.einsum("ry,guy->rgu", weights, arrays["TEST"])
    choices = checkpoint_indices(val)
    fixed = test[:, np.arange(6), original]
    reselected = np.take_along_axis(test, choices[:, :, None], axis=2)[:, :, 0]
    observed = arrays["TEST"].mean(axis=-1)[np.arange(6), original]
    return {
        "schema_version": 1,
        "analysis": "checkpoint_reselection_existing_scores",
        "scope": "Exploratory sensitivity conditional on existing fits, saved checkpoints and ensemble draws; previously exposed TEST period.",
        "limits": ["No retraining or new model samples", "No new-fit uncertainty",
                   "No nominal coverage claim under temporal dependence",
                   "Accepted fixed-selection intervals and historical evidence unchanged"],
        "procedure": {"generator": "PCG64", "seed": SEED, "replicates": REPLICATES,
                      "years": YEARS.tolist(), "validation_evaluations": 50,
                      "test_evaluations": 50, "ensemble_members": 39,
                      "updates": list(UPDATES), "groups": list(GROUPS),
                      "weighting": "Equal represented-month means; repeated year occurrences retain multiplicity",
                      "pairing": "Identical year multiplicities for VAL and TEST, all groups, fixed and reselected comparisons",
                      "selection": "VAL only; exact ties to earlier update; all three fits retained"},
        "original_choices": {group: UPDATES[original[i]] for i, group in enumerate(GROUPS)},
        "observed_TEST": {"D": float(observed[:3].mean()), "O-S": float(observed[3:].mean()),
                          "O-S_minus_D": float(observed[3:].mean() - observed[:3].mean())},
        "fixed_selection": contrast_summary(fixed),
        "reselected": contrast_summary(reselected),
        "selection_counts": {group: {str(update): int(np.sum(choices[:, i] == j))
                                     for j, update in enumerate(UPDATES)} for i, group in enumerate(GROUPS)},
    }


def build_result(input_path):
    payload = Path(input_path).read_bytes()
    result = analyze(json.loads(payload))
    result["input"] = {"path": INPUT, "sha256": hashlib.sha256(payload).hexdigest()}
    result["source"] = {"path": "scripts/analyze_prigm_checkpoint_reselection.py",
                        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result["helper_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in ("prigm/scoring.py", "prigm/select.py", "prigm/config.py")}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / INPUT)
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("output must not overwrite input evidence")
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists; choose a new path")
    result = build_result(args.input)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
