"""Recompute the four scores from the saved forecasts using independent reductions."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

from experiments.g5_auxiliary.evaluate import METRICS, choices
from experiments.g5_auxiliary.train import EPOCHS, HIDDEN, PROTOCOL
from prigm.recovery import sha256_file


def reduce_samples(standardized, observed, mean, scale):
    k = len(standardized)
    x = standardized[:, :, HIDDEN].transpose(0, 2, 1).astype(np.float64)
    primary = x.reshape(k, -1)
    target = ((observed[:, :5] - mean[:5]) / scale[:5]).ravel()
    es = np.linalg.norm(primary - target, axis=1).mean() - pdist(primary).mean() / 2
    physical = x * scale[:5] + mean[:5]
    variance = physical[:, :, 3]**2 + physical[:, :, 4] - np.einsum("kij,kij->ki", physical[:, :, :3], physical[:, :, :3])
    ordered = np.sort(variance, axis=0)
    half_spread = (np.arange(1-k, k, 2) @ ordered) / (k*(k-1))
    obs = observed[:, 5]
    crps = (np.abs(variance - obs).mean(axis=0) - half_spread).mean()
    lower, upper = ordered[1], ordered[-2]
    return [es, crps, np.mean((lower <= obs) & (obs <= upper)), np.mean(upper - lower)]


def audited_record(folder, epoch, truth, ids, mean, scale):
    path = folder / f"epoch-{epoch:02}.npz"
    record = json.loads(path.with_suffix(".json").read_text())
    if sha256_file(path) != record["archive"] or record["windows"] != len(ids):
        raise ValueError("forecast archive identity mismatch")
    with np.load(path, allow_pickle=False) as f:
        if not np.array_equal(ids, f["ids"]) or f["draws"].shape != (len(ids), 39, 5, 48):
            raise ValueError("window IDs or forecast shape mismatch")
        values = np.array([reduce_samples(x, y[HIDDEN], mean, scale)
                           for x, y in zip(f["draws"], truth)])
        np.testing.assert_allclose(values, f["metrics"], rtol=2e-12, atol=2e-12)
    months = np.array([s[:7] for s in ids])
    by_month = {m: values[months == m].mean(axis=0) for m in sorted(set(months))}
    average = np.mean(list(by_month.values()), axis=0)
    np.testing.assert_allclose(average, [record["metrics"][k] for k in METRICS], rtol=2e-12, atol=2e-12)
    return by_month, record


def as_metrics(v):
    return dict(zip(METRICS, (float(x) for x in v)))


def summarize(root, val):
    p = json.loads(PROTOCOL.read_text())
    selected = json.loads((root / "selection.json").read_text())
    if selected != choices(root) or sha256_file(val) != p["training_inputs"]["val.npz"]:
        raise ValueError("invalid validation source or selection")
    with np.load(val, allow_pickle=False) as f:
        val_truth, val_ids, mean, scale = [f[k] for k in ("z","ids","mean","scale")]
    inv = json.loads((root / "holdout.json").read_text())
    if sha256_file(root / "holdout.npz") != inv["arrays_sha256"]:
        raise ValueError("holdout inventory differs from receipt")
    with np.load(root / "holdout.npz", allow_pickle=False) as f:
        truth, ids = f["z"], f["ids"]
    if list(ids) != inv["ids"] or set(val_ids).intersection(ids):
        raise ValueError("holdout IDs disagree or overlap validation")
    monthly = {}
    fits = {}
    for seed in p["seeds"]:
        for arm in p["arms"]:
            name = f"{arm}_{seed}"
            candidates = {}
            for epoch in EPOCHS:
                _, rec = audited_record(root/"scores"/"VAL"/name, epoch, val_truth, val_ids, mean, scale)
                candidates[epoch] = rec["metrics"]["energy"]
            epoch = selected[name]["epoch"]
            if min(candidates, key=lambda e:(candidates[e], e)) != epoch:
                raise ValueError("independent selection disagrees")
            by_month, rec = audited_record(root/"scores"/"HOLDOUT2025"/name, epoch, truth, ids, mean, scale)
            if (rec["checkpoint"] != selected[name]["checkpoint"] or rec["arrays"] != inv["arrays_sha256"]
                    or rec["split"] != "HOLDOUT2025" or rec["arm"] != arm or rec["seed"] != seed):
                raise ValueError("holdout forecast binding mismatch")
            monthly[name] = by_month
            fits[name] = {"validation_energy_by_epoch": candidates, "selected_epoch": epoch,
                          "checkpoint": rec["checkpoint"], "forecast_archive": rec["archive"],
                          "mean": rec["metrics"], "by_month": {m: as_metrics(v) for m,v in by_month.items()}}
    diffs = {}
    for seed in p["seeds"]:
        a,b = monthly[f"G5aux_{seed}"], monthly[f"G5_{seed}"]
        diffs[seed] = {m: a[m] - b[m] for m in b}
    pooled = {m: np.mean([d[m] for d in diffs.values()], axis=0) for m in by_month}
    return {"protocol_sha256": sha256_file(PROTOCOL), "source_sha256": inv["source_sha256"],
            "evaluation_year": 2025, "windows_per_month": inv["selected_by_month"], "fits": fits,
            "difference_by_seed": {s: as_metrics(np.mean(list(d.values()), axis=0)) for s,d in diffs.items()},
            "difference_by_month": {m: as_metrics(v) for m,v in pooled.items()},
            "difference": as_metrics(np.mean(list(pooled.values()), axis=0))}


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("root", type=Path)
    args.add_argument("validation", type=Path)
    args.add_argument("output", type=Path)
    a = args.parse_args()
    results = summarize(a.root, a.validation)
    a.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(results["difference"])
