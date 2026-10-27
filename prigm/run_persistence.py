"""Historical persistence baseline on frozen test arrays. CPU, fresh output.

Hold the last observed bin at index 17 constant across the hidden hour.
The baseline has no fitted parameters and uses no random draws.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prigm.config import HIDDEN_END, HIDDEN_START, PRIMARY_CHANNELS
from prigm.normalize import AffineNormalizer
from prigm.scoring import energy_score, month_balanced_mean


def primary_std(z: np.ndarray, nrm: AffineNormalizer) -> np.ndarray:
    return (z[..., :PRIMARY_CHANNELS] - nrm.mean[:PRIMARY_CHANNELS]) / nrm.scale[:PRIMARY_CHANNELS]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arrays", required=True)
    p.add_argument("--windows-json", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    blob = np.load(args.arrays)
    z = blob["z"]
    n = z.shape[0]
    nrm = AffineNormalizer(mean=blob["mean"], scale=blob["scale"],
                           b_ref_sq=float(blob["b_ref_sq"]), n_bins=int(blob["n_bins"]))
    windows = json.loads(Path(args.windows_json).read_text())
    months = np.array([int(w["year"]) * 100 + int(w["month"]) for w in windows[:n]], dtype=np.int64)
    years = np.array([int(w["year"]) for w in windows[:n]], dtype=np.int64)

    # persistence: copy bin 17 into all hidden bins
    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        hold = z[i, HIDDEN_START - 1]  # bin 17
        pred = np.repeat(hold[None, :], HIDDEN_END - HIDDEN_START, axis=0)
        xs = primary_std(pred[None, ...], nrm).reshape(-1)
        X = np.repeat(xs[None, :], 39, axis=0)
        y = primary_std(z[i, HIDDEN_START:HIDDEN_END][None, ...], nrm).reshape(-1)
        scores[i] = energy_score(X, y)

    # increment strata to compare with interpolation directly
    inc = np.linalg.norm(np.diff(z[:, HIDDEN_START:HIDDEN_END, :3], axis=1), axis=-1).mean(axis=1)
    edges = np.quantile(inc, [1/3, 2/3])
    tinc = np.digitize(inc, edges)

    rec = {
        "n": n,
        "overall": {
            "month_balanced": float(month_balanced_mean(scores, months)),
            "mean": float(scores.mean()),
        },
        "by_increment_tertile": {
            str(k): {
                "n": int((tinc == k).sum()),
                "month_balanced": float(month_balanced_mean(scores[tinc == k], months[tinc == k])),
            } for k in range(3)
        },
        "by_cycle_phase": {},
        "note": "Persistence: bin 17 held constant across the gap. Point mass, 39 copies for ES.",
    }
    phases = [(2005, 2009), (2010, 2015), (2016, 2019), (2020, 2024)]
    for lo, hi in phases:
        m = (years >= lo) & (years <= hi)
        rec["by_cycle_phase"][f"{lo}-{hi}"] = {
            "n": int(m.sum()),
            "month_balanced": float(month_balanced_mean(scores[m], months[m])),
        }
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
