"""Historical solar-cycle and field-strength strata on frozen neural test scores.

Cycle phases follow sunspot-cycle convention for cycles 23/24/25: the
23/24 minimum sits in 2008-2009, cycle 24 maximum near 2014, the 24/25
minimum near December 2019, and cycle 25 rises through 2024.

Field-strength strata split windows by the mean hidden-bin |B| summary m.
Gaussian rho=0 is recomputed deterministically exactly as run_strata
(same training residual fit, same seed 13 draw stream).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prigm.config import GAUSSIAN_JITTER, HIDDEN_END, HIDDEN_START, PRIMARY_CHANNELS
from prigm.controls import gaussian_fit, gaussian_sample, linear_interpolate
from prigm.keys import pcg64
from prigm.normalize import AffineNormalizer
from prigm.observation import residual_obs
from prigm.scoring import energy_score, month_balanced_mean, month_balanced_mean_replicated

ARMS = {
    "D": ["D_s7_g16252.json", "D_s13_g9560.json", "D_s42_g16252.json"],
    "IS": ["IS_s7_g16252.json", "IS_s13_g9560.json", "IS_s42_g2868.json"],
    "OS": ["OS_s7_g16252.json", "OS_s13_g9560.json", "OS_s42_g2868.json"],
    "G5": ["G5_s7_g9560.json", "G5_s13_g9560.json", "G5_s42_g2868.json"],
}
ENSEMBLES_ROB = {
    "D": ["D_s7_g16252.npz", "D_s13_g9560.npz", "D_s42_g16252.npz"],
    "OS": ["OS_s7_g16252.npz", "OS_s13_g9560.npz", "OS_s42_g2868.npz"],
}

PHASES = (
    (range(2005, 2010), "2005-2009 decline-to-min (cyc 23/24)"),
    (range(2010, 2016), "2010-2015 rise-and-max (cyc 24)"),
    (range(2016, 2020), "2016-2019 decline (cyc 24)"),
    (range(2020, 2025), "2020-2024 min-and-rise (cyc 24/25)"),
)


def decode(ens: np.ndarray, nrm: AffineNormalizer) -> np.ndarray:
    u = np.asarray(ens, dtype=np.float64)
    hid = np.transpose(u[:, :, :, HIDDEN_START:HIDDEN_END], (0, 1, 3, 2))
    c = hid.shape[-1]
    if c == 5:
        z5 = hid * nrm.scale[:5] + nrm.mean[:5]
        vv = z5[..., 3] ** 2 + z5[..., 4] - z5[..., 0] ** 2 - z5[..., 1] ** 2 - z5[..., 2] ** 2
        return np.concatenate((z5, vv[..., None]), axis=-1)
    return hid * nrm.scale[:6] + nrm.mean[:6]


def primary_std(z: np.ndarray, nrm: AffineNormalizer) -> np.ndarray:
    return (z[..., :PRIMARY_CHANNELS] - nrm.mean[:PRIMARY_CHANNELS]) / nrm.scale[:PRIMARY_CHANNELS]


def summarize(scores: np.ndarray, months: np.ndarray, mask: np.ndarray) -> dict:
    return {
        "n": int(mask.sum()),
        "month_balanced": float(month_balanced_mean(scores[mask], months[mask])),
        "mean": float(scores[mask].mean()),
    }


def gap_ci(neural: np.ndarray, base: np.ndarray, mask: np.ndarray, months: np.ndarray,
           reps: int = 2000, seed: int = 7010) -> dict:
    """Bootstrap 95% CI of the month-balanced (neural - base) gap within a phase.

    Resample whole years with replacement, preserving Aug+Sep pairing.
    Month ids use YYYYMM; each drawn occurrence retains separate month weights.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    phase_months = months[mask]
    phase_gap = neural[mask] - base[mask]
    year_list = np.unique(phase_months // 100)
    gaps = np.empty(reps, dtype=np.float64)
    for r in range(reps):
        chosen = rng.choice(len(year_list), size=len(year_list), replace=True)
        gaps[r] = month_balanced_mean_replicated(
            phase_gap, phase_months, lambda m: int(m) // 100, year_list[chosen],
        )
    lo, hi = np.quantile(gaps, [0.025, 0.975])
    return {"ci95": [float(lo), float(hi)]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arrays", required=True)
    p.add_argument("--train-arrays", required=True)
    p.add_argument("--windows-json", required=True)
    p.add_argument("--score-dir", required=True)
    p.add_argument("--ensemble-dir", required=True)
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
    years = np.array([int(w["year"]) for w in windows[:n]], dtype=np.int64)
    months = np.array([int(w["year"]) * 100 + int(w["month"]) for w in windows[:n]], dtype=np.int64)

    # frozen neural seed-mean per-window scores
    arm_scores = {}
    for arm, files in ARMS.items():
        ss = []
        for fn in files:
            d = json.loads((Path(args.score_dir) / fn).read_text())
            s = np.array(d["scores"], dtype=np.float64)
            if s.shape[0] != n:
                raise ValueError(fn)
            ss.append(s)
        arm_scores[arm] = np.mean(np.stack(ss, axis=0), axis=0)

    # deterministic interpolation scores
    interp = np.empty(n, dtype=np.float64)
    for i in range(n):
        pred = linear_interpolate(z[i])[HIDDEN_START:HIDDEN_END]
        true = z[i, HIDDEN_START:HIDDEN_END]
        xs = primary_std(pred[None, ...], nrm).reshape(-1)
        X = np.repeat(xs[None, :], 39, axis=0)
        y = primary_std(true[None, ...], nrm).reshape(-1)
        interp[i] = energy_score(X, y)

    # deterministic Gaussian rho=0 exactly as run_strata
    train = np.load(args.train_arrays)
    z_tr = train["z"]
    hid_tr = z_tr[:, HIDDEN_START:HIDDEN_END]
    int_tr = np.stack([linear_interpolate(w)[HIDDEN_START:HIDDEN_END] for w in z_tr], axis=0)
    u_true = nrm.transform(hid_tr.reshape(-1, 6)).reshape(hid_tr.shape)
    u_int = nrm.transform(int_tr.reshape(-1, 6)).reshape(int_tr.shape)
    resid = (u_true - u_int).reshape(z_tr.shape[0], 72)
    mu, cov = gaussian_fit(resid, 0.0, GAUSSIAN_JITTER)
    rng = pcg64(13)
    gauss = np.empty(n, dtype=np.float64)
    for i in range(n):
        ref = nrm.transform(linear_interpolate(z[i])[HIDDEN_START:HIDDEN_END].reshape(-1, 6)).reshape(72)
        su = gaussian_sample(ref, mu, cov, 39, rng).reshape(39, 12, 6)
        zs = su * nrm.scale + nrm.mean
        xs = primary_std(zs, nrm).reshape(39, -1)
        ys = primary_std(z[i, HIDDEN_START:HIDDEN_END][None], nrm).reshape(-1)
        gauss[i] = energy_score(xs, ys)

    # field strength: mean hidden-bin m (channel 3, physical nT)
    m_win = z[:, HIDDEN_START:HIDDEN_END, 3].mean(axis=1)
    m_edges = np.quantile(m_win, [1 / 3, 2 / 3])
    m_grp = np.digitize(m_win, m_edges)

    # mean |R_obs| per window for D and OS from frozen ensembles
    rob = {}
    for arm, files in ENSEMBLES_ROB.items():
        per_seed = []
        for fn in files:
            e = np.load(Path(args.ensemble_dir) / fn)["ensemble"]
            d = decode(e.astype(np.float64), nrm)
            per_seed.append(np.mean(np.abs(residual_obs(d)), axis=(1, 2)))
        rob[arm] = np.mean(np.stack(per_seed, axis=0), axis=0)

    groups = {}
    for bounds, label in PHASES:
        mask = np.isin(years, list(bounds))
        groups[label] = mask
    groups["all"] = np.ones(n, dtype=bool)
    m_groups = {
        "weak-B": m_grp == 0,
        "mid-B": m_grp == 1,
        "strong-B": m_grp == 2,
    }

    rec = {
        "n": n,
        "m_edges_nT": [float(x) for x in m_edges],
        "years_present": sorted(int(y) for y in np.unique(years)),
        "counts_by_year": {str(y): int((years == y).sum()) for y in np.unique(years)},
        "phases": {},
        "mstrata": {},
        "note": ("Frozen neural scores; interp and Gaussian rho=0 recomputed "
                 "deterministically (seed 13). Bootstrap resamples whole years "
                 "within each phase, 2000 replicates, seed 7010."),
    }

    def fill(dct, mask, key):
        entry = {"n": int(mask.sum()),
                 "arms": {}}
        allscores = {"interp": interp, "gauss": gauss, **arm_scores}
        for name, s in allscores.items():
            entry["arms"][name] = summarize(s, months, mask)
        for arm in ("D", "IS", "OS", "G5"):
            g = summarize(arm_scores[arm], months, mask)["month_balanced"] - \
                summarize(interp, months, mask)["month_balanced"]
            entry["arms"][arm]["gap_vs_interp"] = float(g)
        # D-vs-interp bootstrap CI
        entry["gap_D_interp_ci"] = gap_ci(arm_scores["D"], interp, mask, months)
        # residual reduction OS relative to D within group
        rd = float(rob["OS"][mask].mean() / rob["D"][mask].mean())
        entry["rob_OS_over_D"] = rd
        entry["rob_D_mean"] = float(rob["D"][mask].mean())
        entry["rob_OS_mean"] = float(rob["OS"][mask].mean())
        dct[key] = entry

    for label, mask in groups.items():
        fill(rec["phases"], mask, label)
    for label, mask in m_groups.items():
        fill(rec["mstrata"], mask, label)

    out.write_text(json.dumps(rec, indent=2) + "\n")
    for label, entry in rec["phases"].items():
        if label == "all":
            continue
        print(label, "n=", entry["n"],
              "interp", round(entry["arms"]["interp"]["month_balanced"], 3),
              "D", round(entry["arms"]["D"]["month_balanced"], 3),
              "G5", round(entry["arms"]["G5"]["month_balanced"], 3),
              "gapD", round(entry["arms"]["D"]["gap_vs_interp"], 3),
              "ci", [round(x, 3) for x in entry["gap_D_interp_ci"]["ci95"]],
              "robOS/D", round(entry["rob_OS_over_D"], 3))
    for label, entry in rec["mstrata"].items():
        print(label, "n=", entry["n"],
              "interp", round(entry["arms"]["interp"]["month_balanced"], 3),
              "gauss", round(entry["arms"]["gauss"]["month_balanced"], 3),
              "G5", round(entry["arms"]["G5"]["month_balanced"], 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
