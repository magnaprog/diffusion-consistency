"""Compare historical predictions within groups of observed hidden-hour behavior.

Divide windows at the one-third and two-thirds quantiles of either the mean
step between consecutive hidden mean vectors or the mean absolute R_obs
residual of the observed hidden quantities. Within each group, report scores
and the fraction of windows where the mean neural score is below linear
interpolation's score. These comparisons describe the recorded windows.

Neural scores are read from saved files. Gaussian interpolation is fitted
and sampled again with seed 13, both with a joint hidden-hour residual and
with one pooled six-coordinate residual law sampled independently at each
position. The legacy dominance section tabulates score and residual means;
it does not decide dominance or compute uncertainty intervals. Run on CPU
and write to a new output path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prigm.config import GAUSSIAN_JITTER, HIDDEN_END, HIDDEN_START, N_DRAWS, PRIMARY_CHANNELS
from prigm.controls import gaussian_fit, gaussian_sample, linear_interpolate
from prigm.keys import pcg64
from prigm.normalize import AffineNormalizer
from prigm.observation import residual_obs
from prigm.scoring import energy_score, month_balanced_mean


ARMS = {
    "D": ["D_s7_g16252.json", "D_s13_g9560.json", "D_s42_g16252.json"],
    "IS": ["IS_s7_g16252.json", "IS_s13_g9560.json", "IS_s42_g2868.json"],
    "OS": ["OS_s7_g16252.json", "OS_s13_g9560.json", "OS_s42_g2868.json"],
    "G5": ["G5_s7_g9560.json", "G5_s13_g9560.json", "G5_s42_g2868.json"],
}

ENSEMBLES = {
    "D": ["D_s7_g16252.npz", "D_s13_g9560.npz", "D_s42_g16252.npz"],
    "IS": ["IS_s7_g16252.npz", "IS_s13_g9560.npz", "IS_s42_g2868.npz"],
    "OS": ["OS_s7_g16252.npz", "OS_s13_g9560.npz", "OS_s42_g2868.npz"],
    "G5": ["G5_s7_g9560.npz", "G5_s13_g9560.npz", "G5_s42_g2868.npz"],
}


def primary_std_hid(z_hid: np.ndarray, nrm: AffineNormalizer) -> np.ndarray:
    u = (z_hid[..., :PRIMARY_CHANNELS] - nrm.mean[:PRIMARY_CHANNELS]) / nrm.scale[:PRIMARY_CHANNELS]
    return np.reshape(u, u.shape[:-2] + (12 * PRIMARY_CHANNELS,))


def interp_scores(z: np.ndarray, nrm: AffineNormalizer) -> np.ndarray:
    out = np.empty(z.shape[0], dtype=np.float64)
    for i in range(z.shape[0]):
        pred = linear_interpolate(z[i])[HIDDEN_START:HIDDEN_END]
        true = z[i, HIDDEN_START:HIDDEN_END]
        xs = primary_std_hid(pred[None, ...], nrm)
        X = np.repeat(xs, N_DRAWS, axis=0)
        y = primary_std_hid(true[None, ...], nrm)[0]
        out[i] = energy_score(X, y)
    return out


def hidden_increment(z: np.ndarray) -> np.ndarray:
    """Mean Euclidean step of hidden mean-vector (mu_x, mu_y, mu_z)."""
    hid = z[:, HIDDEN_START:HIDDEN_END, :3]
    step = np.linalg.norm(np.diff(hid, axis=1), axis=-1)
    return step.mean(axis=1)


def truth_abs_r(z: np.ndarray) -> np.ndarray:
    hid = z[:, HIDDEN_START:HIDDEN_END]
    return np.mean(np.abs(residual_obs(hid)), axis=1)


def tertiles(x: np.ndarray) -> np.ndarray:
    q = np.quantile(x, [1.0 / 3.0, 2.0 / 3.0])
    return np.digitize(x, q)


def summarize(scores: np.ndarray, months: np.ndarray, mask: np.ndarray) -> dict:
    if not np.any(mask):
        return {"n": 0, "month_balanced": None, "mean": None}
    return {
        "n": int(mask.sum()),
        "month_balanced": float(month_balanced_mean(scores[mask], months[mask])),
        "mean": float(scores[mask].mean()),
    }


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
    nrm = AffineNormalizer(
        mean=blob["mean"], scale=blob["scale"], b_ref_sq=float(blob["b_ref_sq"]), n_bins=int(blob["n_bins"])
    )
    windows = json.loads(Path(args.windows_json).read_text())
    months = np.array([int(w["year"]) * 100 + int(w["month"]) for w in windows[: z.shape[0]]])
    inc = hidden_increment(z)
    rabs = truth_abs_r(z)
    t_inc = tertiles(inc)
    t_r = tertiles(rabs)
    interp = interp_scores(z, nrm)
    # Mean absolute observation-identity residual for the interpolated predictions.
    # Later tabulate this beside energy score; neither quantity alone decides dominance.
    r_interp = np.mean(np.abs(residual_obs(
        np.stack([linear_interpolate(w)[HIDDEN_START:HIDDEN_END] for w in z], axis=0)
    )), axis=1)
    train = np.load(args.train_arrays)
    z_tr = train["z"]
    hid_true_tr = z_tr[:, HIDDEN_START:HIDDEN_END]
    interp_tr = np.stack([linear_interpolate(w)[HIDDEN_START:HIDDEN_END] for w in z_tr], axis=0)
    u_true = nrm.transform(hid_true_tr.reshape(-1, 6)).reshape(hid_true_tr.shape)
    u_int = nrm.transform(interp_tr.reshape(-1, 6)).reshape(interp_tr.shape)
    resid = (u_true - u_int).reshape(z_tr.shape[0], 72)
    mu, cov = gaussian_fit(resid, 0.0, GAUSSIAN_JITTER)
    rng = pcg64(13)
    gauss = np.empty(z.shape[0], dtype=np.float64)
    for i in range(z.shape[0]):
        ref = nrm.transform(linear_interpolate(z[i])[HIDDEN_START:HIDDEN_END].reshape(-1, 6)).reshape(72)
        samples_u = gaussian_sample(ref, mu, cov, N_DRAWS, rng).reshape(N_DRAWS, 12, 6)
        z_s = samples_u * nrm.scale + nrm.mean
        true = z[i, HIDDEN_START:HIDDEN_END]
        xs = primary_std_hid(z_s, nrm)
        ys = primary_std_hid(true[None], nrm)[0]
        gauss[i] = energy_score(xs, ys)
    rec = {
        "n": int(z.shape[0]),
        "increment_tertile_edges": [float(x) for x in np.quantile(inc, [1.0 / 3.0, 2.0 / 3.0])],
        "rabs_tertile_edges": [float(x) for x in np.quantile(rabs, [1.0 / 3.0, 2.0 / 3.0])],
        "interp": {},
        "gaussian": {},
        "arms": {},
        "note": "Tertiles of true hidden-hour mean-vector increment. Frozen neural scores. Gaussian rho=0, seed 13, same as run_controls.",
    }
    rec["interp"]["overall"] = summarize(interp, months, np.ones(z.shape[0], dtype=bool))
    rec["interp"]["inc"] = {str(k): summarize(interp, months, t_inc == k) for k in range(3)}
    rec["interp"]["rabs"] = {str(k): summarize(interp, months, t_r == k) for k in range(3)}
    rec["gaussian"]["overall"] = summarize(gauss, months, np.ones(z.shape[0], dtype=bool))
    rec["gaussian"]["inc"] = {str(k): summarize(gauss, months, t_inc == k) for k in range(3)}
    rec["gaussian"]["rabs"] = {str(k): summarize(gauss, months, t_r == k) for k in range(3)}
    # Draw independently at 12 hidden positions from one pooled 6-d residual
    # Gaussian: shared fitted mean/covariance, no cross-position covariance.
    resid_b = resid.reshape(-1, 6)
    mu_b, cov_b = gaussian_fit(resid_b, 0.0, GAUSSIAN_JITTER)
    rng_b = pcg64(13)
    gauss_b = np.empty(z.shape[0], dtype=np.float64)
    for i in range(z.shape[0]):
        ref = nrm.transform(linear_interpolate(z[i])[HIDDEN_START:HIDDEN_END].reshape(-1, 6))
        samples = np.empty((N_DRAWS, 12, 6), dtype=np.float64)
        for t in range(12):
            samples[:, t, :] = gaussian_sample(ref[t], mu_b, cov_b, N_DRAWS, rng_b)
        z_s = samples * nrm.scale + nrm.mean
        true = z[i, HIDDEN_START:HIDDEN_END]
        xs = primary_std_hid(z_s, nrm)
        ys = primary_std_hid(true[None], nrm)[0]
        gauss_b[i] = energy_score(xs, ys)
    rec["gaussian_perbin"] = {
        "overall": summarize(gauss_b, months, np.ones(z.shape[0], dtype=bool)),
        "inc": {str(k): summarize(gauss_b, months, t_inc == k) for k in range(3)},
        "note": "12 independent 6-d residual Gaussians. No cross-lag covariance.",
    }
    score_dir = Path(args.score_dir)
    for arm, files in ARMS.items():
        seed_scores = []
        for fn in files:
            d = json.loads((score_dir / fn).read_text())
            s = np.array(d["scores"], dtype=np.float64)
            if s.shape[0] != z.shape[0]:
                raise ValueError(fn)
            seed_scores.append(s)
        mean_s = np.mean(np.stack(seed_scores, axis=0), axis=0)
        rec["arms"][arm] = {
            "overall": summarize(mean_s, months, np.ones(z.shape[0], dtype=bool)),
            "inc": {str(k): summarize(mean_s, months, t_inc == k) for k in range(3)},
            "rabs": {str(k): summarize(mean_s, months, t_r == k) for k in range(3)},
            "frac_windows_better_than_interp": float(np.mean(mean_s < interp)),
            "inc_frac_better": {str(k): float(np.mean(mean_s[t_inc == k] < interp[t_inc == k])) for k in range(3)},
        }
    # Tabulate energy score and mean absolute residual for each arm.
    ens_dir = args.ensemble_dir
    arm_r = {}
    for arm, spec in ENSEMBLES.items():
        vals = []
        for fn in spec:
            pack = np.load(Path(ens_dir) / fn)
            e = pack["ensemble"]  # (n, K, C, L)
            n_, K_, C_, L_ = e.shape
            win_r = np.empty(n_, dtype=np.float64)
            for i in range(n_):
                u = e[i].astype(np.float64)
                hid = np.transpose(u[:, :, HIDDEN_START:HIDDEN_END], (0, 2, 1))
                if C_ == 5:
                    z5 = hid * nrm.scale[:5] + nrm.mean[:5]
                    zphys = np.concatenate((z5, z5[..., 3:4] ** 2 + z5[..., 4:5]
                                            - z5[..., 0:1] ** 2 - z5[..., 1:2] ** 2 - z5[..., 2:3] ** 2), axis=-1)
                else:
                    zphys = hid * nrm.scale[:6] + nrm.mean[:6]
                win_r[i] = float(np.mean(np.abs(residual_obs(zphys))))
            vals.append(win_r)
        arm_r[arm] = np.mean(np.stack(vals, axis=0), axis=0)
    rec["dominance"] = {}
    for arm in ("D", "IS", "OS", "G5"):
        rec["dominance"][arm] = {
            "mean_r_obs": float(arm_r[arm].mean()),
            "mean_es": float(np.mean([json.loads((Path(args.score_dir) / f).read_text())["mean"]
                                       for f in ARMS[arm]])),
        }
    rec["dominance"]["interp_pointmass"] = {
        "mean_r_obs": float(r_interp.mean()),
        "mean_es": rec["interp"]["overall"]["mean"],
    }
    # Retain the historical metadata below for payload compatibility. This
    # section tabulates mean energy score and mean absolute residual; it does
    # not compute dominance decisions or uncertainty intervals.
    rec["dominance"]["note"] = ("Dominance on (energy score, mean |R_obs|). "
                                 "Lower is better on both axes.")
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({k: rec["arms"][k]["overall"] for k in rec["arms"]}, indent=2))
    print("interp", rec["interp"]["overall"])
    print("gaussian", rec["gaussian"]["overall"], rec["gaussian"]["inc"])
    print("gaussian_perbin", rec["gaussian_perbin"]["overall"], rec["gaussian_perbin"]["inc"])
    print("inc edges", rec["increment_tertile_edges"])
    for arm in rec["arms"]:
        print(arm, "inc", rec["arms"][arm]["inc"], "frac_better", rec["arms"][arm]["frac_windows_better_than_interp"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
