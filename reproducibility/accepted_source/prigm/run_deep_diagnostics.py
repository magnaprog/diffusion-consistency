"""Deep diagnostics on frozen ensembles. CPU. No redraw. Fresh output.

1. Ensemble-size sensitivity: energy score at K in {5, 10, 20, 39}
   (subsample draws without replacement, 8 repetitions, mean).
2. Per-coordinate CRPS breakdown: which channels (mu, m, vs) drive the gap.
3. Calibration-by-bin: empirical coverage of the hidden mean vector norm
   at nominal 90% per hidden bin position.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prigm.config import HIDDEN_END, HIDDEN_START, PRIMARY_CHANNELS
from prigm.normalize import AffineNormalizer
from prigm.scoring import crps_univariate, energy_score


ARMS = {
    "D": ["D_s7_g16252.npz", "D_s13_g9560.npz", "D_s42_g16252.npz"],
    "OS": ["OS_s7_g16252.npz", "OS_s13_g9560.npz", "OS_s42_g2868.npz"],
    "G5": ["G5_s7_g9560.npz", "G5_s13_g9560.npz", "G5_s42_g2868.npz"],
}
GAUSS_FILES = None  # filled at runtime from controls if requested


def decode(ens: np.ndarray, nrm: AffineNormalizer) -> np.ndarray:
    """ens (n,K,C,L) standardized -> (n,K,12,6) physical hidden.

    Channel axis is axis 2, length axis is axis 3. Slice channels 18:30 on
    the LENGTH axis? No: stored layout is (n, K, C=6, L=48), and the hidden
    window is a slice of L. Transpose to (n, K, L_hidden, C).
    """
    u = np.asarray(ens, dtype=np.float64)
    hid = np.transpose(u[:, :, :, HIDDEN_START:HIDDEN_END], (0, 1, 3, 2))  # n,K,12,C
    c = hid.shape[-1]
    if c == 5:
        z5 = hid * nrm.scale[:5] + nrm.mean[:5]
        vv = z5[..., 3] ** 2 + z5[..., 4] - z5[..., 0] ** 2 - z5[..., 1] ** 2 - z5[..., 2] ** 2
        return np.concatenate((z5, vv[..., None]), axis=-1)
    return hid * nrm.scale[:6] + nrm.mean[:6]


def primary_std(z_phys: np.ndarray, nrm: AffineNormalizer) -> np.ndarray:
    u = (z_phys[..., :PRIMARY_CHANNELS] - nrm.mean[:PRIMARY_CHANNELS]) / nrm.scale[:PRIMARY_CHANNELS]
    return u


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arrays", required=True)
    p.add_argument("--ensemble-dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    blob = np.load(args.arrays)
    z = blob["z"]
    nrm = AffineNormalizer(mean=blob["mean"], scale=blob["scale"],
                           b_ref_sq=float(blob["b_ref_sq"]), n_bins=int(blob["n_bins"]))
    truth = z[:, HIDDEN_START:HIDDEN_END]  # (n,12,6)
    ys = primary_std(truth, nrm)  # (n,12,5)
    flat_y = ys.reshape(z.shape[0], -1)

    rec = {"n": int(z.shape[0]), "arms": {}}

    for arm, files in ARMS.items():
        arm_rec = {"seeds": {}}
        ens_list, dec_list = [], []
        for fn in files:
            e = np.load(Path(args.ensemble_dir) / fn)["ensemble"].astype(np.float64)
            d = decode(e, nrm)
            ens_list.append(e)
            dec_list.append(d)
            arm_rec["seeds"][fn] = {"n_draws": int(e.shape[1])}

        # --- ensemble size sensitivity (seed-mean) ---
        ks = [5, 10, 20, 39]
        rng = np.random.default_rng(77)
        es_by_k = {k: [] for k in ks}
        for d in dec_list:
            flats = primary_std(d, nrm).reshape(d.shape[0], d.shape[1], -1)  # (n,K,60)
            for k in ks:
                vals = []
                for rep in range(8):
                    idx = rng.choice(d.shape[1], size=k, replace=False)
                    acc = 0.0
                    for w in range(d.shape[0]):
                        acc += energy_score(flats[w][idx], flat_y[w])
                    vals.append(acc / d.shape[0])
                es_by_k[k].append(float(np.mean(vals)))
        arm_rec["es_by_k"] = {str(k): float(np.mean(v)) for k, v in es_by_k.items()}
        arm_rec["es_by_k_sem"] = {str(k): float(np.std(v) / np.sqrt(len(v))) for k, v in es_by_k.items()}

        # --- per-coordinate CRPS (seed-mean), channels: mu_x,mu_y,mu_z,m,vs ---
        crps_chan = np.zeros((5,), dtype=np.float64)
        n_tot = 0
        for d in dec_list:
            flatC = primary_std(d, nrm)  # (n,K,12,5)
            n_, K_ = flatC.shape[0], flatC.shape[1]
            n_tot += n_
            for w in range(n_):
                cvals = crps_univariate(flatC[w].reshape(K_, -1), flat_y[w])
                crps_chan += cvals.reshape(12, 5).mean(axis=0)
        arm_rec["crps_by_channel"] = {k: float(v / n_tot) for k, v in
                                       zip(("mu_x", "mu_y", "mu_z", "m", "vs"), crps_chan)}

        # --- per-bin coverage at nominal 90% (seed-ensembled draws) ---
        # concatenate draws across seeds: (n, 3K, 12, 5)
        big = np.concatenate([primary_std(d, nrm) for d in dec_list], axis=1)
        lo = np.quantile(big, 0.05, axis=1)
        hi = np.quantile(big, 0.95, axis=1)
        cov = ((ys >= lo) & (ys <= hi)).mean(axis=0)  # (12,5)
        arm_rec["cov90_by_bin_mu"] = [float(x) for x in cov[:, 0]]
        arm_rec["cov90_by_bin_m"] = [float(x) for x in cov[:, 3]]
        arm_rec["cov90_mean"] = float(((ys >= lo) & (ys <= hi)).mean())

        rec["arms"][arm] = arm_rec
        print(arm, "es_by_k", arm_rec["es_by_k"], "cov90", arm_rec["cov90_mean"], flush=True)

    # Gaussian per-window CRPS would need redraw; skip. Note it.
    rec["note"] = ("Frozen ensembles only. K-sensitivity subsamples draws without "
                   "replacement (8 reps). CRPS channels averaged over 12 bins. "
                   "Coverage concatenates 3x39 draws per window.")
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
