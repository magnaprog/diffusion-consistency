"""Fixed-noisy-posterior and replicated-context toy diagnostics, CPU only.

Defaults give 36 contexts per timestep, 144 records for each diagnostic.
The input boundary law and oracle information set must be declared explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prigm.config import HIDDEN_END, HIDDEN_START, N_HIDDEN, ORACLE_FORWARD_SEED, ORACLE_REF_SEED, TOY_CORRUPT_SEED
from prigm.keys import pcg64
from prigm.normalize import AffineNormalizer
from prigm.observation import residual_obs
from prigm.oracle import gaussian_log_emissions, infer_boundaries, infer_context, nearest_state, oracle_metadata, posterior_hidden, two_step_exact
from prigm.schedule import build_schedule
from prigm.scoring import pairwise_l2_offdiag
from prigm.toy import ALPHAS, PS, hmm_posterior, sample_hmm_paths, sample_mixture_paths, state_summary


TIMESTEPS = (1, 25, 50, 100)
N_CONTEXT_PER_REGIME = 4
N_PATHS = 128
N_REGIMES = 9


def _windows_by_regime(alphas: np.ndarray, ps: np.ndarray) -> list[np.ndarray]:
    """Descriptive groups only; generating p is not a pooled-oracle input."""
    return [np.flatnonzero((ps == p) & (alphas == alpha)) for p in PS for alpha in ALPHAS]


def _boundaries(z_window: np.ndarray, alpha: float, nrm: AffineNormalizer) -> tuple[int, int]:
    return infer_boundaries(z_window, alpha, nrm)


def _forward_corrupt(u0_hid: np.ndarray, t: int, sched: dict, rng: np.random.Generator) -> np.ndarray:
    return sched["a"][int(t)] * u0_hid + sched["s"][int(t)] * rng.standard_normal(u0_hid.shape)


def _sample_paths_given_gamma(
    a: int, b: int, p: float, emit: np.ndarray | None, rng: np.random.Generator, n: int, *, log_emit=None,
) -> np.ndarray:
    """Legacy regime-known FFBS API; log_emit is preferred for noisy inputs."""
    return sample_hmm_paths(hmm_posterior(a, b, p, emit, log_emit=log_emit), rng, n)


def _emit_from_u(u_t: np.ndarray, alpha: float, t: int, nrm: AffineNormalizer, sched: dict) -> np.ndarray:
    """Legacy row-scaled likelihood helper; inference uses log emissions directly."""
    le = gaussian_log_emissions(u_t, alpha, t, nrm, sched)
    return np.exp(le - le.max(axis=1, keepdims=True))


def _path_to_primary(paths: np.ndarray, alpha: float, nrm: AffineNormalizer) -> np.ndarray:
    states = np.array([state_summary(alpha, s) for s in (0, 1)])
    return nrm.transform(states[paths])[..., :5].reshape(paths.shape[0], N_HIDDEN * 5)


def _two_step_from_exact(
    u_t: np.ndarray, t: int, alpha: float, p: float | None, a: int, b: int, nrm: AffineNormalizer, sched: dict,
    *, conditioning="regime-known", boundary_law=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep the old return API, sharing the repaired two-step implementation."""
    z_hat, z_star, gamma = two_step_exact(u_t, t, alpha, p, a, b, nrm, sched,
                                         conditioning=conditioning, boundary_law=boundary_law)
    return z_hat, z_star, residual_obs(z_star), residual_obs(z_hat), gamma


def _replicated_context_endpoints(
    alpha: float, p: float | None, a: int, b: int, nrm: AffineNormalizer, sched: dict,
    t: int, rng: np.random.Generator, n: int, *, conditioning="regime-known", boundary_law=None,
) -> np.ndarray:
    """Generate repeated two-step estimates for one observed endpoint context.

    For each replicate, draw a fresh hidden path conditional on alpha and the
    endpoint states a,b, then add fresh forward noise at timestep t. The chosen
    conditioning mode determines whether p is supplied or averaged out. Apply
    the two-step estimator to that noisy path and return its physical endpoint.

    No recorded hidden truth is reused as the path for these replicates.
    Comparison paths must be drawn separately with an independent random stream.
    """
    kwargs = dict(conditioning=conditioning, boundary_law=boundary_law)
    context = posterior_hidden(alpha, p, a, b, nrm, **kwargs)
    states = np.array([state_summary(alpha, s) for s in (0, 1)])
    ends = []
    for _ in range(n):
        path = sample_mixture_paths(context, rng, 1)[0]
        u_t = _forward_corrupt(nrm.transform(states[path]), t, sched, rng)
        _, z_star, _ = two_step_exact(u_t, t, alpha, p, a, b, nrm, sched, **kwargs)
        ends.append(z_star)
    return np.array(ends)


def _entropy_fields(post: dict) -> dict:
    return {key: post[key] for key in (
        "path_entropy", "regime_entropy", "path_entropy_given_regime", "joint_regime_path_entropy",
    )}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toy-eval", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conditioning", choices=("pooled", "regime-known"), required=True)
    parser.add_argument("--boundary-law", choices=("iid", "stationary"), required=True)
    parser.add_argument("--n-paths", type=int, default=N_PATHS)
    parser.add_argument("--n-context-per-regime", type=int, default=N_CONTEXT_PER_REGIME)
    args = parser.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    if args.n_paths < 2 or args.n_context_per_regime < 1:
        raise ValueError("need n_paths>=2 and n_context_per_regime>=1")
    with np.load(args.toy_eval, allow_pickle=False) as blob:
        rec = oracle_metadata(Path(args.toy_eval), args.conditioning, args.boundary_law, blob)
        z, ps = blob["z"], blob["p"].astype(np.float64)
        nrm = AffineNormalizer(blob["mean"], blob["scale"], float(blob["b_ref_sq"]), int(blob["n_bins"]))
    if len(z) == 0 or len(ps) != len(z) or not np.all(np.isin(ps, PS)):
        raise ValueError("input needs nonempty aligned windows and declared generating-p strata")
    contexts = [infer_context(zw, nrm) for zw in z]
    groups = _windows_by_regime(np.array([c[0] for c in contexts]), ps)
    sched = build_schedule()
    rng_c, rng_f, rng_r = (pcg64(seed) for seed in (TOY_CORRUPT_SEED, ORACLE_FORWARD_SEED, ORACLE_REF_SEED))
    kwargs = dict(conditioning=args.conditioning, boundary_law=args.boundary_law)
    noisy_recs, ctx_recs = [], []
    for g_i, idx in enumerate(groups):
        for wi in idx[:args.n_context_per_regime]:
            alpha, a, b = contexts[wi]
            pval = float(ps[wi]) if args.conditioning == "regime-known" else None
            u0 = nrm.transform(z[wi, HIDDEN_START:HIDDEN_END])
            context = posterior_hidden(alpha, pval, a, b, nrm, entropies=True, **kwargs)
            for t in TIMESTEPS:
                common = {"regime": g_i, "window": int(wi), "t": int(t), "alpha": alpha,
                          "generating_p": float(ps[wi]), "a": a, "b": b,
                          "denoiser_evaluations_per_endpoint": 1 if t == 1 else 2}
                # One fixed noisy observation, anchored to this recorded target.
                # A concentrated posterior at low noise is not mode collapse.
                u_t = _forward_corrupt(u0, t, sched, rng_c)
                post = posterior_hidden(alpha, pval, a, b, nrm, u_t, t, sched, entropies=True, **kwargs)
                paths = sample_mixture_paths(post, rng_c, args.n_paths)
                X = _path_to_primary(paths, alpha, nrm)
                z_hat, z_star, r_star, r_hat, gamma = _two_step_from_exact(
                    u_t, t, alpha, pval, a, b, nrm, sched, **kwargs)
                noisy_recs.append({
                    **common, **_entropy_fields(post),
                    "posterior_p_weights": post["weights"].tolist(),
                    "log_noisy_evidence_given_context": post["log_evidence"],
                    "posterior_self_energy_mc": pairwise_l2_offdiag(X) / 2.0,
                    "mean_abs_r_hat": float(np.mean(np.abs(r_hat))),
                    "mean_abs_r_star": float(np.mean(np.abs(r_star))),
                    "mean_pi": float(np.mean(gamma[:, 1])),
                    "frac_nearest_1": float(np.mean([nearest_state(row, alpha, nrm) for row in z_hat])),
                    "n_unique_paths": int(len(np.unique(paths, axis=0))),
                })

                # References have their own stream; replicated corruptions
                # integrate over fresh context paths, not one recorded target.
                ref_paths = sample_mixture_paths(context, rng_r, args.n_paths)
                Yref = _path_to_primary(ref_paths, alpha, nrm)
                ends = _replicated_context_endpoints(alpha, pval, a, b, nrm, sched, t, rng_f,
                                                     args.n_paths, **kwargs)
                Xend = nrm.transform(ends)[..., :5].reshape(args.n_paths, -1)
                # Mean ES against independent context draws; pair term computed once.
                mean_es = np.linalg.norm(Xend[:, None] - Yref[None, :], axis=-1).mean() - pairwise_l2_offdiag(Xend) / 2.0
                ctx_recs.append({
                    **common, **_entropy_fields(context),
                    "context_p_weights": context["weights"].tolist(),
                    "mean_es_vs_ref": float(mean_es),
                    "reference_self_energy_mc": pairwise_l2_offdiag(Yref) / 2.0,
                    "mean_abs_r_star": float(np.mean(np.abs(residual_obs(ends)))),
                    "n_unique_ref": int(len(np.unique(ref_paths, axis=0))),
                })
        print(f"regime {g_i + 1}/{N_REGIMES} noisy={len(noisy_recs)} ctx={len(ctx_recs)}", flush=True)

    rec.update({
        "n_noisy": len(noisy_recs), "n_context": len(ctx_recs),
        "timesteps": list(TIMESTEPS), "n_paths": args.n_paths,
        "n_context_per_regime": args.n_context_per_regime,
        "seeds": {"corrupt": TOY_CORRUPT_SEED, "forward": ORACLE_FORWARD_SEED, "ref": ORACLE_REF_SEED},
        "entropy_units": "nats",
        "entropy_definitions": {
            "path_entropy": "H(S_1,...,S_12 | conditioning), with latent p marginalized",
            "regime_entropy": "H(p | conditioning)",
            "path_entropy_given_regime": "E_p H(S_1,...,S_12 | p, conditioning)",
            "joint_regime_path_entropy": "H(p,S_1,...,S_12 | conditioning)",
        },
        "noisy_conditioning": "context plus one fixed noisy input from the recorded target; low-noise concentration is legitimate posterior concentration, not mode collapse",
        "context_conditioning": "context only; each forward corruption starts from a fresh mixture-p then bridge draw; reference paths use an independent stream",
        "score_note": "Monte Carlo reference/self-scores are not an exact Bayes floor. Two-step endpoints are finite estimators without support projection.",
        "noisy_by_t": {}, "context_by_t": {}, "noisy": noisy_recs, "context": ctx_recs,
    })
    for t in TIMESTEPS:
        for group, records, fields in (
            ("noisy_by_t", noisy_recs, ("n_unique_paths", "path_entropy", "regime_entropy", "joint_regime_path_entropy", "mean_abs_r_star", "posterior_self_energy_mc")),
            ("context_by_t", ctx_recs, ("mean_es_vs_ref", "reference_self_energy_mc", "mean_abs_r_star", "path_entropy", "regime_entropy", "joint_regime_path_entropy")),
        ):
            subset = [r for r in records if r["t"] == t]
            rec[group][str(t)] = {"n": len(subset), **{
                key if key.startswith("mean_") else f"mean_{key}": float(np.mean([r[key] for r in subset]))
                for key in fields
            }}
    payload = json.dumps(rec, indent=2, allow_nan=False) + "\n"
    with out.open("x") as handle:
        handle.write(payload)
    print(json.dumps({"n_noisy": len(noisy_recs), "n_context": len(ctx_recs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
