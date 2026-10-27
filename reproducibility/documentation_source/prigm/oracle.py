"""Information-matched toy references and finite exact-denoiser samplers.

The input boundary law must be declared: frozen historical toy_eval windows
used iid boundaries; the current generator uses stationary boundaries. Neither
law describes a full 48-bin trajectory: the flanks repeat the two endpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prigm.config import HIDDEN_END, HIDDEN_START, N_BINS, N_DRAWS, N_HIDDEN, ORACLE_REF_SEED
from prigm.keys import noise_key, pcg64
from prigm.normalize import AffineNormalizer
from prigm.recovery import sha256_file
from prigm.schedule import build_schedule, ddim_timesteps
from prigm.scoring import energy_score, pairwise_l2_offdiag
from prigm.toy import ALPHAS, PS, exact_mean_from_gamma, mixture_hmm_posterior, sample_mixture_paths, state_summary


DDIM_BUDGETS = (25, 50, 100)


def nearest_state(z5: np.ndarray, alpha: float, nrm: AffineNormalizer) -> int:
    """Nearest of the two energy-matched states in standardized 5-d."""
    states = np.array([state_summary(alpha, s)[:5] for s in (0, 1)])
    distances = np.sum(((np.asarray(z5)[:5] - states) / nrm.scale[:5]) ** 2, axis=1)
    return int(np.argmin(distances))


def infer_boundaries(z_window: np.ndarray, alpha: float, nrm: AffineNormalizer) -> tuple[int, int]:
    return nearest_state(z_window[0], alpha, nrm), nearest_state(z_window[-1], alpha, nrm)


def infer_context(z_window: np.ndarray, nrm: AffineNormalizer) -> tuple[float, int, int]:
    """Identify alpha and endpoints from observed repeated flanks, never targets.

    This exact toy oracle rejects flanks outside its declared finite family.
    Repeated flank bins encode endpoints; they are not independent transitions.
    """
    observed = np.concatenate((z_window[:HIDDEN_START], z_window[HIDDEN_END:]))
    for alpha in ALPHAS:
        a, b = infer_boundaries(z_window, alpha, nrm)
        expected = np.concatenate((
            np.tile(state_summary(alpha, a), (HIDDEN_START, 1)),
            np.tile(state_summary(alpha, b), (N_BINS - HIDDEN_END, 1)),
        ))
        if np.allclose(observed, expected, rtol=1e-6, atol=1e-7):
            return float(alpha), a, b
    raise ValueError("observed flanks do not identify a declared toy alpha and boundary pair")


def gaussian_log_emissions(u_t: np.ndarray, alpha: float, t: int, nrm: AffineNormalizer, sched: dict) -> np.ndarray:
    """Normalized six-coordinate Gaussian log densities for each hidden state."""
    u_t = np.asarray(u_t, dtype=np.float64)
    if u_t.shape != (N_HIDDEN, 6) or not np.all(np.isfinite(u_t)):
        raise ValueError("u_t must be finite with shape (12, 6)")
    if t is None or not 1 <= int(t) <= sched["n_steps"]:
        raise ValueError("a noisy timestep in 1..n_steps is required")
    sigma = float(sched["s"][int(t)])
    if sigma <= 0:
        raise ValueError("Gaussian noise scale must be positive")
    states = nrm.transform(np.array([state_summary(alpha, s) for s in (0, 1)]))
    residual = (u_t[:, None, :] - sched["a"][int(t)] * states) / sigma
    return -0.5 * np.sum(residual ** 2, axis=2) - u_t.shape[1] * np.log(sigma * np.sqrt(2 * np.pi))


def posterior_hidden(
    alpha: float, p: float | None, a: int, b: int, nrm: AffineNormalizer,
    u_t=None, t=None, sched=None, *, conditioning="regime-known", boundary_law=None, entropies=False,
) -> dict:
    """Full posterior; old calls with a supplied p retain regime-known semantics.

    Pooled calls must pass p=None and an explicit boundary law. Context weights
    are updated with the current input's evidence on every invocation.
    """
    if conditioning == "pooled":
        if p is not None:
            raise ValueError("pooled inference must not receive held-out p; pass None")
        ps = PS
    elif conditioning == "regime-known":
        if p is None:
            raise ValueError("regime-known inference requires p")
        ps = (float(p),)
        # For a known p, conditioning on endpoints cancels the boundary law.
        boundary_law = "iid" if boundary_law is None else boundary_law
    else:
        raise ValueError("conditioning must be pooled or regime-known")
    log_emit = None if u_t is None else gaussian_log_emissions(u_t, alpha, t, nrm, sched)
    return mixture_hmm_posterior(a, b, boundary_law=boundary_law, ps=ps, log_emit=log_emit, entropies=entropies)


def exact_mean_hidden(
    alpha: float, p: float | None, a: int, b: int, nrm: AffineNormalizer,
    u_t=None, t=None, sched=None, *, conditioning="regime-known", boundary_law=None,
):
    """Physical posterior mean and state marginals, without support projection."""
    post = posterior_hidden(alpha, p, a, b, nrm, u_t, t, sched,
                            conditioning=conditioning, boundary_law=boundary_law)
    return exact_mean_from_gamma(alpha, post["gamma"]), post["gamma"]


def two_step_exact(
    u_t: np.ndarray, t: int, alpha: float, p: float | None, a: int, b: int,
    nrm: AffineNormalizer, sched: dict, *, conditioning="regime-known", boundary_law=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact mean, DDIM to t=1, then a newly conditioned exact mean.

    Returns first mean, endpoint, first marginals. One evaluation at t=1,
    otherwise two. Both are estimators, not exact conditional path samples.
    """
    kwargs = dict(conditioning=conditioning, boundary_law=boundary_law)
    z_hat, gamma = exact_mean_hidden(alpha, p, a, b, nrm, u_t, t, sched, **kwargs)
    if int(t) == 1:
        return z_hat, z_hat, gamma
    eps = (u_t - sched["a"][int(t)] * nrm.transform(z_hat)) / sched["s"][int(t)]
    u1 = sched["a"][1] * nrm.transform(z_hat) + sched["s"][1] * eps
    z_star, _ = exact_mean_hidden(alpha, p, a, b, nrm, u1, 1, sched, **kwargs)
    return z_hat, z_star, gamma


def ddim_exact(
    z_window: np.ndarray, alpha: float, p: float | None, a: int, b: int,
    nrm: AffineNormalizer, noise: np.ndarray, n_eval: int, sched: dict,
    *, conditioning="regime-known", boundary_law=None,
) -> np.ndarray:
    """noise (6, 48) standardized; physical hidden (12, 6) output.

    z_window is retained for API compatibility, but no hidden target is read.
    Exact conditioning is encoded by alpha, a, b and the declared information
    set. The same initial noise can be reused across computational budgets.
    """
    if not 2 <= n_eval <= sched["n_steps"]:
        raise ValueError("n_eval must be between 2 and n_steps")
    u = np.asarray(noise, dtype=np.float64).T[HIDDEN_START:HIDDEN_END].copy()
    steps = ddim_timesteps(n_eval, sched["n_steps"])
    for i, t_i in enumerate(steps):
        z_mean, _ = exact_mean_hidden(alpha, p, a, b, nrm, u, int(t_i), sched,
                                     conditioning=conditioning, boundary_law=boundary_law)
        u0_hat = nrm.transform(z_mean)
        if i == len(steps) - 1:
            return z_mean
        eps = (u - sched["a"][int(t_i)] * u0_hat) / sched["s"][int(t_i)]
        t_next = int(steps[i + 1])
        u = sched["a"][t_next] * u0_hat + sched["s"][t_next] * eps


def oracle_metadata(path: Path, conditioning: str, boundary_law: str, blob) -> dict:
    """Record caller-declared law without inventing historical source lineage."""
    recorded = str(blob["boundary_law"].item()) if "boundary_law" in blob else None
    if recorded is not None and recorded != boundary_law:
        raise ValueError(f"declared boundary law {boundary_law} disagrees with input {recorded}")
    return {
        "schema_version": 2,
        "toy_eval_sha256": sha256_file(path),
        "source_hashes": {
            name: sha256_file(Path(__file__).parent / name)
            for name in ("oracle.py", "hmm_diagnostics.py", "toy.py", "toy_windows.py", "keys.py",
                         "schedule.py", "scoring.py", "normalize.py", "observation.py", "config.py")
        },
        "conditioning": conditioning,
        "boundary_law": boundary_law,
        "boundary_law_source": "input field and caller" if recorded is not None else "caller declaration; input has no boundary_law field",
        "window_law": f"{boundary_law}_boundary_markov_bridge_with_repeated_endpoint_flanks",
        "p_prior": {str(p): 1.0 / len(PS) for p in PS},
        "p_prior_source": "declared equal PS family mixture in prigm.toy_windows.make_toy_training, independent of alpha; finite training counts and historical execution not certified by this input",
        "p_prior_role": "family prior used in pooled mode; regime-known mode instead fixes the supplied p",
        "context_p_weighting": "prior(p)" if boundary_law == "iid" else "prior(p) * (T_p^13)[a,b], normalized",
        "information": "observed alpha and endpoints; p marginalized" if conditioning == "pooled" else "observed alpha and endpoints plus supplied regime p (extra information)",
        "p_strata": "held-out generating p is descriptive only in pooled mode",
        "historical_law_note": "frozen historical toy_eval used iid boundaries; current sample_boundaries uses stationary boundaries; caller must establish input lineage",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toy-eval", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conditioning", choices=("pooled", "regime-known"), required=True)
    parser.add_argument("--boundary-law", choices=("iid", "stationary"), required=True)
    parser.add_argument("--n-eval", type=int, help="Optional additional DDIM budget; always compares 2/25/50/100 evaluations")
    parser.add_argument("--n-windows", type=int, default=288)
    parser.add_argument("--n-draws", type=int, default=N_DRAWS)
    args = parser.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    if args.n_windows < 1 or args.n_draws < 2:
        raise ValueError("need n_windows>=1 and n_draws>=2")
    with np.load(args.toy_eval, allow_pickle=False) as blob:
        rec = oracle_metadata(Path(args.toy_eval), args.conditioning, args.boundary_law, blob)
        z = blob["z"][:args.n_windows]
        ps = blob["p"][:args.n_windows]
        nrm = AffineNormalizer(blob["mean"], blob["scale"], float(blob["b_ref_sq"]), int(blob["n_bins"]))
    if len(z) == 0 or len(ps) != len(z):
        raise ValueError("input needs nonempty aligned z and p arrays")
    sched = build_schedule()
    budgets = sorted(set(DDIM_BUDGETS + (() if args.n_eval is None else (args.n_eval,))))
    if any(not 2 <= n <= sched["n_steps"] for n in budgets):
        raise ValueError("DDIM budget outside 2..n_steps")
    methods = {"exact_reference": [], "two_step": []}
    methods.update({f"ddim_{n}": [] for n in budgets})
    self_scores, contexts = [], []
    rng_ref = pcg64(ORACLE_REF_SEED)
    kwargs = dict(conditioning=args.conditioning, boundary_law=args.boundary_law)
    for i, zw in enumerate(z):
        alpha, a, b = infer_context(zw, nrm)
        pval = float(ps[i]) if args.conditioning == "regime-known" else None
        contexts.append({"window": i, "alpha": alpha, "generating_p": float(ps[i]), "a": a, "b": b})
        target = nrm.transform(zw[HIDDEN_START:HIDDEN_END])[:, :5].reshape(-1)
        post = posterior_hidden(alpha, pval, a, b, nrm, **kwargs)
        paths = sample_mixture_paths(post, rng_ref, args.n_draws)
        states = np.array([state_summary(alpha, s) for s in (0, 1)])
        ref = nrm.transform(states[paths])[..., :5].reshape(args.n_draws, -1)
        methods["exact_reference"].append(energy_score(ref, target))
        self_scores.append(pairwise_l2_offdiag(ref) / 2.0)
        draws = {name: [] for name in methods if name != "exact_reference"}
        for d in range(args.n_draws):
            rng = pcg64(noise_key("eval", "TOY", f"TOY:{i}", d, 7))
            noise = rng.standard_normal((N_BINS, 6))
            _, endpoint, _ = two_step_exact(noise[HIDDEN_START:HIDDEN_END], sched["n_steps"],
                                           alpha, pval, a, b, nrm, sched, **kwargs)
            draws["two_step"].append(nrm.transform(endpoint)[:, :5].reshape(-1))
            for n in budgets:
                endpoint = ddim_exact(zw, alpha, pval, a, b, nrm, noise.T, n, sched, **kwargs)
                draws[f"ddim_{n}"].append(nrm.transform(endpoint)[:, :5].reshape(-1))
        for name, samples in draws.items():
            methods[name].append(energy_score(np.array(samples), target))
        if (i + 1) % 16 == 0:
            print(f"{i + 1}/{len(z)}", flush=True)
    rec.update({
        "n": len(z), "n_draws": args.n_draws, "contexts": contexts,
        "initialization": "shared standard Gaussian hidden noise per window/draw across all finite samplers, t=n_steps",
        "reference_seed": ORACLE_REF_SEED,
        "finite_sampler_noise_key": "noise_key('eval', 'TOY', f'TOY:{window_index}', draw, 7), PCG64, float64 standard_normal((48,6))",
        "reference_self_energy_mc": float(np.mean(self_scores)),
        "note": "Exact conditional path reference draws; finite two-step/DDIM estimators. Computational budgets are not convergence claims. Neither finite sampler scores nor Monte Carlo self-scores are an exact Bayes floor. Posterior means are not support-projected.",
        "methods": {},
    })
    for name, values in methods.items():
        count = 0 if name == "exact_reference" else 2 if name == "two_step" else int(name.split("_")[1])
        rec["methods"][name] = {
            "denoiser_evaluations_per_draw": count,
            "total_denoiser_evaluations": count * len(z) * args.n_draws,
            "mean_energy": float(np.mean(values)), "scores": values,
            "mean_energy_by_generating_p": {str(p): float(np.mean(np.array(values)[ps == p])) for p in np.unique(ps)},
        }
    payload = json.dumps(rec, indent=2, allow_nan=False) + "\n"
    with out.open("x") as handle:
        handle.write(payload)
    print(json.dumps({name: item["mean_energy"] for name, item in rec["methods"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
