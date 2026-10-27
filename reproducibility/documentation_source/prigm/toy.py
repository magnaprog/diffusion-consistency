"""Energy-matched two-state Markov toy (plan Section 8)."""

from __future__ import annotations

import numpy as np

from prigm.config import N_HIDDEN
from prigm.observation import summaries_from_vectors


C_J = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
ALPHAS = (0, 1, 2)
PS = (0.1, 0.25, 0.5)


def d_alpha(alpha: float) -> float:
    return float(np.sqrt(1.0 + 2.0 * alpha * alpha))


def kappa_alpha(alpha: float) -> float:
    a = float(alpha)
    return (1.0 + 2.0 * np.sqrt(1.0 + a * a) + 2.0 * np.sqrt(1.0 + 4.0 * a * a)) / 5.0


def vectors_state(alpha: float, state: int) -> np.ndarray:
    a = float(alpha)
    da = d_alpha(a)
    if state == 0:
        return np.stack([(1.0, a * c, 0.0) for c in C_J], axis=0) / da
    if state == 1:
        return np.stack(
            [
                2.0
                * np.array(
                    (1.0 + 2.0 * a * c, 2.0 - 2.0 * a * c, 2.0 + a * c),
                    dtype=np.float64,
                )
                / (3.0 * da)
                for c in C_J
            ],
            axis=0,
        )
    raise ValueError(state)


def state_summary(alpha: float, state: int) -> np.ndarray:
    return summaries_from_vectors(vectors_state(alpha, state))


def pooled_toy_observations() -> np.ndarray:
    rows = [state_summary(a, s) for a in ALPHAS for s in (0, 1)]
    return np.stack(rows, axis=0)


def transition_matrix(p: float) -> np.ndarray:
    p = float(p)
    return np.array([[1.0 - p, p], [p, 1.0 - p]], dtype=np.float64)


def bits_from_index(k: int, n: int = N_HIDDEN) -> np.ndarray:
    return np.array([(k >> i) & 1 for i in range(n)], dtype=np.int64)


def path_probability(states: np.ndarray, a: int, b: int, p: float) -> float:
    T = transition_matrix(p)
    s = np.concatenate(([int(a)], np.asarray(states, dtype=int), [int(b)]))
    num = 1.0
    for i in range(len(s) - 1):
        num *= T[s[i], s[i + 1]]
    den = float(np.linalg.matrix_power(T, len(s) - 1)[a, b])
    return float(num / den)


def all_path_probs(a: int, b: int, p: float) -> np.ndarray:
    n = 1 << N_HIDDEN
    out = np.empty(n, dtype=np.float64)
    for k in range(n):
        out[k] = path_probability(bits_from_index(k), a, b, p)
    return out


def exact_conditional_mean_residual_coeff(alpha: float) -> float:
    a = float(alpha)
    da = d_alpha(a)
    k = kappa_alpha(a)
    return (11.0 / 3.0 - k * k) / (da * da)


def _log_emissions(emit=None, log_emit=None) -> tuple[np.ndarray, float]:
    if emit is not None and log_emit is not None:
        raise ValueError("supply emit or log_emit, not both")
    if log_emit is None:
        emit = np.ones((N_HIDDEN, 2)) if emit is None else np.asarray(emit, dtype=np.float64)
        if np.any(emit < 0) or not np.all(np.isfinite(emit)):
            raise ValueError("emissions must be finite nonnegative likelihoods")
        with np.errstate(divide="ignore"):
            log_emit = np.log(emit)
    log_emit = np.asarray(log_emit, dtype=np.float64)
    if log_emit.shape != (N_HIDDEN, 2) or np.any(np.isnan(log_emit)) or np.any(np.isposinf(log_emit)):
        raise ValueError("log emissions must have shape (12, 2), without NaN or +inf")
    offsets = np.max(log_emit, axis=1)
    if not np.all(np.isfinite(offsets)):
        raise ValueError("every emission row must allow at least one state")
    return log_emit - offsets[:, None], float(offsets.sum())


def _entropy(log_probs: np.ndarray) -> float:
    finite = np.isfinite(log_probs)
    return float(-np.sum(np.exp(log_probs[finite]) * log_probs[finite]))


def hmm_posterior(a: int, b: int, p: float, emit=None, *, log_emit=None) -> dict:
    """Log-space bridge inference, including log P(emissions | a,b,p).

    Row offsets are retained in evidence, but removed during inference. Prefer
    log_emit for Gaussian observations: exponentiating first can erase a row.
    path_entropy is the joint entropy of all 12 states, in nats.
    """
    if a not in (0, 1) or b not in (0, 1) or not 0 < p < 1:
        raise ValueError("binary boundaries and 0 < p < 1 required")
    le, offset = _log_emissions(emit, log_emit)
    T = transition_matrix(p)
    lt = np.log(T)
    log_boundary = float(np.log(np.linalg.matrix_power(T, N_HIDDEN + 1)[a, b]))
    forward = np.empty((N_HIDDEN, 2))
    scales = np.empty(N_HIDDEN)
    forward[0] = lt[a] + le[0]
    scales[0] = np.logaddexp.reduce(forward[0])
    forward[0] -= scales[0]
    for t in range(1, N_HIDDEN):
        forward[t] = np.logaddexp.reduce(forward[t - 1, :, None] + lt, axis=0) + le[t]
        scales[t] = np.logaddexp.reduce(forward[t])
        forward[t] -= scales[t]
    backward = np.empty_like(forward)
    backward[-1] = lt[:, b]
    for t in range(N_HIDDEN - 2, -1, -1):
        backward[t] = np.logaddexp.reduce(lt + le[t + 1] + backward[t + 1], axis=1)
        backward[t] -= np.logaddexp.reduce(backward[t])
    lg = forward + backward
    lg -= np.logaddexp.reduce(lg, axis=1, keepdims=True)
    # H(S_1) + sum H(S_{t+1} | S_t), not a sum of marginal entropies.
    entropy = _entropy(lg[0])
    for t in range(N_HIDDEN - 1):
        pair = forward[t, :, None] + lt + le[t + 1] + backward[t + 1]
        pair -= np.logaddexp.reduce(pair.ravel())
        row = np.logaddexp.reduce(pair, axis=1, keepdims=True)
        finite = np.isfinite(pair)
        conditional = np.zeros_like(pair)
        np.subtract(pair, row, out=conditional, where=finite)
        entropy -= float(np.sum(np.exp(pair[finite]) * conditional[finite]))
    centered_evidence = float(scales.sum() + np.logaddexp.reduce(forward[-1] + lt[:, b]) - log_boundary)
    return {
        "gamma": np.exp(lg),
        "log_evidence": centered_evidence + offset,
        "log_evidence_centered": centered_evidence,
        "log_emission_offset": offset,
        "log_boundary_probability": log_boundary,
        "path_entropy": entropy,
        "log_forward": forward,
        "log_transition": lt,
        "right_boundary": b,
    }


def hmm_forward_backward(a: int, b: int, p: float, emit=None, *, log_emit=None) -> np.ndarray:
    """Backward-compatible marginal API; log_emit avoids likelihood underflow."""
    return hmm_posterior(a, b, p, emit, log_emit=log_emit)["gamma"]


def mixture_hmm_posterior(
    a: int, b: int, *, boundary_law: str, ps=PS, prior=None,
    emit=None, log_emit=None, entropies: bool = False,
) -> dict:
    """Marginalize persistence before scoring or taking a posterior mean.

    The equal prior over PS is the generator's equal mixture, independent of
    alpha. Stationary boundaries reweight it by (T_p^13)[a,b]; historical iid
    boundaries do not. Noisy evidence then updates these context weights.
    Enumeration for the non-Markov mixture's joint path entropy is optional.
    """
    if boundary_law not in ("iid", "stationary"):
        raise ValueError("boundary_law must explicitly be iid or stationary")
    ps = np.asarray(ps, dtype=np.float64)
    if ps.ndim != 1 or ps.size == 0 or np.unique(ps).size != ps.size:
        raise ValueError("ps must be a nonempty vector of distinct regimes")
    prior = np.ones(len(ps)) if prior is None else np.asarray(prior, dtype=np.float64)
    if prior.shape != ps.shape or not np.all(np.isfinite(prior)) or np.any(prior < 0) or not 0 < prior.sum() < np.inf:
        raise ValueError("prior must be nonnegative with positive finite mass")
    prior = prior / prior.sum()
    le, offset = _log_emissions(emit, log_emit)
    components = [hmm_posterior(a, b, float(p), log_emit=le) for p in ps]
    with np.errstate(divide="ignore"):
        log_context = np.log(prior)
    if boundary_law == "stationary":
        log_context += np.array([c["log_boundary_probability"] for c in components])
    log_context -= np.logaddexp.reduce(log_context)
    # Work with centered evidence: huge shared Gaussian offsets must not
    # destroy the small between-regime likelihood ratios by cancellation.
    log_weights = log_context + np.array([c["log_evidence_centered"] for c in components])
    log_evidence = float(np.logaddexp.reduce(log_weights))
    log_weights -= log_evidence
    weights = np.exp(log_weights)
    result = {
        "ps": ps,
        "context_weights": np.exp(log_context),
        "weights": weights,
        "gamma": np.einsum("p,ptk->tk", weights, np.array([c["gamma"] for c in components])),
        "log_evidence": log_evidence + offset,
        "components": components,
    }
    if entropies:
        paths = ((np.arange(1 << N_HIDDEN)[:, None] >> np.arange(N_HIDDEN)) & 1)
        full = np.column_stack((np.full(len(paths), a), paths, np.full(len(paths), b)))
        emission = le[np.arange(N_HIDDEN), paths].sum(axis=1)
        log_paths = []
        for c in components:
            lp = c["log_transition"][full[:, :-1], full[:, 1:]].sum(axis=1) + emission
            lp -= np.logaddexp.reduce(lp)
            log_paths.append(lp)
        mixture_paths = np.logaddexp.reduce(np.array(log_paths) + log_weights[:, None], axis=0)
        result["path_entropy"] = _entropy(mixture_paths)
        result["regime_entropy"] = _entropy(log_weights)
        result["path_entropy_given_regime"] = float(weights @ np.array([c["path_entropy"] for c in components]))
        result["joint_regime_path_entropy"] = result["regime_entropy"] + result["path_entropy_given_regime"]
    return result


def sample_hmm_paths(posterior: dict, rng: np.random.Generator, n: int) -> np.ndarray:
    """Log-space FFBS from an hmm_posterior result, including noisy evidence."""
    if n < 0:
        raise ValueError("n must be nonnegative")
    forward, lt = posterior["log_forward"], posterior["log_transition"]
    paths = np.empty((n, N_HIDDEN), dtype=np.int64)
    last = forward[-1] + lt[:, posterior["right_boundary"]]
    last -= np.logaddexp.reduce(last)
    paths[:, -1] = rng.random(n) < np.exp(last[1])
    for t in range(N_HIDDEN - 2, -1, -1):
        lp = forward[t] + lt[:, paths[:, t + 1]].T
        lp -= np.logaddexp.reduce(lp, axis=1, keepdims=True)
        paths[:, t] = rng.random(n) < np.exp(lp[:, 1])
    return paths


def sample_mixture_paths(posterior: dict, rng: np.random.Generator, n: int) -> np.ndarray:
    """First sample latent p from its posterior, then the corresponding bridge."""
    regimes = rng.choice(len(posterior["weights"]), size=n, p=posterior["weights"])
    paths = np.empty((n, N_HIDDEN), dtype=np.int64)
    for j, component in enumerate(posterior["components"]):
        idx = np.flatnonzero(regimes == j)
        paths[idx] = sample_hmm_paths(component, rng, len(idx))
    return paths


def gamma_from_enumeration(a: int, b: int, p: float) -> np.ndarray:
    probs = all_path_probs(a, b, p)
    gamma = np.zeros((N_HIDDEN, 2), dtype=np.float64)
    for k, pk in enumerate(probs):
        bits = bits_from_index(k)
        for t, s in enumerate(bits):
            gamma[t, s] += pk
    return gamma


def exact_mean_from_gamma(alpha: float, gamma: np.ndarray) -> np.ndarray:
    z0 = state_summary(alpha, 0)
    z1 = state_summary(alpha, 1)
    pi = gamma[:, 1]
    return (1.0 - pi)[:, None] * z0 + pi[:, None] * z1
