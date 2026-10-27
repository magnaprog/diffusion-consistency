"""Embed a two-state Markov bridge with twelve hidden positions in a length-48
toy window (plan Section 8).

Sample endpoint a from its stationary probability 1/2, then endpoint b
using T^13[a, b]. The 13 transitions connect the two observed endpoints
through 12 hidden states. Forward filtering and backward sampling (FFBS)
draw the hidden path conditional on those endpoints.

The 18 observed bins on the left all repeat state a's six quantities; the
18 on the right repeat state b's quantities. These flanks encode endpoints,
not 36 additional transitions of a length-48 Markov trajectory. Each state's
energy remains fixed across alpha, although the two states differ in energy.
"""

from __future__ import annotations

import numpy as np

from prigm.config import HIDDEN_END, HIDDEN_START, N_BINS, N_HIDDEN
from prigm.keys import pcg64
from prigm.toy import ALPHAS, PS, state_summary, transition_matrix


def sample_boundaries(p: float, rng: np.random.Generator) -> tuple[int, int]:
    """Stationary P(a)=1/2, P(b|a)=(T^13)_{ab} for 12 hidden bins plus two flanks."""
    T = transition_matrix(p)
    T13 = np.linalg.matrix_power(T, 13)
    a = int(rng.integers(0, 2))
    b = int(rng.choice(2, p=T13[a]))
    return a, b


def sample_path(p: float, a: int, b: int, rng: np.random.Generator) -> np.ndarray:
    """Sample the two-state bridge over twelve hidden positions by exact FFBS given endpoints a, b."""
    return _ffbs(p, a, b, rng)


def _ffbs(p: float, a: int, b: int, rng: np.random.Generator) -> np.ndarray:
    T = transition_matrix(p)
    n = N_HIDDEN
    alpha = np.zeros((n, 2))
    alpha[0] = T[a]
    alpha[0] /= alpha[0].sum()
    for t in range(1, n):
        alpha[t] = alpha[t - 1] @ T
        alpha[t] /= alpha[t].sum()
    s = np.empty(n, dtype=np.int64)
    # last: P(s_n | a) * T[s_n, b]
    last = alpha[-1] * T[:, b]
    last /= last.sum()
    s[-1] = rng.choice(2, p=last)
    for t in range(n - 2, -1, -1):
        post = alpha[t] * T[:, s[t + 1]]
        post /= post.sum()
        s[t] = rng.choice(2, p=post)
    return s


def window_from_path(alpha: float, p: float, a: int, b: int, path: np.ndarray) -> np.ndarray:
    z = np.empty((N_BINS, 6), dtype=np.float64)
    za = state_summary(alpha, a)
    zb = state_summary(alpha, b)
    z[:HIDDEN_START] = za
    for i, st in enumerate(path):
        z[HIDDEN_START + i] = state_summary(alpha, int(st))
    z[HIDDEN_END:] = zb
    return z


def make_toy_training(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign n windows cyclically across nine regimes.

    Counts differ by at most one and are equal when n is divisible by nine.
    Return z (n, 48, 6), alpha, and p.
    """
    rng = pcg64(seed)
    regimes = [(a, p) for a in ALPHAS for p in PS]
    z = np.empty((n, N_BINS, 6), dtype=np.float64)
    al = np.empty(n, dtype=np.float64)
    pv = np.empty(n, dtype=np.float64)
    for i in range(n):
        alpha, p = regimes[i % 9]
        a, b = sample_boundaries(p, rng)
        path = _ffbs(p, a, b, rng)
        z[i] = window_from_path(alpha, p, a, b, path)
        al[i] = alpha
        pv[i] = p
    return z, al, pv
