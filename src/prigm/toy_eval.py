"""Build paired toy evaluation windows, with 288 windows by default.

For each of three transition probabilities p, draw n_per_regime hidden
paths and endpoint pairs, then reuse them for all three alpha settings.
Thus the default n_per_regime=32 yields 3 * 3 * 32 windows. Endpoint pairs
follow the current stationary law in prigm.toy_windows; saved historical
inputs must retain their own recorded or explicitly declared law (plan Section 8).
"""

from __future__ import annotations

import numpy as np

from prigm.config import TOY_PATH_SEED
from prigm.keys import pcg64
from prigm.toy import ALPHAS, PS
from prigm.toy_windows import _ffbs, sample_boundaries, window_from_path


def build_eval_windows(n_per_regime: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reuse each sampled hidden path and endpoint pair across the three alphas.

    Draw n_per_regime paths for each p using seed 7001. The default is 32 paths
    per p and 288 returned windows; changing n_per_regime changes that count.
    Return the physical windows and their aligned alpha and p labels.
    """
    rng = pcg64(TOY_PATH_SEED)
    paths = {}
    bounds = {}
    for p in PS:
        plist = []
        blist = []
        for _ in range(n_per_regime):
            a, b = sample_boundaries(p, rng)
            path = _ffbs(p, a, b, rng)
            plist.append(path)
            blist.append((a, b))
        paths[p] = plist
        bounds[p] = blist
    zs = []
    alphas = []
    ps = []
    for p in PS:
        for alpha in ALPHAS:
            for path, (a, b) in zip(paths[p], bounds[p]):
                zs.append(window_from_path(alpha, p, a, b, path))
                alphas.append(alpha)
                ps.append(p)
    return np.stack(zs, axis=0), np.array(alphas), np.array(ps)
