"""Prespecified 288 toy evaluation windows (plan Section 8)."""

from __future__ import annotations

import numpy as np

from prigm.config import TOY_PATH_SEED
from prigm.keys import pcg64
from prigm.toy import ALPHAS, PS
from prigm.toy_windows import _ffbs, sample_boundaries, window_from_path


def build_eval_windows(n_per_regime: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reuse 32 latent paths per p across the three alphas. Seed 7001."""
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
