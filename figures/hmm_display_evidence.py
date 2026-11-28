"""Reproduce the HMM display supplement from the published stationary window and RNG records.

Run with OPENBLAS_NUM_THREADS=1 for the small dense path sums. Only NumPy is required.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


INPUTS = {
    "toy.npz": "dbf80b22d72ddeaf8b609455be797cc7862edf9c585d5b76f7abdc3d6f4dcaf7",
    "hmm-stationary-01.json": "6fc53c027cd1c5ded40ef025cc1cdb6930fab8a1a750465d354d82164f2ae93e",
    "oracle-stationary-01.json": "fee403a2d4f823181ea08a7d726193d2ffcc50e5c85e4ded5810426877397c5f",
}
PROBS = np.array([0.1, 0.25, 0.5])
PATH_IDS = np.arange(2**12)
BITS = (PATH_IDS[:, None] >> np.arange(12)) % 2


def bridges():
    """Bridge regime weights, backwards transition probabilities and enumerated path law."""
    powers = np.array([[np.linalg.matrix_power([[1-p, p], [p, 1-p]], j)
                        for j in range(14)] for p in PROBS])
    result = {}
    for a in (0, 1):
        for b in (0, 1):
            endpoint = powers[:, 13, a, b]
            backward = np.stack([powers[:, j, a, 1, None] * powers[:, 1, 1, :] / powers[:, j+1, a, :]
                                 for j in range(1, 13)], axis=1)
            paths = np.column_stack((np.full(len(BITS), a), BITS, np.full(len(BITS), b)))
            switches = np.count_nonzero(np.diff(paths), axis=1)
            joint = PROBS[:, None]**switches * (1-PROBS[:, None])**(13-switches)
            np.testing.assert_allclose(joint.sum(axis=1), endpoint, rtol=0, atol=2e-14)
            pmf = joint.sum(axis=0)
            result[a, b] = endpoint / endpoint.sum(), backward, pmf / pmf.sum()
    return result


def draw(bridge, b, rng, n):
    weights, back, _ = bridge
    regimes = rng.choice(len(PROBS), size=n, p=weights)
    result = np.zeros((n, 12), dtype=int)
    for regime in range(len(PROBS)):
        selected = np.flatnonzero(regimes == regime)
        state = np.full(len(selected), b)
        for j in reversed(range(12)):
            state = (rng.random(len(selected)) < back[regime, j, state]).astype(int)
            result[selected, j] = state
    return result


def denoise(obs, alpha_bar, states, prior):
    """Posterior binwise state-1 probabilities, summing over all paths."""
    lo, hi = [-((obs - np.sqrt(alpha_bar)*state)**2).sum(-1) / (2*(1-alpha_bar))
              for state in states]
    ll = (hi-lo) @ BITS.T + np.log(prior)
    weights = np.exp(ll - ll.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ BITS


def distances(x, y):
    return np.linalg.norm(x[:, None] - y[None, :], axis=-1)


def replay_cases(record, laws, states, mean, scale):
    forward = np.random.default_rng(record["seeds"]["forward"])
    reference = np.random.default_rng(record["seeds"]["ref"])
    f = np.cos((np.arange(101)/100 + .008) / 1.008 * np.pi/2)**2
    abar = np.r_[1, np.cumprod(np.maximum(f[1:]/f[:-1], .001))]
    n = record["n_paths"]
    pairs = np.triu_indices(n, 1)
    rows = []
    for row in record["context"]:
        a, b, t = row["a"], row["b"], row["t"]
        physical = states[int(row["alpha"])]
        std = (physical-mean)/scale
        law = laws[a, b]
        ys = std[draw(law, b, reference, n)][..., :5].reshape(n, -1)
        obs = np.empty((n, 12, 6))
        for j in range(n):
            source = draw(law, b, forward, 1)[0]
            obs[j] = np.sqrt(abar[t])*std[source] + np.sqrt(1-abar[t])*forward.standard_normal((12, 6))
        q = denoise(obs, abar[t], std, law[2])
        if t > 1:
            first = std[0] + q[..., None]*(std[1]-std[0])
            noise = (obs-np.sqrt(abar[t])*first)/np.sqrt(1-abar[t])
            step = np.sqrt(abar[1])*first + np.sqrt(1-abar[1])*noise
            q = denoise(step, abar[1], std, law[2])
        z = physical[0] + q[..., None]*(physical[1]-physical[0])
        x = ((z-mean)/scale)[..., :5].reshape(n, -1)
        residual = z[..., 3]**2 + z[..., 4] - (z[..., :3]**2).sum(-1) - z[..., 5]
        np.testing.assert_allclose(
            [distances(x, ys).mean() - distances(x, x)[pairs].mean()/2,
             distances(ys, ys)[pairs].mean()/2, np.abs(residual).mean()],
            [row[k] for k in ("mean_es_vs_ref", "reference_self_energy_mc", "mean_abs_r_star")],
            rtol=0, atol=2e-11, err_msg=f"window {row['window']}, timestep {t}")
        if t == 100:
            decoded = np.column_stack((np.full(n, a), q > .5, np.full(n, b)))
            switches = np.count_nonzero(np.diff(decoded), axis=1)
            rows.append({k: row[k] for k in ("window", "alpha", "a", "b")} |
                        {"n": n, "minimum_switch_paths": int(np.count_nonzero(switches == (a != b))),
                         # Bit j is the decoded state in hidden bin j, counting from zero.
                         "decoded_path_ids": (decoded[:, 1:-1] @ (2**np.arange(12))).tolist()})
    return rows


def score_targets(laws, states, hidden, alpha, scale):
    """On the realized targets compute E|X-y| - E|X-X'|/2 with independent X,X'."""
    pmfs = np.column_stack([law[2] for law in laws.values()])
    distance = np.sqrt(BITS.sum(axis=1))
    target_term = np.empty_like(pmfs)
    for start in range(0, len(BITS), 64):
        ids = PATH_IDS[start:start+64]
        target_term[ids] = distance[ids[:, None] ^ PATH_IDS] @ pmfs
    half_self = (pmfs*target_term).sum(axis=0)/2
    unit = np.linalg.norm((states[:, 1, :5]-states[:, 0, :5])/scale[:5], axis=1)
    endpoints = 2*hidden[:, 0] + hidden[:, -1]
    targets = hidden[:, 18:30] @ (2**np.arange(12))
    return float(np.mean(unit[alpha]*(target_term[targets, endpoints]-half_self[endpoints])))


def compute(root):
    for name, expected in INPUTS.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"unexpected published input: {name}")
    hmm = json.loads((root/"hmm-stationary-01.json").read_text())
    oracle = json.loads((root/"oracle-stationary-01.json").read_text())
    with np.load(root/"toy.npz", allow_pickle=False) as data:
        z, alpha, mean, scale = (data[k] for k in ("z", "alpha", "mean", "scale"))
    hidden = (z[..., 3]**2 + z[..., 4] > 2.5).astype(int)
    states = np.array([[z[alpha == g][hidden[alpha == g] == s][0] for s in (0, 1)] for g in range(3)])
    np.testing.assert_array_equal(z, states[alpha[:, None], hidden])
    for i, ctx in enumerate(oracle["contexts"]):
        np.testing.assert_array_equal([ctx[k] for k in ("window", "alpha", "a", "b")],
                                      [i, alpha[i], hidden[i, 0], hidden[i, -1]])
    laws = bridges()
    rows = replay_cases(hmm, laws, states, mean, scale)
    return {"schema_version": 2, "inputs": INPUTS,
            "source": {"path": Path(__file__).name, "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "checked_context_rows": len(hmm["context"]), "t100": rows,
            "enumerated_target_score": score_targets(laws, states, hidden, alpha, scale)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=Path(__file__).parent/"evidence/toy")
    parser.add_argument("--output", type=Path, required=True, help="new diagnostic JSON path")
    args = parser.parse_args()
    result = compute(args.evidence_dir)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
