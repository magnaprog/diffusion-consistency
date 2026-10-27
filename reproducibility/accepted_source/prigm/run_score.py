"""Score frozen ensembles in float64. Fresh output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from prigm.config import HIDDEN_END, HIDDEN_START, PRIMARY_CHANNELS
from prigm.artifacts import json_bytes, load_arrays, read_ensemble, require_array_match, require_procedure_match, require_source_match
from prigm.barrier import load_protocol, require_test_started
from prigm.recovery import ContractMismatch, atomic_write_bytes, sha256_file
from prigm.normalize import AffineNormalizer
from prigm.observation import closure_correct, derive_vv
from prigm.scoring import energy_score, month_balanced_mean


def hidden_primary_std(z_phys_hid: np.ndarray, nrm: AffineNormalizer) -> np.ndarray:
    """z_phys_hid (..., 12, 6) -> (..., 60) standardized primary coords."""
    prim = z_phys_hid[..., :PRIMARY_CHANNELS]
    mean = nrm.mean[:PRIMARY_CHANNELS]
    scale = nrm.scale[:PRIMARY_CHANNELS]
    u = (prim - mean) / scale
    return np.reshape(u, u.shape[:-2] + (12 * PRIMARY_CHANNELS,))


def decode_ensemble(ens: np.ndarray, nrm: AffineNormalizer) -> np.ndarray:
    """ens (K, C, L) standardized f32/f64 -> physical hidden (K, 12, 6) float64."""
    u = np.asarray(ens, dtype=np.float64)
    hid = np.transpose(u[:, :, HIDDEN_START:HIDDEN_END], (0, 2, 1))  # K,12,C
    c = hid.shape[-1]
    if c == 5:
        z5 = hid * nrm.scale[:5] + nrm.mean[:5]
        return derive_vv(z5)
    if c == 6:
        return hid * nrm.scale[:6] + nrm.mean[:6]
    raise ValueError(c)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ensemble", required=True)
    p.add_argument("--arrays", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--correct", action="store_true")
    p.add_argument("--split", required=True, choices=("VAL", "TEST", "TOY"))
    p.add_argument("--study-root", required=True)
    p.add_argument("--protocol-sha256", required=True)
    args = p.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    protocol = load_protocol(Path(args.study_root), args.protocol_sha256)
    require_source_match(protocol)
    selection = require_test_started(Path(args.study_root), args.split, args.protocol_sha256)
    ens, ids, meta = read_ensemble(Path(args.ensemble))
    if meta["protocol_sha256"] != args.protocol_sha256:
        raise ContractMismatch("ensemble belongs to a different protocol")
    require_procedure_match(meta, protocol, selection)
    z, target_ids, nrm, identity = load_arrays(
        Path(args.arrays), args.split, protocol["arrays"][args.split]["sha256"]
    )
    require_array_match(meta, identity)
    if ids != target_ids:
        raise ContractMismatch("scoring requires complete ensembles in target window order")
    n = len(ids)
    months = (np.zeros(n, dtype=np.int64) if args.split == "TOY" else
              np.array([int(w[:4]) * 100 + int(w[5:7]) for w in ids], dtype=np.int64))
    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        z_s = decode_ensemble(ens[i], nrm)
        if args.correct:
            z_s = closure_correct(z_s)
        y = z[i, HIDDEN_START:HIDDEN_END]
        xs = hidden_primary_std(z_s, nrm)
        ys = hidden_primary_std(y[None, ...], nrm)[0]
        scores[i] = energy_score(xs, ys)
    mb = float(month_balanced_mean(scores, months))
    rec = {
        "schema_version": 2,
        "protocol_sha256": args.protocol_sha256,
        "ensemble_commit_sha256": sha256_file(Path(args.ensemble) / "gen_00000000/COMMIT.json"),
        "arrays": identity,
        "procedure": meta.get("procedure", "diffusion"),
        "checkpoint_sha256": meta.get("checkpoint_sha256"),
        "control_fit_sha256": meta.get("control_fit_sha256"),
        "ids": ids,
        "corrected": bool(args.correct),
        "n": int(n),
        "month_balanced": mb,
        "mean": float(scores.mean()),
        "scores": scores.tolist(),
        "months": months.tolist(),
    }
    atomic_write_bytes(out, json_bytes(rec))
    print(json.dumps({"month_balanced": rec["month_balanced"], "mean": rec["mean"], "n": n}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
