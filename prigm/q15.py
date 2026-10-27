"""Compare two reconstructions of a 15-minute magnetic second moment.

Q15 is the mean squared vector magnitude over 15 consecutive minute vectors,
in nT^2. The full reconstruction averages ||mu||^2 + v_v over three
five-minute bins. The mean-only reconstruction averages ||mu||^2 and omits
the variance within each bin. Both are compared with the original vectors.

Use complete, UTC-aligned blocks within eligible training windows, with
timestamp and instrument-source joins checked against the minute records.
This tests aggregation of the observed quantities; it does not evaluate
instrument quality or a model's ability to transfer between time resolutions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prigm.calendar import select_quota_indices
from prigm.config import N_BINS, N_MINUTES_PER_BIN, ROLE_TRAIN, YEARS
from prigm.data import (
    Bin,
    consecutive_runs,
    file_sha256,
    form_bins,
    hash_bytes,
    magnetic_valid,
    omni_path,
    parse_omni_year,
    validate_minutes,
)


def q15_from_three(z3: np.ndarray) -> tuple[float, float]:
    """z3: (3, 6). Returns (Q15 from three bins, loss if each bin replaced by mu)."""
    q_keep = float(np.mean(z3[:, 0] ** 2 + z3[:, 1] ** 2 + z3[:, 2] ** 2 + z3[:, 5]))
    q_mean_only = float(np.mean(z3[:, 0] ** 2 + z3[:, 1] ** 2 + z3[:, 2] ** 2))
    lost = q_keep - q_mean_only  # = mean v_v for valid summaries
    return q_keep, lost


def eligible_q15_blocks(bins: list[Bin]) -> tuple[list[list[Bin]], dict]:
    """UTC-aligned triples wholly inside the union of eligible four-hour windows."""
    train_bins = [b for b in bins if b.role == ROLE_TRAIN]
    runs = consecutive_runs(train_bins)
    # With stride one, windows_from_run covers every bin of a consecutive run
    # iff len(run) >= N_BINS. Avoid materializing all overlapping window arrays.
    eligible = [run for run in runs if len(run) >= N_BINS]
    blocks = []
    for run in eligible:
        for i in range(len(run) - 2):
            if int(run[i].left.astype("datetime64[m]").astype(np.int64)) % 15 == 0:
                blocks.append(run[i : i + 3])
    counts = {
        "train_bins": len(train_bins),
        "train_runs": len(runs),
        "eligible_train_runs": len(eligible),
        "eligible_train_bins": sum(len(run) for run in eligible),
        "eligible_train_starts": sum(len(run) - N_BINS + 1 for run in eligible),
        "eligible_blocks": len(blocks),
    }
    return blocks, counts


def compare_q15_raw(minutes: np.ndarray, blocks: list[list[Bin]]) -> np.ndarray:
    """Return columns (raw Q15, full-summary Q15, mean-only Q15), in nT^2.

    Raw Q15 uses the original 15 vectors, without calling the summary operator.
    Timestamp/source joins are checked independently of the supplied summaries.
    """
    validate_minutes(minutes)
    if not blocks:
        return np.empty((0, 3), dtype=np.float64)
    for block in blocks:
        if len(block) != 3:
            raise ValueError("Q15 requires three bins")
        start = block[0].left.astype("datetime64[m]")
        if int(start.astype(np.int64)) % 15:
            raise ValueError("Q15 block must be UTC 15-minute aligned")
        if any(b.left != start + np.timedelta64(N_MINUTES_PER_BIN * i, "m")
               or b.source != block[0].source for i, b in enumerate(block)):
            raise ValueError("Q15 bins must be consecutive and same-source")
    lefts = np.array([block[0].left for block in blocks], dtype="datetime64[m]")
    expected = lefts[:, None] + np.arange(15).astype("timedelta64[m]")[None, :]
    idx = np.searchsorted(minutes["ts"], expected)
    if np.any(idx >= len(minutes)) or np.any(minutes["ts"][idx] != expected):
        raise ValueError("Q15 block lacks its 15 original consecutive minute records")
    raw = minutes[idx]
    sources = np.array([block[0].source for block in blocks])
    if not np.all(magnetic_valid(raw)) or np.any(raw["source"] != sources[:, None]):
        raise ValueError("Q15 raw vectors must be valid and same-source")
    B = np.stack((raw["bx"], raw["by"], raw["bz"]), axis=-1)
    direct = np.mean(np.sum(B ** 2, axis=-1), axis=-1)
    z = np.array([[b.z for b in block] for block in blocks], dtype=np.float64)
    if z.shape != (len(blocks), 3, 6) or not np.all(np.isfinite(z)):
        raise ValueError("Q15 summaries must be finite with shape (3, 6) per block")
    mu_sq = np.sum(z[..., :3] ** 2, axis=-1)
    full = np.mean(mu_sq + z[..., 5], axis=-1)
    mean_only = np.mean(mu_sq, axis=-1)
    return np.column_stack((direct, full, mean_only))


def _source_counts(sources: np.ndarray) -> dict:
    ids, counts = np.unique(sources, return_counts=True)
    return {str(int(s)): int(n) for s, n in zip(ids, counts)}


def _block_digest(lefts: np.ndarray, sources: np.ndarray) -> str:
    records = "".join(f"{left},{int(src)}\n" for left, src in zip(lefts, sources))
    return hash_bytes(records.encode("ascii"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--omni-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-blocks", type=int, default=None,
                   help="Optional deterministic spread-out cap across the full training population; default: all")
    args = p.parse_args(argv)
    if args.max_blocks is not None and args.max_blocks <= 0:
        p.error("--max-blocks must be positive")
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    if not out.parent.is_dir():
        raise FileNotFoundError(out.parent)
    comparisons, all_lefts, all_sources, block_years = [], [], [], []
    inputs = {}
    population_by_year = {}
    for year in YEARS:
        path = omni_path(year, args.omni_root)
        digest = file_sha256(path)
        minutes = parse_omni_year(path)
        blocks, counts = eligible_q15_blocks(form_bins(minutes))
        comparisons.append(compare_q15_raw(minutes, blocks))
        lefts = np.array([b[0].left for b in blocks], dtype="datetime64[m]")
        sources = np.array([b[0].source for b in blocks], dtype=np.int32)
        all_lefts.append(lefts)
        all_sources.append(sources)
        block_years.append(np.full(len(blocks), year, dtype=np.int32))
        inputs[str(year)] = {"path": str(path.resolve()), "sha256": digest, "n_minutes": len(minutes)}
        population_by_year[str(year)] = {**counts, "source_block_counts": _source_counts(sources)}
        print(f"{year} eligible_blocks={len(blocks)}", flush=True)
    values = np.concatenate(comparisons)
    lefts = np.concatenate(all_lefts)
    sources = np.concatenate(all_sources)
    years = np.concatenate(block_years)
    n_population = len(values)
    selected = (np.arange(n_population) if args.max_blocks is None
                else select_quota_indices(n_population, args.max_blocks))
    direct, full, mean_only = values[selected].T
    full_loss = direct - full
    mean_only_loss = direct - mean_only
    selection_by_year = {}
    for year in YEARS:
        mask = years[selected] == year
        selection_by_year[str(year)] = {
            "n_blocks": int(mask.sum()),
            "source_block_counts": _source_counts(sources[selected][mask]),
        }
    rec = {
        "schema_version": 2,
        "n_blocks": len(selected),
        "population_n_blocks": n_population,
        "max_abs_err_vs_raw": float(np.max(np.abs(full_loss))) if len(selected) else None,
        "mean_full_moment_loss_nT2": float(np.mean(full_loss)) if len(selected) else None,
        "mean_abs_full_moment_loss_nT2": float(np.mean(np.abs(full_loss))) if len(selected) else None,
        "mean_only_loss_mean_nT2": float(np.mean(mean_only_loss)) if len(selected) else None,
        "mean_only_loss_median_nT2": float(np.median(mean_only_loss)) if len(selected) else None,
        "mean_only_loss_max_nT2": float(np.max(mean_only_loss)) if len(selected) else None,
        "raw_inputs": inputs,
        "population_by_year": population_by_year,
        "population_source_block_counts": _source_counts(sources),
        "population_blocks_sha256": _block_digest(lefts, sources),
        "selection": {
            "method": "all" if args.max_blocks is None else "chronological_midpoint_indices",
            "max_blocks": args.max_blocks,
            "index_rule": "floor((j+0.5)*N/q), j=0..q-1; q=min(N,max_blocks) when capped",
            "weighting": "equal block weight",
            "by_year": selection_by_year,
            "blocks_sha256": _block_digest(lefts[selected], sources[selected]),
            "first_left": str(lefts[selected[0]]) if len(selected) else None,
            "last_left": str(lefts[selected[-1]]) if len(selected) else None,
        },
        "definitions": {
            "population": "Complete same-source UTC 15-minute blocks; all three bins in the stride-one four-hour training-window union, with no month/role/source/gap crossing.",
            "raw_Q15": "mean over the original 15 vectors of bx^2+by^2+bz^2, nT^2",
            "full_Q15": "mean over three bins of ||mu||^2+v_v, nT^2",
            "mean_only_Q15": "mean over three bins of ||mu||^2, nT^2",
            "loss": "raw_Q15 minus the stated summary reconstruction, nT^2",
            "block_digest": "SHA256 of ordered ASCII records UTC-left,source followed by newline",
        },
        "note": "Read-only aggregation check, not an instrument-quality or learned-resolution-transfer test. Cap selection uses the full all-year training population, never model outputs.",
    }
    with out.open("x") as f:
        f.write(json.dumps(rec, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: rec[k] for k in ("n_blocks", "population_n_blocks", "max_abs_err_vs_raw",
                                         "mean_only_loss_mean_nT2")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
