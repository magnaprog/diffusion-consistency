"""Training-only OMNI quality report. Does not change eligibility or manifests.

Plan Section 5: extra columns are metadata, not denoiser inputs.
Zero-based: 6 IMF point count, 8 PFN percent interpolated, 9-10 shift,
11 PFN spread, 12 DBOT1. DBOT sentinels 9999999 and 999999.
"""

from __future__ import annotations

import argparse
import calendar
import json
from pathlib import Path

import numpy as np

from prigm.calendar import role_for_date
from prigm.config import (
    COL_BX,
    COL_BY,
    COL_BZ,
    COL_DOY,
    COL_HOUR,
    COL_IMF_SOURCE,
    COL_MINUTE,
    COL_YEAR,
    HIDDEN_END,
    HIDDEN_START,
    N_BINS,
    N_MINUTES_PER_BIN,
    OMNI_DIR,
    ROLE_TRAIN,
    YEARS,
)
from prigm.data import (
    _minutes_from_raw,
    consecutive_runs,
    file_sha256,
    form_bins,
    magnetic_valid,
    omni_path,
    windows_from_run,
)


COL_POINTS = 6
COL_PFN_PCT = 8
COL_SHIFT = 9
COL_SHIFT_STD = 10
COL_PFN_SPREAD = 11
COL_DBOT = 12
DBOT_FILL = {999999.0, 9999999.0}
PFN_PCT_FILL = 999.0


def refuse_existing(path: Path) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refuse existing output {path}")
    return path


def parse_quality_year(path: Path) -> np.ndarray:
    raw = np.loadtxt(
        path,
        usecols=(
            COL_YEAR,
            COL_DOY,
            COL_HOUR,
            COL_MINUTE,
            COL_IMF_SOURCE,
            COL_BX,
            COL_BY,
            COL_BZ,
            COL_POINTS,
            COL_PFN_PCT,
            COL_SHIFT,
            COL_SHIFT_STD,
            COL_PFN_SPREAD,
            COL_DBOT,
        ),
        dtype=np.float64,
        ndmin=2,
    )
    dtype = np.dtype(
        [
            ("year", np.int32),
            ("doy", np.int32),
            ("hour", np.int32),
            ("minute", np.int32),
            ("source", np.int32),
            ("bx", np.float64),
            ("by", np.float64),
            ("bz", np.float64),
            ("points", np.float64),
            ("pfn_pct", np.float64),
            ("shift", np.float64),
            ("shift_std", np.float64),
            ("pfn_spread", np.float64),
            ("dbot", np.float64),
            ("ts", "datetime64[m]"),
        ]
    )
    minutes = _minutes_from_raw(raw[:, :8])
    out = np.empty(len(raw), dtype=dtype)
    for name in minutes.dtype.names:
        out[name] = minutes[name]
    out["points"], out["pfn_pct"] = raw[:, 8], raw[:, 9]
    out["shift"], out["shift_std"], out["pfn_spread"] = raw[:, 10], raw[:, 11], raw[:, 12]
    out["dbot"] = raw[:, 13]
    return out


def dbot_missing(v: np.ndarray) -> np.ndarray:
    return ~np.isfinite(v) | np.isin(v, list(DBOT_FILL))


def pfn_missing(v: np.ndarray) -> np.ndarray:
    return ~np.isfinite(v) | (v == PFN_PCT_FILL)


def mag_valid(row) -> np.ndarray:
    return magnetic_valid(row)


def candidate_train_grid_starts(year: int) -> int:
    """Five-minute grid starts with four hours wholly in one training month.

    This calendar denominator includes starts lost to missing/invalid vectors or
    source transitions, and does not depend on which minute records are present.
    Training days form one contiguous interval per month under role_for_date.
    """
    total = 0
    for month in range(1, 13):
        days = sum(role_for_date(year, month, day) == ROLE_TRAIN
                   for day in range(1, calendar.monthrange(year, month)[1] + 1))
        total += max(0, days * (24 * 60 // N_MINUTES_PER_BIN) - N_BINS + 1)
    return total


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--omni-root", default=OMNI_DIR)
    args = p.parse_args(argv)
    out = refuse_existing(Path(args.output))
    train_minutes_total = 0
    train_valid = 0
    dbot_neg_minutes = 0
    dbot_known_minutes = 0
    pfn_pos_minutes = 0
    pfn_known_minutes = 0
    one_point_minutes = 0
    bin_dbot_any_neg = 0
    n_train_bins = 0
    n_target_bins = 0
    mag_union = []
    inc_union = []
    sources = {}
    years_starts = {}
    candidate_starts = {str(year): candidate_train_grid_starts(year) for year in YEARS}
    raw_inputs = {}
    for year in YEARS:
        path = omni_path(year, args.omni_root)
        digest = file_sha256(path)
        q = parse_quality_year(path)
        raw_inputs[str(year)] = {"path": str(path.resolve()), "sha256": digest, "n_minutes": len(q)}
        # training-role minutes
        days = q["ts"].astype("datetime64[D]")
        ymd = days.astype(object)
        # object datetime.date
        years_d = np.array([d.year for d in ymd], dtype=np.int32)
        months_d = np.array([d.month for d in ymd], dtype=np.int32)
        days_d = np.array([d.day for d in ymd], dtype=np.int32)
        tr = np.array(
            [role_for_date(int(y), int(m), int(d)) == ROLE_TRAIN for y, m, d in zip(years_d, months_d, days_d)]
        )
        train_minutes_total += int(tr.sum())
        v = mag_valid(q) & tr
        train_valid += int(v.sum())
        dbot = q["dbot"]
        known = v & ~dbot_missing(dbot)
        dbot_known_minutes += int(known.sum())
        dbot_neg_minutes += int((known & (dbot < 0)).sum())
        pfn = q["pfn_pct"]
        pfn_ok = v & ~pfn_missing(pfn)
        pfn_known_minutes += int(pfn_ok.sum())
        pfn_pos_minutes += int((pfn_ok & (pfn > 0)).sum())
        pts = q["points"]
        one_point_minutes += int((v & np.isfinite(pts) & (pts <= 1)).sum())
        bins = form_bins(q)
        train_bins = [b for b in bins if b.role == ROLE_TRAIN]
        runs = consecutive_runs(train_bins)
        eligible_left = set()
        target_left = set()
        n_starts = 0
        for run in runs:
            wins = windows_from_run(run, stride=1)
            n_starts += len(wins)
            if not wins:
                continue
            for b in run:
                eligible_left.add((str(b.left), b.source))
            for w in wins:
                # hidden bins 18-29 of each window
                for k in range(HIDDEN_START, HIDDEN_END):
                    left = w.start_left + np.timedelta64(5 * k, "m")
                    target_left.add((str(left), w.source))
        years_starts[str(year)] = n_starts
        # join quality onto eligible bins via timestamp
        qmap = {str(ts): i for i, ts in enumerate(q["ts"])}
        for left, src in eligible_left:
            # five minutes in the bin
            t0 = np.datetime64(left)
            idx = []
            ok = True
            for dt in range(5):
                key = str(t0 + np.timedelta64(dt, "m"))
                if key not in qmap:
                    ok = False
                    break
                idx.append(qmap[key])
            if not ok:
                raise ValueError("eligible bin is missing its original quality records")
            n_train_bins += 1
            sl = q[idx]
            dbot_sl = sl["dbot"]
            known_sl = ~dbot_missing(dbot_sl)
            if np.any(known_sl & (dbot_sl < 0)):
                bin_dbot_any_neg += 1
            B = np.stack((sl["bx"], sl["by"], sl["bz"]), axis=1)
            mag = np.linalg.norm(B, axis=1)
            mag_union.append(float(np.mean(mag)))
            inc = np.linalg.norm(np.diff(B, axis=0), axis=1)
            inc_union.extend(inc.tolist())
            sources[int(src)] = sources.get(int(src), 0) + 1
        n_target_bins += len(target_left)
        print(f"{year} train_starts={n_starts}", flush=True)
    mag_union = np.array(mag_union, dtype=np.float64)
    inc_union = np.array(inc_union, dtype=np.float64)
    rec = {
        "train_role_minutes": train_minutes_total,
        "train_valid_magnetic_minutes": train_valid,
        "eligible_bins_scanned": n_train_bins,
        "distinct_potential_target_bins": n_target_bins,
        "dbot_known_minutes": dbot_known_minutes,
        "dbot_negative_minutes": dbot_neg_minutes,
        "dbot_negative_minute_frac": (dbot_neg_minutes / dbot_known_minutes) if dbot_known_minutes else None,
        "bins_with_any_negative_dbot": bin_dbot_any_neg,
        "bins_with_any_negative_dbot_frac": (bin_dbot_any_neg / n_train_bins) if n_train_bins else None,
        "pfn_known_minutes": pfn_known_minutes,
        "pfn_missing_minutes": train_valid - pfn_known_minutes,
        "pfn_positive_minutes": pfn_pos_minutes,
        "pfn_positive_frac_of_valid": (pfn_pos_minutes / train_valid) if train_valid else None,
        "pfn_positive_frac_of_known": (pfn_pos_minutes / pfn_known_minutes) if pfn_known_minutes else None,
        "one_point_minutes": one_point_minutes,
        "one_point_frac_of_valid": (one_point_minutes / train_valid) if train_valid else None,
        "vector_magnitude_median_nT": float(np.median(mag_union)) if mag_union.size else None,
        "vector_magnitude_p999_nT": float(np.quantile(mag_union, 0.999)) if mag_union.size else None,
        "increment_p999_nT": float(np.quantile(inc_union, 0.999)) if inc_union.size else None,
        "source_bin_counts": {str(k): v for k, v in sorted(sources.items())},
        "train_starts_by_year": years_starts,
        "candidate_train_grid_starts_by_year": candidate_starts,
        "candidate_train_grid_starts": sum(candidate_starts.values()),
        "eligible_train_starts": sum(years_starts.values()),
        "eligible_start_frac_of_candidate_grid": (sum(years_starts.values()) / sum(candidate_starts.values())) if sum(candidate_starts.values()) else None,
        "vector_magnitude_sample_count": len(mag_union),
        "increment_sample_count": len(inc_union),
        "raw_inputs": raw_inputs,
        "population_definitions": {
            "train_role_minutes": "Observed records assigned TRAIN by calendar date, before magnetic validity; not the eligible-window union or a complete calendar grid.",
            "train_valid_magnetic_minutes": "Training-role records with documented non-fill source and finite non-fill vector components; no amplitude cut.",
            "minute_metadata": "DBOT, PFN and point-count minute statistics use all valid training-role magnetic records, not only the eligible-window union.",
            "eligible_bins_scanned": "Distinct five-minute bins in the union of eligible stride-one four-hour training windows.",
            "distinct_potential_target_bins": "Distinct bins appearing at hidden positions in at least one eligible training window; not all eligible bins.",
            "dbot_negative_minute_frac": "dbot_negative_minutes / dbot_known_minutes; exact fills 999999 and 9999999 and nonfinite DBOT are missing.",
            "bins_with_any_negative_dbot_frac": "bins_with_any_negative_dbot / eligible_bins_scanned, including bins with missing DBOT in the denominator.",
            "pfn_positive_frac_of_valid": "pfn_positive_minutes / train_valid_magnetic_minutes, including missing PFN in the denominator; PFN fill 999 and nonfinite values never count as positive.",
            "pfn_positive_frac_of_known": "pfn_positive_minutes / pfn_known_minutes, excluding PFN fill 999 and nonfinite values.",
            "one_point_frac_of_valid": "Finite point counts <=1 / train_valid_magnetic_minutes; the existing counter includes zero, not only exactly one.",
            "vector_magnitude": "Quantiles of per-bin mean minute-vector magnitude over distinct eligible bins, one value per bin; not pooled minute magnitudes.",
            "increment": "Quantile of four consecutive one-minute vector increments within each eligible five-minute bin; excludes between-bin increments.",
            "candidate_train_grid_starts": "All UTC five-minute starts whose complete four hours lie in one TRAIN calendar month, before magnetic validity, missing-record or source checks.",
            "eligible_start_frac_of_candidate_grid": "eligible_train_starts / candidate_train_grid_starts; overlapping starts, not independent trajectories.",
        },
        "note": "Read-only metadata scan. Metadata does not alter eligibility. DBOT/PFN are processing flags, not denoiser inputs or proof of invalid vectors.",
    }
    with out.open("x") as f:
        f.write(json.dumps(rec, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: rec[k] for k in rec if k not in ("source_bin_counts", "train_starts_by_year")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
