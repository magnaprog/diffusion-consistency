"""Materialize hashes, bins, windows, and selection manifests. Fresh output only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from prigm.config import (
    OMNI_DIR,
    ROLE_TEST,
    ROLE_TRAIN,
    ROLE_VAL,
)
from prigm.data import (
    consecutive_runs,
    file_sha256,
    form_bins,
    omni_path,
    pack_nonoverlapping,
    parse_omni_year,
    select_eval_windows,
    windows_from_run,
)
from prigm.normalize import fit_normalizer


def refuse_existing(path: Path) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refuse existing output {path}")
    return path


def scan_year(year: int, root: Path) -> dict:
    path = omni_path(year, root)
    digest = file_sha256(path)
    minutes = parse_omni_year(path)
    bins = form_bins(minutes)
    runs = consecutive_runs(bins)
    train_starts = 0
    packed_train = 0
    train_bins = []
    val_packed = []
    test_packed = []
    for run in runs:
        role = run[0].role
        if role == ROLE_TRAIN:
            wins = windows_from_run(run, stride=1)
            train_starts += len(wins)
            packed_train += len(pack_nonoverlapping(run))
            # bins in eligible windows: union of bins belonging to any window
            if wins:
                for b in run:
                    train_bins.append(b)
        elif role == ROLE_VAL:
            val_packed.extend(pack_nonoverlapping(run))
        elif role == ROLE_TEST:
            test_packed.extend(pack_nonoverlapping(run))
    return {
        "year": year,
        "sha256": digest,
        "n_minutes": int(minutes.size),
        "n_bins": len(bins),
        "n_runs": len(runs),
        "train_starts": train_starts,
        "packed_train": packed_train,
        "train_bin_rows": train_bins,
        "val_packed": val_packed,
        "test_packed": test_packed,
    }


def window_record(w) -> dict:
    return {
        "start": str(w.start_left),
        "source": w.source,
        "role": w.role,
        "year": w.year,
        "month": w.month,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build PriGM manifests into a fresh directory")
    p.add_argument("--output", required=True)
    p.add_argument("--omni-root", default=OMNI_DIR)
    p.add_argument("--years", default="2005-2024")
    args = p.parse_args(argv)
    out = refuse_existing(Path(args.output))
    lo, hi = (int(x) for x in args.years.split("-"))
    years = range(lo, hi + 1)
    out.mkdir(parents=False)
    summary = {
        "train_starts": 0,
        "packed_train": 0,
        "val_selected": 0,
        "test_selected": 0,
        "years": {},
    }
    all_train_z = []
    all_val = []
    all_test = []
    seen_train_left = set()
    for year in years:
        rec = scan_year(year, Path(args.omni_root))
        summary["train_starts"] += rec["train_starts"]
        summary["packed_train"] += rec["packed_train"]
        summary["years"][str(year)] = {
            "sha256": rec["sha256"],
            "n_minutes": rec["n_minutes"],
            "n_bins": rec["n_bins"],
            "train_starts": rec["train_starts"],
        }
        for b in rec["train_bin_rows"]:
            key = (str(b.left), b.source)
            if key not in seen_train_left:
                seen_train_left.add(key)
                all_train_z.append(b.z)
        all_val.extend(rec["val_packed"])
        all_test.extend(rec["test_packed"])
        print(
            f"{year} starts={rec['train_starts']} bins={rec['n_bins']} sha={rec['sha256'][:12]}",
            flush=True,
        )
    val_sel = select_eval_windows(all_val, ROLE_VAL)
    test_sel = select_eval_windows(all_test, ROLE_TEST)
    summary["val_selected"] = len(val_sel)
    summary["test_selected"] = len(test_sel)
    summary["train_bin_union"] = len(all_train_z)
    if all_train_z:
        z = np.stack(all_train_z, axis=0)
        nrm = fit_normalizer(z)
        np.savez(
            out / "normalizer.npz",
            mean=nrm.mean,
            scale=nrm.scale,
            b_ref_sq=np.array(nrm.b_ref_sq),
            n_bins=np.array(nrm.n_bins),
        )
    (out / "val_windows.json").write_text(
        json.dumps([window_record(w) for w in val_sel], indent=2) + "\n"
    )
    (out / "test_windows.json").write_text(
        json.dumps([window_record(w) for w in test_sel], indent=2) + "\n"
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in summary if k != "years"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
