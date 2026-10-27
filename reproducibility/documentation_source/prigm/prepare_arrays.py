"""Write physical window arrays + normalizer into a fresh directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from prigm.build_inventory import refuse_existing, window_record
from prigm.config import OMNI_DIR, ROLE_TEST, ROLE_TRAIN, ROLE_VAL, YEARS
from prigm.data import (
    consecutive_runs,
    form_bins,
    omni_path,
    pack_nonoverlapping,
    parse_omni_year,
    select_eval_windows,
    windows_from_run,
)
from prigm.normalize import fit_normalizer
from prigm.arrays import window_ids


def collect(omni_root: Path):
    from prigm.data import file_sha256

    train_w, val_p, test_p = [], [], []
    hashes = {}
    train_bins = []
    seen = set()
    starts = 0
    for year in YEARS:
        path = omni_path(year, omni_root)
        hashes[str(year)] = file_sha256(path)
        minutes = parse_omni_year(path)
        bins = form_bins(minutes)
        n_starts = 0
        for run in consecutive_runs(bins):
            if run[0].role == ROLE_TRAIN:
                wins = windows_from_run(run, stride=1)
                n_starts += len(wins)
                train_w.extend(wins)
                if wins:
                    for b in run:
                        key = (str(b.left), b.source)
                        if key not in seen:
                            seen.add(key)
                            train_bins.append(b.z)
            elif run[0].role == ROLE_VAL:
                val_p.extend(pack_nonoverlapping(run))
            elif run[0].role == ROLE_TEST:
                test_p.extend(pack_nonoverlapping(run))
        starts += n_starts
        print(f"{year} starts={n_starts} sha={hashes[str(year)][:12]}", flush=True)
    return train_w, val_p, test_p, train_bins, hashes, starts


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--omni-root", default=OMNI_DIR)
    args = p.parse_args(argv)
    out = refuse_existing(Path(args.output))
    out.mkdir()
    train_w, val_p, test_p, train_bins, hashes, starts = collect(Path(args.omni_root))
    val_sel = select_eval_windows(val_p, ROLE_VAL)
    test_sel = select_eval_windows(test_p, ROLE_TEST)
    z_union = np.stack(train_bins, axis=0)
    nrm = fit_normalizer(z_union)
    np.savez(
        out / "normalizer.npz",
        mean=nrm.mean,
        scale=nrm.scale,
        b_ref_sq=np.array(nrm.b_ref_sq),
        n_bins=np.array(nrm.n_bins),
    )
    train_z = np.stack([w.z for w in train_w], axis=0)
    val_z = np.stack([w.z for w in val_sel], axis=0) if val_sel else np.zeros((0, 48, 6))
    test_z = np.stack([w.z for w in test_sel], axis=0) if test_sel else np.zeros((0, 48, 6))
    for split, values, windows in (
        ("train", train_z, train_w), ("val", val_z, val_sel), ("test", test_z, test_sel)
    ):
        ids = np.asarray(window_ids(windows))
        if len(set(ids.tolist())) != len(ids):
            raise ValueError(f"duplicate {split} window IDs")
        np.savez(
            out / f"{split}.npz", z=values, ids=ids, split=np.array(split.upper()),
            mean=nrm.mean, scale=nrm.scale, b_ref_sq=np.array(nrm.b_ref_sq),
            n_bins=np.array(nrm.n_bins),
        )
    (out / "val_windows.json").write_text(json.dumps([window_record(w) for w in val_sel], indent=2) + "\n")
    (out / "test_windows.json").write_text(json.dumps([window_record(w) for w in test_sel], indent=2) + "\n")
    summary = {
        "train_starts": starts,
        "train_windows_saved": int(train_z.shape[0]),
        "train_bin_union": int(z_union.shape[0]),
        "val_selected": len(val_sel),
        "test_selected": len(test_sel),
        "hashes": hashes,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in summary if k != "hashes"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
