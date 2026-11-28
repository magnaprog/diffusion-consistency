"""Construct the prespecified 2025 evaluation inventory from modified OMNI ASCII."""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from prigm.arrays import window_ids, windows_to_physical
from prigm.config import ROLE_TEST
from prigm.data import consecutive_runs, form_bins, pack_nonoverlapping, parse_omni_year, select_eval_windows
from prigm.recovery import sha256_file


def prepare(source, output):
    digest = sha256_file(source)
    minutes = parse_omni_year(source)
    if set(minutes["year"]) != {2025}:
        raise ValueError("expected a file covering 2025")
    bins = [replace(b, role=ROLE_TEST) for b in form_bins(minutes)]
    pool = [w for run in consecutive_runs(bins) for w in pack_nonoverlapping(run)]
    selected = select_eval_windows(pool, ROLE_TEST)
    times = np.array([w.start_left for w in selected])
    ids = window_ids(selected)
    if len(ids) != len(set(ids)) or np.any(np.diff(times) < np.timedelta64(4, "h")):
        raise ValueError("duplicate or overlapping windows")
    if sha256_file(source) != digest:
        raise ValueError("input changed while selecting windows")
    with output.open("xb") as f:
        np.savez_compressed(f, z=windows_to_physical(selected), ids=np.array(ids))
    receipt = {"source_sha256": digest, "source_rows": len(minutes),
               "endpoint_minutes": [str(minutes["ts"][i]) for i in (0,-1)],
               "source_url": "https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/modified/omni_min2025.asc",
               "arrays_sha256": sha256_file(output), "selected": len(ids),
               "eligible_by_month": dict(Counter(w.month for w in pool)),
               "selected_by_month": dict(Counter(w.month for w in selected)),
               "ids": ids}
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)
