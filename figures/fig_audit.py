"""Plot hash-pinned schema-2 paired comparisons into a new output directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Protocol counts, not plotted scores; prigm.config.BOOTSTRAP_REPLICATES.
BOOTSTRAP_REPLICATES = 10_000
METHOD = "O-S_raw"
BASELINES = (
    ("D_raw", "D (raw)"),
    ("D_corrected", "D (corrected)"),
    ("G5_raw", "G5 (raw)"),
    ("G5_corrected", "G5 (corrected)"),
    ("gaussian_raw", "Gaussian (raw)"),
    ("interpolation_raw", "Interpolation (raw)"),
    ("persistence_raw", "Persistence (raw)"),
)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _json_float(value):
    return _finite(float(value), "JSON number")


def _sha256(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("expected a lowercase hexadecimal SHA256 digest")
    return value


def load_comparison(path, expected_hash):
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != _sha256(expected_hash):
        raise ValueError(f"comparison SHA256 mismatch: {path}")
    record = json.loads(payload, object_pairs_hook=_unique_object,
                        parse_float=_json_float, parse_constant=_json_float)
    if (type(record.get("schema_version")) is not int or record["schema_version"] != 2
            or record.get("exposure") != "historical_test_exposed"):
        raise ValueError("requires schema_version=2 and exposure=historical_test_exposed")
    if type(record.get("n_replicates")) is not int or record["n_replicates"] != BOOTSTRAP_REPLICATES:
        raise ValueError(f"requires {BOOTSTRAP_REPLICATES} bootstrap replicates")
    comparisons = record["comparisons"]
    if not isinstance(comparisons, dict) or not comparisons:
        raise ValueError("comparisons must be a nonempty object")
    indexed, seen = {}, set()
    for name, row in comparisons.items():
        method, baseline = row["method"], row["baseline"]
        if not all(isinstance(v, str) and v for v in (method, baseline)):
            raise ValueError(f"invalid method/baseline: {name}")
        pair = frozenset((method, baseline))
        if method == baseline or pair in seen:
            raise ValueError(f"duplicate or ambiguous comparison: {name}")
        seen.add(pair)
        _finite(row["mean_diff"], f"{name}.mean_diff")
        for scheme in ("year_95", "twoyear_95"):
            interval = row[scheme]
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError(f"{name}.{scheme} requires two endpoints")
            lo, hi = [_finite(v, f"{name}.{scheme}") for v in interval]
            if lo > hi:
                raise ValueError(f"unordered interval: {name}.{scheme}")
        indexed[method, baseline] = (name, row)
    plotted = []
    for baseline, label in BASELINES:
        if (METHOD, baseline) not in indexed:
            raise ValueError(f"missing exact comparison: {METHOD} minus {baseline}")
        name, row = indexed[METHOD, baseline]
        if row["estimand"] != "validation_selected_procedures":
            raise ValueError(f"unexpected estimand: {name}")
        plotted.append({
            "comparison": name, "method": METHOD, "baseline": baseline,
            "label": label, "family": row["family"], "estimand": row["estimand"],
            **{key: row[key] for key in ("mean_diff", "year_95", "twoyear_95")},
        })
    return record, digest, plotted


def plot_comparisons(rows):
    fig, ax = plt.subplots(figsize=(5.5, 3.4), layout="constrained")
    ax.axvline(0, color="0.55", linestyle=":", linewidth=1, zorder=0)
    for y, row in enumerate(rows):
        for field, offset, style, label in (
            ("year_95", -0.11, "-", "Year-block 95%"),
            ("twoyear_95", 0.11, "--", "Two-year-block 95%"),
        ):
            # Draw endpoints directly: a percentile CI need not contain the point.
            ax.plot(row[field], [y + offset] * 2, linestyle=style, marker="|",
                    markersize=6, linewidth=1.5, color="0.2",
                    label=label if y == 0 else None)
        ax.plot(row["mean_diff"], y, "D", markersize=4, color="black",
                label="Observed" if y == 0 else None, zorder=3)
    ax.set_yticks(range(len(rows)), [row["label"] for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("O-S (raw) minus baseline: primary energy score\n"
                  "Negative differences favor O-S")
    ax.set_title("Historically exposed test comparisons", fontsize=9)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.11), ncol=3,
              frameon=False, fontsize=8, handlelength=2.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.08, y=0.08)
    return fig


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--comparison-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="must not exist; its parent must exist")
    args = parser.parse_args(argv)
    try:
        out = args.output_dir.absolute()
        if out.exists() or out.is_symlink():
            raise FileExistsError(f"output directory must be fresh: {out}")
        source = Path(__file__).resolve()
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        comparison = args.comparison.resolve()
        record, digest, rows = load_comparison(comparison, args.comparison_sha256)
        manifest = {
            "schema_version": 1, "figure": "fig_audit",
            "source": {"path": source.name, "sha256": source_hash},
            "inputs": {"comparison": {"path": comparison.name, "sha256": digest}},
            "input_schema_version": record["schema_version"],
            "exposure": record["exposure"], "n_replicates": record["n_replicates"],
            "plotted_rows": rows,
            "interval_note": "Marginal paired test-year percentile intervals, not familywise corrected; fixed trained models, not refitting or selection uncertainty.",
            "matplotlib_version": matplotlib.__version__, "backend": "Agg",
            "output_sha256": {},
        }
        with plt.rc_context({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
                             "ytick.labelsize": 9, "axes.unicode_minus": False,
                             "text.usetex": False, "pdf.fonttype": 42}):
            fig = plot_comparisons(rows)
            try:
                out.mkdir()  # Atomic refusal even if another writer won the race.
                for suffix in ("pdf", "png"):
                    target = out / f"fig_audit.{suffix}"
                    with target.open("xb") as handle:
                        fig.savefig(handle, format=suffix, dpi=180)
                    manifest["output_sha256"][target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
            finally:
                plt.close(fig)
        if (hashlib.sha256(source.read_bytes()).hexdigest() != source_hash
                or hashlib.sha256(comparison.read_bytes()).hexdigest() != digest):
            raise ValueError("source or comparison changed while rendering; no manifest published")
        manifest_path = out / "fig_audit.manifest.json"
        with manifest_path.open("x") as handle:
            handle.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
        parser.error(str(exc))
    print(json.dumps({"manifest": str(manifest_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
