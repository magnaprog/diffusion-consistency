"""Plot raw coefficient responses from hash-bound fixed-final TEST evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

COEFFICIENTS = (0.0, 0.00005, 0.0005, 0.005)
LABELS = ("0", "0.00005", "0.0005", "0.005")
SEEDS = (7, 13, 42)
FINAL = 29636
PROTOCOL = "1408cd52f306af0095f808dfe1d34aa8493fb26db9a4e75dae422839157bc3b4"
SELECTION = "predetermined_final_update_29636_not_VAL_selected"
STYLE_PATH = Path(__file__).with_name("prigm.mplstyle")
STYLE_SHA256 = hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest()
STYLE = matplotlib.rc_params_from_file(STYLE_PATH, use_default_template=False)
if hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest() != STYLE_SHA256:
    raise ValueError("style changed during import")
COLORS = STYLE["axes.prop_cycle"].by_key()["color"]


def _finite(value):
    if type(value) not in (float, int):
        raise ValueError("expected a finite plotted number, not a boolean or string")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError("expected a finite plotted number") from exc
    if not math.isfinite(number):
        raise ValueError("expected a finite plotted number")
    return number


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def _json_float(value):
    return _finite(float(value))


def _ordered_rows(rows):
    """Validate the plotting contract and order by coefficient, then seed."""
    if not isinstance(rows, list) or len(rows) != 12:
        raise ValueError("requires the complete 12-row coefficient/seed grid")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"coefficient", "seed", "energy", "residual"}:
            raise ValueError("invalid plotted row fields")
        coefficient, seed = _finite(row["coefficient"]), row["seed"]
        if coefficient not in COEFFICIENTS or type(seed) is not int or seed not in SEEDS:
            raise ValueError("changed coefficient or seed in fixed grid")
        key = (coefficient, seed)
        if key in indexed:
            raise ValueError("duplicate coefficient/seed row")
        energy, residual = _finite(row["energy"]), _finite(row["residual"])
        if energy < 0 or residual < 0:
            raise ValueError("energy score and mean absolute residual must be nonnegative")
        indexed[key] = {"coefficient": coefficient, "seed": seed,
                        "energy": energy, "residual": residual}
    expected = [(coefficient, seed) for coefficient in COEFFICIENTS for seed in SEEDS]
    if set(indexed) != set(expected):
        raise ValueError("requires the complete coefficient/seed grid")
    return [indexed[key] for key in expected]


def load_rows(path, expected_sha256):
    """Return 12 coefficient/seed-ordered dicts: coefficient, seed, energy, residual.

    Energy and residual are the raw month-balanced es5 and r_obs_mean values.
    Means are not additional observations and are not returned by this loader.
    """
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("fixed-final evidence SHA256 mismatch")
    record = json.loads(payload, object_pairs_hook=_unique_object,
                        parse_constant=_reject_constant, parse_float=_json_float)
    try:
        if (type(record["schema_version"]) is not int or record["schema_version"] != 2
                or record["complete_grid"] is not True or record["protocol_sha256"] != PROTOCOL
                or record["selection"] != SELECTION or record["arrays"]["split"] != "TEST"
                or type(record["arrays"]["n_windows"]) is not int
                or record["arrays"]["n_windows"] != 480):
            raise ValueError("requires the complete fixed-final TEST evidence contract")
        expected_ids = {f"c{label}_s{seed}" for label in LABELS for seed in SEEDS}
        endpoints = record["endpoints"]
        if not isinstance(endpoints, dict) or set(endpoints) != expected_ids:
            raise ValueError("requires the complete fixed-final endpoint inventory")
        raw_rows = record["rows"]
        if not isinstance(raw_rows, list) or len(raw_rows) != 12:
            raise ValueError("requires the complete 12-row coefficient/seed grid")
        rows = []
        for row in raw_rows:
            coefficient, seed = _finite(row["c_r"]), row["seed"]
            if coefficient not in COEFFICIENTS or type(seed) is not int or seed not in SEEDS:
                raise ValueError("changed coefficient or seed in fixed grid")
            label = LABELS[COEFFICIENTS.index(coefficient)]
            identity = f"c{label}_s{seed}"
            if (row["id"] != identity or row["arm"] != ("D" if coefficient == 0 else "O-S")
                    or type(row["generation"]) is not int or row["generation"] != FINAL
                    or type(endpoints[identity]["generation"]) is not int
                    or endpoints[identity]["generation"] != FINAL):
                raise ValueError("changed fixed-final row identity or generation")
            summary = row["results"]["raw"]["summary"]
            rows.append({"coefficient": coefficient, "seed": seed,
                         "energy": summary["es5"]["month_balanced"],
                         "residual": summary["r_obs_mean"]["month_balanced"]})
    except (KeyError, TypeError) as exc:
        raise ValueError("missing or malformed fixed-final evidence fields") from exc
    return _ordered_rows(rows)


def _mean_rows(rows):
    return [{"coefficient": coefficient,
             **{metric: math.fsum(row[metric] / len(SEEDS) for row in rows
                                 if row["coefficient"] == coefficient)
                for metric in ("energy", "residual")}}
            for coefficient in COEFFICIENTS]


def plot(rows):
    """Return a two-panel Figure; compute arithmetic seed means internally."""
    rows = _ordered_rows(rows)
    means = _mean_rows(rows)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.7), layout="constrained")
        x = np.arange(len(COEFFICIENTS))
        styles = (("o", "-"), ("s", "--"), ("^", ":"))
        for ax, metric, title, ylabel in zip(
                axes, ("energy", "residual"), ("(a) Raw energy score", "(b) Identity residual"),
                ("Energy score (standardized)", r"Mean absolute residual (nT$^2$)")):
            for seed, (marker, linestyle) in zip(SEEDS, styles):
                values = [row[metric] for row in rows if row["seed"] == seed]
                ax.plot(x, values, color=COLORS[1], marker=marker, linestyle=linestyle,
                        markersize=3, linewidth=0.9, label=f"Seed {seed}")
            ax.plot(x, [row[metric] for row in means], color="black", marker="D",
                    linestyle="none", markersize=4, zorder=4, label="Mean across seeds")
            ax.set_xticks(x, LABELS)
            ax.set_xlabel(r"Penalty coefficient $c_R$")
            ax.set_ylabel(ylabel)
            ax.set_title(title, loc="left")
            ax.grid(axis="y")
            ax.margins(x=0.1, y=0.15)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(length=2, pad=2)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center",
                   ncol=4, frameon=False, fontsize=7, handlelength=1.8)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--evidence-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest() != STYLE_SHA256:
        raise ValueError("style changed since import")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError("figure output directory must be fresh")
    font_path = font_manager.findfont(
        font_manager.FontProperties(family=STYLE["font.family"]), fallback_to_default=False)
    if Path(font_path).suffix.lower() != ".ttf":
        raise ValueError("DejaVu Serif TrueType font required; see docs/prigm_release.md")
    source = Path(__file__).resolve()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    style = source.with_name("prigm.mplstyle")
    style_hash = STYLE_SHA256
    rows = load_rows(args.evidence, args.evidence_sha256)
    manifest = {
        "schema_version": 1, "figure": "fig_fixed_final",
        "source": {"path": source.name, "sha256": source_hash},
        "style": {"path": style.name, "sha256": style_hash},
        "inputs": {"fixed_final": {"path": "evidence/deadline/fixed-final.json",
                                    "sha256": args.evidence_sha256}},
        "plotted_rows": rows, "mean_rows": _mean_rows(rows),
        "scope": "Raw month-balanced TEST metrics at the predetermined final update 29636. "
                 "Historical TEST exposure; descriptive coefficient response, no coefficient selection. "
                 "Three fixed seed traces and their arithmetic mean, not uncertainty intervals. "
                 "Straight segments connect observed seed values; no fitted curve, uncertainty bands, "
                 "corrected results, or VAL-selected results.",
        "axes": {"x": {"scale": "categorical", "positions": [0, 1, 2, 3],
                       "coefficient_labels": list(LABELS), "coefficients": list(COEFFICIENTS)},
                 "a": {"y_scale": "linear", "metric": "results.raw.summary.es5.month_balanced",
                       "units": "TRAIN-standardized"},
                 "b": {"y_scale": "linear", "metric": "results.raw.summary.r_obs_mean.month_balanced",
                       "units": "nT^2", "quantity": "mean absolute observation residual"}},
        "matplotlib_version": matplotlib.__version__, "output_sha256": {}}
    with plt.rc_context(STYLE):
        fig = plot(rows)
        try:
            args.output_dir.mkdir()
            for suffix in ("pdf", "png"):
                path = args.output_dir / f"fig_fixed_final.{suffix}"
                with path.open("xb") as handle:
                    fig.savefig(handle, format=suffix, dpi=200)
                manifest["output_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        finally:
            plt.close(fig)
    if (hashlib.sha256(source.read_bytes()).hexdigest() != source_hash
            or hashlib.sha256(style.read_bytes()).hexdigest() != style_hash
            or hashlib.sha256(args.evidence.read_bytes()).hexdigest() != args.evidence_sha256):
        raise ValueError("figure source or input changed during rendering")
    with (args.output_dir / "fig_fixed_final.manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
