"""Plot paired TEST score changes and absolute residuals across penalty coefficients at update 29636.

Check the input hash and the complete four-coefficient, three-seed grid before
plotting. Show each seed and the arithmetic mean across seeds; the
predetermined update is independent of validation selection.
"""
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
from matplotlib.lines import Line2D

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
        raise ValueError("penalty sensitivity evidence SHA256 mismatch")
    record = json.loads(payload, object_pairs_hook=_unique_object,
                        parse_constant=_reject_constant, parse_float=_json_float)
    try:
        if (type(record["schema_version"]) is not int or record["schema_version"] != 2
                or record["complete_grid"] is not True or record["protocol_sha256"] != PROTOCOL
                or record["selection"] != SELECTION or record["arrays"]["split"] != "TEST"
                or type(record["arrays"]["n_windows"]) is not int
                or record["arrays"]["n_windows"] != 480):
            raise ValueError("requires the complete penalty sensitivity TEST evidence contract")
        expected_ids = {f"c{label}_s{seed}" for label in LABELS for seed in SEEDS}
        endpoints = record["endpoints"]
        if not isinstance(endpoints, dict) or set(endpoints) != expected_ids:
            raise ValueError("requires the complete penalty sensitivity endpoint inventory")
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
                raise ValueError("changed penalty sensitivity row identity or generation")
            summary = row["results"]["raw"]["summary"]
            rows.append({"coefficient": coefficient, "seed": seed,
                         "energy": summary["es5"]["month_balanced"],
                         "residual": summary["r_obs_mean"]["month_balanced"]})
    except (KeyError, TypeError) as exc:
        raise ValueError("missing or malformed penalty sensitivity evidence fields") from exc
    return _ordered_rows(rows)


def _mean_rows(rows):
    means = []
    for coefficient in COEFFICIENTS:
        fits = [row for row in rows if row["coefficient"] == coefficient]
        metrics = ("energy", "residual", "energy_minus_D") if coefficient else ("energy", "residual")
        means.append({"coefficient": coefficient,
                      **{m: math.fsum(row[m] / len(SEEDS) for row in fits) for m in metrics}})
    return means


# Preserve the reviewed print layout and vector typography at the manuscript width.
PRINT_STYLE = {**matplotlib.rcParamsDefault, "font.family": STYLE["font.family"],
               "pdf.fonttype": STYLE["pdf.fonttype"], "font.size": 7, "axes.titlesize": 8,
               "xtick.labelsize": 6.7, "ytick.labelsize": 7, "axes.linewidth": .5,
               "axes.spines.right": False, "axes.spines.top": False, "axes.spines.left": False,
               "xtick.major.size": 2, "ytick.major.size": 0}


SEED_MARKERS = ("o", "s", "^")
SEED_KEY_COLOR = "#333333"
MEAN_COLOR = "#B07800"
MEAN_WIDTH = 1.7
# Paul Tol's qualitative colors: https://sronpersonalpages.nl/~pault/
SCORE_COLOR = "#0077BB"
RESIDUAL_COLOR = "#EE3377"


def seed_points(axis, y, values, color):
    average = math.fsum(value / len(values) for value in values)
    for x, offset, marker in zip(values, (.23, 0, -.23), SEED_MARKERS):
        axis.plot(x, y + offset, marker=marker, markersize=3.4, color=color,
                  markeredgewidth=.8, linestyle="none", zorder=3)
    axis.plot([average, average], [y - .4, y + .4], color=MEAN_COLOR,
              linewidth=MEAN_WIDTH, solid_capstyle="butt", zorder=2)


def fit_legend(fig, height):
    handles = [Line2D([], [], linestyle="none", marker=m, markersize=3.7,
                      color=SEED_KEY_COLOR, label=label)
               for m, label in zip(SEED_MARKERS, ("Seed 7", "Seed 13", "Seed 42"))]
    handles.append(Line2D([], [], marker="|", markersize=8, markeredgewidth=MEAN_WIDTH,
                          color=MEAN_COLOR, linestyle="none", label="Mean"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, height), ncol=4,
               frameon=False, handletextpad=.05, columnspacing=.86, borderaxespad=0)


def panel(fig, box, rows, labels, title, xlabel, ticks, limits):
    width, height = fig.get_size_inches()
    x, y, w, h = box
    axis = fig.add_axes((x / width, y / height, w / width, h / height))
    axis.set(yticks=rows, yticklabels=labels, ylim=(min(rows) - .55, max(rows) + .55),
             xlabel=xlabel, xticks=ticks, xlim=limits)
    axis.set_title(title, loc="center", pad=4)
    axis.tick_params(pad=2)
    for level in rows[1::2]:
        lower = axis.get_ylim()[0] if level == rows[-1] else level - .5
        axis.axhspan(lower, level + .5, facecolor="#f0f0f0", linewidth=0, zorder=-2)
    axis.grid(axis="x", color=".85", linewidth=.38, zorder=0)
    return axis


def paired_rows(rows):
    baseline = {row["seed"]: row["energy"] for row in rows if row["coefficient"] == 0}
    return [{**row, "energy_minus_D": row["energy"] - baseline[row["seed"]]}
            if row["coefficient"] else row.copy() for row in rows]


def plot(rows):
    rows = paired_rows(_ordered_rows(rows))
    fig = plt.figure(figsize=(5.5, 2.08))
    levels = (3, 2, 1, 0)
    a = panel(fig, (.83, .68, 1.92, 1.07), levels, ("D at 0", *LABELS[1:]),
              "(a) Raw energy score", "O-S minus D", [-.15, -.10, -.05, 0, .05], (-.201, .08))
    b = panel(fig, (3.31, .68, 1.91, 1.07), levels, [""] * 4,
              "(b) Averaging residual", "Mean absolute value (nT²)", [0, 1, 2, 3, 4], (0, 4.45))
    a.set_ylabel("Penalty coefficient")
    a.plot([0, 0], [a.get_ylim()[0], (levels[0] + levels[1]) / 2],
           color=".4", linewidth=.7, solid_capstyle="butt")
    for y, coefficient in zip(levels, COEFFICIENTS):
        fits = [row for row in rows if row["coefficient"] == coefficient]
        if coefficient:
            seed_points(a, y, [row["energy_minus_D"] for row in fits], SCORE_COLOR)
        seed_points(b, y, [row["residual"] for row in fits], RESIDUAL_COLOR)
    for ax in (a, b):
        for separator in (.5, 1.5, 2.5):
            ax.axhline(separator, color=".9", linewidth=.4)
    fit_legend(fig, .04)
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
        "schema_version": 3, "figure": "fig_penalty_sensitivity",
        "source": {"path": source.name, "sha256": source_hash},
        "style": {"path": style.name, "sha256": style_hash},
        "inputs": {"fixed_final": {"path": "evidence/deadline/penalty-sensitivity.json",
                                    "sha256": args.evidence_sha256}},
        "plotted_rows": paired_rows(rows), "mean_rows": _mean_rows(paired_rows(rows)),
        "scope": "Raw month-balanced TEST metrics at the predetermined final update 29636. "
                 "Historical TEST exposure; descriptive coefficient response, no coefficient selection. "
                 "Three fixed seeds and their arithmetic mean. Score contrasts pair fits with D at the same seed.",
        "axes": {"y": {"scale": "categorical", "positions": [3, 2, 1, 0],
                       "coefficient_labels": list(LABELS), "coefficients": list(COEFFICIENTS)},
                 "a": {"x_scale": "linear", "metric": "energy_minus_D", "units": "TRAIN-standardized",
                       "comparison_coefficients": list(COEFFICIENTS[1:])},
                 "b": {"x_scale": "linear", "metric": "results.raw.summary.r_obs_mean.month_balanced",
                       "units": "nT^2", "quantity": "mean absolute observation residual",
                       "marks": "three seed points and a vertical bar at their arithmetic mean"}},
        "matplotlib_version": matplotlib.__version__, "output_sha256": {}}
    with plt.rc_context(PRINT_STYLE):
        fig = plot(rows)
        try:
            args.output_dir.mkdir()
            for suffix in ("pdf", "png"):
                path = args.output_dir / f"fig_penalty_sensitivity.{suffix}"
                with path.open("xb") as handle:
                    fig.savefig(handle, format=suffix, dpi=200,
                                metadata={"CreationDate": None} if suffix == "pdf" else None)
                manifest["output_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        finally:
            plt.close(fig)
    if (hashlib.sha256(source.read_bytes()).hexdigest() != source_hash
            or hashlib.sha256(style.read_bytes()).hexdigest() != style_hash
            or hashlib.sha256(args.evidence.read_bytes()).hexdigest() != args.evidence_sha256):
        raise ValueError("figure source or input changed during rendering")
    with (args.output_dir / "fig_penalty_sensitivity.manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
