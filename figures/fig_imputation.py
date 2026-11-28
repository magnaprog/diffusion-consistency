"""Draw the recorded magnetic predictions for two missing hours.

At each coordinate and bin, the twentieth sorted draw gives the median and
draws 2 and 38 give a nominal 90% interval. These marginal medians can form
a curve absent from the 39 trajectories. The variance detail uses a common
scale for D, O-S and G5 in each row and a separate scale for Gaussian regression.
Its steps show the full five-minute support of each statistic, not interpolated
values or a continuous magnetic trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator, NullLocator
import numpy as np

METHODS = ("D", "O-S", "G5", "full_flank_gaussian")
LABELS = ("D", "O-S", "G5", "Gaussian\nregression")
CHANNELS = ("mu_x", "mu_y", "mu_z", "m", "v_s")
UNITS = ("nT", "nT", "nT", "nT", "nT^2")
# Platform and OMNI arrival date of the two fixed, caption-identified cases.
CASE_LABELS = {120: "ACE: August 3, 2010", 360: "Wind: August 4, 2020"}
TIME_LABEL = "Minutes from the start of the missing hour"
LEGEND_LABELS = ("Measurements outside gap", "Measurements in gap", "Predictive median", "Nominal 90% interval")
SELECTION = {"rule": "midpoints_of_two_equal_ordered_TEST_partitions",
             "zero_based_indices": [120, 360], "original_window_count": 480,
             "neural_seed": 7, "displayed_draws": [0, 1, 2], "after_test_exposure": True}
STYLE_PATH = Path(__file__).with_name("prigm.mplstyle")
STYLE_SHA256 = hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest()
RC = matplotlib.rc_params_from_file(STYLE_PATH, use_default_template=False)
if hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest() != STYLE_SHA256:
    raise ValueError("style changed during import")
RC["text.parse_math"] = False
# Paul Tol's vibrant blue, magenta, orange, and teal.
# Palette definitions: https://sronpersonalpages.nl/~pault/
CASE_COLORS = ("#0077BB", "#EE3377", "#EE7733", "#009988")
MEDIAN_WIDTH = 1.55
SAMPLE_PATH_WIDTH = 0.56
# Matplotlib scales dashes by line width. Specify the final lengths in print points.
SAMPLE_PATH_STYLE = (0, tuple(length / SAMPLE_PATH_WIDTH for length in (1.98, 0.54, 0.72, 0.54)))
INTERVAL_ALPHA = 0.28
CASE_FORECAST_LIMITS = (-5, 65)
VS_LINTHRESH = 0.05
VS_TICKS = (-1.0, -0.1, 0.0, 0.1, 1.0)
Y_SCALES = (
    {"scale": "linear"},
    {"scale": "linear"},
    {"scale": "linear"},
    {"scale": "linear"},
    {"scale": "symlog", "linthresh": VS_LINTHRESH, "ticks": list(VS_TICKS)},
)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def _json_float(value):
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("nonfinite JSON number")
    return number


def _same(actual, expected):
    """Equality with JSON types preserved (True must not pass as 1)."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_same(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_same(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _hashed_bytes(path, expected):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("input SHA256 mismatch")
    return payload


def _validate(data):
    for key, shape in (("truth", (2, 48, 6)), ("samples", (4, 2, 39, 12, 5))):
        array = data[key]
        if array.dtype != np.dtype("float64") or array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"{key} requires finite float64 {shape}")
    ids, methods = data["ids"], data["methods"]
    if (ids.dtype.kind != "U" or ids.shape != (2,) or len(set(ids.tolist())) != 2
            or any(not value.strip() for value in ids)):
        raise ValueError("requires two distinct nonempty Unicode case IDs")
    if methods.dtype.kind != "U" or methods.shape != (4,) or methods.tolist() != list(METHODS):
        raise ValueError("ordered methods differ")
    record = data["provenance"]
    if (not isinstance(record, dict) or not _same(record.get("schema_version"), 1)
            or record.get("analysis") != "frozen_raw_imputation_examples"
            or not _same(record.get("selection"), SELECTION)
            or not _same(record.get("case_ids"), ids.tolist())
            or not _same(record.get("method_order"), list(METHODS))
            or not _same(record.get("channels"), list(CHANNELS))
            or not _same(record.get("units"), list(UNITS))
            or not isinstance(record.get("methods"), dict)
            or set(record["methods"]) != set(METHODS)
            or any(not isinstance(value, dict) for value in record["methods"].values())
            or not _same(record.get("files"), {"sample-excerpt.npz": data["excerpt_sha256"]})):
        raise ValueError("excerpt provenance or fixed selection differs")


def load_cases(excerpt, excerpt_sha256, provenance, provenance_sha256):
    """Return validated arrays plus provenance and hashes; paths accept str or Path."""
    payload = _hashed_bytes(excerpt, excerpt_sha256)
    record = json.loads(_hashed_bytes(provenance, provenance_sha256),
                        object_pairs_hook=_unique, parse_constant=_constant, parse_float=_json_float)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if len(archive.files) != 4 or set(archive.files) != {"truth", "samples", "ids", "methods"}:
            raise ValueError("excerpt keys differ")
        data = {key: archive[key] for key in archive.files}
    data.update(provenance=record, excerpt_sha256=excerpt_sha256,
                provenance_sha256=provenance_sha256)
    _validate(data)
    return data


def _coordinates(data, case_index):
    _validate(data)
    if type(case_index) is not int or case_index not in (120, 360):
        raise ValueError("case_index must be the original TEST index 120 or 360")
    slot = (120, 360).index(case_index)
    x = np.arange(48) * 5 - 90
    truth = data["truth"][slot, :, :5]
    samples = data["samples"][:, slot]
    ordered = np.sort(samples, axis=1)
    return {"time_bin_start_minutes": x.tolist(), "gap_minutes": [0, 60],
            "context_intervals_half_open_minutes": [[-90, 0], [60, 150]],
            "truth_channel_indices": [0, 1, 2, 3, 4],
            "observed_left": {"x": x[:18].tolist(), "y": truth[:18].tolist()},
            "withheld_truth": {"x": x[18:30].tolist(), "y": truth[18:30].tolist()},
            "observed_right": {"x": x[30:].tolist(), "y": truth[30:].tolist()},
            "ranks": {"draw_axis_in_excerpt": 2, "draw_count": 39, "zero_based_low": 1,
                      "zero_based_median": 19, "zero_based_high": 37,
                      "displayed_draws": [0, 1, 2], "interpolation": "none"},
            "methods": {name: {"x": x[18:30].tolist(),
                               "low": ordered[i, 1].tolist(), "median": ordered[i, 19].tolist(),
                               "high": ordered[i, 37].tolist(), "draws": samples[i, :3].tolist()}
                        for i, name in enumerate(METHODS)}}


class MethodKey(HandlerBase):
    """Show the four method swatches in reading order in two rows."""

    def create_artists(self, legend, handles, xdescent, ydescent, width, height, fontsize, trans):
        column_gap, row_gap = 0.28 * fontsize, 0.22 * fontsize
        swatch_width = (width - column_gap) / 2
        swatch_height = (height - row_gap) / 2
        artists = []
        for index, handle in enumerate(handles):
            row, column = divmod(index, 2)
            x = -xdescent + column * (swatch_width + column_gap)
            y = -ydescent + (1 - row) * (swatch_height + row_gap)
            if isinstance(handle, Line2D):
                center = y + swatch_height / 2
                artist = Line2D([x, x + swatch_width], [center, center])
            else:
                artist = Rectangle((x, y), swatch_width, swatch_height)
            self.update_prop(artist, handle, legend)
            artist.set_transform(trans)
            artists.append(artist)
        return artists


def add_forecast_legend(fig, *, observation_width, include_paths):
    def lines(width, linestyle="-"):
        return tuple(Line2D([], [], color=color, lw=width, linestyle=linestyle, solid_capstyle="butt")
                     for color in CASE_COLORS)

    handles = [Line2D([], [], color="black", lw=observation_width),
               Line2D([], [], color="black", lw=observation_width, ls="--"),
               lines(MEDIAN_WIDTH),
               tuple(Patch(facecolor=color, alpha=INTERVAL_ALPHA) for color in CASE_COLORS)]
    labels = list(LEGEND_LABELS)
    if include_paths:
        handles.append(lines(SAMPLE_PATH_WIDTH, SAMPLE_PATH_STYLE))
        labels.append("Ensemble members 1\u20133")
    fig.legend(handles, labels, handler_map={tuple: MethodKey()}, handlelength=3, handleheight=1.05,
               loc="lower center", bbox_to_anchor=(0.5, 0.008 if include_paths else 0.007),
               ncol=3 if include_paths else 2, columnspacing=1.2 if include_paths else 2,
               frameon=False, fontsize=7)


def _plot_case_inputs(ax, coords, row, limits):
    """Draw the supplied measurements on either side as independent step paths."""
    for key in ("observed_left", "observed_right"):
        part = coords[key]
        starts = np.asarray(part["x"])
        visible = (starts >= limits[0]) & (starts < limits[1])
        x = starts[visible]
        edges = np.append(x, x[-1] + 5)
        ax.stairs(np.asarray(part["y"])[visible, row], edges, baseline=None,
                  color="black", linewidth=0.85, gid=key)


def plot_case(data, case_index):
    """Show the full context and adjacent input bins beside each missing-hour forecast."""
    coords = _coordinates(data, case_index)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(5, 5, figsize=(5.225, 7.125), sharey="row",
                                 gridspec_kw={"width_ratios": [1.15, 1, 1, 1, 1]})
        fig.subplots_adjust(left=0.115, right=0.987, top=0.92, bottom=0.135,
                            wspace=0.19, hspace=0.19)
        fig.suptitle(CASE_LABELS[case_index], fontsize=9, y=0.984)
        forecast_center = (axes[0, 1].get_position().x0 + axes[0, -1].get_position().x1) / 2
        fig.text(forecast_center, 0.95, "Forecasts for the missing hour", ha="center", fontsize=8)
        for ax, label in zip(axes[0], ("Context", *LABELS)):
            ax.set_title(label, fontsize=8, pad=3)
        symbols = (r"$\mu_x$", r"$\mu_y$", r"$\mu_z$", "$m$", "$v_s$")
        for row, (context, *forecasts) in enumerate(axes):
            unit = r"nT$^2$" if row == 4 else "nT"
            context.set_ylabel(f"{symbols[row]} ({unit})", labelpad=1, parse_math=True)
            context.axvspan(*coords["gap_minutes"], color="0.92", linewidth=0, zorder=0, gid="gap")
            _plot_case_inputs(context, coords, row, (-90, 150))
            context.set(xlim=(-90, 150), xticks=[-90, 0, 60, 150])
            for ax, name, color in zip(forecasts, METHODS, CASE_COLORS):
                ax.axvspan(*coords["gap_minutes"], color="0.5", alpha=0.08,
                           linewidth=0, zorder=0, gid="gap")
                _plot_case_inputs(ax, coords, row, CASE_FORECAST_LIMITS)
                pred = coords["methods"][name]
                edges, low = _bin_steps(pred["x"], np.asarray(pred["low"])[:, row])
                _, high = _bin_steps(pred["x"], np.asarray(pred["high"])[:, row])
                ax.fill_between(edges, low, high, step="post", facecolor=color, alpha=INTERVAL_ALPHA,
                                linewidth=0, gid="nominal90")
                for i, draw in enumerate(np.asarray(pred["draws"])):
                    ax.stairs(draw[:, row], edges, baseline=None, color=color,
                              linewidth=SAMPLE_PATH_WIDTH, linestyle=SAMPLE_PATH_STYLE, gid=f"draw_{i}")
                ax.stairs(np.asarray(pred["median"])[:, row], edges, baseline=None,
                          color=color, linewidth=MEDIAN_WIDTH, gid="median")
                ax.stairs(np.asarray(coords["withheld_truth"]["y"])[:, row], edges,
                          baseline=None, color="black", linestyle="--", linewidth=0.85,
                          gid="withheld_truth")
                ax.set(xlim=CASE_FORECAST_LIMITS, xticks=[0, 30, 60])
            for ax in axes[row]:
                ax.margins(y=0.08)
                ax.tick_params(labelsize=6.8, pad=1.8, length=2, labelbottom=row == 4)
            if row == 4:
                context.set_yscale("symlog", linthresh=VS_LINTHRESH)
                context.set_yticks(VS_TICKS, ["\u22121", "\u22120.1", "0", "0.1", "1"])
                context.yaxis.set_minor_locator(NullLocator())
            else:
                context.yaxis.set_major_locator(MaxNLocator(nbins=3))
        fig.text(0.52, 0.078, TIME_LABEL, ha="center", fontsize=8)
        add_forecast_legend(fig, observation_width=0.85, include_paths=True)
        fig.canvas.draw()
    return fig


def detail_coordinates(data):
    """Linear detail with common neural limits and separate Gaussian limits."""
    cases = []
    for slot, index in enumerate((120, 360)):
        coords = _coordinates(data, index)
        truth = data["truth"][slot, 16:32, 4]
        limits = {}
        for group in (METHODS[:3], METHODS[3:]):
            bounds = [truth]
            for name in group:
                bounds.extend(np.asarray(coords["methods"][name][key])[:, 4]
                              for key in ("low", "high"))
            values = np.concatenate(bounds)
            lower, upper = float(values.min()), float(values.max())
            padding = 0.08 * max(upper - lower, 1e-6)
            for name in group:
                limits[name] = [lower - padding, upper + padding]
        gaussian_variance = data["samples"][3, slot, :, :, 4]
        cases.append({"case_index": index, "coordinates": coords,
                      "gaussian_variance": {"negative": int(np.count_nonzero(gaussian_variance < 0)),
                                            "total": gaussian_variance.size},
                      "x_limits": [-10, 70], "y_limits": limits})
    return cases


def _bin_steps(x, y):
    """Extend bin-start values to the right edge for a post-step display."""
    return np.append(x, x[-1] + 5), np.append(y, y[-1])


def plot_detail(data):
    """Show full bin supports; retain separate Gaussian limits and raw values."""
    cases = detail_coordinates(data)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 4, figsize=(5.5, 3.8), sharex=True)
        fig.subplots_adjust(left=0.13, right=0.92, top=0.78, bottom=0.27,
                            hspace=0.32, wspace=0.17)
        for row, case in enumerate(cases):
            coords = case["coordinates"]
            for col, name in enumerate(METHODS):
                ax = axes[row, col]
                pred = coords["methods"][name]
                x = np.asarray(pred["x"])
                ax.axvspan(0, 60, color="0.5", alpha=0.08, linewidth=0, gid="gap")
                ax.axhline(0, color="0.6", linewidth=0.5, zorder=1.5, gid="zero")
                edges, low = _bin_steps(x, np.asarray(pred["low"])[:, 4])
                _, high = _bin_steps(x, np.asarray(pred["high"])[:, 4])
                ax.fill_between(edges, low, high, step="post", facecolor=CASE_COLORS[col], alpha=INTERVAL_ALPHA,
                                linewidth=0, gid="nominal90")
                ax.plot(*_bin_steps(x, np.asarray(pred["median"])[:, 4]),
                        drawstyle="steps-post", color=CASE_COLORS[col], linewidth=MEDIAN_WIDTH, gid="median")
                for key in ("observed_left", "withheld_truth", "observed_right"):
                    part = coords[key]
                    xx = np.asarray(part["x"])
                    keep = (xx >= -10) & (xx <= 65)
                    ax.plot(*_bin_steps(xx[keep], np.asarray(part["y"])[keep, 4]),
                            drawstyle="steps-post", color="black",
                            linestyle="--" if key == "withheld_truth" else "-",
                            linewidth=0.9, gid=key)
                ax.set(xlim=case["x_limits"], ylim=case["y_limits"][name], xticks=[0, 30, 60])
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.spines[["top", "right"]].set_visible(False)
                if row == 0:
                    if col == 3:
                        ax.set_title("Gaussian regression", pad=11)
                    else:
                        ax.set_title(LABELS[col], pad=8)
                if col == 0:
                    # Separate the full, rotated dates without moving the plots.
                    ax.set_ylabel(CASE_LABELS[case["case_index"]] + "\n" + r"$v_s$ (nT$^2$)",
                                  y=(0.6, 0.4)[row], parse_math=True)
                elif col == 3:
                    counts = case["gaussian_variance"]
                    ax.text(0.5, 1.012, f"{counts['negative']}/{counts['total']} below zero",
                            ha="center", va="bottom", fontsize=7, transform=ax.transAxes,
                            gid="negative_variance_count")
                    ax.yaxis.tick_right()
                    ax.spines["left"].set_visible(False)
                    ax.spines["right"].set_visible(True)
                else:
                    ax.tick_params(axis="y", labelleft=False)
        fig.suptitle("Variance of field magnitude", fontsize=8, y=0.99)
        for columns, guide in ((axes[0, :3], "Left axis (first three columns)"),
                               (axes[0, 3:], "Right axis")):
            center = (columns[0].get_position().x0 + columns[-1].get_position().x1) / 2
            fig.text(center, 0.89, guide, ha="center", fontsize=7)
        fig.supxlabel(TIME_LABEL, y=0.145, fontsize=8)
        add_forecast_legend(fig, observation_width=0.9, include_paths=False)
        fig.canvas.draw()
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excerpt", required=True, type=Path)
    parser.add_argument("--excerpt-sha256", required=True)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--provenance-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--case-index", type=int, choices=(120, 360))
    mode.add_argument("--detail", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest() != STYLE_SHA256:
        raise ValueError("style changed since import")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError("figure output directory must be fresh")
    font_path = font_manager.findfont(
        font_manager.FontProperties(family=RC["font.family"]), fallback_to_default=False)
    if Path(font_path).suffix.lower() != ".ttf":
        raise ValueError("DejaVu Serif TrueType font required; see docs/prigm_release.md")
    source = Path(__file__).resolve()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    style = source.with_name("prigm.mplstyle")
    style_hash = STYLE_SHA256
    data = load_cases(args.excerpt, args.excerpt_sha256, args.provenance, args.provenance_sha256)
    if args.detail:
        manifest = {"schema_version": 2, "figure": "fig_imputation",
                    "source": {"path": "fig_imputation.py", "sha256": source_hash},
                    "style": {"path": style.name, "sha256": style_hash},
                    "inputs": {"excerpt": {"path": "evidence/imputation/sample-excerpt.npz",
                                           "sha256": args.excerpt_sha256},
                               "provenance": {"path": "evidence/imputation/MANIFEST.json",
                                              "sha256": args.provenance_sha256}},
                    "cases": detail_coordinates(data), "case_labels": CASE_LABELS, "y_scale": "linear",
                    "limit_rule": "truth bins 16:32 and zero-based sorted indices 1 and 37 (one-based ranks 2 and 38); neural methods share limits, Gaussian separate; padding 8% of range, floor 1e-6",
                    "axis_sides": {"neural": "left", "gaussian": "right"},
                    "bin_display": {"style": "steps-post", "width_minutes": 5,
                                    "support": "left-closed, right-open",
                                    "interpolation": "none; repeated terminal value closes final bin"},
                    "matplotlib_version": matplotlib.__version__, "output_sha256": {}}
        fig = plot_detail(data)
        try:
            args.output_dir.mkdir()
            with plt.rc_context(RC):
                for suffix in ("pdf", "png"):
                    path = args.output_dir / f"fig_imputation.{suffix}"
                    with path.open("xb") as handle:
                        fig.savefig(handle, format=suffix, dpi=200,
                                    metadata={"CreationDate": None} if suffix == "pdf" else None)
                    manifest["output_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            _hashed_bytes(source, source_hash)
            _hashed_bytes(style, style_hash)
            _hashed_bytes(args.excerpt, args.excerpt_sha256)
            _hashed_bytes(args.provenance, args.provenance_sha256)
            with (args.output_dir / "fig_imputation.manifest.json").open("x") as handle:
                json.dump(manifest, handle, indent=2, allow_nan=False)
                handle.write("\n")
        finally:
            plt.close(fig)
        return 0
    manifest = {"schema_version": 2, "figure": "fig_imputation",
                "source": {"path": "fig_imputation.py", "sha256": source_hash},
                "style": {"path": style.name, "sha256": style_hash},
                "inputs": {"excerpt": {"path": "evidence/imputation/sample-excerpt.npz",
                                       "sha256": args.excerpt_sha256},
                           "provenance": {"path": "evidence/imputation/MANIFEST.json",
                                          "sha256": args.provenance_sha256}},
                "case_index": args.case_index, "case_id": str(data["ids"][(120, 360).index(args.case_index)]),
                "case_label": CASE_LABELS[args.case_index],
                "columns": ["context", *METHODS],
                "x_limits_minutes": {"context": [-90, 150], "forecast": list(CASE_FORECAST_LIMITS)},
                "bin_display": {"style": "steps-post", "width_minutes": 5,
                                "support": "left-closed, right-open"},
                "channels": list(CHANNELS), "units": list(UNITS), "method_order": list(METHODS),
                "selection": data["provenance"]["selection"],
                "coordinates": _coordinates(data, args.case_index),
                "scope": "Raw draws, no pooled seeds, smoothing, sampling or clipping. Nominal marginal bands are not a calibration claim. Examples selected after TEST exposure.",
                "matplotlib_version": matplotlib.__version__, "output_sha256": {}}
    fig = plot_case(data, args.case_index)
    try:
        args.output_dir.mkdir()
        with plt.rc_context(RC):
            for suffix in ("pdf", "png"):
                path = args.output_dir / f"fig_imputation.{suffix}"
                with path.open("xb") as handle:
                    fig.savefig(handle, format=suffix, dpi=200,
                                    metadata={"CreationDate": None} if suffix == "pdf" else None)
                manifest["output_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        _hashed_bytes(source, source_hash)
        _hashed_bytes(style, style_hash)
        _hashed_bytes(args.excerpt, args.excerpt_sha256)
        _hashed_bytes(args.provenance, args.provenance_sha256)
        with plt.rc_context(RC):
            fig.canvas.draw()
        manifest["y_limits"] = [list(fig.axes[row * 5].get_ylim()) for row in range(5)]
        manifest["y_scale"] = [dict(spec) for spec in Y_SCALES]
        for row, expected in enumerate(Y_SCALES):
            axis = fig.axes[row * 5]
            if axis.get_yscale() != expected["scale"]:
                raise ValueError("recorded y scale differs from plotted artists")
            if expected["scale"] == "symlog":
                if float(axis.yaxis.get_transform().linthresh) != expected["linthresh"]:
                    raise ValueError("recorded v_s linthresh differs from plotted artists")
                if tuple(axis.get_yticks()) != tuple(expected["ticks"]):
                    raise ValueError("recorded v_s ticks differ from plotted artists")
                for label in axis.get_yticklabels():
                    text = label.get_text()
                    if "$" in text or r"\mathdefault" in text or not text:
                        raise ValueError("v_s tick labels must be plain text")
                    if label.get_parse_math():
                        raise ValueError("v_s tick labels must not parse math")
        with (args.output_dir / "fig_imputation.manifest.json").open("x") as handle:
            json.dump(manifest, handle, indent=2, allow_nan=False)
            handle.write("\n")
    finally:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
