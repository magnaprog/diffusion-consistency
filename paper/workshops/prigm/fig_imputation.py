"""Draw the recorded magnetic predictions for two missing hours.

At each coordinate and bin, the twentieth sorted draw gives the median and
draws 2 and 38 give a nominal 90% interval. These marginal medians can form
a curve absent from the 39 trajectories. The variance detail uses a common
scale for D, O-S and G5 in each row and a separate scale for Gaussian regression.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, NullFormatter
import numpy as np

METHODS = ("D", "O-S", "G5", "full_flank_gaussian")
LABELS = ("D", "O-S", "G5", "Gaussian\nregression")
CHANNELS = ("mu_x", "mu_y", "mu_z", "m", "v_s")
UNITS = ("nT", "nT", "nT", "nT", "nT^2")
SELECTION = {"rule": "midpoints_of_two_equal_ordered_TEST_partitions",
             "zero_based_indices": [120, 360], "original_window_count": 480,
             "neural_seed": 7, "displayed_draws": [0, 1, 2], "after_test_exposure": True}
STYLE_PATH = Path(__file__).with_name("prigm.mplstyle")
STYLE_SHA256 = hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest()
RC = matplotlib.rc_params_from_file(STYLE_PATH, use_default_template=False)
if hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest() != STYLE_SHA256:
    raise ValueError("style changed during import")
RC["text.parse_math"] = False
COLORS = RC["axes.prop_cycle"].by_key()["color"]
VS_LINTHRESH = 0.05
VS_TICKS = (-1.0, -0.1, 0.0, 0.1, 1.0)
Y_SCALES = (
    {"scale": "linear"},
    {"scale": "linear"},
    {"scale": "linear"},
    {"scale": "linear"},
    {"scale": "symlog", "linthresh": VS_LINTHRESH, "ticks": list(VS_TICKS)},
)


def _plain_signed_tick(value):
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if magnitude >= 1:
        return f"{sign}{magnitude:g}"
    return f"{sign}0.{int(round(10 * magnitude))}"


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


def plot_case(data, case_index):
    """Return a matplotlib Figure (5 rows x 4 columns); caller closes it."""
    coords = _coordinates(data, case_index)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(5, 4, figsize=(5.5, 7.5), sharex=True, sharey="row")
        for row in range(5):
            for col, name in enumerate(METHODS):
                ax = axes[row, col]
                values = coords["methods"][name]
                x = values["x"]
                ax.axvspan(0, 60, color="0.5", alpha=0.08, linewidth=0, gid="gap")
                ax.fill_between(x, np.asarray(values["low"])[:, row],
                                np.asarray(values["high"])[:, row], color=COLORS[col],
                                alpha=0.16, linewidth=0, gid="nominal90")
                for draw in range(3):
                    ax.plot(x, np.asarray(values["draws"])[draw, :, row], color=COLORS[col],
                            alpha=0.45, linewidth=0.55, gid=f"draw_{draw}")
                ax.plot(x, np.asarray(values["median"])[:, row], color=COLORS[col],
                        linewidth=1.1, gid="median")
                for key in ("observed_left", "observed_right", "withheld_truth"):
                    part = coords[key]
                    ax.plot(part["x"], np.asarray(part["y"])[:, row], color="black",
                            linestyle="--" if key == "withheld_truth" else "-",
                            linewidth=0.9, gid=key)
                ax.set_xlim(-90, 150)
                ax.set_xticks([-60, 0, 60, 120])
                if row == 4:
                    ax.set_yscale("symlog", linthresh=VS_LINTHRESH)
                    ax.set_yticks(VS_TICKS, [_plain_signed_tick(value) for value in VS_TICKS])
                    ax.yaxis.set_minor_formatter(NullFormatter())
                    for label in ax.get_yticklabels():
                        label.set_parse_math(False)
                else:
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.margins(y=0.08)
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(length=2, pad=2)
                if row == 0:
                    ax.set_title(LABELS[col])
                if col == 0:
                    symbol = (r"$\mu_x$", r"$\mu_y$", r"$\mu_z$", "$m$", "$v_s$")[row]
                    unit = r"nT$^2$" if row == 4 else "nT"
                    ax.set_ylabel(f"{symbol} ({unit})", parse_math=True)
        fig.suptitle(f"TEST window {case_index}", fontsize=8, y=0.988, parse_math=False)
        fig.supxlabel("Time from start of missing hour (min)", y=0.077, fontsize=8)
        handles = [Line2D([], [], color="black", lw=0.9, label="Available measurements"),
                   Line2D([], [], color="black", ls="--", lw=0.9, label="Withheld measurements"),
                   Line2D([], [], color="0.4", lw=1.1, label="Predictive median"),
                   Line2D([], [], color="0.4", lw=0.55, alpha=0.45, label="Three ensemble members"),
                   Patch(facecolor="0.4", alpha=0.16, label="Nominal 90% marginal interval")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.003),
                   ncol=2, fontsize=7, frameon=False, handlelength=1.6)
        fig.subplots_adjust(left=0.16, right=0.99, top=0.93, bottom=0.13, hspace=0.17, wspace=0.13)
        fig.canvas.draw()
    return fig


def detail_coordinates(data):
    """Linear detail with common neural limits and separate Gaussian limits."""
    cases = []
    for index in (120, 360):
        coords = _coordinates(data, index)
        truth = data["truth"][(120, 360).index(index), 16:32, 4]
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
        cases.append({"case_index": index, "coordinates": coords,
                      "x_limits": [-10, 65], "y_limits": limits})
    return cases


def plot_detail(data):
    """Two cases; Gaussian intervals use a separate, fully displayed right axis."""
    cases = detail_coordinates(data)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 4, figsize=(5.5, 3.8), sharex=True)
        for row, case in enumerate(cases):
            coords = case["coordinates"]
            for col, name in enumerate(METHODS):
                ax = axes[row, col]
                pred = coords["methods"][name]
                x = np.asarray(pred["x"])
                ax.axvspan(0, 60, color="0.5", alpha=0.08, linewidth=0, gid="gap")
                ax.axhline(0, color="0.6", linewidth=0.5, zorder=1, gid="zero")
                ax.fill_between(x, np.asarray(pred["low"])[:, 4],
                                np.asarray(pred["high"])[:, 4], color=COLORS[col],
                                alpha=0.16, linewidth=0, gid="nominal90")
                ax.plot(x, np.asarray(pred["median"])[:, 4], color=COLORS[col],
                        linewidth=1.1, gid="median")
                for key in ("observed_left", "withheld_truth", "observed_right"):
                    part = coords[key]
                    xx = np.asarray(part["x"])
                    keep = (xx >= -10) & (xx <= 65)
                    ax.plot(xx[keep], np.asarray(part["y"])[keep, 4], color="black",
                            linestyle="--" if key == "withheld_truth" else "-",
                            linewidth=0.9, gid=key)
                ax.set(xlim=case["x_limits"], ylim=case["y_limits"][name], xticks=[0, 30, 60])
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.spines[["top", "right"]].set_visible(False)
                if row == 0:
                    ax.set_title(LABELS[col], pad=8)
                if col == 0:
                    ax.set_ylabel(f"TEST {case['case_index']}\n" + r"$v_s$ (nT$^2$)",
                                  parse_math=True)
                elif col == 3:
                    ax.yaxis.tick_right()
                    ax.spines["left"].set_visible(False)
                    ax.spines["right"].set_visible(True)
                else:
                    ax.tick_params(axis="y", labelleft=False)
        fig.suptitle("Variance of magnetic field magnitude over five minutes", fontsize=8, y=0.99)
        fig.text(0.5, 0.89, "D, O-S and G5 share each row's scale; Gaussian uses the right axis.",
                 ha="center", fontsize=7)
        fig.supxlabel("Time from start of missing hour (min)", y=0.145, fontsize=8)
        handles = [Line2D([], [], color="black", lw=0.9, label="Available measurements"),
                   Line2D([], [], color="black", ls="--", lw=0.9, label="Withheld measurements"),
                   Line2D([], [], color="0.4", lw=1.1, label="Predictive median"),
                   Patch(facecolor="0.4", alpha=0.16, label="Nominal 90% interval")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.007),
                   ncol=2, fontsize=7, frameon=False)
        fig.subplots_adjust(left=0.13, right=0.92, top=0.78, bottom=0.27,
                            hspace=0.32, wspace=0.17)
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
        manifest = {"schema_version": 1, "figure": "fig_imputation",
                    "source": {"path": "fig_imputation.py", "sha256": source_hash},
                    "style": {"path": style.name, "sha256": style_hash},
                    "inputs": {"excerpt": {"path": "evidence/imputation/sample-excerpt.npz",
                                           "sha256": args.excerpt_sha256},
                               "provenance": {"path": "evidence/imputation/MANIFEST.json",
                                              "sha256": args.provenance_sha256}},
                    "cases": detail_coordinates(data), "y_scale": "linear",
                    "limit_rule": "truth bins 16:32 and ranks 1,37; neural methods share limits, Gaussian separate; padding 8% of range, floor 1e-6",
                    "axis_sides": {"neural": "left", "gaussian": "right"},
                    "matplotlib_version": matplotlib.__version__, "output_sha256": {}}
        fig = plot_detail(data)
        try:
            args.output_dir.mkdir()
            with plt.rc_context(RC):
                for suffix in ("pdf", "png"):
                    path = args.output_dir / f"fig_imputation.{suffix}"
                    with path.open("xb") as handle:
                        fig.savefig(handle, format=suffix, dpi=200)
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
    manifest = {"schema_version": 1, "figure": "fig_imputation",
                "source": {"path": "fig_imputation.py", "sha256": source_hash},
                "style": {"path": style.name, "sha256": style_hash},
                "inputs": {"excerpt": {"path": "evidence/imputation/sample-excerpt.npz",
                                       "sha256": args.excerpt_sha256},
                           "provenance": {"path": "evidence/imputation/MANIFEST.json",
                                          "sha256": args.provenance_sha256}},
                "case_index": args.case_index, "case_id": str(data["ids"][(120, 360).index(args.case_index)]),
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
                    fig.savefig(handle, format=suffix, dpi=200)
                manifest["output_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        _hashed_bytes(source, source_hash)
        _hashed_bytes(style, style_hash)
        _hashed_bytes(args.excerpt, args.excerpt_sha256)
        _hashed_bytes(args.provenance, args.provenance_sha256)
        with plt.rc_context(RC):
            fig.canvas.draw()
        manifest["y_limits"] = [list(fig.axes[row * 4].get_ylim()) for row in range(5)]
        manifest["y_scale"] = [dict(spec) for spec in Y_SCALES]
        for row, expected in enumerate(Y_SCALES):
            axis = fig.axes[row * 4]
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
                        raise ValueError("v_s tick labels must be plain ASCII")
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
