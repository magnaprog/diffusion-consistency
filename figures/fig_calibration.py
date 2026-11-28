"""Plot energy-score components and errors in nominal prediction intervals.

The first subplot compares G5 with D at each training seed. The score subtracts
half the mean distance over pairs with distinct member indices from the mean
distance to the observation. The other subplots show the percentages below the
second or above the 38th ordered value among 39 members. Months receive equal
weight within a fit. For diffusion, the summary gives each fit equal weight
within its method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np

PROTOCOL = "25d0cdefdd5dbb160c32c7b5f4a6053bab99e4a9f64832d9eb6502232bf24e6b"
SEEDS = (7, 13, 42)
CHANNELS = ("mu_x", "mu_y", "mu_z", "m", "v_s")
METHODS = ("D", "O-S", "G5", "full_flank_gaussian")
LABELS = ("D", "O-S", "G5", "Gaussian regression")
STYLE_PATH = Path(__file__).with_name("prigm.mplstyle")
STYLE_SHA256 = hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest()
STYLE = matplotlib.rc_params_from_file(STYLE_PATH, use_default_template=False)
if hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest() != STYLE_SHA256:
    raise ValueError("style changed during import")
# Method colors use Paul Tol's palettes: https://sronpersonalpages.nl/~pault/
METHOD_COLORS = {"D": "#0077BB", "I-S": "#AA3377", "O-S": "#EE3377", "G5": "#EE7733",
                 "gaussian": "#666666", "full_flank_gaussian": "#009988"}
SEED_LEGEND_COLOR = "#333333"
SEED_MARKERS = ("o", "s", "^")
SEED_OFFSETS = (-.26, 0., .26)
MEAN_MARKER = {"marker": "|", "markersize": 14., "markeredgecolor": "#B07800", "markeredgewidth": 1.7}
MEAN_KEY = {**MEAN_MARKER, "markersize": 8.}
ROW_SHADE = "#f0f0f0"
FIGURE_INCHES = (5.5, 3.6)
# Match the approved print proof's type and line weights, retaining shared font settings.
PLOT_STYLE = {"font.family": STYLE["font.family"], "pdf.fonttype": STYLE["pdf.fonttype"],
              "font.size": 7, "axes.linewidth": .5, "xtick.major.width": .5, "ytick.major.width": .5}
# Blue and purple distinguish the two terms. Dark endpoints give their sum.
DISTANCE_COLOR = "#0077BB"
SPREAD_COLOR = "#AA3377"
SUMMARY_COLOR = "#111111"
ALL_METHODS = {*METHODS, "I-S", "gaussian"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_record(path, expected_sha256):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"input SHA256 mismatch: {path}")
    return json.loads(payload, object_pairs_hook=unique_object)


def numbers(value, shape):
    values = np.asarray(value)
    if (values.shape != shape or values.dtype.kind not in "iuf"
            or not np.isfinite(values).all()):
        raise ValueError(f"expected finite numeric shape {shape}")
    return values.astype(float)


def load_decomposition(path, digest):
    record = read_record(path, digest)
    if (record.get("schema_version") != 1 or record.get("analysis") != "paired_operator_mechanism"
            or record.get("protocol_sha256") != PROTOCOL or record.get("n_windows") != 480):
        raise ValueError("requires the original 480-window operator analysis")
    terms = ("target_term", "dispersion_term", "energy_score")
    data = {}
    for name in ("D", "G5"):
        method = record["methods"][name]
        if set(method["seedwise"]) != set(map(str, SEEDS)) or method["n_realizations"] != 3:
            raise ValueError("operator requires all three fixed seeds")
        values = numbers([[method["seedwise"][str(seed)]["metrics"]["raw_" + term]["month_balanced"]
                           for term in terms] for seed in SEEDS], (3, 3))
        mean = numbers([method["metrics"]["raw_" + term]["month_balanced"] for term in terms], (3,))
        if (not np.allclose(values[:, 0] - values[:, 1], values[:, 2], rtol=1e-12, atol=1e-14)
                or not np.allclose(mean, values.mean(axis=0), rtol=1e-12, atol=1e-14)):
            raise ValueError("operator score identity or equal-seed mean differs")
        data[name] = values
    difference = data["G5"] - data["D"]
    values = np.vstack([difference, difference.mean(axis=0)])
    return [{"seed": seed, "delta_target": float(row[0]),
             "negative_delta_dispersion": float(-row[1]), "delta_energy": float(row[2])}
            for seed, row in zip((*SEEDS, "mean"), values)]


def load_calibration(path, digest, *, methods=METHODS):
    record = read_record(path, digest)
    if (record.get("schema_version") != 1 or record.get("n_windows") != 480
            or record.get("n_draws") != 39 or record.get("interval_order_statistics") != [2, 38]
            or record.get("channels") != list(CHANNELS)
            or record.get("outcomes") != ["below", "inside", "above"]
            or record.get("continuous_exchangeability_reference") != [.05, .90, .05]
            or set(record.get("methods", {})) != ALL_METHODS
            or record.get("bootstrap") != {"replicates": 10000, "seed": 7003,
                                           "schemes": ["year", "twoyear"]}):
        raise ValueError("requires the complete frozen tail-analysis schema and bootstrap")
    rows = {}
    for name, method in record["methods"].items():
        seeds = list(SEEDS) if name in ("D", "I-S", "O-S", "G5") else [None]
        if method["fixed_model_seeds"] != seeds:
            raise ValueError("tail-analysis fixed-model inventory differs")
        model_means = numbers(method["model_means"], (len(seeds), 5, 3))
        mean = numbers(method["mean"], (5, 3))
        ci = numbers(method["year_95"], (2, 5, 3))
        if (np.any(model_means < 0) or np.any(model_means > 1)
                or not np.allclose(model_means.sum(axis=-1), 1, rtol=0, atol=1e-14)
                or not np.allclose(mean, model_means.mean(axis=0), rtol=0, atol=1e-14)
                or np.any(ci < 0) or np.any(ci > 1) or np.any(ci[0] > ci[1])):
            raise ValueError("tail fractions, model reduction, or interval bounds differ")
        # An amended report carries the paired contrast; it is not drawn as an error bar here.
        contrast = numbers(method["below_minus_above"], (5,))
        contrast_ci = numbers(method["year_below_minus_above_95"], (2, 5))
        if (not np.allclose(contrast, mean[:, 0] - mean[:, 2], rtol=0, atol=1e-14)
                or np.any(contrast_ci < -1) or np.any(contrast_ci > 1)
                or np.any(contrast_ci[0] > contrast_ci[1])):
            raise ValueError("paired tail contrast differs")
        if name in methods:
            rows[name] = {"fixed_model_seeds": seeds, "mean_percent": (100 * mean).tolist(),
                          "model_means_percent": (100 * model_means).tolist(),
                          "year_95_percent": (100 * ci).tolist()}
    return rows, {"bootstrap": record["bootstrap"], "per_window_sha256": record["per_window_sha256"]}


def plot(decomposition, tails):
    fig = plt.figure(figsize=FIGURE_INCHES)
    width, height = FIGURE_INCHES

    def axis(left, bottom, w, h):
        return fig.add_axes([left / width, bottom / height, w / width, h / height])

    ax = axis(.47, .97, 1.18, 2.03)
    score_center = ax.get_position().x0 + ax.get_position().width / 2
    fig.text(score_center, 3.35 / height, "(a) Energy score, Δ = G5 − D", ha="center", fontsize=8)
    # The second segment starts where the first ends. Its signed length is minus
    # the change in half the mean distance between distinct ensemble members.
    extent = max(abs(row[key]) for row in decomposition for key in ("delta_target", "delta_energy"))
    limit = max(.39, 1.1 * extent)
    ax.set(xlim=(-limit, limit), ylim=(-.52, 3.67))
    positions = (3, 2, 1, 0)
    for y in positions[1::2]:
        ax.axhspan(y - .5, y + .5, facecolor=ROW_SHADE, edgecolor="none", zorder=-2, gid=f"band_{y}")
    ax.set_yticks(positions, ["7", "13", "42", "Mean"])
    if limit == .39:
        ax.set_xticks([-.3, 0, .3], ["−0.3", "0", "0.3"])
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", length=2, pad=2)
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.axvline(0, color=".47", lw=.7, zorder=0)
    ax.axhline(.5, color=".72", lw=.55)
    ax.text(-limit * (.38 / .39), 3.60, "Seed", fontsize=7, va="bottom", ha="right")
    ax.set_xlabel("Score difference", labelpad=3)
    for y, row in zip(positions, decomposition):
        target, energy = row["delta_target"], row["delta_energy"]
        seed = row["seed"]
        upper, lower = (y + .23, y - .23) if seed == "mean" else (y + .14, y - .13)
        ax.plot([0, target], [upper, upper], color=DISTANCE_COLOR, lw=1.8,
                solid_capstyle="butt", gid=f"distance_{seed}")
        ax.plot([target, target], [upper, lower], color=".57", lw=.5, gid=f"connector_{seed}")
        endpoint = MEAN_MARKER if seed == "mean" else {"marker": SEED_MARKERS[SEEDS.index(seed)]}
        ax.plot(target, upper, color=DISTANCE_COLOR, gid=f"distance_endpoint_{seed}", **({"markersize": 3.3} | endpoint))
        ax.add_patch(FancyArrowPatch((target, lower), (energy, lower), transform=ax.transData,
                                     arrowstyle="-|>", mutation_scale=5, shrinkA=0, shrinkB=2,
                                     color=SPREAD_COLOR, lw=1.1, gid=f"spread_{seed}"))
        ax.plot(energy, lower, color=SUMMARY_COLOR, gid=f"energy_{seed}", **({"markersize": 3.7} | endpoint))

    symbols = (r"$\mu_x$", r"$\mu_y$", r"$\mu_z$", r"$m$", r"$v_s$")
    labels = [label for symbol in symbols for label in (symbol + "  Below", "Above")]
    # Keep one common percentage scale across all methods and include every displayed rate.
    maximum = max(np.asarray(tails[name]["model_means_percent"])[:, :, (0, 2)].max() for name in METHODS)
    xmax = max(88, maximum + 2)
    for index, (name, label) in enumerate(zip(METHODS, LABELS)):
        axis_tail = axis(2.42 + .775 * index, .71, .66, 2.37)
        axis_tail.set(xlim=(-4, xmax), ylim=(9.5, -.5))
        axis_tail.axvline(5, color=".48", linewidth=.55, linestyle=(0, (2, 2)), zorder=1,
                          gid="tail_reference")
        for channel in (1, 3):
            axis_tail.axhspan(2 * channel - .5, 2 * channel + 1.5, facecolor=ROW_SHADE,
                              edgecolor="none", zorder=-2, gid=f"band_{CHANNELS[channel]}")
        models = np.asarray(tails[name]["model_means_percent"])
        means = np.asarray(tails[name]["mean_percent"])
        for parity, outcome in enumerate((0, 2)):
            y = 2 * np.arange(len(CHANNELS)) + parity
            if name == "full_flank_gaussian":
                axis_tail.plot(models[0, :, outcome], y, marker="P", markersize=3.4, linewidth=0,
                               color=METHOD_COLORS[name], gid=f"{name}_{outcome}_ensemble")
                continue
            for model, seed, marker, offset in zip(models, SEEDS, SEED_MARKERS, SEED_OFFSETS, strict=True):
                axis_tail.plot(model[:, outcome], y + offset, marker=marker, color=METHOD_COLORS[name], markersize=2.55,
                               linewidth=0, markeredgecolor="white", markeredgewidth=.35, zorder=4,
                               gid=f"{name}_{outcome}_seed_{seed}")
            axis_tail.plot(means[:, outcome], y, linewidth=0, zorder=2,
                           gid=f"{name}_{outcome}_mean", **MEAN_MARKER)
        axis_tail.set_yticks(range(10), labels if index == 0 else [])
        axis_tail.set_xticks([0, 40, 80])
        axis_tail.tick_params(axis="x", pad=1.3, length=2, labelsize=6.7)
        axis_tail.tick_params(axis="y", length=0, pad=2, labelsize=6.6)
        axis_tail.set_title(label.replace(" ", "\n"), pad=2, fontsize=7.2)
        axis_tail.spines[["left", "right", "top"]].set_visible(False)
    tails_center = (fig.axes[1].get_position().x0 + fig.axes[-1].get_position().x1) / 2
    fig.text(tails_center, 3.35 / height, "(b) Observations outside 90% intervals", ha="center", fontsize=8)
    fig.text(3.85 / width, .49 / height, "Observations (%)", ha="center")
    terms = [Line2D([], [], color=DISTANCE_COLOR, lw=1.8),
             Line2D([], [], marker=">", color=SPREAD_COLOR, lw=1, ms=3.2),
             tuple(Line2D([], [], marker=marker, color=SUMMARY_COLOR, ls="", ms=3.1)
                   for marker in SEED_MARKERS) + (Line2D([], [], ls="", **MEAN_KEY),)]
    fig.legend(terms, ["Δ mean distance to observation", "−½ Δ(mean pairwise distance)", "Δ energy score"],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=.18)}, loc="lower left",
               bbox_to_anchor=(.004, .026), frameon=False, fontsize=6.8, handlelength=3.15,
               handletextpad=.4, labelspacing=.42)
    handles = [Line2D([], [], marker=marker, color=SEED_LEGEND_COLOR, linestyle="", markersize=3.3, label=f"Seed {seed}")
               for seed, marker in zip(SEEDS, SEED_MARKERS)]
    handles.extend([Line2D([], [], linestyle="", label="Mean", **MEAN_KEY),
                    Line2D([], [], marker="P", color=METHOD_COLORS["full_flank_gaussian"], linestyle="", markersize=3.2, label="Gaussian regression"),
                    Line2D([], [], color=".45", linestyle="--", linewidth=.65, label="5% per side")])
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(.985, .005), ncol=3, frameon=False,
               fontsize=6.6, handlelength=1.2, columnspacing=.7, handletextpad=.4, labelspacing=.45)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("operator", "calibration"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest() != STYLE_SHA256:
        raise ValueError("style changed since import")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError("figure output directory must be fresh")
    font = font_manager.findfont(STYLE["font.family"][0], fallback_to_default=False)
    if Path(font).suffix.lower() != ".ttf":
        raise ValueError("DejaVu Serif TrueType is required; see docs/prigm_release.md")
    source = Path(__file__).resolve()
    source_hash = sha256(source)
    decomposition = load_decomposition(args.operator, args.operator_sha256)
    tails, provenance = load_calibration(args.calibration, args.calibration_sha256)
    inputs = {name: {"path": getattr(args, name).name, "sha256": getattr(args, name + "_sha256")}
              for name in ("operator", "calibration")}
    manifest = {
        "schema_version": 10, "figure": "fig_calibration", "protocol_sha256": PROTOCOL,
        "source": {"path": source.name, "sha256": source_hash}, "inputs": inputs,
        "style": {"path": STYLE_PATH.name, "sha256": STYLE_SHA256},
        "decomposition": decomposition, "tails": tails, "channels": list(CHANNELS),
        "outcomes": ["below", "inside", "above"], "method_order": list(METHODS),
        "method_colors": {name: METHOD_COLORS[name] for name in METHODS},
        "seed_legend_color": SEED_LEGEND_COLOR,
        "score_colors": {"delta_target": DISTANCE_COLOR, "negative_delta_dispersion": SPREAD_COLOR,
                         "delta_energy": SUMMARY_COLOR},
        "mean_marker": MEAN_MARKER,
        "row_bands": {"colors": ["#ffffff", ROW_SHADE], "a": "seed or mean", "b": "quantity"},
        "score_encoding": {"distance_segment": ["zero", "delta_target"],
                           "spread_arrow": ["delta_target", "delta_energy"],
                           "arrow_displacement": "negative_delta_dispersion",
                           "point_endpoints": ["delta_target", "delta_energy"],
                           "endpoint_shapes": "seed markers or vertical mean bar"},
        "tail_encoding": {"columns": list(METHODS), "row_order_within_quantity": ["below", "above"],
                          "neural_mean": {"marker": "|", "vertical_offset": 0},
                          "gaussian_ensemble": {"marker": "P", "vertical_offset": 0}},
        "calibration_provenance": provenance,
        "pointwise_intervals_plotted": False,
        "seed_glyphs": {"seeds": list(SEEDS), "markers": list(SEED_MARKERS),
                        "vertical_offsets_within_tail": list(SEED_OFFSETS),
                        "scope": "individual fixed neural models; not confidence intervals"},
        "figure_inches": list(FIGURE_INCHES), "interval_order_statistics": [2, 38],
        "axes": {"a": "linear; G5 minus D in 60 TRAIN-standardized coordinates; target minus dispersion",
                 "b": "linear percent; below second and above 38th order statistics, paired within quantity"},
        "scope": "Raw-selected frozen TEST ensembles, 480 windows. Equal months and fixed neural seeds, "
                 "each scored separately. Gold bars average three fixed neural fits. Crosses show one "
                 "Gaussian regression ensemble per window. Pointwise year-bootstrap intervals remain in the "
                 "numerical record only. No causal or full conditional-calibration claim.",
        "tail_reference": {"percent": 5, "assumption": "at each time and coordinate, the observation "
                           "and 39 ensemble members are exchangeable, with no ties"},
        "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__, "output_sha256": {},
    }
    with plt.rc_context(PLOT_STYLE):
        fig = plot(decomposition, tails)
        try:
            args.output_dir.mkdir()
            for suffix in ("pdf", "png"):
                path = args.output_dir / f"fig_calibration.{suffix}"
                metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else {"Software": "PriGM fig_calibration"}
                with path.open("xb") as handle:
                    fig.savefig(handle, format=suffix, dpi=200, metadata=metadata)
                manifest["output_sha256"][path.name] = sha256(path)
        finally:
            plt.close(fig)
    if (sha256(source) != source_hash or sha256(args.operator) != args.operator_sha256
            or sha256(args.calibration) != args.calibration_sha256
            or sha256(STYLE_PATH) != manifest["style"]["sha256"]):
        raise ValueError("figure source or input changed during rendering")
    with (args.output_dir / "fig_calibration.manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
