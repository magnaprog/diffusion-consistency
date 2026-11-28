"""Compare validation-selected raw TEST energy scores and fixed-fit interval coverage.

The two panels use the saved paired comparison and calibration summary.
Supply both files and their SHA256 digests to render into a fresh directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

PAPER = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_producer(name):
    path = PAPER / (name + ".py")
    digest = sha256(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if sha256(path) != digest:
        raise ValueError("loader source changed during import")
    return module, {"path": path.name, "sha256": digest}


audit, audit_source = load_producer("fig_audit")
calibration, calibration_source = load_producer("fig_calibration")
DEPENDENCY_SOURCES = {"audit": audit_source, "calibration": calibration_source}
STYLE_PATH, STYLE, STYLE_SHA256 = calibration.STYLE_PATH, calibration.STYLE, calibration.STYLE_SHA256
SEEDS, CHANNELS = calibration.SEEDS, calibration.CHANNELS
METHODS = ("D", "I-S", "O-S", "G5", "gaussian", "full_flank_gaussian")
COMPARATORS = ("D", "I-S", "G5")
COMPARISON_PATH = "evidence/real/comparison.json"
CALIBRATION_PATH = "evidence/followup/calibration/summary.json"
SEED_MARKERS = calibration.SEED_MARKERS
SCORE_SEED_SIZE = 5.
SCORE_SEED_OFFSETS = (.34, 0., -.34)
MEAN_MARKER = {**calibration.MEAN_MARKER, "markersize": 12.}
SCORE_COLOR = "#0077BB"
SEED_OFFSETS = (.24, 0., -.24)
FIGURE_INCHES = (5.5, 2.3)


def load_overview(comparison_path, comparison_sha256, calibration_path, calibration_sha256):
    """Load both panels. A legacy CLI call gets its comparison digest from calibration."""
    summary = calibration.read_record(calibration_path, calibration_sha256)
    linked_digest = summary["input_sha256"]["accepted-evidence/real/comparison.json"]
    if comparison_sha256 is not None and comparison_sha256 != linked_digest:
        raise ValueError("comparison SHA256 differs from calibration input binding")
    comparison, digest, _ = audit.load_comparison(comparison_path, linked_digest)
    if (comparison["protocol_sha256"] != calibration.PROTOCOL
            or comparison["fixed_training_seeds"] != list(SEEDS)):
        raise ValueError("requires the original study protocol and fixed training seeds")

    seed_means = comparison["seed_means"]
    contrasts = {}
    for name in COMPARATORS:
        baseline = name + "_raw"
        row = comparison["comparisons"]["O-S_raw_minus_" + baseline]
        if (row["method"] != "O-S_raw" or row["baseline"] != baseline
                or row["estimand"] != "validation_selected_procedures"
                or row["method_fixed_models"] != len(SEEDS) or row["baseline_fixed_models"] != len(SEEDS)):
            raise ValueError(f"unexpected paired raw comparison for {name}")
        diffs = calibration.numbers(row["seed_diffs"], (len(SEEDS),))
        expected = [seed_means["O-S_raw"][str(s)] - seed_means[baseline][str(s)] for s in SEEDS]
        if not np.array_equal(diffs, expected) or row["mean_diff"] != diffs.mean():
            raise ValueError(f"paired seed differences or their mean differ for {name}")
        contrasts[name] = {key: row[key] for key in ("seed_diffs", "mean_diff", "year_95")}

    tails, provenance = calibration.load_calibration(calibration_path, calibration_sha256, methods=METHODS)
    inside = summary["outcomes"].index("inside")
    coverage = {
        name: {"fixed_model_seeds": tails[name]["fixed_model_seeds"],
               "model_means_percent": np.asarray(tails[name]["model_means_percent"])[:, :, inside].tolist()}
        for name in METHODS
    }
    return {
        "inputs": {"comparison": {"path": COMPARISON_PATH, "sha256": digest},
                   "calibration": {"path": CALIBRATION_PATH, "sha256": calibration_sha256}},
        "panel_a": {"comparators": list(COMPARATORS), "seeds": list(SEEDS), "contrasts": contrasts},
        "panel_b": coverage, "calibration_provenance": provenance,
    }


def plot(record):
    """Draw paired score differences and one coverage point per fixed fit and quantity."""
    width, height = FIGURE_INCHES
    fig = plt.figure(figsize=FIGURE_INCHES)

    def panel(left, span):
        return fig.add_axes((left / width, .70 / height, span / width, 1.13 / height))

    a = panel(.79, 1.40)
    box = a.get_position()
    fig.text((box.x0 + box.x1) / 2, 2.15 / height, "(a) Raw energy score differences",
             ha="center", fontsize=8)
    a.set(xlim=(-.105, .255), ylim=(-.52, 2.52))
    a.axvline(0, color=".46", linewidth=.7, zorder=0, gid="zero")
    comparators = record["panel_a"]["comparators"]
    centers = np.arange(len(comparators))[::-1]
    for center in centers[1::2]:
        a.axhspan(center - .5, center + .5, facecolor=calibration.ROW_SHADE, edgecolor="none", zorder=-2, gid=f"band_{center}")
    for name, center in zip(comparators, centers, strict=True):
        contrast = record["panel_a"]["contrasts"][name]
        # Separate the three fits vertically so nearly equal scores stay visible.
        for seed, marker, offset, value in zip(
            SEEDS, SEED_MARKERS, SCORE_SEED_OFFSETS, contrast["seed_diffs"], strict=True
        ):
            a.plot(value, center + offset, marker=marker, color=SCORE_COLOR, linestyle="none",
                   markersize=SCORE_SEED_SIZE, markeredgecolor="white", markeredgewidth=.4, zorder=4, gid=f"{name}_seed_{seed}")
        a.plot(contrast["year_95"], [center] * 2, color=SCORE_COLOR, linewidth=1.1,
               solid_capstyle="butt", gid=f"{name}_year_95")
        a.plot(contrast["mean_diff"], center, zorder=3,
               gid=f"{name}_mean", **MEAN_MARKER)
    a.set_xticks([-.1, 0, .1, .2], ["−0.1", "0", "0.1", "0.2"])
    a.set_yticks(centers, [f"O-S minus {name}" for name in comparators])
    a.tick_params(axis="y", length=0, labelsize=7, pad=3)
    a.spines["left"].set_visible(False)
    a.tick_params(axis="x", pad=2)
    fig.text(1.21 / width, .385 / height, "Negative values favor O-S", ha="center", fontsize=7)

    centers = np.arange(len(CHANNELS))[::-1]
    symbols = (r"$\mu_x$", r"$\mu_y$", r"$\mu_z$", "$m$", "$v_s$")
    labels = ("D", "I-S", "O-S", "G5", "Gaussian interpolation", "Gaussian regression")
    # Leave room between the full Gaussian method names at print size.
    left_edges = (2.46, 2.93, 3.40, 3.87, 4.34, 4.985)
    coverage_width = .415
    coverage_center = (left_edges[0] + left_edges[-1] + coverage_width) / 2
    fig.text(coverage_center / width, 2.15 / height, "(b) Nominal 90% intervals",
             ha="center", fontsize=8)
    for index, (name, label, left) in enumerate(zip(METHODS, labels, left_edges, strict=True)):
        ax = panel(left, coverage_width)
        ax.set(xlim=(-6, 106), ylim=(-.51, 4.51))
        ax.set_title(label.replace(" ", "\n"), fontsize=7, pad=3)
        ax.axvline(90, lw=.65, color=".36", linestyle=(0, (2.3, 2)), zorder=1, gid="coverage_reference")
        for center in centers[1::2]:
            ax.axhspan(center - .5, center + .5, facecolor=calibration.ROW_SHADE,
                       edgecolor="none", zorder=-2, gid=f"band_{center}")
        row = record["panel_b"][name]
        for seed, values in zip(row["fixed_model_seeds"], row["model_means_percent"]):
            if seed is None:
                offset, marker = 0., "P"
            else:
                k = SEEDS.index(seed)
                offset, marker = SEED_OFFSETS[k], SEED_MARKERS[k]
            ax.scatter(values, centers + offset, marker=marker, color=calibration.METHOD_COLORS[name], s=9,
                       linewidths=.35, zorder=3, gid=f"{name}_coverage_{seed}")
        ax.set_xticks([0, 50, 90], ["0", "50", "90"])
        ax.set_yticks(centers, symbols)
        ax.tick_params(axis="y", labelleft=(index == 0), length=0, pad=3)
        ax.tick_params(axis="x", labelsize=6.7, pad=2)
        ax.spines["left"].set_visible(False)
    fig.text(3.97 / width, .385 / height, "Coverage (%)", ha="center", fontsize=7)
    handles = [Line2D([], [], marker=marker, color=calibration.SEED_LEGEND_COLOR, linestyle="none", markersize=4.4, label=f"Seed {seed}")
               for seed, marker in zip(SEEDS, SEED_MARKERS)]
    handles.append(Line2D([], [], color=SCORE_COLOR, linewidth=1,
                          label="Mean, 95% year interval", **calibration.MEAN_KEY))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .008), ncol=len(handles),
               handletextpad=.3, columnspacing=1.2, handlelength=1.4, fontsize=7, frameon=False)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--comparison", type=Path, help="raw paired comparison JSON")
    source.add_argument("--evidence-dir", type=Path, help="legacy published evidence/followup/checkpoints directory")
    parser.add_argument("--comparison-sha256", help="required with --comparison")
    parser.add_argument("--manifest-sha256", help="required with --evidence-dir")
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--calibration-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if ((args.comparison is None) != (args.comparison_sha256 is None)
            or (args.evidence_dir is None) != (args.manifest_sha256 is None)):
        parser.error("pair --comparison with --comparison-sha256 or --evidence-dir with --manifest-sha256")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError("figure output directory must be fresh")
    comparison_path = args.comparison
    if args.evidence_dir is not None:
        legacy = calibration.read_record(args.evidence_dir / "MANIFEST.json", args.manifest_sha256)
        if legacy["analysis"] != "checkpoint_followup_public_export":
            raise ValueError("expected the published checkpoint manifest")
        comparison_path = args.evidence_dir.parent.parent / "real/comparison.json"

    source_hash = sha256(__file__)
    if sha256(STYLE_PATH) != STYLE_SHA256:
        raise ValueError("style changed since import")
    font = font_manager.findfont(STYLE["font.family"][0], fallback_to_default=False)
    if Path(font).suffix.lower() != ".ttf":
        raise ValueError("DejaVu Serif TrueType is required")
    record = load_overview(comparison_path, args.comparison_sha256, args.calibration, args.calibration_sha256)
    manifest = {
        "schema_version": 11, "figure": "fig_overview", "protocol_sha256": calibration.PROTOCOL,
        "source": {"path": Path(__file__).name, "sha256": source_hash},
        "dependency_sources": DEPENDENCY_SOURCES,
        "style": {"path": STYLE_PATH.name, "sha256": STYLE_SHA256}, **record,
        "figure_inches": list(FIGURE_INCHES), "channels": list(CHANNELS), "method_order": list(METHODS),
        "mean_marker": MEAN_MARKER,
        "score_lanes": {"seed_offsets": list(SCORE_SEED_OFFSETS), "seed_markersize": SCORE_SEED_SIZE, "mean_offset": 0., "interval_offset": 0.},
        "row_bands": {"colors": ["#ffffff", calibration.ROW_SHADE], "a": "comparison", "b": "quantity"},
        "method_colors": calibration.METHOD_COLORS, "score_color": SCORE_COLOR,
        "seed_legend_color": calibration.SEED_LEGEND_COLOR,
        "seed_markers": dict(zip(map(str, SEEDS), SEED_MARKERS)),
        "axes": {
            "a": "Raw TEST energy score in 60 TRAIN-standardized coordinates. Each symbol gives a "
                 "paired O-S minus comparator difference after validation selection. Gold bars give the "
                 "three-seed mean. Lines give its marginal 95% paired year-resampling interval.",
            "b": "Each diffusion symbol gives the coverage percentage for one fixed fit. "
                 "Each Gaussian cross gives coverage for one ensemble realization per window. "
                 "Coverage averages over twelve hidden positions with equal month weights."},
        "interval_order_statistics": [2, 38], "n_draws": 39, "n_windows": 480,
        "coverage_reference": {"percent": 90, "assumption": "at each time and coordinate, "
                               "the observation and 39 ensemble members are exchangeable, with no ties"},
        "pointwise_intervals_plotted": False,
        "scope": "Previously examined TEST period. Year resampling holds selected fits and ensemble draws fixed.",
        "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__,
        "output_sha256": {}}
    with plt.rc_context(STYLE):
        fig = plot(record)
        try:
            args.output_dir.mkdir(parents=True)
            for suffix in ("pdf", "png"):
                path = args.output_dir / f"fig_overview.{suffix}"
                metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else {"Software": "PriGM fig_overview"}
                with path.open("xb") as handle:
                    fig.savefig(handle, format=suffix, dpi=200, metadata=metadata)
                manifest["output_sha256"][path.name] = sha256(path)
        finally:
            plt.close(fig)
    if (sha256(__file__) != source_hash or sha256(STYLE_PATH) != STYLE_SHA256
            or any(sha256(PAPER / row["path"]) != row["sha256"] for row in DEPENDENCY_SOURCES.values())):
        raise ValueError("source changed during rendering")
    if sha256(comparison_path) != record["inputs"]["comparison"]["sha256"] or sha256(args.calibration) != args.calibration_sha256:
        raise ValueError("input changed during rendering")
    with (args.output_dir / "fig_overview.manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
