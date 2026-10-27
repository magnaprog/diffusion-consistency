"""Plot energy-score components and errors in nominal prediction intervals.

The first subplot compares G5 with D at each seed. The spread term is half
the mean distance between distinct draws. The other subplots count values
below and above the inclusive interval from draws 2 and 38 in sorted order.
Each model retains its own 39 draws; months and models receive equal weight.
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
from matplotlib.patches import Patch
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
COLORS = tuple(STYLE["axes.prop_cycle"].by_key()["color"])
HATCHES = ("", "///", "...", "xxx")
# Same score-component colors as Figure 1; method colors remain separate.
TERM_COLORS = ("#56B4E9", "#E69F00")
SUMMARY_COLOR = "#7B2CBF"
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


def load_calibration(path, digest):
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
        if name in METHODS:
            rows[name] = {"fixed_model_seeds": seeds, "mean_percent": (100 * mean).tolist(),
                          "model_means_percent": (100 * model_means).tolist(),
                          "year_95_percent": (100 * ci).tolist()}
    return rows, {"bootstrap": record["bootstrap"], "per_window_sha256": record["per_window_sha256"]}


def plot(decomposition, tails):
    fig = plt.figure(figsize=(5.5, 5.0))
    grid = fig.add_gridspec(2, 2, height_ratios=(0.85, 1), left=.12, right=.97,
                           bottom=.22, top=.94, hspace=.60, wspace=.20)
    a = fig.add_subplot(grid[0, :])
    b = fig.add_subplot(grid[1, 0])
    c = fig.add_subplot(grid[1, 1], sharey=b)
    x = np.arange(4)
    a.axhline(0, color="0.45", linewidth=.7, zorder=0)
    a.bar(x - .18, [r["delta_target"] for r in decomposition], .34,
          color=TERM_COLORS[0], label="Distance to measurements")
    a.bar(x + .18, [r["negative_delta_dispersion"] for r in decomposition], .34,
          color=TERM_COLORS[1], hatch="///", edgecolor="0.25", linewidth=0.4,
          label="Negative spread term")
    a.plot(x, [r["delta_energy"] for r in decomposition], "D", color=SUMMARY_COLOR, markersize=3.5,
           label="Energy score", gid="energy_difference")
    a.set_xticks(x, ["Seed 7", "Seed 13", "Seed 42", "Mean"])
    a.set_ylabel("Score difference")
    a.set_title("(a) G5 − D: score components", loc="left")
    a.margins(y=.32)
    a.legend(frameon=False, fontsize=7, ncol=3, loc="upper center",
             handlelength=1.4, columnspacing=1.2)

    upper = 10.
    for ax, outcome, title in ((b, 0, "(b) Below nominal 90% interval"), (c, 2, "(c) Above nominal 90% interval")):
        ax.axhline(5, linestyle="--", color="0.35", linewidth=.8, gid="tail_reference")
        for index, name in enumerate(METHODS):
            row = tails[name]
            mean = np.asarray(row["mean_percent"])[:, outcome]
            models = np.asarray(row["model_means_percent"])[:, :, outcome]
            positions = np.arange(5) + (index - 1.5) * .16
            bars = ax.bar(positions, mean, width=.14, color=COLORS[index],
                          hatch=HATCHES[index], edgecolor="white", linewidth=0, zorder=2)
            for channel, bar in enumerate(bars):
                bar.set_gid(f"{name}_{outcome}_mean_{channel}")
            if name != "full_flank_gaussian":
                for seed_index, seed in enumerate(SEEDS):
                    ax.plot(positions + (seed_index - 1) * .035, models[seed_index],
                            linestyle="none", marker="o", color="0.4", markeredgecolor="white",
                            markeredgewidth=.3, markersize=2.8, zorder=3,
                            gid=f"{name}_{outcome}_seed_{seed}")
            upper = max(upper, float(max(mean.max(), models.max())) * 1.08)
        ax.set_xticks(range(5), [r"$\mu_x$", r"$\mu_y$", r"$\mu_z$", r"$m$", r"$v_s$"])
        ax.set_xlim(-.5, 4.5)
        ax.set_title(title, loc="left")
    b.set_ylim(0, min(100, upper))
    b.set_ylabel("Measurements (%)")
    c.tick_params(labelleft=False)
    for ax in (a, b, c):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, pad=2)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    handles = [Patch(facecolor=color, hatch=hatch, edgecolor="white", linewidth=0)
               for color, hatch in zip(COLORS, HATCHES)]
    fig.legend(handles, LABELS, frameon=False, fontsize=7, ncol=4, loc="lower center",
               bbox_to_anchor=(.53, .125), columnspacing=1.3, handletextpad=.4)
    fig.text(.53, .095, "Bars: mean of three diffusion fits or one Gaussian. Gray dots: individual fits.",
             ha="center", fontsize=7)
    fig.text(.53, .059, "The 5% line assumes continuous, exchangeable measurements and draws.",
             ha="center", fontsize=7)
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
        "schema_version": 1, "figure": "fig_calibration", "protocol_sha256": PROTOCOL,
        "source": {"path": source.name, "sha256": source_hash}, "inputs": inputs,
        "style": {"path": STYLE_PATH.name, "sha256": STYLE_SHA256},
        "decomposition": decomposition, "tails": tails, "channels": list(CHANNELS),
        "outcomes": ["below", "inside", "above"], "method_order": list(METHODS),
        "colors": dict(zip(METHODS, COLORS)),
        "hatches": dict(zip(METHODS, ("none", "diagonal", "dots", "crossed"))),
        "calibration_provenance": provenance,
        "pointwise_intervals_plotted": False,
        "seed_dots": {"seeds": list(SEEDS), "offsets_within_bar": [-.035, 0, .035],
                      "color": "0.4", "scope": "individual fixed neural models; not confidence intervals"},
        "figure_inches": [5.5, 5.0], "interval_order_statistics": [2, 38],
        "axes": {"a": "linear; G5 minus D in 60 TRAIN-standardized coordinates; target minus dispersion",
                 "b": "linear percent; target strictly below second order statistic",
                 "c": "linear percent; target strictly above 38th order statistic"},
        "scope": "Raw-selected frozen TEST ensembles, 480 windows. Equal months and fixed neural seeds, with each model scored separately; one Gaussian regression realization. No confidence intervals are plotted. Tail bars are model means; gray dots are the three fixed neural seeds, not pooled draws or a mixture. Pointwise year-bootstrap intervals remain in the numerical record only. No causal or full conditional-calibration claim.",
        "tail_reference": {"percent": 5, "assumption": "continuous exchangeability of target and 39 ensemble draws"},
        "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__, "output_sha256": {},
    }
    with plt.rc_context(STYLE):
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
