"""Plot changes after applying the clipping map to the same magnetic predictions.

Displacement uses 60 standardized coordinates. Clipping changes both the mean
distance to the observation and half the mean distance over pairs with distinct
member indices. The score subtracts the second term from the first. Diffusion
markers show individual fits. Each Gaussian method has one ensemble per window.
Panel (b) separates clipping D at its original step from choosing the step by
the clipped validation score.
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
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np

STYLE_PATH = Path(__file__).with_name("prigm.mplstyle")
STYLE_SHA256 = hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest()
STYLE = matplotlib.rc_params_from_file(STYLE_PATH, use_default_template=False)
if hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest() != STYLE_SHA256:
    raise ValueError("style changed during import")
# Paul Tol's vibrant blue and bright purple (Figs. 3 and 1): https://sronpersonalpages.nl/~pault/
TARGET_COLOR = "#0077BB"
PAIR_COLOR = "#AA3377"
GAUSSIAN_INPUTS = {
    "observations": {"path": "evidence/observations/test.npz",
                     "sha256": "2cbe75399a190d1522379701cced5c74718aaced251bdf4179f246b72b46c1c7"},
    "gaussian_ensemble": {"path": "evidence/deadline/gaussian/test/lambda_0.1_rho_0/gen_00000000/ensemble.npz",
                          "sha256": "b2298486f615c685debd3c59026ca7ee2d6c6ed235442fc9726e9fa8d1afada4"},
}
SELECTION_TERMS = ("fixed_checkpoint_correction", "selection_effect", "total_change")

METHODS = ("D", "I-S", "O-S", "G5", "gaussian")
LABELS = ("D", "I-S", "O-S", "G5", "Gaussian\ninterpolation")
SEEDS = ("7", "13", "42")
PROTOCOL = "25d0cdefdd5dbb160c32c7b5f4a6053bab99e4a9f64832d9eb6502232bf24e6b"


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def finite(value):
    if type(value) not in (float, int) or not np.isfinite(value):
        raise ValueError("expected a finite plotted number")
    return float(value)


def load_rows(path, expected_sha256):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("operator evidence SHA256 mismatch")
    record = json.loads(payload, object_pairs_hook=unique_object)
    if (record.get("schema_version") != 1 or record.get("analysis") != "paired_operator_mechanism"
            or record.get("protocol_sha256") != PROTOCOL or record.get("n_windows") != 480
            or set(record.get("methods", {})) != {*METHODS, "interpolation", "persistence"}):
        raise ValueError("requires the complete original paired-operator analysis")
    rows = []
    for name, label in zip(METHODS, LABELS):
        method = record["methods"][name]
        keys = SEEDS if name != "gaussian" else ("shared",)
        if set(method["seedwise"]) != set(keys) or method["n_realizations"] != len(keys):
            raise ValueError("incomplete fixed-seed or control realization inventory")
        metrics = method["metrics"]
        movement = finite(metrics["mean_movement"]["month_balanced"])
        seed_movement = [finite(method["seedwise"][key]["metrics"]["mean_movement"]["month_balanced"])
                         for key in keys]
        target = finite(metrics["delta_target_term"]["month_balanced"])
        dispersion = finite(metrics["delta_dispersion_term"]["month_balanced"])
        score = finite(metrics["delta_energy_score"]["month_balanced"])
        if (movement <= 0 or any(x <= 0 for x in seed_movement)
                or not np.isclose(movement, np.mean(seed_movement), rtol=1e-12, atol=1e-14)
                or not np.isclose(target - dispersion, score, rtol=1e-12, atol=1e-14)):
            raise ValueError("movement reduction or energy-component identity differs")
        seed_energy = [finite(method["seedwise"][key]["metrics"]["delta_energy_score"]["month_balanced"])
                       for key in keys]
        if not np.isclose(np.mean(seed_energy), score, rtol=1e-12, atol=1e-14):
            raise ValueError("mean score change differs from seed changes")
        rows.append({"seed_energy": seed_energy, "method": name, "label": label, "movement": movement,
                     "seed_movement": seed_movement, "delta_target": target,
                     "negative_delta_dispersion": -dispersion, "delta_energy": score,
                     "violation_fraction": finite(metrics["raw_support_violation_fraction"]["month_balanced"]),
                     "mean_window_p90_m_deficit": finite(metrics["raw_m_minus_r_severity_quantiles"]["month_balanced"][2])})
    source = record["D_selection_decomposition"]
    if set(source["seedwise"]) != set(SEEDS):
        raise ValueError("D selection decomposition requires all three fixed seeds")
    selection = {key: finite(source["metrics"][key]["month_balanced"]) for key in
                 ("fixed_checkpoint_correction", "selection_effect", "total_change")}
    if not np.isclose(selection["fixed_checkpoint_correction"] + selection["selection_effect"],
                      selection["total_change"], rtol=1e-12, atol=1e-14):
        raise ValueError("D selection decomposition does not add up")
    if not np.isclose(selection["fixed_checkpoint_correction"], rows[0]["delta_energy"], rtol=1e-12, atol=1e-14):
        raise ValueError("D clipping effect differs between the operator and checkpoint records")
    fits = {key: [finite(source["seedwise"][seed]["metrics"][key]["month_balanced"]) for seed in SEEDS]
            for key in SELECTION_TERMS}
    if (not np.allclose(np.add(fits[SELECTION_TERMS[0]], fits[SELECTION_TERMS[1]]),
                        fits[SELECTION_TERMS[2]], rtol=1e-12, atol=1e-14)
            or not np.allclose(fits[SELECTION_TERMS[0]], rows[0]["seed_energy"], rtol=1e-12, atol=1e-14)
            or any(not np.isclose(np.mean(fits[k]), selection[k], rtol=1e-12, atol=1e-14)
                   for k in SELECTION_TERMS)):
        raise ValueError("D selection decomposition differs across fitted seeds")
    selection["seedwise"] = fits
    return rows, selection


def gaussian_regression(paper):
    """Measure clipping of the saved regression draws in 60 TRAIN-standardized coordinates."""
    inputs = {name: paper / record["path"] for name, record in GAUSSIAN_INPUTS.items()}
    for name, path in inputs.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != GAUSSIAN_INPUTS[name]["sha256"]:
            raise ValueError("Gaussian figure input SHA256 mismatch: " + name)
    with np.load(inputs["observations"], allow_pickle=False) as obs, np.load(
            inputs["gaussian_ensemble"], allow_pickle=False) as forecast:
        ids = obs["ids"]
        if not np.array_equal(ids, forecast["ids"]):
            raise ValueError("Gaussian forecast and observation IDs differ")
        mean, scale = obs["mean"][:5], obs["scale"][:5]
        targets = obs["z"][:, 18:30, :5]
        draws = forecast["ensemble"][:, :, :5, 18:30].transpose(0, 1, 3, 2) * scale + mean
    if draws.shape != (480, 39, 12, 5) or targets.shape != (480, 12, 5):
        raise ValueError("Gaussian hidden window dimensions differ")
    first, second = np.triu_indices(39, k=1)
    changes = []
    for raw, target in zip(draws, targets):
        corrected = raw.copy()
        radius = np.linalg.norm(raw[..., :3], axis=-1)
        corrected[..., 3] = np.maximum(raw[..., 3], radius)
        corrected[..., 4] = np.clip(raw[..., 4], 0, 1.5 * corrected[..., 3]**2 + 2.5 * radius**2)
        movement = np.linalg.norm((corrected - raw) / scale, axis=(1, 2)).mean()
        distances = [np.linalg.norm((v - target) / scale, axis=(1, 2)).mean() for v in (raw, corrected)]
        spreads = [np.linalg.norm((v[first] - v[second]) / scale, axis=(1, 2)).mean() / 2
                   for v in (raw, corrected)]
        changes.append((movement, distances[1] - distances[0], spreads[0] - spreads[1]))
    values = np.asarray(changes)
    months = np.asarray([window[:7] for window in ids])
    moved, target, spread = np.mean([values[months == m].mean(axis=0) for m in np.unique(months)], axis=0).tolist()
    return {"method": "full_flank_gaussian", "label": "Gaussian regression", "movement": moved,
            "delta_target": target, "negative_delta_dispersion": spread, "delta_energy": target + spread}


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
METHOD_COLORS = {
    "D": "#0077BB", "I-S": "#AA3377", "O-S": "#EE3377", "G5": "#EE7733",
    "gaussian": "#666666", "full_flank_gaussian": "#009988",
}


def seed_points(axis, y, values, color, seed_offset):
    average = np.mean(values)
    for x, offset, marker in zip(values, (seed_offset, 0, -seed_offset), SEED_MARKERS):
        axis.plot(x, y + offset, marker=marker, markersize=3.4, color=color,
                  markeredgewidth=.8, linestyle="none", zorder=3)
    axis.plot([average, average], [y - .39, y + .39], color=MEAN_COLOR,
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


def panel_b_rows(rows, selection):
    """Group same-checkpoint clipping comparisons above the extra checkpoint effect and total for D."""
    changes = [(row["method"], "fixed_checkpoint_correction", row["seed_energy"]) for row in rows[:4]]
    for effect in ("selection_effect", "total_change"):
        changes.append(("D", effect, selection["seedwise"][effect]))
    return [{"method": method, "effect": effect, "seed_changes": values, "mean_change": float(np.mean(values))}
            for method, effect, values in changes]


def gaussian_reductions(rows):
    """The signed score change is the reduction in half pair distance minus that in target distance."""
    return [{"method": row["method"], "target_reduction": -row["delta_target"],
             "half_pair_reduction": row["negative_delta_dispersion"], "score_change": row["delta_energy"]}
            for row in rows[4:]]


def plot(rows, selection):
    fig = plt.figure(figsize=(5.5, 3.1))
    methods = [row["method"] for row in rows[:4]]
    left_x, left_width = 1.20, 1.43
    a = panel(fig, (left_x, 1.60, left_width, 1.16), range(5, -1, -1),
              methods + ["Gaussian interpolation", "Gaussian regression"], "(a) Mean distance moved",
              "", [.1, 1, 2.5], (.045, 3))
    a.set_xscale("log")
    a.set_xticks([.1, 1, 2.5], ["0.1", "1", "2.5"])
    a.xaxis.set_minor_locator(NullLocator())
    # A half-row gap separates the four fixed-checkpoint comparisons from the two D rows below.
    b_levels = (5.5, 4.5, 3.5, 2.5, 1, 0)
    b_labels = [name + ", clip" for name in methods] + ["D, checkpoint\nchoice", "D, total change"]
    b = panel(fig, (3.53, .59, 1.78, 2.17), b_levels, b_labels,
              "(b) Clipping and checkpoint choice", "Change in energy score", [-.04, -.02, 0], (-.049, .009))
    for y, row in zip(b_levels, panel_b_rows(rows, selection)):
        seed_points(b, y, row["seed_changes"], METHOD_COLORS[row["method"]], seed_offset=.21)
    b.axvline(0, color=".36", linewidth=.65, zorder=0)
    for y, row in zip((5, 4, 3, 2), rows[:4]):
        seed_points(a, y, row["seed_movement"], METHOD_COLORS[row["method"]], seed_offset=.16)
    for y, row in zip((1, 0), rows[4:]):
        a.plot(row["movement"], y, "P", color=METHOD_COLORS[row["method"]], markersize=4)
    pairs = gaussian_reductions(rows)
    c = panel(fig, (left_x, .59, left_width, .68), (1, 0), ("Interpolation", "Regression"),
              "(c) Gaussian score changes", "Reduction after clipping", [0, .1, .2, .3, .4], (0, .46))
    c.set(ylim=(-.42, 1.48), xticklabels=["0", "0.1", "0.2", "0.3", "0.4"])
    c.set_axisbelow(True)
    for y, pair in zip((1, 0), pairs):
        target, spread = pair["target_reduction"], pair["half_pair_reduction"]
        c.plot([target, spread], [y, y], lw=1.4, color=".22", solid_capstyle="butt")
        c.plot(target, y, linestyle="none", marker="<", markeredgewidth=.5, markersize=3.8, color=TARGET_COLOR)
        c.plot(spread, y, linestyle="none", marker=">", markeredgewidth=.5, markersize=3.8, color=PAIR_COLOR)
        c.annotate(f"{pair['score_change']:+.3f}", ((target + spread) / 2, y),
                   xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=6.8)
    handles = [
        Line2D([], [], linestyle="none", marker="<", markeredgewidth=.5, markersize=3.8,
               color=TARGET_COLOR, label="Mean distance to observation"),
        Line2D([], [], linestyle="none", marker=">", markersize=4,
               color=PAIR_COLOR, label="½ mean distance between members"),
    ]
    fit_legend(fig, .94)
    center_c = c.get_position().x0 + c.get_position().width / 2
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(center_c, .006),
               fontsize=6.7, frameon=False, handlelength=.9, handletextpad=.1,
               labelspacing=.25, borderpad=.15, borderaxespad=0)
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
    font = font_manager.findfont(STYLE["font.family"][0], fallback_to_default=False)
    if Path(font).suffix.lower() != ".ttf":
        raise ValueError("DejaVu Serif TrueType is required; see docs/prigm_release.md")
    source = Path(__file__).resolve()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    rows, selection = load_rows(args.evidence, args.evidence_sha256)
    rows.append(gaussian_regression(args.evidence.parents[2]))
    manifest = {"schema_version": 5, "figure": "fig_operator", "protocol_sha256": PROTOCOL,
                "source": {"path": source.name, "sha256": source_hash},
                "style": {"path": STYLE_PATH.name, "sha256": STYLE_SHA256},
                "inputs": {"operator": {"path": "evidence/deadline/operator-mechanism.json",
                                        "sha256": args.evidence_sha256}, **GAUSSIAN_INPUTS},
                "plotted_rows": rows, "D_selection": selection, "gaussian_pairs": gaussian_reductions(rows),
                "panel_b": panel_b_rows(rows, selection), "seeds": [7, 13, 42],
                "scope": "Equal month weights. Clipping uses the raw-validation checkpoint. "
                         "Panel b also shows the effect of changing the D checkpoint and the total change. Three fitted seeds and their mean.",
                "axes": {"a": "log displacement in 60 TRAIN-standardized coordinates",
                         "b": "linear energy score change by fit: four clipping effects, then checkpoint change and total for D",
                         "c": "linear reductions from raw to clipped: left triangle for target distance, right triangle for half pair distance, signed score change above the gap"},
                "matplotlib_version": matplotlib.__version__, "output_sha256": {}}
    with plt.rc_context(PRINT_STYLE):
        fig = plot(rows, selection)
        try:
            args.output_dir.mkdir()
            for suffix in ("pdf", "png"):
                path = args.output_dir / f"fig_operator.{suffix}"
                with path.open("xb") as handle:
                    metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
                    fig.savefig(handle, format=suffix, dpi=200, metadata=metadata)
                manifest["output_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        finally:
            plt.close(fig)
    if (hashlib.sha256(source.read_bytes()).hexdigest() != source_hash
            or any(hashlib.sha256((args.evidence.parents[2] / data["path"]).read_bytes()).hexdigest()
                   != data["sha256"] for data in GAUSSIAN_INPUTS.values())
            or hashlib.sha256(args.evidence.read_bytes()).hexdigest() != args.evidence_sha256
            or hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest() != manifest["style"]["sha256"]):
        raise ValueError("figure source or input changed during rendering")
    with (args.output_dir / "fig_operator.manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
