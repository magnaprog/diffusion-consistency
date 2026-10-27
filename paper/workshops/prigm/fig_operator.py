"""Plot changes after applying the clipping map to the same magnetic predictions.

Displacement uses 60 standardized coordinates. The energy-score change is
its change in distance to measurements less its change in the spread term,
which is half the mean distance between distinct draws. Diffusion markers
retain individual seeds; the Gaussian method supplies one ensemble.
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
import numpy as np

STYLE_PATH = Path(__file__).with_name("prigm.mplstyle")
STYLE_SHA256 = hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest()
STYLE = matplotlib.rc_params_from_file(STYLE_PATH, use_default_template=False)
if hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest() != STYLE_SHA256:
    raise ValueError("style changed during import")
# Score components are distinct from the method colors used in other figures.
TERM_COLORS = ("#56B4E9", "#E69F00")
SUMMARY_COLOR = "#7B2CBF"

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
        rows.append({"method": name, "label": label, "movement": movement,
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
        raise ValueError("D fixed-weight effect differs between panels")
    return rows, selection


def plot(rows):
    fig, axes = plt.subplots(1, 2, figsize=(4.95, 2.30), sharey=True,
                             gridspec_kw={"width_ratios": (1, 1.35)})
    fig.subplots_adjust(left=.18, right=.98, bottom=.32, top=.86, wspace=.30)
    a, b = axes
    for index, row in enumerate(rows):
        a.plot(row["seed_movement"], [index] * len(row["seed_movement"]), "o", markersize=3,
               color=TERM_COLORS[0], markeredgecolor="#0072B2", markeredgewidth=0.4, zorder=2)
        a.plot(row["movement"], index, "D", markersize=3.5, color=SUMMARY_COLOR, zorder=3)
    a.set_xscale("log")
    a.set_xticks([0.1, 0.3, 1, 3], ["0.1", "0.3", "1", "3"])
    a.set_yticks(range(len(rows)), [row["label"] for row in rows])
    a.invert_yaxis()
    a.set_xlabel("Distance moved")
    a.set_title("(a) Mean displacement", loc="left")
    a.grid(axis="x", which="major")
    a.margins(x=0.12, y=0.12)

    y = np.arange(len(rows))
    b.axvline(0, color="0.45", linewidth=0.7, zorder=0)
    b.barh(y - 0.19, [r["delta_target"] for r in rows], height=0.36,
          color=TERM_COLORS[0], label="Distance to measurements")
    b.barh(y + 0.19, [r["negative_delta_dispersion"] for r in rows], height=0.36,
          color=TERM_COLORS[1], hatch="///", edgecolor="0.25", linewidth=0.4,
          label="Negative spread term")
    b.plot([r["delta_energy"] for r in rows], y, "D", color=SUMMARY_COLOR, markersize=3, label="Score change (b)")
    g5 = next(row for row in rows if row["method"] == "G5")
    value = g5["delta_energy"]
    label = f"{value:.5f}" if abs(value) < 0.001 else f"{value:.4f}"
    b.annotate(label, (value, list(METHODS).index("G5")),
               textcoords="offset points", xytext=(7, 0), va="center", fontsize=7, color="0.15")
    b.set_xscale("symlog", linthresh=0.005)
    b.set_xticks([-0.4, -0.02, 0, 0.02, 0.4], ["-0.4", "-0.02", "0", "0.02", "0.4"])
    b.set_xlabel("Clipped − raw")
    b.set_title("(b) Score components", loc="left")
    b.legend(frameon=False, fontsize=7, loc="lower center", ncol=3,
             bbox_to_anchor=(.55, .01), bbox_transform=fig.transFigure,
             columnspacing=1, handlelength=1.2)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, pad=2)
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
    manifest = {"schema_version": 1, "figure": "fig_operator", "protocol_sha256": PROTOCOL,
                "source": {"path": source.name, "sha256": source_hash},
                "style": {"path": STYLE_PATH.name, "sha256": STYLE_SHA256},
                "inputs": {"operator": {"path": args.evidence.name, "sha256": args.evidence_sha256}},
                "plotted_rows": rows, "D_selection": selection,
                "scope": "Same raw-selected weights and draws in both panels. Three fixed seeds, not uncertainty intervals. D_selection supplies the distinct selection effect for the caption, not a plotted postprocessing effect.",
                "axes": {"a": "logarithmic x; 60-dimensional TRAIN-standardized Euclidean movement",
                         "b": "symmetric-log x, linear within +/-0.005; energy = target - dispersion"},
                "matplotlib_version": matplotlib.__version__, "output_sha256": {}}
    with plt.rc_context(STYLE):
        fig = plot(rows)
        try:
            args.output_dir.mkdir()
            for suffix in ("pdf", "png"):
                path = args.output_dir / f"fig_operator.{suffix}"
                with path.open("xb") as handle:
                    fig.savefig(handle, format=suffix, dpi=200)
                manifest["output_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        finally:
            plt.close(fig)
    if (hashlib.sha256(source.read_bytes()).hexdigest() != source_hash
            or hashlib.sha256(args.evidence.read_bytes()).hexdigest() != args.evidence_sha256
            or hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest() != manifest["style"]["sha256"]):
        raise ValueError("figure source or input changed during rendering")
    with (args.output_dir / "fig_operator.manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
