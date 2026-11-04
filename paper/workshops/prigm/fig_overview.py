"""Plot checkpoint-policy contrasts and marginal coverage from frozen evidence.

No predictions are recomputed. Existing checkpoint and calibration loaders
validate the records; every displayed coordinate is included in the manifest.
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


checkpoints, checkpoint_source = load_producer("fig_checkpoints")
calibration, calibration_source = load_producer("fig_calibration")
DEPENDENCY_SOURCES = {"checkpoints": checkpoint_source, "calibration": calibration_source}
STYLE_PATH, STYLE, STYLE_SHA256 = calibration.STYLE_PATH, calibration.STYLE, calibration.STYLE_SHA256
SEEDS, METHODS, CHANNELS = checkpoints.SEEDS, calibration.METHODS, calibration.CHANNELS
SEED_COLOR = "#9C27E8"
SEED_MARKERS = ("o", "s", "^")
METHOD_MARKERS = ("o", "s", "^", "D")
FIGURE_INCHES = (5.5, 2.3)


def load_overview(evidence_dir, manifest_sha256, calibration_path, calibration_sha256):
    """Return authenticated input bindings and the complete plotted coordinates."""
    transform = checkpoints.read(evidence_dir / "MANIFEST.json", manifest_sha256)
    checkpoints.require(
        transform["source_protocol_sha256"] == checkpoints.PROTOCOL
        and transform["source_scope_sha256"] == checkpoints.SCOPE
        and set(transform["files"]) == set(checkpoints.FILES), "public export inventory differs")
    records = {name: checkpoints.read(evidence_dir / name, transform["files"][name])
               for name in checkpoints.FILES}
    pairs = {"summary": "SUMMARY.json", "saved_val25": "SAVED_VAL25.json", "selection": "SELECTION.json"}
    pins = {key: transform["original_sha256"][name] for key, name in pairs.items()}
    trajectories = checkpoints.export(*(records[name] for name in pairs.values()), pins)
    policies = trajectories["policies"]
    checkpoints.require(all(row["P25"] == row["P50"] for row in policies.values()),
                        "overview requires the recorded P25/P50 agreement")
    score_by_arm = {}
    for arm in checkpoints.ARMS:
        score_by_arm[arm] = [[trajectories["curves"]["TEST50"][f"{arm}_s{seed}"][
            checkpoints.UPDATES.index(policies[f"{arm}_s{seed}"][policy])]
            for policy in ("F", "P25")] for seed in SEEDS]
    differences = np.asarray(score_by_arm["O-S"]) - np.asarray(score_by_arm["D"])
    panel_a = {"policy_order": ["F", "P25"], "x": [0, 1], "seeds": list(SEEDS),
               "score_by_arm": score_by_arm, "seedwise": differences.tolist(),
               "mean": differences.mean(axis=0).tolist(), "policies": policies,
               "validation_choices_agree": True}
    tails, provenance = calibration.load_calibration(calibration_path, calibration_sha256)
    panel_b = {}
    for index, name in enumerate(METHODS):
        row = tails[name]
        positions = np.arange(5) + (index - 1.5) * .15
        panel_b[name] = {
            "fixed_model_seeds": row["fixed_model_seeds"],
            "mean_percent": np.asarray(row["mean_percent"])[:, 1].tolist(),
            "model_means_percent": np.asarray(row["model_means_percent"])[:, :, 1].tolist(),
            "y": positions.tolist(),
            "individual_y": [(positions + (seed_index - 1) * .028).tolist()
                             for seed_index in range(3)] if len(row["fixed_model_seeds"]) == 3 else []}
    inputs = {name: {"path": "evidence/followup/checkpoints/" + name, "sha256": digest}
              for name, digest in {**transform["files"], "MANIFEST.json": manifest_sha256}.items()}
    inputs["calibration"] = {"path": "evidence/followup/calibration/summary.json",
                             "sha256": calibration_sha256}
    return {"inputs": inputs, "panel_a": panel_a, "panel_b": panel_b,
            "calibration_provenance": provenance}


def plot(record):
    a_data, b_data = record["panel_a"], record["panel_b"]
    fig, (a, b) = plt.subplots(1, 2, figsize=FIGURE_INCHES,
                              gridspec_kw={"width_ratios": [1, 1.22]})
    fig.subplots_adjust(left=.12, right=.985, bottom=.32, top=.88, wspace=.52)
    for seed, marker, values in zip(SEEDS, SEED_MARKERS, a_data["seedwise"]):
        a.plot(a_data["x"], values, color=".65", linewidth=.7,
               marker=marker, markerfacecolor=SEED_COLOR, markeredgecolor=SEED_COLOR,
               markersize=3.8, label=f"Seed {seed}", gid=f"seed_{seed}")
    a.plot(a_data["x"], a_data["mean"], linestyle="none", marker="D", color="black",
           markeredgecolor="white", markeredgewidth=.5, markersize=5, label="Mean", gid="mean")
    for x, value in zip(a_data["x"], a_data["mean"]):
        a.annotate(f"{value:.4f}", (x, value), xytext=(6, -13 if x == 0 else 8),
                   textcoords="offset points", fontsize=7)
    a.axhline(0, color=".5", linewidth=.7, gid="zero")
    a.set_xticks([0, 1], ["Final update", "Validation\nchoices"])
    a.set(xlim=(-.22, 1.55), ylim=(-.105, .09), yticks=[-.10, -.05, 0, .05],
          ylabel="O-S minus D energy score")
    a.set_title("(a) Checkpoint choice", loc="left")
    a.legend(frameon=False, ncol=2, fontsize=7, loc="upper left",
             bbox_to_anchor=(-.15, -.24), columnspacing=.7, handlelength=1.4)
    for index, (name, label, marker) in enumerate(zip(METHODS, calibration.LABELS, METHOD_MARKERS)):
        row, color = b_data[name], calibration.COLORS[index]
        for seed, values, y in zip(SEEDS, row["model_means_percent"], row["individual_y"]):
            b.scatter(values, y, s=7, facecolors="none", edgecolors=color, linewidths=.45,
                      alpha=.7, zorder=2, gid=f"{name}_seed_{seed}")
        b.scatter(row["mean_percent"], row["y"], s=16, marker=marker, color=color,
                  label=label, zorder=3, gid=f"{name}_mean")
    b.axvline(90, color=".4", linestyle="--", linewidth=.7, gid="coverage_reference")
    b.set(xlim=(0, 103), xticks=[0, 50, 90], ylim=(4.5, -.6),
          xlabel="Nominal 90% coverage (%)")
    b.set_yticks(range(5), [r"$\mu_x$", r"$\mu_y$", r"$\mu_z$", "$m$", "$v_s$"])
    b.set_title("(b) Raw predictions", loc="left")
    b.legend(frameon=False, ncol=2, fontsize=7, loc="upper left",
             bbox_to_anchor=(-.16, -.24), columnspacing=.7, handletextpad=.25)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, pad=2)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--calibration-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError("figure output directory must be fresh")
    source_hash = sha256(__file__)
    checkpoints.require(sha256(STYLE_PATH) == STYLE_SHA256, "style changed since import")
    font = font_manager.findfont(STYLE["font.family"][0], fallback_to_default=False)
    checkpoints.require(Path(font).suffix.lower() == ".ttf", "DejaVu Serif TrueType is required")
    record = load_overview(args.evidence_dir, args.manifest_sha256,
                           args.calibration, args.calibration_sha256)
    manifest = {
        "schema_version": 1, "figure": "fig_overview",
        "protocol_sha256": checkpoints.PROTOCOL, "scope_sha256": checkpoints.SCOPE,
        "source": {"path": Path(__file__).name, "sha256": source_hash},
        "dependency_sources": DEPENDENCY_SOURCES,
        "style": {"path": STYLE_PATH.name, "sha256": STYLE_SHA256}, **record,
        "figure_inches": list(FIGURE_INCHES), "channels": list(CHANNELS),
        "method_order": list(METHODS), "colors": dict(zip(METHODS, calibration.COLORS)),
        "seed_color": SEED_COLOR, "seed_markers": dict(zip(map(str, SEEDS), SEED_MARKERS)),
        "method_markers": dict(zip(METHODS, METHOD_MARKERS)),
        "axes": {
            "a": "linear; O-S minus D energy score in 60 TRAIN-standardized coordinates; "
                 "one paired contrast per fixed seed, black diamonds average three seeds; "
                 "F is update 29636; P25 uses separate raw-VAL choices for D and O-S; P25=P50",
            "b": "linear percent; marginal interval coverage over 12 hidden positions per "
                 "window, then equal months; filled diffusion markers average three fixed fits; "
                 "open circles show individual fits; Gaussian regression uses one ensemble "
                 "realization per window and no averaging across model seeds"},
        "interval_order_statistics": [2, 38], "n_draws": 39, "n_windows": 480,
        "coverage_reference": {"percent": 90, "assumption": "at each time and coordinate, "
                               "the observation and 39 ensemble members are exchangeable, with no ties"},
        "pointwise_intervals_plotted": False,
        "scope": "Existing, previously examined TEST period. Equal months; each fixed model "
                 "retains its own 39 draws. No pooled samples, confidence intervals, new fits, "
                 "or uncertainty from repeated selection. Connecting lines identify paired "
                 "seed contrasts under two policies, not a causal effect or training trajectory.",
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
    checkpoints.require(sha256(__file__) == source_hash and sha256(STYLE_PATH) == STYLE_SHA256
                        and all(sha256(PAPER / row["path"]) == row["sha256"]
                                for row in DEPENDENCY_SOURCES.values()), "source changed during rendering")
    for name, row in record["inputs"].items():
        path = args.calibration if name == "calibration" else args.evidence_dir / name
        checkpoints.require(sha256(path) == row["sha256"], "input changed during rendering")
    with (args.output_dir / "fig_overview.manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
