"""Export and plot complete, hash-bound retrospective checkpoint trajectories."""
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
import numpy as np

PROTOCOL = "eeca7c2e14d982f0b3e972256cc9acf7a5aaa8dc03da6a9299a708c78a7b5946"
SCOPE = "5210fbe5432aec70433ed9d7c10a1ba61503e5c7c15e43cc2fa37bfac478674a"
ARMS, SEEDS = ("D", "O-S"), (7, 13, 42)
UPDATES = (2868, 9560, 16252, 22944, 29636)
NAMES = tuple(f"{arm}_s{seed}" for arm in ARMS for seed in SEEDS)
KEYS = {f"{name}-g{g}" for name in NAMES for g in UPDATES}
FILES = ("PROTOCOL.json", "SCOPE.json", "SELECTION.json", "SAVED_VAL25.json", "SUMMARY.json")
STYLE_PATH = Path(__file__).with_name("prigm.mplstyle")
STYLE_SHA256 = hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest()
STYLE = matplotlib.rc_params_from_file(STYLE_PATH, use_default_template=False)
if hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest() != STYLE_SHA256:
    raise ValueError("style changed during import")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path, expected):
    payload = Path(path).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == expected, "input SHA256 mismatch")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(payload, object_pairs_hook=unique)


def public_export(study, output):
    """Replace declared root prefixes and environment fields; preserve all numbers."""
    pins = {name: sha(study / name) for name in FILES}
    require(pins["PROTOCOL.json"] == PROTOCOL and pins["SCOPE.json"] == SCOPE, "unrecognized frozen study")
    records = {name: read(study / name, pins[name]) for name in FILES}
    protocol = records["PROTOCOL.json"]
    replacements = dict(sorted({str(study.resolve()): "checkpoint-study", protocol["original_root"]: "original-study",
                                protocol["extension_root"]: "fixed-final-study"}.items(), key=lambda kv: -len(kv[0])))
    fields = {("PROTOCOL.json", "environment", "executable"): (protocol["environment"]["executable"], "environment-python"),
              ("PROTOCOL.json", "environment", "hostname"): (protocol["environment"]["hostname"], "execution-host")}
    def replace(value, reverse=False, trail=()):
        if trail in fields:
            old, new = fields[trail]
            require(value == (new if reverse else old), "environment field differs")
            return old if reverse else new
        if isinstance(value, dict):
            pairs = [(replace(k, reverse), replace(v, reverse, trail + (k,))) for k, v in value.items()]
            require(len({k for k, _ in pairs}) == len(pairs), "replacement key collision")
            return dict(pairs)
        if isinstance(value, list):
            return [replace(v, reverse) for v in value]
        if isinstance(value, str):
            for old, new in replacements.items():
                if reverse: old, new = new, old
                if value == old or value.startswith(old + "/"):
                    return new + value[len(old):]
            require(not value.startswith("/"), "unknown absolute path; export stopped")
        return value
    require(not output.exists() and not output.is_symlink(), "fresh public export required")
    output.mkdir(parents=True)
    public_pins = {}
    for name, record in records.items():
        transformed = replace(record, trail=(name,))
        # Reversing the declared replacements must restore the original record.
        require(replace(transformed, reverse=True, trail=(name,)) == record,
                "export changed content beyond declared replacements")
        with (output / name).open("x") as handle:
            json.dump(transformed, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
        public_pins[name] = sha(output / name)
    require(all(sha(study / name) == pin for name, pin in pins.items()), "source changed during export")
    manifest = {"schema_version": 1, "analysis": "checkpoint_followup_public_export",
                "source_protocol_sha256": PROTOCOL, "source_scope_sha256": SCOPE,
                "source": {"path": Path(__file__).name, "sha256": sha(__file__)},
                "files": public_pins,
                "original_sha256": pins,
                "transformations": {
                    "root_replacements": [{"original_root_sha256": hashlib.sha256(k.encode()).hexdigest(), "public_root": v}
                                          for k, v in replacements.items()],
                    "field_replacements": [{"file": trail[0], "field": ".".join(trail[1:]),
                                            "original_value_sha256": hashlib.sha256(old.encode()).hexdigest(), "public_value": new}
                                           for trail, (old, new) in fields.items()]}}
    with (output / "MANIFEST.json").open("x") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return manifest


def reduce_curves(rows, ids, months, count):
    require(set(rows) == KEYS and len(ids) == len(set(ids)) == count, "incomplete trajectory population")
    m = np.asarray(months)
    require(m.tolist() == [int(w[:4]) * 100 + int(w[5:7]) for w in ids], "month/ID mismatch")
    curves = {}
    for name in NAMES:
        x = np.asarray([rows[f"{name}-g{g}"] for g in UPDATES])
        require(x.dtype.kind in "fi" and x.shape == (5, count) and np.isfinite(x).all(), "invalid raw scores")
        curves[name] = np.stack([x[:, m == month].mean(axis=1) for month in np.unique(m)], axis=1).mean(axis=1).tolist()
    return curves


def export(summary, saved, selection, pins):
    require(all(r["protocol_sha256"] == PROTOCOL for r in (summary, saved, selection)), "protocol differs")
    require(saved["scope_sha256"] == selection["scope_sha256"] == SCOPE, "scientific scope differs")
    require(summary["selection_sha256"] == pins["selection"], "summary/selection binding differs")
    require(pins["saved_val25"] in summary["input_sha256"].values(), "saved VAL25 is not bound by summary")
    require(summary["policies"] == selection["policies"] and set(selection["policies"]) == set(NAMES), "policy receipt differs")
    require(selection["test_jobs"] == [[name, g] for name in NAMES for g in UPDATES], "incomplete TEST authorization")
    require(len(selection["test_checkpoint_hashes"]) == len(set(selection["test_checkpoint_hashes"])) == 30,
            "thirty distinct checkpoints required")
    curves = {"VAL25": reduce_curves(saved["scores"], saved["ids"], saved["months"], 319)}
    require(set(summary["scores"]) == {"VAL", "TEST"}, "both complete splits required")
    for split, count in (("VAL", 319), ("TEST", 480)):
        row = summary["scores"][split]
        reduced = reduce_curves(row["data"], row["ids"], row["months"], count)
        require(set(summary["curves"][split]) == set(NAMES), "missing reported curve")
        for name in NAMES:
            require(np.allclose(reduced[name], summary["curves"][split][name], rtol=0, atol=1e-12), "score reduction differs")
        curves[split + "50"] = reduced
    require(saved["ids"] == summary["scores"]["VAL"]["ids"]
            and saved["months"] == summary["scores"]["VAL"]["months"], "VAL budgets use different populations")
    for name in NAMES:
        expected = {"P25": UPDATES[int(np.argmin(curves["VAL25"][name]))],
                    "P50": UPDATES[int(np.argmin(curves["VAL50"][name]))], "F": UPDATES[-1]}
        require(selection["policies"][name] == expected, "policies do not reproduce from VAL alone")
    return {"schema_version": 1, "analysis": "retrospective_checkpoint_trajectories",
            "protocol_sha256": PROTOCOL, "scope_sha256": SCOPE, "input_sha256": pins,
            "arms": list(ARMS), "seeds": list(SEEDS), "updates": list(UPDATES),
            "curves": curves, "policies": selection["policies"],
            "policy_means": summary["policy_means"], "contrast_changes": summary["contrast_changes"],
            "common_update_TEST_contrasts": summary["common_update_TEST_contrasts"],
            "final_minus_selected_TEST": summary["final_minus_selected_TEST"],
            "validation_budget_comparison": summary["validation_budget_comparison"],
            "VAL25_selection_diagnostics": saved["curves"],
            "VAL50_selection_diagnostics": summary["VAL50_selection_diagnostics"],
            "scores": {"VAL25": {"ids": saved["ids"], "months": saved["months"], "data": saved["scores"]},
                       **{s + "50": summary["scores"][s] for s in ("VAL", "TEST")}},
            "scope": "Raw month-balanced energy scores; three fixed models per arm, all five milestones. "
                     "P25/P50 selected solely on complete VAL25/VAL50; F is update 29636. "
                     "TEST is historically exposed and no TEST selection is performed. "
                     "Fixed original weights and initial noise isolate budget/policy dependence; "
                     "training causes and Monte Carlo selection uncertainty remain unidentified."}


def plot(record):
    x = np.asarray(UPDATES) / 1000
    styles = (("o", "-"), ("s", "--"), ("^", ":"))
    colors = STYLE["axes.prop_cycle"].by_key()["color"][:2]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.7), sharey=True, layout="constrained")
        for ax, panel, title in zip(axes, ("VAL25", "VAL50", "TEST50"),
                                   ("(a) Validation\n25 evaluations", "(b) Validation\n50 evaluations", "(c) Test\n50 evaluations")):
            for arm, color in zip(ARMS, colors):
                for seed, (marker, line) in zip(SEEDS, styles):
                    name = f"{arm}_s{seed}"
                    y = record["curves"][panel][name]
                    ax.plot(x, y, color=color, marker=marker, linestyle=line, markersize=3,
                            markerfacecolor="white" if arm == "D" else color,
                            markeredgewidth=.7, linewidth=.9)
                    policies = ("P25",) if panel == "VAL25" else ("P50",) if panel == "VAL50" else ("P25", "P50")
                    for policy in policies:
                        i = UPDATES.index(record["policies"][name][policy])
                        ax.plot(x[i], y[i], marker="o" if policy == "P25" else "x", color="black",
                                markerfacecolor="none", linestyle="none", markersize=6 if policy == "P25" else 4,
                                markeredgewidth=.65, zorder=5)
            ax.set_title(title, loc="left")
            ax.set_xticks(x, ["2.9", "9.6", "16.3", "22.9", "29.6"])
            ax.text(x[-1], 1.01, "F", transform=ax.get_xaxis_transform(), ha="center", fontsize=7)
            ax.grid(axis="y")
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(length=2, pad=2)
            ax.margins(x=.07, y=.10)
        axes[0].set_ylabel("Raw energy score")
        fig.supxlabel("Training steps (thousands)", fontsize=8)
        handles = [Line2D([], [], color=c, marker="o", label=a, linewidth=1.2,
                          markerfacecolor="white" if a == "D" else c)
                   for a, c in zip(ARMS, colors)]
        handles += [Line2D([], [], color=".4", marker=m, linestyle=ls, markersize=3, label=f"Seed {s}")
                    for s, (m, ls) in zip(SEEDS, styles)]
        handles += [Line2D([], [], color="black", marker="o", markerfacecolor="none", linestyle="none", markersize=6, label="P25"),
                    Line2D([], [], color="black", marker="x", linestyle="none", markersize=4, label="P50"),
                    Line2D([], [], linestyle="none", label="F: final")]
        # Matplotlib fills legend columns before rows. Group methods and
        # selection policies above the seed keys and final-update label.
        handles = [handles[i] for i in (0, 2, 1, 3, 5, 4, 6, 7)]
        fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False, fontsize=7)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-study-root", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest() != STYLE_SHA256:
        raise ValueError("style changed since import")
    require(not args.output_dir.exists() and not args.output_dir.is_symlink(), "fresh output directory required")
    if args.export_study_root is not None:
        require(args.evidence_dir is None, "choose public export or figure")
        public_export(args.export_study_root, args.output_dir)
        return 0
    require(args.evidence_dir is not None and args.manifest_sha256 is not None, "pinned public evidence required")
    transform = read(args.evidence_dir / "MANIFEST.json", args.manifest_sha256)
    require(transform["source_protocol_sha256"] == PROTOCOL and transform["source_scope_sha256"] == SCOPE
            and set(transform["files"]) == set(FILES), "public export inventory differs")
    records = {name: read(args.evidence_dir / name, transform["files"][name]) for name in FILES}
    pairs = {"summary": "SUMMARY.json", "saved_val25": "SAVED_VAL25.json", "selection": "SELECTION.json"}
    pins = {key: transform["original_sha256"][name] for key, name in pairs.items()}
    exported = export(*(records[name] for name in pairs.values()), pins)
    font = font_manager.findfont(STYLE["font.family"][0], fallback_to_default=False)
    if Path(font).suffix.lower() != ".ttf":
        raise ValueError("DejaVu Serif TrueType is required; see docs/prigm_release.md")
    source_hash = sha(__file__)
    style_hash = STYLE_SHA256
    args.output_dir.mkdir(parents=True)
    fig = plot(exported)
    try:
        with plt.rc_context(STYLE):
            for suffix in ("pdf", "png"):
                with (args.output_dir / f"fig_checkpoints.{suffix}").open("xb") as handle:
                    fig.savefig(handle, format=suffix, dpi=200)
    finally:
        plt.close(fig)
    require(sha(__file__) == source_hash
            and sha(STYLE_PATH) == style_hash
            and sha(args.evidence_dir / "MANIFEST.json") == args.manifest_sha256
            and all(sha(args.evidence_dir / k) == v for k, v in transform["files"].items()),
            "source or input changed during export")
    manifest = {"schema_version": 1, "figure": "fig_checkpoints", "protocol_sha256": PROTOCOL,
                "source": {"path": Path(__file__).name, "sha256": source_hash},
                "style": {"path": STYLE_PATH.name, "sha256": style_hash},
                "evidence_source": transform["source"],
                "inputs": {k: {"path": "evidence/followup/checkpoints/" + k, "sha256": v}
                           for k, v in {**transform["files"], "MANIFEST.json": args.manifest_sha256}.items()},
                "curves": exported["curves"], "policies": exported["policies"],
                "caption": "Raw month-balanced energy score at all five saved updates, for D and O-S. "
                           "D uses open markers; O-S uses filled markers. "
                           "Circles/solid, squares/dashed, triangles/dotted identify seeds 7, 13, 42. "
                           "Black hollow rings mark P25 and black crosses P50; F marks the final update. "
                           "TEST displays the policies frozen from VAL, with no TEST selection. "
                           "Lines connect observed checkpoints and are not fitted curves or uncertainty intervals. "
                           "VAL and TEST contain different windows; their score levels are not a paired population contrast.",
                "matplotlib_version": matplotlib.__version__,
                "output_sha256": {p.name: sha(p) for p in sorted(args.output_dir.iterdir())}}
    with (args.output_dir / "fig_checkpoints.manifest.json").open("x") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
