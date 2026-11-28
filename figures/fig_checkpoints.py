"""Export checkpoint records or plot raw scores, TEST differences, and validation choices.

The plot checks the input hashes and complete validation and TEST score grids.
Validation scores determine the two choices compared for each fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.textpath import TextToPath
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


def checkpoint_display(record):
    """Pair TEST scores at common updates and list choices made on validation."""
    policies = record["policies"]
    require(all(policies[name]["P25"] == policies[name]["P50"] for name in NAMES),
            "one validation choice per fit requires coincident VAL25 and VAL50 selections")
    curves = record["curves"]["TEST50"]
    return {
        "updates": list(UPDATES),
        "test50_os_minus_d": {
            str(seed): np.subtract(curves[f"O-S_s{seed}"], curves[f"D_s{seed}"]).tolist()
            for seed in SEEDS
        },
        "validation_choices": [
            {"seed": seed, **{arm: policies[f"{arm}_s{seed}"]["P25"] for arm in ARMS}}
            for seed in SEEDS
        ],
    }


def manuscript_table_font():
    """Use the same regular Type 1 face as the workshop's Times tables."""
    path = subprocess.check_output(["kpsewhich", "utmr8a.pfb"], text=True).strip()
    require(path and sha(path) == "2ef9d47303d25f3c9553a43255dae8c39160e130ad5ed34444e39dee03d796a1",
            "TeX Live NimbusRomNo9L-Regu (utmr8a.pfb) is required")
    return font_manager.FontProperties(fname=path)


# Paul Tol's blue and pink are the manuscript colors for D and O-S.
METHOD_COLORS = {"D": "#0077BB", "O-S": "#EE3377"}
DIFFERENCE_COLOR = "#EE7733"
STAGES = (("VAL25", "Validation, 25 evaluations"), ("VAL50", "Validation, 50 evaluations"),
          ("TEST50", "TEST, 50 evaluations"))
SEED_MARKERS = ("o", "s", "^")


def step_ticks(axis, steps, show_labels=True):
    axis.set_xlim(1.4, 31.2)
    axis.set_xticks(steps, [f"{step:.1f}" for step in steps])
    axis.tick_params(axis="x", labelbottom=show_labels, length=2, pad=1.5, labelsize=6.3)


# The open circle encloses the square, and the white-edged triangle stays visible inside it.
RAW_STYLES = (
    dict(linestyle="-", markersize=8.4, markerfacecolor="none", markeredgewidth=.8),
    dict(linestyle="--", markersize=4.8, markeredgewidth=.5),
    dict(linestyle="-.", markersize=3.6, markeredgecolor="white", markeredgewidth=.65),
)


def raw_score_grid(fig, curves, steps):
    left, right, top = .109, .984, .864
    width, height = (right - left - 2 * .030) / 3, (top - .468 - .050) / 2
    first_axis = None
    for row, arm in enumerate(ARMS):
        bottom = top - (row + 1) * height - row * .050
        for column, (stage, title) in enumerate(STAGES):
            axis = fig.add_axes([left + column * (width + .030), bottom, width, height],
                                sharey=first_axis, label=f"{stage}_{arm}")
            if first_axis is None:
                first_axis = axis
            for seed, marker, style in zip(SEEDS, SEED_MARKERS, RAW_STYLES):
                name = f"{arm}_s{seed}"
                axis.plot(steps, curves[stage][name], label=f"Seed {seed}", gid=f"{stage}_{name}",
                          color=METHOD_COLORS[arm], marker=marker, linewidth=.9, **style)
            step_ticks(axis, steps, show_labels=row == 1)
            axis.set(xlim=(1.0, 31.5), ylim=(1.45, 2.75), yticks=[1.6, 2.0, 2.4])
            axis.tick_params(axis="y", labelleft=column == 0, length=2, pad=2, labelsize=6.4)
            if column == 0:
                axis.set_ylabel(arm, fontsize=8.0, labelpad=3, rotation=0, va="center", ha="right")
            if row == 0:
                axis.set_title(title, fontsize=7.2, pad=4)
            if row == 1 and column == 1:
                axis.set_xlabel("Optimizer steps (thousands)", fontsize=7, labelpad=2)
    first_axis.text((left + right) / 2, .935, "(a) Raw energy scores during training", transform=fig.transFigure,
                    ha="center", va="center", fontsize=8.1)
    seeds = [Line2D([], [], label=f"Seed {seed}", color=".25", marker=marker, linewidth=.9, **style)
             for seed, marker, style in zip(SEEDS, SEED_MARKERS, RAW_STYLES)]
    fig.legend(handles=seeds, loc="center", bbox_to_anchor=(.5, .022),
               frameon=False, ncol=3, fontsize=6.8, borderpad=0,
               handletextpad=.45, columnspacing=1.1)


def selection_table(fig, record, display):
    font = manuscript_table_font()
    left, right = .575, .990
    def table_y(position):
        return .103 + position * .287

    def text(x, y, label, align="right", size=7.1):
        fig.text(x, table_y(y), label, ha=align, fontproperties=font, fontsize=size)

    step_group = (.638, .847)
    score_group = (.859, right)
    text(sum(step_group) / 2, .72, "Training step", "center")
    text(sum(score_group) / 2, .72, "TEST score", "center")
    for start, end, y, weight in (
        (left, right, .86, .8), (*step_group, .675, .34), (*score_group, .675, .34),
        (left, right, .55, .52), (left, right, .012, .8),
    ):
        fig.add_artist(Line2D([start, end], [table_y(y)] * 2, transform=fig.transFigure,
                              color="black", linewidth=weight, solid_capstyle="butt"))
    seed_left = left + .008
    step_width = step_group[1] - step_group[0]
    centers = (step_group[0] + step_width / 4, step_group[1] - step_width / 4, sum(score_group) / 2)
    text(seed_left, .59, "Seed", "left")
    for x, heading in zip(centers, (*ARMS, "O-S minus D")):
        text(x, .59, heading, "center")
    rows = []
    for row in display["validation_choices"]:
        seed = row["seed"]
        selected = {arm: record["curves"]["TEST50"][f"{arm}_s{seed}"][UPDATES.index(row[arm])]
                    for arm in ARMS}
        difference = selected["O-S"] - selected["D"]
        rows.append((str(seed), f"{row['D']:,}", f"{row['O-S']:,}", f"{difference:+.3f}".replace("-", "−")))

    # Center each right-aligned block of numbers under its heading using the printed font metrics.
    font.set_size(7.1)
    metrics = TextToPath()
    right_edges = []
    for column, center in enumerate(centers, 1):
        width = max(metrics.get_text_width_height_descent(row[column], font, False)[0] for row in rows)
        right_edges.append(center + width / (2 * 72 * fig.get_figwidth()))
    for cells, y in zip(rows, (.445, .265, .085)):
        text(seed_left, y, cells[0], "left")
        for x, cell in zip(right_edges, cells[1:]):
            text(x, y, cell)


def plot(record):
    """Compare raw trajectories, same-step TEST differences, and selected-step TEST scores."""
    display = checkpoint_display(record)
    steps = np.asarray(display["updates"]) / 1000
    # The open circle allows the triangle to remain visible at near-coincident points.
    comparison_styles = (
        dict(linestyle="-", markersize=8.4, markerfacecolor="none", markeredgewidth=.8),
        dict(linestyle="--", markersize=3.5),
        dict(linestyle="-.", markersize=4.7, markeredgecolor="white", markeredgewidth=.55),
    )
    with plt.rc_context(plt.rcParamsDefault), plt.rc_context(STYLE):
        fig = plt.figure(figsize=(5.5, 4.45))
        raw_score_grid(fig, record["curves"], steps)
        axis = fig.add_axes([.109, .103, .408, .252], label="TEST50_contrast")
        for seed, marker, style in zip(SEEDS, SEED_MARKERS, comparison_styles):
            axis.plot(steps, display["test50_os_minus_d"][str(seed)], color=DIFFERENCE_COLOR,
                      marker=marker, linewidth=1, label=f"Seed {seed}", gid=f"TEST50_s{seed}", **style)
        axis.axhline(0, color=".27", linewidth=.65, zorder=0)
        axis.set(ylim=(-.325, .16), yticks=[-.3, -.2, -.1, 0, .1])
        axis.tick_params(axis="y", labelsize=6.4, length=2, pad=2)
        step_ticks(axis, steps)
        axis.set_title("(b) TEST differences at the same step", fontsize=8.1, pad=4)
        # Use the same heading face and baseline on both lower panels. Table cells retain the manuscript face.
        heading_y = axis.get_position().y1 + 4 / (72 * fig.get_figheight())
        axis.text((.575 + .990) / 2, heading_y, "(c) TEST at selected steps", transform=fig.transFigure,
                  ha="center", va="baseline", fontsize=8.1)
        axis.set_xlabel("Optimizer steps (thousands)", fontsize=7, labelpad=2)
        axis.set_ylabel("O-S minus D", fontsize=7.3, labelpad=2)
        selection_table(fig, record, display)
    return fig


def save_table_pdf(fig, output):
    """Embed the manuscript face as Type 1 text beside the Matplotlib plot."""
    width, height = fig.get_size_inches() * 72
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for text in fig.texts:
            text.set_visible(False)
        try:
            fig.savefig(folder / "plot.pdf", dpi=200, metadata={"CreationDate": None, "ModDate": None})
        finally:
            for text in fig.texts:
                text.set_visible(True)

        # The table uses TeX's 8r Times mapping, including its mathematical minus at slot 12.
        # All positions and strings come from the same artists used for the native PNG.
        cells = []
        for text in fig.texts:
            x, y = text.get_position()
            label = text.get_text().replace("−", r"\char12{}")
            cell = rf"\font\tablefont=ptmr8r at {text.get_fontsize()}bp\tablefont {label}"
            if text.get_ha() in ("right", "center"):
                cell = r"\hss" + cell
            if text.get_ha() in ("left", "center"):
                cell += r"\hss"
            cells.append(rf"\rlap{{\kern{x * width:.6f}bp\raise{y * height:.6f}bp"
                         rf"\hbox to0pt{{{cell}}}}}%")
        source = [
            r"\pdfinfoomitdate=1 \pdftrailerid{} \pdfsuppressptexinfo=15",
            r"\input glyphtounicode.tex \pdfgentounicode=1",
            rf"\pdfpagewidth={width:.6f}bp \pdfpageheight={height:.6f}bp",
            r"\pdfhorigin=0bp \pdfvorigin=0bp",
            r"\pdfximage{plot.pdf}\setbox0=\hbox{\pdfrefximage\pdflastximage}",
            r"\shipout\hbox to\wd0{\rlap{\copy0}\pdfliteral direct{0.15 g}%",
            *cells, r"\hss}\end",
        ]
        (folder / "table.tex").write_text("\n".join(source) + "\n", encoding="ascii")
        subprocess.run(["pdftex", "-no-shell-escape", "-halt-on-error", "-interaction=batchmode", "table.tex"],
                       cwd=folder, check=True)
        output.write((folder / "table.pdf").read_bytes())


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
                    if suffix == "pdf":
                        save_table_pdf(fig, handle)
                    else:
                        fig.savefig(handle, format="png", dpi=200, metadata={"Software": "PriGM fig_checkpoints"})
    finally:
        plt.close(fig)
    require(sha(__file__) == source_hash
            and sha(STYLE_PATH) == style_hash
            and sha(args.evidence_dir / "MANIFEST.json") == args.manifest_sha256
            and all(sha(args.evidence_dir / k) == v for k, v in transform["files"].items()),
            "source or input changed during export")
    manifest = {"schema_version": 2, "figure": "fig_checkpoints", "protocol_sha256": PROTOCOL,
                "source": {"path": Path(__file__).name, "sha256": source_hash},
                "style": {"path": STYLE_PATH.name, "sha256": style_hash},
                "evidence_source": transform["source"],
                "inputs": {k: {"path": "evidence/followup/checkpoints/" + k, "sha256": v}
                           for k, v in {**transform["files"], "MANIFEST.json": args.manifest_sha256}.items()},
                "curves": exported["curves"], "policies": exported["policies"],
                "display": checkpoint_display(exported),
                "caption": "Each method row shows three seeds on validation at 25 and 50 evaluations per sample and on TEST at 50. "
                           "The lower plot compares O-S minus D on TEST at the same step and seed. Negative values "
                           "favor O-S. The table compares TEST scores at the steps with the lowest validation score "
                           "for each fit. Both validation budgets give the same choices. TEST was previously examined.",
                "matplotlib_version": matplotlib.__version__,
                "output_sha256": {p.name: sha(p) for p in sorted(args.output_dir.iterdir())}}
    with (args.output_dir / "fig_checkpoints.manifest.json").open("x") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
