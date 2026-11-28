"""Plot casewise finite HMM scores, continuous residuals and decoded path rates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
import numpy as np


STYLE_PATH = Path(__file__).with_name("prigm.mplstyle")
STYLE_SHA256 = hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest()
STYLE = matplotlib.rc_params_from_file(STYLE_PATH, use_default_template=False)
if hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest() != STYLE_SHA256:
    raise ValueError("style changed during import")

TIMESTEPS = (1, 25, 50, 100)
SAMPLERS = (("two_step", 2), ("ddim_25", 25), ("ddim_50", 50), ("ddim_100", 100))
# Full-study counts required for this figure, not estimated or plotted scores.
N_WINDOWS, N_DRAWS, N_CONTEXT_PER_T = 288, 39, 36
# Relevant lineage recorded by prigm.oracle.oracle_metadata. Extra coordinator
# or experiment hashes need not be identical between independently run inputs.
ORACLE_SOURCES = {
    "oracle.py", "hmm_diagnostics.py", "toy.py", "toy_windows.py", "keys.py",
    "schedule.py", "scoring.py", "normalize.py", "observation.py", "config.py",
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _json_float(value):
    return _finite(float(value), "JSON number")


def _sha256(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("expected a lowercase hexadecimal SHA256 digest")
    return value


def _count(value, expected, name):
    if type(value) is not int or value != expected:
        raise ValueError(f"{name} must equal {expected}")


def _mean_matches(value, scores, name, *, abs_tol=1e-12):
    _finite(value, name)
    for score in scores:
        _finite(score, name)
    if not math.isclose(value, math.fsum(scores) / len(scores), rel_tol=1e-12, abs_tol=abs_tol):
        raise ValueError(f"{name} does not match its individual records")


def load_evidence(path, expected_hash):
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != _sha256(expected_hash):
        raise ValueError(f"input SHA256 mismatch: {path}")
    record = json.loads(payload, object_pairs_hook=_unique_object,
                        parse_float=_json_float, parse_constant=_json_float)
    _count(record["schema_version"], 2, "schema_version")
    if record["conditioning"] != "pooled" or record["boundary_law"] not in ("iid", "stationary"):
        raise ValueError("requires pooled conditioning and a declared iid/stationary boundary law")
    _sha256(record["toy_eval_sha256"])
    sources = record["source_hashes"]
    if not isinstance(sources, dict) or not sources:
        raise ValueError("source_hashes must be a nonempty object")
    for name, value in sources.items():
        if not name:
            raise ValueError("empty source hash name")
        _sha256(value)
    return record, digest


def validate_evidence(hmm, oracle):
    for key in ("toy_eval_sha256", "boundary_law", "conditioning"):
        if hmm[key] != oracle[key]:
            raise ValueError(f"HMM/oracle mismatch: {key}")
    shared = sorted(name for name in hmm["source_hashes"].keys() & oracle["source_hashes"].keys()
                    if Path(name).name in ORACLE_SOURCES)
    if not shared:
        raise ValueError("no common relevant HMM/oracle source hashes")
    for name in shared:
        if hmm["source_hashes"][name] != oracle["source_hashes"][name]:
            raise ValueError(f"HMM/oracle source hash mismatch: {name}")

    _count(oracle["n"], N_WINDOWS, "oracle.n")
    _count(oracle["n_draws"], N_DRAWS, "oracle.n_draws")
    contexts = oracle["contexts"]
    if not isinstance(contexts, list) or len(contexts) != N_WINDOWS:
        raise ValueError("oracle requires all window contexts")
    by_window = {}
    for row in contexts:
        window = row["window"]
        if type(window) is not int or not 0 <= window < N_WINDOWS or window in by_window:
            raise ValueError("invalid or duplicate oracle context window")
        for key in ("alpha", "generating_p"):
            _finite(row[key], f"oracle.contexts.{key}")
        for key in ("a", "b"):
            if type(row[key]) is not int or row[key] not in (0, 1):
                raise ValueError(f"invalid oracle context boundary: {key}")
        by_window[window] = row
    sampler_rows, reference = [], None
    for method, budget in (("exact_reference", 0), *SAMPLERS):
        row = oracle["methods"][method]
        _count(row["denoiser_evaluations_per_draw"], budget, f"{method}.budget")
        _count(row["total_denoiser_evaluations"], budget * N_WINDOWS * N_DRAWS,
               f"{method}.total_denoiser_evaluations")
        scores = row["scores"]
        if not isinstance(scores, list) or len(scores) != N_WINDOWS:
            raise ValueError(f"{method} requires all {N_WINDOWS} scores")
        _mean_matches(row["mean_energy"], scores, f"{method}.mean_energy")
        plotted = {"method": method, "denoiser_evaluations_per_draw": budget,
                   "mean_energy": row["mean_energy"], "n": oracle["n"], "n_draws": oracle["n_draws"]}
        if method == "exact_reference":
            reference = plotted
        else:
            sampler_rows.append(plotted)

    _count(hmm["n_context"], N_CONTEXT_PER_T * len(TIMESTEPS), "hmm.n_context")
    _count(hmm["n_context_per_regime"], 4, "hmm.n_context_per_regime")
    if (hmm["timesteps"] != list(TIMESTEPS)
            or any(type(t) is not int for t in hmm["timesteps"])
            or set(hmm["context_by_t"]) != set(map(str, TIMESTEPS))):
        raise ValueError(f"requires exactly timesteps {TIMESTEPS}")
    if type(hmm["n_paths"]) is not int or hmm["n_paths"] < 2:
        raise ValueError("hmm.n_paths must be an integer >= 2")
    records = hmm["context"]
    if not isinstance(records, list) or len(records) != hmm["n_context"]:
        raise ValueError("HMM requires all individual context records")
    grouped = {t: [] for t in TIMESTEPS}
    seen = set()
    for row in records:
        t, window, regime = row["t"], row["window"], row["regime"]
        if (type(t) is not int or t not in grouped or type(window) is not int
                or window not in by_window or (t, window) in seen
                or type(regime) is not int or not 0 <= regime < 9):
            raise ValueError("invalid or duplicate HMM context identity")
        seen.add((t, window))
        for key in ("alpha", "generating_p", "a", "b"):
            _finite(row[key], f"hmm.context.{key}")
            if row[key] != by_window[window][key]:
                raise ValueError(f"HMM/oracle context mismatch: window {window}, {key}")
        _count(row["denoiser_evaluations_per_endpoint"], 1 if t == 1 else 2,
               "hmm.context.denoiser_evaluations_per_endpoint")
        grouped[t].append(row)
    context_rows, residuals = [], {}
    selected = None
    for t in TIMESTEPS:
        summary, group = hmm["context_by_t"][str(t)], grouped[t]
        _count(summary["n"], N_CONTEXT_PER_T, f"context_by_t.{t}.n")
        identities = {(row["window"], row["regime"]) for row in group}
        if (len(group) != N_CONTEXT_PER_T or (selected is not None and identities != selected)
                or any(sum(row["regime"] == r for row in group) != 4 for r in range(9))):
            raise ValueError("requires the same four contexts per regime at every timestep")
        selected = identities
        for field, individual in (
            ("mean_es_vs_ref", "mean_es_vs_ref"),
            ("mean_reference_self_energy_mc", "reference_self_energy_mc"),
            ("mean_abs_r_star", "mean_abs_r_star"),
        ):
            _mean_matches(summary[field], [row[individual] for row in group],
                          f"context_by_t.{t}.{field}",
                          abs_tol=0.0 if field == "mean_abs_r_star" else 1e-12)
        if summary["mean_abs_r_star"] <= 0 or any(row["mean_abs_r_star"] < 0 for row in group):
            raise ValueError("residuals must be nonnegative with a positive mean for the logarithmic axis")
        context_rows.append({"t": t, "n": summary["n"],
                             "mean_es_vs_ref": summary["mean_es_vs_ref"],
                             "mean_reference_self_energy_mc": summary["mean_reference_self_energy_mc"]})
        residuals[str(t)] = summary["mean_abs_r_star"]
    return context_rows, sampler_rows, reference, shared, residuals


DIAGNOSTIC_SHA256 = "26a264d21bdc152c75a739f7cc59b8155226953522bb1bae90aee69ba8820a01"


def load_diagnostic(path, expected_hash, inputs, replay):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != _sha256(expected_hash):
        raise ValueError("diagnostic SHA256 mismatch")
    record = json.loads(
        payload, object_pairs_hook=_unique_object, parse_float=_json_float, parse_constant=_json_float
    )
    _count(record["schema_version"], 2, "diagnostic schema")
    _count(record["checked_context_rows"], 144, "replayed case count")
    if record["inputs"] != inputs or record["source"] != replay:
        raise ValueError("diagnostic replay provenance differs")
    _finite(record["enumerated_target_score"], "enumerated score")
    return record


def display_rows(hmm, diagnostic):
    if hmm["boundary_law"] != "stationary" or hmm["conditioning"] != "pooled":
        raise ValueError("path rates require the pooled stationary bridge law")
    _count(hmm["n_paths"], 128, "S estimates per case")
    panels = {"panel_a": [], "panel_b": {"t": 100, "cases": []}, "panel_c": []}
    for t in TIMESTEPS:
        group = [r for r in hmm["context"] if r["t"] == t]
        windows = [r["window"] for r in group]
        differences = [r["mean_es_vs_ref"] - r["reference_self_energy_mc"] for r in group]
        panels["panel_a"].append(
            {"t": t, "windows": windows, "values": differences, "mean": float(np.mean(differences))}
        )
        if t == 100:
            panels["panel_b"]["cases"] = [
                {"window": r["window"], "matching_endpoints": r["a"] == r["b"],
                 "mean_abs_r_star": r["mean_abs_r_star"], "score_difference": diff}
                for r, diff in zip(group, differences, strict=True)
            ]

    paths = diagnostic["t100"]
    group = [r for r in hmm["context"] if r["t"] == 100]
    for row, case in zip(paths, group, strict=True):
        if any(row[k] != case[k] for k in ("window", "alpha", "a", "b")):
            raise ValueError("decoded path case identity differs")
        _count(row["n"], hmm["n_paths"], "decoded paths per case")
    p = np.array([0.1, 0.25, 0.5])
    for same in (True, False):
        subset = [r for r in paths if (r["a"] == r["b"]) == same]
        if not subset:
            raise ValueError("both endpoint groups are required")
        parity = int(not same)
        counts = np.zeros(14, dtype=int)
        locations = np.zeros(13, dtype=int)
        for row in subset:
            ids = row["decoded_path_ids"]
            n = row["n"]
            if len(ids) != n or any(type(i) is not int or not 0 <= i < 2**12 for i in ids):
                raise ValueError("invalid decoded path IDs")
            hidden = (np.array(ids)[:, None] >> np.arange(12)) % 2
            paths_with_ends = np.column_stack((np.full(n, row["a"]), hidden, np.full(n, row["b"])))
            changed = np.diff(paths_with_ends, axis=1) != 0
            switches = changed.sum(axis=1)
            counts += np.bincount(switches, minlength=14)
            locations += changed[switches == 1].sum(axis=0)
            _count(row["minimum_switch_paths"], int(np.sum(switches == parity)), "minimum-switch paths")
        # Condition the joint regime/count mass on parity, including both boundary links.
        endpoint = (1 + (1 if same else -1) * (1 - 2 * p) ** 13) / 2
        exact = np.zeros(14)
        for k in range(parity, 14, 2):
            exact[k] = math.comb(13, k) * (p**k * (1-p)**(13-k)).sum() / endpoint.sum()
        panels["panel_c"].append({
            "matching_endpoints": same, "n_cases": len(subset), "n_paths": int(counts.sum()),
            "minimum_switch_paths": int(counts[parity]), "exact_probability": float(exact[parity]),
            "switch_counts": counts.tolist(), "exact_probabilities": exact.tolist(),
            "one_switch_positions": locations.tolist(),
        })
    return panels


def plot_evidence(data):
    blue = "#0077BB"
    decoded_color = "#A64B00"
    height = 4.65
    fig = plt.figure(figsize=(5.5, height))

    def panel(left, bottom, width, span):
        # Vertical measurements are in inches so labels keep their print spacing.
        return fig.add_axes([left, bottom / height, width, span / height])

    left, right, width = .11, .615, .35
    a = panel(left, 3.39, width, 1.)
    b = panel(right, 3.39, width, 1.)
    category_gray = "#f0f0f0"
    mean_marker = dict(marker="D", linestyle="none", markersize=4.5, markerfacecolor="#B07800",
                       markeredgecolor="white", markeredgewidth=.65)
    for i in range(len(TIMESTEPS)):
        differences = np.asarray(data["panel_a"][i]["values"])
        y = 3 - i
        if i % 2:
            a.axhspan(y - .5, y + .5, facecolor=category_gray, edgecolor="none", zorder=-2)
        # Offsets separate repeated cases and preserve every numerical abscissa.
        a.scatter(differences, y + np.linspace(-0.27, 0.27, len(differences)),
                  s=5, marker="o", color=blue, alpha=0.77, edgecolors="none", zorder=2)
        a.plot(differences.mean(), y, **mean_marker, zorder=3)
    a.axvline(0, lw=0.65, color="#888888", ls="--", zorder=0)
    a.set(xlim=(-0.22, 4.5), xticks=[0, 1, 2, 3, 4], ylim=(-0.48, 3.48))
    a.set_yticks([3, 2, 1, 0], ["1", "25", "50", "100"])
    a.set_ylabel("Noise index $t$", fontsize=7)
    a.set_title("(a) 36 cases per noise index", loc="center")
    cases = data["panel_b"]["cases"]
    residual = np.array([r["mean_abs_r_star"] for r in cases])
    difference = np.array([r["score_difference"] for r in cases])
    matching = np.array([r["matching_endpoints"] for r in cases])
    for mask, marker, color in ((matching, "o", blue), (~matching, "^", "#EE3377")):
        b.scatter(residual[mask], difference[mask], s=10, marker=marker, color=color,
                  alpha=0.80, edgecolors="none", zorder=2)
    b.plot(residual.mean(), difference.mean(), **mean_marker, zorder=3)
    b.set_title("(b) End states at $t=100$", loc="center")
    b.set_xscale("symlog", linthresh=1e-14, linscale=0.5)
    b.margins(x=0.065, y=0.14)
    b.set_xticks([0, 1e-12, 1e-8, 1e-4], ["0", r"$10^{-12}$", r"$10^{-8}$", r"$10^{-4}$"])
    b.xaxis.set_minor_formatter(NullFormatter())
    b.annotate(f"{matching.sum()} same",
               (np.median(residual[matching]), np.median(difference[matching])),
               xytext=(4, 4), textcoords="offset points", fontsize=6.7)
    b.annotate(f"{(~matching).sum()} different",
               (np.median(residual[~matching]), difference[~matching].min()),
               xytext=(-8, -9), textcoords="offset points", fontsize=6.7, ha="center")
    b.set_ylabel("Energy score (S − baseline)", fontsize=7, labelpad=1)
    a.set_xlabel("Energy score (S − baseline)", fontsize=7, labelpad=1)
    b.set_xlabel("Mean absolute residual in model units", fontsize=7, labelpad=1)
    for axis in (a, b):
        axis.tick_params(labelsize=6.7, pad=2)
    group_center = (left + right + width) / 2
    fig.text(group_center, 2.68 / height, "(c) Switches per path at $t=100$", ha="center", fontsize=8)
    for parity, row in enumerate(data["panel_c"]):
        count_ax = panel((left, right)[parity], 1.70, width, .67)
        k = np.arange(parity, 14, 2)
        exact = 100 * np.asarray(row["exact_probabilities"])[k]
        counts = np.asarray(row["switch_counts"])[k]
        observed = counts > 0
        percentages = 100 * counts / row["n_paths"]
        count_ax.vlines(k, 0, exact, color=".56", lw=.75)
        count_ax.scatter(k, exact, marker="o", s=12, facecolors="white", edgecolors=".25",
                     linewidths=.8, zorder=3)
        count_ax.scatter(k[observed], percentages[observed], marker="s", s=13, color=decoded_color, zorder=4)
        if parity:
            count_ax.annotate(f"{counts[1]:,} of {row['n_paths']:,} paths", (3, percentages[1]), xytext=(4.5, 51),
                          fontsize=6.7, color=decoded_color,
                              arrowprops={"arrowstyle": "-", "lw": .5, "color": decoded_color})
        count_ax.set(xticks=k, ylim=(-4, 109), yticks=[0, 50, 100], xlim=(parity-.8, parity+12.8))
        count_ax.set_title("Different end states" if parity else "Same end states", fontsize=7.1, pad=2.5)
        if parity:
            count_ax.tick_params(axis="y", labelleft=False)
        else:
            count_ax.set_ylabel("Paths (%)", fontsize=7, labelpad=.8)
        count_ax.tick_params(labelsize=6.7, pad=1)
    # The location distribution is conditional on one switch, unlike the count panels.
    different = data["panel_c"][1]
    locations = np.asarray(different["one_switch_positions"])
    links = np.arange(1, len(locations) + 1)
    n_one_switch = different["switch_counts"][1]
    percent = 100 * locations / n_one_switch
    d = panel(left, .45, right + width - left, .66)
    d.set_title("(d) Position of the switch in paths with one switch", loc="center", pad=4)
    uniform = 100 / len(links)
    d.axhline(uniform, color=".35", ls=(0, (3, 2)), lw=.75, gid="bridge_position")
    d.text(1.1, uniform + 7, "Conditional law (1/13)", fontsize=6.7, color=".2")
    d.bar(links, percent, width=.62, color=blue, edgecolor="none", zorder=1)
    peak = int(np.argmax(locations))
    d.annotate(f"{locations[peak]:,} of {n_one_switch:,} paths",
               (links[peak], percent[peak]), xytext=(12, 0), textcoords="offset points",
               va="center", fontsize=6.7, color=blue)
    d.set(xticks=links, xlim=(.5, 13.5), yticks=[0, 50, 100], ylim=(0, 104))
    d.set_xlabel("Transition from the left end of the gap", fontsize=7, labelpad=1)
    d.set_ylabel("Paths (%)", fontsize=7, labelpad=.8)
    d.tick_params(labelsize=6.7, pad=1)
    key_style = dict(loc="center", frameon=False, borderpad=0, handlelength=1.2,
                     handletextpad=.35, columnspacing=1.4, fontsize=7)
    fig.legend(handles=[Line2D([], [], **mean_marker, label="Mean")],
               bbox_to_anchor=(group_center, 3.02 / height), **key_style)
    fig.legend(handles=[
        Line2D([], [], marker="o", color=".25", markerfacecolor="white", ls="", markersize=4,
               label="Model probability"),
        Line2D([], [], marker="s", color=decoded_color, ls="", markersize=4, label="Decoded S frequency"),
    ], bbox_to_anchor=(group_center, 1.45 / height), ncol=2, **key_style)
    return fig


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hmm", required=True, type=Path)
    parser.add_argument("--hmm-sha256", required=True)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--oracle-sha256", required=True)
    parser.add_argument("--diagnostic", type=Path, help="default: path-diagnostic.json beside --hmm")
    parser.add_argument("--diagnostic-sha256", default=DIAGNOSTIC_SHA256)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="must not exist; its parent must exist")
    args = parser.parse_args(argv)
    try:
        if hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest() != STYLE_SHA256:
            raise ValueError("style changed since import")
        out = args.output_dir.absolute()
        if out.exists() or out.is_symlink():
            raise FileExistsError(f"output directory must be fresh: {out}")
        font_path = font_manager.findfont(
            font_manager.FontProperties(family=STYLE["font.family"]), fallback_to_default=False)
        if Path(font_path).suffix.lower() != ".ttf":
            raise ValueError("DejaVu Serif TrueType font required; see docs/prigm_release.md")
        source = Path(__file__).resolve()
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        style = source.with_name("prigm.mplstyle")
        style_hash = STYLE_SHA256
        hmm_path, oracle_path = args.hmm.resolve(), args.oracle.resolve()
        hmm, hmm_hash = load_evidence(hmm_path, args.hmm_sha256)
        oracle, oracle_hash = load_evidence(oracle_path, args.oracle_sha256)
        _, _, _, shared, _ = validate_evidence(hmm, oracle)
        replay_path = source.with_name("hmm_display_evidence.py")
        replay = {"path": replay_path.name, "sha256": hashlib.sha256(replay_path.read_bytes()).hexdigest()}
        diagnostic_path = args.diagnostic or hmm_path.with_name("path-diagnostic.json")
        diagnostic = load_diagnostic(diagnostic_path, args.diagnostic_sha256,
                                     {"toy.npz": hmm["toy_eval_sha256"], "hmm-stationary-01.json": hmm_hash,
                                      "oracle-stationary-01.json": oracle_hash}, replay)
        rows = display_rows(hmm, diagnostic)
        manifest = {
            "schema_version": 5, "figure": "fig_hmm",
            "source": {"path": source.name, "sha256": source_hash},
            "style": {"path": style.name, "sha256": style_hash},
            "dependency_sources": {"replay": replay},
            "inputs": {
                "diagnostic": {"path": diagnostic_path.name, "sha256": args.diagnostic_sha256},
                "hmm": {"path": hmm_path.name, "sha256": hmm_hash, "source_hashes": hmm["source_hashes"]},
                "oracle": {"path": oracle_path.name, "sha256": oracle_hash, "source_hashes": oracle["source_hashes"]},
            },
            "input_schema_version": 2, "toy_eval_sha256": hmm["toy_eval_sha256"],
            "boundary_law": hmm["boundary_law"], "conditioning": hmm["conditioning"],
            "verified_common_source_hashes": {name: hmm["source_hashes"][name] for name in shared},
            "hmm_n_context": hmm["n_context"], "hmm_n_paths": hmm["n_paths"],
            "plotted_rows": rows,
            "matplotlib_version": matplotlib.__version__, "backend": "Agg", "output_sha256": {},
        }
        with plt.rc_context(STYLE):
            fig = plot_evidence(rows)
            try:
                out.mkdir()
                for suffix in ("pdf", "png"):
                    target = out / f"fig_hmm.{suffix}"
                    with target.open("xb") as handle:
                        fig.savefig(handle, format=suffix, dpi=192,
                                    metadata={"CreationDate": None, "ModDate": None} if suffix == "pdf" else None)
                    manifest["output_sha256"][target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
            finally:
                plt.close(fig)
        for path, digest in ((source, source_hash), (style, style_hash),
                             (hmm_path, hmm_hash), (oracle_path, oracle_hash),
                             (replay_path, replay["sha256"]), (diagnostic_path, args.diagnostic_sha256)):
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"input/source changed while rendering; no manifest published: {path}")
        manifest_path = out / "fig_hmm.manifest.json"
        with manifest_path.open("x") as handle:
            handle.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
        parser.error(str(exc))
    print(json.dumps({"manifest": str(manifest_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
