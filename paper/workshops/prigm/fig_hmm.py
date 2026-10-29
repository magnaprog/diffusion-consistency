"""Plot residuals and scores for a hidden Markov model with an exact conditional law.

The first subplot compares denoiser outputs with conditional draws on repeated
contexts; the adjacent subplot shows their mean absolute residual. The bottom
subplot evaluates generated ensembles on 288 windows. Its line
uses exact_reference.mean_energy, a different quantity from the conditional
expected score in the first subplot. Both are finite Monte Carlo estimates.
"""

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


STYLE_PATH = Path(__file__).with_name("prigm.mplstyle")
STYLE_SHA256 = hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest()
STYLE = matplotlib.rc_params_from_file(STYLE_PATH, use_default_template=False)
if hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest() != STYLE_SHA256:
    raise ValueError("style changed during import")
COLORS = STYLE["axes.prop_cycle"].by_key()["color"]

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


def plot_evidence(context_rows, sampler_rows, reference, residuals):
    t = [row["t"] for row in context_rows]
    residual_values = [_finite(residuals[str(step)], "mean_abs_r_star") for step in t]
    if any(value <= 0 for value in residual_values):
        raise ValueError("mean absolute residual must be positive for the logarithmic axis")
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(5.5, 5.0))
        grid = fig.add_gridspec(2, 2, hspace=1.1, wspace=0.48)
        axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
                fig.add_subplot(grid[1, :]))
        fig.subplots_adjust(left=0.13, right=0.99, top=0.93, bottom=0.18)
        # Markers only: no interpolation or fitted trend between measured budgets.
        axes[0].plot(t, [row["mean_es_vs_ref"] for row in context_rows], "s",
                     color=COLORS[2], markersize=4, label="Denoising estimate")
        axes[0].plot(t, [row["mean_reference_self_energy_mc"] for row in context_rows], "o",
                     color=COLORS[3], markerfacecolor="none", markersize=7,
                     label="Exact conditional samples")
        axes[0].set_xticks(t)
        axes[0].set_xlabel("Corruption timestep $t$")
        axes[0].set_title("(a) Denoising", loc="left", fontsize=8)
        axes[1].plot(t, residual_values, "s", color=COLORS[2], markersize=4)
        axes[1].set_yscale("log")
        axes[1].set_xticks(t)
        axes[1].set_xlabel("Corruption timestep $t$")
        axes[1].set_ylabel(r"Mean $|R_{\mathrm{obs}}|$ (HMM units)")
        axes[1].set_title("(b) Residual", loc="left", fontsize=8)
        mantissa, exponent = f"{residual_values[-1]:.2e}".split("e")
        axes[1].annotate(rf"${mantissa}\times 10^{{{int(exponent)}}}$", (t[-1], residual_values[-1]),
                         xytext=(-4, -11), textcoords="offset points", ha="right", fontsize=7)
        budgets = [row["denoiser_evaluations_per_draw"] for row in sampler_rows]
        axes[2].plot(budgets, [row["mean_energy"] for row in sampler_rows], "s",
                     color=COLORS[2], markersize=4, label="DDIM estimates")
        axes[2].axhline(reference["mean_energy"], color=COLORS[3], linestyle="--",
                       label="Exact conditional samples")
        axes[2].set_xticks(budgets)
        axes[2].set_xlabel("Denoiser evaluations")
        axes[2].set_title("(c) Sampling", loc="left", fontsize=8)
        for ax in (axes[0], axes[2]):
            ax.set_ylabel("Mean energy score")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), fontsize=7,
                      frameon=False, handlelength=2)
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
            ax.margins(x=0.1, y=0.15)
    return fig


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hmm", required=True, type=Path)
    parser.add_argument("--hmm-sha256", required=True)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--oracle-sha256", required=True)
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
        context_rows, sampler_rows, reference, shared, residuals = validate_evidence(hmm, oracle)
        manifest = {
            "schema_version": 2, "figure": "fig_hmm",
            "source": {"path": source.name, "sha256": source_hash},
            "style": {"path": style.name, "sha256": style_hash},
            "inputs": {
                "hmm": {"path": hmm_path.name, "sha256": hmm_hash, "source_hashes": hmm["source_hashes"]},
                "oracle": {"path": oracle_path.name, "sha256": oracle_hash, "source_hashes": oracle["source_hashes"]},
            },
            "input_schema_version": 2, "toy_eval_sha256": hmm["toy_eval_sha256"],
            "boundary_law": hmm["boundary_law"], "conditioning": hmm["conditioning"],
            "verified_common_source_hashes": {name: hmm["source_hashes"][name] for name in shared},
            "hmm_n_context": hmm["n_context"], "hmm_n_paths": hmm["n_paths"],
            "plotted_rows": {"panel_a_context": context_rows,
                             "panel_b_context_residuals": [
                                 {"t": row["t"], "n": row["n"],
                                  "mean_abs_r_star": residuals[str(row["t"])]}
                                 for row in context_rows],
                             "panel_c_samplers": sampler_rows,
                             "panel_c_horizontal_reference": reference},
            "score_note": "Panels (a,b) use the same 36 cases at each timestep. Panel (a) uses independent context reference draws. Panel (c) uses final samples on 288 windows; its line is exact_reference.mean_energy, not reference_self_energy_mc. All scores are finite Monte Carlo estimates.",
            "residual_note": "Panel (b) plots the validated mean absolute residual on a logarithmic axis, without a floor or zero reference. The estimator uses one denoiser evaluation at t=1 and two at the other timesteps. This is an absolute residual, not the squared training penalty.",
            "matplotlib_version": matplotlib.__version__, "backend": "Agg", "output_sha256": {},
        }
        with plt.rc_context(STYLE):
            fig = plot_evidence(context_rows, sampler_rows, reference, residuals)
            try:
                out.mkdir()
                for suffix in ("pdf", "png"):
                    target = out / f"fig_hmm.{suffix}"
                    with target.open("xb") as handle:
                        fig.savefig(handle, format=suffix, dpi=180)
                    manifest["output_sha256"][target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
            finally:
                plt.close(fig)
        for path, digest in ((source, source_hash), (style, style_hash),
                             (hmm_path, hmm_hash), (oracle_path, oracle_hash)):
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
