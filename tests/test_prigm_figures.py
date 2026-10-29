"""Figure contracts use synthetic values, never replacement scientific evidence."""

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.text
import pytest

PAPER = Path(__file__).resolve().parents[1] / "paper/workshops/prigm"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, PAPER / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value))
    return digest(path)


@pytest.fixture
def figures():
    return load_module("fig_audit"), load_module("fig_hmm")


@pytest.fixture
def evidence(figures):
    audit, hmm_plot = figures
    comparison = {"schema_version": 2, "exposure": "historical_test_exposed",
                  "n_replicates": 10000, "comparisons": {}}
    for index, (baseline, _) in reversed(list(enumerate(audit.BASELINES))):
        point = (index - 3) * 0.128739123456789
        comparison["comparisons"][f"row-{index}"] = {
            "method": "O-S_raw", "baseline": baseline,
            "family": "primary_O-S_repair" if index < 4 else "exploratory_control_comparison",
            "estimand": "validation_selected_procedures", "mean_diff": point,
            "year_95": [point - 0.1, point + 0.2],
            "twoyear_95": [point + 0.2, point + 0.4] if index == 0 else [point - 0.3, point + 0.3],
        }
    base = {"schema_version": 2, "conditioning": "pooled", "boundary_law": "iid",
            "toy_eval_sha256": "a" * 64,
            "source_hashes": {name: hashlib.sha256(name.encode()).hexdigest()
                              for name in hmm_plot.ORACLE_SOURCES}}
    oracle = copy.deepcopy(base)
    oracle.update(n=288, n_draws=39, contexts=[], methods={}, reference_self_energy_mc=99.)
    for window in range(288):
        regime = window // 32
        oracle["contexts"].append({"window": window, "alpha": float(regime % 3),
                                   "generating_p": (0.1, 0.25, 0.5)[regime // 3],
                                   "a": window % 2, "b": (window // 2) % 2})
    for index, (method, budget) in enumerate((("exact_reference", 0), *hmm_plot.SAMPLERS)):
        value = (3.123456789, 5.123456789, 4.123456789, 5.623456789, 4.623456789)[index]
        oracle["methods"][method] = {"denoiser_evaluations_per_draw": budget,
                                     "total_denoiser_evaluations": budget * 288 * 39,
                                     "mean_energy": value, "scores": [value] * 288}
    hmm = copy.deepcopy(base)
    hmm.update(n_context=144, n_context_per_regime=4, n_paths=128,
               timesteps=[1, 25, 50, 100], context_by_t={}, context=[])
    for index, timestep in enumerate(hmm["timesteps"]):
        energy, reference = 3.23456789 + index * 0.5, 3.01234567
        residual = (2e-16, 4e-15, 0.02, 0.04)[index]
        hmm["context_by_t"][str(timestep)] = {
            "n": 36, "mean_es_vs_ref": energy,
            "mean_reference_self_energy_mc": reference, "mean_abs_r_star": residual}
        for regime in range(9):
            for index in range(4):
                row = dict(oracle["contexts"][regime * 32 + index])
                row.update(regime=regime, t=timestep,
                           denoiser_evaluations_per_endpoint=1 if timestep == 1 else 2,
                           mean_es_vs_ref=energy, reference_self_energy_mc=reference,
                           mean_abs_r_star=residual)
                hmm["context"].append(row)
    return comparison, hmm, oracle


@pytest.mark.parametrize("name", ["audit", "hmm"])
def test_real_entrypoint_writes_portable_exact_manifest(tmp_path, figures, evidence, name):
    audit, hmm_plot = figures
    comparison, hmm, oracle = evidence
    output = tmp_path / "figure"
    if name == "audit":
        path = tmp_path / "comparison.json"
        arguments = ["--comparison", str(path), "--comparison-sha256", save(path, comparison)]
        module = audit
    else:
        hp, op = tmp_path / "hmm.json", tmp_path / "oracle.json"
        arguments = ["--hmm", str(hp), "--hmm-sha256", save(hp, hmm),
                     "--oracle", str(op), "--oracle-sha256", save(op, oracle)]
        module = hmm_plot
    assert module.main([*arguments, "--output-dir", str(output)]) == 0
    manifest = json.loads((output / f"fig_{name}.manifest.json").read_text())
    assert manifest["source"] == {"path": f"fig_{name}.py", "sha256": digest(PAPER / f"fig_{name}.py")}
    for item in manifest["inputs"].values():
        assert not Path(item["path"]).is_absolute()
        assert Path(item["path"]).name == item["path"]
        assert digest(tmp_path / item["path"]) == item["sha256"]
    assert manifest["output_sha256"] == {f"fig_{name}.{suffix}": digest(output / f"fig_{name}.{suffix}")
                                        for suffix in ("pdf", "png")}
    if name == "audit":
        assert [r["baseline"] for r in manifest["plotted_rows"]] == [b for b, _ in audit.BASELINES]
        for row in manifest["plotted_rows"]:
            for key in ("mean_diff", "year_95", "twoyear_95"):
                assert row[key] == comparison["comparisons"][row["comparison"]][key]
    else:
        assert manifest["style"] == {"path": "prigm.mplstyle",
                                     "sha256": digest(PAPER / "prigm.mplstyle")}
        rows = manifest["plotted_rows"]
        assert manifest["schema_version"] == 2
        assert "unplotted_context_mean_abs_r_star" not in manifest
        assert rows["panel_c_horizontal_reference"]["mean_energy"] == oracle["methods"]["exact_reference"]["mean_energy"]
        assert rows["panel_c_horizontal_reference"]["mean_energy"] != oracle["reference_self_energy_mc"]
        assert rows["panel_b_context_residuals"] == [
            {"t": t, "n": 36, "mean_abs_r_star": hmm["context_by_t"][str(t)]["mean_abs_r_star"]}
            for t in hmm_plot.TIMESTEPS]
    hashes = {p.name: digest(p) for p in output.iterdir()}
    with pytest.raises(SystemExit) as error:
        module.main([*arguments, "--output-dir", str(output)])
    assert error.value.code == 2
    assert hashes == {p.name: digest(p) for p in output.iterdir()}
    bad = list(arguments)
    bad[bad.index("--comparison-sha256" if name == "audit" else "--hmm-sha256") + 1] = "0" * 64
    with pytest.raises(SystemExit):
        module.main([*bad, "--output-dir", str(tmp_path / "wrong-hash")])
    assert not (tmp_path / "wrong-hash").exists()


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(schema_version=1), lambda d: d.update(exposure="blind"),
    lambda d: d.update(n_replicates=9999), lambda d: d["comparisons"].pop("row-0"),
    lambda d: d["comparisons"].update(duplicate=copy.deepcopy(d["comparisons"]["row-0"])),
    lambda d: d["comparisons"]["row-0"].update(mean_diff=float("nan")),
    lambda d: d["comparisons"]["row-0"].update(mean_diff=True),
    lambda d: d["comparisons"]["row-0"].update(year_95=[2, 1]),
    lambda d: d["comparisons"]["row-0"].update(estimand="other"),
])
def test_audit_rejects_incompatible_evidence(tmp_path, figures, evidence, mutation):
    comparison = evidence[0]
    mutation(comparison)
    path = tmp_path / "comparison.json"
    sha = save(path, comparison)
    with pytest.raises((ValueError, KeyError)):
        figures[0].load_comparison(path, sha)


@pytest.mark.parametrize("mutation", [
    lambda h, o: o.update(n=287), lambda h, o: o.update(n_draws=38),
    lambda h, o: o.update(toy_eval_sha256="d" * 64),
    lambda h, o: o.update(boundary_law="stationary"),
    lambda h, o: o["source_hashes"].update({"oracle.py": "e" * 64}),
    lambda h, o: h.update(n_context=140), lambda h, o: h["context"].pop(),
    lambda h, o: h["context_by_t"]["1"].update(n=35),
    lambda h, o: h["context_by_t"]["1"].update(mean_es_vs_ref=999.),
    lambda h, o: h["context"][0].update(window=h["context"][1]["window"]),
    lambda h, o: h["context"][0].update(alpha=9),
    lambda h, o: o["methods"].pop("ddim_100"),
    lambda h, o: o["methods"]["ddim_25"]["scores"].pop(),
    lambda h, o: o["methods"]["ddim_25"].update(mean_energy=999.),
    lambda h, o: o["methods"]["two_step"].update(total_denoiser_evaluations=10),
    lambda h, o: o["contexts"][0].update(window=1),
])
def test_hmm_rejects_mixed_or_incomplete_populations(figures, evidence, mutation):
    _, hmm, oracle = evidence
    mutation(hmm, oracle)
    with pytest.raises((ValueError, KeyError)):
        figures[1].validate_evidence(hmm, oracle)


def test_figures_fit_single_column_without_hidden_clipping(tmp_path, figures, evidence):
    audit, hmm_plot = figures
    comparison, hmm, oracle = evidence
    path = tmp_path / "comparison.json"
    _, _, rows = audit.load_comparison(path, save(path, comparison))
    for row in rows:
        row["mean_diff"] *= 1000
        for key in ("year_95", "twoyear_95"):
            row[key] = [value * 1000 for value in row[key]]
    context, samplers, reference, _, residuals = hmm_plot.validate_evidence(hmm, oracle)
    plots = [audit.plot_comparisons(rows), hmm_plot.plot_evidence(context, samplers, reference, residuals)]
    try:
        values = [v for r in rows for v in (r["mean_diff"], *r["year_95"], *r["twoyear_95"])]
        low, high = plots[0].axes[0].get_xlim()
        assert low < min(values) and high > max(values)
        for ax in plots[1].axes:
            assert ax.xaxis.label.get_fontsize() == ax.yaxis.label.get_fontsize() == 8
            assert ax.xaxis.label.get_fontfamily() == ["DejaVu Serif"]
            assert ax.lines[0].get_color() == hmm_plot.COLORS[2]
        for ax in (plots[1].axes[0], plots[1].axes[2]):
            assert all(text.get_fontsize() == 7 for text in ax.get_legend().get_texts())
            assert ax.lines[1].get_color() == hmm_plot.COLORS[3]
        for line in plots[1].axes[0].lines:
            assert line.get_linestyle() == "None"
            assert list(line.get_xdata()) == [1, 25, 50, 100]
        assert list(plots[1].axes[2].lines[1].get_ydata()) == [reference["mean_energy"]] * 2
        for figure in plots:
            assert figure.get_size_inches()[0] <= 5.5
            figure.canvas.draw()
            for axis in figure.axes:
                low, high = sorted(axis.get_xlim())
                boxes = [tick.label1.get_window_extent(figure.canvas.get_renderer())
                         for tick in axis.xaxis.get_major_ticks() if low <= tick.get_loc() <= high]
                assert all(not left.overlaps(right) for left, right in zip(boxes, boxes[1:]))
            # Matplotlib retains out-of-view tick artists without rendering them.
            unrendered_ticks = set()
            for axis in figure.axes:
                for coordinate, limits in ((axis.xaxis, axis.get_xlim()), (axis.yaxis, axis.get_ylim())):
                    low, high = sorted(limits)
                    for tick in (*coordinate.get_major_ticks(), *coordinate.get_minor_ticks()):
                        if not low <= tick.get_loc() <= high:
                            unrendered_ticks.update((tick.label1, tick.label2))
            for text in figure.findobj(match=matplotlib.text.Text):
                if text.get_text() and text.get_visible() and text not in unrendered_ticks:
                    assert text.get_fontsize() >= 7
                    box = text.get_window_extent(figure.canvas.get_renderer())
                    assert box.x0 >= -1 and box.x1 <= figure.bbox.x1 + 1, text.get_text()
                    assert box.y0 >= -1 and box.y1 <= figure.bbox.y1 + 1, text.get_text()
    finally:
        for figure in plots:
            plt.close(figure)


def test_hmm_existing_values_reach_all_three_axes_exactly(figures):
    module = figures[1]
    root = PAPER / "evidence/toy"
    hmm = json.loads((root / "hmm-stationary-01.json").read_text())
    oracle = json.loads((root / "oracle-stationary-01.json").read_text())
    context, samplers, reference, _, residuals = module.validate_evidence(hmm, oracle)
    fig = module.plot_evidence(context, samplers, reference, residuals)
    try:
        assert len(fig.axes) == 3
        a, b, c = fig.axes
        assert list(a.lines[0].get_ydata()) == [hmm["context_by_t"][str(t)]["mean_es_vs_ref"]
                                              for t in module.TIMESTEPS]
        assert list(a.lines[1].get_ydata()) == [hmm["context_by_t"][str(t)]["mean_reference_self_energy_mc"]
                                              for t in module.TIMESTEPS]
        assert b.get_yscale() == "log" and len(b.lines) == 1
        assert list(b.lines[0].get_xdata()) == list(module.TIMESTEPS)
        assert list(b.lines[0].get_ydata()) == [hmm["context_by_t"][str(t)]["mean_abs_r_star"]
                                              for t in module.TIMESTEPS]
        assert b.lines[0].get_linestyle() == "None"
        assert list(c.lines[0].get_ydata()) == [oracle["methods"][name]["mean_energy"]
                                              for name, _ in module.SAMPLERS]
        assert list(c.lines[1].get_ydata()) == [oracle["methods"]["exact_reference"]["mean_energy"]] * 2
        assert a.lines[1].get_label() == c.lines[1].get_label() == "Exact conditional samples"
        assert b.texts[0].get_text() == r"$4.17\times 10^{-5}$"
        assert c.get_position().width > a.get_position().width
    finally:
        plt.close(fig)


@pytest.mark.parametrize("invalid", [0.0, -1e-16, float("nan"), float("inf"), -float("inf")])
def test_hmm_log_axis_refuses_invalid_values_before_creating_figure(figures, evidence, invalid):
    module = figures[1]
    context, samplers, reference, _, residuals = module.validate_evidence(evidence[1], evidence[2])
    residuals["1"] = invalid
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        module.plot_evidence(context, samplers, reference, residuals)
    assert plt.get_fignums() == before


@pytest.mark.parametrize("invalid", [0.0, -1e-16, 1e-15, float("nan"), float("inf")])
def test_hmm_residual_aggregate_matches_small_individual_values(figures, invalid):
    module = figures[1]
    root = PAPER / "evidence/toy"
    hmm = json.loads((root / "hmm-stationary-01.json").read_text())
    oracle = json.loads((root / "oracle-stationary-01.json").read_text())
    module.validate_evidence(hmm, oracle)
    hmm["context_by_t"]["1"]["mean_abs_r_star"] = invalid
    with pytest.raises(ValueError):
        module.validate_evidence(hmm, oracle)


@pytest.mark.parametrize("value", [0.0, -1e-16])
def test_hmm_log_residual_rejects_consistent_nonpositive_evidence(figures, evidence, value):
    module = figures[1]
    _, hmm, oracle = evidence
    module.validate_evidence(hmm, oracle)
    hmm["context_by_t"]["1"]["mean_abs_r_star"] = value
    for row in hmm["context"]:
        if row["t"] == 1:
            row["mean_abs_r_star"] = value
    with pytest.raises(ValueError, match="positive mean"):
        module.validate_evidence(hmm, oracle)


@pytest.mark.parametrize("value", [0.0, -1e-16])
def test_hmm_individual_residual_sign_with_positive_aggregate(figures, evidence, value):
    module = figures[1]
    _, hmm, oracle = evidence
    module.validate_evidence(hmm, oracle)
    group = [row for row in hmm["context"] if row["t"] == 1]
    group[0]["mean_abs_r_star"] = value
    mean = math.fsum(row["mean_abs_r_star"] for row in group) / len(group)
    assert mean > 0
    hmm["context_by_t"]["1"]["mean_abs_r_star"] = mean
    if value == 0:
        assert module.validate_evidence(hmm, oracle)[4]["1"] == mean
    else:
        with pytest.raises(ValueError, match="nonnegative"):
            module.validate_evidence(hmm, oracle)


def test_figure_counts_match_scientific_configuration(figures):
    from prigm.config import BOOTSTRAP_REPLICATES, N_DRAWS
    assert figures[0].BOOTSTRAP_REPLICATES == BOOTSTRAP_REPLICATES
    assert figures[1].N_DRAWS == N_DRAWS


def test_hmm_style_rechecked_after_save(tmp_path, figures, evidence, monkeypatch):
    module = figures[1]
    hp, op = tmp_path / "hmm.json", tmp_path / "oracle.json"
    args = ["--hmm", str(hp), "--hmm-sha256", save(hp, evidence[1]),
            "--oracle", str(op), "--oracle-sha256", save(op, evidence[2])]
    source = tmp_path / "fig_hmm.py"
    source.write_bytes((PAPER / source.name).read_bytes())
    style = tmp_path / "prigm.mplstyle"
    style.write_bytes((PAPER / style.name).read_bytes())
    monkeypatch.setattr(module, "__file__", str(source))
    real_save = matplotlib.figure.Figure.savefig

    def changing_save(fig, *args, **kwargs):
        assert module.matplotlib.rcParams["pdf.fonttype"] == 42
        assert module.matplotlib.rcParams["font.family"] == ["DejaVu Serif"]
        result = real_save(fig, *args, **kwargs)
        if kwargs["format"] == "png":
            style.write_bytes(style.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", changing_save)
    output = tmp_path / "changed"
    with pytest.raises(SystemExit):
        module.main([*args, "--output-dir", str(output)])
    assert not (output / "fig_hmm.manifest.json").exists()


def test_hmm_refuses_nimbus_opentype_fallback(tmp_path, figures, evidence, monkeypatch):
    module = figures[1]
    hp, op = tmp_path / "hmm.json", tmp_path / "oracle.json"
    args = ["--hmm", str(hp), "--hmm-sha256", save(hp, evidence[1]),
            "--oracle", str(op), "--oracle-sha256", save(op, evidence[2]),
            "--output-dir", str(tmp_path / "output")]

    def opentype_font(*args, **kwargs):
        assert kwargs["fallback_to_default"] is False
        return "/fonts/DejaVuSerif.otf"

    monkeypatch.setattr(module.font_manager, "findfont", opentype_font)
    with pytest.raises(SystemExit):
        module.main(args)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("name", [
    "hmm", "operator", "calibration", "checkpoints", "fixed_final", "imputation",
])
@pytest.mark.parametrize("phase", ["before_call", "after_guard"])
def test_style_changed_since_import_refused(tmp_path, capsys, monkeypatch, name, phase):
    """Import a private copy, then refuse a style revision the module never loaded."""
    script = tmp_path / f"fig_{name}.py"
    script.write_bytes((PAPER / script.name).read_bytes())
    style = tmp_path / "prigm.mplstyle"
    original_style = (PAPER / style.name).read_bytes()
    style.write_bytes(original_style)
    spec = importlib.util.spec_from_file_location(f"style_mutation_{name}", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    imported_hash = hashlib.sha256(original_style).hexdigest()
    assert module.STYLE_SHA256 == imported_hash

    # Use valid CLI inputs so a separate input error cannot satisfy this test.
    inputs = {
        "hmm": [("hmm", "evidence/toy/hmm-stationary-01.json"),
                ("oracle", "evidence/toy/oracle-stationary-01.json")],
        "operator": [("evidence", "evidence/deadline/operator-mechanism.json")],
        "calibration": [("operator", "evidence/deadline/operator-mechanism.json"),
                        ("calibration", "evidence/followup/calibration/summary.json")],
        "fixed_final": [("evidence", "evidence/deadline/fixed-final.json")],
        "imputation": [("excerpt", "evidence/imputation/sample-excerpt.npz"),
                       ("provenance", "evidence/imputation/MANIFEST.json")],
    }
    args = []
    if name == "checkpoints":
        directory = PAPER / "evidence/followup/checkpoints"
        args = ["--evidence-dir", str(directory),
                "--manifest-sha256", digest(directory / "MANIFEST.json")]
    else:
        for option, relative in inputs[name]:
            path = PAPER / relative
            args.extend([f"--{option}", str(path), f"--{option}-sha256", digest(path)])
    if name == "imputation":
        args.extend(["--case-index", "120"])
    output = tmp_path / "output"
    args.extend(["--output-dir", str(output)])

    if phase == "before_call":
        style.write_bytes(original_style + b"\n# changed after module import\n")
    else:
        # Font lookup follows the entry guard and precedes the old hash reads
        # in all six renderers. Change only the private style, once per case.
        original_findfont = module.font_manager.findfont
        mutated = False

        def changing_font_lookup(*args, **kwargs):
            nonlocal mutated
            result = original_findfont(*args, **kwargs)
            if not mutated:
                style.write_bytes(original_style + b"\n# changed after entry guard\n")
                mutated = True
            return result

        monkeypatch.setattr(module.font_manager, "findfont", changing_font_lookup)
    assert module.STYLE_SHA256 == imported_hash
    with pytest.raises((ValueError, SystemExit)) as error:
        module.main(args)
    message = f"{error.value}\n{capsys.readouterr().err}"
    assert digest(style) != imported_hash
    assert module.STYLE_SHA256 == imported_hash
    if phase == "before_call":
        assert "style changed since import" in message
        assert not output.exists()
    else:
        assert mutated
        expected = {
            "hmm": "input/source changed while rendering",
            "operator": "figure source or input changed during rendering",
            "calibration": "figure source or input changed during rendering",
            "checkpoints": "source or input changed during export",
            "fixed_final": "figure source or input changed during rendering",
            "imputation": "input SHA256 mismatch",
        }
        assert expected[name] in message
    assert not (output / f"fig_{name}.manifest.json").exists()
