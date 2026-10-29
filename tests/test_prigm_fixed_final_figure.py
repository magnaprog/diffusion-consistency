"""Check the coefficient figure using invented results at one final update.

Each coefficient has three seed records. Raw scores averaged with equal
weight per month are intentionally different from pooled-window averages
and scores after correction, so selecting the wrong field is detectable.
Tests check the individual records, arithmetic means, plotting geometry
and invalid-input rejection.

These fixtures verify what the renderer displays; they do not supply new
evidence about how a coefficient changes model performance.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "paper/workshops/prigm/fig_fixed_final.py"
spec = importlib.util.spec_from_file_location("prigm_fixed_final_figure", SOURCE)
figure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figure)

COEFFICIENTS = (0.0, 0.00005, 0.0005, 0.005)
LABELS = ("0", "0.00005", "0.0005", "0.005")
SEEDS = (7, 13, 42)


@pytest.fixture
def evidence():
    """Invented metrics, with deliberately different pooled/corrected values."""
    rows, endpoints = [], {}
    for i, (coefficient, label) in enumerate(zip(COEFFICIENTS, LABELS)):
        for j, seed in enumerate(SEEDS):
            identity = f"c{label}_s{seed}"
            energy = 1.0 + 0.17 * i + 0.09 * j + 0.02 * i * j
            residual = 8.0 - 0.8 * i + 0.27 * j + 0.03 * i * j
            summary = {"es5": {"month_balanced": energy, "mean": energy + 100},
                       "r_obs_mean": {"month_balanced": residual, "mean": residual + 100}}
            rows.append({"id": identity, "c_r": coefficient, "seed": seed,
                         "arm": "D" if coefficient == 0 else "O-S", "generation": 29636,
                         "results": {"raw": {"summary": summary}, "corrected": {"summary": {
                             metric: {"month_balanced": 200.0, "mean": 300.0}
                             for metric in ("es5", "r_obs_mean")}}}})
            endpoints[identity] = {"generation": 29636}
    return {"schema_version": 2, "complete_grid": True,
            "protocol_sha256": "1408cd52f306af0095f808dfe1d34aa8493fb26db9a4e75dae422839157bc3b4",
            "selection": "predetermined_final_update_29636_not_VAL_selected",
            "arrays": {"split": "TEST", "n_windows": 480}, "endpoints": endpoints, "rows": rows}


def save(path, record):
    payload = json.dumps(record).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def arguments(path, digest, output):
    return ["--evidence", str(path), "--evidence-sha256", digest, "--output-dir", str(output)]


def expected_rows(record):
    return [{"coefficient": row["c_r"], "seed": row["seed"],
             "energy": row["results"]["raw"]["summary"]["es5"]["month_balanced"],
             "residual": row["results"]["raw"]["summary"]["r_obs_mean"]["month_balanced"]}
            for row in record["rows"]]


def test_load_returns_exact_raw_rows_in_canonical_order(evidence, tmp_path):
    expected = expected_rows(evidence)
    evidence["rows"].reverse()
    path = tmp_path / "input.json"
    rows = figure.load_rows(str(path), save(path, evidence))
    assert rows == expected
    assert len(rows) == 12
    assert all(type(row["coefficient"]) is float and type(row["seed"]) is int for row in rows)
    assert all(set(row) == {"coefficient", "seed", "energy", "residual"} for row in rows)


@pytest.mark.parametrize("change", [
    lambda d: d.update(schema_version=True),
    lambda d: d.update(schema_version=2.0),
    lambda d: d.update(schema_version=1),
    lambda d: d.update(complete_grid=False),
    lambda d: d.update(complete_grid=1),
    lambda d: d.update(protocol_sha256="0" * 64),
    lambda d: d.update(selection="VAL_selected"),
    lambda d: d["arrays"].update(split="VAL"),
    lambda d: d["arrays"].update(n_windows=479),
    lambda d: d["arrays"].update(n_windows=480.0),
    lambda d: d["arrays"].update(n_windows=True),
    lambda d: d.pop("arrays"),
    lambda d: d["rows"].pop(),
    lambda d: d["rows"].append(copy.deepcopy(d["rows"][0])),
    lambda d: d["rows"].__setitem__(1, copy.deepcopy(d["rows"][0])),
    lambda d: d["rows"][0].update(id="c0_s13"),
    lambda d: d["rows"][0].update(c_r=0.00001),
    lambda d: d["rows"][3].update(c_r=float(np.nextafter(0.00005, 1.0))),
    lambda d: d["rows"][0].update(c_r=False),
    lambda d: d["rows"][0].update(c_r=True),
    lambda d: d["rows"][0].update(c_r="0"),
    lambda d: d["rows"][0].update(seed=8),
    lambda d: d["rows"][0].update(seed=13),
    lambda d: d["rows"][0].update(seed=7.0),
    lambda d: d["rows"][0].update(seed=True),
    lambda d: d["rows"][0].update(seed="7"),
    lambda d: d["rows"][0].update(arm="O-S"),
    lambda d: d["rows"][3].update(arm="D"),
    lambda d: d["rows"][0].update(generation=9560),
    lambda d: d["rows"][0].update(generation=29636.0),
    lambda d: d["rows"][0].update(generation=True),
    lambda d: d["endpoints"].pop("c0_s7"),
    lambda d: d["endpoints"].update(c0_s8={"generation": 29636}),
    lambda d: d["endpoints"]["c0_s7"].update(generation=9560),
    lambda d: d["endpoints"]["c0_s7"].update(generation=True),
    lambda d: d["rows"][0]["results"].pop("raw"),
    lambda d: d["rows"][0]["results"]["raw"]["summary"]["es5"].pop("month_balanced"),
    lambda d: d["rows"][0]["results"]["raw"].update(summary=[]),
])
def test_changed_contract_or_grid_is_rejected(evidence, tmp_path, change):
    change(evidence)
    path = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        figure.load_rows(path, save(path, evidence))


@pytest.mark.parametrize("record", [None, [], 1, True, "input", {"rows": []}])
def test_malformed_root_is_rejected(tmp_path, record):
    path = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        figure.load_rows(path, save(path, record))


@pytest.mark.parametrize("metric", ["es5", "r_obs_mean"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"),
                                   True, False, "1.0", None, [1.0], -0.1, 10 ** 400])
def test_invalid_plotted_numbers_are_rejected(evidence, tmp_path, metric, value):
    evidence["rows"][0]["results"]["raw"]["summary"][metric]["month_balanced"] = value
    path = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        figure.load_rows(path, save(path, evidence))


def test_zero_metrics_remain_zero(evidence, tmp_path):
    for metric in ("es5", "r_obs_mean"):
        evidence["rows"][0]["results"]["raw"]["summary"][metric]["month_balanced"] = 0
    path = tmp_path / "input.json"
    rows = figure.load_rows(path, save(path, evidence))
    assert rows[0]["energy"] == rows[0]["residual"] == 0.0


@pytest.mark.parametrize("location", ["root", "nested", "overflow"])
def test_duplicate_json_keys_and_overflow_are_rejected(evidence, tmp_path, location):
    payload = json.dumps(evidence)
    if location == "root":
        payload = payload[:-1] + ', "schema_version": 2}'
    elif location == "nested":
        payload = payload.replace('"month_balanced": 1.0',
                                  '"month_balanced": 1.0, "month_balanced": 1.0', 1)
    else:
        payload = payload.replace('"month_balanced": 1.0', '"month_balanced": 1e400', 1)
    path = tmp_path / "invalid.json"
    path.write_text(payload)
    with pytest.raises(ValueError, match="duplicate|finite"):
        figure.load_rows(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_wrong_hash_rejected_before_plot_or_output(evidence, tmp_path, monkeypatch):
    path, output = tmp_path / "input.json", tmp_path / "figure"
    save(path, evidence)
    monkeypatch.setattr(figure, "plot", lambda *args: pytest.fail("must not plot invalid input"))
    with pytest.raises(ValueError, match="SHA256"):
        figure.main(arguments(path, "0" * 64, output))
    assert not output.exists()


def test_artist_arrays_units_seed_means_and_unclipped_labels(evidence, tmp_path):
    path = tmp_path / "input.json"
    rows = figure.load_rows(path, save(path, evidence))
    before = copy.deepcopy(rows)
    fig = figure.plot(list(reversed(rows)))
    try:
        assert rows == before
        np.testing.assert_allclose(fig.get_size_inches()[0], 5.5)
        assert len(fig.axes) == 2
        assert fig.axes[0].get_ylabel() == "Energy score (standardized)"
        assert fig.axes[1].get_ylabel() == r"Mean absolute residual (nT$^2$)"
        assert fig.axes[1].get_title(loc="left") == "(b) Mean absolute residual"
        for ax, metric in zip(fig.axes, ("energy", "residual")):
            assert ax.get_xscale() == ax.get_yscale() == "linear"
            assert ax.get_xlabel() == r"Penalty coefficient $c_R$"
            np.testing.assert_array_equal(ax.get_xticks(), [0, 1, 2, 3])
            assert [label.get_text() for label in ax.get_xticklabels()] == list(LABELS)
            assert ax.get_xlim()[0] < 0 < ax.get_xlim()[1]
            assert len(ax.lines) == 4
            assert not ax.collections and not ax.patches and not ax.texts
            assert {line.get_color() for line in ax.lines[:3]} == {figure.COLORS[1]}
            assert len({line.get_linestyle() for line in ax.lines[:3]}) == 3
            assert len({line.get_marker() for line in ax.lines[:3]}) == 3
            for line, seed in zip(ax.lines[:3], SEEDS):
                assert line.get_label() == f"Seed {seed}"
                np.testing.assert_array_equal(line.get_xdata(), [0, 1, 2, 3])
                np.testing.assert_array_equal(line.get_ydata(),
                                              [r[metric] for r in rows if r["seed"] == seed])
            mean = ax.lines[3]
            assert mean.get_label() == "Mean across seeds"
            assert mean.get_color() == "black" and mean.get_marker() == "D"
            assert mean.get_linestyle() == "None"
            np.testing.assert_array_equal(mean.get_xdata(), [0, 1, 2, 3])
            np.testing.assert_allclose(mean.get_ydata(), [
                np.mean([r[metric] for r in rows if r["coefficient"] == c]) for c in COEFFICIENTS],
                rtol=1e-15, atol=0)
            assert ax.xaxis.label.get_fontsize() == ax.yaxis.label.get_fontsize() == 8
            assert ax.xaxis.label.get_fontfamily() == ["DejaVu Serif"]
            assert all(t.get_fontsize() == 7 for t in ax.get_xticklabels() + ax.get_yticklabels())
        assert [t.get_text() for t in fig.legends[0].get_texts()] == [
            "Seed 7", "Seed 13", "Seed 42", "Mean across seeds"]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = fig.bbox
        texts = [text for ax in fig.axes for text in (
            ax.xaxis.label, ax.yaxis.label, ax._left_title,
            *ax.get_xticklabels(), *ax.get_yticklabels())]
        for artist in [*texts, *fig.legends]:
            box = artist.get_window_extent(renderer)
            assert bounds.x0 <= box.x0 < box.x1 <= bounds.x1
            assert bounds.y0 <= box.y0 < box.y1 <= bounds.y1
    finally:
        figure.plt.close(fig)


@pytest.mark.parametrize("change", [
    lambda rows: rows.pop(),
    lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])),
    lambda rows: rows[0].update(coefficient=False),
    lambda rows: rows[0].update(seed=7.0),
    lambda rows: rows[0].update(energy=float("nan")),
    lambda rows: rows[0].update(residual=True),
    lambda rows: rows[0].update(mean=1.0),
])
def test_plot_helper_also_rejects_invalid_rows(evidence, change):
    rows = expected_rows(evidence)
    change(rows)
    with pytest.raises(ValueError):
        figure.plot(rows)


def test_fresh_render_and_exact_manifest(evidence, tmp_path, monkeypatch):
    path, output = tmp_path / "input.json", tmp_path / "figure"
    digest = save(path, evidence)
    real_plot = figure.plot

    def check_style(rows):
        assert figure.matplotlib.rcParams["pdf.fonttype"] == 42
        assert figure.matplotlib.rcParams["axes.unicode_minus"] is False
        assert figure.matplotlib.rcParams["font.size"] == 8
        fig = real_plot(rows)
        real_save = fig.savefig

        def check_save(*args, **kwargs):
            assert figure.matplotlib.rcParams["pdf.fonttype"] == 42
            assert figure.matplotlib.rcParams["font.family"] == ["DejaVu Serif"]
            return real_save(*args, **kwargs)

        fig.savefig = check_save
        return fig

    monkeypatch.setattr(figure, "plot", check_style)
    args = arguments(path, digest, output)
    assert figure.main(args) == 0
    manifest = json.loads((output / "fig_fixed_final.manifest.json").read_text())
    assert manifest["schema_version"] == 1 and manifest["figure"] == "fig_fixed_final"
    assert manifest["source"] == {"path": "fig_fixed_final.py",
                                  "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
    assert manifest["style"] == {"path": "prigm.mplstyle",
                                 "sha256": hashlib.sha256(SOURCE.with_name("prigm.mplstyle").read_bytes()).hexdigest()}
    assert manifest["inputs"] == {"fixed_final": {
        "path": "evidence/deadline/fixed-final.json", "sha256": digest}}
    assert manifest["plotted_rows"] == expected_rows(evidence)
    assert len(manifest["mean_rows"]) == 4
    for mean, coefficient in zip(manifest["mean_rows"], COEFFICIENTS):
        assert set(mean) == {"coefficient", "energy", "residual"}
        assert mean["coefficient"] == coefficient
        for metric in ("energy", "residual"):
            assert mean[metric] == pytest.approx(np.mean([
                r[metric] for r in expected_rows(evidence) if r["coefficient"] == coefficient]))
    assert manifest["axes"]["x"] == {"scale": "categorical", "positions": [0, 1, 2, 3],
                                       "coefficient_labels": list(LABELS),
                                       "coefficients": list(COEFFICIENTS)}
    assert manifest["axes"]["a"]["units"] == "TRAIN-standardized"
    assert manifest["axes"]["b"]["units"] == "nT^2"
    assert "Raw month-balanced TEST" in manifest["scope"]
    assert "no coefficient selection" in manifest["scope"]
    assert "not uncertainty intervals" in manifest["scope"]
    assert set(manifest["output_sha256"]) == {"fig_fixed_final.pdf", "fig_fixed_final.png"}
    for name, expected in manifest["output_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    assert (output / "fig_fixed_final.pdf").read_bytes().startswith(b"%PDF")
    assert (output / "fig_fixed_final.png").read_bytes().startswith(b"\x89PNG")
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        figure.main(args)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


@pytest.mark.parametrize("kind", ["directory", "file", "symlink", "dangling_symlink"])
def test_existing_output_paths_rejected_before_loading(tmp_path, monkeypatch, kind):
    output = tmp_path / "figure"
    if kind == "directory":
        output.mkdir()
    elif kind == "file":
        output.write_bytes(b"preserve")
    else:
        target = tmp_path / "target"
        if kind == "symlink":
            target.mkdir()
        output.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(figure, "load_rows", lambda *args: pytest.fail("must reject output first"))
    with pytest.raises(FileExistsError):
        figure.main(arguments(tmp_path / "absent.json", "0" * 64, output))
    if kind == "file":
        assert output.read_bytes() == b"preserve"
    elif "symlink" in kind:
        assert output.is_symlink()


def test_output_created_during_plot_is_not_reused(evidence, tmp_path, monkeypatch):
    path, output = tmp_path / "input.json", tmp_path / "figure"
    digest = save(path, evidence)
    real_plot = figure.plot

    def competing_output(rows):
        fig = real_plot(rows)
        output.mkdir()
        (output / "preserve").write_bytes(b"owned elsewhere")
        return fig

    monkeypatch.setattr(figure, "plot", competing_output)
    with pytest.raises(FileExistsError):
        figure.main(arguments(path, digest, output))
    assert [p.name for p in output.iterdir()] == ["preserve"]
    assert (output / "preserve").read_bytes() == b"owned elsewhere"


@pytest.mark.parametrize("target", ["source", "input", "style"])
@pytest.mark.parametrize("phase", ["plot", "last_save"])
def test_source_and_input_rechecked_after_render(evidence, tmp_path, monkeypatch, target, phase):
    path, output = tmp_path / "input.json", tmp_path / "figure"
    digest = save(path, evidence)
    source = tmp_path / "fig_fixed_final.py"
    original = SOURCE.read_bytes()
    source.write_bytes(original)
    style = source.with_name("prigm.mplstyle")
    style.write_bytes(SOURCE.with_name("prigm.mplstyle").read_bytes())
    monkeypatch.setattr(figure, "__file__", str(source))
    changed = {"source": source, "input": path, "style": style}[target]
    real_plot = figure.plot

    def mutate():
        changed.write_bytes(changed.read_bytes() + b"\n")

    def changing_plot(rows):
        fig = real_plot(rows)
        if phase == "plot":
            mutate()
        else:
            real_save = fig.savefig

            def changing_save(*args, **kwargs):
                real_save(*args, **kwargs)
                if kwargs["format"] == "png":
                    mutate()

            fig.savefig = changing_save
        return fig

    monkeypatch.setattr(figure, "plot", changing_plot)
    open_before = figure.plt.get_fignums()
    with pytest.raises(ValueError, match="source or input changed"):
        figure.main(arguments(path, digest, output))
    assert not (output / "fig_fixed_final.manifest.json").exists()
    assert figure.plt.get_fignums() == open_before
    assert SOURCE.read_bytes() == original


@pytest.mark.parametrize("missing", ["--evidence", "--evidence-sha256", "--output-dir"])
def test_cli_requires_all_arguments(tmp_path, missing):
    args = arguments(tmp_path / "absent.json", "0" * 64, tmp_path / "figure")
    index = args.index(missing)
    del args[index:index + 2]
    with pytest.raises(SystemExit) as exc:
        figure.main(args)
    assert exc.value.code == 2


def test_standalone_archive_script_has_no_repository_dependencies(evidence, tmp_path):
    imports = set()
    for node in ast.walk(ast.parse(SOURCE.read_text())):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imports.add(node.module.split(".")[0])
    assert imports <= sys.stdlib_module_names | {"numpy", "matplotlib"}
    script = tmp_path / "fig_fixed_final.py"
    script.write_bytes(SOURCE.read_bytes())
    script.with_name("prigm.mplstyle").write_bytes(SOURCE.with_name("prigm.mplstyle").read_bytes())
    path, output = tmp_path / "input.json", tmp_path / "figure"
    digest = save(path, evidence)
    env = {**os.environ, "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1",
           "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MPLCONFIGDIR": str(tmp_path / "mplconfig")}
    result = subprocess.run([sys.executable, "-B", str(script), *arguments(path, digest, output)],
                            cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "fig_fixed_final.manifest.json").is_file()
    assert not list(tmp_path.rglob("__pycache__"))


def test_refuses_nimbus_opentype_fallback(evidence, tmp_path, monkeypatch):
    def opentype_font(*args, **kwargs):
        assert kwargs["fallback_to_default"] is False
        return "/fonts/DejaVuSerif.otf"

    monkeypatch.setattr(figure.font_manager, "findfont", opentype_font)
    path = tmp_path / "input.json"
    with pytest.raises(ValueError, match="DejaVu Serif TrueType font required"):
        figure.main(arguments(path, save(path, evidence), tmp_path / "output"))
    assert not (tmp_path / "output").exists()
