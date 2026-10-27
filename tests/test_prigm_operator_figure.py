"""Check that the clipping-map figure keeps score and selection effects separate.

Invented records distinguish corrected-minus-raw score at the same selected
weights from the additional effect of choosing a different D checkpoint.
Tests check the target-distance and dispersion signs and reject inconsistent
component sums, missing seeds and altered input identities.

The figures use these records to test coordinates and validation behavior,
not to establish an empirical benefit from clipping.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "paper/workshops/prigm/fig_operator.py"
spec = importlib.util.spec_from_file_location("prigm_operator_figure", SOURCE)
figure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figure)


@pytest.fixture
def evidence():
    methods = {}
    for name in (*figure.METHODS, "interpolation", "persistence"):
        seeds = figure.SEEDS if name in ("D", "I-S", "O-S", "G5") else ("shared",)
        metrics = {"mean_movement": {"month_balanced": 0.1},
                   "delta_target_term": {"month_balanced": -0.01},
                   "delta_dispersion_term": {"month_balanced": -0.002},
                   "delta_energy_score": {"month_balanced": -0.008},
                   "raw_support_violation_fraction": {"month_balanced": 0.5},
                   "raw_m_minus_r_severity_quantiles": {"month_balanced": [0, 0.01, 0.02, 0.03, 0.04]}}
        methods[name] = {"n_realizations": len(seeds), "metrics": metrics,
                         "seedwise": {seed: {"metrics": copy.deepcopy(metrics)} for seed in seeds}}
    return {"schema_version": 1, "analysis": "paired_operator_mechanism", "protocol_sha256": figure.PROTOCOL,
            "n_windows": 480, "methods": methods, "D_selection_decomposition": {
                "seedwise": {seed: {} for seed in figure.SEEDS}, "metrics": {
                    "fixed_checkpoint_correction": {"month_balanced": -0.008},
                    "selection_effect": {"month_balanced": -0.012},
                    "total_change": {"month_balanced": -0.02}}}}


def save(path, value):
    payload = json.dumps(value).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_plotted_signs_and_distinct_selection(evidence, tmp_path):
    path = tmp_path / "input.json"
    rows, selection = figure.load_rows(path, save(path, evidence))
    assert [r["method"] for r in rows] == list(figure.METHODS)
    assert rows[0]["negative_delta_dispersion"] == 0.002
    assert rows[0]["delta_energy"] == -0.008
    assert selection["selection_effect"] == -0.012
    assert rows[0]["delta_energy"] != selection["total_change"]


@pytest.mark.parametrize("change", [
    lambda d: d.update(protocol_sha256="0" * 64),
    lambda d: d.update(n_windows=479),
    lambda d: d["methods"].pop("G5"),
    lambda d: d["methods"]["D"]["seedwise"].pop("7"),
    lambda d: d["methods"]["D"]["metrics"]["delta_energy_score"].update(month_balanced=-0.009),
    lambda d: d["methods"]["D"]["metrics"]["mean_movement"].update(month_balanced=0.2),
    lambda d: d["methods"]["D"]["metrics"]["delta_target_term"].update(month_balanced=float("nan")),
    lambda d: d["D_selection_decomposition"]["metrics"]["total_change"].update(month_balanced=-0.1),
    lambda d: d["D_selection_decomposition"]["seedwise"].pop("42"),
])
def test_changed_scientific_inputs_refused(evidence, tmp_path, change):
    change(evidence)
    path = tmp_path / "changed.json"
    digest = save(path, evidence)
    with pytest.raises(ValueError):
        figure.load_rows(path, digest)


def test_hash_and_duplicate_keys(evidence, tmp_path):
    path = tmp_path / "input.json"
    save(path, evidence)
    with pytest.raises(ValueError, match="SHA256"):
        figure.load_rows(path, "0" * 64)
    payload = path.read_bytes()[:-1] + b', "n_windows":480}'
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="duplicate"):
        figure.load_rows(path, hashlib.sha256(payload).hexdigest())


def test_axes_preserve_signed_values(evidence, tmp_path):
    path = tmp_path / "input.json"
    rows, _ = figure.load_rows(path, save(path, evidence))
    fig = figure.plot(rows)
    try:
        assert len(fig.axes) == 2
        assert fig.axes[0].get_xscale() == "log"
        assert fig.axes[1].get_xscale() == "symlog"
        np.testing.assert_allclose([p.get_width() for p in fig.axes[1].patches[:5]], [-0.01] * 5)
        np.testing.assert_allclose([p.get_width() for p in fig.axes[1].patches[5:]], [0.002] * 5)
        assert "Gaussian\ninterpolation" in [label.get_text() for label in fig.axes[0].get_yticklabels()]
        assert fig.axes[0].get_ylim() == fig.axes[1].get_ylim()
        assert fig.axes[0]._left_title.get_text() == "(a) Mean displacement"
        assert fig.axes[1].texts and fig.axes[1].texts[0].get_text() == "-0.0080"
    finally:
        figure.plt.close(fig)


def test_fresh_render_and_manifest(evidence, tmp_path):
    path = tmp_path / "input.json"
    digest = save(path, evidence)
    output = tmp_path / "figure"
    args = ["--evidence", str(path), "--evidence-sha256", digest, "--output-dir", str(output)]
    assert figure.main(args) == 0
    manifest = json.loads((output / "fig_operator.manifest.json").read_text())
    assert manifest["source"]["sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert manifest["inputs"]["operator"]["sha256"] == digest
    for name, expected in manifest["output_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    assert manifest["D_selection"]["total_change"] == -0.02
    assert "not uncertainty intervals" in manifest["scope"]
    before = (output / "fig_operator.pdf").read_bytes()
    with pytest.raises(FileExistsError):
        figure.main(args)
    assert (output / "fig_operator.pdf").read_bytes() == before


def test_released_legend_does_not_cover_bars():
    manifest = SOURCE.parent / "figures/operator/fig_operator.manifest.json"
    rows = json.loads(manifest.read_text())["plotted_rows"]
    fig = figure.plot(rows)
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axis = fig.axes[1]
        legend = axis.get_legend().get_window_extent(renderer)
        assert all(not legend.overlaps(bar.get_window_extent(renderer)) for bar in axis.patches)
    finally:
        figure.plt.close(fig)
