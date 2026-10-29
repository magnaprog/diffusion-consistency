"""Check plotted score signs, tail coordinates, and reproducible input binding."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "paper/workshops/prigm/fig_calibration.py"
spec = importlib.util.spec_from_file_location("prigm_calibration_figure", SOURCE)
figure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figure)


@pytest.fixture
def records():
    operator = {"schema_version": 1, "analysis": "paired_operator_mechanism",
                "protocol_sha256": figure.PROTOCOL, "n_windows": 480, "methods": {}}
    for name, target, dispersion in (("D", [3., 4., 5.], [1., 2., 3.]),
                                     ("G5", [3., 3.6, 5.3], [1.2, 1.65, 3.35])):
        metrics = []
        for t, d in zip(target, dispersion):
            metrics.append({"raw_target_term": {"month_balanced": t},
                            "raw_dispersion_term": {"month_balanced": d},
                            "raw_energy_score": {"month_balanced": t - d}})
        operator["methods"][name] = {"n_realizations": 3,
            "seedwise": {str(s): {"metrics": m} for s, m in zip(figure.SEEDS, metrics)},
            "metrics": {k: {"month_balanced": np.mean([m[k]["month_balanced"] for m in metrics])}
                        for k in metrics[0]}}
    calibration = {"schema_version": 1, "n_windows": 480, "n_draws": 39,
        "channels": list(figure.CHANNELS), "outcomes": ["below", "inside", "above"],
        "interval_order_statistics": [2, 38], "continuous_exchangeability_reference": [.05, .9, .05],
        "bootstrap": {"replicates": 10000, "seed": 7003, "schemes": ["year", "twoyear"]},
        "per_window_sha256": "a" * 64, "methods": {}}
    for i, name in enumerate(sorted(figure.ALL_METHODS)):
        seeds = list(figure.SEEDS) if name in ("D", "I-S", "O-S", "G5") else [None]
        below = np.array([.03, .06, .09, .12, .50]) + .01 * i
        above = np.array([.15, .13, .11, .09, .02]) + .003 * i
        mean = np.stack([below, 1 - below - above, above], axis=-1)
        models = np.tile(mean, (len(seeds), 1, 1))
        if len(seeds) == 3:
            models[0, :, 0] += .02
            models[0, :, 1] -= .02
            models[2, :, 0] -= .02
            models[2, :, 1] += .02
        if name == "G5":
            models[:, 4] = [[.68, .29, .03], [.78, .18, .04], [.01, .75, .24]]
        mean = models.mean(axis=0)
        ci = np.stack([np.maximum(0, mean - .015), np.minimum(1, mean + .02)])
        contrast = mean[:, 0] - mean[:, 2]
        calibration["methods"][name] = {"fixed_model_seeds": seeds,
            "model_means": models.tolist(), "mean": mean.tolist(),
            "year_95": ci.tolist(), "below_minus_above": contrast.tolist(),
            "year_below_minus_above_95": np.stack([contrast - .02, contrast + .02]).tolist()}
    return operator, calibration


def save(path, record):
    path.write_text(json.dumps(record))
    return figure.sha256(path)


def inputs(tmp_path, records):
    operator, calibration = tmp_path / "operator.json", tmp_path / "summary.json"
    return operator, save(operator, records[0]), calibration, save(calibration, records[1])


def args(paths, output):
    op, oh, cal, ch = paths
    return ["--operator", str(op), "--operator-sha256", oh, "--calibration", str(cal),
            "--calibration-sha256", ch, "--output-dir", str(output)]


def test_decomposition_signs_seed_order_and_mean(records, tmp_path):
    op, oh, _, _ = inputs(tmp_path, records)
    rows = figure.load_decomposition(op, oh)
    assert [r["seed"] for r in rows] == [7, 13, 42, "mean"]
    np.testing.assert_allclose([r["delta_target"] for r in rows], [0, -.4, .3, -.1 / 3], atol=1e-14)
    np.testing.assert_allclose([r["negative_delta_dispersion"] for r in rows], [-.2, .35, -.35, -.2 / 3], atol=1e-14)
    np.testing.assert_allclose([r["delta_energy"] for r in rows], [-.2, -.05, -.05, -.1], atol=1e-14)


def test_artists_keep_tail_direction_percent_scale_and_each_unpooled_seed(records, tmp_path):
    # Numerical confidence intervals are retained but must not affect the drawn marks.
    records[1]["methods"]["D"]["year_95"][0][0][0] = .20
    records[1]["methods"]["D"]["year_95"][1][0][0] = .25
    op, oh, cal, ch = inputs(tmp_path, records)
    rows = figure.load_decomposition(op, oh)
    tails, _ = figure.load_calibration(cal, ch)
    fig = figure.plot(rows, tails)
    try:
        np.testing.assert_allclose(fig.get_size_inches(), [5.5, 5])
        for ax, outcome in zip(fig.axes[1:], (0, 2)):
            for name in figure.METHODS:
                bars = [next(p for p in ax.patches if p.get_gid() == f"{name}_{outcome}_mean_{channel}")
                        for channel in range(5)]
                expected = 100 * np.array(records[1]["methods"][name]["mean"])[:, outcome]
                np.testing.assert_allclose([bar.get_height() for bar in bars], expected)
                if name != "full_flank_gaussian":
                    for index, seed in enumerate(figure.SEEDS):
                        line = next(x for x in ax.lines if x.get_gid() == f"{name}_{outcome}_seed_{seed}")
                        expected = 100 * np.array(records[1]["methods"][name]["model_means"])[index, :, outcome]
                        np.testing.assert_allclose(line.get_ydata(), expected)
            assert not ax.collections  # No interval segments or pooled-sample scatter.
            reference = next(x for x in ax.lines if x.get_gid() == "tail_reference")
            np.testing.assert_array_equal(reference.get_ydata(), [5, 5])
        assert fig.axes[1].get_ylim() == fig.axes[2].get_ylim()
        note = fig.texts[-1]
        assert note.get_text() == (
            "The 5% line assumes that, at each time and coordinate,\n"
            "the observation and ensemble members are exchangeable, with no ties.")
        assert note.get_fontsize() == 7
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = note.get_window_extent(renderer)
        assert fig.bbox.contains(bounds.x0, bounds.y0)
        assert fig.bbox.contains(bounds.x1, bounds.y1)
        assert bounds.y1 < fig.texts[-2].get_window_extent(renderer).y0
    finally:
        figure.plt.close(fig)


@pytest.mark.parametrize("change", [
    lambda d: d.update(n_windows=479),
    lambda d: d.update(interval_order_statistics=[1, 37]),
    lambda d: d.update(outcomes=["above", "inside", "below"]),
    lambda d: d["methods"]["G5"].update(fixed_model_seeds=[7, 13]),
    lambda d: d["methods"]["D"]["mean"][0].__setitem__(0, .9),
    lambda d: d["methods"]["D"]["year_95"][0][0].__setitem__(0, .9),
    lambda d: d["methods"]["D"]["year_95"][0][0].__setitem__(0, float("nan")),
    lambda d: d["methods"]["D"]["below_minus_above"].__setitem__(0, .9),
])
def test_rehashed_incompatible_calibration_rejected(records, tmp_path, change):
    record = copy.deepcopy(records[1])
    change(record)
    path = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        figure.load_calibration(path, save(path, record))


def test_hash_and_operator_identity_fail_before_output(records, tmp_path):
    paths = inputs(tmp_path, records)
    paths[0].write_text(paths[0].read_text() + " ")
    with pytest.raises(ValueError, match="SHA256"):
        figure.main(args(paths, tmp_path / "new"))
    assert not (tmp_path / "new").exists()
    records[0]["methods"]["D"]["seedwise"]["7"]["metrics"]["raw_energy_score"]["month_balanced"] = 9
    digest = save(paths[0], records[0])
    with pytest.raises(ValueError, match="score identity"):
        figure.load_decomposition(paths[0], digest)


def test_repeatable_render_and_numeric_manifest(records, tmp_path):
    paths = inputs(tmp_path, records)
    manifests = []
    for suffix in ("first", "repeat"):
        output = tmp_path / suffix
        assert figure.main(args(paths, output)) == 0
        m = json.loads((output / "fig_calibration.manifest.json").read_text())
        assert m["source"]["sha256"] == figure.sha256(SOURCE)
        assert m["inputs"]["operator"]["sha256"] == paths[1]
        assert m["inputs"]["calibration"]["sha256"] == paths[3]
        assert m["decomposition"] == figure.load_decomposition(*paths[:2])
        assert m["tails"] == figure.load_calibration(*paths[2:])[0]
        assert m["calibration_provenance"]["per_window_sha256"] == "a" * 64
        assert m["pointwise_intervals_plotted"] is False
        assert m["tail_reference"] == {"percent": 5, "assumption": (
            "at each time and coordinate, the observation and 39 ensemble members "
            "are exchangeable, with no ties")}
        for name, digest in m["output_sha256"].items():
            assert figure.sha256(output / name) == digest
        manifests.append(m)
    assert manifests[0] == manifests[1]
    with pytest.raises(FileExistsError):
        figure.main(args(paths, tmp_path / "first"))


def test_input_change_during_render_prevents_manifest(records, tmp_path, monkeypatch):
    paths = inputs(tmp_path, records)
    original = figure.plot
    def changed(*values):
        fig = original(*values)
        paths[2].write_text(paths[2].read_text() + " ")
        return fig
    monkeypatch.setattr(figure, "plot", changed)
    with pytest.raises(ValueError, match="changed during rendering"):
        figure.main(args(paths, tmp_path / "changed"))
    assert not (tmp_path / "changed/fig_calibration.manifest.json").exists()
