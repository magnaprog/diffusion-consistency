"""Check that imputation plots preserve the supplied observations and predictions.

Synthetic arrays give cases, methods, coordinates, times and draws distinct
values. Permuted draw ranks separate the first three stored trajectories
from the median and interval endpoints computed by sorting. Negative
variance predictions are intentional: the renderer must display them.

Tests inspect the plotted lines and filled intervals, hashes, and axis
limits. The detail view keeps a shared scale for the three neural methods
and a separate visible right axis for Gaussian regression. Passing these
checks says nothing about the scientific calibration of the invented data.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "paper/workshops/prigm/fig_imputation.py"
spec = importlib.util.spec_from_file_location("prigm_imputation_figure", SOURCE)
figure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figure)
METHODS = ["D", "O-S", "G5", "full_flank_gaussian"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def bundle(tmp_path):
    # Distinct case/method/channel/time offsets; permuted draw ranks ensure
    # the displayed raw draws cannot be confused with order statistics.
    c, t, k = np.indices((2, 48, 6))
    truth = (300 + c * 40 + t * 0.25 + k * 10).astype(np.float64)
    truth[:, :, 5] = 1e9  # v_v must never affect plotted coordinates or limits.
    m, c, d, t, k = np.indices((4, 2, 39, 12, 5))
    samples = (m * 30 + c * 40 + ((d * 17) % 39) * 2 + t * 0.3 + k * 3).astype(np.float64)
    samples[:, :, :, :, 4] -= 220  # Retain negative variance, including extremes.
    arrays = {"truth": truth, "samples": samples,
              "ids": np.array(["case-A-$literal$", "case-B-unknown-date"]),
              "methods": np.array(METHODS)}
    record = {"schema_version": 1, "analysis": "frozen_raw_imputation_examples",
              "selection": {"rule": "midpoints_of_two_equal_ordered_TEST_partitions",
                            "zero_based_indices": [120, 360], "original_window_count": 480,
                            "neural_seed": 7, "displayed_draws": [0, 1, 2], "after_test_exposure": True},
              "case_ids": arrays["ids"].tolist(), "method_order": METHODS.copy(),
              "methods": {name: {} for name in METHODS},
              "channels": ["mu_x", "mu_y", "mu_z", "m", "v_s"],
              "units": ["nT", "nT", "nT", "nT", "nT^2"]}
    return arrays, record, tmp_path


def write_bundle(bundle):
    arrays, record, root = bundle
    excerpt, provenance = root / "sample-excerpt.npz", root / "MANIFEST.json"
    np.savez(excerpt, **arrays)
    record["files"] = {"sample-excerpt.npz": digest(excerpt)}
    provenance.write_text(json.dumps(record))
    return excerpt, digest(excerpt), provenance, digest(provenance)


@pytest.mark.parametrize("case_index,slot", [(120, 0), (360, 1)])
def test_actual_artists_match_independent_raw_arrays(bundle, case_index, slot):
    arrays, _, _ = bundle
    data = figure.load_cases(*write_bundle(bundle))
    before = {name: value.copy() for name, value in arrays.items()}
    fig = figure.plot_case(data, case_index)
    x = np.arange(48) * 5 - 90
    try:
        assert len(fig.axes) == 20
        assert tuple(fig.get_size_inches()) == (5.5, 7.5)
        title = {120: "ACE \u00b7 3 August 2010", 360: "Wind \u00b7 4 August 2020"}[case_index]
        assert fig._suptitle.get_text() == title
        assert not fig._suptitle.get_parse_math()
        assert [fig.axes[j].get_title().replace("\n", " ") for j in range(4)] == [
            "D", "O-S", "G5", "Gaussian regression"]
        assert len(fig.legends) == 1
        assert [t.get_text() for t in fig.legends[0].texts] == [
            "Available measurements", "Withheld measurements", "Predictive median",
            "Three ensemble members", "Nominal 90% marginal interval"]
        assert [text.get_text() for text in fig.texts] == [
            title, "Time from start of missing hour (min)"]
        for row in range(5):
            limits = [fig.axes[row * 4 + col].get_ylim() for col in range(4)]
            assert all(lim == limits[0] for lim in limits)
            for col in range(4):
                ax = fig.axes[row * 4 + col]
                raw = arrays["samples"][col, slot, :, :, row]
                ordered = np.sort(raw, axis=0)
                lines = {line.get_gid(): line for line in ax.lines}
                assert set(lines) == {"draw_0", "draw_1", "draw_2", "median",
                                      "observed_left", "observed_right", "withheld_truth"}
                for draw in range(3):
                    np.testing.assert_array_equal(lines[f"draw_{draw}"].get_ydata(), raw[draw])
                    np.testing.assert_array_equal(lines[f"draw_{draw}"].get_xdata(), x[18:30])
                np.testing.assert_array_equal(lines["median"].get_ydata(), ordered[19])
                np.testing.assert_array_equal(lines["median"].get_xdata(), x[18:30])
                for name, part in (("observed_left", slice(0, 18)),
                                   ("observed_right", slice(30, 48)),
                                   ("withheld_truth", slice(18, 30))):
                    line = lines[name]
                    np.testing.assert_array_equal(line.get_xdata(), x[part])
                    np.testing.assert_array_equal(line.get_ydata(), arrays["truth"][slot, part, row])
                    assert np.all(line.get_ydata() != 0)
                    assert line.get_color() == "black"
                    assert line.get_linestyle() == ("--" if name == "withheld_truth" else "-")
                assert len(ax.collections) == 1
                band = ax.collections[0]
                assert band.get_gid() == "nominal90"
                vertices = band.get_paths()[0].vertices
                for index, time in enumerate(x[18:30]):
                    ys = vertices[vertices[:, 0] == time, 1]
                    np.testing.assert_array_equal(np.unique(ys), [ordered[1, index], ordered[37, index]])
                assert set(vertices[:, 0]) == set(x[18:30])
                assert lines["median"].get_color() == figure.COLORS[col]
                assert ax.get_xlim() == (-90, 150)
                np.testing.assert_array_equal(ax.get_xticks(), [-60, 0, 60, 120])
                gap = ax.patches[0]
                assert gap.get_gid() == "gap"
                if hasattr(gap, "get_width"):
                    assert gap.get_x() == 0 and gap.get_width() == 60
                else:
                    assert set(gap.get_xy()[:, 0]) == {0, 60}
                assert ax.get_yscale() == ("symlog" if row == 4 else "linear")
                displayed = np.concatenate([line.get_ydata() for line in ax.lines] + [vertices[:, 1]])
                assert limits[col][0] < displayed.min() <= displayed.max() < limits[col][1]
                assert limits[col][1] < 1e9
                if row == 4:
                    assert limits[col][0] < 0
                    assert ax.get_yscale() == "symlog"
                    assert ax.yaxis.get_transform().linthresh == figure.VS_LINTHRESH
                    np.testing.assert_array_equal(ax.get_yticks(), figure.VS_TICKS)
                    if col == 0:
                        labels = [text.get_text() for text in ax.get_yticklabels()]
                        assert labels == ["-1", "-0.1", "0", "0.1", "1"]
                        assert all("$" not in text and r"\mathdefault" not in text for text in labels)
                        assert all(not text.get_parse_math() for text in ax.get_yticklabels())
                if col == 0:
                    assert ("nT$^2$" if row == 4 else "nT") in ax.get_ylabel()
                    assert ax.yaxis.label.get_fontsize() == 8
                    assert ax.yaxis.label.get_fontfamily() == ["DejaVu Serif"]
        for key in arrays:
            np.testing.assert_array_equal(data[key], before[key])
    finally:
        figure.plt.close(fig)


@pytest.mark.parametrize("key,change", [
    ("truth", lambda a: a.astype(np.float32)),
    ("truth", lambda a: a.astype(np.int64)),
    ("truth", lambda a: a.astype(bool)),
    ("truth", lambda a: a[:1]),
    ("truth", lambda a: a[:, :, :5]),
    ("truth", lambda a: np.full_like(a, np.nan)),
    ("truth", lambda a: np.full_like(a, np.inf)),
    ("samples", lambda a: a.astype(np.float32)),
    ("samples", lambda a: a[:, :, :38]),
    ("samples", lambda a: a[:, :, :, :11]),
    ("samples", lambda a: np.full_like(a, -np.inf)),
    ("ids", lambda a: a.astype("S")),
    ("ids", lambda a: a.astype(object)),
    ("ids", lambda a: a[:1]),
    ("ids", lambda a: np.array(["same", "same"])),
    ("ids", lambda a: np.array(["", "other"])),
    ("methods", lambda a: a[::-1]),
    ("methods", lambda a: a.astype("S")),
])
def test_invalid_arrays(bundle, key, change):
    bundle[0][key] = change(bundle[0][key])
    with pytest.raises(ValueError):
        figure.load_cases(*write_bundle(bundle))


@pytest.mark.parametrize("change", [
    lambda r: r.update(schema_version=True),
    lambda r: r.update(schema_version=1.0),
    lambda r: r.update(analysis="other"),
    lambda r: r.update(case_ids=["unknown", "ids"]),
    lambda r: r.update(method_order=METHODS[::-1]),
    lambda r: r["methods"].pop("G5"),
    lambda r: r["methods"].update(D=[]),
    lambda r: r.update(channels=["mu_x", "mu_y", "mu_z", "m", "v_v"]),
    lambda r: r.update(units=["nT"] * 5),
    lambda r: r["selection"].update(rule="picked_by_error"),
    lambda r: r["selection"].update(zero_based_indices=[360, 120]),
    lambda r: r["selection"].update(zero_based_indices=[120.0, 360]),
    lambda r: r["selection"].update(original_window_count=479),
    lambda r: r["selection"].update(neural_seed=13),
    lambda r: r["selection"].update(displayed_draws=[0, True, 2]),
    lambda r: r["selection"].update(displayed_draws=[0, 1, 3]),
    lambda r: r["selection"].update(displayed_draws=[0, 1, 1]),
    lambda r: r["selection"].update(after_test_exposure=1),
    lambda r: r["selection"].update(after_test_exposure=False),
])
def test_invalid_provenance(bundle, change):
    change(bundle[1])
    with pytest.raises(ValueError):
        figure.load_cases(*write_bundle(bundle))


def test_extra_archive_key_and_embedded_hash(bundle):
    bundle[0]["extra"] = np.zeros(1)
    with pytest.raises(ValueError, match="keys"):
        figure.load_cases(*write_bundle(bundle))
    bundle[0].pop("extra")
    excerpt, eh, provenance, _ = write_bundle(bundle)
    bundle[1]["files"]["sample-excerpt.npz"] = "0" * 64
    provenance.write_text(json.dumps(bundle[1]))
    with pytest.raises(ValueError, match="provenance"):
        figure.load_cases(excerpt, eh, provenance, digest(provenance))


@pytest.mark.parametrize("which", [1, 3])
def test_input_hashes(bundle, which):
    args = list(write_bundle(bundle))
    args[which] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        figure.load_cases(*args)


@pytest.mark.parametrize("addition", [', "schema_version":1}', ', "extra":NaN}',
                                       ', "extra":Infinity}', ', "extra":-Infinity}',
                                       ', "extra":1e999}', ', "extra":{"x":1,"x":2}}'])
def test_duplicate_and_nonfinite_json(bundle, addition):
    excerpt, eh, provenance, _ = write_bundle(bundle)
    provenance.write_text(provenance.read_text()[:-1] + addition)
    with pytest.raises(ValueError):
        figure.load_cases(excerpt, eh, provenance, digest(provenance))


@pytest.mark.parametrize("index", [0, 1, True, 120.0, "120", 361])
def test_invalid_case_index(bundle, index):
    data = figure.load_cases(*write_bundle(bundle))
    with pytest.raises(ValueError, match="case_index"):
        figure.plot_case(data, index)


def cli_args(inputs, output, index=120):
    excerpt, eh, provenance, ph = inputs
    return ["--excerpt", str(excerpt), "--excerpt-sha256", eh,
            "--provenance", str(provenance), "--provenance-sha256", ph,
            "--case-index", str(index), "--output-dir", str(output)]


def test_fresh_render_manifest_and_independent_coordinates(bundle):
    arrays, _, root = bundle
    inputs = write_bundle(bundle)
    for slot, index in enumerate((120, 360)):
        output = root / f"case-{index}"
        args = cli_args(inputs, output, index)
        assert figure.main(args) == 0
        manifest_path = output / "fig_imputation.manifest.json"
        text = manifest_path.read_text()
        manifest = json.loads(text)
        assert str(root) not in text
        assert manifest["schema_version"] == 1
        assert manifest["figure"] == "fig_imputation"
        assert manifest["source"] == {"path": "fig_imputation.py", "sha256": digest(SOURCE)}
        assert manifest["style"] == {"path": "prigm.mplstyle",
                                     "sha256": digest(SOURCE.with_name("prigm.mplstyle"))}
        assert manifest["inputs"] == {
            "excerpt": {"path": "evidence/imputation/sample-excerpt.npz", "sha256": inputs[1]},
            "provenance": {"path": "evidence/imputation/MANIFEST.json", "sha256": inputs[3]}}
        assert manifest["case_index"] == index
        assert manifest["case_id"] == arrays["ids"][slot]
        assert set(manifest["output_sha256"]) == {"fig_imputation.pdf", "fig_imputation.png"}
        for name, expected in manifest["output_sha256"].items():
            assert digest(output / name) == expected
        assert manifest["y_scale"] == [
            {"scale": "linear"}, {"scale": "linear"}, {"scale": "linear"}, {"scale": "linear"},
            {"scale": "symlog", "linthresh": 0.05, "ticks": [-1.0, -0.1, 0.0, 0.1, 1.0]}]
        coords = manifest["coordinates"]
        assert coords["ranks"] == {"draw_axis_in_excerpt": 2, "draw_count": 39, "zero_based_low": 1,
                                   "zero_based_median": 19, "zero_based_high": 37,
                                   "displayed_draws": [0, 1, 2], "interpolation": "none"}
        x = np.arange(48) * 5 - 90
        np.testing.assert_array_equal(coords["time_bin_start_minutes"], x)
        assert coords["gap_minutes"] == [0, 60]
        assert coords["context_intervals_half_open_minutes"] == [[-90, 0], [60, 150]]
        assert coords["truth_channel_indices"] == [0, 1, 2, 3, 4]
        for key, part in (("observed_left", slice(0, 18)), ("withheld_truth", slice(18, 30)),
                          ("observed_right", slice(30, 48))):
            np.testing.assert_array_equal(coords[key]["x"], x[part])
            np.testing.assert_array_equal(coords[key]["y"], arrays["truth"][slot, part, :5])
        for col, name in enumerate(METHODS):
            values = coords["methods"][name]
            raw = arrays["samples"][col, slot]
            ordered = np.sort(raw, axis=0)
            np.testing.assert_array_equal(values["x"], x[18:30])
            np.testing.assert_array_equal(values["draws"], raw[:3])
            for key, rank in (("low", 1), ("median", 19), ("high", 37)):
                np.testing.assert_array_equal(values[key], ordered[rank])
        hashes = {p.name: digest(p) for p in output.iterdir()}
        with pytest.raises(FileExistsError):
            figure.main(args)
        assert hashes == {p.name: digest(p) for p in output.iterdir()}


@pytest.mark.parametrize("kind", ["directory", "file", "dangling_symlink"])
def test_nonfresh_paths(bundle, kind):
    inputs = write_bundle(bundle)
    output = bundle[2] / "occupied"
    if kind == "directory":
        output.mkdir()
    elif kind == "file":
        output.write_text("keep")
    else:
        output.symlink_to(bundle[2] / "nonexistent")
    with pytest.raises(FileExistsError):
        figure.main(cli_args(inputs, output))


@pytest.mark.parametrize("target", ["excerpt", "provenance", "source", "style"])
def test_mutation_during_render_refused(bundle, monkeypatch, target):
    inputs = write_bundle(bundle)
    original_plot = figure.plot_case
    source_copy = bundle[2] / "source-copy.py"
    source_copy.write_bytes(SOURCE.read_bytes())
    style = source_copy.with_name("prigm.mplstyle")
    style.write_bytes(SOURCE.with_name("prigm.mplstyle").read_bytes())
    monkeypatch.setattr(figure, "__file__", str(source_copy))
    path = {"excerpt": inputs[0], "provenance": inputs[2], "source": source_copy, "style": style}[target]

    def changed_plot(data, index):
        fig = original_plot(data, index)
        path.write_bytes(path.read_bytes() + b" ")
        return fig

    monkeypatch.setattr(figure, "plot_case", changed_plot)
    output = bundle[2] / "changed-output"
    with pytest.raises(ValueError, match="SHA256"):
        figure.main(cli_args(inputs, output))
    assert not (output / "fig_imputation.manifest.json").exists()


def test_detail_artists_show_all_intervals_on_declared_scales(bundle):
    arrays, _, _ = bundle
    arrays["samples"][3, :, :, :, 4] *= 100
    data = figure.load_cases(*write_bundle(bundle))
    original_samples = data["samples"].copy()
    fig = figure.plot_detail(data)
    cases = figure.detail_coordinates(data)
    try:
        assert len(fig.axes) == 8
        for slot, case in enumerate(cases):
            ordered = np.sort(arrays["samples"][:, slot, :, :, 4], axis=1)
            for col, name in enumerate(METHODS):
                group = slice(0, 3) if col < 3 else slice(3, 4)
                bounds = np.concatenate([arrays["truth"][slot, 16:32, 4],
                                         ordered[group, (1, 37)].reshape(-1)])
                pad = .08 * max(np.ptp(bounds), 1e-6)
                expected = [bounds.min() - pad, bounds.max() + pad]
                np.testing.assert_array_equal(case["y_limits"][name], expected)
                ax = fig.axes[slot * 4 + col]
                assert ax.get_yscale() == "linear"
                if col == 0:
                    label = ("ACE \u00b7 3 Aug 2010", "Wind \u00b7 4 Aug 2020")[slot]
                    assert ax.get_ylabel() == label + "\n" + r"$v_s$ (nT$^2$)"
                    assert ax.yaxis.label.get_parse_math()
                elif col == 3:
                    assert ax.yaxis.get_ticks_position() == "right"
                    assert not ax.spines["left"].get_visible()
                    assert ax.spines["right"].get_visible()
                else:
                    assert not any(t.get_visible() for t in ax.get_yticklabels())
                np.testing.assert_array_equal(ax.get_ylim(), expected)
                assert ax.get_xlim() == (-10, 65)
                lines = {line.get_gid(): line for line in ax.lines}
                assert set(lines) == {"median", "zero", "observed_left", "observed_right", "withheld_truth"}
                np.testing.assert_array_equal(lines["median"].get_ydata(), ordered[col, 19])
                np.testing.assert_array_equal(lines["median"].get_xdata(), np.arange(0, 60, 5))
                np.testing.assert_array_equal(lines["zero"].get_ydata(), [0, 0])
                vertices = ax.collections[0].get_paths()[0].vertices
                for i, x in enumerate(range(0, 60, 5)):
                    np.testing.assert_array_equal(np.unique(vertices[vertices[:, 0] == x, 1]),
                                                  ordered[col, [1, 37], i])
                for key, part in (("observed_left", slice(16, 18)),
                                  ("withheld_truth", slice(18, 30)), ("observed_right", slice(30, 32))):
                    np.testing.assert_array_equal(lines[key].get_ydata(), arrays["truth"][slot, part, 4])
                    np.testing.assert_array_equal(lines[key].get_xdata(), np.arange(48)[part] * 5 - 90)
                displayed = np.concatenate([ordered[col, [1, 19, 37]].ravel(),
                                            arrays["truth"][slot, 16:32, 4]])
                assert expected[0] < displayed.min() <= displayed.max() < expected[1]
            assert case["y_limits"][METHODS[0]] == case["y_limits"][METHODS[1]] == case["y_limits"][METHODS[2]]
            assert case["y_limits"][METHODS[3]] != case["y_limits"][METHODS[0]]
        assert fig.texts[2].get_text() == "Time from start of missing hour (min)"
        assert "Gaussian uses the right axis" in fig.texts[1].get_text()
        assert len(fig.legends) == 1
        assert [t.get_text() for t in fig.legends[0].texts] == [
            "Available measurements", "Withheld measurements", "Predictive median", "Nominal 90% interval"]
        renderer = fig.canvas.get_renderer()
        for text in fig.texts:
            assert fig.bbox.contains(*text.get_window_extent(renderer).get_points()[0])
            assert fig.bbox.contains(*text.get_window_extent(renderer).get_points()[1])
        np.testing.assert_array_equal(data["samples"], original_samples)
    finally:
        figure.plt.close(fig)


def test_detail_cli_manifest(bundle):
    inputs = write_bundle(bundle)
    output = bundle[2] / "detail"
    args = cli_args(inputs, output)
    pos = args.index("--case-index")
    args[pos:pos + 2] = ["--detail"]
    assert figure.main(args) == 0
    manifest = json.loads((output / "fig_imputation.manifest.json").read_text())
    assert manifest["cases"] == figure.detail_coordinates(figure.load_cases(*inputs))
    assert manifest["y_scale"] == "linear"
    assert manifest["style"] == {"path": "prigm.mplstyle",
                                 "sha256": digest(SOURCE.with_name("prigm.mplstyle"))}
    for name, expected in manifest["output_sha256"].items():
        assert digest(output / name) == expected
    with pytest.raises(FileExistsError):
        figure.main(args)


def test_source_is_ascii():
    SOURCE.read_bytes().decode("ascii")
    Path(__file__).read_bytes().decode("ascii")


@pytest.mark.parametrize("detail", [False, True])
def test_save_keeps_shared_font_context(bundle, monkeypatch, detail):
    inputs = write_bundle(bundle)
    output = bundle[2] / "font-context"
    args = cli_args(inputs, output)
    if detail:
        pos = args.index("--case-index")
        args[pos:pos + 2] = ["--detail"]
    real_save = figure.matplotlib.figure.Figure.savefig
    formats = []

    def checked_save(fig, *args, **kwargs):
        assert figure.matplotlib.rcParams["pdf.fonttype"] == 42
        assert figure.matplotlib.rcParams["font.family"] == ["DejaVu Serif"]
        assert figure.matplotlib.rcParams["mathtext.fontset"] == "dejavuserif"
        formats.append(kwargs["format"])
        return real_save(fig, *args, **kwargs)

    monkeypatch.setattr(figure.matplotlib.figure.Figure, "savefig", checked_save)
    assert figure.main(args) == 0
    assert formats == ["pdf", "png"]


@pytest.mark.parametrize("detail", [False, True])
def test_refuses_nimbus_opentype_fallback(bundle, monkeypatch, detail):
    args = cli_args(write_bundle(bundle), bundle[2] / "output")
    if detail:
        pos = args.index("--case-index")
        args[pos:pos + 2] = ["--detail"]

    def opentype_font(*args, **kwargs):
        assert kwargs["fallback_to_default"] is False
        return "/fonts/DejaVuSerif.otf"

    monkeypatch.setattr(figure.font_manager, "findfont", opentype_font)
    with pytest.raises(ValueError, match="DejaVu Serif TrueType font required"):
        figure.main(args)
    assert not (bundle[2] / "output").exists()
