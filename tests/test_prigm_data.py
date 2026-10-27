"""Synthetic data-path tests: bins, windows, masks, no real OMNI required."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prigm.calendar import timestamp_from_fields
from prigm.config import HIDDEN_END, HIDDEN_START, N_BINS, ROLE_TRAIN
from prigm.data import (
    Bin,
    consecutive_runs,
    context_mask,
    form_bins,
    pack_nonoverlapping,
    parse_omni_year,
    relative_positions,
    vector_valid,
    windows_from_run,
)
from prigm.normalize import mask_hidden
from prigm.quality import mag_valid, parse_quality_year


def _minute_dtype():
    return np.dtype(
        [
            ("year", np.int32),
            ("doy", np.int32),
            ("hour", np.int32),
            ("minute", np.int32),
            ("source", np.int32),
            ("bx", np.float64),
            ("by", np.float64),
            ("bz", np.float64),
            ("ts", "datetime64[m]"),
        ]
    )


def make_minutes(n, year=2010, doy=20, source=71, start_hour=0, start_min=0, fill_at=None):
    rows = []
    for i in range(n):
        total = start_hour * 60 + start_min + i
        hour, minute = divmod(total, 60)
        bx = by = bz = 1.0
        src = source
        if fill_at is not None and i == fill_at:
            bx = 9999.99
        ts = timestamp_from_fields(year, doy, hour, minute)
        rows.append((year, doy, hour, minute, src, bx, by, bz, ts))
    return np.array(rows, dtype=_minute_dtype())


def test_bins_require_five_valid_same_source():
    m = make_minutes(10)
    bins = form_bins(m)
    assert len(bins) == 2
    assert bins[0].z.shape == (6,)
    m_bad = make_minutes(10, fill_at=2)
    bins_bad = form_bins(m_bad)
    # first bin invalid, second starting at minute 5 still complete
    assert len(bins_bad) == 1
    assert bins_bad[0].left == bins[1].left
    np.testing.assert_array_equal(bins_bad[0].z, bins[1].z)
    for b in bins:
        np.testing.assert_array_equal(b.z, [1.0, 1.0, 1.0, np.sqrt(3.0), 0.0, 0.0])


def test_source_transition_splits_run():
    a = make_minutes(10, source=71)
    b = make_minutes(10, source=52, start_min=10)
    m = np.concatenate([a, b])
    bins = form_bins(m)
    runs = consecutive_runs(bins)
    assert len(runs) == 2
    assert runs[0][0].source != runs[-1][0].source


def test_windows_hidden_mask():
    m = make_minutes(48 * 5)
    bins = form_bins(m)
    wins = windows_from_run(bins, stride=1)
    assert len(wins) == 1
    assert wins[0].z.shape == (48, 6)
    packed = pack_nonoverlapping(bins)
    assert len(packed) == 1
    u = np.ones((48, 6))
    masked = mask_hidden(u)
    assert np.all(masked[HIDDEN_START:HIDDEN_END] == 0)
    assert np.all(masked[:HIDDEN_START] == 1)
    cm = context_mask()
    assert cm[17, 0] == 1 and cm[18, 0] == 0 and cm[29, 0] == 0 and cm[30, 0] == 1
    rp = relative_positions()
    assert rp.shape == (48, 1)


def test_real_omni_2005_schema():
    from prigm.data import omni_path

    path = omni_path(2005, Path(PROJECT_ROOT) / "data/omni")
    if not path.is_file():
        pytest.skip("local 2005 OMNI cache not present")
    arr = parse_omni_year(path)
    assert arr.size == 365 * 24 * 60
    assert np.all(np.diff(arr["ts"]) == np.timedelta64(1, "m"))
    first = arr[0]
    with path.open() as f:
        parts = next(f).split()
    assert int(parts[4]) == first["source"]
    assert abs(float(parts[14]) - first["bx"]) < 1e-12
    assert abs(float(parts[15]) - first["by"]) < 1e-12
    assert abs(float(parts[16]) - first["bz"]) < 1e-12


def test_parse_ascii(tmp_path):
    line = (
        "2010  20  0  0 71 71   3   1 100   1865      4  0.00     41    7.07"
        "   -6.63    2.37   -0.66    2.46   -0.00    0.02    0.17   411.8"
        "  -410.7    -2.7    29.0   5.06  157004.  1.72   0.11   1.23   6.6"
        "  231.43  -39.66   19.35   13.40   -0.89   -0.94    68   -58    10"
        "    -2    -2    12    16  9.999  4.6\n"
    )
    p = tmp_path / "omni_min2010.asc"
    p.write_text(line)
    arr = parse_omni_year(p)
    assert arr.size == 1
    assert arr[0]["source"] == 71
    # Zero-based field index 13 contains the native scalar magnitude.
    parts = line.split()
    assert tuple(arr[0][k] for k in ("bx", "by", "bz")) == tuple(map(float, parts[14:17]))
    assert arr[0]["ts"] == np.datetime64("2010-01-20T00:00", "m")


def write_omni_fixture(path, minutes):
    """Only the documented magnetic and quality columns are needed by these readers."""
    raw = np.zeros((len(minutes), 17), dtype=np.float64)
    for col, name in zip((0, 1, 2, 3, 4, 14, 15, 16),
                         ("year", "doy", "hour", "minute", "source", "bx", "by", "bz")):
        raw[:, col] = minutes[name]
    raw[:, 6] = 3  # IMF points
    raw[:, 8] = 0  # PFN percent interpolation
    raw[:, 12] = 10  # DBOT
    np.savetxt(path, raw, fmt="%.17g")
    return raw


@pytest.mark.parametrize("parser", [parse_omni_year, parse_quality_year])
@pytest.mark.parametrize("col,value,field", [
    (0, 2010.5, "year"), (0, np.nan, "year"), (0, np.inf, "year"),
    (0, 0, "year"), (0, -1, "year"), (0, 10000, "year"),
    (1, 0, "doy"), (1, 366, "doy"), (1, 367, "doy"), (1, 20.5, "doy"),
    (2, -1, "hour"), (2, 24, "hour"), (2, 0.5, "hour"), (2, np.nan, "hour"),
    (3, -1, "minute"), (3, 60, "minute"), (3, 0.5, "minute"),
    (4, 0, "source"), (4, 51, "source"), (4, 72, "source"),
    (4, 71.5, "source"), (4, 99.5, "source"), (4, np.nan, "source"),
])
def test_parsers_reject_invalid_fields_before_cast(tmp_path, parser, col, value, field):
    path = tmp_path / "bad.asc"
    raw = write_omni_fixture(path, make_minutes(1))
    raw[0, col] = value
    np.savetxt(path, raw, fmt="%.17g")
    with pytest.raises(ValueError, match=field):
        parser(path)


@pytest.mark.parametrize("parser", [parse_omni_year, parse_quality_year])
@pytest.mark.parametrize("year,doy,valid", [(2000, 366, True), (2012, 366, True),
                                          (1900, 366, False), (2012, 367, False)])
def test_parser_leap_year_doy(tmp_path, parser, year, doy, valid):
    path = tmp_path / "leap.asc"
    raw = write_omni_fixture(path, make_minutes(1))
    raw[0, :4] = [year, doy, 23, 59]
    np.savetxt(path, raw, fmt="%.17g")
    if valid:
        assert parser(path)[0]["ts"] == np.datetime64(f"{year}-12-31T23:59", "m")
    else:
        with pytest.raises(ValueError, match="doy"):
            parser(path)


@pytest.mark.parametrize("source", [50, 52, 60, 71, 99])
def test_documented_sources_and_source_fill(tmp_path, source):
    minutes = make_minutes(5, source=source)
    path = tmp_path / "source.asc"
    write_omni_fixture(path, minutes)
    parsed = parse_omni_year(path)
    quality = parse_quality_year(path)
    np.testing.assert_array_equal(parsed, minutes)
    for name in parsed.dtype.names:
        np.testing.assert_array_equal(quality[name], parsed[name])
    assert len(form_bins(parsed)) == (0 if source == 99 else 1)
    assert vector_valid(parsed[0]) == (source != 99)
    assert mag_valid(quality).all() == (source != 99)


@pytest.mark.parametrize("kind", ["duplicate_left", "duplicate_end", "out_of_order"])
def test_duplicate_or_unordered_minutes_rejected_everywhere(tmp_path, kind):
    m = make_minutes(10)
    if kind == "duplicate_left":
        m = np.concatenate((m[:1], m))
    elif kind == "duplicate_end":
        m = np.concatenate((m[:5], m[4:]))
    else:
        m[[3, 4]] = m[[4, 3]]
    path = tmp_path / "ambiguous.asc"
    write_omni_fixture(path, m)
    for parser in (parse_omni_year, parse_quality_year):
        with pytest.raises(ValueError, match="unique and strictly increasing"):
            parser(path)
    with pytest.raises(ValueError, match="unique and strictly increasing"):
        form_bins(m)


@pytest.mark.parametrize("field,value,match", [("doy", 366, "doy"), ("source", 72, "source"),
                                               ("ts", np.datetime64("NaT"), "timestamp"),
                                               ("minute", 1, "timestamp")])
def test_bins_validate_in_memory_records(field, value, match):
    m = make_minutes(1)
    m[field][0] = value
    with pytest.raises(ValueError, match=match):
        form_bins(m)


@pytest.mark.parametrize("value,valid", [(9999.99, False), (np.nan, False), (np.inf, False),
                                        (-np.inf, False), (9999.9, True), (10000.0, True),
                                        (-9999.99, True), (-10000.0, True), (70.01, True)])
@pytest.mark.parametrize("component", ["bx", "by", "bz"])
def test_exact_component_fill_no_amplitude_filter(component, value, valid):
    m = make_minutes(5)
    m[component][2] = value
    assert vector_valid(m[2]) == valid
    assert bool(mag_valid(m)[2]) == valid
    assert len(form_bins(m)) == int(valid)


def test_utc_bin_alignment_missing_minute_and_mixed_source():
    m = make_minutes(16, start_min=2)
    bins = form_bins(m)
    assert [b.left for b in bins] == [np.datetime64("2010-01-20T00:05", "m"),
                                    np.datetime64("2010-01-20T00:10", "m")]
    # A gap at 00:07 invalidates only the first complete UTC bin.
    gapped = m[m["minute"] != 7]
    assert [b.left for b in form_bins(gapped)] == [bins[1].left]
    m["source"][m["minute"] == 7] = 52
    assert [b.left for b in form_bins(m)] == [bins[1].left]


def test_valid_nonconstant_fixture_preserves_summaries_and_window_union(tmp_path):
    m = make_minutes(50 * 5)
    B = np.random.default_rng(17).normal(size=(len(m), 3))
    for i, k in enumerate(("bx", "by", "bz")):
        m[k] = B[:, i]
    path = tmp_path / "valid.asc"
    write_omni_fixture(path, m)
    parsed = parse_omni_year(path)
    np.testing.assert_array_equal(parsed, m)
    bins = form_bins(parsed)
    assert len(bins) == 50
    for i, b in enumerate(bins):
        raw = B[5 * i : 5 * (i + 1)]
        mu = raw.mean(axis=0)
        mag = np.linalg.norm(raw, axis=1)
        expected = np.r_[mu, mag.mean(), ((mag - mag.mean()) ** 2).mean(),
                         ((raw - mu) ** 2).sum(axis=1).mean()]
        np.testing.assert_array_equal(b.z, expected)
    runs = consecutive_runs(bins)
    assert len(runs) == 1
    wins = windows_from_run(runs[0])
    assert len(wins) == 3
    for i, w in enumerate(wins):
        np.testing.assert_array_equal(w.z, np.stack([b.z for b in bins[i:i + N_BINS]]))
