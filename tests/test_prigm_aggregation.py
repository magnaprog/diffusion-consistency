"""Independent raw-vector Q15 checks and metadata denominator regressions."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from tests.test_prigm_data import make_minutes, write_omni_fixture
from prigm import q15, quality
from prigm.config import N_BINS, YEARS
from prigm.data import consecutive_runs, file_sha256, form_bins, windows_from_run


def varying_minutes(n, **kwargs):
    m = make_minutes(n, **kwargs)
    m["bx"] = np.arange(n) % 5
    return m


@pytest.mark.parametrize("n_bins,start_min,n_blocks", [(47, 0, 0), (48, 0, 16),
                                                       (48, 5, 15), (50, 5, 16)])
def test_q15_utc_alignment_and_four_hour_union(n_bins, start_min, n_blocks):
    m = varying_minutes(n_bins * 5, start_min=start_min)
    bins = form_bins(m)
    blocks, counts = q15.eligible_q15_blocks(bins)
    assert len(blocks) == n_blocks
    eligible = set()
    starts = 0
    for run in consecutive_runs(bins):
        wins = windows_from_run(run)
        starts += len(wins)
        for w in wins:
            eligible.update(w.start_left + np.arange(N_BINS) * np.timedelta64(5, "m"))
    assert counts["eligible_train_bins"] == len(eligible)
    assert counts["eligible_train_starts"] == starts
    expected = {b.left for b in bins if int(b.left.astype(np.int64)) % 15 == 0
                and all(b.left + np.timedelta64(i * 5, "m") in eligible for i in range(3))}
    assert {b[0].left for b in blocks} == expected
    for block in blocks:
        assert all(b.left in eligible for b in block)
    values = q15.compare_q15_raw(m, blocks)
    np.testing.assert_array_equal(values, np.tile([8.0, 8.0, 6.0], (n_blocks, 1)))


@pytest.mark.parametrize("boundary", ["source", "gap", "month", "guard", "role"])
def test_q15_never_promotes_short_runs_across_boundaries(boundary):
    if boundary == "month":
        m = np.concatenate((make_minutes(120, doy=31, start_hour=22), make_minutes(120, doy=32)))
    elif boundary == "guard":
        m = np.concatenate((make_minutes(120, doy=14, start_hour=22), make_minutes(120, doy=15)))
    elif boundary == "role":
        m = np.concatenate((make_minutes(120, doy=181, start_hour=22), make_minutes(120, doy=182)))
    else:
        m = make_minutes(240)
        if boundary == "source":
            m["source"][120:] = 52
        else:
            m = np.delete(m, 120)
    blocks, counts = q15.eligible_q15_blocks(form_bins(m))
    assert not blocks
    assert counts["eligible_train_bins"] == 0
    assert counts["eligible_train_starts"] == 0


def test_q15_excludes_heldout_and_short_training_runs():
    m = np.concatenate((make_minutes(240), make_minutes(15, doy=21),
                        make_minutes(240, doy=213), make_minutes(240, doy=305)))
    blocks, counts = q15.eligible_q15_blocks(form_bins(m))
    assert counts == {"train_bins": 51, "train_runs": 2, "eligible_train_runs": 1,
                      "eligible_train_bins": 48, "eligible_train_starts": 1, "eligible_blocks": 16}
    assert all(str(block[0].left).startswith("2010-01-20") for block in blocks)


def test_q15_corrupted_summary_is_detected_by_independent_raw_moment():
    m = varying_minutes(240)
    blocks, _ = q15.eligible_q15_blocks(form_bins(m))
    z = blocks[0][0].z.copy()
    z[5] += 3.0
    blocks[0][0] = replace(blocks[0][0], z=z)
    # The old check still passes this corruption: both sides use the bad v_v.
    z3 = np.stack([b.z for b in blocks[0]])
    _, old_loss = q15.q15_from_three(z3)
    assert old_loss == np.mean(z3[:, 5])
    values = q15.compare_q15_raw(m, blocks)
    np.testing.assert_array_equal(values[0], [8.0, 9.0, 6.0])
    assert np.max(np.abs(values[:, 0] - values[:, 1])) == 1.0
    np.testing.assert_array_equal(values[1:], np.tile([8.0, 8.0, 6.0], (15, 1)))


@pytest.mark.parametrize("damage", ["duplicate", "missing", "source", "fill", "nonfinite"])
def test_q15_rechecks_original_minute_join(damage):
    m = varying_minutes(240)
    blocks, _ = q15.eligible_q15_blocks(form_bins(m))
    if damage == "duplicate":
        m = np.concatenate((m[:1], m))
    elif damage == "missing":
        m = np.delete(m, 2)
    elif damage == "source":
        m["source"][2] = 52
    elif damage == "fill":
        m["bx"][2] = 9999.99
    else:
        m["bz"][2] = np.nan
    with pytest.raises(ValueError):
        q15.compare_q15_raw(m, blocks)


@pytest.mark.parametrize("max_blocks,n_selected", [(None, 32), (4, 4), (100, 32)])
def test_q15_report_all_year_training_selection_hashes_and_fresh_output(tmp_path, monkeypatch,
                                                                      max_blocks, n_selected):
    monkeypatch.setattr(q15, "YEARS", (2005, 2024))
    original_hashes = {}
    for year, source in ((2005, 52), (2024, 71)):
        m = varying_minutes(240, year=year, source=source)
        heldout = make_minutes(240, year=year, source=source, doy=220)
        heldout["bx"] = 1e6  # Valid, but must not enter the training-only result.
        path = tmp_path / f"omni_min{year}.asc"
        write_omni_fixture(path, np.concatenate((m, heldout)))
        original_hashes[str(year)] = file_sha256(path)
    out = tmp_path / "q15.json"
    argv = ["--omni-root", str(tmp_path), "--output", str(out)]
    if max_blocks is not None:
        argv += ["--max-blocks", str(max_blocks)]
    assert q15.main(argv) == 0
    rec = json.loads(out.read_text())
    assert rec["population_n_blocks"] == 32
    assert rec["n_blocks"] == n_selected
    assert rec["max_abs_err_vs_raw"] == 0.0
    assert rec["mean_full_moment_loss_nT2"] == 0.0
    assert rec["mean_only_loss_mean_nT2"] == 2.0
    assert rec["mean_only_loss_median_nT2"] == 2.0
    assert rec["population_source_block_counts"] == {"52": 16, "71": 16}
    for year in ("2005", "2024"):
        assert rec["population_by_year"][year]["eligible_train_bins"] == 48
        assert rec["population_by_year"][year]["eligible_train_starts"] == 1
        assert rec["selection"]["by_year"][year]["n_blocks"] == n_selected // 2
        assert rec["raw_inputs"][year]["sha256"] == original_hashes[year]
        assert rec["raw_inputs"][year]["n_minutes"] == 480
        assert file_sha256(tmp_path / f"omni_min{year}.asc") == original_hashes[year]
    if n_selected == 32:
        assert rec["population_blocks_sha256"] == rec["selection"]["blocks_sha256"]
    else:
        assert rec["population_blocks_sha256"] != rec["selection"]["blocks_sha256"]
        assert rec["selection"]["first_left"] == "2005-01-20T01:00"
        assert rec["selection"]["last_left"] == "2024-01-20T03:00"
    # The selection and complete record are deterministic on unchanged inputs.
    second = tmp_path / "q15-second.json"
    argv2 = list(argv)
    argv2[argv2.index(str(out))] = str(second)
    assert q15.main(argv2) == 0
    assert second.read_bytes() == out.read_bytes()
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        q15.main(argv)
    assert out.read_bytes() == before


def test_q15_report_exposes_corrupted_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(q15, "YEARS", (2010,))
    write_omni_fixture(tmp_path / "omni_min2010.asc", varying_minutes(240))

    def corrupt(minutes):
        bins = form_bins(minutes)
        z = bins[0].z.copy()
        z[5] += 3.0
        bins[0] = replace(bins[0], z=z)
        return bins

    monkeypatch.setattr(q15, "form_bins", corrupt)
    out = tmp_path / "corruption.json"
    assert q15.main(["--omni-root", str(tmp_path), "--output", str(out)]) == 0
    rec = json.loads(out.read_text())
    assert rec["n_blocks"] == 16
    assert rec["max_abs_err_vs_raw"] == 1.0
    assert rec["mean_abs_full_moment_loss_nT2"] == 1.0 / 16
    assert rec["mean_only_loss_mean_nT2"] == 2.0


@pytest.mark.parametrize("cap", [0, -1])
def test_q15_rejects_nonpositive_cap_before_scanning(tmp_path, cap):
    out = tmp_path / "never-created.json"
    with pytest.raises(SystemExit):
        q15.main(["--omni-root", str(tmp_path), "--output", str(out), "--max-blocks", str(cap)])
    assert not out.exists()


def test_quality_metadata_missingness_uses_exact_sentinels():
    values = np.array([0, -1, 999999, 9999999, 1e6, -1e6, np.nan, np.inf])
    np.testing.assert_array_equal(quality.dbot_missing(values),
                                  [False, False, True, True, False, False, True, True])
    np.testing.assert_array_equal(quality.pfn_missing(np.array([0, 100, 999, np.nan, np.inf])),
                                  [False, False, True, True, True])


def test_candidate_start_denominator_is_calendar_grid_not_valid_records():
    assert quality.candidate_train_grid_starts(2010) == 47814
    assert quality.candidate_train_grid_starts(2012) == 48102
    assert sum(quality.candidate_train_grid_starts(year) for year in YEARS) == 957720


def test_quality_report_pfn_fill_and_explicit_populations(tmp_path, monkeypatch):
    monkeypatch.setattr(quality, "YEARS", (2010,))
    m = np.concatenate((make_minutes(240), make_minutes(5, doy=21)))
    path = tmp_path / "omni_min2010.asc"
    raw = write_omni_fixture(path, m)
    raw[:4, 8] = [999, np.nan, np.inf, 100]
    raw[240:, 8] = 100
    raw[:4, 12] = [-1, 999999, 9999999, np.nan]
    raw[240, 12] = -1
    raw[:3, 6] = [1, 0, 999]
    np.savetxt(path, raw, fmt="%.17g")
    digest = file_sha256(path)
    out = tmp_path / "quality.json"
    argv = ["--omni-root", str(tmp_path), "--output", str(out)]
    assert quality.main(argv) == 0
    rec = json.loads(out.read_text())
    assert rec["train_role_minutes"] == rec["train_valid_magnetic_minutes"] == 245
    assert rec["eligible_bins_scanned"] == 48
    assert rec["distinct_potential_target_bins"] == 12
    assert rec["dbot_known_minutes"] == 242
    assert rec["dbot_negative_minutes"] == 2
    assert rec["dbot_negative_minute_frac"] == 2 / 242
    assert rec["bins_with_any_negative_dbot"] == 1
    assert rec["bins_with_any_negative_dbot_frac"] == 1 / 48
    assert rec["pfn_known_minutes"] == 242
    assert rec["pfn_missing_minutes"] == 3
    assert rec["pfn_positive_minutes"] == 6
    assert rec["pfn_positive_frac_of_valid"] == 6 / 245
    assert rec["pfn_positive_frac_of_known"] == 6 / 242
    assert rec["one_point_minutes"] == 2  # Preserve the existing <=1 counter, now documented.
    assert rec["vector_magnitude_sample_count"] == 48
    assert rec["increment_sample_count"] == 192
    assert rec["candidate_train_grid_starts"] == 47814
    assert rec["eligible_train_starts"] == 1
    assert rec["eligible_start_frac_of_candidate_grid"] == 1 / 47814
    assert "not only the eligible-window union" in rec["population_definitions"]["minute_metadata"]
    assert "excludes between-bin increments" in rec["population_definitions"]["increment"]
    assert rec["raw_inputs"]["2010"]["sha256"] == digest
    assert file_sha256(path) == digest
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        quality.main(argv)
    assert out.read_bytes() == before
