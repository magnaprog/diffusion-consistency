"""Check averaging and whole-year resampling for score comparisons on CPU.

Average scores within each represented month, then give those month means
equal weight. Bootstrap resampling draws years with replacement: every
selected occurrence contributes its represented months again. A missing
month contributes no observations, and repeated years are not deduplicated.

Independent calculations and controlled random draws check this convention
for the score difference within each time-period group.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import Mock

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prigm import run_cycle
from prigm.scoring import month_balanced_mean_replicated


@pytest.fixture
def phase_scores():
    months = []
    gaps = []
    for phase, (bounds, _) in enumerate(run_cycle.PHASES):
        a, b, c = list(bounds)[:3]
        months.extend([
            a * 100 + 8, a * 100 + 8, a * 100 + 9,
            b * 100 + 8, b * 100 + 9, b * 100 + 9, b * 100 + 9,
            c * 100 + 8, c * 100 + 8, c * 100 + 9,
        ])
        gaps.extend(np.array([1, 3, 10, 20, 26, 30, 34, 100, 104, 200]) + 50 * phase)
    months = np.asarray(months, dtype=np.int64)
    base = np.arange(len(months), dtype=np.float64) + 10.0
    return base + np.asarray(gaps), base, months


def manual_replicated_mean(values, months, years_draw):
    """Average each represented month separately for every drawn occurrence."""
    month_means = []
    for year in years_draw:
        for month in sorted(set(months[months // 100 == year])):
            within_month = values[months == month].tolist()
            month_means.append(sum(within_month) / len(within_month))
    return sum(month_means) / len(month_means)


@pytest.mark.parametrize("missing_september", [False, True])
def test_existing_replicated_helper_keeps_months_and_year_multiplicity(missing_september):
    months = np.array([200508, 200508, 200509, 200608, 200609, 200609, 200609, 200708])
    values = np.array([1.0, 3.0, 10.0, 20.0, 26.0, 30.0, 34.0, 9999.0])
    if missing_september:
        mask = months != 200609
        values, months = values[mask], months[mask]
    draw = np.array([2005, 2005, 2006])
    # A's August and September appear twice; B's represented months appear once.
    month_means = [(1 + 3) / 2, 10, (1 + 3) / 2, 10, 20]
    if not missing_september:
        month_means.append((26 + 30 + 34) / 3)
    expected = sum(month_means) / len(month_means)
    assert manual_replicated_mean(values, months, draw) == pytest.approx(expected)
    assert month_balanced_mean_replicated(
        values, months, lambda m: int(m) // 100, draw,
    ) == pytest.approx(expected)


@pytest.mark.parametrize("phase", range(len(run_cycle.PHASES)))
@pytest.mark.parametrize("restricted", [False, True], ids=["full-phase", "masked-windows"])
def test_gap_ci_repeated_year_draw_matches_manual(monkeypatch, phase_scores, phase, restricted):
    neural, base, months = phase_scores
    bounds, _ = run_cycle.PHASES[phase]
    a, b, c = list(bounds)[:3]
    mask = np.isin(months // 100, list(bounds))
    if restricted:
        # Also exclude windows inside sampled years, as a field-strength mask can.
        mask[np.flatnonzero(months == a * 100 + 8)[0]] = False
        mask[months == b * 100 + 9] = False
    rng = Mock()
    rng.choice.return_value = np.array([0, 0, 1])  # [A, A, B], never C.
    monkeypatch.setattr(run_cycle.np.random, "Generator", lambda bitgen: rng)
    expected = manual_replicated_mean(
        (neural - base)[mask], months[mask], [a, a, b],
    )
    assert expected == pytest.approx((46 / 5 if restricted else 37 / 3) + 50 * phase)

    result = run_cycle.gap_ci(neural, base, mask, months, reps=5)

    assert set(result) == {"ci95"}
    assert isinstance(result["ci95"], list)
    assert all(isinstance(v, float) for v in result["ci95"])
    assert result["ci95"] == pytest.approx([expected, expected])
    assert rng.choice.call_count == 5
    rng.choice.assert_called_with(3, size=3, replace=True)


@pytest.mark.parametrize("phase", range(len(run_cycle.PHASES)))
def test_gap_ci_seeded_phase_and_point_estimate_match_manual(phase_scores, phase):
    neural, base, months = phase_scores
    bounds, _ = run_cycle.PHASES[phase]
    mask = np.isin(months // 100, list(bounds))
    years = np.unique(months[mask] // 100)
    values = (neural - base)[mask]
    expected_point = manual_replicated_mean(values, months[mask], years)
    point = (
        run_cycle.summarize(neural, months, mask)["month_balanced"]
        - run_cycle.summarize(base, months, mask)["month_balanced"]
    )
    assert point == pytest.approx(expected_point)
    assert point == pytest.approx(182 / 3 + 50 * phase)

    reps, seed = 31, 7010
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = []
    for _ in range(reps):
        chosen = rng.choice(len(years), size=len(years), replace=True)
        samples.append(manual_replicated_mean(values, months[mask], years[chosen]))
    expected_ci = np.quantile(samples, [0.025, 0.975])
    result = run_cycle.gap_ci(neural, base, mask, months, reps=reps, seed=seed)
    assert result["ci95"] == pytest.approx(expected_ci)
    assert run_cycle.gap_ci(neural, base, mask, months, reps=reps, seed=seed) == result
