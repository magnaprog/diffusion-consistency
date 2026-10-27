"""Role assignment and UTC-aligned five-minute bins (plan Section 5)."""

from __future__ import annotations

import datetime as dt

import numpy as np

from prigm.config import (
    BUFFER_MONTHS,
    DEC_EXCLUDED_FIRST_DAY,
    JAN_EXCLUDED_LAST_DAY,
    ROLE_BUFFER,
    ROLE_EXCLUDED,
    ROLE_TEST,
    ROLE_TRAIN,
    ROLE_VAL,
    TEST_MONTHS,
    TRAIN_MONTHS,
    VAL_MONTHS,
)


def role_for_date(year: int, month: int, day: int) -> str:
    if month == 1 and day <= JAN_EXCLUDED_LAST_DAY:
        return ROLE_EXCLUDED
    if month == 12 and day >= DEC_EXCLUDED_FIRST_DAY:
        return ROLE_EXCLUDED
    if month in TRAIN_MONTHS:
        return ROLE_TRAIN
    if month in TEST_MONTHS:
        return ROLE_TEST
    if month in VAL_MONTHS:
        return ROLE_VAL
    if month in BUFFER_MONTHS:
        return ROLE_BUFFER
    return ROLE_EXCLUDED


def doy_to_date(year: int, doy: int) -> dt.date:
    return dt.date(year, 1, 1) + dt.timedelta(days=int(doy) - 1)


def timestamp_from_fields(year: int, doy: int, hour: int, minute: int) -> np.datetime64:
    d = doy_to_date(year, doy)
    return np.datetime64(
        dt.datetime(d.year, d.month, d.day, hour, minute), "m"
    )


def bin_left_edge(ts: np.datetime64) -> np.datetime64:
    """UTC-aligned five-minute left edge of the minute timestamp."""
    epoch = np.datetime64("1970-01-01T00:00", "m")
    minutes = int((ts.astype("datetime64[m]") - epoch) / np.timedelta64(1, "m"))
    aligned = minutes - (minutes % 5)
    return epoch + np.timedelta64(aligned, "m")


def month_quota(role: str) -> int:
    from prigm.config import TEST_QUOTA, VAL_QUOTA

    if role == ROLE_VAL:
        return VAL_QUOTA
    if role == ROLE_TEST:
        return TEST_QUOTA
    raise ValueError(f"quota undefined for role {role}")


def select_quota_indices(n: int, q0: int) -> np.ndarray:
    """Spread-out indices floor((j+0.5)*N/q) for j=0..q-1. Empty if N=0."""
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    q = min(n, q0)
    j = np.arange(q, dtype=np.float64)
    idx = np.floor((j + 0.5) * n / q).astype(np.int64)
    return idx
