"""Build five-minute bins and four-hour windows from OMNI minute records.

A bin requires five consecutive valid vectors from one instrument source,
aligned to a UTC five-minute start. Each bin stores the six quantities
defined in prigm.observation. A window contains 48 consecutive bins from
one source, calendar month, year, and assigned data role. Roles identify
training, validation, testing, buffer periods, or excluded dates.

Window construction and monthly evaluation selection follow plan Section 5.
Missing vectors are not filled to make an otherwise ineligible bin usable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from prigm.calendar import (
    bin_left_edge,
    month_quota,
    role_for_date,
    select_quota_indices,
    timestamp_from_fields,
)
from prigm.config import (
    COL_BX,
    COL_BY,
    COL_BZ,
    COL_DOY,
    COL_HOUR,
    COL_IMF_SOURCE,
    COL_MINUTE,
    COL_YEAR,
    FILL_COMPONENT,
    FILL_SOURCE,
    HIDDEN_END,
    HIDDEN_START,
    N_BINS,
    N_MINUTES_PER_BIN,
    OMNI_DIR,
    OMNI_NAME,
)
from prigm.observation import summaries_from_vectors


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def omni_path(year: int, root: Path | str = OMNI_DIR) -> Path:
    return Path(root) / OMNI_NAME.format(year=year)


_MINUTE_DTYPE = np.dtype(
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


# HRO modified format, footnote D; original audit: PISCES-dev/analysis/omni_source_audit.py.
# The 2005-2024 audit observed Wind (52) and ACE (71); 50 and 60 are documented.
VALID_IMF_SOURCES = (50, 52, 60, 71)


def _validated_timestamps(fields: np.ndarray) -> np.ndarray:
    """Validate year/DOY/hour/minute/source before any integer conversion."""
    for i, name in enumerate(("year", "doy", "hour", "minute", "source")):
        v = fields[:, i]
        if np.any(~np.isfinite(v) | (v != np.floor(v))):
            raise ValueError(f"{name} must be finite and integral")
    year, doy, hour, minute, source = fields.T
    if np.any((year < 1) | (year > 9999)):
        raise ValueError("year must be in 1..9999")
    leap = (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))
    if np.any((doy < 1) | (doy > 365 + leap)):
        raise ValueError("doy outside the calendar year")
    if np.any((hour < 0) | (hour > 23)):
        raise ValueError("hour must be in 0..23")
    if np.any((minute < 0) | (minute > 59)):
        raise ValueError("minute must be in 0..59")
    if np.any(~np.isin(source, (*VALID_IMF_SOURCES, FILL_SOURCE))):
        raise ValueError("source must be a documented IMF identifier or source fill 99")
    year, doy, hour, minute = fields[:, :4].astype(np.int32).T
    jan1 = (np.datetime64("1970", "Y") + (year - 1970).astype("timedelta64[Y]")).astype(
        "datetime64[D]"
    )
    days = jan1 + (doy - 1).astype("timedelta64[D]")
    ts = days.astype("datetime64[m]") + hour.astype("timedelta64[h]") + minute.astype("timedelta64[m]")
    if np.any(ts[1:] <= ts[:-1]):
        raise ValueError("minute timestamps must be unique and strictly increasing")
    return ts


def _minutes_from_raw(raw: np.ndarray) -> np.ndarray:
    """Shared magnetic-column parser for the data and metadata readers."""
    ts = _validated_timestamps(raw[:, :5])
    out = np.empty(len(raw), dtype=_MINUTE_DTYPE)
    for i, name in enumerate(_MINUTE_DTYPE.names[:-1]):
        out[name] = raw[:, i]
    out["ts"] = ts
    return out


def parse_omni_year(path: Path) -> np.ndarray:
    """Read ordered, unique minute records. Reject malformed metadata; never fill."""
    raw = np.loadtxt(
        path,
        usecols=(COL_YEAR, COL_DOY, COL_HOUR, COL_MINUTE, COL_IMF_SOURCE, COL_BX, COL_BY, COL_BZ),
        dtype=np.float64,
        ndmin=2,
    )
    return _minutes_from_raw(raw)


def validate_minutes(minutes: np.ndarray) -> None:
    """Also validate callers that supply structured records without the ASCII parser."""
    fields = np.column_stack([minutes[k] for k in ("year", "doy", "hour", "minute", "source")])
    ts = _validated_timestamps(fields)
    if np.any(np.isnat(minutes["ts"])) or np.any(minutes["ts"] != ts):
        raise ValueError("timestamp does not match year/doy/hour/minute fields")


def magnetic_valid(row) -> np.ndarray:
    """Exact documented component fill and nonfinite checks, not an amplitude cut."""
    valid = np.isin(row["source"], VALID_IMF_SOURCES)
    for k in ("bx", "by", "bz"):
        valid = valid & np.isfinite(row[k]) & (row[k] != FILL_COMPONENT)
    return valid


def vector_valid(row: np.void) -> bool:
    return bool(magnetic_valid(row))


@dataclass(frozen=True)
class Bin:
    left: np.datetime64
    source: int
    z: np.ndarray  # (6,)
    role: str
    year: int
    month: int


def form_bins(minutes: np.ndarray) -> list[Bin]:
    """Fixed UTC-aligned 5-minute bins: five unique consecutive valid vectors, one source."""
    if minutes.size == 0:
        return []
    validate_minutes(minutes)
    m = minutes
    mins = m["ts"].astype("datetime64[m]").astype(np.int64)
    valid = magnetic_valid(m)
    n = m.size
    if n < N_MINUTES_PER_BIN:
        return []
    starts = np.flatnonzero(mins % N_MINUTES_PER_BIN == 0)
    starts = starts[starts + N_MINUTES_PER_BIN - 1 < n]
    if starts.size == 0:
        return []
    idx = starts[:, None] + np.arange(N_MINUTES_PER_BIN)[None, :]
    ts_ok = np.all(np.diff(mins[idx], axis=1) == 1, axis=1)
    val_ok = np.all(valid[idx], axis=1)
    src = m["source"][idx]
    src_ok = np.all(src == src[:, :1], axis=1)
    keep = ts_ok & val_ok & src_ok
    starts = starts[keep]
    idx = idx[keep]
    if starts.size == 0:
        return []
    B = np.stack((m["bx"][idx], m["by"][idx], m["bz"][idx]), axis=-1)
    z_all = summaries_from_vectors(B)
    lefts = m["ts"][starts]
    days = lefts.astype("datetime64[D]").astype("O")
    bins: list[Bin] = []
    for k, s in enumerate(starts):
        py = days[k]
        bins.append(
            Bin(
                left=lefts[k],
                source=int(m["source"][s]),
                z=z_all[k],
                role=role_for_date(int(py.year), int(py.month), int(py.day)),
                year=int(py.year),
                month=int(py.month),
            )
        )
    return bins


def consecutive_runs(bins: list[Bin]) -> list[list[Bin]]:
    """Maximal same-source, same-role, 5-minute-stride runs. No month/role/source cross."""
    if not bins:
        return []
    runs: list[list[Bin]] = []
    cur = [bins[0]]
    for b in bins[1:]:
        prev = cur[-1]
        ok = (
            b.role == prev.role
            and b.source == prev.source
            and b.month == prev.month
            and b.year == prev.year
            and (b.left - prev.left) / np.timedelta64(5, "m") == 1
        )
        if ok:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    return runs


@dataclass(frozen=True)
class Window:
    start_left: np.datetime64
    source: int
    role: str
    year: int
    month: int
    z: np.ndarray  # (48, 6)


def windows_from_run(run: list[Bin], stride: int = 1) -> list[Window]:
    out: list[Window] = []
    if len(run) < N_BINS:
        return out
    for s in range(0, len(run) - N_BINS + 1, stride):
        block = run[s : s + N_BINS]
        if (block[-1].left - block[0].left) / np.timedelta64(5, "m") != N_BINS - 1:
            continue
        z = np.stack([b.z for b in block], axis=0)
        out.append(
            Window(
                start_left=block[0].left,
                source=block[0].source,
                role=block[0].role,
                year=block[0].year,
                month=block[0].month,
                z=z,
            )
        )
    return out


def pack_nonoverlapping(run: list[Bin]) -> list[Window]:
    return windows_from_run(run, stride=N_BINS)


def select_eval_windows(windows: list[Window], role: str) -> list[Window]:
    """Greedy packed pool already assumed; apply monthly quota."""
    q0 = month_quota(role)
    by_key: dict[tuple[int, int], list[Window]] = {}
    for w in windows:
        by_key.setdefault((w.year, w.month), []).append(w)
    selected: list[Window] = []
    for key in sorted(by_key):
        pool = sorted(by_key[key], key=lambda w: w.start_left)
        idx = select_quota_indices(len(pool), q0)
        for j in idx:
            selected.append(pool[int(j)])
    return selected


def hidden_slice(z_window: np.ndarray) -> np.ndarray:
    return z_window[HIDDEN_START:HIDDEN_END]


def context_mask() -> np.ndarray:
    m = np.ones((N_BINS, 1), dtype=np.float64)
    m[HIDDEN_START:HIDDEN_END] = 0.0
    return m


def relative_positions() -> np.ndarray:
    t = np.arange(N_BINS, dtype=np.float64)
    center = 0.5 * (N_BINS - 1)
    return ((t - center) / center)[:, None]


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
