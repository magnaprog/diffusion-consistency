"""Select the lowest validation energy score; break exact ties by smaller c_R
or earlier milestone."""

from __future__ import annotations

from pathlib import Path


def better_score(a: float, b: float) -> bool:
    return a < b


def select_coefficient(scores: dict[float, float]) -> float:
    """Select the positive c_r with lowest primary VAL score; prefer smaller c_r on exact ties."""
    pos = {c: s for c, s in scores.items() if c > 0}
    if not pos:
        raise ValueError("no positive coefficients")
    best_s = min(pos.values())
    cands = [c for c, s in pos.items() if s == best_s]
    return min(cands)


def select_milestone(rows: list[dict]) -> dict:
    """rows have epoch, score. Exact score tie -> earlier epoch."""
    best = None
    for r in rows:
        if best is None:
            best = r
            continue
        if r["score"] < best["score"] or (
            r["score"] == best["score"] and r["epoch"] < best["epoch"]
        ):
            best = r
    return best
