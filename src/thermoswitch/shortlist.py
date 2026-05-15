"""Prediction ranking, diversity filtering, QC, and order-sheet utilities."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

from thermoswitch.config import (
    GC_MAX,
    GC_MIN,
    GUARD_SLOPE_MIN,
    GUARD_T50_RANGE,
    HOMO_N,
    TOP_K_SAVE,
    USE_GUARDS,
)


PREDICTION_COLUMNS = ["id", "seq", "p_pass", "t50_pred", "slope70_pred"]
SHORTLIST_COLUMNS = ["rank", "id", "seq", "p_pass", "t50_pred", "slope70_pred"]
ORDER_SHEET_COLUMNS = [
    "rank",
    "id",
    "seq",
    "len_nt",
    "gc_frac",
    "p_pass",
    "t50_pred",
    "slope70_pred",
]


def _as_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _require_columns(rows: list[Mapping[str, Any]], required_columns: list[str]) -> None:
    if not rows:
        raise ValueError("No rows provided.")
    missing = [column for column in required_columns if column not in rows[0]]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _as_float(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Column {field_name} must be numeric.") from exc
    if math.isnan(number):
        raise ValueError(f"Column {field_name} must not be NaN.")
    return number


def _coerce_prediction_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["p_pass"] = _as_float(result["p_pass"], "p_pass")
    result["t50_pred"] = _as_float(result["t50_pred"], "t50_pred")
    result["slope70_pred"] = _as_float(result["slope70_pred"], "slope70_pred")
    return result


def rank_key(row: Mapping[str, Any]) -> tuple[float, float, float]:
    """Rank by predicted pass probability, T50 closeness to 70 C, then slope."""
    coerced = _coerce_prediction_row(row)
    return (
        coerced["p_pass"],
        -abs(coerced["t50_pred"] - 70.0),
        coerced["slope70_pred"],
    )


def kmer_set(seq: str, k: int = 3) -> set[str]:
    """Return the set of sequence k-mers used by the notebook diversity logic."""
    return {seq[i : i + k] for i in range(len(seq) - k + 1)} if len(seq) >= k else set()


def jaccard_distance(seq_a: str, seq_b: str, k: int = 3) -> float:
    """Return 1 - Jaccard similarity over k-mer sets."""
    kmers_a = kmer_set(seq_a, k)
    kmers_b = kmer_set(seq_b, k)
    if not kmers_a and not kmers_b:
        return 0.0
    union = kmers_a | kmers_b
    if not union:
        return 0.0
    return 1.0 - (len(kmers_a & kmers_b) / len(union))


def has_homopolymer(seq: str, n: int = HOMO_N) -> bool:
    """Return True when a sequence contains a homopolymer run of length n or more."""
    return re.search(rf"(A{{{n},}}|U{{{n},}}|G{{{n},}}|C{{{n},}})", seq) is not None


def gc_fraction_for_shortlist(seq: str) -> float:
    """Return GC fraction for shortlist QC and order-sheet reporting."""
    return 0.0 if len(seq) == 0 else (seq.count("G") + seq.count("C")) / len(seq)


def passes_prediction_guards(
    row: Mapping[str, Any],
    t50_range: tuple[float, float] = GUARD_T50_RANGE,
    slope_min: float = GUARD_SLOPE_MIN,
) -> bool:
    """Check predicted T50 and slope guards from notebook cells 18 and 21."""
    coerced = _coerce_prediction_row(row)
    t50_low, t50_high = t50_range
    return (
        t50_low <= coerced["t50_pred"] <= t50_high
        and coerced["slope70_pred"] >= slope_min
    )


def passes_qc(
    row: Mapping[str, Any],
    gc_min: float = GC_MIN,
    gc_max: float = GC_MAX,
    homo_n: int = HOMO_N,
) -> bool:
    """Check sequence GC and homopolymer QC rules."""
    seq = str(row["seq"])
    gc = gc_fraction_for_shortlist(seq)
    return gc_min <= gc <= gc_max and not has_homopolymer(seq, homo_n)


def select_top_k(
    prediction_rows: Iterable[Mapping[str, Any]],
    k: int = TOP_K_SAVE,
    use_guards: bool = USE_GUARDS,
) -> list[dict[str, Any]]:
    """Select the top K predictions by notebook ranking logic."""
    rows = _as_rows(prediction_rows)
    _require_columns(rows, PREDICTION_COLUMNS)
    rows = [_coerce_prediction_row(row) for row in rows]
    pool = [row for row in rows if passes_prediction_guards(row)] if use_guards else list(rows)
    if not pool:
        pool = list(rows)
    selected = sorted(pool, key=rank_key, reverse=True)[:k]
    return [_shortlist_row(row, rank) for rank, row in enumerate(selected, 1)]


def _min_distance_to_selected(row: Mapping[str, Any], selected: list[Mapping[str, Any]]) -> float:
    if not selected:
        return 1.0
    return min(jaccard_distance(str(row["seq"]), str(chosen["seq"]), k=3) for chosen in selected)


def diversify_top_k(
    prediction_rows: Iterable[Mapping[str, Any]],
    k: int = TOP_K_SAVE,
    pool_multiplier: int = 3,
) -> list[dict[str, Any]]:
    """Greedily diversify the top probability pool using 3-mer Jaccard distance."""
    rows = _as_rows(prediction_rows)
    _require_columns(rows, PREDICTION_COLUMNS)
    rows = [_coerce_prediction_row(row) for row in rows]
    pool = sorted(rows, key=rank_key, reverse=True)[: k * pool_multiplier]

    selected: list[dict[str, Any]] = []
    while pool and len(selected) < k:
        if not selected:
            selected.append(pool.pop(0))
            continue
        best = max(pool, key=lambda row: _min_distance_to_selected(row, selected))
        pool.remove(best)
        selected.append(best)
    return [_shortlist_row(row, rank) for rank, row in enumerate(selected, 1)]


def refine_shortlist(
    prediction_rows: Iterable[Mapping[str, Any]],
    shortlist_rows: Iterable[Mapping[str, Any]],
    k: int = TOP_K_SAVE,
) -> list[dict[str, Any]]:
    """Apply QC/guard rules to a shortlist and refill with diverse valid candidates."""
    predictions = [_coerce_prediction_row(row) for row in _as_rows(prediction_rows)]
    shortlist = _as_rows(shortlist_rows)
    _require_columns(predictions, PREDICTION_COLUMNS)
    _require_columns(shortlist, ["id", "seq"])

    by_id = {str(row["id"]): row for row in predictions}
    selected: list[dict[str, Any]] = []
    selected_ids = {str(row.get("id")) for row in selected}
    shortlist_ids = [str(row["id"]) for row in shortlist]
    target_k = min(k, len(shortlist_ids)) if shortlist_ids else k

    for row_id in shortlist_ids:
        row = by_id.get(row_id)
        if row and passes_qc(row) and passes_prediction_guards(row):
            selected.append(row)
            selected_ids.add(str(row["id"]))

    pool: list[dict[str, Any]] = []
    for row in sorted(predictions, key=rank_key, reverse=True):
        row_id = str(row["id"])
        if row_id in selected_ids:
            continue
        if not passes_qc(row) or not passes_prediction_guards(row):
            continue
        pool.append(row)
        if len(pool) >= 5 * target_k:
            break

    while len(selected) < target_k and pool:
        best = max(pool, key=lambda row: (_min_distance_to_selected(row, selected),) + rank_key(row))
        pool.remove(best)
        selected.append(best)
        selected_ids.add(str(best["id"]))

    return [_shortlist_row(row, rank) for rank, row in enumerate(selected[:target_k], 1)]


def make_order_sheet(shortlist_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Create practical order-sheet rows from a clean shortlist."""
    rows = _as_rows(shortlist_rows)
    _require_columns(rows, SHORTLIST_COLUMNS)
    order_rows: list[dict[str, Any]] = []
    for row in rows:
        seq = str(row["seq"])
        order_rows.append(
            {
                "rank": row["rank"],
                "id": row["id"],
                "seq": seq,
                "len_nt": len(seq),
                "gc_frac": f"{gc_fraction_for_shortlist(seq):.2f}",
                "p_pass": row["p_pass"],
                "t50_pred": row["t50_pred"],
                "slope70_pred": row["slope70_pred"],
            }
        )
    return order_rows


def _shortlist_row(row: Mapping[str, Any], rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "id": row["id"],
        "seq": row["seq"],
        "p_pass": row["p_pass"],
        "t50_pred": row["t50_pred"],
        "slope70_pred": row["slope70_pred"],
    }
