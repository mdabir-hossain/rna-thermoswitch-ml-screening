"""Simulation-derived target and label utilities."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from thermoswitch.config import SPEC


TARGET_COLUMNS = ["closed_low", "open_midhi", "open_high", "T50", "slope70"]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _row_get(row: Mapping[str, Any] | pd.Series, key: str) -> Any:
    if isinstance(row, pd.Series):
        return row[key]
    return row[key]


def _as_float(value: Any) -> float:
    if _is_missing(value):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _iter_rows(rows_or_df: Iterable[Mapping[str, Any]] | pd.DataFrame) -> list[Mapping[str, Any]]:
    if isinstance(rows_or_df, pd.DataFrame):
        return rows_or_df.to_dict(orient="records")
    return list(rows_or_df)


def validate_target_columns(rows_or_df: Iterable[Mapping[str, Any]] | pd.DataFrame) -> None:
    """Validate that simulation-derived target columns are present."""
    if isinstance(rows_or_df, pd.DataFrame):
        columns = set(rows_or_df.columns)
    else:
        rows = _iter_rows(rows_or_df)
        columns = set(rows[0].keys()) if rows else set()
    missing = [column for column in TARGET_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"Missing required target columns: {', '.join(missing)}")


def meets_spec(row: Mapping[str, Any] | pd.Series, spec: Mapping[str, float] = SPEC) -> bool:
    """Return True when a row meets the simulation-derived design spec."""
    closed_low = _as_float(_row_get(row, "closed_low"))
    open_midhi = _as_float(_row_get(row, "open_midhi"))
    open_high = _as_float(_row_get(row, "open_high"))
    t50 = _as_float(_row_get(row, "T50"))
    slope70 = _as_float(_row_get(row, "slope70"))

    values = [closed_low, open_midhi, open_high, t50, slope70]
    if any(math.isnan(value) for value in values):
        return False

    return (
        closed_low >= spec["min_closed_low"]
        and open_midhi >= spec["min_open_midhi"]
        and open_high >= spec["min_open_high"]
        and spec["t50_low"] <= t50 <= spec["t50_high"]
        and slope70 >= spec["min_slope70"]
    )


def create_pass_labels(rows_or_df: Iterable[Mapping[str, Any]] | pd.DataFrame) -> np.ndarray:
    """Create binary labels from simulation-derived target columns."""
    validate_target_columns(rows_or_df)
    rows = _iter_rows(rows_or_df)
    return np.array([int(meets_spec(row)) for row in rows], dtype=int)


def extract_regression_targets(
    rows_or_df: Iterable[Mapping[str, Any]] | pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return T50 and slope70 regression targets from simulation outputs."""
    validate_target_columns(rows_or_df)
    rows = _iter_rows(rows_or_df)
    t50 = np.array([_as_float(row["T50"]) for row in rows], dtype=float)
    slope70 = np.array([_as_float(row["slope70"]) for row in rows], dtype=float)
    return t50, slope70
