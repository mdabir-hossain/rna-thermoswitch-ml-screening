"""Data loading and column validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from thermoswitch.config import LIBRARY_COMBINED_PATH


METADATA_COLUMNS = ["id", "seq", "len", "rbs_start", "rbs_end"]
DESIGN_FEATURE_COLUMNS = ["extra_pairs", "pGC", "loop_len"]
TARGET_COLUMNS = ["closed_low", "open_midhi", "open_high", "T50", "slope70"]
SIMULATION_ONLY_COLUMNS = ["slope_max", "reentrant_pen", "score", "curve"]
PREDICTION_OUTPUT_COLUMNS = ["p_pass", "t50_pred", "slope70_pred", "rank", "gc_frac", "len_nt"]

LIBRARY_REQUIRED_COLUMNS = [
    "id",
    "seq",
    "len",
    "rbs_start",
    "rbs_end",
    "extra_pairs",
    "pGC",
    "loop_len",
    "closed_low",
    "open_midhi",
    "open_high",
    "slope70",
    "T50",
    "slope_max",
    "reentrant_pen",
    "score",
    "curve",
]

PREDICTIONS_REQUIRED_COLUMNS = [
    "id",
    "seq",
    "len",
    "p_pass",
    "t50_pred",
    "slope70_pred",
]

SHORTLIST_REQUIRED_COLUMNS = [
    "rank",
    "id",
    "seq",
    "p_pass",
    "t50_pred",
    "slope70_pred",
]


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Raise a clear error when a DataFrame is missing required columns."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def get_library_required_columns() -> list[str]:
    """Return the canonical required columns for the processed library."""
    return list(LIBRARY_REQUIRED_COLUMNS)


def split_library_columns() -> dict[str, list[str]]:
    """Describe canonical column roles without using targets as inference features."""
    inference_features = METADATA_COLUMNS + DESIGN_FEATURE_COLUMNS
    return {
        "metadata": list(METADATA_COLUMNS),
        "inference_features": list(inference_features),
        "targets": list(TARGET_COLUMNS),
        "simulation_only": list(SIMULATION_ONLY_COLUMNS),
        "prediction_outputs": list(PREDICTION_OUTPUT_COLUMNS),
    }


def load_library(path: str | Path = LIBRARY_COMBINED_PATH) -> pd.DataFrame:
    """Load the canonical processed library and validate target-bearing columns."""
    df = pd.read_csv(path)
    validate_required_columns(df, LIBRARY_REQUIRED_COLUMNS)
    return df


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Load prediction outputs for reporting or shortlist generation only."""
    df = pd.read_csv(path)
    validate_required_columns(df, PREDICTIONS_REQUIRED_COLUMNS)
    return df


def load_shortlist(path: str | Path) -> pd.DataFrame:
    """Load a Top-K shortlist output file."""
    df = pd.read_csv(path)
    validate_required_columns(df, SHORTLIST_REQUIRED_COLUMNS)
    return df
