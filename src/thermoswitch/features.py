"""Feature extraction copied from the canonical notebook cell 18."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd


ALPH = "AUGC"
NUC_IDX = {base: i for i, base in enumerate(ALPH)}
DINUCS = [a + b for a in ALPH for b in ALPH]
TRIN = [a + b + c for a in ALPH for b in ALPH for c in ALPH]
DEFAULT_RBS_LEN = 6

LEAKAGE_COLUMNS = {
    "closed_low",
    "open_midhi",
    "open_high",
    "T50",
    "slope70",
    "slope_max",
    "reentrant_pen",
    "score",
    "curve",
    "p_pass",
    "t50_pred",
    "slope70_pred",
    "rank",
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _clean_seq(seq: Any) -> str:
    if _is_missing(seq):
        return ""
    return str(seq)


def _row_get(row: Mapping[str, Any] | pd.Series, key: str, default: Any = None) -> Any:
    if isinstance(row, pd.Series):
        return row.get(key, default)
    return row.get(key, default)


def _float_or_default(value: Any, default: float = 0.0) -> float:
    if _is_missing(value):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    return number


def _int_or_none(value: Any) -> int | None:
    if _is_missing(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def sequence_length(seq: Any) -> int:
    """Return sequence length, treating missing values as an empty sequence."""
    return len(_clean_seq(seq))


def gc_fraction(seq: Any) -> float:
    """Return the G/C fraction for an RNA sequence."""
    cleaned = _clean_seq(seq)
    length = len(cleaned)
    if length == 0:
        return 0.0
    return (cleaned.count("G") + cleaned.count("C")) / length


def dinuc_freqs(seq: Any) -> np.ndarray:
    """Return canonical dinucleotide frequency features in notebook order."""
    cleaned = _clean_seq(seq)
    if len(cleaned) < 2:
        return np.zeros(len(DINUCS), float)
    counts = Counter(cleaned[i : i + 2] for i in range(len(cleaned) - 1))
    total = len(cleaned) - 1
    return np.array([counts.get(motif, 0) / total for motif in DINUCS], float)


def trinuc_freqs(seq: Any) -> np.ndarray:
    """Return canonical trinucleotide frequency features in notebook order."""
    cleaned = _clean_seq(seq)
    if len(cleaned) < 3:
        return np.zeros(len(TRIN), float)
    counts = Counter(cleaned[i : i + 3] for i in range(len(cleaned) - 2))
    total = len(cleaned) - 2
    return np.array([counts.get(motif, 0) / total for motif in TRIN], float)


def rbs_window_onehot(
    seq: Any,
    rbs_start: Any,
    rbs_end: Any,
    left: int = 10,
    right: int = 20,
    rbs_len_default: int = DEFAULT_RBS_LEN,
) -> np.ndarray:
    """Return fixed-length RBS-window one-hot features."""
    cleaned = _clean_seq(seq)
    start_coord = _int_or_none(rbs_start)
    end_coord = _int_or_none(rbs_end)
    window_len = left + rbs_len_default + right

    if start_coord is None or end_coord is None or end_coord <= start_coord:
        return np.zeros(window_len * len(ALPH), float)

    start = start_coord - left
    vec = np.zeros((window_len, len(ALPH)), float)
    for i in range(window_len):
        j = start + i
        if 0 <= j < len(cleaned):
            k = NUC_IDX.get(cleaned[j])
            if k is not None:
                vec[i, k] = 1.0
    return vec.reshape(-1)


def make_features(row: Mapping[str, Any] | pd.Series) -> np.ndarray:
    """Build the model-ready feature vector from non-target candidate fields."""
    seq = _clean_seq(_row_get(row, "seq", ""))
    base = np.array(
        [
            sequence_length(seq),
            gc_fraction(seq),
            _float_or_default(_row_get(row, "extra_pairs", 0.0)),
            _float_or_default(_row_get(row, "pGC", 0.0)),
            _float_or_default(_row_get(row, "loop_len", 0.0)),
        ],
        float,
    )
    return np.concatenate(
        [
            base,
            dinuc_freqs(seq),
            trinuc_freqs(seq),
            rbs_window_onehot(seq, _row_get(row, "rbs_start"), _row_get(row, "rbs_end")),
        ]
    )


def _iter_rows(rows_or_df: Iterable[Mapping[str, Any]] | pd.DataFrame) -> list[Mapping[str, Any]]:
    if isinstance(rows_or_df, pd.DataFrame):
        return rows_or_df.to_dict(orient="records")
    return list(rows_or_df)


def build_feature_matrix(rows_or_df: Iterable[Mapping[str, Any]] | pd.DataFrame) -> np.ndarray:
    """Build a numeric feature matrix without target or prediction columns."""
    rows = _iter_rows(rows_or_df)
    if not rows:
        return np.empty((0, len(get_feature_names())), dtype=float)
    return np.vstack([make_features(row) for row in rows]).astype(float)


def get_feature_names() -> list[str]:
    """Return stable feature names aligned to make_features output."""
    base = ["seq_len", "gc_from_seq", "extra_pairs", "pGC", "loop_len"]
    dinuc = [f"dinuc_{motif}" for motif in DINUCS]
    trinuc = [f"trinuc_{motif}" for motif in TRIN]
    rbs = [
        f"rbs_window_pos_{position:02d}_{base_name}"
        for position in range(10 + DEFAULT_RBS_LEN + 20)
        for base_name in ALPH
    ]
    names = base + dinuc + trinuc + rbs
    leaked = LEAKAGE_COLUMNS.intersection(names)
    if leaked:
        raise RuntimeError(f"Feature schema contains leakage columns: {sorted(leaked)}")
    return names
