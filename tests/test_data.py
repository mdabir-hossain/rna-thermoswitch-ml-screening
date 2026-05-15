from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoswitch.data import (  # noqa: E402
    TARGET_COLUMNS,
    get_library_required_columns,
    load_library,
    split_library_columns,
    validate_required_columns,
)


def test_load_library_reads_canonical_processed_dataset():
    df = load_library()

    assert not df.empty
    assert "seq" in df.columns
    assert len(df) == 700


def test_required_columns_are_present():
    df = load_library()

    for column in get_library_required_columns():
        assert column in df.columns


def test_missing_required_columns_raise_value_error():
    df = pd.DataFrame({"seq": ["AUGC"]})

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_required_columns(df, ["seq", "T50"])


def test_split_library_columns_excludes_targets_from_inference_features():
    column_roles = split_library_columns()

    for target in TARGET_COLUMNS:
        assert target not in column_roles["inference_features"]
