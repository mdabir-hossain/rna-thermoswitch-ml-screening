from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoswitch.labels import (  # noqa: E402
    create_pass_labels,
    extract_regression_targets,
    meets_spec,
    validate_target_columns,
)


def passing_row():
    return {
        "closed_low": 0.90,
        "open_midhi": 0.60,
        "open_high": 0.80,
        "T50": 70.0,
        "slope70": 0.04,
    }


def test_meets_spec_returns_true_for_row_that_meets_thresholds():
    assert meets_spec(passing_row()) is True


def test_meets_spec_returns_false_when_closed_low_fails():
    row = passing_row()
    row["closed_low"] = 0.899

    assert meets_spec(row) is False


def test_meets_spec_returns_false_when_t50_is_outside_range():
    row = passing_row()
    row["T50"] = 72.6

    assert meets_spec(row) is False


def test_meets_spec_returns_false_when_slope70_fails():
    row = passing_row()
    row["slope70"] = 0.039

    assert meets_spec(row) is False


def test_create_pass_labels_returns_same_number_of_labels_as_rows():
    df = pd.DataFrame([passing_row(), {**passing_row(), "closed_low": 0.1}])

    labels = create_pass_labels(df)

    assert len(labels) == len(df)


def test_extract_regression_targets_returns_expected_length():
    df = pd.DataFrame([passing_row(), {**passing_row(), "T50": 71.0}])

    t50, slope70 = extract_regression_targets(df)

    assert len(t50) == len(df)
    assert len(slope70) == len(df)


def test_missing_target_columns_raise_value_error():
    df = pd.DataFrame([passing_row()]).drop(columns=["T50"])

    with pytest.raises(ValueError, match="Missing required target columns"):
        validate_target_columns(df)
    with pytest.raises(ValueError, match="Missing required target columns"):
        create_pass_labels(df)
    with pytest.raises(ValueError, match="Missing required target columns"):
        extract_regression_targets(df)
