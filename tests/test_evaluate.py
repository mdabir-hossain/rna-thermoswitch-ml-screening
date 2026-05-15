from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoswitch.data import load_library  # noqa: E402
from thermoswitch.evaluate import (  # noqa: E402
    choose_threshold,
    cross_validate_models,
    evaluate_classifier,
    evaluate_regression,
    evaluate_train_test,
    precision_recall_at_k,
    random_ranking_baseline_at_k,
)
from thermoswitch.features import build_feature_matrix  # noqa: E402
from thermoswitch.labels import create_pass_labels, extract_regression_targets  # noqa: E402


def test_choose_threshold_returns_valid_threshold():
    threshold = choose_threshold(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.3, 0.7, 0.9]),
    )

    assert 0.0 <= threshold <= 1.0


def test_precision_recall_at_k_on_known_arrays():
    result = precision_recall_at_k(
        np.array([1, 0, 1, 0]),
        np.array([0.9, 0.8, 0.7, 0.1]),
        2,
    )

    assert result["precision"] == 0.5
    assert np.isclose(result["recall"], 0.5)


def test_evaluate_classifier_returns_pr_auc_and_roc_auc_keys():
    metrics = evaluate_classifier(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
        threshold=0.5,
    )

    assert "average_precision" in metrics
    assert "roc_auc" in metrics
    assert "brier_score" in metrics
    assert "confusion_matrix" in metrics
    assert "random_ranking_baseline_at_k" in metrics


def test_random_ranking_baseline_uses_expected_prevalence_and_k_fraction():
    baseline = random_ranking_baseline_at_k(np.array([1, 0, 0, 0]), k=2)

    assert baseline["precision"] == 0.25
    assert baseline["recall"] == 0.5


def test_evaluate_regression_returns_mae_and_r2_keys():
    metrics = evaluate_regression(np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.9, 3.2]))

    assert "mae" in metrics
    assert "r2" in metrics


def test_train_test_evaluation_runs_on_canonical_dataset():
    df = load_library()
    X = build_feature_matrix(df)
    y = create_pass_labels(df)
    y_t50, y_slope70 = extract_regression_targets(df)

    metrics = evaluate_train_test(X, y, y_t50, y_slope70)

    assert "classifier" in metrics
    assert "regression" in metrics
    assert "average_precision" in metrics["classifier"]


def test_cross_validation_runs_with_reduced_folds_on_fixture():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(36, 8))
    y = np.array([0, 1] * 18)
    y_t50 = 70.0 + X[:, 0]
    y_slope70 = 0.04 + 0.002 * X[:, 1]

    metrics = cross_validate_models(X, y, y_t50, y_slope70, n_splits=3)

    assert metrics["skipped"] is False
    assert metrics["n_splits"] == 3
    assert "average_precision" in metrics["summary"]
    assert "brier_score" in metrics["summary"]
