from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoswitch.models import (  # noqa: E402
    fit_classifier,
    fit_model_bundle,
    fit_regressors,
    predict_classifier_proba,
    predict_model_bundle,
    predict_regression,
)


def synthetic_training_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 6))
    y = np.array([0, 1] * 15)
    y_t50 = 69.0 + X[:, 0]
    y_slope70 = 0.04 + 0.005 * X[:, 1]
    return X, y, y_t50, y_slope70


def test_classifier_can_fit_on_small_subset_with_both_classes():
    X, y, _, _ = synthetic_training_data()

    model = fit_classifier(X, y)

    assert hasattr(model, "predict_proba")


def test_classifier_probability_output_shape_and_range():
    X, y, _, _ = synthetic_training_data()
    model = fit_classifier(X, y)

    probabilities = predict_classifier_proba(model, X)

    assert probabilities.shape == (len(X),)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_regressors_fit_and_predict_expected_lengths():
    X, _, y_t50, y_slope70 = synthetic_training_data()

    t50_model, slope70_model = fit_regressors(X, y_t50, y_slope70)
    t50_pred, slope70_pred = predict_regression(t50_model, slope70_model, X)

    assert len(t50_pred) == len(X)
    assert len(slope70_pred) == len(X)


def test_model_bundle_returns_prediction_arrays():
    X, y, y_t50, y_slope70 = synthetic_training_data()

    bundle = fit_model_bundle(X, y, y_t50, y_slope70)
    predictions = predict_model_bundle(bundle, X)

    assert set(predictions) == {"p_pass", "t50_pred", "slope70_pred"}
    assert predictions["p_pass"].shape == (len(X),)
    assert predictions["t50_pred"].shape == (len(X),)
    assert predictions["slope70_pred"].shape == (len(X),)
