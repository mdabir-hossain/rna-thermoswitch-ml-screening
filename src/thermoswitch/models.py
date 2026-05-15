"""Model construction and prediction helpers for Phase 2B."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

@dataclass
class ModelBundle:
    """In-memory model bundle for Phase 2B evaluation."""

    classifier: Pipeline
    t50_regressor: Pipeline
    slope70_regressor: Pipeline


def _calibrated_classifier(base_model: GradientBoostingClassifier) -> CalibratedClassifierCV:
    """Build a calibrated classifier across scikit-learn API versions."""
    try:
        return CalibratedClassifierCV(estimator=base_model, method="isotonic", cv=3)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base_model, method="isotonic", cv=3)


def build_classifier(random_state: int = 0) -> Pipeline:
    """Build the calibrated GradientBoosting classifier pipeline from cell 18."""
    base = GradientBoostingClassifier(random_state=random_state)
    return make_pipeline(StandardScaler(), _calibrated_classifier(base))


def fit_classifier(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """Fit the classifier using balanced sample weights where supported."""
    model = build_classifier()
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    try:
        model.fit(X_train, y_train, calibratedclassifiercv__sample_weight=sample_weight)
    except TypeError:
        warnings.warn(
            "sample_weight not accepted by calibrated classifier; fitting without it.",
            RuntimeWarning,
            stacklevel=2,
        )
        model.fit(X_train, y_train)
    return model


def build_t50_regressor(random_state: int = 0) -> Pipeline:
    """Build the T50 regressor pipeline from notebook cell 18."""
    return make_pipeline(
        StandardScaler(),
        GradientBoostingRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=random_state,
        ),
    )


def build_slope70_regressor(random_state: int = 1) -> Pipeline:
    """Build the slope70 regressor pipeline from notebook cell 18."""
    return make_pipeline(
        StandardScaler(),
        RandomForestRegressor(n_estimators=400, random_state=random_state, n_jobs=-1),
    )


def fit_regressors(
    X_train: np.ndarray,
    y_t50_train: np.ndarray,
    y_slope70_train: np.ndarray,
) -> tuple[Pipeline, Pipeline]:
    """Fit the T50 and slope70 regressors."""
    t50_model = build_t50_regressor()
    slope70_model = build_slope70_regressor()
    t50_model.fit(X_train, y_t50_train)
    slope70_model.fit(X_train, y_slope70_train)
    return t50_model, slope70_model


def predict_classifier_proba(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Predict pass probabilities from a fitted classifier pipeline."""
    return model.predict_proba(X)[:, 1]


def predict_regression(
    t50_model: Pipeline,
    slope70_model: Pipeline,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict T50 and slope70 values."""
    return t50_model.predict(X), slope70_model.predict(X)


def fit_model_bundle(
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_t50_train: np.ndarray,
    y_slope70_train: np.ndarray,
) -> ModelBundle:
    """Fit classifier and regressors into an in-memory bundle."""
    classifier = fit_classifier(X_train, y_train)
    t50_model, slope70_model = fit_regressors(X_train, y_t50_train, y_slope70_train)
    return ModelBundle(
        classifier=classifier,
        t50_regressor=t50_model,
        slope70_regressor=slope70_model,
    )


def predict_model_bundle(bundle: ModelBundle, X: np.ndarray) -> dict[str, np.ndarray]:
    """Return pass probability, T50 prediction, and slope70 prediction arrays."""
    p_pass = predict_classifier_proba(bundle.classifier, X)
    t50_pred, slope70_pred = predict_regression(
        bundle.t50_regressor,
        bundle.slope70_regressor,
        X,
    )
    return {
        "p_pass": p_pass,
        "t50_pred": t50_pred,
        "slope70_pred": slope70_pred,
    }
