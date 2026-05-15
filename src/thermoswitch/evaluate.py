"""Structured evaluation helpers extracted from notebook cell 18."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from thermoswitch.config import BETA, RANDOM_STATE, TARGET_RECALL, THRESHOLD_MODE, TOP_KS
from thermoswitch.models import fit_model_bundle, predict_model_bundle


def _as_array(values: Any) -> np.ndarray:
    return np.asarray(values)


def _has_two_classes(y_true: np.ndarray) -> bool:
    return np.unique(y_true).size >= 2


def _safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if not _has_two_classes(y_true):
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _safe_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    if y_true.size == 0 or np.sum(y_true == 1) == 0:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def _safe_brier_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(brier_score_loss(y_true, scores))


def train_test_split_data(
    X: np.ndarray,
    y: np.ndarray,
    y_t50: np.ndarray,
    y_slope70: np.ndarray,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> dict[str, np.ndarray]:
    """Create the deterministic stratified split used by notebook cell 18."""
    stratify = y if _has_two_classes(y) else None
    split = train_test_split(
        X,
        y,
        y_t50,
        y_slope70,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    X_train, X_test, y_train, y_test, t50_train, t50_test, slope_train, slope_test = split
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_t50_train": t50_train,
        "y_t50_test": t50_test,
        "y_slope70_train": slope_train,
        "y_slope70_test": slope_test,
    }


def choose_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    mode: str = THRESHOLD_MODE,
    beta: float = BETA,
    target_recall: float = TARGET_RECALL,
) -> float:
    """Return a probability threshold from a training precision-recall curve."""
    y_true = _as_array(y_true)
    scores = _as_array(scores)
    if scores.size == 0 or not _has_two_classes(y_true):
        return 0.5

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return 0.5

    mode = mode.lower()
    if mode == "recall":
        idx = np.where(recall[:-1] >= target_recall)[0]
        return float(thresholds[idx[-1]]) if idx.size else float(np.median(thresholds))
    if mode == "f_beta":
        f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-9)
        return float(thresholds[np.argmax(f_beta[:-1])])
    if mode == "pos_quantile":
        pos_scores = scores[y_true == 1]
        if pos_scores.size == 0:
            return 0.5
        q = max(0.0, min(1.0, 1.0 - target_recall))
        return float(np.quantile(pos_scores, q))
    return 0.5


def precision_recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> dict[str, float]:
    """Compute Precision@K and Recall@K using descending score order."""
    y_true = _as_array(y_true)
    scores = _as_array(scores)
    k = min(int(k), len(scores))
    if k <= 0:
        return {"precision": 0.0, "recall": 0.0}
    idx = np.argsort(-scores)[:k]
    true_positives = int((y_true[idx] == 1).sum())
    positives = int((y_true == 1).sum())
    return {
        "precision": float(true_positives / k),
        "recall": float(true_positives / (positives + 1e-9)),
    }


def random_ranking_baseline_at_k(y_true: np.ndarray, k: int) -> dict[str, float]:
    """Return expected random-ranking Precision@K and Recall@K for a label vector."""
    y_true = _as_array(y_true)
    n = int(y_true.size)
    if n == 0:
        return {"precision": 0.0, "recall": 0.0}
    k_eff = min(int(k), n)
    positives = int((y_true == 1).sum())
    prevalence = positives / n
    return {
        "precision": float(prevalence),
        "recall": float(k_eff / n) if positives else 0.0,
    }


def random_ranking_baselines(y_true: np.ndarray, top_ks: list[int] | None = None) -> dict[int, dict[str, float]]:
    """Return expected random-ranking baselines for the configured K values."""
    top_ks = TOP_KS if top_ks is None else top_ks
    return {int(k): random_ranking_baseline_at_k(y_true, int(k)) for k in top_ks}


def evaluate_classifier(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    top_ks: list[int] | None = None,
) -> dict[str, Any]:
    """Return structured classification and ranking metrics."""
    y_true = _as_array(y_true)
    scores = _as_array(scores)
    y_pred = (scores >= threshold).astype(int)
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    top_ks = TOP_KS if top_ks is None else top_ks
    return {
        "average_precision": _safe_average_precision(y_true, scores),
        "roc_auc": _safe_roc_auc(y_true, scores),
        "brier_score": _safe_brier_score(y_true, scores),
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "classification_report": report,
        "precision_recall_at_k": {
            int(k): precision_recall_at_k(y_true, scores, int(k)) for k in top_ks
        },
        "random_ranking_baseline_at_k": random_ranking_baselines(y_true, top_ks),
    }


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE and R2 for a regression target."""
    y_true = _as_array(y_true)
    y_pred = _as_array(y_pred)
    if y_true.size == 0:
        return {"mae": float("nan"), "r2": float("nan")}
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if y_true.size >= 2 else float("nan"),
    }


def evaluate_train_test(
    X: np.ndarray,
    y: np.ndarray,
    y_t50: np.ndarray,
    y_slope70: np.ndarray,
) -> dict[str, Any]:
    """Fit models on the train split and evaluate on the held-out split."""
    split = train_test_split_data(X, y, y_t50, y_slope70)
    bundle = fit_model_bundle(
        split["X_train"],
        split["y_train"],
        split["y_t50_train"],
        split["y_slope70_train"],
    )
    train_preds = predict_model_bundle(bundle, split["X_train"])
    threshold = choose_threshold(split["y_train"], train_preds["p_pass"])
    test_preds = predict_model_bundle(bundle, split["X_test"])
    return {
        "split": {
            "train_size": int(len(split["y_train"])),
            "test_size": int(len(split["y_test"])),
            "train_positives": int(np.sum(split["y_train"] == 1)),
            "test_positives": int(np.sum(split["y_test"] == 1)),
            "train_negatives": int(np.sum(split["y_train"] == 0)),
            "test_negatives": int(np.sum(split["y_test"] == 0)),
        },
        "classifier": evaluate_classifier(split["y_test"], test_preds["p_pass"], threshold),
        "regression": {
            "t50": evaluate_regression(split["y_t50_test"], test_preds["t50_pred"]),
            "slope70": evaluate_regression(split["y_slope70_test"], test_preds["slope70_pred"]),
        },
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    finite = np.array([value for value in values if not math.isnan(value)], dtype=float)
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(np.mean(finite)), "std": float(np.std(finite))}


def cross_validate_models(
    X: np.ndarray,
    y: np.ndarray,
    y_t50: np.ndarray,
    y_slope70: np.ndarray,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Run stratified cross-validation and return structured metric summaries."""
    class_counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
    feasible_splits = int(min(n_splits, class_counts.min()))
    if feasible_splits < 2:
        return {"skipped": True, "reason": "Need at least two examples per class for stratified CV."}

    skf = StratifiedKFold(n_splits=feasible_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics: list[dict[str, Any]] = []
    for train_idx, valid_idx in skf.split(X, y):
        bundle = fit_model_bundle(
            X[train_idx],
            y[train_idx],
            y_t50[train_idx],
            y_slope70[train_idx],
        )
        train_preds = predict_model_bundle(bundle, X[train_idx])
        threshold = choose_threshold(y[train_idx], train_preds["p_pass"])
        valid_preds = predict_model_bundle(bundle, X[valid_idx])
        fold_metrics.append(
            {
                "classifier": evaluate_classifier(y[valid_idx], valid_preds["p_pass"], threshold),
                "regression": {
                    "t50": evaluate_regression(y_t50[valid_idx], valid_preds["t50_pred"]),
                    "slope70": evaluate_regression(y_slope70[valid_idx], valid_preds["slope70_pred"]),
                },
            }
        )

    top_k_summary: dict[int, dict[str, dict[str, float]]] = {}
    for k in TOP_KS:
        top_k_summary[k] = {
            "precision": _mean_std(
                [fold["classifier"]["precision_recall_at_k"][k]["precision"] for fold in fold_metrics]
            ),
            "recall": _mean_std(
                [fold["classifier"]["precision_recall_at_k"][k]["recall"] for fold in fold_metrics]
            ),
        }

    return {
        "skipped": False,
        "n_splits": feasible_splits,
        "folds": fold_metrics,
        "summary": {
            "average_precision": _mean_std(
                [fold["classifier"]["average_precision"] for fold in fold_metrics]
            ),
            "roc_auc": _mean_std([fold["classifier"]["roc_auc"] for fold in fold_metrics]),
            "brier_score": _mean_std([fold["classifier"]["brier_score"] for fold in fold_metrics]),
            "precision_recall_at_k": top_k_summary,
            "t50_mae": _mean_std([fold["regression"]["t50"]["mae"] for fold in fold_metrics]),
            "t50_r2": _mean_std([fold["regression"]["t50"]["r2"] for fold in fold_metrics]),
            "slope70_mae": _mean_std(
                [fold["regression"]["slope70"]["mae"] for fold in fold_metrics]
            ),
            "slope70_r2": _mean_std([fold["regression"]["slope70"]["r2"] for fold in fold_metrics]),
        },
    }


def _fmt(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_mean_std(summary: dict[str, float]) -> str:
    return f"{_fmt(summary['mean'])} +/- {_fmt(summary['std'])}"


def format_metrics_summary(metrics: dict[str, Any]) -> str:
    """Format evaluation metrics as recruiter-ready Markdown."""
    generated_at = metrics.get("generated_at") or datetime.now().strftime("%Y-%m-%d %H:%M")
    dataset = metrics.get("dataset", {})
    features = metrics.get("features", {})
    shortlist = metrics.get("shortlist", {})
    figures = metrics.get("figures", [])

    lines = [
        "# RNA Thermoswitch Model Evaluation",
        "",
        f"Generated: {generated_at}",
        "",
        "## Disclaimer",
        "",
        "The labels and regression targets in this report are thermodynamic simulation-derived. The outputs support candidate prioritisation under simulation assumptions; no wet-lab validation has been performed. These results should not be interpreted as experimental proof or production-ready genetic control.",
        "",
        "## Dataset Summary",
        "",
        f"- Canonical dataset: `{dataset.get('path', 'data/processed/library_combined.csv')}`",
        f"- Rows: {dataset.get('row_count', 'n/a')}",
        f"- Simulation-derived positives: {dataset.get('positive_count', 'n/a')}",
        f"- Simulation-derived negatives: {dataset.get('negative_count', 'n/a')}",
        f"- Positive rate: {dataset.get('positive_rate', 'n/a')}",
        "- Target/label source: thermodynamic simulation-derived columns `closed_low`, `open_midhi`, `open_high`, `T50`, and `slope70`.",
        "",
        "## Split Protocol",
        "",
        "- Deterministic stratified train/test split with `random_state=42`.",
        "- 5-fold stratified cross-validation is implemented when each class has enough examples.",
        "- Threshold selection uses the configured F-beta mode on training predictions.",
        "",
        "## Feature Summary",
        "",
        f"- Feature count: {features.get('feature_count', 'n/a')}",
        "- Input features include sequence length, GC fraction, design metadata (`extra_pairs`, `pGC`, `loop_len`), dinucleotide frequencies, trinucleotide frequencies, and fixed-length RBS-window one-hot features.",
        "- Target and output columns are excluded from inference features, including `closed_low`, `open_midhi`, `open_high`, `T50`, `slope70`, `slope_max`, `reentrant_pen`, `score`, `curve`, `p_pass`, `t50_pred`, and `slope70_pred`.",
        "",
        "## Model Summary",
        "",
        "- Classifier: calibrated scikit-learn GradientBoostingClassifier pipeline with standard scaling and isotonic calibration.",
        "- T50 regressor: GradientBoostingRegressor pipeline with standard scaling.",
        "- slope70 regressor: RandomForestRegressor pipeline with standard scaling.",
        "- Model artefacts are currently fitted in memory by the canonical scripts; production-style saved artefacts are future work.",
        "",
    ]
    if "train_test" in metrics:
        tt = metrics["train_test"]
        clf = tt["classifier"]
        reg = tt["regression"]
        lines.extend(
            [
                "## Train/Test Evaluation",
                "",
                f"- Train size: {tt['split']['train_size']}",
                f"- Test size: {tt['split']['test_size']}",
                f"- Test positives: {tt['split']['test_positives']}",
                f"- Test negatives: {tt['split']['test_negatives']}",
                f"- Average Precision / PR-AUC: {_fmt(clf['average_precision'])}",
                f"- ROC-AUC: {_fmt(clf['roc_auc'])}",
                f"- Brier score: {_fmt(clf['brier_score'])}",
                f"- Threshold: {_fmt(clf['threshold'])}",
                f"- Confusion matrix: TN={clf['confusion_matrix']['tn']}, FP={clf['confusion_matrix']['fp']}, FN={clf['confusion_matrix']['fn']}, TP={clf['confusion_matrix']['tp']}",
                f"- T50 MAE: {_fmt(reg['t50']['mae'])}",
                f"- T50 R2: {_fmt(reg['t50']['r2'])}",
                f"- slope70 MAE: {_fmt(reg['slope70']['mae'])}",
                f"- slope70 R2: {_fmt(reg['slope70']['r2'])}",
                "",
                "### Ranking Metrics",
                "",
                "| K | Model Precision@K | Model Recall@K | Expected Random Precision@K | Expected Random Recall@K |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for k, values in clf["precision_recall_at_k"].items():
            baseline = clf["random_ranking_baseline_at_k"][k]
            lines.append(
                f"| {k} | {_fmt(values['precision'])} | {_fmt(values['recall'])} | {_fmt(baseline['precision'])} | {_fmt(baseline['recall'])} |"
            )
        lines.append("")
        lines.extend(
            [
                "The random baseline is the expected value from random ranking at the same K on the held-out test labels. It provides context for ranking metrics without introducing a new model family.",
                "",
            ]
        )
    if "cross_validation" in metrics:
        cv = metrics["cross_validation"]
        lines.extend(["## Cross-Validation", ""])
        if cv.get("skipped"):
            lines.append(f"- Skipped: {cv['reason']}")
        else:
            summary = cv["summary"]
            lines.extend(
                [
                    f"- Folds: {cv['n_splits']}",
                    f"- Average Precision / PR-AUC: {_fmt_mean_std(summary['average_precision'])}",
                    f"- ROC-AUC: {_fmt_mean_std(summary['roc_auc'])}",
                    f"- Brier score: {_fmt_mean_std(summary['brier_score'])}",
                    f"- T50 MAE: {_fmt_mean_std(summary['t50_mae'])}",
                    f"- T50 R2: {_fmt_mean_std(summary['t50_r2'])}",
                    f"- slope70 MAE: {_fmt_mean_std(summary['slope70_mae'])}",
                    f"- slope70 R2: {_fmt_mean_std(summary['slope70_r2'])}",
                    "",
                    "### Cross-Validated Ranking Metrics",
                    "",
                    "| K | Precision@K mean +/- sd | Recall@K mean +/- sd |",
                    "|---:|---:|---:|",
                ]
            )
            for k, values in summary["precision_recall_at_k"].items():
                lines.append(
                    f"| {k} | {_fmt_mean_std(values['precision'])} | {_fmt_mean_std(values['recall'])} |"
                )
        lines.append("")
    lines.extend(
        [
            "## Calibration Summary",
            "",
            "- Brier score is reported above for the held-out split and cross-validation folds.",
            "- The calibration figure is generated at `reports/figures/f4_calibration.png` from the current prediction artefact and simulation-derived labels.",
            "",
            "## Shortlist Summary",
            "",
            f"- Clean shortlist path: `{shortlist.get('path', 'reports/results/ml_topk_diverse_clean.csv')}`",
            f"- Clean shortlist size: {shortlist.get('row_count', 'n/a')}",
            f"- Order-sheet path: `{shortlist.get('order_sheet_path', 'reports/results/ml_order_sheet.csv')}`",
            f"- GC range in order sheet: {shortlist.get('gc_range', 'n/a')}",
            f"- Length range in order sheet: {shortlist.get('length_range', 'n/a')}",
            f"- Homopolymer warning count at n>=7: {shortlist.get('homopolymer_warning_count', 'n/a')}",
            "- Ranking uses predicted pass probability first, then closeness of predicted T50 to 70 C, then predicted slope70.",
            "- Diversity filtering uses pairwise 3-mer Jaccard distance over the top-ranked pool.",
            "- QC filtering checks GC fraction and homopolymer runs before creating the practical order sheet.",
            "",
            "## Generated Figures",
            "",
        ]
    )
    if figures:
        lines.extend([f"- `reports/figures/{figure}`" for figure in figures])
    else:
        lines.append("- Figure list not provided by the report script.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These metrics support a careful claim that the canonical scripts can train and evaluate a simulation-guided ML ranking workflow on thermodynamic simulation-derived targets. The ranking outputs are useful for candidate prioritisation and documentation of a reproducible scientific ML workflow.",
            "",
            "They do not support claims of wet-lab validation, biological proof, deployment readiness, or guaranteed thermoswitch behaviour. Full-library prediction and figure artefacts are generated after fitting an in-memory model on the same canonical dataset, so they should be treated as scoring artefacts rather than held-out performance estimates.",
            "",
            "## Limitations",
            "",
            "- Labels and regression targets are thermodynamic simulation-derived.",
            "- The canonical processed dataset is small for biological sequence modelling.",
            "- Sequence-similarity-aware splitting is not yet implemented.",
            "- Only an expected random-ranking baseline is currently included; richer rule-based or model baselines remain future work.",
            "- Model artefacts are not yet saved as production-style files.",
            "- F1 openness plotting uses stored simulation curve values in the lightweight public workflow rather than fresh NUPACK recomputation.",
            "",
            "## Reproducibility Commands",
            "",
            "```powershell",
            "& '.\\.venv\\Scripts\\python.exe' -m pytest tests",
            "& '.\\.venv\\Scripts\\python.exe' scripts\\evaluate_model.py",
            "& '.\\.venv\\Scripts\\python.exe' scripts\\predict_candidates.py",
            "& '.\\.venv\\Scripts\\python.exe' scripts\\make_shortlist.py",
            "& '.\\.venv\\Scripts\\python.exe' scripts\\make_figures.py",
            "```",
            "",
            "## Interpretation Guardrails",
            "",
            "- Labels are thermodynamic simulation-derived, not wet-lab validated.",
            "- Metrics are from the canonical evaluation script.",
            "- Results support candidate prioritisation under simulation assumptions.",
            "- README claims should quote this report only with the same caveats.",
        ]
    )
    return "\n".join(lines) + "\n"
