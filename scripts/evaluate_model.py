"""Run Phase 2B model evaluation and write a Markdown report."""

from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from thermoswitch.config import FIGURES_DIR, RESULTS_DIR  # noqa: E402
from thermoswitch.data import load_library  # noqa: E402
from thermoswitch.evaluate import (  # noqa: E402
    cross_validate_models,
    evaluate_train_test,
    format_metrics_summary,
)
from thermoswitch.features import build_feature_matrix, get_feature_names  # noqa: E402
from thermoswitch.labels import create_pass_labels, extract_regression_targets  # noqa: E402
from thermoswitch.shortlist import has_homopolymer  # noqa: E402


def _shortlist_summary() -> dict[str, str | int]:
    shortlist_path = RESULTS_DIR / "ml_topk_diverse_clean.csv"
    order_sheet_path = RESULTS_DIR / "ml_order_sheet.csv"
    summary: dict[str, str | int] = {
        "path": "reports/results/ml_topk_diverse_clean.csv",
        "order_sheet_path": "reports/results/ml_order_sheet.csv",
        "row_count": "not generated",
        "gc_range": "not available",
        "length_range": "not available",
        "homopolymer_warning_count": "not available",
    }
    if not shortlist_path.exists() or not order_sheet_path.exists():
        return summary

    shortlist_df = pd.read_csv(shortlist_path)
    order_df = pd.read_csv(order_sheet_path)
    summary["row_count"] = int(len(shortlist_df))
    if "gc_frac" in order_df.columns and not order_df.empty:
        gc_values = order_df["gc_frac"].astype(float)
        summary["gc_range"] = f"{gc_values.min():.2f}-{gc_values.max():.2f}"
    if "len_nt" in order_df.columns and not order_df.empty:
        lengths = order_df["len_nt"].astype(int)
        summary["length_range"] = f"{lengths.min()}-{lengths.max()} nt"
    if "seq" in shortlist_df.columns:
        summary["homopolymer_warning_count"] = int(
            shortlist_df["seq"].astype(str).map(has_homopolymer).sum()
        )
    return summary


def _figure_list() -> list[str]:
    if not FIGURES_DIR.exists():
        return []
    return sorted(path.name for path in FIGURES_DIR.glob("*.png"))


def main() -> None:
    df = load_library()
    X = build_feature_matrix(df)
    y = create_pass_labels(df)
    y_t50, y_slope70 = extract_regression_targets(df)

    metrics = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset": {
            "path": "data/processed/library_combined.csv",
            "row_count": int(len(df)),
            "positive_count": int(y.sum()),
            "negative_count": int(len(y) - y.sum()),
            "positive_rate": f"{(y.sum() / len(y)):.3f}",
        },
        "features": {
            "feature_count": len(get_feature_names()),
        },
        "shortlist": _shortlist_summary(),
        "figures": _figure_list(),
        "train_test": evaluate_train_test(X, y, y_t50, y_slope70),
        "cross_validation": cross_validate_models(X, y, y_t50, y_slope70, n_splits=5),
    }
    report = format_metrics_summary(metrics)
    output_path = PROJECT_ROOT / "reports" / "model_evaluation.md"
    output_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
