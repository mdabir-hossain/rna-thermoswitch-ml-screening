"""Generate Top-K, diversified, cleaned shortlist, and order sheet outputs."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from thermoswitch.config import RESULTS_DIR, TOP_K_SAVE  # noqa: E402
from thermoswitch.shortlist import (  # noqa: E402
    ORDER_SHEET_COLUMNS,
    SHORTLIST_COLUMNS,
    diversify_top_k,
    make_order_sheet,
    refine_shortlist,
    select_top_k,
)


PREDICTIONS_PATH = RESULTS_DIR / "ml_predictions.csv"
TOPK_SUMMARY_PATH = RESULTS_DIR / "ml_topk_summary.csv"
TOPK_DIVERSE_PATH = RESULTS_DIR / "ml_topk_diverse.csv"
TOPK_DIVERSE_CLEAN_PATH = RESULTS_DIR / "ml_topk_diverse_clean.csv"
ORDER_SHEET_PATH = RESULTS_DIR / "ml_order_sheet.csv"


def _write_rows(path: Path, rows: list[dict], columns: list[str]) -> None:
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def main() -> None:
    prediction_rows = pd.read_csv(PREDICTIONS_PATH).to_dict(orient="records")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    topk = select_top_k(prediction_rows, k=TOP_K_SAVE)
    _write_rows(TOPK_SUMMARY_PATH, topk, SHORTLIST_COLUMNS)

    diverse = diversify_top_k(prediction_rows, k=TOP_K_SAVE)
    _write_rows(TOPK_DIVERSE_PATH, diverse, SHORTLIST_COLUMNS)

    clean = refine_shortlist(prediction_rows, diverse, k=TOP_K_SAVE)
    _write_rows(TOPK_DIVERSE_CLEAN_PATH, clean, SHORTLIST_COLUMNS)

    order_sheet = make_order_sheet(clean)
    _write_rows(ORDER_SHEET_PATH, order_sheet, ORDER_SHEET_COLUMNS)

    print(f"Wrote {TOPK_SUMMARY_PATH} with {len(topk)} rows.")
    print(f"Wrote {TOPK_DIVERSE_PATH} with {len(diverse)} rows.")
    print(f"Wrote {TOPK_DIVERSE_CLEAN_PATH} with {len(clean)} rows.")
    print(f"Wrote {ORDER_SHEET_PATH} with {len(order_sheet)} rows.")


if __name__ == "__main__":
    main()
