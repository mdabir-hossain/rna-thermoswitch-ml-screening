"""Train the Phase 2B bundle in memory and export candidate predictions."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from thermoswitch.config import LIBRARY_COMBINED_PATH, RESULTS_DIR  # noqa: E402
from thermoswitch.data import load_library  # noqa: E402
from thermoswitch.features import build_feature_matrix  # noqa: E402
from thermoswitch.labels import create_pass_labels, extract_regression_targets  # noqa: E402
from thermoswitch.models import fit_model_bundle, predict_model_bundle  # noqa: E402


OUTPUT_PATH = RESULTS_DIR / "ml_predictions.csv"
REPORTING_TARGET_COLUMNS = ["closed_low", "open_midhi", "open_high", "T50", "slope70"]


def main() -> None:
    df = load_library(LIBRARY_COMBINED_PATH)
    X = build_feature_matrix(df)

    # Targets are used only to fit the temporary Phase 2B model bundle.
    # They are not part of the inference feature matrix.
    y = create_pass_labels(df)
    y_t50, y_slope70 = extract_regression_targets(df)
    bundle = fit_model_bundle(X, y, y_t50, y_slope70)
    predictions = predict_model_bundle(bundle, X)

    output = pd.DataFrame(
        {
            "id": df["id"],
            "seq": df["seq"],
            "len": df["seq"].astype(str).str.len(),
            "p_pass": predictions["p_pass"],
            "t50_pred": predictions["t50_pred"],
            "slope70_pred": predictions["slope70_pred"],
        }
    )
    for column in REPORTING_TARGET_COLUMNS:
        if column in df.columns:
            output[column] = df[column]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH} with {len(output)} rows.")
    print("Observed target columns, if present, are included only as reporting/reference columns.")


if __name__ == "__main__":
    main()
