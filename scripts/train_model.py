"""Train the Phase 2B in-memory model bundle without saving artefacts."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from thermoswitch.data import load_library  # noqa: E402
from thermoswitch.features import build_feature_matrix  # noqa: E402
from thermoswitch.labels import create_pass_labels, extract_regression_targets  # noqa: E402
from thermoswitch.models import fit_model_bundle  # noqa: E402


def main() -> None:
    df = load_library()
    X = build_feature_matrix(df)
    y = create_pass_labels(df)
    y_t50, y_slope70 = extract_regression_targets(df)
    fit_model_bundle(X, y, y_t50, y_slope70)
    print(
        "Training completed for Phase 2B in-memory model bundle. "
        f"Rows={len(df)}, features={X.shape[1]}, positives={int(y.sum())}."
    )
    print("No model artefacts were saved in Phase 2B.")


if __name__ == "__main__":
    main()
