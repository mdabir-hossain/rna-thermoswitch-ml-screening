from pathlib import Path
import subprocess
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase2c_scripts_create_expected_outputs():
    predict = subprocess.run(
        [sys.executable, "scripts/predict_candidates.py"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "ml_predictions.csv" in predict.stdout

    shortlist = subprocess.run(
        [sys.executable, "scripts/make_shortlist.py"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "ml_order_sheet.csv" in shortlist.stdout

    expected = {
        "ml_predictions.csv": ["id", "seq", "len", "p_pass", "t50_pred", "slope70_pred"],
        "ml_topk_summary.csv": ["rank", "id", "seq", "p_pass", "t50_pred", "slope70_pred"],
        "ml_topk_diverse.csv": ["rank", "id", "seq", "p_pass", "t50_pred", "slope70_pred"],
        "ml_topk_diverse_clean.csv": ["rank", "id", "seq", "p_pass", "t50_pred", "slope70_pred"],
        "ml_order_sheet.csv": [
            "rank",
            "id",
            "seq",
            "len_nt",
            "gc_frac",
            "p_pass",
            "t50_pred",
            "slope70_pred",
        ],
    }
    for filename, columns in expected.items():
        path = PROJECT_ROOT / "reports" / "results" / filename
        assert path.exists()
        df = pd.read_csv(path)
        assert not df.empty
        for column in columns:
            assert column in df.columns
