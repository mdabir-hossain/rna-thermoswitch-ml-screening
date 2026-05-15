"""Project constants for the RNA thermoswitch ML screening workflow."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
LIBRARY_COMBINED_PATH = DATA_PROCESSED_PATH / "library_combined.csv"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RANDOM_STATE = 42

TOP_KS = [10, 20, 50]
TOP_K_SAVE = 20

SPEC = {
    "min_closed_low": 0.90,
    "min_open_midhi": 0.60,
    "min_open_high": 0.80,
    "t50_low": 69.0,
    "t50_high": 72.5,
    "min_slope70": 0.04,
}

THRESHOLD_MODE = "f_beta"
BETA = 2.0
TARGET_RECALL = 0.70

USE_GUARDS = True
GUARD_T50_RANGE = (69.0, 72.5)
GUARD_SLOPE_MIN = 0.040

HOMO_N = 7
GC_MIN = 0.25
GC_MAX = 0.70
