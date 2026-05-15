"""Figure generation for the RNA thermoswitch screening workflow."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

from thermoswitch.config import (
    FIGURES_DIR,
    GUARD_SLOPE_MIN,
    GUARD_T50_RANGE,
    LIBRARY_COMBINED_PATH,
    RESULTS_DIR,
    SPEC,
    TOP_KS,
)
from thermoswitch.labels import TARGET_COLUMNS, meets_spec
from thermoswitch.shortlist import gc_fraction_for_shortlist, jaccard_distance


PREDICTIONS_PATH = RESULTS_DIR / "ml_predictions.csv"
SHORTLIST_PATH = RESULTS_DIR / "ml_topk_diverse_clean.csv"

EXPECTED_FIGURE_FILENAMES = [
    "f1_openness_best.png",
    "f2_pr_curve.png",
    "f3_precision_at_k.png",
    "f3_recall_at_k.png",
    "f4_calibration.png",
    "f5_top20_t50_hist.png",
    "f5_top20_slope_hist.png",
    "f6_gc_hist.png",
    "f6_len_hist.png",
    "s4_diversity_hist.png",
]

OKABE_ITO = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
]


def set_plot_style() -> None:
    """Apply a small, reproducible matplotlib style for saved report figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )


def ensure_output_dir(output_dir: str | Path = FIGURES_DIR) -> Path:
    """Create and return the figure output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_figure(fig: plt.Figure, path: str | Path) -> Path:
    """Save a figure with consistent layout and close it."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def load_plotting_inputs(
    library_path: str | Path = LIBRARY_COMBINED_PATH,
    predictions_path: str | Path = PREDICTIONS_PATH,
    shortlist_path: str | Path = SHORTLIST_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the canonical data artefacts used by the Phase 2D figures."""
    return (
        pd.read_csv(library_path),
        pd.read_csv(predictions_path),
        pd.read_csv(shortlist_path),
    )


def _require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {', '.join(missing)}")


def simulation_labels_from_predictions(predictions_df: pd.DataFrame) -> np.ndarray:
    """Create simulation-derived pass labels from prediction reporting columns."""
    _require_columns(predictions_df, TARGET_COLUMNS, "predictions_df")
    return np.array([int(meets_spec(row)) for _, row in predictions_df.iterrows()], dtype=int)


def _series_from_curve(value: object) -> list[float]:
    if pd.isna(value):
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(";") if part.strip()]
        return [float(part) for part in parts]
    return list(value)


def plot_f1_openness_best(
    library_df: pd.DataFrame,
    shortlist_df: pd.DataFrame,
    output_dir: str | Path = FIGURES_DIR,
) -> Path:
    """Plot the simulated openness curve for the first clean shortlisted candidate."""
    _require_columns(shortlist_df, ["id"], "shortlist_df")
    _require_columns(library_df, ["id", "curve"], "library_df")
    output_path = ensure_output_dir(output_dir) / "f1_openness_best.png"

    first_id = shortlist_df.iloc[0]["id"]
    match = library_df[library_df["id"] == first_id]
    if match.empty:
        match = library_df.iloc[[0]]
    row = match.iloc[0]

    curve = _series_from_curve(row["curve"])
    if not curve:
        raise ValueError("No simulated openness curve values are available for F1.")
    temps = np.arange(50, 50 + 2 * len(curve), 2)

    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(temps, curve, color=OKABE_ITO[0], marker="o", linewidth=2.0, label="RBS openness")
    ax.axvspan(GUARD_T50_RANGE[0], GUARD_T50_RANGE[1], color=OKABE_ITO[2], alpha=0.15, label="T50 target band")
    if "T50" in row and pd.notna(row["T50"]):
        ax.axvline(float(row["T50"]), color=OKABE_ITO[1], linestyle="--", linewidth=1.8, label="Simulated T50")
    ax.axvline(70.0, color="#333333", linestyle=":", linewidth=1.5, label="70 C")
    ax.set_title(f"Best shortlisted candidate openness profile ({first_id})")
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Simulated RBS openness")
    ax.set_ylim(0, min(1.05, max(1.0, max(curve) * 1.15)))
    ax.legend(loc="best")
    return save_figure(fig, output_path)


def plot_pr_curve(predictions_df: pd.DataFrame, output_dir: str | Path = FIGURES_DIR) -> Path:
    """Plot precision-recall curve from simulation-derived pass labels."""
    _require_columns(predictions_df, ["p_pass"], "predictions_df")
    y_true = simulation_labels_from_predictions(predictions_df)
    scores = predictions_df["p_pass"].to_numpy(dtype=float)
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)

    f2_scores = (5 * precision * recall) / np.maximum((4 * precision) + recall, 1e-12)
    best_idx = int(np.nanargmax(f2_scores))
    threshold_text = "n/a"
    if len(thresholds) and best_idx < len(thresholds):
        threshold_text = f"{thresholds[best_idx]:.3f}"

    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.step(recall, precision, where="post", color=OKABE_ITO[0], linewidth=2.0, label=f"AP = {ap:.3f}")
    ax.scatter(
        recall[best_idx],
        precision[best_idx],
        color=OKABE_ITO[1],
        zorder=3,
        label=f"Best F2 threshold = {threshold_text}",
    )
    ax.set_title("Precision-recall curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="best")
    return save_figure(fig, ensure_output_dir(output_dir) / "f2_pr_curve.png")


def _precision_recall_at_k_arrays(
    y_true: np.ndarray,
    scores: np.ndarray,
    ks: Iterable[int] = TOP_KS,
) -> tuple[list[int], list[float], list[float]]:
    order = np.argsort(scores)[::-1]
    total_positive = int(np.sum(y_true))
    used_ks: list[int] = []
    precisions: list[float] = []
    recalls: list[float] = []
    for k in ks:
        k_eff = min(int(k), len(scores))
        if k_eff <= 0:
            continue
        top = y_true[order[:k_eff]]
        used_ks.append(k_eff)
        precisions.append(float(np.mean(top)))
        recalls.append(float(np.sum(top) / total_positive) if total_positive else 0.0)
    return used_ks, precisions, recalls


def plot_precision_recall_at_k(
    predictions_df: pd.DataFrame,
    output_dir: str | Path = FIGURES_DIR,
    ks: Iterable[int] = TOP_KS,
) -> list[Path]:
    """Plot Precision@K and Recall@K as separate canonical figures."""
    _require_columns(predictions_df, ["p_pass"], "predictions_df")
    y_true = simulation_labels_from_predictions(predictions_df)
    scores = predictions_df["p_pass"].to_numpy(dtype=float)
    used_ks, precisions, recalls = _precision_recall_at_k_arrays(y_true, scores, ks)
    output_path = ensure_output_dir(output_dir)

    set_plot_style()
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.bar([str(k) for k in used_ks], precisions, color=OKABE_ITO[0])
    ax.set_title("Precision at K")
    ax.set_xlabel("K")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, min(1.0, max(precisions + [0.1]) * 1.25))
    p_path = save_figure(fig, output_path / "f3_precision_at_k.png")

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.bar([str(k) for k in used_ks], recalls, color=OKABE_ITO[2])
    ax.set_title("Recall at K")
    ax.set_xlabel("K")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, min(1.0, max(recalls + [0.1]) * 1.25))
    r_path = save_figure(fig, output_path / "f3_recall_at_k.png")
    return [p_path, r_path]


def plot_calibration(predictions_df: pd.DataFrame, output_dir: str | Path = FIGURES_DIR) -> Path:
    """Plot probability calibration against simulation-derived labels."""
    _require_columns(predictions_df, ["p_pass"], "predictions_df")
    y_true = simulation_labels_from_predictions(predictions_df)
    scores = predictions_df["p_pass"].to_numpy(dtype=float)
    frac_pos, mean_pred = calibration_curve(y_true, scores, n_bins=8, strategy="quantile")

    set_plot_style()
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    ax.plot([0, 1], [0, 1], color="#666666", linestyle=":", linewidth=1.5, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, color=OKABE_ITO[3], marker="o", linewidth=2.0, label="Model")
    ax.set_title("Calibration curve")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction passing simulation spec")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    return save_figure(fig, ensure_output_dir(output_dir) / "f4_calibration.png")


def plot_top20_histograms(shortlist_df: pd.DataFrame, output_dir: str | Path = FIGURES_DIR) -> list[Path]:
    """Plot predicted T50 and slope distributions for the clean Top-K shortlist."""
    _require_columns(shortlist_df, ["t50_pred", "slope70_pred"], "shortlist_df")
    output_path = ensure_output_dir(output_dir)
    paths: list[Path] = []

    t50_values = shortlist_df["t50_pred"].to_numpy(dtype=float)
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(t50_values, bins=min(8, max(3, len(t50_values))), color=OKABE_ITO[0], edgecolor="#222222")
    ax.axvspan(GUARD_T50_RANGE[0], GUARD_T50_RANGE[1], color=OKABE_ITO[2], alpha=0.18, label="Target band")
    ax.axvline(np.mean(t50_values), color=OKABE_ITO[1], linestyle="--", label="Mean")
    ax.set_title("Top 20 predicted T50")
    ax.set_xlabel("Predicted T50 (C)")
    ax.set_ylabel("Candidate count")
    ax.legend(loc="best")
    paths.append(save_figure(fig, output_path / "f5_top20_t50_hist.png"))

    slope_values = shortlist_df["slope70_pred"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(slope_values, bins=min(8, max(3, len(slope_values))), color=OKABE_ITO[4], edgecolor="#222222")
    ax.axvline(GUARD_SLOPE_MIN, color=OKABE_ITO[1], linestyle="--", label="Minimum guard")
    ax.axvline(np.mean(slope_values), color=OKABE_ITO[0], linestyle=":", label="Mean")
    ax.set_title("Top 20 predicted slope at 70 C")
    ax.set_xlabel("Predicted slope70")
    ax.set_ylabel("Candidate count")
    ax.legend(loc="best")
    paths.append(save_figure(fig, output_path / "f5_top20_slope_hist.png"))
    return paths


def plot_library_qc_histograms(predictions_df: pd.DataFrame, output_dir: str | Path = FIGURES_DIR) -> list[Path]:
    """Plot candidate library sequence length and GC distributions."""
    _require_columns(predictions_df, ["seq"], "predictions_df")
    output_path = ensure_output_dir(output_dir)
    seqs = predictions_df["seq"].astype(str)
    lengths = seqs.str.len().to_numpy(dtype=int)
    gc_values = np.array([gc_fraction_for_shortlist(seq) for seq in seqs], dtype=float)

    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(gc_values, bins=16, color=OKABE_ITO[2], edgecolor="#222222")
    ax.set_title("Candidate library GC fraction")
    ax.set_xlabel("GC fraction")
    ax.set_ylabel("Candidate count")
    gc_path = save_figure(fig, output_path / "f6_gc_hist.png")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(lengths, bins=16, color=OKABE_ITO[5], edgecolor="#222222")
    ax.set_title("Candidate library sequence length")
    ax.set_xlabel("Length (nt)")
    ax.set_ylabel("Candidate count")
    len_path = save_figure(fig, output_path / "f6_len_hist.png")
    return [gc_path, len_path]


def plot_diversity_histogram(shortlist_df: pd.DataFrame, output_dir: str | Path = FIGURES_DIR) -> Path:
    """Plot pairwise 3-mer Jaccard distances among shortlisted candidates."""
    _require_columns(shortlist_df, ["seq"], "shortlist_df")
    seqs = shortlist_df["seq"].astype(str).tolist()
    distances: list[float] = []
    for idx, seq_a in enumerate(seqs):
        for seq_b in seqs[idx + 1 :]:
            distances.append(jaccard_distance(seq_a, seq_b, k=3))
    if not distances:
        distances = [0.0]

    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(distances, bins=12, color=OKABE_ITO[6], edgecolor="#222222")
    ax.set_title("Top shortlist sequence diversity")
    ax.set_xlabel("Pairwise 3-mer Jaccard distance")
    ax.set_ylabel("Pair count")
    return save_figure(fig, ensure_output_dir(output_dir) / "s4_diversity_hist.png")


def make_all_figures(
    library_path: str | Path = LIBRARY_COMBINED_PATH,
    predictions_path: str | Path = PREDICTIONS_PATH,
    shortlist_path: str | Path = SHORTLIST_PATH,
    output_dir: str | Path = FIGURES_DIR,
) -> list[Path]:
    """Generate the canonical Phase 2D figure set."""
    library_df, predictions_df, shortlist_df = load_plotting_inputs(
        library_path=library_path,
        predictions_path=predictions_path,
        shortlist_path=shortlist_path,
    )
    output_path = ensure_output_dir(output_dir)
    paths: list[Path] = []
    paths.append(plot_f1_openness_best(library_df, shortlist_df, output_path))
    paths.append(plot_pr_curve(predictions_df, output_path))
    paths.extend(plot_precision_recall_at_k(predictions_df, output_path))
    paths.append(plot_calibration(predictions_df, output_path))
    paths.extend(plot_top20_histograms(shortlist_df, output_path))
    paths.extend(plot_library_qc_histograms(predictions_df, output_path))
    paths.append(plot_diversity_histogram(shortlist_df, output_path))
    return paths
