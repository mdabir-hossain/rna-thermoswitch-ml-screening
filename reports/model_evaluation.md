# RNA Thermoswitch Model Evaluation

Generated: 2026-05-15 07:37

## Disclaimer

The labels and regression targets in this report are thermodynamic simulation-derived. The outputs support candidate prioritisation under simulation assumptions; no wet-lab validation has been performed. These results should not be interpreted as experimental proof or production-ready genetic control.

## Dataset Summary

- Canonical dataset: `data/processed/library_combined.csv`
- Rows: 700
- Simulation-derived positives: 112
- Simulation-derived negatives: 588
- Positive rate: 0.160
- Target/label source: thermodynamic simulation-derived columns `closed_low`, `open_midhi`, `open_high`, `T50`, and `slope70`.

## Split Protocol

- Deterministic stratified train/test split with `random_state=42`.
- 5-fold stratified cross-validation is implemented when each class has enough examples.
- Threshold selection uses the configured F-beta mode on training predictions.

## Feature Summary

- Feature count: 229
- Input features include sequence length, GC fraction, design metadata (`extra_pairs`, `pGC`, `loop_len`), dinucleotide frequencies, trinucleotide frequencies, and fixed-length RBS-window one-hot features.
- Target and output columns are excluded from inference features, including `closed_low`, `open_midhi`, `open_high`, `T50`, `slope70`, `slope_max`, `reentrant_pen`, `score`, `curve`, `p_pass`, `t50_pred`, and `slope70_pred`.

## Model Summary

- Classifier: calibrated scikit-learn GradientBoostingClassifier pipeline with standard scaling and isotonic calibration.
- T50 regressor: GradientBoostingRegressor pipeline with standard scaling.
- slope70 regressor: RandomForestRegressor pipeline with standard scaling.
- Model artefacts are currently fitted in memory by the canonical scripts; production-style saved artefacts are future work.

## Train/Test Evaluation

- Train size: 560
- Test size: 140
- Test positives: 22
- Test negatives: 118
- Average Precision / PR-AUC: 0.385
- ROC-AUC: 0.736
- Brier score: 0.212
- Threshold: 0.615
- Confusion matrix: TN=107, FP=11, FN=13, TP=9
- T50 MAE: 1.168
- T50 R2: 0.353
- slope70 MAE: 0.006
- slope70 R2: 0.347

### Ranking Metrics

| K | Model Precision@K | Model Recall@K | Expected Random Precision@K | Expected Random Recall@K |
|---:|---:|---:|---:|---:|
| 10 | 0.500 | 0.227 | 0.157 | 0.071 |
| 20 | 0.450 | 0.409 | 0.157 | 0.143 |
| 50 | 0.280 | 0.636 | 0.157 | 0.357 |

The random baseline is the expected value from random ranking at the same K on the held-out test labels. It provides context for ranking metrics without introducing a new model family.

## Cross-Validation

- Folds: 5
- Average Precision / PR-AUC: 0.285 +/- 0.074
- ROC-AUC: 0.624 +/- 0.060
- Brier score: 0.226 +/- 0.009
- T50 MAE: 1.278 +/- 0.123
- T50 R2: 0.354 +/- 0.094
- slope70 MAE: 0.006 +/- 0.000
- slope70 R2: 0.337 +/- 0.082

### Cross-Validated Ranking Metrics

| K | Precision@K mean +/- sd | Recall@K mean +/- sd |
|---:|---:|---:|
| 10 | 0.320 +/- 0.147 | 0.143 +/- 0.067 |
| 20 | 0.290 +/- 0.102 | 0.260 +/- 0.093 |
| 50 | 0.248 +/- 0.039 | 0.554 +/- 0.090 |

## Calibration Summary

- Brier score is reported above for the held-out split and cross-validation folds.
- The calibration figure is generated at `reports/figures/f4_calibration.png` from the current prediction artefact and simulation-derived labels.

## Shortlist Summary

- Clean shortlist path: `reports/results/ml_topk_diverse_clean.csv`
- Clean shortlist size: 20
- Order-sheet path: `reports/results/ml_order_sheet.csv`
- GC range in order sheet: 0.26-0.33
- Length range in order sheet: 46-50 nt
- Homopolymer warning count at n>=7: 0
- Ranking uses predicted pass probability first, then closeness of predicted T50 to 70 C, then predicted slope70.
- Diversity filtering uses pairwise 3-mer Jaccard distance over the top-ranked pool.
- QC filtering checks GC fraction and homopolymer runs before creating the practical order sheet.

## Generated Figures

- `reports/figures/f1_openness_best.png`
- `reports/figures/f2_pr_curve.png`
- `reports/figures/f3_precision_at_k.png`
- `reports/figures/f3_recall_at_k.png`
- `reports/figures/f4_calibration.png`
- `reports/figures/f5_top20_slope_hist.png`
- `reports/figures/f5_top20_t50_hist.png`
- `reports/figures/f6_gc_hist.png`
- `reports/figures/f6_len_hist.png`
- `reports/figures/s4_diversity_hist.png`

## Interpretation

These metrics support a careful claim that the canonical scripts can train and evaluate a simulation-guided ML ranking workflow on thermodynamic simulation-derived targets. The ranking outputs are useful for candidate prioritisation and documentation of a reproducible scientific ML workflow.

They do not support claims of wet-lab validation, biological proof, deployment readiness, or guaranteed thermoswitch behaviour. Full-library prediction and figure artefacts are generated after fitting an in-memory model on the same canonical dataset, so they should be treated as scoring artefacts rather than held-out performance estimates.

## Limitations

- Labels and regression targets are thermodynamic simulation-derived.
- The canonical processed dataset is small for biological sequence modelling.
- Sequence-similarity-aware splitting is not yet implemented.
- Only an expected random-ranking baseline is currently included; richer rule-based or model baselines remain future work.
- Model artefacts are not yet saved as production-style files.
- F1 openness plotting uses stored simulation curve values in the lightweight public workflow rather than fresh NUPACK recomputation.

## Reproducibility Commands

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests
& '.\.venv\Scripts\python.exe' scripts\evaluate_model.py
& '.\.venv\Scripts\python.exe' scripts\predict_candidates.py
& '.\.venv\Scripts\python.exe' scripts\make_shortlist.py
& '.\.venv\Scripts\python.exe' scripts\make_figures.py
```

## Interpretation Guardrails

- Labels are thermodynamic simulation-derived, not wet-lab validated.
- Metrics are from the canonical evaluation script.
- Results support candidate prioritisation under simulation assumptions.
- README claims should quote this report only with the same caveats.
