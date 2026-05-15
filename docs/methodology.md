# Methodology

## Project Goal

RNA Thermoswitch ML Screening is a simulation-guided machine learning workflow for ranking RNA thermoswitch candidates from thermodynamic features. The goal is not to prove that a sequence works in the lab. The goal is to create a reproducible candidate-prioritisation pipeline that turns simulated design outputs into a ranked, QC-filtered shortlist for future experimental follow-up.

## Screening Problem

An RNA thermoswitch is a sequence designed to change accessibility or structure as temperature changes. In this project, the practical screening question is:

> Which simulated candidate sequences look most promising under a thermodynamic model, especially around a target temperature near 70 C?

The project frames that question as a scientific ML task:

- create simulation-derived pass/fail labels from thermodynamic outputs;
- engineer sequence and design features available before target observation;
- train a calibrated classifier to estimate probability of meeting the simulation-derived specification;
- train regressors for T50 and slope at 70 C;
- rank candidates and create a practical shortlist.

## Canonical Dataset

The canonical public modelling dataset is:

```text
data/processed/library_combined.csv
```

It contains 700 processed candidate rows copied from the original working data into the public project structure. Large raw simulation sweeps are intentionally kept out of the lightweight public workflow.

The dataset includes:

- candidate metadata such as `id`, `seq`, `len`, `rbs_start`, and `rbs_end`;
- design inputs such as `extra_pairs`, `pGC`, and `loop_len`;
- thermodynamic simulation-derived target columns such as `closed_low`, `open_midhi`, `open_high`, `T50`, and `slope70`;
- supporting simulation-only columns such as `slope_max`, `reentrant_pen`, `score`, and `curve`.

Observed simulation target columns are never used as inference features.

## Feature Engineering

Feature extraction is implemented in:

```text
src/thermoswitch/features.py
```

The canonical feature vector uses only fields that are available before observing target outputs:

- sequence length;
- GC fraction calculated from the RNA sequence;
- design metadata: `extra_pairs`, `pGC`, and `loop_len`;
- dinucleotide frequency features;
- trinucleotide frequency features;
- fixed-length one-hot encoding around the RBS window.

The feature schema intentionally excludes simulation targets and generated prediction outputs. Tests check that target/output column names do not enter the model feature list.

## Simulation-Derived Labels

Label creation is implemented in:

```text
src/thermoswitch/labels.py
```

The binary pass/fail label is created from thermodynamic simulation-derived target columns. A candidate passes the current specification when it meets all configured threshold rules:

- sufficiently closed at low temperature;
- sufficiently open in the mid/high target range;
- sufficiently open at high temperature;
- simulated T50 within the target band;
- slope at 70 C above the configured minimum.

These labels are useful for ML screening under the simulation assumptions, but they are not wet-lab labels.

## Model Approach

Model construction is implemented in:

```text
src/thermoswitch/models.py
```

The workflow fits three model components:

- a calibrated GradientBoostingClassifier for estimated probability of passing the simulation-derived specification;
- a GradientBoostingRegressor for predicted T50;
- a RandomForestRegressor for predicted slope at 70 C.

The classifier uses scikit-learn calibration so that `p_pass` can be treated as a ranking score rather than only a hard class label. The regressors support practical ranking and guard filters around the target transition temperature and slope.

Models are currently fitted in memory by the canonical scripts. Saved production-style model artefacts are planned future work.

## Evaluation Workflow

Evaluation logic is implemented in:

```text
src/thermoswitch/evaluate.py
scripts/evaluate_model.py
```

The canonical evaluation report is:

```text
reports/model_evaluation.md
```

The current evaluation workflow includes:

- deterministic stratified train/test split;
- 5-fold stratified cross-validation when class counts allow it;
- Average Precision / PR-AUC as the lead classification metric;
- ROC-AUC as a secondary classification metric;
- Brier score for calibration context;
- Precision@K and Recall@K for K = 10, 20, and 50;
- confusion matrix at the selected threshold;
- T50 MAE/R2 and slope70 MAE/R2;
- expected random-ranking baseline values for Top-K ranking context.

Average Precision is prioritised over ROC-AUC because this is an imbalanced screening problem where ranking likely positives is more important than overall discrimination across all thresholds.

## Prediction Export

Prediction export is implemented in:

```text
scripts/predict_candidates.py
```

It writes:

```text
reports/results/ml_predictions.csv
```

This file includes candidate IDs, sequences, predicted pass probabilities, predicted T50 values, predicted slope70 values, and observed simulation target columns for reporting/reference.

Important: `ml_predictions.csv` is an output artefact. It must not be used as a training input.

## Ranking and Shortlist Generation

Shortlist logic is implemented in:

```text
src/thermoswitch/shortlist.py
scripts/make_shortlist.py
```

The ranking workflow:

1. ranks candidates by higher predicted pass probability;
2. breaks ties using closeness of predicted T50 to 70 C;
3. uses predicted slope70 as an additional ranking signal;
4. applies prediction guards for T50 and slope;
5. greedily diversifies candidates using pairwise 3-mer Jaccard distance;
6. applies sequence QC filters;
7. refills the clean shortlist when needed;
8. writes practical order-sheet rows.

Canonical outputs:

```text
reports/results/ml_topk_summary.csv
reports/results/ml_topk_diverse.csv
reports/results/ml_topk_diverse_clean.csv
reports/results/ml_order_sheet.csv
```

## QC Filtering

The current QC filters are lightweight and designed for a public portfolio workflow:

- GC fraction must stay within the configured range;
- homopolymer runs of length 7 or more are flagged/excluded;
- predicted T50 and slope guards are applied before final clean shortlist selection.

These checks are practical screening filters, not guarantees of biological performance.

## Figure Generation

Figure generation is implemented in:

```text
src/thermoswitch/plotting.py
scripts/make_figures.py
```

Canonical figures are written to:

```text
reports/figures/
```

The public figure set includes precision-recall, calibration, Top-K ranking summaries, predicted shortlist distributions, library GC/length distributions, diversity histograms, and the simulated openness profile for the top clean shortlisted candidate.

For the lightweight public workflow, the F1 openness plot uses stored simulation curve values from the processed dataset rather than requiring fresh NUPACK recomputation.

## Reproducibility Workflow

Recommended command sequence:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests
& '.\.venv\Scripts\python.exe' scripts\evaluate_model.py
& '.\.venv\Scripts\python.exe' scripts\predict_candidates.py
& '.\.venv\Scripts\python.exe' scripts\make_shortlist.py
& '.\.venv\Scripts\python.exe' scripts\make_figures.py
```

The project should be interpreted as a reproducible scientific ML and candidate-prioritisation workflow built from thermodynamic simulation-derived targets.
