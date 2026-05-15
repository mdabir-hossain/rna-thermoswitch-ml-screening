# Limitations

This project is intentionally framed as simulation-guided candidate prioritisation. The limitations below should travel with any README, portfolio description, or presentation that uses these outputs.

## Simulation-Derived Labels

The classification labels and regression targets are derived from thermodynamic simulation outputs, not wet-lab assays. They reflect the assumptions, thresholds, and approximations of the simulation workflow used to create the processed dataset.

Appropriate interpretation:

- "candidate prioritisation under thermodynamic simulation assumptions";
- "simulation-guided ranking";
- "shortlist for future experimental validation."

Inappropriate interpretation:

- "experimentally validated thermoswitches";
- "biologically proven candidates";
- "production-ready genetic control devices."

## No Wet-Lab Validation

No candidate in this repository should be described as experimentally validated. The shortlist is useful for deciding what might be worth testing next, not for claiming that a sequence works in cells or in vitro.

## Small Canonical Dataset

The canonical public dataset contains 700 processed rows. That is enough for a compact portfolio workflow, but it is small for biological sequence modelling. Reported metrics may be sensitive to the exact split, label balance, sequence similarity, and design generation process.

## Possible Sequence Similarity Leakage

The current evaluation uses deterministic stratified train/test splitting and 5-fold stratified cross-validation. It does not yet use a sequence-similarity-aware split or grouped split by generation family.

If similar sequences appear in both train and test folds, held-out metrics may overstate performance on genuinely novel design families. A future robustness phase should add near-duplicate checks and sequence-similarity-aware validation.

## Full-Library Scoring Is In-Sample

`scripts/predict_candidates.py` currently fits the Phase 2B model bundle in memory on the canonical 700-row dataset and then scores that same library to create:

```text
reports/results/ml_predictions.csv
```

That file is useful as a reproducible scoring artefact and for shortlist generation, but figures generated from it should not be presented as held-out performance. Held-out and cross-validation metrics should come from `reports/model_evaluation.md`.

## Baselines Are Limited

Phase 3B adds an expected random-ranking baseline for Top-K ranking context. This is useful, but it is not a complete baseline suite.

Future baseline work should include:

- simple rule-based ranking;
- logistic regression or another simple classifier baseline;
- GC/length-only baseline if useful;
- baseline-vs-model ranking figures.

Until those are implemented, avoid claims that the model broadly outperforms alternative approaches.

## No Saved Production Model Artefacts Yet

The current canonical scripts fit models in memory. The project does not yet publish saved model artefacts such as:

```text
models/thermoswitch_classifier.joblib
models/t50_regressor.joblib
models/slope70_regressor.joblib
models/feature_schema.json
models/model_metadata.json
```

This is acceptable for the current portfolio phase, but it is not a deployment workflow.

## No Sequence-Similarity Split Yet

The current split protocol is standard stratified splitting. It is not yet designed to answer "how well does the model generalise to structurally or sequence-distant candidates?"

Future evaluation should include sequence similarity groups or near-duplicate checks before making stronger generalisation claims.

## Lightweight Public Figure Caveat

The original exploratory notebook included NUPACK-oriented openness plotting logic. The public lightweight figure workflow uses stored simulation curve values from `data/processed/library_combined.csv` for `reports/figures/f1_openness_best.png`.

This keeps the repository easy to run, but it means the public F1 figure is not a fresh NUPACK recomputation.

## Domain Assumptions

The workflow assumes that the selected thermodynamic outputs and thresholds are useful proxies for candidate quality. Those assumptions may not capture experimental context, cellular environment, expression system effects, RNA degradation, off-target behaviour, manufacturability, or assay-specific constraints.

## Intended Interpretation

The safest project summary is:

> A physics-guided ML pipeline for ranking simulated RNA thermoswitch candidates from thermodynamic features.

The project demonstrates applied ML engineering, scientific feature extraction, imbalanced classification, calibration, regression, Top-K ranking, QC filtering, and honest documentation of simulation-derived labels. It does not demonstrate wet-lab validation.
