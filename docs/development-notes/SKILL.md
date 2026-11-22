# SKILL.md — RNA Thermoswitch ML Screening

## Project Identity

This project is **RNA Thermoswitch ML Screening**.

It is a portfolio-focused Machine Learning / Data Science project that builds a reproducible, simulation-guided pipeline for ranking RNA thermoswitch candidates using thermodynamic simulation features, calibrated classification, regression, and Top-K shortlist generation.

The intended public repository name is:

```text
rna-thermoswitch-ml-screening
```

The project should be positioned as:

> A physics-guided machine learning pipeline for ranking simulated RNA thermoswitch candidates from thermodynamic features.

This is not a wet-lab validated biotechnology product. It is a scientific ML and candidate-prioritisation workflow built from simulation-derived labels.

The target audience is UK recruiters and hiring managers for:

- Machine Learning Engineer roles
- Data Scientist roles
- Bioinformatics Data Scientist roles
- Scientific ML / Applied AI roles
- R&D analytics roles

---

## Core Portfolio Goal

The goal is to turn this project from a dissertation-style working folder into a polished, recruiter-facing GitHub portfolio project.

The final repo should demonstrate:

- applied machine learning
- scientific feature engineering
- responsible handling of simulation-derived labels
- imbalanced classification
- model calibration
- regression
- ranking metrics
- Top-K candidate selection
- reproducible scripts
- clean documentation
- honest limitations
- professional GitHub structure

The project should feel like an ML engineering project, not just an academic hand-in folder.

---

## Required Tone and Honesty Rules

Always describe the labels as:

- simulation-derived labels
- thermodynamic simulation-derived targets
- physics-guided screening labels
- candidate-prioritisation labels

Do **not** describe the candidates as:

- wet-lab validated
- experimentally proven
- biologically confirmed
- production-ready genetic control devices
- guaranteed thermoswitches

Preferred wording:

> The model is trained on thermodynamic simulation-derived targets, so results should be interpreted as candidate prioritisation under simulation assumptions rather than experimental validation.

Use careful language such as:

- candidate prioritisation
- shortlist generation
- simulation-guided ranking
- under thermodynamic model assumptions
- designed for future experimental validation

Avoid overclaiming from one split or one metric.

Do not lead with the strongest historical metric unless it has been reproduced by the canonical evaluation script.

---

## Canonical Repository Structure

The public repository should follow this structure:

```text
rna-thermoswitch-ml-screening/
  README.md
  LICENSE
  .gitignore
  requirements.txt
  pyproject.toml

  data/
    README.md
    raw/
    interim/
    processed/
      library_combined.csv
      library_combined_sample.csv

  notebooks/
    01_thermoswitch_pipeline_exploration.ipynb
    02_model_evaluation_report.ipynb

  src/
    thermoswitch/
      __init__.py
      config.py
      data.py
      features.py
      labels.py
      models.py
      evaluate.py
      shortlist.py
      plotting.py

  scripts/
    generate_library.py
    train_model.py
    evaluate_model.py
    predict_candidates.py
    make_shortlist.py
    make_figures.py

  models/
    README.md

  reports/
    PROJECT_AUDIT.md
    PHASE_0_CANONICALISATION_PLAN.md
    PHASE_1_STRUCTURE_SETUP_REPORT.md
    model_evaluation.md
    results/
      ml_predictions.csv
      ml_topk_diverse_clean.csv
      ml_order_sheet.csv
    figures/
      f1_openness_best.png
      f2_pr_curve.png
      f3_precision_at_k.png
      f3_recall_at_k.png
      f4_calibration.png
      f5_top20_t50_hist.png
      f5_top20_slope_hist.png
      f6_gc_hist.png
      f6_len_hist.png
      s4_diversity_hist.png

  docs/
    methodology.md
    limitations.md
    data_dictionary.md
    model_card.md

  tests/
    test_features.py
    test_labels.py
    test_shortlist.py
```

Do not use spaces in folder names.

Do not use informal filenames.

Do not keep duplicate zip bundles in the public repository.

---

## Canonical Assets

### Canonical Processed Dataset

Use this as the Phase 1 canonical modelling dataset:

```text
data/processed/library_combined.csv
```

Original source:

```text
final data file/library_combined.csv
```

This file contains the 700-row processed design library and should be treated as the canonical public dataset for the first reproducible pipeline.

---

### Canonical Prediction Output

Use this as an example output artefact:

```text
reports/results/ml_predictions.csv
```

Important:

- This is an output artefact.
- It must not be used as a training input.
- It contains predictions and observed target columns together for reporting purposes.

---

### Canonical Shortlist Output

Use this as the final Top-K shortlist example:

```text
reports/results/ml_topk_diverse_clean.csv
```

---

### Canonical Order Sheet

Use this as the practical export example:

```text
reports/results/ml_order_sheet.csv
```

---

### Canonical Notebook

The current source notebook is:

```text
notebooks/01_thermoswitch_pipeline_exploration.ipynb
```

This notebook is not yet production-ready. It should be treated as the extraction source for reusable code.

Eventually, the project should have:

```text
notebooks/01_thermoswitch_pipeline_exploration.ipynb
notebooks/02_model_evaluation_report.ipynb
```

The notebook should not be the only place where core project logic exists.

---

## Files and Folders That Must Stay Out of Public GitHub

Do not commit these to the public repository:

```text
KiTTY/
KiTTY.zip
final data file/zip/
library_60nt_sweep_with_seq.csv
*.zip
*.exe
*.dll
*.cmd
*.bat
*.docx
*AutoRecovered*
*The Final Boss*
```

Do not publish academic drafts unless fully sanitised:

```text
Data Science Project_Md Abir Hossain.pdf
Data Science Project_Md Abir Hossain_29 August.pdf
final data file/Data Science Project_Md Abir Hossain.docx
final data file/AJS Comments - Data Science Project_Md Abir Hossain_plusAAcomments.docx
final data file/Updated Data Science Project.docx
Front_Matter_Title_and_Abstract (AutoRecovered).docx
```

Reasons:

- may contain student ID or academic metadata
- may contain supervisor or institution details
- may contain private comments
- may look unprofessional
- may include unrelated executable/tool files
- may expose session or machine details

---

## Data Handling Rules

The public repository should stay lightweight and recruiter-friendly.

### Keep in Git

Allowed public data:

```text
data/processed/library_combined.csv
data/processed/library_combined_sample.csv
reports/results/ml_predictions.csv
reports/results/ml_topk_diverse_clean.csv
reports/results/ml_order_sheet.csv
```

### Exclude from Normal Git

Large or generated files should not be committed directly:

```text
library_60nt_sweep_with_seq.csv
nupack_sweep_results.csv
thermoswitch_10k_features.csv
thermoswitch_10k_features.xlsx
Generated_1000_RNA_Sequences.csv
generated_10k_thermoswitches.csv
generated_10k_scaffold_thermoswitches.csv
top20_around70.csv
```

### Data Documentation Required

Maintain:

```text
data/README.md
docs/data_dictionary.md
```

These should explain:

- which dataset is canonical
- which columns are features
- which columns are targets
- which columns are metadata
- which files are output artefacts
- which files are excluded due to size
- that labels are simulation-derived
- that full raw sweeps are not included in normal GitHub

Never mix target columns into inference inputs.

---

## Modelling Rules

Use scikit-learn pipelines where possible.

The project should include:

- a calibrated classifier for pass probability
- regressors for T50 and slope at 70°C
- a stable feature schema
- deterministic random seeds
- saved model artefacts
- saved model metadata
- clear threshold-selection logic
- clear Top-K ranking logic

Future saved artefacts should include:

```text
models/thermoswitch_classifier.joblib
models/t50_regressor.joblib
models/slope70_regressor.joblib
models/feature_schema.json
models/model_metadata.json
```

Model binaries may stay ignored until intentionally added.

Use `joblib` for scikit-learn artefacts unless there is a strong reason not to.

Do not leave canonical training logic only inside notebooks.

---

## Required Source Modules

Reusable logic should live in:

```text
src/thermoswitch/
```

Recommended modules:

```text
src/thermoswitch/config.py
src/thermoswitch/data.py
src/thermoswitch/features.py
src/thermoswitch/labels.py
src/thermoswitch/models.py
src/thermoswitch/evaluate.py
src/thermoswitch/shortlist.py
src/thermoswitch/plotting.py
```

### Module Responsibilities

#### `config.py`

Responsible for:

- project paths
- random seed
- default thresholds
- constants

#### `data.py`

Responsible for:

- loading processed dataset
- validating required columns
- splitting metadata, features, and targets

#### `features.py`

Responsible for:

- sequence length
- GC content
- dinucleotide features
- trinucleotide features if retained
- RBS-window features if retained
- feature schema handling

#### `labels.py`

Responsible for:

- pass/fail target creation
- simulation-derived target rules
- threshold documentation

#### `models.py`

Responsible for:

- classifier and regressor construction
- scikit-learn pipeline creation
- calibration setup
- model saving/loading helpers

#### `evaluate.py`

Responsible for:

- PR-AUC
- ROC-AUC
- Precision@K
- Recall@K
- calibration metrics
- confusion matrix
- regression metrics
- baseline comparison

#### `shortlist.py`

Responsible for:

- Top-K ranking
- diversity filtering
- GC and homopolymer QC
- order sheet creation

#### `plotting.py`

Responsible for:

- PR curve
- ROC curve
- calibration plot
- Precision@K / Recall@K
- T50 and slope histograms
- GC and length plots

---

## Required Scripts

Scripts should be thin wrappers around `src/thermoswitch/`.

Recommended scripts:

```text
scripts/train_model.py
scripts/evaluate_model.py
scripts/predict_candidates.py
scripts/make_shortlist.py
scripts/make_figures.py
scripts/generate_library.py
```

### Expected Commands

The final README should eventually support commands like:

```bash
python scripts/train_model.py
python scripts/evaluate_model.py
python scripts/predict_candidates.py --input data/processed/library_combined.csv --output reports/results/ml_predictions.csv
python scripts/make_shortlist.py --input reports/results/ml_predictions.csv --output reports/results/ml_topk_diverse_clean.csv
python scripts/make_figures.py
```

Do not invent commands in the README until scripts actually exist.

---

## Canonical ML Workflow

The canonical workflow is:

1. Load `data/processed/library_combined.csv`
2. Validate required columns
3. Separate metadata, features, and targets
4. Generate sequence/design features
5. Create simulation-derived pass/fail labels
6. Use deterministic train/test split
7. Train calibrated classifier
8. Train T50 and slope regressors
9. Evaluate classification, ranking, calibration, and regression performance
10. Generate predictions
11. Generate Top-K shortlist
12. Apply diversity and QC filtering
13. Export order sheet
14. Generate figures
15. Update `reports/model_evaluation.md`

---

## Evaluation Rules

Official metrics must come from canonical scripts, not old notebook outputs.

### Classification and Ranking Metrics

Report:

- PR-AUC / Average Precision as the lead classification metric
- ROC-AUC as secondary
- Precision@K
- Recall@K
- confusion matrix at selected threshold
- classification report at selected threshold
- calibration curve or Brier score

Recommended K values:

```text
K = 10, 20, 50
```

### Regression Metrics

Report:

- T50 MAE
- T50 R2
- slope70 MAE
- slope70 R2

### Practical Shortlist Metrics

Report:

- number of candidates selected
- GC range
- length range
- homopolymer warnings
- diversity summary
- predicted T50 distribution
- predicted slope70 distribution

### Split Protocol

Phase 1:

- deterministic stratified train/test split
- 5-fold stratified cross-validation
- random or rule-based baseline

Later robustness phase:

- sequence similarity split
- group/family split if generation family metadata exists
- near-duplicate checks

If robust split metrics are lower, report them honestly.

---

## Baseline Rules

The project should include baselines.

Recommended baselines:

- random ranking baseline
- simple rule-based score baseline
- logistic regression baseline
- simple GC/length-only baseline if useful

Do not present model results without context.

Ranking metrics should be compared against random or rule-based selection where possible.

---

## Visualisation Rules

Figures should live in:

```text
reports/figures/
```

Use lowercase snake_case filenames.

Preferred README/report visuals:

```text
reports/figures/f1_openness_best.png
reports/figures/f2_pr_curve.png
reports/figures/f3_precision_at_k.png
reports/figures/f3_recall_at_k.png
reports/figures/f4_calibration.png
reports/figures/f5_top20_t50_hist.png
reports/figures/f5_top20_slope_hist.png
reports/figures/f6_gc_hist.png
reports/figures/f6_len_hist.png
reports/figures/s4_diversity_hist.png
```

Do not lead with ROC-AUC over PR-AUC for an imbalanced screening task.

Add or regenerate later:

- class balance plot
- target distribution plots
- baseline vs model Precision@K comparison
- pipeline diagram
- similarity/near-duplicate check visual

---

## Notebook Rules

Notebooks should be readable and recruiter-friendly.

Every public notebook must include markdown sections explaining:

- project goal
- data source
- feature families
- target creation
- model approach
- metrics
- results
- limitations
- next steps

The current notebook has code but little/no markdown. It must be cleaned before being treated as presentation-ready.

Notebook logic should be extracted into `src/thermoswitch/` and scripts.

Notebooks should consume reusable code, not define the whole pipeline.

---

## README Rules

The final `README.md` must include:

1. Project title
2. One-line summary
3. Problem statement
4. Why this matters
5. Dataset description
6. Clear simulation-derived label disclaimer
7. Methodology
8. Results from canonical scripts only
9. Key figures
10. How to run
11. Project structure
12. Limitations
13. Next steps
14. Skills demonstrated

Do not write performance claims until canonical scripts reproduce the official metrics.

Do not include raw student/dissertation-style wording.

Keep explanations accessible for recruiters who may not know RNA biology.

---

## Documentation Rules

Maintain these docs:

```text
docs/methodology.md
docs/limitations.md
docs/data_dictionary.md
docs/model_card.md
```

### `docs/methodology.md`

Should explain:

- RNA thermoswitch design problem
- simulation-guided workflow
- feature engineering
- label logic
- ML approach
- shortlist generation

### `docs/limitations.md`

Must include:

- simulation-derived labels
- no wet-lab validation
- small canonical processed dataset
- possible sequence similarity leakage
- domain assumptions
- need for future experimental validation

### `docs/data_dictionary.md`

Should define:

- every important dataset column
- whether it is metadata, feature, target, or output
- whether it is available at inference time

### `docs/model_card.md`

Should explain:

- model purpose
- inputs
- outputs
- intended use
- non-intended use
- training data
- evaluation protocol
- limitations

---

## Testing Rules

Add tests gradually under:

```text
tests/
```

Minimum useful tests:

```text
tests/test_features.py
tests/test_labels.py
tests/test_shortlist.py
```

Test:

- GC content calculation
- sequence length calculation
- k-mer feature extraction
- label creation thresholds
- no target leakage into inference features
- Top-K ranking order
- shortlist QC filters
- homopolymer detection
- required column validation

Tests should be simple and stable.

---

## Environment Rules

Add one reproducible environment file.

Preferred first step:

```text
requirements.txt
```

Later optional improvement:

```text
pyproject.toml
```

Expected packages may include:

```text
pandas
numpy
scikit-learn
matplotlib
joblib
pytest
jupyter
```

Add other dependencies only when actually used.

Do not add heavy or unused packages.

If NUPACK is required for regeneration, document it separately because installation may not be simple for every reviewer.

---

## GitHub Portfolio Rules

The final GitHub repo should be lightweight, clear, and impressive within two minutes.

Recommended GitHub description:

> Physics-guided ML pipeline for ranking RNA thermoswitch candidates from thermodynamic simulation features.

Recommended topics:

```text
machine-learning
data-science
bioinformatics
synthetic-biology
rna
scientific-computing
scikit-learn
model-calibration
ranking
candidate-screening
portfolio-project
```

Optional topics:

```text
nupack
molecular-design
imbalanced-classification
regression
```

The repo should not look like a random project folder.

The public repo should not include:

- irrelevant tools
- zip bundles
- private academic drafts
- giant raw data files
- informal filenames
- duplicate copies
- unexplained outputs

---

## Phase Roadmap

### Phase 1 — Structure Setup

Status: complete.

Completed:

- clean folder skeleton
- `.gitignore`
- canonical dataset copied
- canonical result files copied
- selected figures copied
- placeholder docs created
- package marker created

### Phase 2 — Reproducible Pipeline Extraction

Next focus:

- inspect the notebook
- identify canonical logic
- extract reusable code into `src/thermoswitch/`
- create thin scripts under `scripts/`
- avoid changing scientific logic until baseline reproduction works

Expected files:

```text
src/thermoswitch/data.py
src/thermoswitch/features.py
src/thermoswitch/labels.py
src/thermoswitch/models.py
src/thermoswitch/evaluate.py
src/thermoswitch/shortlist.py
scripts/train_model.py
scripts/evaluate_model.py
scripts/predict_candidates.py
scripts/make_shortlist.py
```

### Phase 3 — Evaluation and Figures

Focus:

- reproduce official metrics
- add baselines
- generate figures from scripts
- create `reports/model_evaluation.md`

### Phase 4 — README and Documentation

Focus:

- write recruiter-ready README
- complete methodology
- complete limitations
- complete data dictionary
- add model card

### Phase 5 — Demo and Polish

Optional but valuable:

- Streamlit demo
- FastAPI endpoint
- Dockerfile
- GitHub Actions tests
- final GitHub metadata

---

## Safety Rules for Future Codex Work

When using Codex:

- make one focused change per prompt
- ask Codex to save reports in `reports/`
- require markdown reports for audits and phases
- do not allow deletion unless explicitly approved
- do not allow model metric claims unless reproduced
- do not allow direct edits to private academic files
- do not allow large raw files into public Git
- require tests when adding reusable feature or label logic
- preserve original files unless migration is explicitly requested

For every phase, Codex should report:

1. what changed
2. which files were created
3. which files were modified
4. which files were not touched
5. risks or warnings
6. recommended next step

---

## Immediate Next Step

After creating this `SKILL.md`, begin Phase 2.

The next Codex task should audit the notebook and create a precise extraction plan before writing modules.

Recommended next report:

```text
reports/PHASE_2_NOTEBOOK_EXTRACTION_PLAN.md
```

Do not extract code until the extraction plan identifies the canonical workflow sections and maps them to target modules/scripts.
