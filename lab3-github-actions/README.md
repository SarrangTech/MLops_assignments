# Lab Assignment 3 — GitHub Actions: CI/CD for an ML Training Pipeline

**Lab implemented:** the GitHub Actions labs (`Labs/Github_Labs/` in the
course repo) — automating tests, training, and evaluation with GitHub
Actions workflows.

**Project:** an original machine-learning pipeline built for this lab —
a regression model that predicts **diabetes progression one year after
baseline** (scikit-learn's diabetes dataset: 442 patients, 10 baseline
features). The pipeline compares three model families by cross-validation,
tunes the winner, tracks the run in MLflow, versions the model by run
timestamp, and evaluates it on a held-out test split — locally by hand, and
automatically in CI on every push.

## How the lab's concepts map to this submission

| GitHub Actions lab concept | Where it is implemented here |
|---|---|
| Workflow triggered on push / pull request | `.github/workflows/lab3-ci.yml` — `on: push`, `pull_request`, plus `workflow_dispatch` and a weekly `schedule` |
| Jobs and steps | Two jobs per Python version: `lint-and-test`, then `train-and-evaluate` |
| Job gating | Stage 2 runs only if stage 1 passes (`needs: lint-and-test`) |
| Matrix builds | Python 3.11 and 3.12, `fail-fast: false` |
| Dependency caching | `actions/setup-python` pip cache keyed to `requirements.txt` |
| Test reporting | pytest writes a JUnit XML report, uploaded as the `lab3-test-results-py<version>` artifact (`if: always()`) |
| Model training in CI | `python -m src.train --timestamp <run timestamp>` (comparison + tuning + MLflow tracking) |
| Model versioning | The trained model is saved as `models/model_<timestamp>.joblib`, the metrics as `metrics/<timestamp>_metrics.json` |
| Model evaluation in CI | `python -m src.evaluate --timestamp <run timestamp>` on the held-out test split |
| Run artifacts | `lab3-model-py<version>` artifact: the versioned model, metrics, CV report, and residuals |
| Run summary | `summarize_run.py` writes the candidate comparison and the test metrics to the Actions run summary |
| Concurrency control | One run per ref; a new push cancels the in-progress run |
| Least-privilege token | `permissions: contents: read` |

## The ML pipeline

1. **Data** (`src/data_pipeline.py`) — loads the diabetes dataset, splits
   it 80/20 (seed 42, stratified by target quintile), and writes
   `data/train.csv` (353 rows) and `data/test.csv` (89 rows). Feature
   scaling is *not* done here: each model is a `Pipeline` whose
   `StandardScaler` is fitted inside cross-validation, so nothing leaks
   from validation or test rows.
2. **Training** (`src/train.py`) — compares Ridge, RandomForest, and
   GradientBoosting by 5-fold cross-validated RMSE on the training split,
   tunes the winner with `GridSearchCV`, refits it, logs parameters and
   metrics to MLflow (`mlruns/`, experiment `diabetes-progression`), and
   saves the model under the run timestamp. The full comparison is
   written to `reports/cv_results.json`.
3. **Evaluation** (`src/evaluate.py`) — loads the versioned model and
   scores it on the exact saved test split (evaluation never regenerates
   data): RMSE / MAE / R² to `metrics/<timestamp>_metrics.json`, per-row
   residuals to `reports/residuals.csv`.
4. **Smoke check** (`verify_outputs.py`) — fails the build unless the
   model, metrics, CV report, and residuals all exist and agree with
   each other (model filename, test-row count, best-model consistency).

Local results (seed 42): Ridge won the comparison (CV RMSE 54.95 vs
RandomForest 57.78 and GradientBoosting 59.73), tuned to `alpha=10.0`;
held-out test RMSE 52.29, MAE 42.07, R² 0.546. On this dataset the tuned
linear model genuinely beats both ensembles — the comparison table in
`reports/cv_results.json` is the evidence, and CI reproduces it.

## Layout

| Path | Purpose |
|---|---|
| `src/data_pipeline.py` | Dataset loading, seeded stratified split, CSV splits |
| `src/train.py` | Model comparison (CV), GridSearchCV tuning, MLflow tracking, versioned model |
| `src/evaluate.py` | Held-out evaluation: metrics JSON + residuals CSV |
| `tests/test_pipeline.py` | Pytest suite: split contract, split reproducibility, train→evaluate round-trip, missing-model error |
| `verify_outputs.py` | Post-run smoke check used locally and as a CI step |
| `summarize_run.py` | Renders the CV comparison + test metrics as Markdown (CI run summary) |
| `requirements.txt` | Pinned dependencies (scikit-learn, pandas, MLflow, pytest, …) |
| `ruff.toml` | Lint configuration for this folder |
| `data/README.md` | What the generated split files are |
| `sample_output/` | One verified local run: `metrics.json`, `cv_results.json`, `train.log` |

## Requirements

- Python 3.11 or 3.12 (CI tests both)
- Dependencies are pinned in `requirements.txt` (`pip` installs them)
- No accounts, keys, or external services: the dataset ships with
  scikit-learn and MLflow tracks to a local `mlruns/` folder

## Run it locally

```bash
git clone https://github.com/SarrangTech/MLops_assignments.git
cd MLops_assignments
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r lab3-github-actions/requirements.txt
cd lab3-github-actions

python -m pytest tests -q
# 4 passed

python -m src.train --timestamp local
# compares the 3 candidates by CV RMSE, tunes the winner, logs to
# MLflow, saves models/model_local.joblib + reports/cv_results.json

python -m src.evaluate --timestamp local
# writes metrics/local_metrics.json + reports/residuals.csv

python verify_outputs.py --timestamp local
python summarize_run.py --timestamp local   # the table CI puts in the run summary
```

## What a successful run looks like

Locally: pytest reports `4 passed`; training prints the three CV RMSE
lines, the best candidate, and its tuned parameters; evaluation writes
the metrics JSON (for seed 42: RMSE 52.29, MAE 42.07, R² 0.546,
n_test 89); `verify_outputs.py` prints `Smoke check passed`.

In CI (Actions tab → **Lab 3 CI**): four jobs — `lint-and-test` and
`train-and-evaluate`, each on Python 3.11 and 3.12 — all green. The run
summary shows the candidate-comparison table and the held-out test
metrics; the artifacts hold the JUnit test report and the versioned
model with its metrics and reports.

## Verification

| Check | Date | Revision | Result |
|---|---|---|---|
| Local run of every command above (Python 3.12 venv, clean folder) | 2026-11-01 | — | Ruff clean; pytest 4 passed; train → evaluate → verify all pass; metrics reproduced exactly (RMSE 52.2927, R² 0.5464) |
| CI runs on `main` | 2026-11-01 | see Actions tab | Runs during the folder restructure failed while the old workflow and the new code were out of sync; the run on the final code and workflow is the one that counts — [Lab 3 CI runs](https://github.com/SarrangTech/MLops_assignments/actions/workflows/lab3-ci.yml) |
