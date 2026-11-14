# Lab Assignment 4 - MLflow: Experiment Tracking and Model Registry

**Lab implemented:** the MLflow experiment-tracking lab
(`Labs/Experiment_Tracking_Labs/Mlflow_Labs/Lab1` in the course repo):
tracking experiments and runs, logging parameters, metrics and model
artifacts, autologging, model signatures, the model registry, and
loading a model back for predictions.

**Project:** an original experiment-tracking and model-registry
system for a regression problem - predicting median house value on
the California housing dataset (20,640 census block groups, 8
features). A designed grid of twelve model configurations is tracked
as MLflow runs, the best run is registered as a versioned model, a
promotion gate decides whether it takes over the `champion` alias,
and a scoring stage loads the champion *from the registry by alias*
and scores it. The full lifecycle runs locally by hand and in CI on
every push.

## How the lab's modules map to this submission

| Course lab module (Mlflow_Lab1) | Where it is implemented here |
|---|---|
| Experiments and runs (course notes: an experiment groups runs; a run records params, metrics, artifacts, source) | One experiment, `california-housing-experiments`, holding 13 runs; every run records its hyperparameters, held-out metrics, tags, and the fitted model as an artifact (`src/run_experiments.py`) |
| Tracking basics: tracking URI, `start_run`, `log_param` / `log_metric` | All runs use explicit `log_params` / `log_metrics`; the tracking URI is set in one shared place (`src/tracking.py`) |
| Storage mechanisms (course notes: local `mlruns/` file store vs a backend store / tracking server) | This project uses a **SQLite backend store** (`mlflow.db`) instead of the default file store - see "What differs from the lab demo" for why |
| Autologging (`autologging-intro`) | One run repeats a RandomForest configuration with `mlflow.sklearn.autolog()`, tagged `logging_mode=autolog`, next to the twelve manually logged runs |
| Comparing runs with different hyperparameters (the demo's alpha / l1_ratio runs) | A designed grid: 12 configurations across Ridge, ElasticNet, RandomForest, GradientBoosting, and XGBoost, exported through the search API to `reports/runs_table.csv` |
| Logging a model with a signature (`infer_signature`, input example) | Every manually logged run stores its model with an inferred signature and an input example |
| Model registry (`registered_model_name`; the demo notes the registry does not work with a file store) | `src/register_and_promote.py` registers the best run as a new version of `CaliforniaHousingModel` in the registry that the backend store makes available |
| Storing and loading a model (starter notebook: "Store ML models", "Load a model") | `src/score_with_champion.py` loads `models:/CaliforniaHousingModel@champion` with `mlflow.sklearn.load_model` and scores the test split |
| Serving concerns: the logged model's pip requirements (serving lab) | Each logged model carries its generated `requirements.txt` artifact in the run; the registry-served model is the same artifact the scoring stage loads |
| Viewing results (the demo opens the MLflow UI) | `mlflow ui --backend-store-uri sqlite:///mlflow.db` works locally after a run; CI instead exports the runs table and writes a summary to the Actions run page |

## What differs from the lab demo, and why

- **Backend store instead of the file store.** The demo's
  `linear_regression.py` registers the model only `if
  tracking_url_type_store != "file"` - with the default store the
  registry path silently never runs. Here the SQLite backend store
  makes registration, versioning, and aliases work end to end.
- **A designed experiment, not single runs.** The demo trains one
  ElasticNet per invocation and compares informally. Here the grid,
  the metrics, and the exported runs table make the comparison the
  deliverable.
- **Registration is gated, not automatic.** The demo registers on
  log (when it can). Here the best run is registered as a new
  version, and the `champion` alias moves only if the candidate
  beats the current champion's RMSE by a 1% relative margin;
  otherwise the decision is `hold`. The decision and both RMSE
  values are recorded in `reports/promotion_report.json`.
- **A consumer that trusts the alias.** Scoring never opens a local
  model file; it loads by registry alias, so whatever passed the
  gate is what gets served - and `verify_outputs.py` checks the
  served model's measured RMSE against the tracked RMSE of the run
  behind the alias.
- **A different problem and dataset** (California housing
  regression), with a seeded, saved train/test split that every
  stage shares.

## Layout

| Path | Purpose |
|---|---|
| `src/data_pipeline.py` | Fetches California housing, writes the seeded 80/20 splits to `data/train.csv` / `data/test.csv` |
| `src/tracking.py` | Shared configuration: tracking URI (SQLite backend store), experiment and model names, metric computation |
| `src/run_experiments.py` | The 12-configuration grid + 1 autologging run; exports `reports/runs_table.csv` |
| `src/register_and_promote.py` | Registers the best run; promotion gate on the `champion` alias; writes `reports/promotion_report.json` |
| `src/score_with_champion.py` | Loads the champion by alias, scores the test split; writes `reports/champion_metrics.json` and `reports/predictions.csv` |
| `tests/test_tracking.py` | Pytest suite on temporary tracking stores: tracking contract, gate promote/hold behaviour, scoring round-trip |
| `verify_outputs.py` | Post-run smoke check used locally and as a CI step |
| `summarize_run.py` | Renders the experiment + registry results as Markdown (CI run summary) |
| `requirements.txt` | Pinned dependencies (MLflow, scikit-learn, XGBoost, pandas, pytest) |
| `data/README.md` | What the generated split files are |
| `sample_output/` | One verified local run: `runs_table.csv`, `promotion_report.json`, `champion_metrics.json`, `experiments.log` |

## Requirements

- Python 3.11 or 3.12 (CI tests both)
- Dependencies are pinned in `requirements.txt`
- No accounts or external services: MLflow tracks to a local SQLite
  file; the dataset downloads once from scikit-learn's cache source

## Run it locally

```bash
git clone https://github.com/SarrangTech/MLops_assignments.git
cd MLops_assignments
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r lab4-mlflow-tracking/requirements.txt
cd lab4-mlflow-tracking

python -m pytest tests -q
# 3 passed

python -m src.run_experiments
# 13 runs tracked (12 manual grid + 1 autolog); writes reports/runs_table.csv

python -m src.register_and_promote
# registers the best run as a new model version; prints the gate decision

python -m src.score_with_champion
# loads models:/CaliforniaHousingModel@champion and scores the test split

python verify_outputs.py
python summarize_run.py            # the summary CI puts on the run page

mlflow ui --backend-store-uri sqlite:///mlflow.db
# optional: browse the experiment and the registry in the MLflow UI
```

## What a successful run looks like

Locally: pytest reports `3 passed`; the experiment stage prints one
line per configuration with its test RMSE and exports a 13-row runs
table; the registry stage prints a promotion report (decision,
candidate and champion RMSE, margin); scoring prints the champion's
test metrics; `verify_outputs.py` prints `Smoke check passed`.

In CI (Actions tab - **Lab 4 CI**): four jobs - `lint-and-test` and
`experiments-and-registry`, each on Python 3.11 and 3.12 - all
green. The run summary shows the best run, the gate decision, and
the champion metrics; the `lab4-experiment-py<version>` artifact
holds the runs table, the promotion report, the champion metrics,
and the predictions.

## Verification

| Check | Date | Revision | Result |
|---|---|---|---|
| Local run of every command above in a clean venv built from `requirements.txt` (Python 3.12) | 2026-11-14 | working tree | Lint clean; pytest 3 passed; 13 runs tracked; best run XGBoost (max_depth 6) at test RMSE 0.4518; gate decision `promote`, champion version 1; champion scored by alias at RMSE 0.4518; smoke check passed. Outputs in `sample_output/` |
