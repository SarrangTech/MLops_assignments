# Lab Assignment 3 — GitHub Actions CI for an ML Training Pipeline

IE7374 / DADS 7305 (MLOps), Lab Assignment 3, based on the **GitHub Actions lab**
(Session 1) from the course repo `SarrangTech/MLOps`.

This lab is self-contained: the folder holds a small training pipeline written
for it (a digits classifier), its unit tests, and its pinned requirements. The
workflow that runs everything lives at the repository root, where GitHub
Actions requires it: `.github/workflows/lab3-ci.yml`.

## The pipeline

- `data.py` — loads the digits dataset (it ships with scikit-learn, so nothing
  is downloaded and every environment sees identical data) and makes a
  stratified, seeded train/test split.
- `train.py` — trains a `StandardScaler + RandomForestClassifier` pipeline,
  evaluates it on the held-out split, and writes three outputs next to itself:
  `models/model.joblib`, `reports/metrics.json`, `logs/train.log`.
- `tests/` — pytest tests for the data split (shape, sizes, stratification,
  reproducibility) and a smoke test that runs a full training pass into a
  temporary directory and checks the saved model and metrics.
- `verify_outputs.py` — a local version of the workflow's output smoke check.

## The workflow

Two stages, each on Python 3.11 and 3.12:

1. **lint-and-test** — installs the pinned requirements, lints the lab folder
   with ruff, and runs the unit tests.
2. **train-and-publish** — runs only if stage 1 passed (`needs:`). It trains
   the model, smoke-checks that the model, metrics, and log outputs exist and
   are non-empty, writes the metrics to the run's summary page, and uploads
   the outputs as a build artifact.

It triggers on pushes and pull requests to `main`, manually
(`workflow_dispatch`), and on a weekly schedule (Mondays 13:00 UTC).

## What a successful run looks like

Open the repository's **Actions** tab and select the `lab3-ci` workflow.
A successful run shows four green jobs — lint-and-test and train-and-publish,
each for Python 3.11 and Python 3.12 — and:

- A run **Summary** containing the training metrics as JSON. The local
  verification run (Python 3.12) produced: accuracy **0.9639**,
  f1_macro **0.9634**, n_train 1437, n_test 360. CI numbers are identical
  because the data and the split are seeded.
- A downloadable artifact named `lab3-model-py3.11` (and `-py3.12`)
  containing `models/model.joblib`, `reports/metrics.json`, and
  `logs/train.log`.

## Local run

Requirements: Python 3.11 or 3.12 and `pip`. Nothing else — the dataset is
built into scikit-learn.

```bash
git clone https://github.com/SarrangTech/MLops_assignments.git
cd MLops_assignments
pip install -r lab3-github-actions/requirements.txt

python -m pytest lab3-github-actions/tests -q   # 4 tests
python lab3-github-actions/train.py             # writes models/, reports/, logs/
python lab3-github-actions/verify_outputs.py    # the CI smoke check, locally
```

Expected results: all 4 tests pass; training logs its metrics to the console
(accuracy ≈ 0.964); `verify_outputs.py` lists the three outputs and prints
"Smoke check passed". Generated outputs are gitignored — CI publishes them
as artifacts instead of committing them.

## What was changed vs. the original lab

The course lab's workflow runs a single job on one Python version and
uploads whatever the run leaves behind. This version rebuilds the CI/CD
pipeline itself:

1. **Two-stage pipeline.** A lint-and-test job gates the training job
   (`needs:`), so a failing test stops the run before any training happens.
2. **Lint gate.** ruff checks the whole lab folder in CI (pinned version).
3. **Unit tests in CI.** The pytest suite runs in the pipeline; the original
   workflow had no test stage.
4. **Python version matrix.** Both stages run on 3.11 and 3.12 in parallel
   (`fail-fast: false`, so one version failing does not hide the other);
   artifact names are version-suffixed so the jobs do not collide.
5. **Output smoke check.** The training job fails if any of the model,
   metrics, or log outputs is missing or empty, instead of a green run
   uploading an incomplete artifact.
6. **Run summary.** The training metrics are written to the Actions run
   summary page, readable without downloading the artifact.
7. **Concurrency control and schedule.** One active run per branch (a newer
   push cancels the stale run), plus a weekly scheduled run; `pip` caching
   is keyed to the lab's `requirements.txt`.

## Layout

| Path | What it is |
|---|---|
| `.github/workflows/lab3-ci.yml` (repo root) | The Lab 3 workflow |
| `lab3-github-actions/data.py` | Dataset loading and seeded split |
| `lab3-github-actions/train.py` | Training, evaluation, output writing |
| `lab3-github-actions/tests/` | pytest suite (`test_data.py`, `test_train.py`) |
| `lab3-github-actions/verify_outputs.py` | Local version of the workflow's smoke check |
| `lab3-github-actions/requirements.txt` | Pinned dependencies |
| `lab3-github-actions/README.md` | This file |
