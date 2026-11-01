# Lab Assignment 3 — GitHub Actions CI for an ML Training Pipeline

IE7374 / DADS 7305 (MLOps), Lab Assignment 3, based on the **GitHub Actions lab**
(Session 1) from the course repo `SarrangTech/MLOps`.

The workflow file itself lives at the repository root, where GitHub Actions
requires it: `.github/workflows/lab3-ci.yml`. This folder documents it.
It runs the Lab 1 pipeline (`lab1-logging/train.py` — Wine dataset,
RandomForest) end-to-end on a clean runner on every push and pull request
to `main`.

## What a successful run looks like

Open the repository's **Actions** tab and select the `lab3-ci` workflow.
A successful run shows, for **both** Python 3.11 and Python 3.12:

- All steps green: Check out repo → Set up Python → Install dependencies →
  Run training pipeline → Smoke check model and log outputs →
  Write run summary → Upload model and logs.
- A run **Summary** page listing the model and log files the run produced.
- A downloadable artifact named `wine-model-and-logs-py3.11` (and `-py3.12`)
  containing `lab1-logging/models/wine_rf.joblib` and the run's log file
  under `lab1-logging/logs/`.

No local installation is needed to check the CI behaviour — the runner
provisions everything. To run the underlying pipeline locally instead:

## Local run (optional)

Requirements: Python 3.11 or 3.12, `pip`.

```bash
git clone https://github.com/SarrangTech/MLops_assignments.git
cd MLops_assignments/lab1-logging
pip install -r requirements.txt
python train.py
```

Expected output: training log lines on the console (INFO level), and two
outputs on disk — `models/wine_rf.joblib` and `logs/train.log`.

## What was changed vs. the original lab

The course lab's workflow runs one job on one Python version and uploads
whatever the run leaves behind. This version changes the CI/CD pipeline
itself in five ways:

1. **Output smoke check.** A dedicated step fails the build if training
   produced no model file or no log file, instead of a green run uploading
   an empty artifact.
2. **Python version matrix.** The job runs on Python 3.11 and 3.12 in
   parallel (`fail-fast: false`, so one version failing does not hide the
   other). Artifact names are version-suffixed so the two jobs do not
   collide.
3. **Weekly scheduled run.** A Monday 13:00 UTC cron trigger exercises the
   pipeline in weeks with no pushes.
4. **Concurrency control.** One active run per branch; a newer push cancels
   the stale run instead of both finishing and racing on artifacts.
5. **Run summary.** A step writes the produced model and log file names to
   the Actions run summary page, so the outputs can be checked without
   downloading the artifact.

Supporting details: `pip` caching keyed to `lab1-logging/requirements.txt`,
and artifacts upload with `if: always()` so logs are available even when a
step fails.

## Layout

| Path | What it is |
|---|---|
| `.github/workflows/lab3-ci.yml` (repo root) | The Lab 3 workflow |
| `lab3-github-actions/README.md` (this file) | Lab 3 documentation |
| `lab1-logging/train.py` | The pipeline the workflow runs |
| `lab1-logging/requirements.txt` | Pinned dependencies the workflow installs |
