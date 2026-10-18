# Docker verification record — Lab 2

This document records the verification runs for the training script
and Dockerfile in this folder: one local run (Section 1) and the
GitHub Actions Docker runs (Section 2).

## 1. Local run (script, pinned dependencies)

Environment: fresh virtual environment, `pip install -r requirements.txt`
(`scikit-learn==1.5.2`, `joblib==1.4.2`, `numpy==1.26.4`), then
`python src/main.py --output-dir output`.

Result: exit code 0.

```text
Dataset: Breast Cancer Wisconsin - 569 samples, 30 features, 2 classes
Train/test split: 455 train, 114 test
Accuracy: 0.9561
F1 score: 0.9655
Model saved to: output/breast_cancer_rf.joblib
Metrics saved to: output/metrics.json
The model training was successful
```

Files produced: `breast_cancer_rf.joblib` (317 KB) and `metrics.json`
(copies of the metrics and the console log are in this `sample_output/`
folder as `metrics.json` and `run_log.txt`).

A second local run with different arguments
(`python src/main.py --n-estimators 10`) also exited 0
(accuracy 0.9386, F1 0.9510).

## 2. Docker run (GitHub Actions, clean runner)

Workflow: `.github/workflows/lab2-docker-build.yml` in this repo
(builds the image from this folder's `Dockerfile`, runs the
container, verifies the outputs).

First verified run — 2026-10-18:
https://github.com/SarrangTech/MLops_assignments/actions/runs/39162001130

- Status: **completed — success** (commit `28d04be`)
- Steps that passed: Build the Lab 2 Docker image; Run the container
  (default arguments, output volume mounted); Verify model and
  metrics came out of the container; Run again with different
  arguments (ENTRYPOINT check, no rebuild); Upload model and metrics
  as artifacts
- Artifact produced by the run: `lab2-docker-output`
  (the model and `metrics.json` from inside the Docker run —
  downloadable from the run page above)

Second verified run — 2026-10-18, after a Compose step was added
to the workflow:
https://github.com/SarrangTech/MLops_assignments/actions/runs/39162488482

- Status: **completed — success** (commit `0df3e08`)
- Every step passed, including the new step “Run via Docker Compose
  (README alternative path)”: `docker compose up --build` built and
  ran the same job, and the model + `metrics.json` were verified in
  the mounted `output/` folder afterwards.

Third run — 2026-10-18:
https://github.com/SarrangTech/MLops_assignments/actions/runs/39162762370

- Status: **completed — success** (commit `dbbaae3`)
- All steps passed, including the Docker Compose step.

Fourth run — 2026-10-18, after README corrections for native Linux
and Git Bash (folder creation before the run commands, and
troubleshooting entries):
https://github.com/SarrangTech/MLops_assignments/actions/runs/39166019969

- Status: **completed — success** (commit `2f980d7`)
- All steps passed, including the Docker Compose step.

Later runs of the same workflow are listed on the repo's Actions
tab: https://github.com/SarrangTech/MLops_assignments/actions
