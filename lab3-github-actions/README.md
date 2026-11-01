# Lab Assignment 3 — GitHub Actions (implementing Github_Lab1)

IE7374 / DADS 7305 (MLOps), Lab Assignment 3.

This submission implements the course lab **`Labs/Github_Labs/Lab1`** from
the course repo `SarrangTech/MLOps` — the GitHub Actions lab from Session 1.
That lab has five parts: create a virtual environment, create a GitHub
repository with a `data/` · `src/` · `test/` structure, write the Python
files (`src/calculator.py`), write test files in **both** pytest and
unittest (`test/test_pytest.py`, `test/test_unittest.py`), and implement
GitHub Actions workflows that run those tests. Each part is implemented
below, then extended — the extensions are listed separately so what is the
lab and what was changed stays visible.

The workflow lives at the repository root, where GitHub Actions requires
it: `.github/workflows/lab3-ci.yml`. Everything else is in this folder.

## How this submission maps to the lab

| Lab component (Labs/Github_Labs/Lab1) | This submission |
|---|---|
| Folder structure `data/`, `src/`, `test/` + `.gitignore` | Same structure in this folder |
| `src/calculator.py` — `fun1` add, `fun2` subtract, `fun3` multiply, `fun4` add three numbers | Same file, same four functions, same behaviour (formatting cleaned) |
| `test/test_pytest.py` — pytest suite for fun1–fun4 | Same suite, same cases, plus added tests (below) |
| `test/test_unittest.py` — unittest suite for fun1–fun4 | Same suite, same cases, plus added tests (below) |
| `workflows/github_lab1_pytest_action.yml` — pytest workflow, uploads the JUnit XML report | Merged and rebuilt as stage 1 of `.github/workflows/lab3-ci.yml`: pytest run with `--junitxml`, report uploaded as an artifact on every run |
| `workflows/github_lab2_unittest_action.yml` — unittest workflow | Same stage 1: a dedicated `python -m unittest discover` step |
| `requirements.txt` (`pytest`) | Pinned `requirements.txt` for the whole project (below) |

## Additions made for this submission

On top of the lab, in the lab's own terms first, then the CI rebuild:

1. **Two more calculator functions.** `fun5` (division, raises
   `ZeroDivisionError` on a zero denominator) and `fun6` (power), written
   in the lab's style with the same `ValueError` input validation as
   fun1–fun3.
2. **Error-case tests in both suites.** The lab's suites only test valid
   inputs. Both suites now also test the new functions and assert that
   non-numeric inputs raise `ValueError` and division by zero raises
   `ZeroDivisionError` — 8 pytest tests and 8 unittest tests for the
   calculator, all passing.
3. **An ML training pipeline as the thing CI operates on.** `data.py` /
   `train.py` train a `StandardScaler + RandomForestClassifier` pipeline
   on scikit-learn's digits dataset (built in, seeded split — identical
   data on every run), with its own pytest suite in `tests/` and a
   `verify_outputs.py` smoke check. This is what the workflow's second
   stage trains and publishes.

## The workflow

Two stages, each on Python 3.11 and 3.12:

1. **lint-and-test** — installs the pinned requirements, lints the lab
   folder with ruff (config in `ruff.toml`), runs the pytest suites
   (`test/` and `tests/`, 20 tests) with a JUnit XML report, runs the
   unittest suite (8 tests), and uploads the test report as the
   `lab3-test-results-py*` artifact — on failure too (`if: always()`).
2. **train-and-publish** — runs only if stage 1 passed (`needs:`). Trains
   the model, smoke-checks that the model, metrics, and log outputs exist
   and are non-empty, writes the metrics to the run's summary page, and
   uploads the outputs as the `lab3-model-py*` artifact.

Triggers: pushes and pull requests to `main`, manual
(`workflow_dispatch`), and a weekly schedule (Mondays 13:00 UTC).
One active run per branch — a newer push cancels the stale run.

## What a successful run looks like

Open the repository's **Actions** tab and select the `lab3-ci` workflow.
A successful run shows four green jobs — lint-and-test and
train-and-publish, each for Python 3.11 and Python 3.12 — and:

- a `lab3-test-results-py3.11` / `-py3.12` artifact (the JUnit XML test
  report);
- a run **Summary** containing the training metrics as JSON — accuracy
  **0.9639**, f1_macro **0.9634**, n_train 1437, n_test 360 (seeded split,
  so CI reproduces these numbers exactly);
- a `lab3-model-py3.11` / `-py3.12` artifact containing
  `models/model.joblib`, `reports/metrics.json`, and `logs/train.log`.

Copies of the metrics and log from a local run are in `sample_output/`.

## Local run

Python 3.11 or 3.12 and `pip`. The dataset is built into scikit-learn;
nothing is downloaded.

```bash
git clone https://github.com/SarrangTech/MLops_assignments.git
cd MLops_assignments
python -m venv .venv && source .venv/bin/activate   # the lab's Step 1
pip install -r lab3-github-actions/requirements.txt

# the lab's two test suites, the way CI runs them
python -m pytest lab3-github-actions/test lab3-github-actions/tests -q
python -m unittest discover -s lab3-github-actions/test -t lab3-github-actions

python lab3-github-actions/train.py           # writes models/, reports/, logs/
python lab3-github-actions/verify_outputs.py  # the CI smoke check, locally
```

Expected results: 20 pytest tests pass, 8 unittest tests pass, training
logs accuracy ≈ 0.964, and `verify_outputs.py` prints "Smoke check
passed". Generated outputs are gitignored — CI publishes them as
artifacts instead of committing them.

## Changes vs. the original lab's workflows

The lab ships two separate single-job workflows (checkout v2,
setup-python v2, Python 3.8, no lint, no gating). The CI here was rebuilt
— that rebuild is the core change of this submission:

1. **One gated two-stage pipeline** instead of two independent
   workflows: the test stage must pass before training runs (`needs:`).
2. **Both of the lab's test frameworks run in CI** — pytest (with the
   JUnit report upload the lab's pytest workflow had) *and* a unittest
   step (the lab's second workflow), instead of either/or.
3. **Current action versions and Python versions:** checkout v4,
   setup-python v5, Python 3.11 + 3.12 matrix (`fail-fast: false`)
   instead of a single Python 3.8 job.
4. **Lint gate:** ruff checks the whole lab folder in stage 1
   (`ruff.toml` documents the one rule exception: the lab's calculator
   raises `ValueError` for non-numeric input by design, so TRY004 is
   ignored rather than changing the lab's behaviour).
5. **Output smoke check:** the training stage fails if any output is
   missing or empty, instead of uploading an incomplete artifact.
6. **Run summary:** training metrics are written to the Actions summary
   page, readable without downloading the artifact.
7. **Concurrency control, schedule, caching:** one active run per
   branch, a weekly scheduled run, and `pip` caching keyed to the lab's
   `requirements.txt`.

## Layout

| Path | What it is |
|---|---|
| `.github/workflows/lab3-ci.yml` (repo root) | The Lab 3 workflow |
| `lab3-github-actions/src/calculator.py` | The lab's calculator, plus fun5/fun6 |
| `lab3-github-actions/test/` | The lab's pytest + unittest suites, extended |
| `lab3-github-actions/data/` | The lab's data folder (see its README) |
| `lab3-github-actions/data.py`, `train.py` | Training pipeline (addition) |
| `lab3-github-actions/tests/` | pytest suite for the training pipeline |
| `lab3-github-actions/verify_outputs.py` | Local version of the workflow's smoke check |
| `lab3-github-actions/ruff.toml` | Lint configuration |
| `lab3-github-actions/requirements.txt` | Pinned dependencies |
| `lab3-github-actions/sample_output/` | Metrics + log from a verified local run |

## Verification

Verification records for this lab are listed below.

**Local run** — fresh virtual environment, dependencies installed
from the pinned `requirements.txt`:

- `ruff check lab3-github-actions` — all checks passed
- `python -m pytest lab3-github-actions/test lab3-github-actions/tests -q`
  — 20 passed
- `python -m unittest discover -s lab3-github-actions/test -t lab3-github-actions`
  — 8 tests, OK
- `python lab3-github-actions/train.py` — exit code 0, accuracy
  0.9639, F1 (macro) 0.9634; `verify_outputs.py` — "Smoke check
  passed" (metrics and log copies in `sample_output/`)

The local checks were also run in a clean `git clone` of this
repository before pushing, because lint import-sorting can differ
between a plain folder and a git checkout (see `ruff.toml`).

**CI runs (GitHub Actions)** — workflow `lab3-ci`:

| Run | Date | Commit | Result |
|---|---|---|---|
| [41363001190](https://github.com/SarrangTech/MLops_assignments/actions/runs/41363001190) | 2026-11-01 | `2b79bff` | Failed at the lint step (import sorting, I001) — fixed in the next commit |
| [41363846239](https://github.com/SarrangTech/MLops_assignments/actions/runs/41363846239) | 2026-11-01 | `39b06f3` | Success — all 4 jobs; artifacts `lab3-test-results-py3.11`/`-py3.12` and `lab3-model-py3.11`/`-py3.12` uploaded |

Later runs are listed on the repository's Actions tab.
