# Wine Classifier Training with Production-Style Logging

IE7374 (MLOps), Lab Assignment 1, based on the **Logging Lab** from
[SarrangTech/MLOps](https://github.com/SarrangTech/MLOps/tree/main/Labs/Experiment_Tracking_Labs/Logging_Labs).

## What this lab teaches

Python's built-in `logging` module is the foundation of experiment tracking and
debugging in MLOps. The original lab covers: log levels, `basicConfig`, named
loggers, `logger.exception`, file logging, runtime level control, and handlers.
This project applies **every one of those concepts** to a realistic ML workflow:
training and saving a scikit-learn classifier.

| Concept (original lab) | Where it appears here |
|---|---|
| Named loggers instead of root | `mlops.data`, `mlops.train`, `mlops.model` in `train.py` |
| Multiple handlers | Console (INFO) + rotating file (DEBUG) in `logging_config.py` |
| Log levels / runtime control | `--console-level`, `--file-level` CLI flags |
| Timestamped format | `%(asctime)s \| %(levelname)s \| %(name)s \| %(message)s` |
| `logger.exception` | Wraps data loading, training, and model saving |
| Key ML events | Dataset shape, class balance, split sizes, hyperparameters, accuracy, model path |

## How to run

```bash
pip install -r requirements.txt
python train.py
```

Outputs: `logs/train.log` (full DEBUG log) and `models/wine_rf.joblib`.

Try the options:

```bash
python train.py --console-level DEBUG            # verbose console
python train.py --console-level WARNING          # quiet console, file still DEBUG
python train.py --n-estimators 200 --max-depth 8 # change hyperparameters (logged)
python train.py --data-path missing.csv          # demo: exception logging, exit code 1
python train.py --n-estimators 0                 # demo: invalid hyperparameter
```

A real example of the output is in [`sample_logs/train_sample.log`](sample_logs/train_sample.log).

## What I changed vs. the original lab

| Original lab | This project |
|---|---|
| Single Jupyter notebook (`Starter.ipynb`) | Runnable Python project: `train.py`, `logging_config.py`, `requirements.txt` |
| Toy examples (`10 / 0`, dummy messages) | Real ML pipeline: load data, split, train RandomForest, evaluate, save model |
| No dataset | **Wine dataset** (178 samples, 13 features, 3 classes) from scikit-learn, plus optional CSV input |
| `basicConfig` on the root logger, ad-hoc handlers | Reusable `setup_logging()` that attaches handlers once to a parent `mlops` logger; child loggers propagate to it |
| Same level everywhere | Different levels per handler: console INFO, file DEBUG |
| Plain `FileHandler` | `RotatingFileHandler` (1 MB, 3 backups), so logs can't grow unbounded |
| `logger.exception` on a contrived division by zero | `logger.exception` around real failure points, re-raised and turned into a CRITICAL line and non-zero exit code |
| Level set in code | Levels, log path, and hyperparameters controlled from the command line |
| Not applicable | Idempotent setup (no duplicate lines if called twice) and warnings for NaNs / low accuracy |

## Project layout

```
train.py             # pipeline: load_data -> train_model -> save_model
logging_config.py    # setup_logging() and get_logger()
requirements.txt     # pinned dependencies
sample_logs/         # example log from a real run
```
