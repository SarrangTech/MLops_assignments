"""Train a RandomForest on the Wine dataset with production-style logging."""
import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from logging_config import get_logger, setup_logging

# Named loggers: one per pipeline stage, so every line says where it came from.
data_log = get_logger("data")
train_log = get_logger("train")
model_log = get_logger("model")


def load_data(data_path=None):
    """Load Wine from sklearn, or from a CSV (last column = label) if given."""
    try:
        if data_path:
            data_log.info("Loading dataset from CSV: %s", data_path)
            raw = np.genfromtxt(data_path, delimiter=",", skip_header=1)
            X, y = raw[:, :-1], raw[:, -1].astype(int)
        else:
            data_log.info("Loading built-in Wine dataset from scikit-learn")
            bunch = load_wine()
            X, y = bunch.data, bunch.target
            data_log.debug("Feature names: %s", list(bunch.feature_names))
            data_log.debug("Class names: %s", [str(n) for n in bunch.target_names])
        data_log.info("Dataset shape: X=%s, y=%s", X.shape, y.shape)
        labels, counts = np.unique(y, return_counts=True)
        data_log.info("Class distribution: %s", {int(k): int(v) for k, v in zip(labels, counts)})
        if np.isnan(X).any():
            data_log.warning("Dataset contains NaN values (%d)", int(np.isnan(X).sum()))
        return X, y
    except Exception:
        data_log.exception("Failed to load dataset")
        raise


def train_model(X, y, n_estimators, max_depth, test_size, seed):
    """Split, fit and evaluate the model; return (model, accuracy)."""
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        train_log.info(
            "Train/test split: %d train rows, %d test rows (test_size=%.2f, seed=%d)",
            len(X_train), len(X_test), test_size, seed,
        )
        params = {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": seed}
        train_log.info("Hyperparameters: %s", params)

        model = RandomForestClassifier(**params)
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_log.info("Training finished in %.3fs", time.perf_counter() - start)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        train_log.info("Test accuracy: %.4f", acc)
        train_log.debug("Classification report:\n%s", classification_report(y_test, preds))
        if acc < 0.9:
            train_log.warning("Accuracy %.4f is below the 0.90 target", acc)
        return model, acc
    except Exception:
        train_log.exception("Training failed")
        raise


def save_model(model, path):
    try:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out)
        model_log.info("Model saved to %s (%d bytes)", out.resolve(), out.stat().st_size)
        return out
    except Exception:
        model_log.exception("Failed to save model to %s", path)
        raise


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--console-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    p.add_argument("--file-level", default="DEBUG",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    p.add_argument("--log-file", default="logs/train.log")
    p.add_argument("--data-path", default=None, help="optional CSV (last column = label)")
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-path", default="models/wine_rf.joblib")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.console_level, args.file_level, args.log_file)
    train_log.info("=== Run started ===")
    try:
        X, y = load_data(args.data_path)
        model, acc = train_model(X, y, args.n_estimators, args.max_depth,
                                 args.test_size, args.seed)
        save_model(model, args.model_path)
    except Exception:
        train_log.critical("Run aborted; see traceback above and in %s", args.log_file)
        return 1
    train_log.info("=== Run finished successfully (accuracy=%.4f) ===", acc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
