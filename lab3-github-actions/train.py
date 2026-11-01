"""
Train and evaluate a digits classifier; save the model and its metrics.

The pipeline is small and fully self-contained: the dataset ships with
scikit-learn, the split is seeded, and every output lands next to this
file — models/model.joblib, reports/metrics.json, logs/train.log — so a
run behaves the same on a laptop and on a CI runner.

Run it with:  python train.py
"""
import json
import logging
from pathlib import Path

import joblib
from data import load_dataset, split_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "model.joblib"
METRICS_PATH = BASE / "reports" / "metrics.json"
LOG_PATH = BASE / "logs" / "train.log"

logger = logging.getLogger("lab3.train")


def train_and_evaluate(model_path=MODEL_PATH, metrics_path=METRICS_PATH):
    """Train on the seeded split, save model + metrics, return the metrics."""
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "model": "StandardScaler + RandomForestClassifier(n_estimators=200)",
        "dataset": "sklearn digits (1797 samples, 64 features, 10 classes)",
        "n_train": len(y_train),
        "n_test": len(y_test),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "f1_macro": round(
            float(f1_score(y_test, predictions, average="macro")), 4
        ),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger.info("starting training run")
    metrics = train_and_evaluate()
    for key, value in metrics.items():
        logger.info("%s: %s", key, value)
    logger.info("model saved to %s", MODEL_PATH)
    logger.info("metrics saved to %s", METRICS_PATH)


if __name__ == "__main__":
    main()
