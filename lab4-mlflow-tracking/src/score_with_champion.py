"""Scoring stage: load the champion from the registry and score it.

Loads the model behind the `champion` alias of the registered model
- by registry URI, with no reference to local model files - and
scores it on the test split. Writes the metrics to
reports/champion_metrics.json and the per-row predictions to
reports/predictions.csv. This is the consumer side of the registry:
whatever the promotion gate last approved is what gets served.
"""

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from .data_pipeline import TARGET, feature_columns
from .tracking import (
    CHAMPION_ALIAS,
    REGISTERED_MODEL_NAME,
    eval_metrics,
    get_tracking_uri,
)

BASE_DIR = Path(__file__).resolve().parent.parent


def score_with_champion(
    base_dir: Path | None = None,
    tracking_uri: str | None = None,
) -> dict:
    """Score the current champion on data/test.csv. Returns metrics."""
    base = base_dir or BASE_DIR
    mlflow.set_tracking_uri(tracking_uri or get_tracking_uri(base))

    model = mlflow.sklearn.load_model(
        f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"
    )
    test = pd.read_csv(base / "data" / "test.csv")
    predictions = model.predict(test[feature_columns(test)])
    if pd.isna(predictions).any():
        raise ValueError("Champion produced NaN predictions on the test split.")

    metrics = eval_metrics(test[TARGET], predictions)
    metrics["n_test"] = len(test)
    metrics["model_uri"] = f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"

    (base / "reports").mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {"y_true": test[TARGET].to_numpy(), "y_pred": predictions.round(4)}
    )
    out.to_csv(base / "reports" / "predictions.csv", index=False)
    (base / "reports" / "champion_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(f"Champion metrics: {json.dumps(metrics)}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    score_with_champion()


if __name__ == "__main__":
    main()
