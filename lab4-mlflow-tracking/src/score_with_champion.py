"""Scoring stage: load the champion from the registry and score it.

Loads the model behind the `champion` alias of the registered model
- by registry URI, with no reference to local model files - and
scores it on the test split. This is the one place the test split
is used for a reported number: the champion was selected on
validation metrics, so its test score is an unbiased estimate of
how it does on unseen data.

Writes the metrics to reports/champion_metrics.json - the measured
test metrics, the tracked validation RMSE the champion was selected
on, and the test RMSE its run recorded during the experiment stage,
so the three can be cross-checked - and the per-row predictions to
reports/predictions.csv. This is the consumer side of the registry:
whatever the promotion gate last approved is what gets served.
"""

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from .data_pipeline import TARGET, feature_columns
from .tracking import (
    CHAMPION_ALIAS,
    REGISTERED_MODEL_NAME,
    eval_metrics,
    get_tracking_uri,
)

BASE_DIR = Path(__file__).resolve().parent.parent


def champion_model_uri() -> str:
    """The registry URI every consumer loads the champion by."""
    return f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"


def score_with_champion(
    base_dir: Path | None = None,
    tracking_uri: str | None = None,
) -> dict:
    """Score the current champion on data/test.csv. Returns metrics."""
    base = base_dir or BASE_DIR
    mlflow.set_tracking_uri(tracking_uri or get_tracking_uri(base))
    client = MlflowClient()
    version = client.get_model_version_by_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS
    )
    tracked = client.get_run(version.run_id).data.metrics

    model = mlflow.sklearn.load_model(champion_model_uri())
    test = pd.read_csv(base / "data" / "test.csv")
    predictions = model.predict(test[feature_columns(test)])
    if pd.isna(predictions).any():
        raise ValueError("Champion produced NaN predictions on the test split.")

    measured = eval_metrics(test[TARGET], predictions)
    metrics = {
        "test_rmse": measured["rmse"],
        "test_mae": measured["mae"],
        "test_r2": measured["r2"],
        "n_test": len(test),
        "model_uri": champion_model_uri(),
        "model_version": version.version,
        "selection_val_rmse": float(tracked["val_rmse"]),
        "tracked_test_rmse": float(tracked["test_rmse"]),
    }

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
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()
    score_with_champion(
        base_dir=Path(args.base_dir) if args.base_dir else None,
        tracking_uri=args.tracking_uri,
    )


if __name__ == "__main__":
    main()
