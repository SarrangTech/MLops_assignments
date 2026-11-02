"""Evaluation stage: score the versioned model on the held-out test split.

Loads ``models/model_<timestamp>.joblib`` and ``data/test.csv`` (the
exact split the data pipeline saved - evaluation never regenerates or
reshuffles data), computes RMSE / MAE / R2, writes the per-row
residuals to ``reports/residuals.csv`` for inspection, and writes the
metrics to ``metrics/<timestamp>_metrics.json``.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data_pipeline import TARGET, feature_columns

BASE_DIR = Path(__file__).resolve().parent.parent


def run_evaluation(
    base_dir: Path | None = None, timestamp: str = "local"
) -> dict:
    """Evaluate the versioned model on the test split. Returns metrics."""
    base = base_dir or BASE_DIR
    model_path = base / "models" / f"model_{timestamp}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No versioned model at {model_path} - run the training "
            "stage first with the same --timestamp."
        )
    model = joblib.load(model_path)

    test = pd.read_csv(base / "data" / "test.csv")
    features = feature_columns(test)
    y_true = test[TARGET].to_numpy()
    y_pred = model.predict(test[features])
    if np.isnan(y_pred).any():
        raise ValueError("Model produced NaN predictions on the test split.")

    metrics = {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "n_test": len(test),
        "model_file": model_path.name,
    }

    (base / "reports").mkdir(parents=True, exist_ok=True)
    residuals = pd.DataFrame(
        {"y_true": y_true, "y_pred": np.round(y_pred, 3)}
    )
    residuals["residual"] = np.round(residuals["y_true"] - residuals["y_pred"], 3)
    residuals.to_csv(base / "reports" / "residuals.csv", index=False)

    (base / "metrics").mkdir(parents=True, exist_ok=True)
    out = base / "metrics" / f"{timestamp}_metrics.json"
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Wrote {out}: {json.dumps(metrics)}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timestamp",
        default="local",
        help="Version tag of the model to evaluate (must match training).",
    )
    args = parser.parse_args()
    run_evaluation(timestamp=args.timestamp)


if __name__ == "__main__":
    main()
