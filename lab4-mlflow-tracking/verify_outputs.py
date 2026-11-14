"""Smoke-check the outputs of a full pipeline run.

Run after the experiment, registry, and scoring stages (CI does this
before uploading artifacts). Fails with a non-zero exit code unless:
the runs table covers the full experiment grid plus the autolog run;
the promotion report is well-formed; the champion metrics were
computed on every test row; the predictions file has one row per
test row; and the RMSE the scoring stage measured for the champion
matches the RMSE recorded in MLflow for the run behind the champion
alias - i.e. the model that was served is the model that was gated.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

EXPECTED_RUNS = 13  # 12 grid configurations + 1 autologging run


def fail(message: str) -> int:
    print(f"FAIL: {message}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=EXPECTED_RUNS,
        help="Runs the experiment stage should have produced.",
    )
    args = parser.parse_args()
    base = Path(args.base_dir)

    runs_file = base / "reports" / "runs_table.csv"
    promo_file = base / "reports" / "promotion_report.json"
    metrics_file = base / "reports" / "champion_metrics.json"
    preds_file = base / "reports" / "predictions.csv"
    test_file = base / "data" / "test.csv"
    for path in (runs_file, promo_file, metrics_file, preds_file, test_file):
        if not path.exists() or path.stat().st_size == 0:
            return fail(f"missing or empty output: {path}")

    runs = pd.read_csv(runs_file)
    if len(runs) != args.expected_runs:
        return fail(
            f"runs table has {len(runs)} rows, expected {args.expected_runs}."
        )
    modes = set(runs.get("tags.logging_mode", pd.Series(dtype=str)))
    if not {"manual", "autolog"} <= modes:
        return fail(f"runs table is missing a logging mode; found {modes}.")

    promo = json.loads(promo_file.read_text())
    if promo["decision"] not in ("promote", "hold"):
        return fail(f"unknown promotion decision: {promo['decision']!r}")
    if not promo["champion_after"]:
        return fail("promotion report has no champion after the gate ran.")

    metrics = json.loads(metrics_file.read_text())
    n_test = len(pd.read_csv(test_file))
    n_pred = len(pd.read_csv(preds_file))
    if metrics["n_test"] != n_test or n_pred != n_test:
        return fail(
            f"row mismatch: test {n_test}, metrics n_test "
            f"{metrics['n_test']}, predictions {n_pred}."
        )

    # The served champion's measured RMSE must equal the tracked RMSE
    # of the run the alias points at.
    import mlflow
    from mlflow.tracking import MlflowClient

    from src.tracking import (
        CHAMPION_ALIAS,
        REGISTERED_MODEL_NAME,
        get_tracking_uri,
    )

    mlflow.set_tracking_uri(get_tracking_uri(base))
    client = MlflowClient()
    version = client.get_model_version_by_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS
    )
    tracked_rmse = float(client.get_run(version.run_id).data.metrics["rmse"])
    if abs(tracked_rmse - metrics["rmse"]) > 0.01:
        return fail(
            f"champion RMSE mismatch: scoring measured {metrics['rmse']} "
            f"but the aliased run recorded {tracked_rmse}."
        )

    print(
        "Smoke check passed: runs table, promotion report, champion "
        "metrics, and predictions are present and consistent."
    )
    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
