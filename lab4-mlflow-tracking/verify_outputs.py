"""Smoke-check the outputs of a full pipeline run.

Run after the experiment, registry, scoring, and serving stages (CI
does this before uploading artifacts). Fails with a non-zero exit
code unless:

- the runs table covers the full experiment grid plus the autolog
  run, ordered by validation RMSE, with validation and test metrics
  recorded for every run;
- the promotion report is well-formed for its decision (a promote
  registered the version the alias now points at; a hold registered
  nothing and left the champion untouched);
- the champion metrics were computed on every test row, and the
  predictions file has one row per test row;
- the champion's measured test RMSE matches the test RMSE its run
  tracked during the experiment stage, and the selection validation
  RMSE recorded at scoring time matches the tracked one - i.e. the
  model that was served is the model that was gated, scored on data
  the selection never used;
- the serving check passed: the model served over HTTP by
  `mlflow models serve` answered with the same predictions as the
  in-process model, within its recorded tolerance.
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
    serve_file = base / "reports" / "serve_check.json"
    test_file = base / "data" / "test.csv"
    required = (
        runs_file,
        promo_file,
        metrics_file,
        preds_file,
        serve_file,
        test_file,
    )
    for path in required:
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
    for column in ("metrics.val_rmse", "metrics.test_rmse"):
        if column not in runs or runs[column].isna().any():
            return fail(f"runs table is missing values in {column}.")
    if not runs["metrics.val_rmse"].is_monotonic_increasing:
        return fail("runs table is not ordered by validation RMSE.")

    promo = json.loads(promo_file.read_text())
    if promo["decision"] not in ("promote", "hold"):
        return fail(f"unknown promotion decision: {promo['decision']!r}")
    if not promo["champion_after"]:
        return fail("promotion report has no champion after the gate ran.")
    if promo["decision"] == "promote":
        registered = promo["candidate"]["registered_version"]
        if registered != promo["champion_after"]["version"]:
            return fail(
                "promotion report: the registered candidate version "
                f"{registered} is not the champion version "
                f"{promo['champion_after']['version']}."
            )
    else:  # hold
        if promo["candidate"]["registered_version"] is not None:
            return fail("hold decision but a version was registered.")
        before = promo["champion_before"]
        if not before or before["version"] != promo["champion_after"]["version"]:
            return fail("hold decision but the champion changed.")

    metrics = json.loads(metrics_file.read_text())
    n_test = len(pd.read_csv(test_file))
    n_pred = len(pd.read_csv(preds_file))
    if metrics["n_test"] != n_test or n_pred != n_test:
        return fail(
            f"row mismatch: test {n_test}, metrics n_test "
            f"{metrics['n_test']}, predictions {n_pred}."
        )

    serve = json.loads(serve_file.read_text())
    if serve["status"] != "pass":
        return fail(f"serving check did not pass: {serve['status']}")
    if serve["max_abs_diff"] > serve["tolerance"]:
        return fail(
            "served predictions differ from in-process predictions by "
            f"{serve['max_abs_diff']} (tolerance {serve['tolerance']})."
        )

    # The scored champion's metrics must match what its run tracked:
    # same validation RMSE it was selected on, same test RMSE.
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
    tracked = client.get_run(version.run_id).data.metrics
    if abs(float(tracked["val_rmse"]) - metrics["selection_val_rmse"]) > 0.001:
        return fail(
            "selection val_rmse mismatch: scoring recorded "
            f"{metrics['selection_val_rmse']} but the aliased run "
            f"tracked {tracked['val_rmse']}."
        )
    if abs(float(tracked["test_rmse"]) - metrics["test_rmse"]) > 0.01:
        return fail(
            "champion test RMSE mismatch: scoring measured "
            f"{metrics['test_rmse']} but the aliased run recorded "
            f"{tracked['test_rmse']}."
        )

    print(
        "Smoke check passed: runs table, promotion report, champion "
        "metrics, predictions, and serving check are present and "
        "consistent."
    )
    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
