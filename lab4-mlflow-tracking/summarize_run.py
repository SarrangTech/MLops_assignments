"""Print a Markdown summary of the experiment + registry pipeline.

CI appends this to the GitHub Actions run summary so the experiment
comparison (on validation metrics), the promotion decision, the
champion's held-out test metrics, and the serving check are visible
on the run page without downloading artifacts.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    args = parser.parse_args()
    base = Path(args.base_dir)

    runs = pd.read_csv(base / "reports" / "runs_table.csv")
    promo = json.loads((base / "reports" / "promotion_report.json").read_text())
    metrics = json.loads(
        (base / "reports" / "champion_metrics.json").read_text()
    )

    best = runs.sort_values("metrics.val_rmse").iloc[0]
    n_manual = int((runs["tags.logging_mode"] == "manual").sum())
    n_autolog = int((runs["tags.logging_mode"] == "autolog").sum())
    lines = [
        "## Lab 4 - MLflow experiment + registry results",
        "",
        (
            f"Runs tracked: **{len(runs)}** "
            f"({n_manual} manual, {n_autolog} autolog)"
        ),
        "",
        (
            f"Best run by validation RMSE: **{best['tags.model_family']}** "
            f"- val RMSE {best['metrics.val_rmse']}, "
            f"val MAE {best['metrics.val_mae']}, "
            f"val R2 {best['metrics.val_r2']} "
            f"(its test RMSE, not used for selection: "
            f"{best['metrics.test_rmse']})"
        ),
        "",
        (
            f"Promotion gate: **{promo['decision']}** - {promo['reason']}. "
            f"Champion is version {promo['champion_after']['version']} "
            f"(tracked val RMSE {promo['champion_after']['val_rmse']})."
        ),
        "",
        "| Champion (loaded by alias, scored once on the test split) | Value |",
        "|---|---|",
        f"| Test RMSE | {metrics['test_rmse']} |",
        f"| Test MAE | {metrics['test_mae']} |",
        f"| Test R2 | {metrics['test_r2']} |",
        f"| Selection val RMSE | {metrics['selection_val_rmse']} |",
        f"| n_test | {metrics['n_test']} |",
    ]

    serve_file = base / "reports" / "serve_check.json"
    if serve_file.exists():
        serve = json.loads(serve_file.read_text())
        lines += [
            "",
            (
                f"Serving check: **{serve['status']}** - "
                f"`mlflow models serve` answered /invocations for "
                f"{serve['n_rows']} test rows; max difference from "
                f"in-process predictions {serve['max_abs_diff']} "
                f"(tolerance {serve['tolerance']}); served via "
                f"{serve['resolution']}."
            ),
        ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
