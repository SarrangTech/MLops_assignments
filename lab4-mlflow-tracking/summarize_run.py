"""Print a Markdown summary of the experiment + registry pipeline.

CI appends this to the GitHub Actions run summary so the experiment
comparison, the promotion decision, and the champion's test metrics
are visible on the run page without downloading artifacts.
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

    best = runs.sort_values("metrics.rmse").iloc[0]
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
            f"Best run: **{best['tags.model_family']}** - test RMSE "
            f"{best['metrics.rmse']}, MAE {best['metrics.mae']}, "
            f"R2 {best['metrics.r2']}"
        ),
        "",
        (
            f"Promotion gate: **{promo['decision']}** - {promo['reason']}. "
            f"Champion is now version {promo['champion_after']['version']} "
            f"(tracked RMSE {promo['champion_after']['rmse']})."
        ),
        "",
        "| Champion (loaded by alias) | Value |",
        "|---|---|",
        f"| Test RMSE | {metrics['rmse']} |",
        f"| Test MAE | {metrics['mae']} |",
        f"| Test R2 | {metrics['r2']} |",
        f"| n_test | {metrics['n_test']} |",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
