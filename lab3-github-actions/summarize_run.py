"""Print a Markdown summary of a train + evaluate run.

CI appends this to the GitHub Actions run summary ($GITHUB_STEP_SUMMARY)
so the model comparison and the held-out test metrics are visible on
the run page without downloading artifacts.
"""

import argparse
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", default="local")
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    args = parser.parse_args()
    base = Path(args.base_dir)

    cv = json.loads((base / "reports" / "cv_results.json").read_text())
    metrics = json.loads(
        (base / "metrics" / f"{args.timestamp}_metrics.json").read_text()
    )

    lines = [
        "## Lab 3 — training results",
        "",
        "| Candidate | CV RMSE (mean) | CV RMSE (std) |",
        "|---|---|---|",
    ]
    for name, res in cv["candidates"].items():
        lines.append(f"| {name} | {res['cv_rmse_mean']} | {res['cv_rmse_std']} |")
    lines += [
        "",
        f"Best model: **{cv['best_model']}** — tuned CV RMSE "
        f"{cv['best_cv_rmse']}, params `{cv['best_params']}`",
        "",
        "| Held-out test metric | Value |",
        "|---|---|",
        f"| RMSE | {metrics['rmse']} |",
        f"| MAE | {metrics['mae']} |",
        f"| R2 | {metrics['r2']} |",
        f"| n_test | {metrics['n_test']} |",
        f"| Model file | `{metrics['model_file']}` |",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
