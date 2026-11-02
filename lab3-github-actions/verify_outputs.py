"""Smoke-check the outputs of a train + evaluate run.

Run after both stages with the same timestamp (CI does this before
uploading artifacts). Fails with a non-zero exit code if the versioned
model, the metrics file, the CV comparison report, or the residuals
are missing or inconsistent with each other.
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", default="local")
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    args = parser.parse_args()
    base = Path(args.base_dir)
    ts = args.timestamp

    model = base / "models" / f"model_{ts}.joblib"
    metrics_file = base / "metrics" / f"{ts}_metrics.json"
    cv_file = base / "reports" / "cv_results.json"
    residuals_file = base / "reports" / "residuals.csv"
    for path in (model, metrics_file, cv_file, residuals_file):
        if not path.exists() or path.stat().st_size == 0:
            print(f"FAIL: missing or empty output: {path}")
            return 1

    metrics = json.loads(metrics_file.read_text())
    cv = json.loads(cv_file.read_text())
    residual_rows = len(residuals_file.read_text().splitlines()) - 1
    if metrics["model_file"] != model.name:
        print("FAIL: metrics were produced for a different model file.")
        return 1
    if residual_rows != metrics["n_test"]:
        print(
            f"FAIL: residuals have {residual_rows} rows but metrics "
            f"report n_test={metrics['n_test']}."
        )
        return 1
    if cv["best_model"] not in cv["candidates"]:
        print("FAIL: CV report's best model is not among the candidates.")
        return 1

    print(
        "Smoke check passed: model, metrics, CV report and residuals "
        "are present and consistent."
    )
    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
