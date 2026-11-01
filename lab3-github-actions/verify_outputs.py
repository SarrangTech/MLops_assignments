"""
Check that a local Lab 3 training run produced everything CI expects.

The GitHub Actions workflow fails the build if the training step leaves no
model, no metrics file, or no log. This script performs the same check on a
local run (and also flags empty files), so the result CI will see can be
confirmed before pushing:

    python verify_outputs.py [lab-dir]

lab-dir defaults to this script's own folder. Exit code 0 means all three
outputs exist and are non-empty; exit code 1 means at least one is missing
or empty.
"""
import json
import sys
from pathlib import Path

EXPECTED = [
    Path("models") / "model.joblib",
    Path("reports") / "metrics.json",
    Path("logs") / "train.log",
]


def main(argv):
    base = (
        Path(argv[1])
        if len(argv) > 1
        else Path(__file__).resolve().parent
    )
    missing = []
    for relative in EXPECTED:
        path = base / relative
        if path.is_file() and path.stat().st_size > 0:
            print(f"  ok  {relative} ({path.stat().st_size} bytes)")
        else:
            print(f"  MISSING or empty  {relative}")
            missing.append(relative)

    metrics_path = base / "reports" / "metrics.json"
    if metrics_path.is_file() and metrics_path.stat().st_size > 0:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        print("\nmetrics from the run:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    if missing:
        print("\nSmoke check FAILED: outputs missing or empty.")
        return 1
    print("\nSmoke check passed: model, metrics and log are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
