"""
Local version of the Lab 3 workflow's smoke check.

The GitHub Actions workflow fails the build if a training run produces no
model file or no log file. This script performs the same check on a local
run (and also flags empty files), so the result CI will see can be confirmed
before pushing:

    python lab3-github-actions/verify_outputs.py [pipeline-dir]

pipeline-dir defaults to lab1-logging; run from the repository root.
Exit code 0 means a non-empty model file and a non-empty log file are both
present; exit code 1 means at least one is missing or empty.
"""
import sys
from pathlib import Path


def find_outputs(base):
    def files_under(subdir):
        folder = base / subdir
        if not folder.is_dir():
            return []
        return sorted(p for p in folder.rglob("*") if p.is_file())

    return files_under("models"), files_under("logs")


def main(argv):
    base = Path(argv[1]) if len(argv) > 1 else Path("lab1-logging")
    if not base.is_dir():
        print(f"pipeline directory not found: {base}")
        return 1

    models, logs = find_outputs(base)
    print("Model files:")
    for path in models:
        print(f"  - {path} ({path.stat().st_size} bytes)")
    if not models:
        print("  - none")
    print("Log files:")
    for path in logs:
        print(f"  - {path} ({path.stat().st_size} bytes)")
    if not logs:
        print("  - none")

    empty = [p for p in models + logs if p.stat().st_size == 0]
    for path in empty:
        print(f"empty output file: {path}")

    if models and logs and not empty:
        print("Smoke check passed: model and log outputs are present.")
        return 0
    print("Smoke check FAILED: outputs missing or empty.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
