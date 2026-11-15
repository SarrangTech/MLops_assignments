"""Serving check: serve the champion over HTTP and compare answers.

The course serving lab runs `mlflow models serve` and posts rows to
the server's /invocations endpoint. This stage does exactly that
against the registered champion, as an automated check:

1. The champion is loaded in-process and predicts the first N rows
   of the test split (default 20).
2. `mlflow models serve` is started as a background process for the
   champion's registry URI (models:/CaliforniaHousingModel@champion)
   on a free port, with the project's tracking URI in its
   environment so it can resolve the registry.
3. The same rows are POSTed to /invocations and the served
   predictions must match the in-process ones to within 1e-6.

If the alias URI cannot be served, the model's artifact path is
resolved through the tracking client and that local path is served
instead; the report records which form was used and why. The server
is always shut down again. The result is written to
reports/serve_check.json. This is a correctness check of the
serving path only - no selection or scoring decision uses it.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import mlflow
import mlflow.artifacts
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from .data_pipeline import feature_columns
from .score_with_champion import champion_model_uri
from .tracking import (
    CHAMPION_ALIAS,
    REGISTERED_MODEL_NAME,
    get_tracking_uri,
)

BASE_DIR = Path(__file__).resolve().parent.parent
TOLERANCE = 1e-6
HEALTH_TIMEOUT_S = 180


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http(method: str, url: str, payload: dict | None = None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url, data=data, headers=headers, method=method
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, response.read().decode()


def _wait_healthy(proc: subprocess.Popen, port: int, log_path: Path) -> None:
    """Poll the server's health endpoints until one answers 200."""
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-1500:]
            raise RuntimeError(
                f"model server exited with code {proc.returncode}.\n{tail}"
            )
        for endpoint in ("/health", "/ping"):
            try:
                status, _ = _http("GET", f"http://127.0.0.1:{port}{endpoint}")
                if status == 200:
                    return
            except (urllib.error.URLError, OSError):
                pass
        time.sleep(2)
    tail = log_path.read_text(errors="replace")[-1500:]
    raise RuntimeError(
        f"model server not healthy after {HEALTH_TIMEOUT_S}s.\n{tail}"
    )


def _start_server(model_ref: str, port: int, tracking_uri: str, log_path: Path):
    env = dict(os.environ)
    env["MLFLOW_TRACKING_URI"] = tracking_uri
    env["MLFLOW_REGISTRY_URI"] = tracking_uri
    # `mlflow models serve` shells out to gunicorn by name. When this
    # stage is run as `python -m src.serve_check` from a virtualenv
    # that was never activated, the venv's bin directory is not on
    # PATH and the server dies with "gunicorn: command not found" -
    # put the running interpreter's bin directory first.
    env["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    )
    log_file = open(log_path, "w")  # noqa: SIM115 - handed to the child
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "models",
            "serve",
            "-m",
            model_ref,
            "--env-manager",
            "local",
            "--host",
            "127.0.0.1",
            "-p",
            str(port),
        ],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc


def _stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=15)


def _serve_and_predict(model_ref: str, tracking_uri: str, features, rows):
    """Start the server for model_ref, POST rows, return predictions."""
    port = _free_port()
    log_path = Path(tempfile.mkstemp(prefix="serve_check_", suffix=".log")[1])
    proc = _start_server(model_ref, port, tracking_uri, log_path)
    try:
        _wait_healthy(proc, port, log_path)
        payload = {
            "dataframe_split": {
                "columns": list(features),
                "data": rows,
            }
        }
        _, body = _http("POST", f"http://127.0.0.1:{port}/invocations", payload)
        return [float(p) for p in json.loads(body)["predictions"]]
    finally:
        _stop_server(proc)


def serve_check(
    base_dir: Path | None = None,
    tracking_uri: str | None = None,
    n_rows: int = 20,
    tolerance: float = TOLERANCE,
) -> dict:
    """Serve the champion and compare served vs in-process predictions."""
    base = base_dir or BASE_DIR
    uri = tracking_uri or get_tracking_uri(base)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    version = client.get_model_version_by_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS
    )

    test = pd.read_csv(base / "data" / "test.csv")
    features = feature_columns(test)
    sample = test[features].head(n_rows)
    rows = sample.to_numpy().tolist()

    model = mlflow.sklearn.load_model(champion_model_uri())
    in_process = [float(p) for p in model.predict(sample)]

    alias_uri = champion_model_uri()
    resolution = "alias_uri"
    reason = None
    served_ref = alias_uri
    try:
        served = _serve_and_predict(alias_uri, uri, features, rows)
    except RuntimeError as exc:
        resolution = "resolved_artifact_path"
        reason = (
            "serving the alias URI failed; resolved the champion "
            f"version's artifact path through the client instead. "
            f"Alias attempt: {str(exc)[:400]}"
        )
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{version.run_id}/model"
        )
        served_ref = local_path
        served = _serve_and_predict(local_path, uri, features, rows)

    diffs = np.abs(np.array(served) - np.array(in_process))
    max_diff = float(diffs.max()) if len(diffs) else 0.0
    report = {
        "status": "pass" if max_diff <= tolerance else "fail",
        "served_model_uri": served_ref,
        "resolution": resolution,
        "reason": reason,
        "model_version": version.version,
        "n_rows": n_rows,
        "tolerance": tolerance,
        "max_abs_diff": max_diff,
    }
    (base / "reports").mkdir(parents=True, exist_ok=True)
    (base / "reports" / "serve_check.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(f"Serve check: {json.dumps(report)}")
    if report["status"] != "pass":
        raise RuntimeError(
            f"served predictions differ from in-process by {max_diff} "
            f"(tolerance {tolerance})"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rows", type=int, default=20)
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()
    serve_check(
        base_dir=Path(args.base_dir) if args.base_dir else None,
        tracking_uri=args.tracking_uri,
        n_rows=args.n_rows,
    )


if __name__ == "__main__":
    main()
