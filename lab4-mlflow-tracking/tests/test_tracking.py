"""Tests for the tracking, registry gate, scoring, and serving stages.

Every test uses a temporary SQLite tracking store, so the project's
real mlflow.db is never created or touched. The promotion tests log
runs with deliberately set validation/test RMSE metrics: the gate
reads tracked val_rmse, so setting it directly is what exercises its
decision logic deterministically (better candidate promotes, repeat
call holds - and registers nothing - a genuinely better run
promotes). The serving test starts a real `mlflow models serve`
process against the temporary store; it is the same check the
pipeline runs as a stage, at test scale.
"""

import subprocess
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pytest
from mlflow.tracking import MlflowClient
from sklearn.linear_model import Ridge

from src.data_pipeline import write_splits
from src.register_and_promote import register_and_promote
from src.run_experiments import run_experiments
from src.score_with_champion import score_with_champion
from src.serve_check import serve_check
from src.tracking import (
    CHAMPION_ALIAS,
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def store(tmp_path):
    """Project layout in a temp dir with its own tracking database."""
    uri = f"sqlite:///{(tmp_path / 'test_mlflow.db').as_posix()}"
    return tmp_path, uri


@pytest.fixture()
def data_dir(store):
    base, _ = store
    write_splits(data_dir=base / "data", seed=42)
    return base


def _fake_run(uri, base, val_rmse: float, test_rmse: float) -> str:
    """Log one real fitted model whose run records the given metrics."""
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    train, _, _ = write_splits(data_dir=base / "data", seed=42)
    features = [c for c in train.columns if c != "MedHouseVal"]
    model = Ridge(alpha=1.0).fit(train[features], train["MedHouseVal"])
    with mlflow.start_run() as run:
        mlflow.log_metrics({"val_rmse": val_rmse, "test_rmse": test_rmse})
        mlflow.sklearn.log_model(model, "model")
        return run.info.run_id


def _version_count(uri) -> int:
    client = MlflowClient(tracking_uri=uri)
    return len(
        client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")
    )


def test_experiments_are_tracked_and_searchable(store):
    base, uri = store
    table = run_experiments(
        base_dir=base,
        tracking_uri=uri,
        families=["ridge"],
        include_autolog=True,
    )
    # 3 ridge configurations + 1 autologging run
    assert len(table) == 4
    assert set(table["tags.logging_mode"]) == {"manual", "autolog"}
    assert table["metrics.val_rmse"].notna().all()
    assert table["metrics.test_rmse"].notna().all()
    assert table["metrics.val_rmse"].is_monotonic_increasing
    assert (base / "reports" / "runs_table.csv").exists()

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 4
    assert all(
        "alpha" in r.data.params
        for r in runs
        if r.data.tags.get("logging_mode") == "manual"
    )


def test_promotion_gate_promotes_then_holds(store, data_dir):
    base, uri = store
    _fake_run(uri, base, val_rmse=0.90, test_rmse=0.95)
    first = register_and_promote(base_dir=base, tracking_uri=uri, margin=0.01)
    assert first["decision"] == "promote"  # no champion existed
    assert first["candidate"]["registered_version"] is not None
    champion_v1 = first["champion_after"]["version"]
    assert _version_count(uri) == 1

    # No better run exists: the same best run is the candidate, and a
    # tie does not beat the champion by the margin -> hold, and the
    # hold registers no new version.
    second = register_and_promote(base_dir=base, tracking_uri=uri, margin=0.01)
    assert second["decision"] == "hold"
    assert second["candidate"]["registered_version"] is None
    assert second["champion_after"]["version"] == champion_v1
    assert _version_count(uri) == 1

    # A genuinely better run appears -> promote, alias moves.
    _fake_run(uri, base, val_rmse=0.50, test_rmse=0.60)
    third = register_and_promote(base_dir=base, tracking_uri=uri, margin=0.01)
    assert third["decision"] == "promote"
    assert third["champion_after"]["version"] != champion_v1
    assert third["champion_after"]["val_rmse"] == 0.50
    assert _version_count(uri) == 2


def test_score_champion_round_trip(store, data_dir):
    base, uri = store
    _fake_run(uri, base, val_rmse=0.90, test_rmse=0.95)
    register_and_promote(base_dir=base, tracking_uri=uri)

    metrics = score_with_champion(base_dir=base, tracking_uri=uri)
    assert metrics["n_test"] == 4128
    assert metrics["model_uri"].endswith(f"@{CHAMPION_ALIAS}")
    assert metrics["test_rmse"] > 0
    # The champion was selected on the fake run's validation metric.
    assert metrics["selection_val_rmse"] == 0.90
    assert metrics["tracked_test_rmse"] == 0.95

    client = MlflowClient(tracking_uri=uri)
    version = client.get_model_version_by_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS
    )
    assert version is not None


def test_rerun_with_fresh_keeps_run_count(store):
    """--fresh makes the experiment stage reproducible; without it,
    runs append (MLflow's normal behaviour). Driven through the CLI
    in subprocesses, exactly as a user runs it."""
    base, uri = store

    def run_cli(*extra: str) -> int:
        cmd = [
            sys.executable,
            "-m",
            "src.run_experiments",
            "--families",
            "ridge",
            "--no-autolog",
            "--base-dir",
            str(base),
            "--tracking-uri",
            uri,
            *extra,
        ]
        result = subprocess.run(
            cmd,
            check=False,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        table = (base / "reports" / "runs_table.csv").read_text()
        return len(table.strip().splitlines()) - 1  # minus header

    assert run_cli("--fresh") == 3
    assert run_cli("--fresh") == 3  # second fresh run: still 3 runs
    assert run_cli() == 6  # without --fresh, runs append


def test_serve_check_matches_in_process(store, data_dir):
    """Real serving: start `mlflow models serve` for the champion in
    the temporary store and compare its /invocations answers with
    the in-process model."""
    base, uri = store
    _fake_run(uri, base, val_rmse=0.90, test_rmse=0.95)
    register_and_promote(base_dir=base, tracking_uri=uri)

    report = serve_check(base_dir=base, tracking_uri=uri, n_rows=20)
    assert report["status"] == "pass"
    assert report["max_abs_diff"] <= report["tolerance"]
    assert report["n_rows"] == 20
    assert (base / "reports" / "serve_check.json").exists()
