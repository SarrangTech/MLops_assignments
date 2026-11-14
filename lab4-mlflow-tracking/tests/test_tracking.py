"""Tests for the tracking, registry gate, and scoring stages.

Every test uses a temporary SQLite tracking store, so the project's
real mlflow.db is never created or touched. The promotion tests log
runs with deliberately set RMSE metrics: the gate reads tracked
metrics, so setting them directly is what exercises its decision
logic deterministically (better candidate promotes, repeat call
holds, by the margin rule).
"""

import mlflow
import mlflow.sklearn
import pytest
from mlflow.tracking import MlflowClient
from sklearn.linear_model import Ridge

from src.data_pipeline import load_and_split
from src.register_and_promote import register_and_promote
from src.run_experiments import run_experiments
from src.score_with_champion import score_with_champion
from src.tracking import (
    CHAMPION_ALIAS,
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
)


@pytest.fixture()
def store(tmp_path):
    """Project layout in a temp dir with its own tracking database."""
    uri = f"sqlite:///{(tmp_path / 'test_mlflow.db').as_posix()}"
    return tmp_path, uri


@pytest.fixture()
def data_dir(store):
    base, _ = store
    load_and_split(data_dir=base / "data", seed=42)
    return base


def _fake_run(uri, base, rmse: float) -> str:
    """Log one real fitted model whose run records the given RMSE."""
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    train, _ = load_and_split(data_dir=base / "data", seed=42)
    features = [c for c in train.columns if c != "MedHouseVal"]
    model = Ridge(alpha=1.0).fit(train[features], train["MedHouseVal"])
    with mlflow.start_run() as run:
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, "model")
        return run.info.run_id


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
    assert table["metrics.rmse"].notna().all()
    assert (base / "reports" / "runs_table.csv").exists()

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 4
    assert all("alpha" in r.data.params for r in runs if r.data.tags.get(
        "logging_mode") == "manual")


def test_promotion_gate_promotes_then_holds(store, data_dir):
    base, uri = store
    _fake_run(uri, base, rmse=0.90)
    first = register_and_promote(base_dir=base, tracking_uri=uri, margin=0.01)
    assert first["decision"] == "promote"  # no champion existed
    champion_v1 = first["champion_after"]["version"]

    # No better run exists: the same best run is the candidate, and a
    # tie does not beat the champion by the margin -> hold.
    second = register_and_promote(base_dir=base, tracking_uri=uri, margin=0.01)
    assert second["decision"] == "hold"
    assert second["champion_after"]["version"] == champion_v1

    # A genuinely better run appears -> promote, alias moves.
    _fake_run(uri, base, rmse=0.50)
    third = register_and_promote(base_dir=base, tracking_uri=uri, margin=0.01)
    assert third["decision"] == "promote"
    assert third["champion_after"]["version"] != champion_v1
    assert third["champion_after"]["rmse"] == 0.50


def test_score_champion_round_trip(store, data_dir):
    base, uri = store
    _fake_run(uri, base, rmse=0.90)
    register_and_promote(base_dir=base, tracking_uri=uri)

    metrics = score_with_champion(base_dir=base, tracking_uri=uri)
    assert metrics["n_test"] == 4128
    assert metrics["model_uri"].endswith(f"@{CHAMPION_ALIAS}")
    assert metrics["rmse"] > 0

    client = MlflowClient(tracking_uri=uri)
    version = client.get_model_version_by_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS
    )
    assert version is not None
