"""Shared tracking configuration for the whole project.

One place decides where MLflow records everything: a SQLite
backend-store file (mlflow.db in the project folder) unless the
MLFLOW_TRACKING_URI environment variable overrides it. A database
backend store is what makes the MLflow Model Registry available -
the course demo's default file store cannot register models, which
is why its registry code path never executes there. Tests point the
URI at a temporary database so the real store is never touched.
"""

import os
import shutil
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
EXPERIMENT_NAME = "california-housing-experiments"
REGISTERED_MODEL_NAME = "CaliforniaHousingModel"
CHAMPION_ALIAS = "champion"

# Runs are selected, and the champion gate is decided, on validation
# metrics only. Test metrics are logged for the record and used once,
# when the finished champion is scored.
PRIMARY_METRIC = "val_rmse"


def get_tracking_uri(base_dir: Path | None = None) -> str:
    """The tracking URI every stage uses (env override wins)."""
    override = os.environ.get("MLFLOW_TRACKING_URI")
    if override:
        return override
    base = base_dir or BASE_DIR
    return f"sqlite:///{(base / 'mlflow.db').as_posix()}"


def reset_tracking_store(
    base_dir: Path | None = None, tracking_uri: str | None = None
) -> list[str]:
    """Delete the local tracking store so a run starts from empty.

    Removes the SQLite database file behind the tracking URI and the
    default local artifact directory (mlruns/) next to the project.
    Only file-backed SQLite stores can be reset this way; for any
    other URI nothing is removed and the returned list says so.
    Returns human-readable descriptions of what was removed.
    """
    base = base_dir or BASE_DIR
    uri = tracking_uri or get_tracking_uri(base)
    removed: list[str] = []
    if uri.startswith("sqlite:///"):
        db_path = Path(uri.removeprefix("sqlite:///"))
        if db_path.exists():
            db_path.unlink()
            removed.append(f"tracking database {db_path}")
    else:
        removed.append(f"no reset for non-SQLite tracking URI {uri!r}")
    artifact_dir = base / "mlruns"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
        removed.append(f"artifact directory {artifact_dir}")
    return removed


def eval_metrics(actual, pred) -> dict[str, float]:
    """Regression metrics, rounded for stable comparison and display."""
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(actual, pred))), 4),
        "mae": round(float(mean_absolute_error(actual, pred)), 4),
        "r2": round(float(r2_score(actual, pred)), 4),
    }
