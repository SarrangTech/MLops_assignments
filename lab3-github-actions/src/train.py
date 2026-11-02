"""Training stage: model comparison, tuning, tracking, versioning.

Candidates - Ridge, RandomForest, GradientBoosting, each behind a
StandardScaler inside one Pipeline - are compared by cross-validated
RMSE on the training split only. The winner is tuned with
GridSearchCV, refitted on the full training split, tracked in MLflow
(params, CV scores for every candidate, best hyperparameters), and
saved as a timestamp-versioned artifact
``models/model_<timestamp>.joblib``. The comparison table is written
to ``reports/cv_results.json`` so the choice of model is auditable
without opening MLflow.
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import mlflow
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_pipeline import TARGET, feature_columns, load_and_split

BASE_DIR = Path(__file__).resolve().parent.parent
SEED = 42

CANDIDATES = {
    "ridge": Ridge(),
    "random_forest": RandomForestRegressor(random_state=SEED),
    "gradient_boosting": GradientBoostingRegressor(random_state=SEED),
}

PARAM_GRIDS = {
    "ridge": {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
    "random_forest": {
        "model__n_estimators": [200],
        "model__max_depth": [None, 6, 10],
        "model__min_samples_leaf": [1, 3],
    },
    "gradient_boosting": {
        "model__n_estimators": [150, 300],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3],
    },
}


def make_pipeline(model) -> Pipeline:
    """Scaler + model as one Pipeline (the scaler is fitted per fold)."""
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def run_training(
    base_dir: Path | None = None,
    timestamp: str = "local",
    seed: int = SEED,
    cv_folds: int = 5,
) -> dict:
    """Compare, tune, fit and version the best model. Returns a summary."""
    base = base_dir or BASE_DIR
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("train")

    train, _ = load_and_split(data_dir=base / "data", seed=seed)
    features = feature_columns(train)
    x_train, y_train = train[features], train[TARGET]
    log.info(
        "Training split: %d rows, %d features", len(train), len(features)
    )

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_results: dict[str, dict] = {}
    for name, model in CANDIDATES.items():
        scores = cross_val_score(
            make_pipeline(model),
            x_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
        )
        cv_results[name] = {
            "cv_rmse_mean": round(float(-scores.mean()), 4),
            "cv_rmse_std": round(float(scores.std()), 4),
        }
        log.info("%s: CV RMSE %.4f (+/- %.4f)", name, -scores.mean(), scores.std())

    best_name = min(cv_results, key=lambda n: cv_results[n]["cv_rmse_mean"])
    log.info("Best candidate by CV RMSE: %s", best_name)

    search = GridSearchCV(
        make_pipeline(CANDIDATES[best_name]),
        PARAM_GRIDS[best_name],
        cv=cv,
        scoring="neg_root_mean_squared_error",
    )
    search.fit(x_train, y_train)
    best_model = search.best_estimator_
    log.info("Best params: %s", search.best_params_)

    (base / "reports").mkdir(parents=True, exist_ok=True)
    report = {
        "candidates": cv_results,
        "best_model": best_name,
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
        "best_cv_rmse": round(float(-search.best_score_), 4),
        "n_train": len(train),
        "features": features,
        "seed": seed,
        "cv_folds": cv_folds,
    }
    (base / "reports" / "cv_results.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    mlflow.set_tracking_uri((base / "mlruns").as_uri())
    mlflow.set_experiment("diabetes-progression")
    with mlflow.start_run(run_name=f"train_{timestamp}"):
        mlflow.log_params(
            {
                "best_model": best_name,
                "seed": seed,
                "cv_folds": cv_folds,
                "n_train": len(train),
                **{k: str(v) for k, v in search.best_params_.items()},
            }
        )
        for name, res in cv_results.items():
            mlflow.log_metric(f"cv_rmse_{name}", res["cv_rmse_mean"])
        mlflow.log_metric("cv_rmse_best_tuned", report["best_cv_rmse"])

    models_dir = base / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"model_{timestamp}.joblib"
    joblib.dump(best_model, model_path)
    log.info("Saved versioned model: %s", model_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timestamp",
        default="local",
        help="Version tag for the saved model (CI passes the run timestamp).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    report = run_training(
        timestamp=args.timestamp, seed=args.seed, cv_folds=args.cv_folds
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
