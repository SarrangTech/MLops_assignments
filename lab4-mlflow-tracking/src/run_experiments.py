"""Experiment stage: a designed grid of tracked MLflow runs.

Trains twelve model configurations across five families on the
training split and records each one as an MLflow run in a single
experiment: hyperparameters and run context as parameters, held-out
test RMSE / MAE / R2 as metrics, the fitted model (with an inferred
signature and an input example) as the run's artifact, and the model
family as a tag. A thirteenth run repeats one configuration with
MLflow autologging instead of manual logging, so the two logging
styles can be compared in the same experiment.

Nothing is registered here on purpose: registration is a separate,
gated step (register_and_promote.py), not a side effect of training.

After the runs, the experiment is exported through the tracking
search API to reports/runs_table.csv - one row per run with its
parameters and metrics - so the comparison is readable without
opening the MLflow UI.
"""

import argparse
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .data_pipeline import TARGET, feature_columns, load_and_split
from .tracking import EXPERIMENT_NAME, eval_metrics, get_tracking_uri

BASE_DIR = Path(__file__).resolve().parent.parent
SEED = 42

# The experiment design: (family, [parameter sets]) - 12 configurations.
GRID: dict[str, list[dict]] = {
    "ridge": [{"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 10.0}],
    "elasticnet": [
        {"alpha": 0.1, "l1_ratio": 0.5},
        {"alpha": 0.5, "l1_ratio": 0.5},
        {"alpha": 0.1, "l1_ratio": 0.9},
    ],
    "random_forest": [
        {"n_estimators": 300, "max_depth": None},
        {"n_estimators": 300, "max_depth": 10},
    ],
    "gradient_boosting": [
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3},
        {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 3},
    ],
    "xgboost": [
        {"n_estimators": 400, "learning_rate": 0.1, "max_depth": 4},
        {"n_estimators": 400, "learning_rate": 0.1, "max_depth": 6},
    ],
}

SCALED_FAMILIES = {"ridge", "elasticnet"}


def build_model(family: str, params: dict):
    """One estimator per family; linear models get a scaler Pipeline."""
    if family == "ridge":
        model = Ridge(random_state=SEED, **params)
    elif family == "elasticnet":
        model = ElasticNet(random_state=SEED, max_iter=5000, **params)
    elif family == "random_forest":
        model = RandomForestRegressor(random_state=SEED, n_jobs=-1, **params)
    elif family == "gradient_boosting":
        model = GradientBoostingRegressor(random_state=SEED, **params)
    elif family == "xgboost":
        model = XGBRegressor(random_state=SEED, n_jobs=-1, **params)
    else:
        raise ValueError(f"Unknown model family: {family}")
    if family in SCALED_FAMILIES:
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    return model


def _fit_and_score(model, train: pd.DataFrame, test: pd.DataFrame):
    features = feature_columns(train)
    model.fit(train[features], train[TARGET])
    return model, eval_metrics(test[TARGET], model.predict(test[features]))


def run_experiments(
    base_dir: Path | None = None,
    tracking_uri: str | None = None,
    families: list[str] | None = None,
    include_autolog: bool = True,
    seed: int = SEED,
) -> pd.DataFrame:
    """Run the grid, export the runs table, and return it."""
    base = base_dir or BASE_DIR
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("experiments")

    train, test = load_and_split(data_dir=base / "data", seed=seed)
    features = feature_columns(train)
    log.info("Splits: %d train / %d test rows", len(train), len(test))

    mlflow.set_tracking_uri(tracking_uri or get_tracking_uri(base))
    mlflow.set_experiment(EXPERIMENT_NAME)

    selected = families or list(GRID)
    for family in selected:
        for i, params in enumerate(GRID[family]):
            with mlflow.start_run(run_name=f"{family}-{i}"):
                mlflow.set_tag("model_family", family)
                mlflow.set_tag("logging_mode", "manual")
                mlflow.log_params(
                    {**params, "seed": seed, "n_train": len(train)}
                )
                model, metrics = _fit_and_score(
                    build_model(family, params), train, test
                )
                mlflow.log_metrics(metrics)
                signature = mlflow.models.infer_signature(
                    test[features], model.predict(test[features])
                )
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    signature=signature,
                    input_example=test[features].head(5),
                )
                log.info("%s %s -> rmse %s", family, params, metrics["rmse"])

    if include_autolog:
        params = GRID["random_forest"][0]
        mlflow.sklearn.autolog()
        try:
            with mlflow.start_run(run_name="random_forest-autolog"):
                mlflow.set_tag("model_family", "random_forest")
                mlflow.set_tag("logging_mode", "autolog")
                model, metrics = _fit_and_score(
                    build_model("random_forest", params), train, test
                )
                # Autologging records params and the model on fit; the
                # held-out metrics are added so this run is comparable
                # with the manually logged ones.
                mlflow.log_metrics(metrics)
                log.info("autolog random_forest -> rmse %s", metrics["rmse"])
        finally:
            mlflow.sklearn.autolog(disable=True)

    table = _export_runs_table(base)
    log.info("Exported %d runs to reports/runs_table.csv", len(table))
    return table


def _export_runs_table(base: Path) -> pd.DataFrame:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
    )
    keep = [
        c
        for c in runs.columns
        if c in ("run_id", "status", "tags.model_family", "tags.logging_mode")
        or c.startswith(("metrics.", "params."))
    ]
    table = runs[keep].copy()
    (base / "reports").mkdir(parents=True, exist_ok=True)
    table.to_csv(base / "reports" / "runs_table.csv", index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families",
        default=None,
        help="Comma-separated subset of the grid (default: all families).",
    )
    parser.add_argument("--no-autolog", action="store_true")
    args = parser.parse_args()
    families = args.families.split(",") if args.families else None
    run_experiments(families=families, include_autolog=not args.no_autolog)


if __name__ == "__main__":
    main()
