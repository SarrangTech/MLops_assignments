"""Experiment stage: a designed grid of tracked MLflow runs.

Trains twelve model configurations across five families on the
training split and records each one as an MLflow run in a single
experiment: hyperparameters and run context as parameters, the
fitted model (with an inferred signature and an input example) as
the run's artifact, and the model family as a tag. Every run logs
validation metrics (val_rmse, val_mae, val_r2) and test metrics
(test_rmse, test_mae, test_r2). Model selection and the promotion
gate use the validation metrics only; the test metrics are recorded
so the champion's final test score can be cross-checked against
them, not for choosing between runs. A thirteenth run repeats one
configuration with MLflow autologging instead of manual logging, so
the two logging styles can be compared in the same experiment.

Nothing is registered here on purpose: registration is a separate,
gated step (register_and_promote.py), not a side effect of training.

After the runs, the experiment is exported through the tracking
search API to reports/runs_table.csv - one row per run with its
parameters and metrics, ordered by val_rmse - so the comparison is
readable without opening the MLflow UI.

Re-running this stage appends runs to the experiment (MLflow's
normal behaviour). Pass --fresh to delete the local tracking store
first, which makes the documented run reproducible: exactly the 13
runs of this grid, no leftovers from earlier executions. The
tracking database is a generated artifact (it is gitignored, like
the split CSVs), so deleting it loses nothing that cannot be
rebuilt by re-running the pipeline.
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

from .data_pipeline import TARGET, feature_columns, write_splits
from .tracking import (
    EXPERIMENT_NAME,
    PRIMARY_METRIC,
    eval_metrics,
    get_tracking_uri,
    reset_tracking_store,
)

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


def _fit_and_score(model, train, validation, test):
    """Fit on train; return the model and prefixed val/test metrics."""
    features = feature_columns(train)
    model.fit(train[features], train[TARGET])
    val = eval_metrics(validation[TARGET], model.predict(validation[features]))
    tst = eval_metrics(test[TARGET], model.predict(test[features]))
    metrics = {
        "val_rmse": val["rmse"],
        "val_mae": val["mae"],
        "val_r2": val["r2"],
        "test_rmse": tst["rmse"],
        "test_mae": tst["mae"],
        "test_r2": tst["r2"],
    }
    return model, metrics


def run_experiments(
    base_dir: Path | None = None,
    tracking_uri: str | None = None,
    families: list[str] | None = None,
    include_autolog: bool = True,
    seed: int = SEED,
    fresh: bool = False,
) -> pd.DataFrame:
    """Run the grid, export the runs table, and return it."""
    base = base_dir or BASE_DIR
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("experiments")

    if fresh:
        for removed in reset_tracking_store(base, tracking_uri):
            log.info("Fresh start: removed %s", removed)

    train, validation, test = write_splits(data_dir=base / "data", seed=seed)
    features = feature_columns(train)
    log.info(
        "Splits: %d train / %d validation / %d test rows",
        len(train),
        len(validation),
        len(test),
    )

    mlflow.set_tracking_uri(tracking_uri or get_tracking_uri(base))
    mlflow.set_experiment(EXPERIMENT_NAME)

    selected = families or list(GRID)
    for family in selected:
        for i, params in enumerate(GRID[family]):
            with mlflow.start_run(run_name=f"{family}-{i}"):
                mlflow.set_tag("model_family", family)
                mlflow.set_tag("logging_mode", "manual")
                mlflow.log_params(
                    {
                        **params,
                        "seed": seed,
                        "n_train": len(train),
                        "n_validation": len(validation),
                    }
                )
                model, metrics = _fit_and_score(
                    build_model(family, params), train, validation, test
                )
                mlflow.log_metrics(metrics)
                signature = mlflow.models.infer_signature(
                    validation[features],
                    model.predict(validation[features]),
                )
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    signature=signature,
                    input_example=train[features].head(5),
                )
                log.info(
                    "%s %s -> val_rmse %s (test_rmse %s)",
                    family,
                    params,
                    metrics["val_rmse"],
                    metrics["test_rmse"],
                )

    if include_autolog:
        params = GRID["random_forest"][0]
        mlflow.sklearn.autolog()
        try:
            with mlflow.start_run(run_name="random_forest-autolog"):
                mlflow.set_tag("model_family", "random_forest")
                mlflow.set_tag("logging_mode", "autolog")
                model, metrics = _fit_and_score(
                    build_model("random_forest", params),
                    train,
                    validation,
                    test,
                )
                # Autologging records params and the model on fit; the
                # held-out metrics are added so this run is comparable
                # with the manually logged ones.
                mlflow.log_metrics(metrics)
                log.info(
                    "autolog random_forest -> val_rmse %s", metrics["val_rmse"]
                )
        finally:
            mlflow.sklearn.autolog(disable=True)

    table = _export_runs_table(base)
    log.info("Exported %d runs to reports/runs_table.csv", len(table))
    return table


def _export_runs_table(base: Path) -> pd.DataFrame:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{PRIMARY_METRIC} ASC"],
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
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the local tracking store before running, so the "
        "experiment holds exactly this execution's runs.",
    )
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()
    families = args.families.split(",") if args.families else None
    run_experiments(
        base_dir=Path(args.base_dir) if args.base_dir else None,
        tracking_uri=args.tracking_uri,
        families=families,
        include_autolog=not args.no_autolog,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    main()
