"""Registry stage: register the best run, gate the champion alias.

The best experiment run by validation RMSE (val_rmse, the project's
selection metric) is the candidate. The `champion` alias - the name
scoring and serving load by - moves to the candidate only if it
beats the current champion's val_rmse by at least the promotion
margin (default 1% relative). With no existing champion, the
candidate is promoted unconditionally.

Registration follows the decision, not the other way round: a
promoted candidate is registered as a new model version and tagged
gate_decision=promote; on a hold decision nothing is registered at
all, so every version in the registry is a model that passed the
gate. A held candidate is not lost - it stays in the experiment as
a run, and the decision record names its run id.

The decision, with the candidate's and champion's validation and
test metrics and the margin, is written to
reports/promotion_report.json. The gate never reads test metrics:
selection on the test split would make the champion's final test
score an optimistic estimate instead of an unbiased one.
"""

import argparse
import json
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from .tracking import (
    CHAMPION_ALIAS,
    EXPERIMENT_NAME,
    PRIMARY_METRIC,
    REGISTERED_MODEL_NAME,
    get_tracking_uri,
)

BASE_DIR = Path(__file__).resolve().parent.parent


def _run_metrics(client: MlflowClient, run_id: str) -> dict:
    metrics = client.get_run(run_id).data.metrics
    return {
        "val_rmse": float(metrics["val_rmse"]),
        "test_rmse": float(metrics["test_rmse"]),
    }


def _best_run(client: MlflowClient) -> tuple[str, dict]:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{PRIMARY_METRIC} ASC"],
        max_results=1,
    )
    if not runs:
        raise LookupError(f"No runs found in experiment {EXPERIMENT_NAME!r}.")
    run = runs[0]
    return run.info.run_id, _run_metrics(client, run.info.run_id)


def _champion(client: MlflowClient) -> dict | None:
    """Version and metrics behind the champion alias, or None."""
    try:
        version = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME, CHAMPION_ALIAS
        )
    except MlflowException:
        return None
    return {
        "version": version.version,
        "run_id": version.run_id,
        **_run_metrics(client, version.run_id),
    }


def register_and_promote(
    base_dir: Path | None = None,
    tracking_uri: str | None = None,
    margin: float = 0.01,
) -> dict:
    """Apply the promotion gate to the best run; register on promote."""
    base = base_dir or BASE_DIR
    mlflow.set_tracking_uri(tracking_uri or get_tracking_uri(base))
    client = MlflowClient()

    candidate_run_id, candidate = _best_run(client)
    before = _champion(client)

    if before is None:
        decision, reason = "promote", "no existing champion"
    elif candidate["val_rmse"] <= before["val_rmse"] * (1 - margin):
        decision = "promote"
        reason = (
            f"candidate val_rmse {candidate['val_rmse']} beats champion "
            f"val_rmse {before['val_rmse']} by more than the "
            f"{margin:.0%} margin"
        )
    else:
        decision = "hold"
        reason = (
            f"candidate val_rmse {candidate['val_rmse']} does not beat "
            f"champion val_rmse {before['val_rmse']} by the "
            f"{margin:.0%} margin"
        )

    registered_version = None
    if decision == "promote":
        version = mlflow.register_model(
            f"runs:/{candidate_run_id}/model", REGISTERED_MODEL_NAME
        )
        registered_version = version.version
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            registered_version,
            "gate_decision",
            "promote",
        )
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, CHAMPION_ALIAS, registered_version
        )

    report = {
        "registered_model": REGISTERED_MODEL_NAME,
        "selection_metric": PRIMARY_METRIC,
        "margin": margin,
        "decision": decision,
        "reason": reason,
        "candidate": {
            "run_id": candidate_run_id,
            **candidate,
            "registered_version": registered_version,
        },
        "champion_before": before,
        "champion_after": _champion(client),
    }
    (base / "reports").mkdir(parents=True, exist_ok=True)
    (base / "reports" / "promotion_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()
    register_and_promote(
        base_dir=Path(args.base_dir) if args.base_dir else None,
        tracking_uri=args.tracking_uri,
        margin=args.margin,
    )


if __name__ == "__main__":
    main()
