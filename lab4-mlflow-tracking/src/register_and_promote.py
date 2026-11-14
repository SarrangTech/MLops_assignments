"""Registry stage: register the best run, gate the champion alias.

The best experiment run by test RMSE is registered as a new version
of the registered model. Registration alone changes nothing for
consumers: the `champion` alias - the name scoring loads by - moves
to the new version only if the candidate beats the current
champion's RMSE by at least the promotion margin (default 1%
relative). With no existing champion, the candidate is promoted
unconditionally. The decision, with both RMSE values and the margin,
is written to reports/promotion_report.json.
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
    REGISTERED_MODEL_NAME,
    get_tracking_uri,
)

BASE_DIR = Path(__file__).resolve().parent.parent


def _best_run(client: MlflowClient) -> tuple[str, float]:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=1,
    )
    if not runs:
        raise LookupError(f"No runs found in experiment {EXPERIMENT_NAME!r}.")
    run = runs[0]
    return run.info.run_id, float(run.data.metrics["rmse"])


def _champion(client: MlflowClient) -> tuple[str, float] | None:
    """(version, rmse) behind the champion alias, or None if unset."""
    try:
        version = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME, CHAMPION_ALIAS
        )
    except MlflowException:
        return None
    run = client.get_run(version.run_id)
    return version.version, float(run.data.metrics["rmse"])


def register_and_promote(
    base_dir: Path | None = None,
    tracking_uri: str | None = None,
    margin: float = 0.01,
) -> dict:
    """Register the best run and apply the promotion gate."""
    base = base_dir or BASE_DIR
    mlflow.set_tracking_uri(tracking_uri or get_tracking_uri(base))
    client = MlflowClient()

    candidate_run_id, candidate_rmse = _best_run(client)
    before = _champion(client)

    version = mlflow.register_model(
        f"runs:/{candidate_run_id}/model", REGISTERED_MODEL_NAME
    )

    if before is None:
        decision, reason = "promote", "no existing champion"
    else:
        _, champion_rmse = before
        if candidate_rmse <= champion_rmse * (1 - margin):
            decision = "promote"
            reason = (
                f"candidate rmse {candidate_rmse} beats champion rmse "
                f"{champion_rmse} by more than the {margin:.0%} margin"
            )
        else:
            decision = "hold"
            reason = (
                f"candidate rmse {candidate_rmse} does not beat champion "
                f"rmse {champion_rmse} by the {margin:.0%} margin"
            )

    if decision == "promote":
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version.version
        )

    champion_now = _champion(client)
    report = {
        "registered_model": REGISTERED_MODEL_NAME,
        "candidate": {
            "run_id": candidate_run_id,
            "rmse": candidate_rmse,
            "registered_version": version.version,
        },
        "champion_before": (
            {"version": before[0], "rmse": before[1]} if before else None
        ),
        "margin": margin,
        "decision": decision,
        "reason": reason,
        "champion_after": (
            {"version": champion_now[0], "rmse": champion_now[1]}
            if champion_now
            else None
        ),
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
    args = parser.parse_args()
    register_and_promote(margin=args.margin)


if __name__ == "__main__":
    main()
