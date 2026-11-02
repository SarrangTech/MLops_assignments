"""Tests for the data pipeline, training, and evaluation stages.

The training test runs the real comparison + tuning flow with fewer
CV folds so the suite stays fast; the assertions check the contract
between stages (versioned model file, CV report, metrics file,
residual rows), not a particular score.
"""

import json

import numpy as np
import pytest

from src.data_pipeline import TARGET, feature_columns, load_and_split
from src.evaluate import run_evaluation
from src.train import CANDIDATES, run_training


@pytest.fixture(scope="module")
def splits(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    return load_and_split(data_dir=data_dir, seed=42)


def test_split_sizes_and_columns(splits):
    train, test = splits
    assert len(train) + len(test) == 442  # full diabetes dataset
    assert len(test) == 89  # 20% of 442, rounded up
    assert TARGET in train.columns
    assert len(feature_columns(train)) == 10
    assert not train.isna().any().any()


def test_split_is_reproducible(tmp_path):
    train_a, test_a = load_and_split(data_dir=tmp_path / "a", seed=42)
    train_b, test_b = load_and_split(data_dir=tmp_path / "b", seed=42)
    assert list(test_a.index) == list(test_b.index)
    assert list(train_a.index) == list(train_b.index)


def test_train_then_evaluate_round_trip(tmp_path):
    report = run_training(base_dir=tmp_path, timestamp="pytest", cv_folds=3)
    assert report["best_model"] in CANDIDATES
    assert set(report["candidates"]) == set(CANDIDATES)
    for res in report["candidates"].values():
        assert np.isfinite(res["cv_rmse_mean"]) and res["cv_rmse_mean"] > 0
    assert (tmp_path / "models" / "model_pytest.joblib").exists()
    assert (tmp_path / "reports" / "cv_results.json").exists()

    metrics = run_evaluation(base_dir=tmp_path, timestamp="pytest")
    assert metrics["n_test"] == 89
    assert metrics["rmse"] > 0 and metrics["mae"] > 0
    assert metrics["r2"] <= 1.0
    saved = json.loads(
        (tmp_path / "metrics" / "pytest_metrics.json").read_text()
    )
    assert saved == metrics

    residuals = (tmp_path / "reports" / "residuals.csv").read_text().splitlines()
    assert len(residuals) == 90  # header + one row per test patient


def test_evaluate_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="run the training"):
        run_evaluation(base_dir=tmp_path, timestamp="does-not-exist")
