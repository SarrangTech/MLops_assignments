"""Smoke test for train.py: a full training run into a temp directory."""
import json

import joblib
from train import train_and_evaluate


def test_training_produces_model_and_metrics(tmp_path):
    model_path = tmp_path / "models" / "model.joblib"
    metrics_path = tmp_path / "reports" / "metrics.json"
    metrics = train_and_evaluate(
        model_path=model_path, metrics_path=metrics_path
    )

    assert model_path.is_file()
    assert metrics_path.is_file()
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved == metrics
    assert metrics["n_train"] + metrics["n_test"] == 1797
    assert metrics["accuracy"] >= 0.90

    model = joblib.load(model_path)
    assert model.predict([[0.0] * 64]).shape == (1,)
