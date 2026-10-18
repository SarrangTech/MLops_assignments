"""Lab 2 (Docker) - training payload.

Original Docker Lab1 payload: train a RandomForest on Iris inside a
container and save iris_model.pkl inside the container, with no
evaluation output and no way to get the model out or to change the
run without rebuilding the image.

Changes in this version (payload only - the Docker changes are in
the Dockerfile / docker-compose.yml and are listed in README.md):
  * Dataset: Breast Cancer Wisconsin (569 samples, 30 features,
    binary) instead of Iris.
  * The run is configurable without rebuilding: --n-estimators,
    --test-size, --random-state and --output-dir are CLI arguments,
    and the output directory can also be set with the OUTPUT_DIR
    environment variable (docker run -e OUTPUT_DIR=...).
  * It evaluates the model (accuracy + F1) and writes both the model
    and a metrics.json file to the output directory, which the
    Dockerfile / compose file mount as a volume so the artifacts
    survive the container.
"""

import argparse
import json
import os

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Train a RandomForest in a container.")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("OUTPUT_DIR", "output"),
        help="Directory for model + metrics (default: $OUTPUT_DIR or ./output)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data = load_breast_cancer()
    X, y = data.data, data.target
    print(f"Dataset: Breast Cancer Wisconsin - {X.shape[0]} samples, {X.shape[1]} features, 2 classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    print(f"Train/test split: {len(X_train)} train, {len(X_test)} test")

    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 score: {f1:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "breast_cancer_rf.joblib")
    metrics_path = os.path.join(args.output_dir, "metrics.json")

    joblib.dump(model, model_path)
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "dataset": "breast_cancer_wisconsin",
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "n_estimators": args.n_estimators,
                "test_size": args.test_size,
                "random_state": args.random_state,
                "accuracy": round(float(accuracy), 4),
                "f1_score": round(float(f1), 4),
            },
            f,
            indent=2,
        )

    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print("The model training was successful")


if __name__ == "__main__":
    main()
