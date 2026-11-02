"""Data pipeline for the diabetes progression regression project.

Loads scikit-learn's diabetes dataset (442 patients, 10 baseline
features, target = a quantitative measure of disease progression one
year later), splits it into train/test with a fixed seed, and writes
the splits to ``data/train.csv`` and ``data/test.csv`` so the training
and evaluation stages - and anyone inspecting the run - work from
exactly the same rows.

Scaling is deliberately *not* done here: each candidate model is a
Pipeline whose StandardScaler is fitted inside cross-validation, so
no information from validation or test rows leaks into the features.
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

TARGET = "target"
BASE_DIR = Path(__file__).resolve().parent.parent


def load_and_split(
    data_dir: Path | None = None,
    seed: int = 42,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the diabetes dataset, split it, and save both splits as CSV.

    Returns the (train, test) DataFrames, each including the target
    column. The split is stratified by target quintile so both parts
    cover the same progression range.
    """
    data_dir = data_dir or (BASE_DIR / "data")
    bunch = load_diabetes(as_frame=True)
    frame = bunch.frame.copy()
    frame.columns = [str(c) for c in frame.columns]

    quintile = pd.qcut(frame[TARGET], q=5, labels=False, duplicates="drop")
    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=quintile,
    )

    data_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    return train, test


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """All columns except the target."""
    return [c for c in frame.columns if c != TARGET]
