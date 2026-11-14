"""Data pipeline: California housing dataset -> train/test CSVs.

Downloads scikit-learn's California housing dataset (20,640 census
block groups, 8 features, target = median house value in $100k),
splits it 80/20 with a fixed seed, and writes data/train.csv and
data/test.csv. Every stage of this project - experiments, registry
promotion, champion scoring - reads these same two files, so a run's
test metrics are always computed on identical rows.
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

TARGET = "MedHouseVal"
BASE_DIR = Path(__file__).resolve().parent.parent


def load_and_split(
    data_dir: Path | None = None,
    seed: int = 42,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch the dataset, split it, save both splits, return them."""
    data_dir = data_dir or (BASE_DIR / "data")
    bunch = fetch_california_housing(as_frame=True)
    frame = bunch.frame.copy()

    train, test = train_test_split(frame, test_size=test_size, random_state=seed)

    data_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    return train, test


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """All columns except the target."""
    return [c for c in frame.columns if c != TARGET]
