"""Data pipeline: California housing dataset -> three CSV splits.

Downloads scikit-learn's California housing dataset (20,640 census
block groups, 8 features, target = median house value in $100k) and
writes three seeded splits to the data folder:

- data/test.csv        20% - held out; scored on once, at the end
- data/validation.csv  16% - model selection and the promotion gate
- data/train.csv       64% - fitting only

The test split is drawn first with seed 42, exactly as the previous
two-way version of this pipeline drew it, so the test rows are
unchanged; the validation split is then carved out of the remaining
80%. Selection never touches the test split: experiment runs are
compared on validation metrics, and the test split is used only by
the final champion scoring (and by the serving check, which compares
predictions on test rows without selecting anything on them).
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

TARGET = "MedHouseVal"
BASE_DIR = Path(__file__).resolve().parent.parent
SEED = 42


def load_frames(
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch the dataset and return (train, validation, test) frames."""
    bunch = fetch_california_housing(as_frame=True)
    frame = bunch.frame.copy()

    rest, test = train_test_split(frame, test_size=0.2, random_state=seed)
    train, validation = train_test_split(
        rest, test_size=0.2, random_state=seed
    )
    return train, validation, test


def write_splits(
    data_dir: Path | None = None,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Write the three split CSVs and return the frames."""
    data_dir = data_dir or (BASE_DIR / "data")
    train, validation, test = load_frames(seed=seed)

    data_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(data_dir / "train.csv", index=False)
    validation.to_csv(data_dir / "validation.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    return train, validation, test


def read_splits(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read back the three split CSVs written by write_splits."""
    data_dir = data_dir or (BASE_DIR / "data")
    return (
        pd.read_csv(data_dir / "train.csv"),
        pd.read_csv(data_dir / "validation.csv"),
        pd.read_csv(data_dir / "test.csv"),
    )


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """All columns except the target."""
    return [c for c in frame.columns if c != TARGET]
