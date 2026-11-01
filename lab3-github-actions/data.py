"""
Dataset loading and splitting for the Lab 3 pipeline.

The digits dataset ships with scikit-learn, so nothing is downloaded and
every environment — laptop or CI runner — sees identical data. The split
is stratified and seeded, so the test set is the same on every run.
"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2
SEED = 42


def load_dataset():
    """Return (X, y) for the digits dataset: 1797 rows, 64 features, 10 classes."""
    digits = load_digits()
    return digits.data, digits.target


def split_dataset(X, y, test_size=TEST_SIZE, seed=SEED):
    """Stratified train/test split with a fixed seed."""
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
