"""Tests for data.py: dataset shape, split sizes, reproducibility."""
import numpy as np

from data import load_dataset, split_dataset


def test_dataset_shape_and_classes():
    X, y = load_dataset()
    assert X.shape == (1797, 64)
    assert y.shape == (1797,)
    assert sorted(np.unique(y)) == list(range(10))


def test_split_sizes_and_stratification():
    X, y = load_dataset()
    _X_train, _X_test, y_train, y_test = split_dataset(X, y)
    assert len(y_train) == 1437
    assert len(y_test) == 360
    # Stratified split: every class is present in the test set.
    assert sorted(np.unique(y_test)) == list(range(10))


def test_split_is_reproducible():
    X, y = load_dataset()
    first = split_dataset(X, y)
    second = split_dataset(X, y)
    for a, b in zip(first, second):
        assert np.array_equal(a, b)
