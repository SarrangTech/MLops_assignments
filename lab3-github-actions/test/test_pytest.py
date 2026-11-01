"""Pytest suite for src/calculator.py.

test_fun1-test_fun4 are the course lab's original tests
(Labs/Github_Labs/Lab1/test/test_pytest.py), with the same cases.
test_fun5, test_fun6, and the error-case tests were added for this
submission to cover the added functions and the input validation.
"""
import pytest

from src import calculator


def test_fun1():
    assert calculator.fun1(2, 3) == 5
    assert calculator.fun1(5, 0) == 5
    assert calculator.fun1(-1, 1) == 0
    assert calculator.fun1(-1, -1) == -2


def test_fun2():
    assert calculator.fun2(2, 3) == -1
    assert calculator.fun2(5, 0) == 5
    assert calculator.fun2(-1, 1) == -2
    assert calculator.fun2(-1, -1) == 0


def test_fun3():
    assert calculator.fun3(2, 3) == 6
    assert calculator.fun3(5, 0) == 0
    assert calculator.fun3(-1, 1) == -1
    assert calculator.fun3(-1, -1) == 1


def test_fun4():
    assert calculator.fun4(2, 3, 5) == 10
    assert calculator.fun4(5, 0, -1) == 4
    assert calculator.fun4(-1, -1, -1) == -3
    assert calculator.fun4(-1, -1, 100) == 98


def test_fun5():
    assert calculator.fun5(6, 3) == 2
    assert calculator.fun5(5, 2) == 2.5
    assert calculator.fun5(-4, 2) == -2
    assert calculator.fun5(0, 5) == 0


def test_fun6():
    assert calculator.fun6(2, 3) == 8
    assert calculator.fun6(5, 0) == 1
    assert calculator.fun6(9, 0.5) == 3
    assert calculator.fun6(2, -1) == 0.5


def test_invalid_inputs_raise_value_error():
    with pytest.raises(ValueError):
        calculator.fun1("2", 3)
    with pytest.raises(ValueError):
        calculator.fun2(2, "3")
    with pytest.raises(ValueError):
        calculator.fun3("2", "3")
    with pytest.raises(ValueError):
        calculator.fun5("6", 3)
    with pytest.raises(ValueError):
        calculator.fun6(2, "3")


def test_fun5_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        calculator.fun5(1, 0)
