"""Unittest suite for src/calculator.py.

The TestCalculator class is the course lab's original suite
(Labs/Github_Labs/Lab1/test/test_unittest.py), with the same cases.
The TestAddedFunctions class was added for this submission to cover
fun5/fun6 and the input validation in the unittest style as well.
"""
import os
import sys
import unittest

# Get the path to the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src import calculator


class TestCalculator(unittest.TestCase):

    def test_fun1(self):
        self.assertEqual(calculator.fun1(2, 3), 5)
        self.assertEqual(calculator.fun1(5, 0), 5)
        self.assertEqual(calculator.fun1(-1, 1), 0)
        self.assertEqual(calculator.fun1(-1, -1), -2)

    def test_fun2(self):
        self.assertEqual(calculator.fun2(2, 3), -1)
        self.assertEqual(calculator.fun2(5, 0), 5)
        self.assertEqual(calculator.fun2(-1, 1), -2)
        self.assertEqual(calculator.fun2(-1, -1), 0)

    def test_fun3(self):
        self.assertEqual(calculator.fun3(2, 3), 6)
        self.assertEqual(calculator.fun3(5, 0), 0)
        self.assertEqual(calculator.fun3(-1, 1), -1)
        self.assertEqual(calculator.fun3(-1, -1), 1)

    def test_fun4(self):
        self.assertEqual(calculator.fun4(2, 3, 5), 10)
        self.assertEqual(calculator.fun4(5, 0, -1), 4)
        self.assertEqual(calculator.fun4(-1, -1, -1), -3)
        self.assertEqual(calculator.fun4(-1, -1, 100), 98)


class TestAddedFunctions(unittest.TestCase):

    def test_fun5(self):
        self.assertEqual(calculator.fun5(6, 3), 2)
        self.assertEqual(calculator.fun5(5, 2), 2.5)
        self.assertEqual(calculator.fun5(-4, 2), -2)
        self.assertEqual(calculator.fun5(0, 5), 0)

    def test_fun6(self):
        self.assertEqual(calculator.fun6(2, 3), 8)
        self.assertEqual(calculator.fun6(5, 0), 1)
        self.assertEqual(calculator.fun6(9, 0.5), 3)
        self.assertEqual(calculator.fun6(2, -1), 0.5)

    def test_invalid_inputs_raise_value_error(self):
        with self.assertRaises(ValueError):
            calculator.fun1("2", 3)
        with self.assertRaises(ValueError):
            calculator.fun3("2", "3")
        with self.assertRaises(ValueError):
            calculator.fun5("6", 3)
        with self.assertRaises(ValueError):
            calculator.fun6(2, "3")

    def test_fun5_divide_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            calculator.fun5(1, 0)


if __name__ == "__main__":
    unittest.main()
