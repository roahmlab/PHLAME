import ipdb
import numpy as np
import unittest

from phlame.utils import apply_func_to_vector
from phlame.utils import interpolate_matrix


class TestUtilsFunctions(unittest.TestCase):
    def setUp(self):
        self.values = np.array([1, 2, 3])
        self.matrix = np.array([[1, 2], [3, 4], [5, 6]])
        self.times = np.array([0, 1, 2])
        self.query_times = np.array([0.5, 1.5])
        self.flipped_query_times = np.array([1.5, 0.5])

    def test_apply_func_to_vector_row_result(self):
        print("####### Running test_utils tests")
        # Test if the function correctly applies the given function to the input vector
        def test_func(x):
            return x ** 2

        result = apply_func_to_vector(test_func, self.values)
        expected_result = np.array([1, 4, 9], dtype=np.double)[:, np.newaxis]
        np.testing.assert_array_equal(result, expected_result)

    def test_apply_func_to_vector_matrix_result(self):
        def test_func(x):
            return np.array([x**2, x**3])

        result = apply_func_to_vector(test_func, self.values)
        expected_result = np.array([[1, 1], [4, 8], [9, 27]])
        np.testing.assert_array_equal(result, expected_result)

    def test_interpolate_matrix(self):
        # Test if the function correctly interpolates the matrix over time
        result = interpolate_matrix(self.matrix, self.times, self.query_times)
        expected_result = np.array([[2, 3], [4, 5]])
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_interpolate_matrix_flipped_query_times(self):
        result = interpolate_matrix(self.matrix, self.times, self.flipped_query_times)
        expected_result = np.array([[4, 5], [2, 3]])
        np.testing.assert_array_almost_equal(result, expected_result)

if __name__ == "__main__":
    unittest.main()