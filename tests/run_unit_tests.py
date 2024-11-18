import unittest
import os

def run_all_tests():
    # Discover all test cases in the 'tests' folder
    loader = unittest.TestLoader()
    test_dir = os.path.join(os.path.dirname(__file__))
    suite = loader.discover(start_dir=test_dir, pattern='test*.py')

    # Run the test suite
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == "__main__":
    run_all_tests()