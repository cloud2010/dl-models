"""Compatibility entry point for actual regression assertions.

Run the full suite with: python -m unittest discover -s tests -v
"""

if __name__ == "__main__":
    import unittest
    unittest.main(module="tests.test_experiments", verbosity=2)
