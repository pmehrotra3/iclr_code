"""Discover and run every test under code/unit_tests (no pytest needed):
    python code/unit_tests/run.py"""
import os
import sys
import unittest

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(here))

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(here, pattern="test_*.py", top_level_dir=os.path.dirname(here))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
