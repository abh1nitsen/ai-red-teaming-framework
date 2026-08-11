"""
tests/conftest.py
==================
Pytest configuration: adds project root to sys.path so all tests can import
src.* without installing the package. Auto-loaded by pytest before any test.
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
