"""
conftest.py — PaySense Guardian test configuration
Ensures src/ is on the Python path regardless of where pytest is invoked from.
Run from PaySense-ML-Backend/:  pytest tests/ -v
"""
import os
import sys

# Make src/ importable from any working directory
sys.path.insert(0, os.path.dirname(__file__))
