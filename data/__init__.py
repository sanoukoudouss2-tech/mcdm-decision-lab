"""
Data Module
Contains example datasets and sample data for MCDM methods.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.examples as examples

__all__ = ['examples']
