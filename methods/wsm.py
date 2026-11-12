"""
Weighted Sum Method (WSM) implementation.
Simple additive ranking method.
"""

import numpy as np
import pandas as pd


def calculate(normalized_matrix, weights):
    """
    Calculate rankings using Weighted Sum Method.

    Parameters:
    -----------
    normalized_matrix : pd.DataFrame or np.ndarray
        Normalized decision matrix (alternatives x criteria)
    weights : np.ndarray or list
        Weights of criteria

    Returns:
    --------
    dict
        Results containing:
        - scores: WSM scores for each alternative
        - rankings: ranking of alternatives
    """
    if isinstance(normalized_matrix, pd.DataFrame):
        matrix = normalized_matrix.values
        alternatives = normalized_matrix.index.tolist()
    else:
        matrix = normalized_matrix.copy()
        alternatives = [f'A{i+1}' for i in range(matrix.shape[0])]

    weights = np.array(weights)

    # Calculate WSM scores: sum of weighted normalized values
    scores = np.dot(matrix, weights)

    # Create ranking (higher is better)
    sorted_indices = np.argsort(-scores)
    rankings = {}
    for rank, idx in enumerate(sorted_indices, 1):
        rankings[alternatives[idx]] = rank

    return {
        'scores': dict(zip(alternatives, scores)),
        'rankings': rankings
    }
