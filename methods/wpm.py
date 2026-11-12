"""
Weighted Product Method (WPM) implementation.
Multiplicative ranking method.
"""

import numpy as np
import pandas as pd


def calculate(normalized_matrix, weights):
    """
    Calculate rankings using Weighted Product Method.

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
        - scores: WPM scores for each alternative
        - rankings: ranking of alternatives
    """
    if isinstance(normalized_matrix, pd.DataFrame):
        matrix = normalized_matrix.values
        alternatives = normalized_matrix.index.tolist()
    else:
        matrix = normalized_matrix.copy()
        alternatives = [f'A{i+1}' for i in range(matrix.shape[0])]

    weights = np.array(weights)

    # Add small constant to avoid log(0) and product of zeros
    epsilon = 1e-10
    matrix = matrix + epsilon

    # Calculate WPM scores: product of (normalized_value ^ weight)
    scores = np.prod(matrix ** weights, axis=1)

    # Create ranking (higher is better)
    sorted_indices = np.argsort(-scores)
    rankings = {}
    for rank, idx in enumerate(sorted_indices, 1):
        rankings[alternatives[idx]] = rank

    return {
        'scores': dict(zip(alternatives, scores)),
        'rankings': rankings
    }
