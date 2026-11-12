"""
TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) implementation.
"""

import numpy as np
import pandas as pd


def calculate(normalized_matrix, weights, criteria_direction=None):
    """
    Calculate rankings using TOPSIS method.

    Parameters:
    -----------
    normalized_matrix : pd.DataFrame or np.ndarray
        Normalized decision matrix (alternatives x criteria)
    weights : np.ndarray or list
        Weights of criteria
    criteria_direction : list, optional
        Direction of each criterion ('max' for benefit, 'min' for cost)
        If None, assumes all are benefit criteria

    Returns:
    --------
    dict
        Results containing:
        - scores: TOPSIS scores (closeness coefficient) for each alternative
        - d_plus: distance to positive ideal solution
        - d_minus: distance to negative ideal solution
        - positive_ideal: positive ideal solution
        - negative_ideal: negative ideal solution
        - rankings: ranking of alternatives
    """
    if isinstance(normalized_matrix, pd.DataFrame):
        matrix = normalized_matrix.values
        alternatives = normalized_matrix.index.tolist()
    else:
        matrix = normalized_matrix.copy()
        alternatives = [f'A{i+1}' for i in range(matrix.shape[0])]

    weights = np.array(weights)

    if criteria_direction is None:
        criteria_direction = ['max'] * matrix.shape[1]

    # Step 1: Apply weights to normalized matrix
    weighted_matrix = matrix * weights

    # Step 2: Determine positive and negative ideal solutions
    n_criteria = weighted_matrix.shape[1]

    positive_ideal = np.zeros(n_criteria)
    negative_ideal = np.zeros(n_criteria)

    for j in range(n_criteria):
        if criteria_direction[j] == 'max':
            positive_ideal[j] = np.max(weighted_matrix[:, j])
            negative_ideal[j] = np.min(weighted_matrix[:, j])
        else:
            positive_ideal[j] = np.min(weighted_matrix[:, j])
            negative_ideal[j] = np.max(weighted_matrix[:, j])

    # Step 3: Calculate distances to ideal solutions
    m = weighted_matrix.shape[0]
    d_plus = np.zeros(m)
    d_minus = np.zeros(m)

    for i in range(m):
        d_plus[i] = np.sqrt(np.sum((weighted_matrix[i, :] - positive_ideal) ** 2))
        d_minus[i] = np.sqrt(np.sum((weighted_matrix[i, :] - negative_ideal) ** 2))

    # Step 4: Calculate closeness coefficient
    # CC_i = d_minus / (d_plus + d_minus)
    scores = d_minus / (d_plus + d_minus)

    # Create ranking (higher is better)
    sorted_indices = np.argsort(-scores)
    rankings = {}
    for rank, idx in enumerate(sorted_indices, 1):
        rankings[alternatives[idx]] = rank

    return {
        'scores': dict(zip(alternatives, scores)),
        'd_plus': dict(zip(alternatives, d_plus)),
        'd_minus': dict(zip(alternatives, d_minus)),
        'positive_ideal': positive_ideal,
        'negative_ideal': negative_ideal,
        'rankings': rankings
    }


def calculate_from_raw(matrix, weights, criteria_direction):
    """
    Calculate TOPSIS rankings from raw (non-normalized) matrix.

    Parameters:
    -----------
    matrix : pd.DataFrame or np.ndarray
        Raw decision matrix
    weights : np.ndarray or list
        Weights of criteria
    criteria_direction : list
        Direction of each criterion ('max' for benefit, 'min' for cost)

    Returns:
    --------
    dict
        TOPSIS results
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.normalization import normalize_vector

    # Normalize using vector normalization
    normalized = normalize_vector(matrix)

    # Calculate TOPSIS
    return calculate(normalized, weights, criteria_direction)
