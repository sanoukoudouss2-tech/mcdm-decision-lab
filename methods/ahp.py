"""
Analytic Hierarchy Process (AHP) method implementation.
Based on Saaty's method for weighting criteria and ranking alternatives.
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.validation import validate_pairwise_matrix, calculate_consistency_index, check_consistency_ratio


def calculate_weights(pairwise_matrix):
    """
    Calculate weights from pairwise comparison matrix using eigenvector method.

    Parameters:
    -----------
    pairwise_matrix : pd.DataFrame or np.ndarray
        Pairwise comparison matrix

    Returns:
    --------
    dict
        Results containing:
        - weights: normalized weights
        - eigenvalue: maximum eigenvalue
        - ci: consistency index
        - cr: consistency ratio
        - consistent: boolean indicating if CR < 0.1
    """
    # Validate matrix
    is_valid, message = validate_pairwise_matrix(pairwise_matrix)
    if not is_valid:
        raise ValueError(message)

    if isinstance(pairwise_matrix, pd.DataFrame):
        matrix = pairwise_matrix.values
    else:
        matrix = pairwise_matrix.copy()

    n = matrix.shape[0]

    # Method 1: Approximate eigenvector method (geometric mean)
    # Calculate geometric mean of each row
    geometric_means = np.prod(matrix, axis=1) ** (1/n)

    # Normalize to get weights
    weights = geometric_means / np.sum(geometric_means)

    # Calculate maximum eigenvalue
    weighted_sum = np.dot(matrix, weights)
    eigenvalue = np.mean(weighted_sum / weights)

    # Calculate consistency indices
    ci = calculate_consistency_index(eigenvalue, n)
    consistent, cr = check_consistency_ratio(ci, n)

    return {
        'weights': weights,
        'eigenvalue': eigenvalue,
        'ci': ci,
        'cr': cr,
        'consistent': consistent
    }


def calculate_weights_approximate(pairwise_matrix):
    """
    Calculate weights using the approximate method from the course (column normalization).

    Parameters:
    -----------
    pairwise_matrix : pd.DataFrame or np.ndarray
        Pairwise comparison matrix

    Returns:
    --------
    dict
        Results containing weights and consistency measures
    """
    # Validate matrix
    is_valid, message = validate_pairwise_matrix(pairwise_matrix)
    if not is_valid:
        raise ValueError(message)

    if isinstance(pairwise_matrix, pd.DataFrame):
        matrix = pairwise_matrix.values
    else:
        matrix = pairwise_matrix.copy()

    n = matrix.shape[0]

    # Step 1: Calculate sum of each column
    column_sums = np.sum(matrix, axis=0)

    # Step 2: Divide each element by column sum
    normalized = matrix / column_sums

    # Step 3: Calculate average of each row to get weights
    weights = np.mean(normalized, axis=1)

    # Calculate maximum eigenvalue
    weighted_sum = np.dot(matrix, weights)
    eigenvalue = np.sum(column_sums * weights)

    # Calculate consistency indices
    ci = calculate_consistency_index(eigenvalue, n)
    consistent, cr = check_consistency_ratio(ci, n)

    return {
        'weights': weights,
        'eigenvalue': eigenvalue,
        'ci': ci,
        'cr': cr,
        'consistent': consistent
    }


def rank_alternatives(pairwise_matrices, criteria_weights):
    """
    Rank alternatives using AHP based on pairwise comparison matrices for each criterion.

    Parameters:
    -----------
    pairwise_matrices : list of pd.DataFrame or np.ndarray
        List of pairwise comparison matrices, one for each criterion
    criteria_weights : np.ndarray
        Weights of criteria

    Returns:
    --------
    dict
        Results containing:
        - local_priorities: local priorities for each criterion
        - global_priorities: final scores for each alternative
        - rankings: ranking of alternatives
    """
    n_alternatives = pairwise_matrices[0].shape[0]
    n_criteria = len(pairwise_matrices)

    # Calculate local priorities for each criterion
    local_priorities = np.zeros((n_alternatives, n_criteria))

    for i, matrix in enumerate(pairwise_matrices):
        result = calculate_weights_approximate(matrix)
        local_priorities[:, i] = result['weights']

    # Calculate global priorities (final scores)
    global_priorities = np.dot(local_priorities, criteria_weights)

    # Create ranking
    sorted_indices = np.argsort(-global_priorities)
    rankings = {}
    for rank, idx in enumerate(sorted_indices, 1):
        rankings[f'A{idx+1}'] = rank

    return {
        'local_priorities': local_priorities,
        'global_priorities': global_priorities,
        'rankings': rankings
    }


def create_pairwise_matrix(n_criteria, comparisons):
    """
    Create a pairwise comparison matrix from a dictionary of comparisons.

    Parameters:
    -----------
    n_criteria : int
        Number of criteria
    comparisons : dict
        Dictionary of {(i, j): value} where i, j are 0-indexed and value is the comparison

    Returns:
    --------
    np.ndarray
        Pairwise comparison matrix
    """
    matrix = np.eye(n_criteria)

    for (i, j), value in comparisons.items():
        matrix[i, j] = value
        matrix[j, i] = 1 / value

    return matrix
