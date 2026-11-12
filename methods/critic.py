"""
CRITIC (CRiteria Importance Through Intercriteria Correlation) method implementation.
Objective weighting method considering both standard deviation and correlation.
"""

import numpy as np
import pandas as pd


def calculate_weights(matrix):
    """
    Calculate weights using CRITIC method.

    Parameters:
    -----------
    matrix : pd.DataFrame or np.ndarray
        Decision matrix (alternatives x criteria)

    Returns:
    --------
    dict
        Results containing:
        - weights: calculated weights
        - std_dev: standard deviation for each criterion
        - correlation_matrix: correlation matrix between criteria
        - C_j: C_j values for each criterion
    """
    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.values

    m, n = matrix.shape  # m alternatives, n criteria

    # Step 1: Normalize the matrix using min-max normalization
    normalized = np.zeros_like(matrix)
    for j in range(n):
        col_min = np.min(matrix[:, j])
        col_max = np.max(matrix[:, j])
        if col_max == col_min:
            normalized[:, j] = 1.0
        else:
            normalized[:, j] = (matrix[:, j] - col_min) / (col_max - col_min)

    # Step 2: Calculate standard deviation for each criterion
    std_dev = np.std(normalized, axis=0, ddof=1)

    # Step 3: Calculate correlation matrix
    correlation_matrix = np.corrcoef(normalized.T)

    # Step 4: Calculate C_j for each criterion
    C_j = np.zeros(n)
    for j in range(n):
        # Sum of (1 - |correlation|) for criterion j
        sum_anti_correlation = np.sum(1 - np.abs(correlation_matrix[j, :]))
        C_j[j] = std_dev[j] * sum_anti_correlation

    # Step 5: Calculate weights
    weights = C_j / np.sum(C_j)

    return {
        'weights': weights,
        'std_dev': std_dev,
        'correlation_matrix': correlation_matrix,
        'C_j': C_j
    }


def calculate_weights_with_criteria_direction(matrix, criteria_direction):
    """
    Calculate weights using CRITIC method with criteria direction consideration.

    Parameters:
    -----------
    matrix : pd.DataFrame or np.ndarray
        Decision matrix (alternatives x criteria)
    criteria_direction : list
        Direction of each criterion ('max' for benefit, 'min' for cost)

    Returns:
    --------
    dict
        Results containing weights and intermediate values
    """
    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.values

    m, n = matrix.shape

    # Normalize with criteria direction
    normalized = np.zeros_like(matrix)
    for j in range(n):
        col_min = np.min(matrix[:, j])
        col_max = np.max(matrix[:, j])
        if col_max == col_min:
            normalized[:, j] = 1.0
        else:
            if criteria_direction[j] == 'max':
                normalized[:, j] = (matrix[:, j] - col_min) / (col_max - col_min)
            else:
                normalized[:, j] = (col_max - matrix[:, j]) / (col_max - col_min)

    # Calculate standard deviation
    std_dev = np.std(normalized, axis=0, ddof=1)

    # Calculate correlation matrix
    correlation_matrix = np.corrcoef(normalized.T)

    # Calculate C_j
    C_j = np.zeros(n)
    for j in range(n):
        sum_anti_correlation = np.sum(1 - np.abs(correlation_matrix[j, :]))
        C_j[j] = std_dev[j] * sum_anti_correlation

    # Calculate weights
    weights = C_j / np.sum(C_j)

    return {
        'weights': weights,
        'std_dev': std_dev,
        'correlation_matrix': correlation_matrix,
        'C_j': C_j,
        'normalized_matrix': normalized
    }
