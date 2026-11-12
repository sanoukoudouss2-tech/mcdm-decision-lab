"""
Entropy method implementation.
Objective weighting method based on information entropy.
"""

import numpy as np
import pandas as pd


def calculate_weights(matrix):
    """
    Calculate weights using entropy method.

    Parameters:
    -----------
    matrix : pd.DataFrame or np.ndarray
        Decision matrix (alternatives x criteria)

    Returns:
    --------
    dict
        Results containing:
        - weights: calculated weights
        - entropy: entropy values for each criterion
        - diversification: diversification degree for each criterion
    """
    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.values

    m, n = matrix.shape  # m alternatives, n criteria

    # Normalize the matrix to ensure all values are positive
    # Using min-max normalization for each column
    normalized = np.zeros_like(matrix)
    for j in range(n):
        col_min = np.min(matrix[:, j])
        col_max = np.max(matrix[:, j])
        if col_max == col_min:
            normalized[:, j] = 1.0
        else:
            normalized[:, j] = (matrix[:, j] - col_min) / (col_max - col_min)

    # Add small constant to avoid log(0)
    epsilon = 1e-10
    normalized = normalized + epsilon

    # Calculate p_ij for each element
    column_sums = np.sum(normalized, axis=0)
    p = normalized / column_sums

    # Calculate entropy for each criterion
    k = 1 / np.log(m)  # Normalization constant
    entropy = -k * np.sum(p * np.log(p), axis=0)

    # Calculate diversification degree
    diversification = 1 - entropy

    # Calculate weights
    weights = diversification / np.sum(diversification)

    return {
        'weights': weights,
        'entropy': entropy,
        'diversification': diversification
    }


def calculate_weights_direct(matrix):
    """
    Calculate weights using entropy method (direct approach without normalization).

    Parameters:
    -----------
    matrix : pd.DataFrame or np.ndarray
        Decision matrix (alternatives x criteria)

    Returns:
    --------
    dict
        Results containing weights and entropy values
    """
    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.values

    m, n = matrix.shape

    # Ensure all values are positive
    matrix = np.abs(matrix)

    # Calculate column sums
    column_sums = np.sum(matrix, axis=0)

    # Calculate p_ij
    p = matrix / column_sums

    # Add small constant to avoid log(0)
    epsilon = 1e-10
    p = p + epsilon

    # Calculate entropy
    k = 1 / np.log(m)
    entropy = -k * np.sum(p * np.log(p), axis=0)

    # Calculate diversification
    diversification = 1 - entropy

    # Calculate weights
    weights = diversification / np.sum(diversification)

    return {
        'weights': weights,
        'entropy': entropy,
        'diversification': diversification
    }
