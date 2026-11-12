"""
Validation utilities for MCDM methods.
Provides functions to validate decision matrices, weights, and pairwise comparisons.
"""

import numpy as np
import pandas as pd


def validate_decision_matrix(matrix):
    """
    Validate that the decision matrix is properly formatted.

    Parameters:
    -----------
    matrix : pd.DataFrame or np.ndarray
        Decision matrix to validate

    Returns:
    --------
    bool
        True if valid, False otherwise
    str
        Error message if invalid
    """
    if matrix is None:
        return False, "Decision matrix is None"

    if isinstance(matrix, pd.DataFrame):
        if matrix.empty:
            return False, "Decision matrix is empty"
        if matrix.isnull().any().any():
            return False, "Decision matrix contains NaN values"
        if (matrix == 0).all().all():
            return False, "Decision matrix contains only zeros"
    elif isinstance(matrix, np.ndarray):
        if matrix.size == 0:
            return False, "Decision matrix is empty"
        if np.isnan(matrix).any():
            return False, "Decision matrix contains NaN values"
        if np.all(matrix == 0):
            return False, "Decision matrix contains only zeros"
    else:
        return False, "Decision matrix must be a pandas DataFrame or numpy array"

    return True, "Valid"


def validate_weights(weights, n_criteria):
    """
    Validate that weights are properly formatted.

    Parameters:
    -----------
    weights : list, np.ndarray, or pd.Series
        Weights to validate
    n_criteria : int
        Number of criteria

    Returns:
    --------
    bool
        True if valid, False otherwise
    str
        Error message if invalid
    """
    if weights is None:
        return False, "Weights are None"

    if len(weights) != n_criteria:
        return False, f"Number of weights ({len(weights)}) does not match number of criteria ({n_criteria})"

    weights = np.array(weights)

    if np.isnan(weights).any():
        return False, "Weights contain NaN values"

    if (weights < 0).any():
        return False, "Weights must be non-negative"

    if np.abs(np.sum(weights) - 1.0) > 0.01:
        return False, f"Weights must sum to 1.0 (current sum: {np.sum(weights):.4f})"

    return True, "Valid"


def validate_pairwise_matrix(matrix):
    """
    Validate a pairwise comparison matrix for AHP.
    Checks for reciprocity and positive values.

    Parameters:
    -----------
    matrix : pd.DataFrame or np.ndarray
        Pairwise comparison matrix

    Returns:
    --------
    bool
        True if valid, False otherwise
    str
        Error message if invalid
    """
    if matrix is None:
        return False, "Pairwise matrix is None"

    if isinstance(matrix, pd.DataFrame):
        if matrix.empty:
            return False, "Pairwise matrix is empty"
        if matrix.shape[0] != matrix.shape[1]:
            return False, "Pairwise matrix must be square"
        if matrix.isnull().any().any():
            return False, "Pairwise matrix contains NaN values"
        if (matrix <= 0).any().any():
            return False, "Pairwise matrix must contain only positive values"

        # Check reciprocity
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if i != j:
                    if abs(matrix.iloc[i, j] - 1/matrix.iloc[j, i]) > 0.01:
                        return False, f"Pairwise matrix is not reciprocal at ({i}, {j})"
    elif isinstance(matrix, np.ndarray):
        if matrix.size == 0:
            return False, "Pairwise matrix is empty"
        if matrix.shape[0] != matrix.shape[1]:
            return False, "Pairwise matrix must be square"
        if np.isnan(matrix).any():
            return False, "Pairwise matrix contains NaN values"
        if (matrix <= 0).any():
            return False, "Pairwise matrix must contain only positive values"

        # Check reciprocity
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i != j:
                    if abs(matrix[i, j] - 1/matrix[j, i]) > 0.01:
                        return False, f"Pairwise matrix is not reciprocal at ({i}, {j})"
    else:
        return False, "Pairwise matrix must be a pandas DataFrame or numpy array"

    return True, "Valid"


def check_consistency_ratio(ci, n):
    """
    Check if consistency ratio is acceptable (< 0.1).

    Parameters:
    -----------
    ci : float
        Consistency index
    n : int
        Number of criteria

    Returns:
    --------
    bool
        True if CR < 0.1, False otherwise
    float
        Consistency ratio
    """
    # Random index values (Saaty)
    ri_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.56}

    ri = ri_dict.get(n, 1.56)  # Default to 1.56 for n > 10

    if ri == 0:
        return True, 0.0

    cr = ci / ri
    return cr < 0.1, cr


def calculate_consistency_index(eigenvalue, n):
    """
    Calculate consistency index for AHP.

    Parameters:
    -----------
    eigenvalue : float
        Maximum eigenvalue
    n : int
        Number of criteria

    Returns:
    --------
    float
        Consistency index
    """
    if n <= 1:
        return 0.0
    return (eigenvalue - n) / (n - 1)
