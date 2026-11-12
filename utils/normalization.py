"""
Normalization utilities for MCDM methods.
Provides various normalization techniques for decision matrices.
"""

import numpy as np
import pandas as pd


def normalize_vector(matrix):
    """
    Vector normalization (Euclidean normalization).
    Each column is normalized by the square root of the sum of squares.

    Parameters:
    -----------
    matrix : pd.DataFrame or np.ndarray
        Decision matrix to normalize

    Returns:
    --------
    pd.DataFrame or np.ndarray
        Normalized matrix
    """
    if isinstance(matrix, pd.DataFrame):
        result = matrix.copy()
        for col in result.columns:
            norm = np.sqrt(np.sum(result[col]**2))
            if norm > 0:
                result[col] = result[col] / norm
        return result
    else:
        result = matrix.copy()
        for j in range(matrix.shape[1]):
            norm = np.sqrt(np.sum(matrix[:, j]**2))
            if norm > 0:
                result[:, j] = result[:, j] / norm
        return result


def normalize_min_max(matrix, criteria_direction):
    """
    Min-Max normalization (0-1 scaling).
    For benefit criteria: (x - min) / (max - min)
    For cost criteria: (max - x) / (max - min)

    Parameters:
    -----------
    matrix : pd.DataFrame or np.ndarray
        Decision matrix to normalize
    criteria_direction : list or dict
        Direction of each criterion ('max' for benefit, 'min' for cost)

    Returns:
    --------
    pd.DataFrame or np.ndarray
        Normalized matrix
    """
    if isinstance(matrix, pd.DataFrame):
        result = matrix.copy()
        for i, col in enumerate(result.columns):
            direction = criteria_direction[i] if isinstance(criteria_direction, list) else criteria_direction[col]
            col_min = result[col].min()
            col_max = result[col].max()
            if col_max == col_min:
                result[col] = 1.0
            else:
                if direction == 'max':
                    result[col] = (result[col] - col_min) / (col_max - col_min)
                else:
                    result[col] = (col_max - result[col]) / (col_max - col_min)
        return result
    else:
        result = matrix.copy()
        for j in range(matrix.shape[1]):
            direction = criteria_direction[j] if isinstance(criteria_direction, list) else criteria_direction[j]
            col_min = np.min(matrix[:, j])
            col_max = np.max(matrix[:, j])
            if col_max == col_min:
                result[:, j] = 1.0
            else:
                if direction == 'max':
                    result[:, j] = (matrix[:, j] - col_min) / (col_max - col_min)
                else:
                    result[:, j] = (col_max - matrix[:, j]) / (col_max - col_min)
        return result


def normalize_sum(matrix, criteria_direction):
    """
    Sum normalization (divides by column sum).
    For benefit criteria: x / sum(x)
    For cost criteria: (1/x) / sum(1/x)

    Parameters:
    -----------
    matrix : pd.DataFrame or np.ndarray
        Decision matrix to normalize
    criteria_direction : list or dict
        Direction of each criterion ('max' for benefit, 'min' for cost)

    Returns:
    --------
    pd.DataFrame or np.ndarray
        Normalized matrix
    """
    if isinstance(matrix, pd.DataFrame):
        result = matrix.copy()
        for i, col in enumerate(result.columns):
            direction = criteria_direction[i] if isinstance(criteria_direction, list) else criteria_direction[col]
            if direction == 'max':
                col_sum = result[col].sum()
                if col_sum > 0:
                    result[col] = result[col] / col_sum
            else:
                inv = 1 / result[col]
                inv_sum = inv.sum()
                if inv_sum > 0:
                    result[col] = inv / inv_sum
        return result
    else:
        result = matrix.copy()
        for j in range(matrix.shape[1]):
            direction = criteria_direction[j] if isinstance(criteria_direction, list) else criteria_direction[j]
            if direction == 'max':
                col_sum = np.sum(matrix[:, j])
                if col_sum > 0:
                    result[:, j] = matrix[:, j] / col_sum
            else:
                inv = 1 / matrix[:, j]
                inv_sum = np.sum(inv)
                if inv_sum > 0:
                    result[:, j] = inv / inv_sum
        return result


def normalize_max(matrix, criteria_direction):
    """
    Max normalization (divides by maximum value).
    For benefit criteria: x / max(x)
    For cost criteria: min(x) / x

    Parameters:
    -----------
    matrix : pd.DataFrame or np.ndarray
        Decision matrix to normalize
    criteria_direction : list or dict
        Direction of each criterion ('max' for benefit, 'min' for cost)

    Returns:
    --------
    pd.DataFrame or np.ndarray
        Normalized matrix
    """
    if isinstance(matrix, pd.DataFrame):
        result = matrix.copy()
        for i, col in enumerate(result.columns):
            direction = criteria_direction[i] if isinstance(criteria_direction, list) else criteria_direction[col]
            col_max = result[col].max()
            col_min = result[col].min()
            if direction == 'max':
                if col_max > 0:
                    result[col] = result[col] / col_max
            else:
                if col_min > 0:
                    result[col] = col_min / result[col]
        return result
    else:
        result = matrix.copy()
        for j in range(matrix.shape[1]):
            direction = criteria_direction[j] if isinstance(criteria_direction, list) else criteria_direction[j]
            col_max = np.max(matrix[:, j])
            col_min = np.min(matrix[:, j])
            if direction == 'max':
                if col_max > 0:
                    result[:, j] = matrix[:, j] / col_max
            else:
                if col_min > 0:
                    result[:, j] = col_min / matrix[:, j]
        return result
