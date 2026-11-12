"""
PROMETHEE (Preference Ranking Organization METHod for Enrichment Evaluations) implementation.
Outranking method for multi-criteria decision making.
"""

import numpy as np
import pandas as pd


def calculate(matrix, weights, criteria_direction, preference_functions=None, p=None, q=None):
    """
    Calculate rankings using PROMETHEE II method.

    Parameters:
    -----------
    matrix : pd.DataFrame or np.ndarray
        Decision matrix (alternatives x criteria)
    weights : np.ndarray or list
        Weights of criteria
    criteria_direction : list
        Direction of each criterion ('max' for benefit, 'min' for cost)
    preference_functions : list, optional
        Preference function for each criterion
        Options: 'usual', 'linear', 'quasi', 'level', 'v-shape', 'gaussian'
        Default: 'linear' for all
    p : list, optional
        Preference threshold for each criterion
    q : list, optional
        Indifference threshold for each criterion

    Returns:
    --------
    dict
        Results containing:
        - phi_plus: positive outranking flows
        - phi_minus: negative outranking flows
        - phi: net outranking flows
        - rankings: ranking of alternatives
        - preference_matrix: preference matrix
    """
    if isinstance(matrix, pd.DataFrame):
        matrix_val = matrix.values
        alternatives = matrix.index.tolist()
    else:
        matrix_val = matrix.copy()
        alternatives = [f'A{i+1}' for i in range(matrix.shape[0])]

    weights = np.array(weights)
    m, n = matrix_val.shape

    # Set default preference function if not provided
    if preference_functions is None:
        preference_functions = ['linear'] * n

    # Set default thresholds if not provided
    if p is None:
        p = [np.std(matrix_val[:, j]) for j in range(n)]
    if q is None:
        q = [0.5 * np.std(matrix_val[:, j]) for j in range(n)]

    # Calculate preference matrix
    preference_matrix = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            if i == j:
                continue

            for k in range(n):
                # Calculate difference
                diff = matrix_val[i, k] - matrix_val[j, k]

                # Adjust for cost criteria
                if criteria_direction[k] == 'min':
                    diff = -diff

                # Calculate preference degree
                pref_degree = _preference_function(diff, preference_functions[k], p[k], q[k])

                # Add weighted preference
                preference_matrix[i, j] += weights[k] * pref_degree

    # Calculate positive and negative flows
    phi_plus = np.sum(preference_matrix, axis=1) / (m - 1)
    phi_minus = np.sum(preference_matrix, axis=0) / (m - 1)

    # Calculate net flow
    phi = phi_plus - phi_minus

    # Create ranking (higher is better)
    sorted_indices = np.argsort(-phi)
    rankings = {}
    for rank, idx in enumerate(sorted_indices, 1):
        rankings[alternatives[idx]] = rank

    return {
        'phi_plus': dict(zip(alternatives, phi_plus)),
        'phi_minus': dict(zip(alternatives, phi_minus)),
        'phi': dict(zip(alternatives, phi)),
        'rankings': rankings,
        'preference_matrix': preference_matrix
    }


def _preference_function(diff, func_type, p, q):
    """
    Calculate preference degree based on difference and function type.

    Parameters:
    -----------
    diff : float
        Difference between two alternatives on a criterion
    func_type : str
        Type of preference function
    p : float
        Preference threshold
    q : float
        Indifference threshold

    Returns:
    --------
    float
        Preference degree (0 to 1)
    """
    if func_type == 'usual':
        # Usual criterion
        if diff > 0:
            return 1.0
        else:
            return 0.0

    elif func_type == 'linear':
        # Linear preference
        if diff <= 0:
            return 0.0
        elif diff < p:
            return diff / p
        else:
            return 1.0

    elif func_type == 'quasi':
        # Quasi-criterion
        if diff <= q:
            return 0.0
        else:
            return 1.0

    elif func_type == 'level':
        # Level criterion
        if diff <= q:
            return 0.0
        elif diff < p:
            return 0.5
        else:
            return 1.0

    elif func_type == 'v-shape':
        # V-shape criterion
        if diff <= 0:
            return 0.0
        elif diff < p:
            return diff / p
        else:
            return 1.0

    elif func_type == 'gaussian':
        # Gaussian criterion
        if diff <= 0:
            return 0.0
        else:
            return 1 - np.exp(-diff**2 / (2 * p**2))

    else:
        # Default to linear
        if diff <= 0:
            return 0.0
        elif diff < p:
            return diff / p
        else:
            return 1.0
