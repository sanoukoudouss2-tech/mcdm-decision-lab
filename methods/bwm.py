"""
Best-Worst Method (BWM) implementation.
Based on Rezaei (2015) method for weighting criteria with fewer comparisons than AHP.
"""

import numpy as np
from scipy.optimize import linprog
import pandas as pd


def calculate_weights(best_criterion, worst_criterion, best_to_others, others_to_worst, criteria_names=None):
    """
    Calculate weights using BWM method with linear programming.

    Parameters:
    -----------
    best_criterion : str or int
        Index or name of the best criterion
    worst_criterion : str or int
        Index or name of the worst criterion
    best_to_others : dict
        Dictionary of {criterion: value} indicating how much better the best criterion is
    others_to_worst : dict
        Dictionary of {criterion: value} indicating how much better each criterion is than worst
    criteria_names : list, optional
        List of criterion names

    Returns:
    --------
    dict
        Results containing:
        - weights: calculated weights
        - xi: optimal inconsistency value
        - constraints_met: boolean
    """
    n = len(best_to_others)

    # Convert names to indices if necessary
    if isinstance(best_criterion, str) and criteria_names is not None:
        best_idx = criteria_names.index(best_criterion)
    else:
        best_idx = best_criterion

    if isinstance(worst_criterion, str) and criteria_names is not None:
        worst_idx = criteria_names.index(worst_criterion)
    else:
        worst_idx = worst_criterion

    # Build linear programming problem
    # Minimize xi
    # Subject to:
    # |w_best - a_Bj * w_j| <= xi for all j
    # |w_j - a_jW * w_worst| <= xi for all j
    # sum(w_j) = 1
    # w_j >= 0

    # Number of variables: n weights + xi
    c = np.zeros(n + 1)
    c[-1] = 1  # Minimize xi

    # Inequality constraints
    A_ub = []
    b_ub = []

    # Best-to-others constraints
    for j in range(n):
        if j == best_idx:
            continue

        a_Bj = best_to_others[j] if isinstance(best_to_others, dict) else best_to_others[list(best_to_others.keys())[j]]

        # w_best - a_Bj * w_j - xi <= 0
        row1 = np.zeros(n + 1)
        row1[best_idx] = 1
        row1[j] = -a_Bj
        row1[-1] = -1
        A_ub.append(row1)
        b_ub.append(0)

        # -w_best + a_Bj * w_j - xi <= 0
        row2 = np.zeros(n + 1)
        row2[best_idx] = -1
        row2[j] = a_Bj
        row2[-1] = -1
        A_ub.append(row2)
        b_ub.append(0)

    # Others-to-worst constraints
    for j in range(n):
        if j == worst_idx:
            continue

        a_jW = others_to_worst[j] if isinstance(others_to_worst, dict) else others_to_worst[list(others_to_worst.keys())[j]]

        # w_j - a_jW * w_worst - xi <= 0
        row1 = np.zeros(n + 1)
        row1[j] = 1
        row1[worst_idx] = -a_jW
        row1[-1] = -1
        A_ub.append(row1)
        b_ub.append(0)

        # -w_j + a_jW * w_worst - xi <= 0
        row2 = np.zeros(n + 1)
        row2[j] = -1
        row2[worst_idx] = a_jW
        row2[-1] = -1
        A_ub.append(row2)
        b_ub.append(0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Equality constraint: sum of weights = 1
    A_eq = np.zeros(n + 1)
    A_eq[:n] = 1
    b_eq = np.array([1])

    # Bounds: weights >= 0, xi unrestricted
    bounds = [(0, None)] * n + [(None, None)]

    # Solve linear program
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if not result.success:
        raise ValueError(f"Linear programming failed: {result.message}")

    weights = result.x[:n]
    xi = result.x[-1]

    return {
        'weights': weights,
        'xi': xi,
        'success': result.success
    }


def calculate_weights_from_vectors(best_idx, worst_idx, best_to_others_vector, others_to_worst_vector):
    """
    Calculate weights from BWM vectors (simplified interface).

    Parameters:
    -----------
    best_idx : int
        Index of best criterion (0-based)
    worst_idx : int
        Index of worst criterion (0-based)
    best_to_others_vector : list or np.ndarray
        Vector of comparisons from best to others
    others_to_worst_vector : list or np.ndarray
        Vector of comparisons from others to worst

    Returns:
    --------
    dict
        Results containing weights and inconsistency value
    """
    n = len(best_to_others_vector)

    best_to_others = {i: best_to_others_vector[i] for i in range(n)}
    others_to_worst = {i: others_to_worst_vector[i] for i in range(n)}

    return calculate_weights(best_idx, worst_idx, best_to_others, others_to_worst)


def check_consistency(xi):
    """
    Check consistency of BWM weights based on xi value.

    Parameters:
    -----------
    xi : float
        Inconsistency value from BWM

    Returns:
    --------
    bool
        True if xi is close to 0 (consistent), False otherwise
    str
        Consistency level description
    """
    if xi < 0.05:
        return True, "Excellent consistency"
    elif xi < 0.1:
        return True, "Good consistency"
    elif xi < 0.2:
        return False, "Acceptable consistency"
    else:
        return False, "Poor consistency"
