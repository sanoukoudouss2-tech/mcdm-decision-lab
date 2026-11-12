"""
VIKOR (VIseKriterijumska Optimizacija I Kompromisno Resenje) implementation.
Compromise ranking method balancing group utility and individual regret.
"""

import numpy as np
import pandas as pd


def calculate(normalized_matrix, weights, criteria_direction, v=0.5):
    """
    Calculate rankings using VIKOR method.

    Parameters:
    -----------
    normalized_matrix : pd.DataFrame or np.ndarray
        Normalized decision matrix (alternatives x criteria)
    weights : np.ndarray or list
        Weights of criteria
    criteria_direction : list
        Direction of each criterion ('max' for benefit, 'min' for cost)
    v : float
        Weight of strategy of maximum group utility (default: 0.5)
        v = 0.5 gives equal weight to S and R
        v > 0.5 emphasizes group utility
        v < 0.5 emphasizes individual regret

    Returns:
    --------
    dict
        Results containing:
        - scores: VIKOR Q scores for each alternative
        - S: group utility measures
        - R: individual regret measures
        - S_star: minimum S
        - S_minus: maximum S
        - R_star: minimum R
        - R_minus: maximum R
        - rankings: ranking of alternatives
        - compromise_set: list of compromise solutions
    """
    if isinstance(normalized_matrix, pd.DataFrame):
        matrix = normalized_matrix.values
        alternatives = normalized_matrix.index.tolist()
    else:
        matrix = normalized_matrix.copy()
        alternatives = [f'A{i+1}' for i in range(matrix.shape[0])]

    weights = np.array(weights)

    m, n = matrix.shape

    # Step 1: Determine best (f*) and worst (f-) values for each criterion
    f_star = np.zeros(n)
    f_minus = np.zeros(n)

    for j in range(n):
        if criteria_direction[j] == 'max':
            f_star[j] = np.max(matrix[:, j])
            f_minus[j] = np.min(matrix[:, j])
        else:
            f_star[j] = np.min(matrix[:, j])
            f_minus[j] = np.max(matrix[:, j])

    # Step 2: Calculate S (group utility) and R (individual regret)
    S = np.zeros(m)
    R = np.zeros(m)

    for i in range(m):
        # S_i = sum of weighted normalized distances
        S[i] = np.sum(weights * (f_star - matrix[i, :]) / (f_star - f_minus))

        # R_i = maximum weighted normalized distance
        R[i] = np.max(weights * (f_star - matrix[i, :]) / (f_star - f_minus))

    # Step 3: Calculate S_star, S_minus, R_star, R_minus
    S_star = np.min(S)
    S_minus = np.max(S)
    R_star = np.min(R)
    R_minus = np.max(R)

    # Step 4: Calculate Q (compromise ranking)
    # Q_i = v * (S_i - S_star) / (S_minus - S_star) + (1 - v) * (R_i - R_star) / (R_minus - R_star)
    if S_minus == S_star:
        S_part = np.zeros(m)
    else:
        S_part = (S - S_star) / (S_minus - S_star)

    if R_minus == R_star:
        R_part = np.zeros(m)
    else:
        R_part = (R - R_star) / (R_minus - R_star)

    Q = v * S_part + (1 - v) * R_part

    # Create ranking (lower is better for VIKOR)
    sorted_indices = np.argsort(Q)
    rankings = {}
    for rank, idx in enumerate(sorted_indices, 1):
        rankings[alternatives[idx]] = rank

    # Step 5: Determine compromise set
    # Condition 1: Acceptable advantage
    # Q(A2) - Q(A1) >= 1/(m-1) where A1 is best, A2 is second best
    if len(sorted_indices) >= 2:
        best_idx = sorted_indices[0]
        second_best_idx = sorted_indices[1]
        advantage_condition = Q[second_best_idx] - Q[best_idx] >= 1 / (m - 1)
    else:
        advantage_condition = True

    # Condition 2: Acceptable stability
    # A1 must be best in terms of both S and R
    best_S_idx = np.argmin(S)
    best_R_idx = np.argmin(R)
    stability_condition = (best_S_idx == sorted_indices[0]) and (best_R_idx == sorted_indices[0])

    # Determine compromise solutions
    if advantage_condition and stability_condition:
        compromise_set = [alternatives[sorted_indices[0]]]
    elif not stability_condition:
        compromise_set = [alternatives[sorted_indices[0]], alternatives[sorted_indices[1]]]
    else:
        # Find all alternatives within acceptable advantage
        k = 1
        while k < m and Q[sorted_indices[k]] - Q[sorted_indices[0]] < 1 / (m - 1):
            k += 1
        compromise_set = [alternatives[sorted_indices[i]] for i in range(k)]

    return {
        'scores': dict(zip(alternatives, Q)),
        'S': dict(zip(alternatives, S)),
        'R': dict(zip(alternatives, R)),
        'S_star': S_star,
        'S_minus': S_minus,
        'R_star': R_star,
        'R_minus': R_minus,
        'rankings': rankings,
        'compromise_set': compromise_set,
        'v': v
    }


def calculate_from_raw(matrix, weights, criteria_direction, v=0.5):
    """
    Calculate VIKOR rankings from raw (non-normalized) matrix.

    Parameters:
    -----------
    matrix : pd.DataFrame or np.ndarray
        Raw decision matrix
    weights : np.ndarray or list
        Weights of criteria
    criteria_direction : list
        Direction of each criterion ('max' for benefit, 'min' for cost)
    v : float
        Strategy parameter (default: 0.5)

    Returns:
    --------
    dict
        VIKOR results
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.normalization import normalize_min_max

    # Normalize using min-max normalization
    normalized = normalize_min_max(matrix, criteria_direction)

    # Calculate VIKOR
    return calculate(normalized, weights, criteria_direction, v)
