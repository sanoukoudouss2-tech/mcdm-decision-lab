"""
ELECTRE III (ELimination Et Choix Traduisant la REalité) implementation.
Outranking method with concordance and discordance analysis.
"""

import numpy as np
import pandas as pd


def calculate(matrix, weights, criteria_direction, p=None, q=None, v=None, alpha=0.3, beta=0.7):
    """
    Calculate rankings using ELECTRE III method.

    Parameters:
    -----------
    matrix : pd.DataFrame or np.ndarray
        Decision matrix (alternatives x criteria)
    weights : np.ndarray or list
        Weights of criteria
    criteria_direction : list
        Direction of each criterion ('max' for benefit, 'min' for cost)
    p : list, optional
        Preference threshold for each criterion
    q : list, optional
        Indifference threshold for each criterion
    v : list, optional
        Veto threshold for each criterion
    alpha : float
        Concordance threshold (default: 0.3)
    beta : float
        Discordance threshold (default: 0.7)

    Returns:
    --------
    dict
        Results containing:
        - concordance_matrix: concordance matrix
        - discordance_matrix: discordance matrix
        - credibility_matrix: credibility matrix
        - rankings: ranking of alternatives
        - kernel: set of non-dominated alternatives
    """
    if isinstance(matrix, pd.DataFrame):
        matrix_val = matrix.values
        alternatives = matrix.index.tolist()
    else:
        matrix_val = matrix.copy()
        alternatives = [f'A{i+1}' for i in range(matrix.shape[0])]

    weights = np.array(weights)
    m, n = matrix_val.shape

    # Set default thresholds if not provided
    if p is None:
        p = [np.std(matrix_val[:, j]) for j in range(n)]
    if q is None:
        q = [0.5 * np.std(matrix_val[:, j]) for j in range(n)]
    if v is None:
        v = [2 * np.std(matrix_val[:, j]) for j in range(n)]

    # Calculate concordance matrix
    concordance = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            if i == j:
                continue

            concordance_sum = 0
            for k in range(n):
                diff = matrix_val[i, k] - matrix_val[j, k]

                # Adjust for cost criteria
                if criteria_direction[k] == 'min':
                    diff = -diff

                # Calculate concordance index
                if diff <= -q[k]:
                    c_ik = 0
                elif diff <= p[k]:
                    c_ik = (diff + q[k]) / (p[k] + q[k])
                else:
                    c_ik = 1

                concordance_sum += weights[k] * c_ik

            concordance[i, j] = concordance_sum

    # Calculate discordance matrix
    discordance = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            if i == j:
                continue

            max_discordance = 0
            for k in range(n):
                diff = matrix_val[j, k] - matrix_val[i, k]

                # Adjust for cost criteria
                if criteria_direction[k] == 'min':
                    diff = -diff

                # Calculate discordance index
                if diff <= -v[k]:
                    d_ik = 0
                elif diff <= -p[k]:
                    d_ik = (diff + v[k]) / (v[k] - p[k])
                else:
                    d_ik = 1

                max_discordance = max(max_discordance, d_ik)

            discordance[i, j] = max_discordance

    # Calculate credibility matrix
    credibility = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            if i == j:
                continue

            # Check concordance condition
            if concordance[i, j] >= alpha:
                # Check discordance condition
                if discordance[i, j] <= beta:
                    credibility[i, j] = concordance[i, j]
                else:
                    credibility[i, j] = 0
            else:
                credibility[i, j] = 0

    # Calculate net outranking scores
    net_scores = np.sum(credibility, axis=1) - np.sum(credibility, axis=0)

    # Create ranking (higher is better)
    sorted_indices = np.argsort(-net_scores)
    rankings = {}
    for rank, idx in enumerate(sorted_indices, 1):
        rankings[alternatives[idx]] = rank

    # Find kernel (non-dominated alternatives)
    dominated = np.zeros(m, dtype=bool)
    for i in range(m):
        for j in range(m):
            if i != j and credibility[i, j] > 0 and credibility[j, i] == 0:
                dominated[j] = True

    kernel = [alternatives[i] for i in range(m) if not dominated[i]]

    return {
        'concordance_matrix': concordance,
        'discordance_matrix': discordance,
        'credibility_matrix': credibility,
        'net_scores': dict(zip(alternatives, net_scores)),
        'rankings': rankings,
        'kernel': kernel,
        'alpha': alpha,
        'beta': beta
    }
