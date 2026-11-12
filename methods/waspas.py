"""
WASPAS (Weighted Aggregates Sum Product Assessment) implementation.
Combines WSM and WPM methods.
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import methods.wsm as wsm
import methods.wpm as wpm


def calculate(normalized_matrix, weights, lambda_param=0.5):
    """
    Calculate rankings using WASPAS method.

    Parameters:
    -----------
    normalized_matrix : pd.DataFrame or np.ndarray
        Normalized decision matrix (alternatives x criteria)
    weights : np.ndarray or list
        Weights of criteria
    lambda_param : float
        Trade-off parameter between WSM and WPM (default: 0.5)

    Returns:
    --------
    dict
        Results containing:
        - scores: WASPAS scores for each alternative
        - wsm_scores: WSM scores
        - wpm_scores: WPM scores
        - rankings: ranking of alternatives
    """
    if isinstance(normalized_matrix, pd.DataFrame):
        alternatives = normalized_matrix.index.tolist()
    else:
        alternatives = [f'A{i+1}' for i in range(normalized_matrix.shape[0])]

    # Calculate WSM scores
    wsm_result = wsm.calculate(normalized_matrix, weights)
    wsm_scores = np.array([wsm_result['scores'][alt] for alt in alternatives])

    # Calculate WPM scores
    wpm_result = wpm.calculate(normalized_matrix, weights)
    wpm_scores = np.array([wpm_result['scores'][alt] for alt in alternatives])

    # Calculate WASPAS scores
    # Q_i = lambda * WSM_i + (1 - lambda) * WPM_i
    waspas_scores = lambda_param * wsm_scores + (1 - lambda_param) * wpm_scores

    # Create ranking (higher is better)
    sorted_indices = np.argsort(-waspas_scores)
    rankings = {}
    for rank, idx in enumerate(sorted_indices, 1):
        rankings[alternatives[idx]] = rank

    return {
        'scores': dict(zip(alternatives, waspas_scores)),
        'wsm_scores': dict(zip(alternatives, wsm_scores)),
        'wpm_scores': dict(zip(alternatives, wpm_scores)),
        'rankings': rankings,
        'lambda': lambda_param
    }
