"""
Sensitivity analysis utilities for MCDM methods.
Provides functions to analyze the robustness of rankings.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def sensitivity_analysis_weights(rankings, base_weights, matrix, criteria_direction, method='topsis', delta=0.1):
    """
    Perform sensitivity analysis by varying weights.

    Parameters:
    -----------
    rankings : dict
        Base rankings
    base_weights : np.ndarray
        Base weights
    matrix : pd.DataFrame
        Decision matrix
    criteria_direction : list
        Direction of criteria ('max' or 'min')
    method : str
        Method to use for ranking
    delta : float
        Percentage variation (0.1 = 10%)

    Returns:
    --------
    dict
        Sensitivity analysis results
    """
    from methods import topsis, vikor, wsm, wpm, waspas

    n_criteria = len(base_weights)
    alternatives = matrix.index.tolist() if isinstance(matrix, pd.DataFrame) else [f"A{i+1}" for i in range(matrix.shape[0])]

    results = {}

    for i in range(n_criteria):
        results[i] = {}

        # Vary weight i by ±delta%
        variations = np.linspace(base_weights[i] * (1 - delta), base_weights[i] * (1 + delta), 5)

        for new_weight in variations:
            adjusted_weights = base_weights.copy()
            adjusted_weights[i] = new_weight

            # Normalize to sum to 1
            adjusted_weights = adjusted_weights / adjusted_weights.sum()

            # Recalculate ranking
            if method == 'topsis':
                from utils.normalization import normalize_vector
                normalized = normalize_vector(matrix)
                weighted = normalized * adjusted_weights
                result = topsis.calculate(weighted, criteria_direction)
            elif method == 'vikor':
                from utils.normalization import normalize_min_max
                normalized = normalize_min_max(matrix, criteria_direction)
                result = vikor.calculate(normalized, adjusted_weights, criteria_direction)
            elif method == 'wsm':
                from utils.normalization import normalize_min_max
                normalized = normalize_min_max(matrix, criteria_direction)
                result = wsm.calculate(normalized, adjusted_weights)
            elif method == 'wpm':
                from utils.normalization import normalize_min_max
                normalized = normalize_min_max(matrix, criteria_direction)
                result = wpm.calculate(normalized, adjusted_weights)
            elif method == 'waspas':
                from utils.normalization import normalize_min_max
                normalized = normalize_min_max(matrix, criteria_direction)
                result = waspas.calculate(normalized, adjusted_weights)
            else:
                result = {alt: 0 for alt in alternatives}

            results[i][new_weight] = result

    return results


def calculate_spearman_correlation(rankings1, rankings2):
    """
    Calculate Spearman rank correlation coefficient between two rankings.

    Parameters:
    -----------
    rankings1 : dict
        First ranking {alternative: rank}
    rankings2 : dict
        Second ranking {alternative: rank}

    Returns:
    --------
    float
        Spearman correlation coefficient
    float
        p-value
    """
    alternatives = list(rankings1.keys())
    ranks1 = [rankings1[alt] for alt in alternatives]
    ranks2 = [rankings2[alt] for alt in alternatives]

    correlation, p_value = spearmanr(ranks1, ranks2)

    return correlation, p_value


def compare_methods_rankings(all_rankings):
    """
    Compare rankings from multiple methods using Spearman correlation.

    Parameters:
    -----------
    all_rankings : dict
        Dictionary of {method_name: {alternative: rank}}

    Returns:
    --------
    pd.DataFrame
        Correlation matrix
    """
    methods = list(all_rankings.keys())
    n_methods = len(methods)

    correlation_matrix = np.zeros((n_methods, n_methods))

    for i in range(n_methods):
        for j in range(n_methods):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                corr, _ = calculate_spearman_correlation(
                    all_rankings[methods[i]],
                    all_rankings[methods[j]]
                )
                correlation_matrix[i, j] = corr

    return pd.DataFrame(correlation_matrix, index=methods, columns=methods)


def robustness_index(rankings, base_ranking):
    """
    Calculate robustness index: percentage of alternatives that maintain their rank.

    Parameters:
    -----------
    rankings : list of dict
        List of rankings from different scenarios
    base_ranking : dict
        Base ranking

    Returns:
    --------
    float
        Robustness index (0-1)
    """
    alternatives = list(base_ranking.keys())
    n_alternatives = len(alternatives)
    stable_count = 0

    for alt in alternatives:
        base_rank = base_ranking[alt]
        all_same = all(ranking[alt] == base_rank for ranking in rankings)
        if all_same:
            stable_count += 1

    return stable_count / n_alternatives


def rank_aggregation(rankings, method='borda'):
    """
    Aggregate multiple rankings into a single ranking.

    Parameters:
    -----------
    rankings : dict
        Dictionary of {method_name: {alternative: rank}}
    method : str
        Aggregation method ('borda', 'copeland', 'average')

    Returns:
    --------
    dict
        Aggregated ranking {alternative: rank}
    """
    alternatives = list(rankings[list(rankings.keys())[0]].keys())
    methods = list(rankings.keys())

    if method == 'borda':
        # Borda count: sum of ranks (lower is better)
        scores = {alt: 0 for alt in alternatives}
        for alt in alternatives:
            for method_name in methods:
                scores[alt] += rankings[method_name][alt]

        # Rank by total score
        sorted_alts = sorted(alternatives, key=lambda x: scores[x])
        aggregated_rank = {alt: i+1 for i, alt in enumerate(sorted_alts)}

    elif method == 'average':
        # Average rank
        scores = {alt: 0 for alt in alternatives}
        for alt in alternatives:
            for method_name in methods:
                scores[alt] += rankings[method_name][alt]
            scores[alt] /= len(methods)

        sorted_alts = sorted(alternatives, key=lambda x: scores[x])
        aggregated_rank = {alt: i+1 for i, alt in enumerate(sorted_alts)}

    elif method == 'copeland':
        # Copeland method: pairwise comparison
        from collections import defaultdict
        wins = defaultdict(int)

        for alt1 in alternatives:
            for alt2 in alternatives:
                if alt1 != alt2:
                    wins1 = 0
                    wins2 = 0
                    for method_name in methods:
                        if rankings[method_name][alt1] < rankings[method_name][alt2]:
                            wins1 += 1
                        elif rankings[method_name][alt1] > rankings[method_name][alt2]:
                            wins2 += 1
                    if wins1 > wins2:
                        wins[alt1] += 1
                    elif wins2 > wins1:
                        wins[alt2] += 1

        sorted_alts = sorted(alternatives, key=lambda x: -wins[x])
        aggregated_rank = {alt: i+1 for i, alt in enumerate(sorted_alts)}

    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    return aggregated_rank
