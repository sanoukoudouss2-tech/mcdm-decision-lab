"""
Fuzzy TOPSIS implementation.
Handles triangular fuzzy numbers for uncertain decision making.
"""

import numpy as np
import pandas as pd


class TriangularFuzzyNumber:
    """Triangular Fuzzy Number (a, b, c)"""

    def __init__(self, a, b, c):
        self.a = a  # Lower bound (pessimistic)
        self.b = b  # Most likely value
        self.c = c  # Upper bound (optimistic)

    def __add__(self, other):
        return TriangularFuzzyNumber(
            self.a + other.a,
            self.b + other.b,
            self.c + other.c
        )

    def __sub__(self, other):
        return TriangularFuzzyNumber(
            self.a - other.c,
            self.b - other.b,
            self.c - other.a
        )

    def __mul__(self, scalar):
        if scalar >= 0:
            return TriangularFuzzyNumber(
                self.a * scalar,
                self.b * scalar,
                self.c * scalar
            )
        else:
            return TriangularFuzzyNumber(
                self.c * scalar,
                self.b * scalar,
                self.a * scalar
            )

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, other):
        return TriangularFuzzyNumber(
            self.a / other.c if other.c != 0 else 0,
            self.b / other.b if other.b != 0 else 0,
            self.c / other.a if other.a != 0 else 0
        )

    def distance(self, other):
        """Calculate distance between two triangular fuzzy numbers"""
        return np.sqrt(
            (self.a - other.a)**2 +
            (self.b - other.b)**2 +
            (self.c - other.c)**2
        ) / np.sqrt(3)

    def defuzzify(self):
        """Defuzzify using centroid method"""
        return (self.a + self.b + self.c) / 3

    def __repr__(self):
        return f"({self.a:.3f}, {self.b:.3f}, {self.c:.3f})"


def calculate(matrix, weights, criteria_direction=None):
    """
    Calculate rankings using Fuzzy TOPSIS with triangular fuzzy numbers.

    Parameters:
    -----------
    matrix : pd.DataFrame or np.ndarray
        Decision matrix where each element is a tuple (a, b, c) representing a triangular fuzzy number
    weights : np.ndarray or list
        Weights of criteria (can be crisp or fuzzy)
    criteria_direction : list, optional
        Direction of each criterion ('max' for benefit, 'min' for cost)
        If None, assumes all are benefit criteria

    Returns:
    --------
    dict
        Results containing:
        - scores: Fuzzy TOPSIS scores (closeness coefficient) for each alternative
        - d_plus: distance to positive ideal solution
        - d_minus: distance to negative ideal solution
        - rankings: ranking of alternatives
        - defuzzified_scores: defuzzified scores
    """
    if isinstance(matrix, pd.DataFrame):
        alternatives = matrix.index.tolist()
        criteria = matrix.columns.tolist()
    else:
        alternatives = [f'A{i+1}' for i in range(matrix.shape[0])]
        criteria = [f'C{j+1}' for j in range(matrix.shape[1])]

    if criteria_direction is None:
        criteria_direction = ['max'] * len(criteria)

    # Convert matrix to triangular fuzzy numbers
    m = len(alternatives)
    n = len(criteria)

    fuzzy_matrix = np.zeros((m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            if isinstance(matrix, pd.DataFrame):
                val = matrix.iloc[i, j]
            else:
                val = matrix[i, j]

            if isinstance(val, (tuple, list, np.ndarray)):
                fuzzy_matrix[i, j] = TriangularFuzzyNumber(val[0], val[1], val[2])
            else:
                # Crisp value converted to fuzzy number
                fuzzy_matrix[i, j] = TriangularFuzzyNumber(val, val, val)

    # Convert weights to fuzzy numbers if necessary
    fuzzy_weights = np.zeros(n, dtype=object)
    for j in range(n):
        w = weights[j]
        if isinstance(w, (tuple, list, np.ndarray)):
            fuzzy_weights[j] = TriangularFuzzyNumber(w[0], w[1], w[2])
        else:
            fuzzy_weights[j] = TriangularFuzzyNumber(w, w, w)

    # Step 1: Normalize the fuzzy decision matrix
    normalized_matrix = np.zeros((m, n), dtype=object)

    for j in range(n):
        # Find max and max_c for normalization
        max_a = max(fuzzy_matrix[i, j].c for i in range(m))
        max_c = max(fuzzy_matrix[i, j].c for i in range(m))

        for i in range(m):
            if criteria_direction[j] == 'max':
                # Benefit criterion
                normalized_matrix[i, j] = TriangularFuzzyNumber(
                    fuzzy_matrix[i, j].a / max_c if max_c != 0 else 0,
                    fuzzy_matrix[i, j].b / max_c if max_c != 0 else 0,
                    fuzzy_matrix[i, j].c / max_c if max_c != 0 else 0
                )
            else:
                # Cost criterion
                min_a = min(fuzzy_matrix[i, j].a for i in range(m))
                normalized_matrix[i, j] = TriangularFuzzyNumber(
                    min_a / fuzzy_matrix[i, j].c if fuzzy_matrix[i, j].c != 0 else 0,
                    min_a / fuzzy_matrix[i, j].b if fuzzy_matrix[i, j].b != 0 else 0,
                    min_a / fuzzy_matrix[i, j].a if fuzzy_matrix[i, j].a != 0 else 0
                )

    # Step 2: Construct weighted normalized fuzzy decision matrix
    weighted_matrix = np.zeros((m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            weighted_matrix[i, j] = normalized_matrix[i, j] * fuzzy_weights[j]

    # Step 3: Determine fuzzy positive ideal solution (FPIS) and fuzzy negative ideal solution (FNIS)
    fpis = np.zeros(n, dtype=object)
    fnis = np.zeros(n, dtype=object)

    for j in range(n):
        if criteria_direction[j] == 'max':
            max_c = max(weighted_matrix[i, j].c for i in range(m))
            max_b = max(weighted_matrix[i, j].b for i in range(m))
            max_a = max(weighted_matrix[i, j].a for i in range(m))
            fpis[j] = TriangularFuzzyNumber(max_a, max_b, max_c)

            min_c = min(weighted_matrix[i, j].c for i in range(m))
            min_b = min(weighted_matrix[i, j].b for i in range(m))
            min_a = min(weighted_matrix[i, j].a for i in range(m))
            fnis[j] = TriangularFuzzyNumber(min_a, min_b, min_c)
        else:
            min_c = min(weighted_matrix[i, j].c for i in range(m))
            min_b = min(weighted_matrix[i, j].b for i in range(m))
            min_a = min(weighted_matrix[i, j].a for i in range(m))
            fpis[j] = TriangularFuzzyNumber(min_a, min_b, min_c)

            max_c = max(weighted_matrix[i, j].c for i in range(m))
            max_b = max(weighted_matrix[i, j].b for i in range(m))
            max_a = max(weighted_matrix[i, j].a for i in range(m))
            fnis[j] = TriangularFuzzyNumber(max_a, max_b, max_c)

    # Step 4: Calculate distance to FPIS and FNIS
    d_plus = np.zeros(m)
    d_minus = np.zeros(m)

    for i in range(m):
        d_plus[i] = np.sqrt(sum(
            weighted_matrix[i, j].distance(fpis[j])**2
            for j in range(n)
        ))

        d_minus[i] = np.sqrt(sum(
            weighted_matrix[i, j].distance(fnis[j])**2
            for j in range(n)
        ))

    # Step 5: Calculate closeness coefficient
    scores = d_minus / (d_plus + d_minus + 1e-10)

    # Create ranking (higher is better)
    sorted_indices = np.argsort(-scores)
    rankings = {}
    for rank, idx in enumerate(sorted_indices, 1):
        rankings[alternatives[idx]] = rank

    return {
        'scores': dict(zip(alternatives, scores)),
        'd_plus': dict(zip(alternatives, d_plus)),
        'd_minus': dict(zip(alternatives, d_minus)),
        'rankings': rankings,
        'defuzzified_scores': dict(zip(alternatives, scores))
    }


def create_fuzzy_matrix_from_crisp(matrix, uncertainty=0.1):
    """
    Convert a crisp decision matrix to a fuzzy matrix by adding uncertainty.

    Parameters:
    -----------
    matrix : pd.DataFrame or np.ndarray
        Crisp decision matrix
    uncertainty : float
        Percentage uncertainty (e.g., 0.1 for 10%)

    Returns:
    --------
    np.ndarray
        Fuzzy matrix with triangular fuzzy numbers
    """
    if isinstance(matrix, pd.DataFrame):
        matrix_val = matrix.values
    else:
        matrix_val = matrix.copy()

    m, n = matrix_val.shape
    fuzzy_matrix = np.zeros((m, n), dtype=object)

    for i in range(m):
        for j in range(n):
            val = matrix_val[i, j]
            delta = val * uncertainty
            fuzzy_matrix[i, j] = (val - delta, val, val + delta)

    return fuzzy_matrix
