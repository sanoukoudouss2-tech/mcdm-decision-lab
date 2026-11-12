"""
Example datasets for MCDM methods.
Contains sample decision matrices from the course.
"""

import pandas as pd
import numpy as np


def get_car_selection_example():
    """
    Car selection example from the course.
    Criteria: Style (+), Reliability (+), Fuel Economy (+), Cost (-)

    Returns:
    --------
    pd.DataFrame
        Decision matrix
    list
        Criteria directions
    """
    data = {
        'Style': [7, 8, 9, 6],
        'Reliability': [9, 7, 6, 7],
        'Fuel Economy': [9, 8, 8, 8],
        'Cost': [8, 7, 9, 6]
    }
    matrix = pd.DataFrame(data, index=['M1', 'M2', 'M3', 'M4'])
    criteria_direction = ['max', 'max', 'max', 'min']
    return matrix, criteria_direction


def get_machine_selection_example():
    """
    Machine selection example from the course.
    4 machines evaluated on 5 criteria.

    Returns:
    --------
    pd.DataFrame
        Decision matrix
    list
        Criteria directions
    """
    data = {
        'C1': [8, 7, 9, 6],
        'C2': [7, 8, 6, 7],
        'C3': [9, 7, 8, 6],
        'C4': [7, 9, 8, 7],
        'C5': [8, 6, 7, 9]
    }
    matrix = pd.DataFrame(data, index=['A1', 'A2', 'A3', 'A4'])
    criteria_direction = ['max', 'max', 'max', 'min', 'max']
    return matrix, criteria_direction


def get_supplier_selection_example():
    """
    Supplier selection example for BWM.
    4 criteria: Quality, Price, Delivery, RSE

    Returns:
    --------
    pd.DataFrame
        Decision matrix
    list
        Criteria directions
    """
    data = {
        'Quality': [9, 7, 8, 6],
        'Price': [6, 8, 7, 9],
        'Delivery': [8, 7, 9, 6],
        'RSE': [7, 6, 8, 7]
    }
    matrix = pd.DataFrame(data, index=['S1', 'S2', 'S3', 'S4'])
    criteria_direction = ['max', 'min', 'max', 'max']
    return matrix, criteria_direction


def get_machine_cutting_example():
    """
    Machine cutting example for CRITIC method.
    4 machines evaluated on thickness, width, quality, maintenance cost.

    Returns:
    --------
    pd.DataFrame
        Decision matrix
    list
        Criteria directions
    """
    data = {
        'Thickness': [10, 8, 12, 9],
        'Width': [2, 3, 1.5, 2.5],
        'Quality': [8, 7, 9, 6],
        'Maintenance Cost': [5000, 4000, 6000, 4500]
    }
    matrix = pd.DataFrame(data, index=['A1', 'A2', 'A3', 'A4'])
    criteria_direction = ['max', 'min', 'max', 'min']
    return matrix, criteria_direction


def get_portfolio_selection_example():
    """
    Portfolio selection example from the course.
    Financial metrics for stock selection.

    Returns:
    --------
    pd.DataFrame
        Decision matrix
    list
        Criteria directions
    """
    data = {
        'Return (%)': [15, 12, 18, 10, 14],
        'Std Dev (%)': [20, 15, 25, 12, 18],
        'Volume': [1000, 800, 1200, 600, 900],
        'Beta': [1.2, 0.9, 1.5, 0.7, 1.1],
        'EPS': [5, 4, 6, 3, 4.5],
        'ROE (%)': [20, 18, 22, 15, 19]
    }
    matrix = pd.DataFrame(data, index=['Stock1', 'Stock2', 'Stock3', 'Stock4', 'Stock5'])
    criteria_direction = ['max', 'min', 'max', 'min', 'max', 'max']
    return matrix, criteria_direction


def get_student_recruitment_example():
    """
    Student recruitment example for Fuzzy TOPSIS.
    3 candidates evaluated on 5 criteria.

    Returns:
    --------
    pd.DataFrame
        Decision matrix
    list
        Criteria directions
    """
    data = {
        'Emotional Stability': [8, 7, 9],
        'Oral Communication': [7, 9, 8],
        'Personality': [9, 8, 7],
        'Past Experience': [8, 7, 9],
        'Self Confidence': [7, 8, 9]
    }
    matrix = pd.DataFrame(data, index=['A1', 'A2', 'A3'])
    criteria_direction = ['max', 'max', 'max', 'max', 'max']
    return matrix, criteria_direction


def get_risk_prioritization_example():
    """
    Risk prioritization example.
    Risks evaluated on occurrence, severity, detection.

    Returns:
    --------
    pd.DataFrame
        Decision matrix
    list
        Criteria directions
    """
    data = {
        'Occurrence': [5, 3, 7, 4, 6],
        'Severity': [8, 6, 9, 5, 7],
        'Detection': [4, 7, 3, 6, 5]
    }
    matrix = pd.DataFrame(data, index=['R1', 'R2', 'R3', 'R4', 'R5'])
    criteria_direction = ['min', 'min', 'max']
    return matrix, criteria_direction


def get_ml_model_selection_example():
    """
    ML model selection example.
    Models evaluated on accuracy, precision, recall, F1, AUC.

    Returns:
    --------
    pd.DataFrame
        Decision matrix
    list
        Criteria directions
    """
    data = {
        'Accuracy': [0.92, 0.89, 0.94, 0.91],
        'Precision': [0.88, 0.91, 0.93, 0.90],
        'Recall': [0.90, 0.87, 0.92, 0.89],
        'F1 Score': [0.89, 0.89, 0.925, 0.895],
        'AUC': [0.95, 0.93, 0.97, 0.94]
    }
    matrix = pd.DataFrame(data, index=['SVM', 'Random Forest', 'XGBoost', 'KNN'])
    criteria_direction = ['max', 'max', 'max', 'max', 'max']
    return matrix, criteria_direction


def get_ahp_pairwise_example():
    """
    AHP pairwise comparison matrix example (car purchase).

    Returns:
    --------
    pd.DataFrame
        Pairwise comparison matrix
    """
    matrix = pd.DataFrame(
        [[1, 7, 3],
         [1/7, 1, 1/3],
         [1/3, 3, 1]],
        index=['Cost', 'Comfort', 'Safety'],
        columns=['Cost', 'Comfort', 'Security']
    )
    return matrix


def get_bwm_vectors_example():
    """
    BWM Best-to-Others and Others-to-Worst vectors example.

    Returns:
    --------
    dict
        BWM vectors
    """
    return {
        'best': 'C1',  # Quality
        'worst': 'C4',  # RSE
        'best_to_others': {'C1': 1, 'C2': 3, 'C3': 4, 'C4': 8},
        'others_to_worst': {'C1': 8, 'C2': 5, 'C3': 3, 'C4': 1}
    }


def get_all_examples():
    """
    Get all available examples.

    Returns:
    --------
    dict
        Dictionary of example names and their data
    """
    return {
        'Car Selection': get_car_selection_example,
        'Machine Selection': get_machine_selection_example,
        'Supplier Selection': get_supplier_selection_example,
        'Machine Cutting (CRITIC)': get_machine_cutting_example,
        'Portfolio Selection': get_portfolio_selection_example,
        'Student Recruitment (Fuzzy)': get_student_recruitment_example,
        'Risk Prioritization': get_risk_prioritization_example,
        'ML Model Selection': get_ml_model_selection_example
    }
