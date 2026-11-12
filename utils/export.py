"""
Export utilities for MCDM results.
Provides functions to export results to Excel and CSV.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def export_to_excel(results, filename='mcdm_results.xlsx'):
    """
    Export MCDM results to an Excel file with multiple sheets.

    Parameters:
    -----------
    results : dict
        Dictionary containing results with keys:
        - 'decision_matrix': original decision matrix
        - 'weights': criteria weights
        - 'normalized_matrix': normalized decision matrix
        - 'rankings': final rankings
        - 'scores': alternative scores
        - 'method': method used
        - 'criteria_direction': criteria directions
    filename : str
        Output filename

    Returns:
    --------
    str
        Path to saved file
    """
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Decision matrix
        if 'decision_matrix' in results:
            dm = results['decision_matrix']
            if isinstance(dm, pd.DataFrame):
                dm.to_excel(writer, sheet_name='Decision Matrix', index=True)
            else:
                pd.DataFrame(dm).to_excel(writer, sheet_name='Decision Matrix', index=True)

        # Weights
        if 'weights' in results:
            weights_df = pd.DataFrame({
                'Criterion': [f'C{i+1}' for i in range(len(results['weights']))],
                'Weight': results['weights']
            })
            weights_df.to_excel(writer, sheet_name='Weights', index=False)

        # Normalized matrix
        if 'normalized_matrix' in results:
            nm = results['normalized_matrix']
            if isinstance(nm, pd.DataFrame):
                nm.to_excel(writer, sheet_name='Normalized Matrix', index=True)
            else:
                pd.DataFrame(nm).to_excel(writer, sheet_name='Normalized Matrix', index=True)

        # Rankings
        if 'rankings' in results:
            rankings_df = pd.DataFrame.from_dict(
                results['rankings'],
                orient='index',
                columns=['Rank']
            )
            rankings_df.index.name = 'Alternative'
            rankings_df.to_excel(writer, sheet_name='Rankings')

        # Scores
        if 'scores' in results:
            scores_df = pd.DataFrame.from_dict(
                results['scores'],
                orient='index',
                columns=['Score']
            )
            scores_df.index.name = 'Alternative'
            scores_df.to_excel(writer, sheet_name='Scores')

        # Summary
        summary_data = {
            'Method': [results.get('method', 'N/A')],
            'Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Number of Alternatives': [len(results.get('rankings', {}))],
            'Number of Criteria': [len(results.get('weights', []))]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    return filename


def export_to_csv(results, filename='mcdm_results.csv'):
    """
    Export MCDM rankings to a CSV file.

    Parameters:
    -----------
    results : dict
        Dictionary containing results
    filename : str
        Output filename

    Returns:
    --------
    str
        Path to saved file
    """
    if 'rankings' in results:
        rankings_df = pd.DataFrame.from_dict(
            results['rankings'],
            orient='index',
            columns=['Rank']
        )
        rankings_df.index.name = 'Alternative'
        rankings_df.to_csv(filename)
    else:
        raise ValueError("No rankings found in results")

    return filename


def export_comparison(all_results, filename='mcdm_comparison.xlsx'):
    """
    Export comparison of multiple methods to an Excel file.

    Parameters:
    -----------
    all_results : dict
        Dictionary of {method_name: results_dict}
    filename : str
        Output filename

    Returns:
    --------
    str
        Path to saved file
    """
    methods = list(all_results.keys())

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Summary comparison
        comparison_data = []
        for method in methods:
            if 'rankings' in all_results[method]:
                for alt, rank in all_results[method]['rankings'].items():
                    comparison_data.append({
                        'Method': method,
                        'Alternative': alt,
                        'Rank': rank
                    })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_excel(writer, sheet_name='Rankings Comparison', index=False)

        # Individual method sheets
        for method in methods:
            sheet_name = method[:31]  # Excel sheet name max 31 chars
            if 'rankings' in all_results[method]:
                rankings_df = pd.DataFrame.from_dict(
                    all_results[method]['rankings'],
                    orient='index',
                    columns=['Rank']
                )
                rankings_df.index.name = 'Alternative'
                rankings_df.to_excel(writer, sheet_name=sheet_name)

    return filename


def export_sensitivity_analysis(sensitivity_results, filename='sensitivity_analysis.xlsx'):
    """
    Export sensitivity analysis results to an Excel file.

    Parameters:
    -----------
    sensitivity_results : dict
        Dictionary of sensitivity analysis results
    filename : str
        Output filename

    Returns:
    --------
    str
        Path to saved file
    """
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for crit_idx, variations in sensitivity_results.items():
            sheet_name = f'Criterion_{crit_idx+1}'
            data = []

            for weight, scores in variations.items():
                for alt, score in scores.items():
                    data.append({
                        'Weight': weight,
                        'Alternative': alt,
                        'Score': score
                    })

            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    return filename


def create_report(results, filename='mcdm_report.txt'):
    """
    Create a text report of MCDM analysis.

    Parameters:
    -----------
    results : dict
        Dictionary containing results
    filename : str
        Output filename

    Returns:
    --------
    str
        Path to saved file
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("MCDM Decision Analysis Report\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Method: {results.get('method', 'N/A')}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if 'weights' in results:
            f.write("Criteria Weights:\n")
            f.write("-" * 40 + "\n")
            for i, w in enumerate(results['weights']):
                f.write(f"Criterion {i+1}: {w:.4f}\n")
            f.write("\n")

        if 'rankings' in results:
            f.write("Final Rankings:\n")
            f.write("-" * 40 + "\n")
            sorted_rankings = sorted(results['rankings'].items(), key=lambda x: x[1])
            for i, (alt, rank) in enumerate(sorted_rankings, 1):
                f.write(f"Rank {rank}: {alt}\n")
            f.write("\n")

        if 'scores' in results:
            f.write("Alternative Scores:\n")
            f.write("-" * 40 + "\n")
            for alt, score in results['scores'].items():
                f.write(f"{alt}: {score:.4f}\n")
            f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("End of Report\n")
        f.write("=" * 60 + "\n")

    return filename
