"""
Visualization utilities for MCDM methods.
Provides functions to create interactive plots using Plotly.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def plot_radar_chart(alternatives, criteria, scores, title="Radar Chart"):
    """
    Create a radar chart for comparing alternatives.

    Parameters:
    -----------
    alternatives : list
        Names of alternatives
    criteria : list
        Names of criteria
    scores : np.ndarray or pd.DataFrame
        Scores of alternatives (alternatives x criteria)
    title : str
        Chart title

    Returns:
    --------
    plotly.graph_objects.Figure
        Radar chart figure
    """
    fig = go.Figure()

    if isinstance(scores, pd.DataFrame):
        scores = scores.values

    for i, alt in enumerate(alternatives):
        fig.add_trace(go.Scatterpolar(
            r=scores[i, :].tolist(),
            theta=criteria,
            fill='toself',
            name=alt
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title=title
    )

    return fig


def plot_bar_chart(alternatives, scores, title="Bar Chart", xaxis_title="Alternatives", yaxis_title="Score"):
    """
    Create a bar chart for comparing alternatives.

    Parameters:
    -----------
    alternatives : list
        Names of alternatives
    scores : list or np.ndarray
        Scores of alternatives
    title : str
        Chart title
    xaxis_title : str
        X-axis label
    yaxis_title : str
        Y-axis label

    Returns:
    --------
    plotly.graph_objects.Figure
        Bar chart figure
    """
    fig = go.Figure(data=[go.Bar(x=alternatives, y=scores)])

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title
    )

    return fig


def plot_weights_bar_chart(criteria, weights, title="Criteria Weights"):
    """
    Create a bar chart for criteria weights.

    Parameters:
    -----------
    criteria : list
        Names of criteria
    weights : list or np.ndarray
        Weights of criteria
    title : str
        Chart title

    Returns:
    --------
    plotly.graph_objects.Figure
        Bar chart figure
    """
    fig = go.Figure(data=[go.Bar(x=criteria, y=weights)])

    fig.update_layout(
        title=title,
        xaxis_title="Criteria",
        yaxis_title="Weight"
    )

    return fig


def plot_heatmap(matrix, row_labels, col_labels, title="Heatmap"):
    """
    Create a heatmap for a matrix.

    Parameters:
    -----------
    matrix : np.ndarray or pd.DataFrame
        Matrix to visualize
    row_labels : list
        Row labels
    col_labels : list
        Column labels
    title : str
        Chart title

    Returns:
    --------
    plotly.graph_objects.Figure
        Heatmap figure
    """
    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.values

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=col_labels,
        y=row_labels,
        colorscale='Viridis'
    ))

    fig.update_layout(title=title)

    return fig


def plot_sensitivity_analysis(alternatives, criteria, base_weights, variations, title="Sensitivity Analysis"):
    """
    Create a line chart for sensitivity analysis.

    Parameters:
    -----------
    alternatives : list
        Names of alternatives
    criteria : list
        Names of criteria
    base_weights : np.ndarray
        Base weights
    variations : dict
        Dictionary of {criterion_index: {weight_value: {alternative: score}}}
    title : str
        Chart title

    Returns:
    --------
    plotly.graph_objects.Figure
        Line chart figure
    """
    fig = go.Figure()

    colors = px.colors.qualitative.Set1

    for alt_idx, alt in enumerate(alternatives):
        for crit_idx in variations.keys():
            x_values = list(variations[crit_idx].keys())
            y_values = [variations[crit_idx][w][alt] for w in x_values]

            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines+markers',
                name=f"{alt} (w{crit_idx+1})",
                line=dict(color=colors[alt_idx % len(colors)])
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Weight Variation",
        yaxis_title="Score",
        hovermode='x unified'
    )

    return fig


def plot_ranking_comparison(rankings, methods, title="Ranking Comparison"):
    """
    Create a grouped bar chart for comparing rankings from different methods.

    Parameters:
    -----------
    rankings : dict
        Dictionary of {method: {alternative: rank}}
    methods : list
        Names of methods
    title : str
        Chart title

    Returns:
    --------
    plotly.graph_objects.Figure
        Grouped bar chart figure
    """
    alternatives = list(rankings[methods[0]].keys())

    fig = go.Figure()

    for method in methods:
        ranks = [rankings[method][alt] for alt in alternatives]
        fig.add_trace(go.Bar(
            name=method,
            x=alternatives,
            y=ranks
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Alternatives",
        yaxis_title="Rank (lower is better)",
        barmode='group'
    )

    return fig


def plot_correlation_heatmap(correlation_matrix, method_names, title="Method Correlation"):
    """
    Create a heatmap for correlation matrix between methods.

    Parameters:
    -----------
    correlation_matrix : np.ndarray
        Correlation matrix
    method_names : list
        Names of methods
    title : str
        Chart title

    Returns:
    --------
    plotly.graph_objects.Figure
        Heatmap figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=method_names,
        y=method_names,
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlation_matrix, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(title=title)

    return fig
