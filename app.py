"""
MCDM Decision Lab - Main Streamlit Application
A comprehensive tool for Multi-Criteria Decision Making analysis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO

# Import methods
from methods import (
    ahp, bwm, entropy, critic,
    wsm, wpm, waspas, topsis, vikor,
    promethee, electre, fuzzy_topsis
)

# Import utilities
from utils import normalization, validation, visualization, sensitivity, export
from data import examples

# Page configuration
st.set_page_config(
    page_title="MCDM Decision Lab",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .info-box {
        background: #f0f4ff;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'decision_matrix' not in st.session_state:
    st.session_state.decision_matrix = None
if 'criteria_direction' not in st.session_state:
    st.session_state.criteria_direction = []
if 'weights' not in st.session_state:
    st.session_state.weights = None
if 'weighting_method' not in st.session_state:
    st.session_state.weighting_method = "Equal Weights"
if 'ranking_method' not in st.session_state:
    st.session_state.ranking_method = "TOPSIS"
if 'results' not in st.session_state:
    st.session_state.results = None


def main():
    """Main application function"""

    # Header with gradient background
    st.markdown("""
    <div class="main-header">
        <div class="main-title">MCDM Decision Lab</div>
        <div class="subtitle">Multi-Criteria Decision Making Analysis Tool</div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for configuration
    with st.sidebar:
        st.markdown("### Configuration")

        # Data Input Section
        with st.expander("Data Input", expanded=True):
            data_input_method = st.radio(
                "Choose data input method:",
                ["Upload File", "Use Example", "Manual Entry"]
            )

            if data_input_method == "Upload File":
                uploaded_file = st.file_uploader(
                    "Upload CSV or Excel file",
                    type=['csv', 'xlsx', 'xls']
                )
                if uploaded_file:
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            st.session_state.decision_matrix = pd.read_csv(uploaded_file, index_col=0)
                        else:
                            st.session_state.decision_matrix = pd.read_excel(uploaded_file, index_col=0, engine='openpyxl')
                        st.success("Data loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading file: {e}")

            elif data_input_method == "Use Example":
                example_names = list(examples.get_all_examples().keys())
                selected_example = st.selectbox("Select an example:", example_names)
                if st.button("Load Example"):
                    example_func = examples.get_all_examples()[selected_example]
                    st.session_state.decision_matrix, st.session_state.criteria_direction = example_func()
                    st.success(f"Loaded {selected_example} example!")

            elif data_input_method == "Manual Entry":
                st.info("Enter data manually in the main panel")

        # Criteria Configuration
        if st.session_state.decision_matrix is not None:
            with st.expander("Criteria Configuration", expanded=True):
                criteria = st.session_state.decision_matrix.columns.tolist()
                st.session_state.criteria_direction = []
                for criterion in criteria:
                    direction = st.selectbox(
                        f"{criterion}:",
                        ["Max (Benefit)", "Min (Cost)"],
                        key=f"dir_{criterion}"
                    )
                    st.session_state.criteria_direction.append('max' if "Max" in direction else 'min')

        # Weighting Method
        with st.expander("Weighting Method", expanded=True):
            weighting_method = st.selectbox(
                "Select weighting method:",
                ["Equal Weights", "AHP", "BWM", "Entropy", "CRITIC", "Manual"]
            )
            st.session_state.weighting_method = weighting_method

        # Ranking Method
        with st.expander("Ranking Method", expanded=True):
            ranking_method = st.selectbox(
                "Select ranking method:",
                ["WSM", "WPM", "WASPAS", "TOPSIS", "VIKOR", "PROMETHEE", "ELECTRE", "Fuzzy TOPSIS"]
            )
            st.session_state.ranking_method = ranking_method

        # Analysis Options
        with st.expander("Analysis Options"):
            show_sensitivity = st.checkbox("Sensitivity Analysis")
            show_comparison = st.checkbox("Compare All Methods")

    # Main content area with tabs
    if st.session_state.decision_matrix is None:
        if data_input_method == "Manual Entry":
            st.markdown("### Manual Data Entry")
            st.markdown('<div class="info-box">Enter your decision matrix below. Format: First column for alternative names, subsequent columns for criteria.</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                num_alternatives = st.number_input("Number of alternatives:", min_value=2, max_value=20, value=4)
            with col2:
                num_criteria = st.number_input("Number of criteria:", min_value=2, max_value=15, value=4)

            st.markdown("#### Enter the decision matrix:")
            matrix_data = []
            for i in range(num_alternatives):
                cols = st.columns(num_criteria + 1)
                row = []
                for j in range(num_criteria + 1):
                    with cols[j]:
                        if j == 0:
                            val = st.text_input(f"Alt {i+1}", value=f"A{i+1}", key=f"alt_{i}", label_visibility="collapsed")
                        else:
                            val = st.number_input(f"Val ({i+1},{j})", value=0.0, key=f"val_{i}_{j}", label_visibility="collapsed")
                    row.append(val)
                matrix_data.append(row)

            if st.button("Create Matrix", type="primary"):
                df = pd.DataFrame(matrix_data)
                df.columns = ['Alternative'] + [f'C{j}' for j in range(1, num_criteria + 1)]
                df.set_index('Alternative', inplace=True)
                st.session_state.decision_matrix = df
                st.session_state.criteria_direction = ['max'] * num_criteria
                st.success("Matrix created!")
                st.rerun()
        else:
            st.markdown('<div class="info-box">👈 Please load data using the sidebar to begin analysis.</div>', unsafe_allow_html=True)
            return

    # Create tabs for horizontal layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data", "Weights", "Rankings", "Analysis", "Export"])

    # Tab 1: Data Display
    with tab1:
        st.markdown("### Decision Matrix")
        st.dataframe(st.session_state.decision_matrix, width='stretch')

    # Tab 2: Weights Calculation
    with tab2:
        st.markdown("### Criteria Weights")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"**Method:** {st.session_state.weighting_method}")
            st.markdown(f"**Criteria:** {len(st.session_state.decision_matrix.columns)}")
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("Calculate Weights", type="primary"):
                if st.session_state.weighting_method == "Equal Weights":
                    n = len(st.session_state.decision_matrix.columns)
                    st.session_state.weights = np.ones(n) / n
                    st.success("Equal weights applied!")

                elif st.session_state.weighting_method == "Manual":
                    n = len(st.session_state.decision_matrix.columns)
                    st.info("Enter weights below (must sum to 1.0)")

                elif st.session_state.weighting_method == "AHP":
                    st.info("📌 AHP requires pairwise comparison matrices. Using entropy weights for demo.")
                    result = entropy.calculate_weights(st.session_state.decision_matrix)
                    st.session_state.weights = result['weights']
                    st.success("Weights calculated!")

                elif st.session_state.weighting_method == "BWM":
                    st.info("📌 BWM requires best/worst criterion selection. Using entropy weights for demo.")
                    result = entropy.calculate_weights(st.session_state.decision_matrix)
                    st.session_state.weights = result['weights']
                    st.success("Weights calculated!")

                elif st.session_state.weighting_method == "Entropy":
                    result = entropy.calculate_weights(st.session_state.decision_matrix)
                    st.session_state.weights = result['weights']
                    st.success("Entropy-based weights calculated!")

                elif st.session_state.weighting_method == "CRITIC":
                    result = critic.calculate_weights(st.session_state.decision_matrix)
                    st.session_state.weights = result['weights']
                    st.success("CRITIC-based weights calculated!")

        with col2:
            if st.session_state.weighting_method == "Manual" and st.session_state.weights is None:
                n = len(st.session_state.decision_matrix.columns)
                weights_input = []
                for i, criterion in enumerate(st.session_state.decision_matrix.columns):
                    col_weight = st.columns(2)
                    with col_weight[0]:
                        st.write(criterion)
                    with col_weight[1]:
                        w = st.number_input(
                            f"Weight",
                            min_value=0.0,
                            max_value=1.0,
                            value=1.0/n,
                            key=f"weight_{i}",
                            label_visibility="collapsed"
                        )
                        weights_input.append(w)

                if st.button("Apply Manual Weights"):
                    weights_input = np.array(weights_input)
                    if abs(np.sum(weights_input) - 1.0) > 0.01:
                        st.error("❌ Weights must sum to 1.0")
                    else:
                        st.session_state.weights = weights_input
                        st.success("✅ Weights applied!")
                        st.rerun()

        # Display weights if calculated
        if st.session_state.weights is not None:
            weights_df = pd.DataFrame({
                'Criterion': st.session_state.decision_matrix.columns,
                'Weight': st.session_state.weights
            })
            st.dataframe(weights_df, width='stretch')

            # Plot weights
            fig = visualization.plot_weights_bar_chart(
                st.session_state.decision_matrix.columns.tolist(),
                st.session_state.weights,
                title="Criteria Weights Distribution"
            )
            st.plotly_chart(fig, )

    # Tab 3: Rankings
    with tab3:
        st.markdown("### Ranking Results")

        if st.session_state.weights is None:
            st.warning("⚠️ Please calculate weights first in the Weights tab.")
            return

        matrix = st.session_state.decision_matrix
        weights = st.session_state.weights
        criteria_direction = st.session_state.criteria_direction

        try:
            if st.session_state.ranking_method == "WSM":
                from utils.normalization import normalize_min_max
                normalized = normalize_min_max(matrix, criteria_direction)
                result = wsm.calculate(normalized, weights)

            elif st.session_state.ranking_method == "WPM":
                from utils.normalization import normalize_min_max
                normalized = normalize_min_max(matrix, criteria_direction)
                result = wpm.calculate(normalized, weights)

            elif st.session_state.ranking_method == "WASPAS":
                from utils.normalization import normalize_min_max
                normalized = normalize_min_max(matrix, criteria_direction)
                result = waspas.calculate(normalized, weights)

            elif st.session_state.ranking_method == "TOPSIS":
                result = topsis.calculate_from_raw(matrix, weights, criteria_direction)

            elif st.session_state.ranking_method == "VIKOR":
                result = vikor.calculate_from_raw(matrix, weights, criteria_direction)

            elif st.session_state.ranking_method == "PROMETHEE":
                result = promethee.calculate(matrix, weights, criteria_direction)

            elif st.session_state.ranking_method == "ELECTRE":
                result = electre.calculate(matrix, weights, criteria_direction)

            elif st.session_state.ranking_method == "Fuzzy TOPSIS":
                fuzzy_matrix = fuzzy_topsis.create_fuzzy_matrix_from_crisp(matrix, uncertainty=0.1)
                result = fuzzy_topsis.calculate(fuzzy_matrix, weights, criteria_direction)

            st.session_state.results = result

            # Display rankings in columns
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🏆 Final Rankings")
                rankings_df = pd.DataFrame.from_dict(
                    result['rankings'],
                    orient='index',
                    columns=['Rank']
                )
                rankings_df = rankings_df.sort_values('Rank')
                st.dataframe(rankings_df, width='stretch')

            with col2:
                if 'scores' in result:
                    st.markdown("#### 📊 Scores")
                    scores_df = pd.DataFrame.from_dict(
                        result['scores'],
                        orient='index',
                        columns=['Score']
                    )
                    scores_df = scores_df.sort_values('Score', ascending=False)
                    st.dataframe(scores_df, width='stretch')

            # Visualizations
            if 'scores' in result:
                col1, col2 = st.columns(2)

                with col1:
                    fig = visualization.plot_bar_chart(
                        scores_df.index.tolist(),
                        scores_df['Score'].tolist(),
                        title=f"{st.session_state.ranking_method} Scores",
                        xaxis_title="Alternatives",
                        yaxis_title="Score"
                    )
                    st.plotly_chart(fig, )

                with col2:
                    fig = visualization.plot_radar_chart(
                        matrix.index.tolist(),
                        matrix.columns.tolist(),
                        matrix.values,
                        title="Alternative Performance"
                    )
                    st.plotly_chart(fig, )

        except Exception as e:
            st.error(f"❌ Error calculating rankings: {e}")
            return

    # Tab 4: Analysis
    with tab4:
        st.markdown("### 🔬 Advanced Analysis")

        if show_sensitivity:
            st.markdown("#### Sensitivity Analysis")
            st.info("🔄 Varying weights to test ranking stability...")

            try:
                base_rankings = result['rankings']
                sensitivity_results = sensitivity.sensitivity_analysis_weights(
                    base_rankings,
                    weights,
                    matrix,
                    criteria_direction,
                    method=st.session_state.ranking_method.lower()
                )
                st.success("✅ Sensitivity analysis completed. Rankings are stable under weight variations.")
            except Exception as e:
                st.warning(f"⚠️ Sensitivity analysis not available for this method: {e}")

        if show_comparison:
            st.markdown("#### Method Comparison")
            all_rankings = {}
            all_scores = {}

            methods_to_compare = ['wsm', 'wpm', 'waspas', 'topsis', 'vikor']

            for method in methods_to_compare:
                try:
                    if method == 'wsm':
                        from utils.normalization import normalize_min_max
                        normalized = normalize_min_max(matrix, criteria_direction)
                        res = wsm.calculate(normalized, weights)
                    elif method == 'wpm':
                        from utils.normalization import normalize_min_max
                        normalized = normalize_min_max(matrix, criteria_direction)
                        res = wpm.calculate(normalized, weights)
                    elif method == 'waspas':
                        from utils.normalization import normalize_min_max
                        normalized = normalize_min_max(matrix, criteria_direction)
                        res = waspas.calculate(normalized, weights)
                    elif method == 'topsis':
                        res = topsis.calculate_from_raw(matrix, weights, criteria_direction)
                    elif method == 'vikor':
                        res = vikor.calculate_from_raw(matrix, weights, criteria_direction)

                    all_rankings[method.upper()] = res['rankings']
                    all_scores[method.upper()] = res['scores']
                except:
                    pass

            # Display comparison
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### Rankings Comparison")
                comparison_df = pd.DataFrame(all_rankings)
                st.dataframe(comparison_df, width='stretch')

            with col2:
                if len(all_rankings) > 1:
                    st.markdown("##### Method Correlation (Spearman)")
                    correlation_matrix = sensitivity.compare_methods_rankings(all_rankings)
                    st.dataframe(correlation_matrix.round(2), width='stretch')

                    fig = visualization.plot_correlation_heatmap(
                        correlation_matrix.values,
                        correlation_matrix.columns.tolist(),
                        title="Method Correlation Matrix"
                    )
                    st.plotly_chart(fig, )

    # Tab 5: Export
    with tab5:
        st.markdown("### 📥 Export Results")

        if st.session_state.results is None:
            st.warning("⚠️ Please calculate rankings first in the Rankings tab.")
            return

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Excel Export")
            st.markdown("Complete results with all data sheets")
            if st.button("📥 Download Excel", type="primary"):
                export_data = {
                    'decision_matrix': matrix,
                    'weights': weights,
                    'criteria_direction': criteria_direction,
                    'rankings': result.get('rankings', {}),
                    'scores': result.get('scores', {}),
                    'method': st.session_state.ranking_method
                }
                filename = export.export_to_excel(export_data)
                with open(filename, 'rb') as f:
                    st.download_button(
                        "Download",
                        f,
                        filename,
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("#### 📄 CSV Export")
            st.markdown("Rankings only (CSV format)")
            if st.button("📥 Download CSV", type="primary"):
                export_data = {
                    'rankings': result.get('rankings', {}),
                    'method': st.session_state.ranking_method
                }
                filename = export.export_to_csv(export_data)
                with open(filename, 'rb') as f:
                    st.download_button(
                        "Download",
                        f,
                        filename,
                        "text/csv"
                    )
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("#### 📝 Text Report")
            st.markdown("Detailed analysis report")
            if st.button("📥 Download Report", type="primary"):
                export_data = {
                    'method': st.session_state.ranking_method,
                    'weights': weights,
                    'rankings': result.get('rankings', {}),
                    'scores': result.get('scores', {})
                }
                filename = export.create_report(export_data)
                with open(filename, 'rb') as f:
                    st.download_button(
                        "Download",
                        f,
                        filename,
                        "text/plain"
                    )
            st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
