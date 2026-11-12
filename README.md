# MCDM Decision Lab

A comprehensive Multi-Criteria Decision Making (MCDM) analysis tool built with Streamlit. This application provides an interactive interface for applying various MCDM methods to decision problems, including weighting methods, ranking techniques, and fuzzy logic approaches.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Implemented Methods](#implemented-methods)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies](#technologies)
- [Deployment](#deployment)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

MCDM Decision Lab is designed to help decision-makers evaluate and rank alternatives based on multiple criteria. It implements both classical (crisp) and fuzzy MCDM methods, making it suitable for problems with precise data as well as those with uncertainty and qualitative judgments.

The application is particularly useful for:
- **Academic Research**: Teaching and learning MCDM methods
- **Business Decisions**: Supplier selection, project prioritization, investment decisions
- **Engineering**: Equipment selection, technology evaluation
- **Healthcare**: Treatment selection, hospital location planning
- **Finance**: Portfolio selection, risk assessment

## ✨ Features

- **Interactive Web Interface**: User-friendly Streamlit-based UI
- **Multiple Weighting Methods**: Subjective (AHP, BWM) and objective (Entropy, CRITIC)
- **Diverse Ranking Methods**: Additive (WSM, WPM, WASPAS, TOPSIS, VIKOR) and outranking (PROMETHEE, ELECTRE)
- **Fuzzy Logic Support**: Fuzzy TOPSIS for handling uncertainty
- **Data Visualization**: Interactive plots with Plotly
- **Sensitivity Analysis**: Test ranking robustness under weight variations
- **Method Comparison**: Compare results from multiple methods with correlation analysis
- **Export Options**: Excel, CSV, and text report generation
- **Example Datasets**: Pre-loaded examples from the course material
- **Flexible Data Input**: Upload files (CSV/Excel), use examples, or manual entry

## 🔬 Implemented Methods

### Weighting Methods

1. **AHP (Analytic Hierarchy Process)**
   - Pairwise comparison-based weighting
   - Consistency ratio calculation
   - Eigenvector method for weight derivation

2. **BWM (Best-Worst Method)**
   - Reduced comparison requirements (2n-3 vs n(n-1)/2 for AHP)
   - Linear programming optimization
   - Consistency indicator (ξ)

3. **Entropy**
   - Objective weighting based on information entropy
   - Measures criterion discrimination power
   - No expert judgment required

4. **CRITIC (CRiteria Importance Through Intercriteria Correlation)**
   - Considers both standard deviation and correlation
   - Accounts for criterion interdependence
   - Objective weighting method

### Ranking Methods

1. **WSM (Weighted Sum Method)**
   - Simple additive approach
   - Weighted sum of normalized values

2. **WPM (Weighted Product Method)**
   - Multiplicative approach
   - Product of weighted normalized values

3. **WASPAS (Weighted Aggregates Sum Product Assessment)**
   - Combines WSM and WPM
   - Adjustable trade-off parameter (λ)

4. **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)**
   - Distance-based ranking
   - Considers both positive and negative ideal solutions
   - Closeness coefficient calculation

5. **VIKOR (VIseKriterijumska Optimizacija I Kompromisno Resenje)**
   - Compromise ranking method
   - Balances group utility (S) and individual regret (R)
   - Adjustable strategy parameter (v)

6. **PROMETHEE (Preference Ranking Organization METHod for Enrichment Evaluations)**
   - Outranking method
   - Preference functions (usual, linear, level, V-shape, Gaussian)
   - Positive and negative flows

7. **ELECTRE III (ELimination Et Choix Traduisant la REalité)**
   - Outranking method with concordance/discordance
   - Threshold-based preference modeling
   - Kernel determination

8. **Fuzzy TOPSIS**
   - Handles triangular fuzzy numbers
   - Suitable for uncertain/imprecise data
   - Defuzzification and ranking

## 🏗️ Architecture

```
mcdm-decision-lab/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
│
├── methods/                    # MCDM method implementations
│   ├── __init__.py
│   ├── ahp.py                  # Analytic Hierarchy Process
│   ├── bwm.py                  # Best-Worst Method
│   ├── entropy.py              # Entropy weighting
│   ├── critic.py               # CRITIC method
│   ├── wsm.py                  # Weighted Sum Method
│   ├── wpm.py                  # Weighted Product Method
│   ├── waspas.py               # WASPAS method
│   ├── topsis.py               # TOPSIS method
│   ├── vikor.py                # VIKOR method
│   ├── promethee.py            # PROMETHEE method
│   ├── electre.py              # ELECTRE III method
│   └── fuzzy_topsis.py         # Fuzzy TOPSIS
│
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── normalization.py        # Normalization techniques
│   ├── validation.py           # Input validation
│   ├── visualization.py       # Plotting functions
│   ├── sensitivity.py         # Sensitivity analysis
│   └── export.py               # Export functions
│
└── data/                       # Example datasets
    ├── __init__.py
    └── examples.py             # Sample decision matrices
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Local Installation

1. **Clone or download the repository**

```bash
cd "Outils d'optimisation"
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Unix/MacOS:
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Step-by-Step Guide

1. **Load Data**
   - Upload a CSV/Excel file with your decision matrix
   - Select a pre-loaded example from the course
   - Or enter data manually

2. **Configure Criteria**
   - Specify whether each criterion should be maximized (benefit) or minimized (cost)

3. **Select Weighting Method**
   - Choose from Equal Weights, AHP, BWM, Entropy, CRITIC, or Manual
   - The system will calculate criteria weights

4. **Select Ranking Method**
   - Choose from WSM, WPM, WASPAS, TOPSIS, VIKOR, PROMETHEE, ELECTRE, or Fuzzy TOPSIS
   - Rankings and scores will be computed

5. **Analyze Results**
   - View rankings in tabular format
   - Examine scores and visualizations
   - Enable sensitivity analysis to test robustness
   - Compare multiple methods with correlation analysis

6. **Export Results**
   - Download results as Excel file
   - Export rankings as CSV
   - Generate a text report

### Data Format

Your decision matrix should be formatted as follows:

- **First column**: Alternative names (row labels)
- **Subsequent columns**: Criterion values
- **No missing values**
- **Numeric data only**

Example CSV format:
```csv
Alternative,C1,C2,C3,C4
A1,7,9,9,8
A2,8,7,8,7
A3,9,6,8,9
A4,6,7,8,6
```

## 🛠️ Technologies

- **Python 3.8+**: Core programming language
- **Streamlit 1.28+**: Web application framework
- **Pandas 2.0+**: Data manipulation and analysis
- **NumPy 1.24+**: Numerical computing
- **SciPy 1.10+**: Scientific computing (optimization)
- **Plotly 5.17+**: Interactive visualizations
- **OpenPyXL 3.1+**: Excel file handling
- **Matplotlib 3.7+**: Additional plotting support

## 🌐 Deployment to Streamlit Community Cloud

### Prerequisites

- A GitHub account
- A Streamlit Community Cloud account (free)

### Deployment Steps

1. **Prepare the Repository**

Ensure your project is ready for deployment:
- All files are committed to Git
- `requirements.txt` is complete
- No absolute file paths in code
- All imports work correctly

2. **Create GitHub Repository**

```bash
git init
git add .
git commit -m "Initial commit: MCDM Decision Lab"
git branch -M main
```

Create a new repository on GitHub, then:

```bash
git remote add origin https://github.com/YOUR_USERNAME/mcdm-decision-lab.git
git push -u origin main
```

3. **Deploy to Streamlit Community Cloud**

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository (`mcdm-decision-lab`)
5. Select the branch (`main`)
6. Set the main file path to `app.py`
7. Click "Deploy"

The application will be deployed and accessible via a public URL.

### Deployment Checklist

Before deploying, verify:
- ✅ `requirements.txt` contains all dependencies
- ✅ No absolute file paths (e.g., `C:\Users\...`)
- ✅ All necessary files are in the repository
- ✅ Imports work from `app.py`
- ✅ Example CSV/XLSX files use relative paths
- ✅ `st.session_state` is used correctly
- ✅ File uploads work properly
- ✅ Exports function correctly
- ✅ `scipy.optimize` works for BWM
- ✅ Plotly visualizations render
- ✅ All Streamlit pages function after deployment

## 📚 Examples

The application includes several example datasets from the course:

1. **Car Selection**: Choose among 4 car models based on style, reliability, fuel economy, and cost
2. **Machine Selection**: Select cutting machines for industrial use
3. **Supplier Selection**: Evaluate suppliers on quality, price, delivery, and RSE
4. **Portfolio Selection**: Select stocks based on financial metrics
5. **Student Recruitment**: Evaluate job candidates using fuzzy logic
6. **Risk Prioritization**: Prioritize risks based on occurrence, severity, and detection
7. **ML Model Selection**: Choose the best machine learning model

### Example Workflow

1. Select "Use Example" in the sidebar
2. Choose "Car Selection"
3. Specify criteria directions (Cost = Min, others = Max)
4. Select "Entropy" weighting method
5. Select "TOPSIS" ranking method
6. View results and visualizations
7. Enable "Compare All Methods" to see different rankings
8. Export results

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional MCDM methods (ANP, DEMATEL, etc.)
- More fuzzy logic variants (Intuitionistic, Pythagorean)
- Group decision making support
- Enhanced visualizations
- Mobile-responsive design
- Additional export formats
- Unit tests
- Documentation improvements

## 📄 License

This project is created for educational purposes based on the MCDM course from École Mohammadia d'Ingénieurs (EMI), Université Mohammed V de Rabat.

## 📧 Contact

For questions or support regarding the course content:
- Prof. LAMRANI ALAOUI: lamrani@emi.ac.ma
- EMI-Rabat

## 🙏 Acknowledgments

This application is based on the MCDM course taught at École Mohammadia d'Ingénieurs (EMI), covering:
- Analytic Hierarchy Process (AHP)
- Best-Worst Method (BWM)
- Entropy and CRITIC methods
- WSM, WPM, WASPAS, TOPSIS, VIKOR
- PROMETHEE and ELECTRE
- Fuzzy Logic and Fuzzy TOPSIS
- Sensitivity analysis and method comparison

---

**Note**: This application is designed for educational and research purposes. It implements well-established MCDM methods but should be used with appropriate domain expertise for real-world decision making.
