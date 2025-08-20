 

# EconoNet: Economic Analysis & Forecasting Hub

EconoNet is a Streamlit-based web application for comprehensive macroeconomic analysis, modeling, and forecasting. It provides an integrated environment for data scientists, economists, and analysts to explore data, build predictive models, and assess financial risks.

## ✨ Key Features

- **Interactive Dashboard**: A user-friendly interface built with Streamlit for seamless navigation and analysis.
- **Data Exploration**: Tools for deep-diving into economic datasets, uncovering trends, and analyzing correlations.
- **Predictive Modeling**: Leverage time-series models like ARIMA to forecast key economic indicators.
- **Financial Risk Analysis**: Evaluate risk using standard methodologies like Value at Risk (VaR), Conditional VaR (CVaR), and Monte Carlo simulations.
- **Notebook Explorer**: Run and interact with detailed analytical Jupyter Notebooks directly within the application.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- `pip` and `virtualenv`

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/EconoNet.git
   cd EconoNet
   ```

2. **Create and activate a virtual environment:**

   ```bash
   # For Windows
   python -m venv venv
   .\venv\Scripts\activate

   # For macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Launch the Streamlit application by running the following command from the root directory:

```bash
streamlit run dashboard/app.py
```

The application will open in your default web browser.

## 📂 Project Structure

| Name             | Description                                                 |
| :--------------- | :---------------------------------------------------------- |
| `dashboard/`     | Contains the Streamlit application scripts.                 |
| `├── app.py`     | The main entry point for the Streamlit app.                 |
| `└── pages/`     | Additional pages for the dashboard (e.g., Notebook Explorer).|
| `data/`          | Directory for storing raw, processed, and cleaned data.     |
| `models/`        | Saved machine learning models.                              |
| `notebooks/`     | Jupyter notebooks for detailed analysis and experimentation.|
| `src/`           | Source code for data processing, modeling, and utilities.   |
| `└── utils/`     | Utility scripts, including the notebook runner.             |
| `requirements.txt`| A list of Python packages required for the project.         |
| `README.md`      | This file.                                                  |


## 🔬 Analyses and Notebooks

The `notebooks/` directory contains detailed walkthroughs of the core analyses performed in this project:

1. **Data Exploration (`Data_Exploration.ipynb`)**:
   - Covers EDA, feature engineering, correlation analysis, and data preprocessing.
2. **Predictive Modeling (`Predictive_Models.ipynb`)**:
   - Demonstrates time-series forecasting using models like ARIMA.
3. **Risk Analysis (`Risk_Analysis.ipynb`)**:
   - Provides a framework for financial risk assessment using VaR, CVaR, and Monte Carlo methods.

These notebooks can be run interactively within the "Notebook Explorer" section of the Streamlit application.

