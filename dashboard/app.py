 
import streamlit as st
from pathlib import Path
import sys

# Add project root to sys.path
# Assumes this script is in the 'dashboard' directory
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

st.set_page_config(
    page_title="EconoNet Dashboard",
    page_icon="https://i.imgur.com/6fJp7ss.png", # A simple chart icon
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Font Awesome CSS ---
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.0/css/all.min.css">
""", unsafe_allow_html=True)

st.title("EconoNet: Economic Analysis & Forecasting Hub")

st.markdown("""
Welcome to EconoNet, your integrated dashboard for macroeconomic analysis, modeling, and forecasting.
This platform provides tools to explore economic data, run predictive models, and assess financial risk.

### <i class="fa-solid fa-star"></i> Key Features:
- **Data Exploration**: Dive deep into economic datasets.
- **Predictive Modeling**: Leverage time-series models to forecast key indicators.
- **Risk Analysis**: Evaluate financial risk using methods like VaR and Monte Carlo simulations.
- **Notebook Explorer**: Run and interact with detailed analytical notebooks directly within the dashboard.

Use the sidebar to navigate to the different sections of the dashboard.
""")

st.info("Navigate to the Notebook Explorer page to run detailed analyses.", icon="ℹ️")
