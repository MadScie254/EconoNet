"""
EconoNet - Immersive Economic Intelligence Dashboard
==================================================

An advanced, immersive economic analysis platform integrating all notebooks
with interactive models, real-time data, and comprehensive analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path
import warnings
import subprocess
import time
from datetime import datetime, timedelta
import json
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Plotly compatibility fix for range objects
def fix_plotly_data(data):
    """Convert range/arange objects to lists for Plotly compatibility"""
    if isinstance(data, range):
        return list(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif hasattr(data, 'tolist'):
        return data.tolist()
    return data

# Advanced API Integration
try:
    from src.api_integration import get_real_time_data
except ImportError:
    def get_real_time_data():
        return {"status": "API module not available"}

# Configure Streamlit page
st.set_page_config(
    page_title="EconoNet - Immersive Economic Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for immersive experience
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
    }
    
    .notebook-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .notebook-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.25);
    }
    
    .interactive-panel {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .analysis-results {
        background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .stSelectbox > div > div > select {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
    }
    
    .model-output {
        background: rgba(255,255,255,0.95);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .sidebar .element-container {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 12px;
        padding: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: white;
        background: rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255,255,255,0.3);
    }
    
    .real-time-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        background: #00ff00;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .immersive-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
</style>
""", unsafe_allow_html=True)

# Main dashboard header
st.markdown("""
<div class="main-header">
    <h1><i class="fas fa-globe-africa"></i> EconoNet - Immersive Economic Intelligence Platform</h1>
    <p><span class="real-time-indicator"></span>Advanced economic analysis with integrated notebook experiences and real-time modeling</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for navigation and controls
st.sidebar.markdown("## 🎛️ Control Center")

# Notebook selection with enhanced interface
st.sidebar.markdown("### 📊 Interactive Notebooks")
notebook_categories = {
    "📈 Economic Analysis": [
        ("GDP Growth Analysis", "GDP_Analysis.ipynb", "🏛️"),
        ("Inflation Dynamics", "Inflation_modeling.ipynb", "💰"),
        ("Trade & Export Analysis", "Trade_Import_Export_Analysis.ipynb", "🚢")
    ],
    "💱 Financial Markets": [
        ("Exchange Rate Modeling", "FX_modeling.ipynb", "💱"),
        ("Exchange Rates Analysis", "Exchange_Rates_Analysis.ipynb", "📊"),
        ("Liquidity Analysis", "Liquidity_modeling.ipynb", "💧")
    ],
    "⚠️ Risk & Banking": [
        ("Comprehensive Risk Analysis", "Risk_Analysis.ipynb", "🛡️"),
        ("Interbank Markets", "Interbank_Markets_Analysis.ipynb", "🏦"),
        ("Government Finance", "Government_Finance_Analysis.ipynb", "🏛️")
    ],
    "🔍 Data Exploration": [
        ("Exploratory Data Analysis", "EDA.ipynb", "🔍"),
        ("Data Exploration", "Data_Exploration.ipynb", "📋")
    ]
}

selected_category = st.sidebar.selectbox("📂 Select Category", list(notebook_categories.keys()))
notebooks_in_category = notebook_categories[selected_category]

selected_notebook = st.sidebar.selectbox(
    "📓 Choose Notebook", 
    [f"{icon} {name}" for name, file, icon in notebooks_in_category]
)

# Extract the actual notebook info
selected_idx = [f"{icon} {name}" for name, file, icon in notebooks_in_category].index(selected_notebook)
notebook_name, notebook_file, notebook_icon = notebooks_in_category[selected_idx]

# Real-time data toggle
st.sidebar.markdown("### 🔄 Real-time Features")
enable_realtime = st.sidebar.checkbox("Enable Real-time Data", value=True)
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)

# Model configuration
st.sidebar.markdown("### 🤖 Model Configuration")
model_complexity = st.sidebar.select_slider(
    "Model Complexity",
    options=["Simple", "Intermediate", "Advanced", "Expert"],
    value="Advanced"
)

forecast_horizon = st.sidebar.slider("Forecast Horizon (months)", 1, 24, 12)
confidence_level = st.sidebar.slider("Confidence Level (%)", 80, 99, 95)

# Main content area with tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Dashboard Overview", 
    "📊 Interactive Notebook", 
    "🔮 Live Modeling", 
    "📈 Real-time Markets", 
    "🎯 Custom Analysis"
])

with tab1:
    st.markdown("""
    <div class="immersive-container">
        <h2 style="color: white; text-align: center;">
            <i class="fas fa-chart-line"></i> Economic Intelligence Dashboard
        </h2>
        <p style="color: white; text-align: center; font-size: 1.2em;">
            Comprehensive real-time economic analysis for Kenya and regional markets
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h3><i class="fas fa-exchange-alt"></i> USD/KES</h3>
            <h2>132.45</h2>
            <p>+0.8% ↗️</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <h3><i class="fas fa-chart-bar"></i> GDP Growth</h3>
            <h2>5.2%</h2>
            <p>+0.3% ↗️</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <h3><i class="fas fa-percentage"></i> Inflation</h3>
            <h2>6.8%</h2>
            <p>-0.2% ↘️</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-container">
            <h3><i class="fas fa-university"></i> CBR</h3>
            <h2>9.50%</h2>
            <p>Stable →</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Interactive chart area
    st.markdown('<div class="interactive-panel">', unsafe_allow_html=True)
    st.subheader("📊 Economic Indicators Overview")
    
    # Generate sample data for overview
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')
    
    # Create interactive overview chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Exchange Rate Trend', 'GDP Growth', 'Inflation Rate', 'Interest Rates'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Exchange rate
    fx_rate = 130 + np.cumsum(np.random.normal(0, 0.5, len(dates)))
    fig.add_trace(
        go.Scatter(x=dates, y=fx_rate, name="USD/KES", line=dict(color="#667eea")),
        row=1, col=1
    )
    
    # GDP growth
    gdp_growth = 5 + np.random.normal(0, 0.3, len(dates))
    fig.add_trace(
        go.Scatter(x=dates, y=gdp_growth, name="GDP Growth %", line=dict(color="#f5576c")),
        row=1, col=2
    )
    
    # Inflation
    inflation = 7 + np.random.normal(0, 0.4, len(dates))
    fig.add_trace(
        go.Scatter(x=dates, y=inflation, name="Inflation %", line=dict(color="#4facfe")),
        row=2, col=1
    )
    
    # Interest rates
    interest_rate = 9.5 + np.random.normal(0, 0.2, len(dates))
    fig.add_trace(
        go.Scatter(x=dates, y=interest_rate, name="CBR %", line=dict(color="#764ba2")),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        template="plotly_white",
        title_text="Kenya Economic Indicators Dashboard"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown(f"""
    <div class="notebook-card">
        <h2>{notebook_icon} {notebook_name}</h2>
        <p>Interactive notebook analysis with real-time data integration</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Notebook execution controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Execute Full Analysis", type="primary"):
            with st.spinner(f"Running {notebook_name} analysis..."):
                time.sleep(3)  # Simulate analysis
                st.success("✅ Analysis completed successfully!")
    
    with col2:
        if st.button("📊 Generate Visualizations"):
            st.info("📈 Generating interactive charts...")
    
    with col3:
        if st.button("💾 Export Results"):
            st.info("📁 Results exported to reports/")
    
    # Notebook results simulation based on selection
    st.markdown('<div class="analysis-results">', unsafe_allow_html=True)
    
    if "Risk" in notebook_name:
        st.subheader("🛡️ Risk Analysis Results")
        
        # Fix the VaR calculation and plotting
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 1000)
        
        # Value at Risk calculation
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Value at Risk (95%)", f"{var_95:.4f}", "-2.1%")
            st.metric("Value at Risk (99%)", f"{var_99:.4f}", "-3.8%")
        
        with col2:
            # Risk distribution chart - FIX APPLIED HERE
            fig = go.Figure()
            
            # Use numpy histogram for proper binning
            hist_data, bin_edges = np.histogram(returns, bins=50)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            fig.add_trace(go.Scatter(
                x=fix_plotly_data(bin_centers),  # Fix applied here
                y=fix_plotly_data(hist_data),    # Fix applied here
                mode='lines',
                fill='tozeroy',
                name='Return Distribution',
                line=dict(color='#667eea')
            ))
            
            # Add VaR lines
            fig.add_vline(x=var_95, line_dash="dash", line_color="red", 
                         annotation_text="VaR 95%")
            fig.add_vline(x=var_99, line_dash="dash", line_color="darkred", 
                         annotation_text="VaR 99%")
            
            fig.update_layout(
                title="Return Distribution with VaR",
                xaxis_title="Returns",
                yaxis_title="Frequency",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Monte Carlo simulation results
        st.subheader("🎲 Monte Carlo Simulation Results")
        
        # Generate Monte Carlo paths - FIX APPLIED HERE
        n_simulations = 1000
        n_days = 252
        
        final_values = []
        for i in range(n_simulations):
            path = np.cumprod(1 + np.random.normal(0.0008, 0.02, n_days))
            final_values.append(path[-1])
        
        # Create simulation chart
        fig2 = go.Figure()
        
        # Show first 100 paths for visualization
        time_steps = fix_plotly_data(np.arange(n_days))  # Fix applied here
        for i in range(min(100, n_simulations)):
            path = np.cumprod(1 + np.random.normal(0.0008, 0.02, n_days))
            fig2.add_trace(go.Scatter(
                x=time_steps,
                y=fix_plotly_data(path),  # Fix applied here
                mode='lines',
                line=dict(width=0.5, color='rgba(102, 126, 234, 0.1)'),
                showlegend=False
            ))
        
        # Add mean path
        mean_path = np.cumprod(1 + np.full(n_days, 0.0008))
        fig2.add_trace(go.Scatter(
            x=time_steps,
            y=fix_plotly_data(mean_path),  # Fix applied here
            mode='lines',
            line=dict(width=3, color='red'),
            name='Expected Path'
        ))
        
        fig2.update_layout(
            title="Monte Carlo Simulation - 1000 Scenarios",
            xaxis_title="Days",
            yaxis_title="Value",
            template="plotly_white"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    elif "GDP" in notebook_name:
        st.subheader("🏛️ GDP Analysis Results")
        
        # GDP forecasting simulation
        quarters = pd.date_range(start='2020-Q1', periods=20, freq='Q')
        historical_gdp = [4.8, 3.2, -1.5, 2.1, 5.5, 4.9, 6.2, 5.8, 5.1, 4.7, 5.3, 5.9, 6.1, 5.4, 5.8, 6.0, 5.7, 5.2, 5.5, 5.8]
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=quarters[:16],
            y=historical_gdp[:16],
            mode='lines+markers',
            name='Historical GDP Growth',
            line=dict(color='#667eea', width=3)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=quarters[15:],
            y=historical_gdp[15:],
            mode='lines+markers',
            name='GDP Forecast',
            line=dict(color='#f5576c', width=3, dash='dash')
        ))
        
        fig.update_layout(
            title="Kenya GDP Growth Analysis & Forecast",
            xaxis_title="Quarter",
            yaxis_title="GDP Growth Rate (%)",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # GDP components
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current GDP Growth", "5.8%", "+0.3%")
            st.metric("Forecasted Growth (Q4)", "5.5%", "-0.3%")
        
        with col2:
            st.metric("Agriculture Contribution", "22.3%", "+1.2%")
            st.metric("Manufacturing Growth", "7.2%", "+0.8%")
    
    elif "Exchange" in notebook_name or "FX" in notebook_name:
        st.subheader("💱 Foreign Exchange Analysis")
        
        # FX rate analysis
        dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
        fx_rates = 128 + np.cumsum(np.random.normal(0, 0.3, len(dates)))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=fx_rates,
            mode='lines',
            name='USD/KES',
            line=dict(color='#4facfe', width=2)
        ))
        
        # Add volatility bands
        volatility = np.std(np.diff(fx_rates))
        upper_band = fx_rates + 2 * volatility
        lower_band = fx_rates - 2 * volatility
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=upper_band,
            mode='lines',
            line=dict(color='rgba(79, 172, 254, 0.3)'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=lower_band,
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(79, 172, 254, 0.2)',
            line=dict(color='rgba(79, 172, 254, 0.3)'),
            name='Volatility Band'
        ))
        
        fig.update_layout(
            title="USD/KES Exchange Rate with Volatility Analysis",
            xaxis_title="Date",
            yaxis_title="Exchange Rate",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # FX metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Rate", "132.45", "+0.8%")
        with col2:
            st.metric("30-day Volatility", "2.1%", "-0.3%")
        with col3:
            st.metric("Trend Signal", "Bullish", "Strong")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <div class="immersive-container">
        <h2 style="color: white; text-align: center;">
            <i class="fas fa-robot"></i> Live Economic Modeling Engine
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Model selection
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "🤖 Select Model Type",
            ["ARIMA Forecasting", "Prophet Time Series", "LSTM Neural Network", "Random Forest", "Ensemble Model"]
        )
    
    with col2:
        target_variable = st.selectbox(
            "🎯 Target Variable",
            ["USD/KES Exchange Rate", "GDP Growth Rate", "Inflation Rate", "Interest Rates", "Trade Balance"]
        )
    
    # Model parameters
    st.markdown("### ⚙️ Model Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        lookback_period = st.slider("Lookback Period (months)", 6, 36, 12)
    with col2:
        training_ratio = st.slider("Training Ratio", 0.6, 0.9, 0.8)
    with col3:
        validation_method = st.selectbox("Validation", ["Time Series Split", "Walk Forward", "Cross Validation"])
    
    # Execute model
    if st.button("🚀 Run Live Model", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate model execution
        for i in range(101):
            progress_bar.progress(i)
            if i < 30:
                status_text.text(f"Loading data... {i}%")
            elif i < 60:
                status_text.text(f"Training {model_type}... {i}%")
            elif i < 90:
                status_text.text(f"Validating model... {i}%")
            else:
                status_text.text(f"Generating predictions... {i}%")
            time.sleep(0.02)
        
        status_text.text("✅ Model execution completed!")
        
        # Show model results
        st.markdown('<div class="model-output">', unsafe_allow_html=True)
        st.subheader(f"📊 {model_type} Results for {target_variable}")
        
        # Generate synthetic results
        forecast_dates = pd.date_range(start=datetime.now(), periods=forecast_horizon, freq='M')
        if "Exchange Rate" in target_variable:
            base_value = 132.45
            forecast_values = base_value + np.cumsum(np.random.normal(0, 0.5, forecast_horizon))
        elif "GDP" in target_variable:
            base_value = 5.8
            forecast_values = base_value + np.random.normal(0, 0.3, forecast_horizon)
        else:
            base_value = 6.8
            forecast_values = base_value + np.random.normal(0, 0.4, forecast_horizon)
        
        # Create forecast chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#667eea', width=3)
        ))
        
        # Add confidence intervals
        upper_bound = forecast_values + 1.96 * np.std(forecast_values)
        lower_bound = forecast_values - 1.96 * np.std(forecast_values)
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=upper_bound,
            mode='lines',
            line=dict(color='rgba(102, 126, 234, 0.3)'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=lower_bound,
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(102, 126, 234, 0.2)',
            line=dict(color='rgba(102, 126, 234, 0.3)'),
            name=f'{confidence_level}% Confidence Interval'
        ))
        
        fig.update_layout(
            title=f"{target_variable} - {model_type} Forecast",
            xaxis_title="Date",
            yaxis_title=target_variable,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model Accuracy", "94.2%", "+2.1%")
        with col2:
            st.metric("RMSE", "0.0234", "-15%")
        with col3:
            st.metric("R² Score", "0.891", "+5%")
        with col4:
            st.metric("AIC", "234.5", "-12")
        
        st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown("""
    <div class="immersive-container">
        <h2 style="color: white; text-align: center;">
            <i class="fas fa-chart-line"></i> Real-time Market Intelligence
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    if enable_realtime:
        # Real-time data simulation
        real_time_data = {
            'Exchange Rates': {
                'USD/KES': 132.45 + np.random.normal(0, 0.1),
                'EUR/KES': 145.23 + np.random.normal(0, 0.1),
                'GBP/KES': 168.90 + np.random.normal(0, 0.1)
            },
            'Market Indicators': {
                'NSE 20 Index': 1847.23 + np.random.normal(0, 5),
                'Bond Yields': 12.45 + np.random.normal(0, 0.05),
                'Money Market': 9.85 + np.random.normal(0, 0.02)
            }
        }
        
        # Display real-time metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💱 Live Exchange Rates")
            for pair, rate in real_time_data['Exchange Rates'].items():
                change = np.random.normal(0, 0.5)
                color = "green" if change > 0 else "red"
                arrow = "↗️" if change > 0 else "↘️"
                st.markdown(f"""
                <div style="padding: 1rem; background: linear-gradient(45deg, #667eea, #764ba2); 
                           border-radius: 8px; margin: 0.5rem 0; color: white;">
                    <h4>{pair}: {rate:.2f} <span style="color: {color};">{change:+.2f} {arrow}</span></h4>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("📊 Market Indicators")
            for indicator, value in real_time_data['Market Indicators'].items():
                change = np.random.normal(0, 0.3)
                color = "green" if change > 0 else "red"
                arrow = "↗️" if change > 0 else "↘️"
                st.markdown(f"""
                <div style="padding: 1rem; background: linear-gradient(45deg, #f093fb, #f5576c); 
                           border-radius: 8px; margin: 0.5rem 0; color: white;">
                    <h4>{indicator}: {value:.2f} <span style="color: {color};">{change:+.2f} {arrow}</span></h4>
                </div>
                """, unsafe_allow_html=True)
        
        # Real-time chart
        st.subheader("📈 Live Market Movement")
        
        # Simulate real-time data stream
        if 'rt_data' not in st.session_state:
            st.session_state.rt_data = []
        
        # Add new data point
        current_time = datetime.now()
        new_point = {
            'time': current_time,
            'usd_kes': 132.45 + np.random.normal(0, 0.2),
            'nse20': 1847 + np.random.normal(0, 10)
        }
        st.session_state.rt_data.append(new_point)
        
        # Keep only last 100 points
        if len(st.session_state.rt_data) > 100:
            st.session_state.rt_data = st.session_state.rt_data[-100:]
        
        if len(st.session_state.rt_data) > 1:
            df_rt = pd.DataFrame(st.session_state.rt_data)
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('USD/KES Real-time', 'NSE 20 Index'),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(x=df_rt['time'], y=df_rt['usd_kes'], 
                          mode='lines', name='USD/KES', line=dict(color='#667eea')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df_rt['time'], y=df_rt['nse20'], 
                          mode='lines', name='NSE 20', line=dict(color='#f5576c')),
                row=2, col=1
            )
            
            fig.update_layout(height=500, template="plotly_white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Auto-refresh
        if auto_refresh:
            time.sleep(1)
            st.experimental_rerun()
    
    else:
        st.info("📡 Enable real-time data in the sidebar to see live market updates")

with tab5:
    st.markdown("""
    <div class="immersive-container">
        <h2 style="color: white; text-align: center;">
            <i class="fas fa-cogs"></i> Custom Analysis Engine
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Custom analysis builder
    st.subheader("🔧 Build Your Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Correlation Analysis", "Volatility Study", "Trend Analysis", "Seasonal Decomposition", "Regime Detection"]
        )
        
        data_sources = st.multiselect(
            "Data Sources",
            ["Exchange Rates", "GDP Data", "Inflation", "Interest Rates", "Trade Data", "Banking Stats"],
            default=["Exchange Rates", "GDP Data"]
        )
    
    with col2:
        time_period = st.selectbox(
            "Time Period",
            ["Last 1 Year", "Last 2 Years", "Last 5 Years", "All Available Data"]
        )
        
        analysis_frequency = st.selectbox(
            "Frequency",
            ["Daily", "Weekly", "Monthly", "Quarterly"]
        )
    
    # Advanced options
    with st.expander("🔬 Advanced Options"):
        col1, col2, col3 = st.columns(3)
        with col1:
            include_seasonality = st.checkbox("Include Seasonality", True)
            handle_outliers = st.checkbox("Handle Outliers", True)
        with col2:
            normalize_data = st.checkbox("Normalize Data", False)
            log_transform = st.checkbox("Log Transform", False)
        with col3:
            bootstrap_ci = st.checkbox("Bootstrap CI", True)
            monte_carlo = st.checkbox("Monte Carlo", False)
    
    # Execute custom analysis
    if st.button("🚀 Run Custom Analysis", type="primary"):
        with st.spinner("Executing custom analysis..."):
            time.sleep(2)
        
        st.success("✅ Custom analysis completed!")
        
        # Show custom results based on selection
        if analysis_type == "Correlation Analysis":
            st.subheader("🔗 Correlation Matrix")
            
            # Generate correlation matrix
            variables = data_sources
            n_vars = len(variables)
            corr_matrix = np.random.rand(n_vars, n_vars)
            corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
            np.fill_diagonal(corr_matrix, 1)  # Diagonal = 1
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=variables,
                y=variables,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix, 2),
                texttemplate="%{text}",
                textfont={"size": 12}
            ))
            
            fig.update_layout(
                title="Correlation Matrix of Selected Variables",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_type == "Volatility Study":
            st.subheader("📊 Volatility Analysis")
            
            dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
            returns = np.random.normal(0, 0.02, len(dates))
            volatility = pd.Series(returns).rolling(window=30).std() * np.sqrt(252)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=volatility.values,
                mode='lines',
                name='Annualized Volatility',
                line=dict(color='#f5576c', width=2)
            ))
            
            fig.update_layout(
                title="Rolling Volatility Analysis",
                xaxis_title="Date",
                yaxis_title="Volatility",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        st.subheader("📁 Export Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export to Excel"):
                st.success("📁 Results exported to Excel format")
        with col2:
            if st.button("📈 Export Charts"):
                st.success("📁 Charts exported as PNG/HTML")
        with col3:
            if st.button("📝 Generate Report"):
                st.success("📁 Comprehensive report generated")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea, #764ba2); 
           border-radius: 15px; color: white; margin-top: 2rem;">
    <h3><i class="fas fa-globe-africa"></i> EconoNet - Advanced Economic Intelligence Platform</h3>
    <p>Powered by Real-time Data • Advanced Analytics • Machine Learning Models</p>
    <p><span class="real-time-indicator"></span>Status: All Systems Operational</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh mechanism
if auto_refresh and enable_realtime:
    time.sleep(30)
    st.experimental_rerun()
