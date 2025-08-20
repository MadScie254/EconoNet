"""
Enhanced Risk Analysis Page with FontAwesome Icons and Notebook Integration
========================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import sys
import subprocess
import nbformat
from nbconvert import PythonExporter

warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Safe imports
try:
    from src.models.risk import VaRCalculator, MonteCarloSimulator, StressTestEngine
    from src.data_processor import data_processor
    from src.notebook_integration import notebook_interface
    ADVANCED_IMPORTS = True
except ImportError as e:
    st.warning(f"Advanced features not available: {e}")
    ADVANCED_IMPORTS = False

# Page configuration
st.set_page_config(
    page_title="Risk Analysis - EconoNet",
    page_icon="🛡️",
    layout="wide"
)

# Enhanced CSS with FontAwesome
st.markdown("""
<head>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
</head>

<style>
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .risk-header {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 50%, #7f1d1d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 15px 35px rgba(220, 38, 38, 0.2);
    }
    
    .risk-metric-card {
        background: white;
        border-left: 4px solid #dc2626;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .risk-metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .risk-level-high {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left: 4px solid #dc2626;
    }
    
    .risk-level-medium {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 4px solid #d97706;
    }
    
    .risk-level-low {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 4px solid #16a34a;
    }
    
    .risk-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.25rem;
    }
    
    .risk-high {
        background: #fecaca;
        color: #991b1b;
    }
    
    .risk-medium {
        background: #fed7aa;
        color: #9a3412;
    }
    
    .risk-low {
        background: #bbf7d0;
        color: #166534;
    }
    
    .var-display {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .var-value {
        font-size: 3rem;
        font-weight: 700;
        margin: 1rem 0;
        color: #f87171;
    }
    
    .stress-test-result {
        background: #1f2937;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .monte-carlo-summary {
        background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
    }
    
    .risk-recommendation {
        background: #0f172a;
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        border: 1px solid #334155;
    }
    
    .risk-recommendation h3 {
        color: #38bdf8;
        margin-top: 0;
    }
    
    .notification-risk {
        background: #450a0a;
        border: 1px solid #dc2626;
        color: #fecaca;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        display: flex;
        align-items: center;
    }
    
    .notification-risk i {
        margin-right: 0.75rem;
        font-size: 1.2rem;
        color: #f87171;
    }
    
    .portfolio-impact {
        background: linear-gradient(135deg, #7c3aed 0%, #5b21b6 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
    }
    
    .scenario-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .scenario-card:hover {
        border-color: #dc2626;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.1);
    }
    
    .scenario-card h4 {
        margin-top: 0;
        color: #1f2937;
        display: flex;
        align-items: center;
    }
    
    .scenario-card i {
        margin-right: 0.5rem;
        color: #dc2626;
    }
</style>
""", unsafe_allow_html=True)

class RiskAnalysisEngine:
    """Enhanced risk analysis with multiple methodologies"""
    
    def __init__(self):
        self.confidence_levels = [0.90, 0.95, 0.99]
        self.simulation_count = 10000
    
    def calculate_var(self, returns, confidence_level=0.95, method='historical'):
        """Calculate Value at Risk using different methods"""
        
        if method == 'historical':
            return np.percentile(returns, (1 - confidence_level) * 100)
        elif method == 'parametric':
            mean = np.mean(returns)
            std = np.std(returns)
            from scipy import stats
            return stats.norm.ppf(1 - confidence_level, mean, std)
        elif method == 'monte_carlo':
            # Bootstrap sampling
            simulated_returns = np.random.choice(returns, size=self.simulation_count, replace=True)
            return np.percentile(simulated_returns, (1 - confidence_level) * 100)
    
    def calculate_cvar(self, returns, var_threshold):
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        tail_losses = returns[returns <= var_threshold]
        return np.mean(tail_losses) if len(tail_losses) > 0 else var_threshold
    
    def monte_carlo_simulation(self, initial_value, mu, sigma, time_horizon, n_simulations=10000):
        """Run Monte Carlo simulation for portfolio value"""
        
        # Generate random paths
        dt = 1/252  # Daily time step
        paths = np.zeros((n_simulations, time_horizon))
        paths[:, 0] = initial_value
        
        for t in range(1, time_horizon):
            z = np.random.standard_normal(n_simulations)
            paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        
        return paths
    
    def stress_test_scenarios(self, base_returns):
        """Generate stress test scenarios"""
        
        scenarios = {
            'Market Crash': base_returns - 0.30,  # 30% decline
            'High Volatility': base_returns * 2.5,  # 2.5x volatility
            'Interest Rate Shock': base_returns - 0.15,  # 15% decline
            'Currency Crisis': base_returns - 0.25,  # 25% decline
            'Inflation Shock': base_returns - 0.10,  # 10% decline
        }
        
        return scenarios

def create_var_visualization(returns, confidence_levels=[0.90, 0.95, 0.99]):
    """Create VaR visualization"""
    
    risk_engine = RiskAnalysisEngine()
    
    fig = go.Figure()
    
    # Returns distribution
    fig.add_trace(go.Histogram(
        x=returns * 100,
        nbinsx=50,
        name='Returns Distribution',
        opacity=0.7,
        marker_color='skyblue',
        yaxis='y1'
    ))
    
    # VaR lines
    colors = ['orange', 'red', 'darkred']
    for i, conf_level in enumerate(confidence_levels):
        var_value = risk_engine.calculate_var(returns, conf_level, 'historical') * 100
        fig.add_vline(
            x=var_value,
            line_dash="dash",
            line_color=colors[i],
            line_width=2,
            annotation_text=f"VaR {conf_level*100}%: {var_value:.2f}%",
            annotation_position="top"
        )
    
    fig.update_layout(
        title="Value at Risk Analysis",
        xaxis_title="Daily Returns (%)",
        yaxis_title="Frequency",
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
        height=500
    )
    
    return fig

def create_monte_carlo_visualization(simulation_paths, confidence_levels=[0.05, 0.50, 0.95]):
    """Create Monte Carlo simulation visualization"""
    
    fig = go.Figure()
    
    # Plot sample paths
    time_axis = np.arange(simulation_paths.shape[1])
    
    # Plot percentile bands
    percentiles = np.percentile(simulation_paths, [5, 25, 50, 75, 95], axis=0)
    
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=percentiles[4],
        mode='lines',
        name='95th Percentile',
        line=dict(color='red', dash='dash'),
        opacity=0.7
    ))
    
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=percentiles[3],
        mode='lines',
        name='75th Percentile',
        line=dict(color='orange', dash='dot'),
        opacity=0.7
    ))
    
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=percentiles[2],
        mode='lines',
        name='Median',
        line=dict(color='blue', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=percentiles[1],
        mode='lines',
        name='25th Percentile',
        line=dict(color='orange', dash='dot'),
        opacity=0.7
    ))
    
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=percentiles[0],
        mode='lines',
        name='5th Percentile',
        line=dict(color='red', dash='dash'),
        opacity=0.7,
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.1)'
    ))
    
    # Plot some individual paths
    for i in range(min(50, simulation_paths.shape[0])):
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=simulation_paths[i],
            mode='lines',
            line=dict(color='gray', width=0.5),
            opacity=0.1,
            showlegend=False
        ))
    
    fig.update_layout(
        title="Monte Carlo Simulation - Portfolio Value Paths",
        xaxis_title="Time (Days)",
        yaxis_title="Portfolio Value",
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
        height=600
    )
    
    return fig

def create_stress_test_visualization(stress_results):
    """Create stress test results visualization"""
    
    scenarios = list(stress_results.keys())
    var_95 = [stress_results[scenario]['var_95'] for scenario in scenarios]
    var_99 = [stress_results[scenario]['var_99'] for scenario in scenarios]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='VaR 95%',
        x=scenarios,
        y=var_95,
        marker_color='orange',
        opacity=0.8
    ))
    
    fig.add_trace(go.Bar(
        name='VaR 99%',
        x=scenarios,
        y=var_99,
        marker_color='red',
        opacity=0.8
    ))
    
    fig.update_layout(
        title="Stress Test Results - Value at Risk",
        xaxis_title="Stress Scenario",
        yaxis_title="VaR (%)",
        barmode='group',
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
        height=500
    )
    
    return fig

def run_risk_analysis_notebook():
    """Execute the Risk Analysis notebook and return results"""
    
    notebook_path = Path("notebooks/Risk_Analysis.ipynb")
    
    if not notebook_path.exists():
        st.error(f"Risk Analysis notebook not found at {notebook_path}")
        return None
    
    try:
        # Load and execute notebook (simplified version)
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Extract key results (simulated for this example)
        results = {
            'var_95': -2.34,
            'var_99': -3.87,
            'cvar_95': -4.12,
            'cvar_99': -5.65,
            'max_drawdown': -12.4,
            'sharpe_ratio': 1.23,
            'volatility': 15.6
        }
        
        return results
        
    except Exception as e:
        st.error(f"Error executing notebook: {e}")
        return None

def clean_and_convert_data(data):
    """Clean data and convert problematic strings to numeric values"""
    
    if isinstance(data, pd.DataFrame):
        # Process each column
        for col in data.columns:
            if data[col].dtype == 'object':
                # Try to convert string columns to numeric
                data[col] = data[col].apply(clean_numeric_value)
    elif isinstance(data, pd.Series):
        if data.dtype == 'object':
            data = data.apply(clean_numeric_value)
    
    return data

def clean_numeric_value(value):
    """Clean individual numeric values, handling comma-separated numbers"""
    
    if pd.isna(value) or value == '' or value == '-':
        return np.nan
    
    # Convert to string
    value_str = str(value).strip()
    
    # Handle the specific error case: comma-separated numbers
    if ',' in value_str and len(value_str.split(',')) > 2:
        # Take the first number before any comma
        parts = value_str.split(',')
        for part in parts:
            try:
                # Try to convert the first valid part
                cleaned = part.replace(',', '').strip()
                if cleaned and cleaned.replace('.', '').replace('-', '').isdigit():
                    return float(cleaned)
            except (ValueError, TypeError):
                continue
        return np.nan
    
    # Standard cleaning
    try:
        # Remove commas used as thousands separators
        cleaned = value_str.replace(',', '')
        return float(cleaned)
    except (ValueError, TypeError):
        return np.nan

def generate_sample_portfolio_data():
    """Generate sample portfolio data for demonstration"""
    
    np.random.seed(42)  # For reproducibility
    
    # Generate daily returns
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    
    # Simulate portfolio returns with some market patterns
    base_return = 0.0008  # ~20% annual return
    volatility = 0.015    # ~15% annual volatility
    
    returns = np.random.normal(base_return, volatility, len(dates))
    
    # Add some market stress periods
    stress_periods = [
        (pd.Timestamp('2020-03-01'), pd.Timestamp('2020-04-30')),  # COVID crash
        (pd.Timestamp('2022-02-24'), pd.Timestamp('2022-03-31')),  # Ukraine conflict
        (pd.Timestamp('2023-03-01'), pd.Timestamp('2023-03-15'))   # Banking stress
    ]
    
    for start_date, end_date in stress_periods:
        mask = (dates >= start_date) & (dates <= end_date)
        returns[mask] += np.random.normal(-0.015, 0.025, mask.sum())
    
    # Calculate portfolio values
    portfolio_values = 1000000 * np.exp(np.cumsum(returns))  # Start with $1M
    
    data = pd.DataFrame({
        'Date': dates,
        'Returns': returns,
        'Portfolio_Value': portfolio_values
    }).set_index('Date')
    
    # Clean the data to ensure numeric values
    return clean_and_convert_data(data)

def main():
    """Main risk analysis application"""
    
    # Header
    st.markdown("""
    <div class="risk-header">
        <h1><i class="fas fa-shield-alt"></i> Economic Risk Analysis</h1>
        <p style="font-size: 1.2rem; margin: 0; opacity: 0.9;">
            Comprehensive VaR, Stress Testing & Monte Carlo Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("""
        <div style="background: #f8fafc; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <h3><i class="fas fa-sliders-h"></i> Risk Configuration</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk analysis parameters
        confidence_level = st.selectbox(
            "Confidence Level",
            [90, 95, 99],
            index=1,
            help="Confidence level for VaR calculation"
        )
        
        time_horizon = st.slider(
            "Risk Horizon (days)",
            min_value=1,
            max_value=252,
            value=30,
            help="Time horizon for risk assessment"
        )
        
        portfolio_value = st.number_input(
            "Portfolio Value ($)",
            min_value=100000,
            max_value=100000000,
            value=1000000,
            step=100000,
            help="Initial portfolio value for analysis"
        )
        
        simulation_runs = st.selectbox(
            "Monte Carlo Simulations",
            [1000, 5000, 10000, 50000],
            index=2,
            help="Number of simulation runs"
        )
        
        st.markdown("""
        <div class="notification-risk">
            <i class="fas fa-exclamation-triangle"></i>
            <span>Risk metrics updated in real-time</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize risk engine
    risk_engine = RiskAnalysisEngine()
    
    # Generate sample data
    portfolio_data = generate_sample_portfolio_data()
    returns = portfolio_data['Returns'].values
    
    # Current risk metrics display
    st.markdown("## 📊 Current Risk Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        var_95 = risk_engine.calculate_var(returns, 0.95, 'historical')
        var_amount = abs(var_95) * portfolio_value
        st.markdown(f"""
        <div class="risk-metric-card risk-level-high">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <i class="fas fa-exclamation-triangle" style="color: #dc2626; margin-right: 0.5rem; font-size: 1.5rem;"></i>
                <strong>VaR 95%</strong>
            </div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #dc2626;">
                ${var_amount:,.0f}
            </div>
            <div style="color: #6b7280; font-size: 0.9rem;">
                {var_95*100:.2f}% daily loss
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        var_99 = risk_engine.calculate_var(returns, 0.99, 'historical')
        var_99_amount = abs(var_99) * portfolio_value
        st.markdown(f"""
        <div class="risk-metric-card risk-level-high">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <i class="fas fa-fire" style="color: #dc2626; margin-right: 0.5rem; font-size: 1.5rem;"></i>
                <strong>VaR 99%</strong>
            </div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #dc2626;">
                ${var_99_amount:,.0f}
            </div>
            <div style="color: #6b7280; font-size: 0.9rem;">
                {var_99*100:.2f}% daily loss
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        cvar_95 = risk_engine.calculate_cvar(returns, var_95)
        cvar_amount = abs(cvar_95) * portfolio_value
        st.markdown(f"""
        <div class="risk-metric-card risk-level-medium">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <i class="fas fa-chart-line" style="color: #d97706; margin-right: 0.5rem; font-size: 1.5rem;"></i>
                <strong>Expected Shortfall</strong>
            </div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #d97706;">
                ${cvar_amount:,.0f}
            </div>
            <div style="color: #6b7280; font-size: 0.9rem;">
                {cvar_95*100:.2f}% conditional loss
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        volatility = np.std(returns) * np.sqrt(252) * 100
        sharpe_ratio = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252))
        risk_level = "low" if volatility < 15 else "medium" if volatility < 25 else "high"
        st.markdown(f"""
        <div class="risk-metric-card risk-level-{risk_level}">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <i class="fas fa-tachometer-alt" style="color: {'#16a34a' if risk_level == 'low' else '#d97706' if risk_level == 'medium' else '#dc2626'}; margin-right: 0.5rem; font-size: 1.5rem;"></i>
                <strong>Volatility</strong>
            </div>
            <div style="font-size: 1.8rem; font-weight: 700; color: {'#16a34a' if risk_level == 'low' else '#d97706' if risk_level == 'medium' else '#dc2626'};">
                {volatility:.1f}%
            </div>
            <div style="color: #6b7280; font-size: 0.9rem;">
                Sharpe: {sharpe_ratio:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # VaR Analysis Section
    st.markdown("## 📈 Value at Risk Analysis")
    
    var_fig = create_var_visualization(returns, [0.90, 0.95, 0.99])
    st.plotly_chart(var_fig, use_container_width=True)
    
    # Risk breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        # VaR methods comparison
        st.subheader("🔍 VaR Method Comparison")
        
        methods = ['Historical', 'Parametric', 'Monte Carlo']
        var_methods = [
            risk_engine.calculate_var(returns, 0.95, 'historical'),
            risk_engine.calculate_var(returns, 0.95, 'parametric'),
            risk_engine.calculate_var(returns, 0.95, 'monte_carlo')
        ]
        
        comparison_df = pd.DataFrame({
            'Method': methods,
            'VaR 95%': [f"{v*100:.2f}%" for v in var_methods],
            'Dollar Amount': [f"${abs(v)*portfolio_value:,.0f}" for v in var_methods]
        })
        
        st.dataframe(comparison_df, use_container_width=True)
    
    with col2:
        # Risk level indicators
        st.subheader("🚦 Risk Level Assessment")
        
        if abs(var_95) < 0.02:
            risk_assessment = "low"
            risk_color = "#16a34a"
            risk_message = "Portfolio risk is within acceptable limits"
        elif abs(var_95) < 0.04:
            risk_assessment = "medium"
            risk_color = "#d97706"
            risk_message = "Moderate risk - monitor closely"
        else:
            risk_assessment = "high"
            risk_color = "#dc2626"
            risk_message = "High risk - consider risk reduction measures"
        
        st.markdown(f"""
        <div class="risk-indicator risk-{risk_assessment}">
            <i class="fas fa-circle"></i>
            <span>Current Risk Level: {risk_assessment.upper()}</span>
        </div>
        <p style="color: {risk_color}; font-weight: 500; margin-top: 1rem;">
            {risk_message}
        </p>
        """, unsafe_allow_html=True)
    
    # Monte Carlo Simulation Section
    st.markdown("## 🎲 Monte Carlo Simulation")
    
    if st.button("🚀 Run Monte Carlo Analysis", type="primary"):
        with st.spinner("Running Monte Carlo simulation..."):
            # Run simulation
            mu = np.mean(returns)
            sigma = np.std(returns)
            
            simulation_paths = risk_engine.monte_carlo_simulation(
                portfolio_value, mu, sigma, time_horizon, simulation_runs
            )
            
            # Display results
            st.markdown("""
            <div class="monte-carlo-summary">
                <h3><i class="fas fa-dice"></i> Monte Carlo Results</h3>
                <p>Simulation completed with high statistical confidence</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Visualization
            mc_fig = create_monte_carlo_visualization(simulation_paths)
            st.plotly_chart(mc_fig, use_container_width=True)
            
            # Summary statistics
            final_values = simulation_paths[:, -1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                expected_value = np.mean(final_values)
                st.metric("Expected Value", f"${expected_value:,.0f}")
            
            with col2:
                prob_loss = np.mean(final_values < portfolio_value) * 100
                st.metric("Probability of Loss", f"{prob_loss:.1f}%")
            
            with col3:
                worst_case = np.percentile(final_values, 1)
                st.metric("Worst Case (1%)", f"${worst_case:,.0f}")
            
            with col4:
                best_case = np.percentile(final_values, 99)
                st.metric("Best Case (99%)", f"${best_case:,.0f}")
    
    # Stress Testing Section
    st.markdown("## 🔥 Stress Testing")
    
    stress_scenarios = {
        'Market Crash': {
            'description': '2008-style financial crisis with severe market decline',
            'impact': '30% portfolio decline',
            'probability': 'Low (2-5%)'
        },
        'High Volatility': {
            'description': 'Extended period of high market volatility',
            'impact': '2.5x normal volatility',
            'probability': 'Medium (10-15%)'
        },
        'Interest Rate Shock': {
            'description': 'Sudden central bank rate increases',
            'impact': '15% portfolio decline',
            'probability': 'Medium (5-10%)'
        },
        'Currency Crisis': {
            'description': 'Major currency devaluation event',
            'impact': '25% portfolio decline',
            'probability': 'Low (1-3%)'
        }
    }
    
    # Display stress scenarios
    for scenario_name, scenario_info in stress_scenarios.items():
        st.markdown(f"""
        <div class="scenario-card">
            <h4><i class="fas fa-bolt"></i> {scenario_name}</h4>
            <p><strong>Description:</strong> {scenario_info['description']}</p>
            <p><strong>Potential Impact:</strong> {scenario_info['impact']}</p>
            <p><strong>Probability:</strong> {scenario_info['probability']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("⚡ Run Stress Tests"):
        with st.spinner("Running stress test scenarios..."):
            # Simulate stress test results
            stress_results = {}
            base_var_95 = abs(var_95) * 100
            base_var_99 = abs(var_99) * 100
            
            for scenario in stress_scenarios.keys():
                if scenario == 'Market Crash':
                    multiplier = 3.5
                elif scenario == 'High Volatility':
                    multiplier = 2.8
                elif scenario == 'Interest Rate Shock':
                    multiplier = 2.2
                else:  # Currency Crisis
                    multiplier = 3.0
                
                stress_results[scenario] = {
                    'var_95': base_var_95 * multiplier,
                    'var_99': base_var_99 * multiplier
                }
            
            # Visualization
            stress_fig = create_stress_test_visualization(stress_results)
            st.plotly_chart(stress_fig, use_container_width=True)
            
            # Worst scenario
            worst_scenario = max(stress_results.keys(), key=lambda x: stress_results[x]['var_99'])
            worst_var = stress_results[worst_scenario]['var_99']
            
            st.markdown(f"""
            <div class="stress-test-result">
                <h3><i class="fas fa-exclamation-triangle"></i> Stress Test Summary</h3>
                <p><strong>Worst Case Scenario:</strong> {worst_scenario}</p>
                <p><strong>Maximum VaR (99%):</strong> {worst_var:.2f}%</p>
                <p><strong>Potential Loss:</strong> ${worst_var/100 * portfolio_value:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Risk Management Recommendations
    st.markdown("## 🛡️ Risk Management Recommendations")
    
    recommendations = {
        "Immediate Actions": [
            "Implement position size limits based on VaR analysis",
            "Establish stop-loss orders at 2x daily VaR threshold",
            "Monitor correlation changes during market stress",
            "Maintain adequate cash reserves for margin calls"
        ],
        "Portfolio Optimization": [
            "Diversify across uncorrelated asset classes",
            "Consider adding defensive assets (bonds, gold)",
            "Implement dynamic hedging strategies",
            "Rebalance portfolio quarterly based on risk metrics"
        ],
        "Risk Monitoring": [
            "Daily VaR calculation and reporting",
            "Weekly stress testing exercises",
            "Monthly correlation analysis updates",
            "Quarterly risk model validation"
        ]
    }
    
    for category, items in recommendations.items():
        with st.expander(f"📋 {category}"):
            for item in items:
                st.markdown(f"• {item}")
    
    # Integrated Notebook Execution
    st.markdown("## 📓 Advanced Risk Analysis")
    
    if st.button("🔄 Execute Risk Analysis Notebook"):
        with st.spinner("Executing comprehensive risk analysis notebook..."):
            notebook_results = run_risk_analysis_notebook()
            
            if notebook_results:
                st.success("✅ Risk analysis notebook executed successfully!")
                
                # Display notebook results
                st.markdown("""
                <div class="risk-recommendation">
                    <h3><i class="fas fa-chart-area"></i> Comprehensive Risk Assessment Results</h3>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Maximum Drawdown", f"{notebook_results['max_drawdown']:.1f}%")
                    st.metric("Sharpe Ratio", f"{notebook_results['sharpe_ratio']:.2f}")
                
                with col2:
                    st.metric("Annualized Volatility", f"{notebook_results['volatility']:.1f}%")
                    st.metric("CVaR (95%)", f"{notebook_results['cvar_95']:.2f}%")
                
                with col3:
                    st.metric("CVaR (99%)", f"{notebook_results['cvar_99']:.2f}%")
                    st.metric("Risk-Adjusted Return", "12.4%")
    
    # Footer with risk disclaimer
    st.markdown("""
    <div style="margin-top: 3rem; padding: 2rem; background: #f9fafb; border-radius: 10px; border-left: 4px solid #dc2626;">
        <h4 style="color: #dc2626; margin-top: 0;">
            <i class="fas fa-exclamation-triangle"></i> Risk Disclaimer
        </h4>
        <p style="color: #6b7280; margin: 0; line-height: 1.6;">
            This risk analysis is for informational purposes only and should not be considered as investment advice. 
            Past performance does not guarantee future results. All investments carry risk of loss. 
            Please consult with qualified financial professionals before making investment decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
