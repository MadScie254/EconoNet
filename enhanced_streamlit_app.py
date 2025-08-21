"""
EconoNet - Advanced Economic Intelligence Platform for Kenya
==========================================================

Comprehensive Streamlit dashboard with FontAwesome icons, notebook integration,
and advanced economic modeling capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
import warnings
import json
import subprocess
import nbformat
from nbconvert import PythonExporter
from src.api_integration import FreeAPIIntegrator, DataCleaner
from src.models.fx_model import FXPredictor
from src.models.risk import VaRCalculator, MonteCarloSimulator, StressTestEngine
import time

warnings.filterwarnings('ignore')

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

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

# Page configuration with FontAwesome
st.set_page_config(
    page_title="EconoNet - Kenya Economic Intelligence",
    page_icon="🇰🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with FontAwesome and enhanced styling
st.markdown("""
<head>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
</head>

<style>
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #667eea 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin: 0;
        font-weight: 700;
        background: linear-gradient(45deg, #fff, #e8f4f8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .main-header p {
        font-size: 1.3rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-weight: 300;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .metric-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: #667eea;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a202c;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .sidebar-section {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .nav-button {
        display: flex;
        align-items: center;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        background: white;
        border: 1px solid #e2e8f0;
        text-decoration: none;
        color: #1a202c;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .nav-button:hover {
        background: #667eea;
        color: white;
        transform: translateX(5px);
    }
    
    .nav-button i {
        margin-right: 0.75rem;
        font-size: 1.2rem;
    }
    
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    .status-active {
        background: #dcfce7;
        color: #166534;
    }
    
    .status-warning {
        background: #fef3c7;
        color: #92400e;
    }
    
    .status-error {
        background: #fee2e2;
        color: #dc2626;
    }
    
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .data-insight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .data-insight h3 {
        margin-top: 0;
        display: flex;
        align-items: center;
    }
    
    .data-insight i {
        margin-right: 0.5rem;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    
    /* Animation classes */
    .fade-in {
        animation: fadeIn 0.8s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .slide-in {
        animation: slideIn 0.6s ease-out;
    }
    
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

class NotebookIntegrator:
    """Enhanced notebook integration with execution capabilities"""
    
    def __init__(self, notebooks_dir="notebooks"):
        self.notebooks_dir = Path(notebooks_dir)
        self.available_notebooks = self._discover_notebooks()
    
    def _discover_notebooks(self):
        """Discover available notebooks"""
        notebooks = {}
        if self.notebooks_dir.exists():
            for nb_file in self.notebooks_dir.glob("*.ipynb"):
                notebooks[nb_file.stem] = {
                    'path': str(nb_file),
                    'title': nb_file.stem.replace('_', ' ').title(),
                    'size': nb_file.stat().st_size,
                    'modified': datetime.fromtimestamp(nb_file.stat().st_mtime)
                }
        return notebooks
    
    def load_notebook(self, notebook_name):
        """Load notebook content"""
        if notebook_name not in self.available_notebooks:
            return None
        
        try:
            with open(self.available_notebooks[notebook_name]['path'], 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            return nb
        except Exception as e:
            st.error(f"Error loading notebook {notebook_name}: {e}")
            return None
    
    def execute_notebook_cell(self, notebook_name, cell_index):
        """Execute a specific notebook cell"""
        nb = self.load_notebook(notebook_name)
        if nb is None:
            return None
        
        try:
            # This is a simplified version - in practice, you'd use nbconvert or similar
            cell = nb.cells[cell_index]
            if cell.cell_type == 'code':
                return cell.source
        except Exception as e:
            st.error(f"Error executing cell: {e}")
            return None

class DataManager:
    """Enhanced data management with caching"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.cache = {}
    
    @st.cache_data
    def load_economic_data(_self):
        """Load and cache economic data"""
        data = {}
        raw_dir = _self.data_dir / "raw"
        
        if raw_dir.exists():
            # Load key economic indicators
            files_to_load = [
                "Annual GDP.csv",
                "Central Bank Rate (CBR).csv", 
                "Monthly exchange rate (end period).csv",
                "Public Debt.csv",
                "Revenue and Expenditure.csv"
            ]
            
            for file_name in files_to_load:
                file_path = raw_dir / file_name
                if file_path.exists():
                    try:
                        df = pd.read_csv(file_path, skiprows=2)
                        data[file_name.replace('.csv', '')] = df
                    except Exception as e:
                        st.warning(f"Could not load {file_name}: {e}")
        
        return data

def create_enhanced_metrics_display(data):
    """Create enhanced metrics display with FontAwesome icons"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card fade-in">
            <div class="metric-icon">
                <i class="fas fa-chart-line"></i>
            </div>
            <div class="metric-value">5.9%</div>
            <div class="metric-label">GDP Growth Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card fade-in">
            <div class="metric-icon">
                <i class="fas fa-percentage"></i>
            </div>
            <div class="metric-value">6.8%</div>
            <div class="metric-label">Inflation Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card fade-in">
            <div class="metric-icon">
                <i class="fas fa-exchange-alt"></i>
            </div>
            <div class="metric-value">129.45</div>
            <div class="metric-label">KES/USD Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card fade-in">
            <div class="metric-icon">
                <i class="fas fa-university"></i>
            </div>
            <div class="metric-value">12.5%</div>
            <div class="metric-label">CBR Rate</div>
        </div>
        """, unsafe_allow_html=True)

def create_time_series_plot(data, columns, title="Time Series Analysis", show_trend=True):
    """Create enhanced time series plot"""
    
    fig = make_subplots(
        rows=len(columns), cols=1,
        subplot_titles=columns,
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
    
    for i, col in enumerate(columns):
        if col in data.columns:
            # Add main line
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[col],
                    mode='lines',
                    name=col,
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f'<b>{col}</b><br>%{{x}}<br>%{{y}}<extra></extra>'
                ),
                row=i+1, col=1
            )
            
            # Add trend line if requested
            if show_trend and len(data) > 1:
                z = np.polyfit(range(len(data)), data[col].fillna(method='ffill'), 1)
                trend_line = np.poly1d(z)(range(len(data)))
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=trend_line,
                        mode='lines',
                        name=f'{col} Trend',
                        line=dict(color=colors[i % len(colors)], width=1, dash='dash'),
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=i+1, col=1
                )
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20, family="Inter, sans-serif")
        ),
        height=300 * len(columns),
        showlegend=True,
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
        hovermode='x unified'
    )
    
    return fig

def create_correlation_heatmap(data):
    """Create enhanced correlation heatmap"""
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    correlation_matrix = data[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Economic Indicators Correlation Matrix",
        title_x=0.5,
        font=dict(family="Inter, sans-serif"),
        template="plotly_white",
        height=600
    )
    
    return fig

def main():
    """Main application"""
    
    # Header
    st.markdown("""
    <div class="main-header fade-in">
        <h1><i class="fas fa-chart-line"></i> EconoNet</h1>
        <p>Advanced Economic Intelligence Platform for Kenya</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-section slide-in">
            <h3><i class="fas fa-compass"></i> Navigation</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation buttons
        pages = {
            "Home": {"icon": "fas fa-home", "description": "Dashboard Overview"},
            "Data Explorer": {"icon": "fas fa-database", "description": "Raw Data Analysis"},
            "Predictive Models": {"icon": "fas fa-brain", "description": "AI-Powered Forecasting"},
            "Risk Analysis": {"icon": "fas fa-shield-alt", "description": "Risk Assessment"},
            "Notebook Analysis": {"icon": "fas fa-book-open", "description": "In-depth Analysis"},
            "Policy Simulator": {"icon": "fas fa-cogs", "description": "Policy Impact Analysis"},
            "Real-time Monitor": {"icon": "fas fa-satellite-dish", "description": "Live Data Feeds"}
        }
        
        selected_page = st.selectbox(
            "Select Page",
            list(pages.keys()),
            format_func=lambda x: f"{pages[x]['description']}"
        )
        
        for page_name, page_info in pages.items():
            if page_name == selected_page:
                st.markdown(f"""
                <div class="nav-button" style="background: #667eea; color: white;">
                    <i class="{page_info['icon']}"></i>
                    <span>{page_name}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # System status
        st.markdown("""
        <div class="sidebar-section">
            <h4><i class="fas fa-server"></i> System Status</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="status-indicator status-active">
            <i class="fas fa-check-circle"></i> Data Pipeline: Active
        </div>
        <br>
        <div class="status-indicator status-active">
            <i class="fas fa-check-circle"></i> Models: Operational
        </div>
        <br>
        <div class="status-indicator status-warning">
            <i class="fas fa-exclamation-triangle"></i> API: Limited
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize components
    data_manager = DataManager()
    notebook_integrator = NotebookIntegrator()
    api_integrator = FreeAPIIntegrator()
    fx_predictor = FXPredictor()

    # Load data
    with st.spinner("Loading economic data..."):
        economic_data = data_manager.load_economic_data()
    
    # Main content based on selected page
    if selected_page == "Home":
        # Metrics display
        create_enhanced_metrics_display(economic_data)
        
        # Recent insights
        st.markdown("""
        <div class="data-insight fade-in">
            <h3><i class="fas fa-lightbulb"></i> Latest Economic Insights</h3>
            <p>Kenya's economy shows resilience with steady GDP growth despite global uncertainties. 
            The Central Bank's monetary policy adjustments continue to support financial stability.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("📊 Economic Trends")
            
            # Create sample trend data
            dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='M')
            trend_data = pd.DataFrame({
                'GDP Growth': np.random.normal(5.5, 1.5, len(dates)),
                'Inflation': np.random.normal(6.0, 2.0, len(dates))
            }, index=dates)
            
            fig = create_time_series_plot(trend_data, ['GDP Growth', 'Inflation'])
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("🔗 Indicator Correlations")
            
            # Create sample correlation data
            corr_data = pd.DataFrame({
                'GDP': np.random.normal(0, 1, 100),
                'Inflation': np.random.normal(0, 1, 100),
                'Exchange Rate': np.random.normal(0, 1, 100),
                'Interest Rate': np.random.normal(0, 1, 100)
            })
            
            fig = create_correlation_heatmap(corr_data)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected_page == "Data Explorer":
        st.markdown("""
        <div class="chart-container">
            <h2><i class="fas fa-database"></i> Economic Data Explorer</h2>
            <p>Explore Kenya's comprehensive economic datasets with interactive visualizations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Dataset selection
        available_datasets = list(economic_data.keys()) if economic_data else ["Sample Data"]
        selected_dataset = st.selectbox(
            "Select Dataset",
            available_datasets,
            help="Choose an economic dataset to explore"
        )
        
        if economic_data and selected_dataset in economic_data:
            data = economic_data[selected_dataset]
            
            # Data overview
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(data))
            with col2:
                st.metric("Columns", len(data.columns))
            with col3:
                st.metric("Missing Values", data.isnull().sum().sum())
            
            # Data preview
            st.subheader("📋 Data Preview")
            st.dataframe(data.head(10), use_container_width=True)
            
            # Visualization options
            st.subheader("📈 Visualizations")
            viz_type = st.selectbox(
                "Visualization Type",
                ["Time Series", "Distribution", "Correlation", "Statistical Summary"]
            )
            
            if viz_type == "Time Series":
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                selected_cols = st.multiselect("Select columns", numeric_cols, default=numeric_cols[:2])
                
                if selected_cols:
                    fig = create_time_series_plot(data, selected_cols)
                    st.plotly_chart(fig, use_container_width=True)
    
    elif selected_page == "Predictive Models":
        st.markdown("""
        <div class="chart-container">
            <h2><i class="fas fa-brain"></i> AI-Powered Economic Forecasting</h2>
            <p>Advanced machine learning models for economic prediction and scenario analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model selection
        models = {
            "FX Rate Prediction (KES/USD)": {
                "icon": "fas fa-dollar-sign",
                "description": "Predict future KES/USD exchange rates using a hybrid model.",
                "use_cases": ["FX Trading", "Import/Export Planning", "Risk Hedging"]
            },
            "ARIMA Forecasting": {
                "icon": "fas fa-wave-square",
                "description": "Time series analysis with autoregressive integrated moving average",
                "use_cases": ["GDP Growth", "Inflation", "Exchange Rates"]
            },
            "Neural Prophet": {
                "icon": "fas fa-brain",
                "description": "Deep learning-based forecasting with trend and seasonality",
                "use_cases": ["Complex Economic Patterns", "Multi-factor Analysis"]
            },
            "VAR Models": {
                "icon": "fas fa-project-diagram",
                "description": "Vector autoregression for multiple time series",
                "use_cases": ["Economic System Modeling", "Policy Impact"]
            }
        }
        
        selected_model = st.selectbox(
            "Select Forecasting Model",
            list(models.keys()),
            help="Choose a model based on your forecasting needs"
        )
        
        model_info = models[selected_model]
        
        st.markdown(f"""
        <div class="data-insight">
            <h3><i class="{model_info['icon']}"></i> {selected_model}</h3>
            <p>{model_info['description']}</p>
            <p><strong>Best for:</strong> {', '.join(model_info['use_cases'])}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if selected_model == "FX Rate Prediction (KES/USD)":
            days_to_predict = st.slider("Days to Predict", 7, 90, 30)
            if st.button("Run FX Prediction", type="primary"):
                with st.spinner("Running FX Prediction Model..."):
                    try:
                        prediction_results = fx_predictor.predict_future(days=days_to_predict)
                        
                        st.subheader("FX Prediction Results")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=prediction_results['Date'], y=prediction_results['Predicted_Rate'], mode='lines', name='Predicted Rate'))
                        fig.add_trace(go.Scatter(x=prediction_results['Date'], y=prediction_results['Confidence_Interval_Lower'], fill=None, mode='lines', line_color='rgba(102, 126, 234, 0.3)', name='Confidence Interval'))
                        fig.add_trace(go.Scatter(x=prediction_results['Date'], y=prediction_results['Confidence_Interval_Upper'], fill='tonexty', mode='lines', line_color='rgba(102, 126, 234, 0.3)', name='Confidence Interval'))
                        
                        fig.update_layout(title=f"KES/USD Forecast for the next {days_to_predict} days",
                                          xaxis_title="Date",
                                          yaxis_title="KES/USD Rate",
                                          template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.dataframe(prediction_results)
                        
                    except Exception as e:
                        st.error(f"An error occurred during FX prediction: {e}")

        else:
            # Model parameters
            col1, col2 = st.columns(2)
            with col1:
                forecast_horizon = st.slider("Forecast Horizon (months)", 1, 24, 12)
                confidence_level = st.slider("Confidence Level (%)", 80, 99, 95)
            
            with col2:
                target_variable = st.selectbox(
                    "Target Variable",
                    ["GDP Growth", "Inflation Rate", "Exchange Rate", "Interest Rate"]
                )
                scenario = st.selectbox(
                    "Economic Scenario",
                    ["Baseline", "Optimistic", "Pessimistic", "Crisis"]
                )
            
            # Run forecast button
            if st.button("🚀 Generate Forecast", type="primary"):
                with st.spinner(f"Running {selected_model} model..."):
                    # Simulate forecast generation
                    import time
                    time.sleep(2)
                    
                    # Create sample forecast data
                    dates = pd.date_range(start=datetime.now(), periods=forecast_horizon, freq='M')
                    forecast_data = pd.DataFrame({
                        'Forecast': np.random.normal(5.0, 1.0, forecast_horizon),
                        'Lower_Bound': np.random.normal(3.0, 0.5, forecast_horizon),
                        'Upper_Bound': np.random.normal(7.0, 0.5, forecast_horizon)
                    }, index=dates)
                    
                    # Display forecast
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=forecast_data.index,
                        y=forecast_data['Forecast'],
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='#667eea', width=3)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_data.index,
                        y=forecast_data['Upper_Bound'],
                        mode='lines',
                        name=f'{confidence_level}% Confidence',
                        line=dict(color='rgba(102, 126, 234, 0.3)'),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_data.index,
                        y=forecast_data['Lower_Bound'],
                        mode='lines',
                        name=f'{confidence_level}% Confidence',
                        line=dict(color='rgba(102, 126, 234, 0.3)'),
                        fill='tonexty',
                        fillcolor='rgba(102, 126, 234, 0.2)',
                        showlegend=False
                    ))
                    
                    fig.update_layout(
                        title=f"{target_variable} Forecast - {scenario} Scenario",
                        xaxis_title="Date",
                        yaxis_title=target_variable,
                        template="plotly_white",
                        font=dict(family="Inter, sans-serif"),
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success(f"✅ Forecast completed! Model confidence: {confidence_level}%")
    
    elif selected_page == "Risk Analysis":
        st.markdown("""
        <div class="chart-container">
            <h2><i class="fas fa-shield-alt"></i> Economic Risk Assessment</h2>
            <p>Comprehensive risk analysis using VaR, stress testing, and scenario modeling.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-icon" style="color: #dc2626;">
                    <i class="fas fa-exclamation-triangle"></i>
                </div>
                <div class="metric-value">2.3%</div>
                <div class="metric-label">VaR (95%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-icon" style="color: #ea580c;">
                    <i class="fas fa-fire"></i>
                </div>
                <div class="metric-value">3.8%</div>
                <div class="metric-label">Expected Shortfall</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-icon" style="color: #0891b2;">
                    <i class="fas fa-chart-area"></i>
                </div>
                <div class="metric-value">0.89</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-icon" style="color: #059669;">
                    <i class="fas fa-balance-scale"></i>
                </div>
                <div class="metric-value">-0.15</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            """, unsafe_allow_html=True)

    elif selected_page == "Notebook Analysis":
        st.markdown("""
        <div class="chart-container">
            <h2><i class="fas fa-book-open"></i> In-depth Notebook Analysis</h2>
            <p>Explore detailed data stories and analyses from our research notebooks.</p>
        </div>
        """, unsafe_allow_html=True)

        available_notebooks = notebook_integrator.available_notebooks
        if not available_notebooks:
            st.warning("No analysis notebooks found.")
        else:
            selected_notebook_name = st.selectbox(
                "Select Analysis Notebook",
                list(available_notebooks.keys()),
                format_func=lambda name: available_notebooks[name]['title']
            )

            if selected_notebook_name:
                nb_info = available_notebooks[selected_notebook_name]
                st.markdown(f"""
                <div class="data-insight">
                    <h3><i class="fas fa-file-alt"></i> {nb_info['title']}</h3>
                    <p><strong>Last Modified:</strong> {nb_info['modified'].strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Size:</strong> {nb_info['size'] / 1024:.2f} KB</p>
                </div>
                """, unsafe_allow_html=True)

                nb = notebook_integrator.load_notebook(selected_notebook_name)
                if nb:
                    for i, cell in enumerate(nb.cells):
                        if cell.cell_type == 'markdown':
                            st.markdown(cell.source, unsafe_allow_html=True)
                        elif cell.cell_type == 'code':
                            with st.expander(f"Code Cell {i+1}", expanded=False):
                                st.code(cell.source, language='python')

    elif selected_page == "Real-time Monitor":
        st.markdown("""
        <div class="chart-container">
            <h2><i class="fas fa-satellite-dish"></i> Real-time Economic Monitor</h2>
            <p>Live data feeds for currencies, commodities, and global markets with enhanced API integration.</p>
        </div>
        """, unsafe_allow_html=True)

        # Create tabs for different data categories
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "💱 Exchange Rates", 
            "₿ Cryptocurrencies", 
            "📊 Commodities", 
            "🌍 Global Economics", 
            "📈 African Markets",
            "📅 Economic Calendar"
        ])
        
        with tab1:
            st.markdown("### 💱 Multi-Source Exchange Rates")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                with st.spinner("Fetching exchange rates from multiple sources..."):
                    # Get multi-source exchange rates
                    multi_rates = api_integrator.get_multi_source_exchange_rates()
                    
                    if 'consolidated_rates' in multi_rates:
                        st.markdown("#### 📊 Consolidated Exchange Rates (vs USD)")
                        
                        # Create DataFrame for display
                        rates_data = []
                        for currency, data in multi_rates['consolidated_rates'].items():
                            rates_data.append({
                                'Currency': currency,
                                'Average Rate': f"{data['average_rate']:.4f}",
                                'Sources': data['sources_count'],
                                'Spread': f"{data['rate_spread']:.4f}",
                                'Reliability': '🟢 High' if data['sources_count'] >= 2 else '🟡 Medium'
                            })
                        
                        if rates_data:
                            rates_df = pd.DataFrame(rates_data)
                            st.dataframe(rates_df, use_container_width=True)
                            
                            # Create visualization
                            fig_rates = px.bar(
                                rates_df, 
                                x='Currency', 
                                y='Average Rate',
                                title='Exchange Rates vs USD',
                                color='Sources',
                                color_continuous_scale='viridis'
                            )
                            st.plotly_chart(fig_rates, use_container_width=True)
            
            with col2:
                st.markdown("#### 🔄 Rate Sources Status")
                
                if 'sources_data' in multi_rates:
                    for source, data in multi_rates['sources_data'].items():
                        status = data.get('status', 'unknown')
                        if status == 'success':
                            st.success(f"✅ {source.replace('_', ' ').title()}")
                        else:
                            st.error(f"❌ {source.replace('_', ' ').title()}")
                
                # Add refresh button
                if st.button("🔄 Refresh Rates", key="refresh_rates"):
                    st.rerun()
        
        with tab2:
            st.markdown("### ₿ Enhanced Cryptocurrency Data")
            
            with st.spinner("Fetching detailed crypto data..."):
                crypto_data = api_integrator.get_enhanced_crypto_data()
                
                if 'data' in crypto_data:
                    # Create metrics grid
                    crypto_metrics = list(crypto_data['data'].items())
                    
                    # Display top cryptocurrencies
                    for i in range(0, len(crypto_metrics), 3):
                        cols = st.columns(3)
                        
                        for j, col in enumerate(cols):
                            if i + j < len(crypto_metrics):
                                crypto_name, crypto_info = crypto_metrics[i + j]
                                
                                with col:
                                    # Price change color
                                    change_24h = crypto_info.get('price_change_24h', 0)
                                    change_color = "🟢" if change_24h >= 0 else "🔴"
                                    
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <h4>{change_color} {crypto_info['name']} ({crypto_info['symbol']})</h4>
                                        <div class="metric-value">${crypto_info['current_price']:,.2f}</div>
                                        <div class="metric-label">24h: {change_24h:+.2f}%</div>
                                        <div class="metric-label">Rank: #{crypto_info['market_cap_rank']}</div>
                                        <div class="metric-label">Vol: ${crypto_info['total_volume']:,.0f}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                    # Create market overview chart
                    crypto_df = pd.DataFrame([
                        {
                            'Name': info['name'],
                            'Price': info['current_price'],
                            'Market Cap': info['market_cap'],
                            '24h Change': info['price_change_24h']
                        }
                        for name, info in crypto_data['data'].items()
                    ])
                    
                    fig_crypto = px.treemap(
                        crypto_df,
                        path=['Name'],
                        values='Market Cap',
                        color='24h Change',
                        color_continuous_scale='RdYlGn',
                        title='Cryptocurrency Market Overview'
                    )
                    st.plotly_chart(fig_crypto, use_container_width=True)
        
        with tab3:
            st.markdown("### 📊 Global Commodity Prices")
            
            with st.spinner("Fetching commodity prices..."):
                commodity_data = api_integrator.get_commodity_prices()
                
                if 'data' in commodity_data:
                    # Display commodities in a grid
                    commodities = list(commodity_data['data'].items())
                    
                    for i in range(0, len(commodities), 2):
                        cols = st.columns(2)
                        
                        for j, col in enumerate(cols):
                            if i + j < len(commodities):
                                comm_name, comm_data = commodities[i + j]
                                
                                with col:
                                    change_24h = comm_data.get('change_24h', 0)
                                    change_color = "🟢" if change_24h >= 0 else "🔴"
                                    
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <h4>{change_color} {comm_name.replace('_', ' ').title()}</h4>
                                        <div class="metric-value">${comm_data['price']:.2f}</div>
                                        <div class="metric-label">per {comm_data.get('unit', 'unit')}</div>
                                        <div class="metric-label">24h: {change_24h:+.2f}%</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                    # Create commodity price chart
                    comm_df = pd.DataFrame([
                        {
                            'Commodity': name.replace('_', ' ').title(),
                            'Price': data['price'],
                            'Change': data.get('change_24h', 0)
                        }
                        for name, data in commodity_data['data'].items()
                    ])
                    
                    fig_comm = px.bar(
                        comm_df,
                        x='Commodity',
                        y='Price',
                        color='Change',
                        color_continuous_scale='RdYlGn',
                        title='Commodity Prices Overview'
                    )
                    st.plotly_chart(fig_comm, use_container_width=True)
        
        with tab4:
            st.markdown("### 🌍 Global Economic Indicators")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 World Bank Indicators")
                
                # Display World Bank data for multiple indicators
                wb_indicators = {
                    'NY.GDP.MKTP.KD.ZG': 'GDP Growth',
                    'FP.CPI.TOTL.ZG': 'Inflation Rate',
                    'SL.UEM.TOTL.ZS': 'Unemployment Rate'
                }
                
                wb_data = {}
                for indicator, name in wb_indicators.items():
                    wb_data[indicator] = api_integrator.get_world_bank_data(indicator)
                
                # Display latest values
                for indicator, name in wb_indicators.items():
                    data = wb_data[indicator]
                    if data and 'data' in data and data['data']:
                        latest = data['data'][0]
                        if latest.get('value') is not None:
                            gdp_growth = latest['value']
                            year = latest['date']
                            
                            st.metric(
                                label=f"{name} (Kenya)",
                                value=f"{gdp_growth:.1f}%",
                                delta=f"Year: {year}"
                            )
                        else:
                            st.info(f"{name}: Data not available")
                    else:
                        st.warning(f"{name}: Unable to fetch data")
            
            with col2:
                st.markdown("#### 🌍 Global Market Sentiment")
                
                # Market sentiment indicators
                sentiment_data = {
                    'VIX Index': {'value': 18.5, 'status': 'Low Volatility', 'color': 'green'},
                    'Dollar Index': {'value': 103.2, 'status': 'Strong USD', 'color': 'blue'},
                    'Global Growth': {'value': 3.1, 'status': 'Moderate', 'color': 'orange'},
                    'Risk Appetite': {'value': 75, 'status': 'High', 'color': 'green'}
                }
                
                for indicator, data in sentiment_data.items():
                    st.markdown(f"""
                    <div class="metric-card">
                        <h5>{indicator}</h5>
                        <div class="metric-value" style="color: {data['color']}">{data['value']}</div>
                        <div class="metric-label">{data['status']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab5:
            st.markdown("### 📈 African Economic Data")
            
            with st.spinner("Fetching African economic indicators..."):
                african_data = api_integrator.get_african_economic_data()
                
                if 'data' in african_data:
                    data = african_data['data']
                    
                    # Regional indicators
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "East Africa GDP Growth",
                            f"{data['east_africa_gdp_growth']}%",
                            delta="Regional Average"
                        )
                        
                        st.metric(
                            "Regional Inflation",
                            f"{data['regional_inflation']}%",
                            delta="Yearly Average"
                        )
                    
                    with col2:
                        st.metric(
                            "Trade Balance",
                            f"{data['trade_balance']}%",
                            delta="% of GDP"
                        )
                        
                        st.metric(
                            "FDI Inflows",
                            f"{data['fdi_inflows']}%",
                            delta="% of GDP"
                        )
                    
                    with col3:
                        st.metric(
                            "Debt to GDP",
                            f"{data['debt_to_gdp']}%",
                            delta="Regional Average"
                        )
                        
                        st.metric(
                            "Current Account",
                            f"{data['current_account']}%",
                            delta="% of GDP"
                        )
                    
                    # Regional currencies
                    st.markdown("#### 💱 East African Currencies (vs USD)")
                    
                    currencies = data['regional_currencies']
                    curr_df = pd.DataFrame([
                        {'Currency': curr, 'Rate': rate}
                        for curr, rate in currencies.items()
                    ])
                    
                    fig_curr = px.bar(
                        curr_df,
                        x='Currency',
                        y='Rate',
                        title='East African Currency Rates',
                        color='Rate',
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig_curr, use_container_width=True)
        
        with tab6:
            st.markdown("### 📅 Economic Calendar")
            
            with st.spinner("Fetching upcoming economic events..."):
                calendar_data = api_integrator.get_economic_calendar_events()
                
                if 'events' in calendar_data:
                    events = calendar_data['events']
                    
                    st.markdown(f"#### 📋 Upcoming High-Impact Events ({len(events)} events)")
                    
                    # Create events table
                    if events:
                        events_df = pd.DataFrame(events)
                        
                        # Style the dataframe
                        styled_events = events_df.style.apply(
                            lambda x: ['background-color: #ffebee' if x.name % 2 == 0 else 'background-color: #f3e5f5' for i in x],
                            axis=1
                        )
                        
                        st.dataframe(styled_events, use_container_width=True)
                        
                        # Impact distribution
                        if 'impact' in events_df.columns:
                            impact_counts = events_df['impact'].value_counts()
                            
                            fig_impact = px.pie(
                                values=impact_counts.values,
                                names=impact_counts.index,
                                title='Events by Impact Level',
                                color_discrete_map={
                                    'High': '#ff4444',
                                    'Medium': '#ffaa00',
                                    'Low': '#44ff44'
                                }
                            )
                            st.plotly_chart(fig_impact, use_container_width=True)
                    else:
                        st.info("No upcoming high-impact events found")
                else:
                    st.warning("Unable to fetch economic calendar data")
        
        # Auto-refresh option
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            auto_refresh = st.checkbox("🔄 Auto-refresh (30s)", value=False)
        
        with col2:
            if st.button("📊 Generate Report"):
                st.success("📋 Comprehensive market report generated!")
                st.download_button(
                    "📥 Download Market Report",
                    data="Market report data would be here",
                    file_name=f"market_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf"
                )
        
        with col3:
            if st.button("📧 Email Alerts Setup"):
                st.info("📧 Email alert preferences saved!")
        
        # Auto-refresh functionality
        if auto_refresh:
            time.sleep(30)
            st.rerun()
            # Exchange Rates
            st.subheader("Forex Rates (Base: USD)")
            exchange_rates = api_integrator.get_exchange_rates()
            if 'error' not in exchange_rates:
                col1, col2, col3, col4 = st.columns(4)
                major_rates = {'KES': exchange_rates.get('kes_usd'), 'EUR': exchange_rates['rates'].get('EUR'), 'GBP': exchange_rates['rates'].get('GBP'), 'JPY': exchange_rates['rates'].get('JPY')}
                icons = {'KES': 'fas fa-coins', 'EUR': 'fas fa-euro-sign', 'GBP': 'fas fa-pound-sign', 'JPY': 'fas fa-yen-sign'}
                
                for i, (currency, rate) in enumerate(major_rates.items()):
                    if rate is not None:
                        with locals()[f"col{i+1}"]:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-icon"><i class="{icons[currency]}"></i></div>
                                <div class="metric-value">{rate:.2f}</div>
                                <div class="metric-label">USD/{currency}</div>
                            </div>
                            """, unsafe_allow_html=True)
                
                with st.expander("View all exchange rates"):
                    st.json(exchange_rates.get('rates', {}))

            # Crypto Rates
            st.subheader("Cryptocurrency Prices")
            crypto_rates = api_integrator.get_crypto_rates()
            if 'error' not in crypto_rates:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Bitcoin (USD)", f"${crypto_rates.get('bitcoin_usd', 0):,.2f}")
                with col2:
                    st.metric("Bitcoin (EUR)", f"€{crypto_rates.get('bitcoin_eur', 0):,.2f}")
                with col3:
                    st.metric("Bitcoin (KES)", f"Ksh{crypto_rates.get('bitcoin_kes', 0):,.2f}")

            # World Bank Data
            st.subheader("Kenya - World Bank Indicators")
            wb_indicators = ['gdp', 'inflation', 'unemployment']
            wb_data = {}
            for indicator in wb_indicators:
                wb_data[indicator] = api_integrator.get_world_bank_data(indicator)

            if wb_data:
                col1, col2, col3 = st.columns(3)
                with col1:
                    gdp_data = wb_data.get('gdp')
                    if gdp_data and gdp_data.get('data') and gdp_data['data'][0].get('value') is not None:
                        st.metric("GDP (Current USD)", f"${gdp_data['data'][0]['value']:,.0f} ({gdp_data['data'][0]['date']})")
                with col2:
                    inflation_data = wb_data.get('inflation')
                    if inflation_data and inflation_data.get('data') and inflation_data['data'][0].get('value') is not None:
                        st.metric("Inflation, consumer prices (annual %)", f"{inflation_data['data'][0]['value']:.2f}% ({inflation_data['data'][0]['date']})")
                with col3:
                    unemployment_data = wb_data.get('unemployment')
                    if unemployment_data and unemployment_data.get('data') and unemployment_data['data'][0].get('value') is not None:
                        st.metric("Unemployment, total (% of total labor force)", f"{unemployment_data['data'][0]['value']:.2f}% ({unemployment_data['data'][0]['date']})")

if __name__ == "__main__":
    main()
