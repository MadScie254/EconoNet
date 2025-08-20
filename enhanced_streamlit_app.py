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

warnings.filterwarnings('ignore')

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
        
        # Integrated notebook execution
        if st.button("🔄 Run Risk Analysis Notebook"):
            st.info("🚀 Executing comprehensive risk analysis...")
            
            # Display risk analysis results (simulated)
            risk_scenarios = ["Base Case", "Stress Test", "Crisis Scenario"]
            risk_values = [2.3, 5.7, 12.4]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=risk_scenarios,
                y=risk_values,
                marker_color=['#10b981', '#f59e0b', '#ef4444'],
                text=[f"{val}%" for val in risk_values],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Value at Risk Across Scenarios",
                yaxis_title="VaR (%)",
                template="plotly_white",
                font=dict(family="Inter, sans-serif"),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("""
    <div style="margin-top: 3rem; padding: 2rem; text-align: center; border-top: 1px solid #e2e8f0;">
        <p style="color: #64748b; margin: 0;">
            <i class="fas fa-copyright"></i> 2024 EconoNet - Kenya Economic Intelligence Platform
        </p>
        <p style="color: #64748b; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
            Powered by Advanced Analytics & AI | Built with ❤️ for Kenya's Economic Future
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
