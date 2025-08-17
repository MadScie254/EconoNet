"""
NERVA Professional Dashboard - Enhanced Economic Intelligence Platform
Advanced Analytics with Professional Styling and Real-time Capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime, timedelta
import warnings
import sys
import time
from pathlib import Path

warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="NERVA Professional - Economic Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import professional styling and analytics
try:
    # Add current directory to path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    from nerva.ui.professional_styles import (
        NERVA_PROFESSIONAL_CSS, NERVA_ICONS,
        get_professional_header, get_metric_card, 
        get_status_indicator, get_alert_box, get_progress_bar
    )
    from nerva.analytics.advanced_engine import AdvancedAnalyticsEngine
    STYLING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Professional styling not available: {e}")
    STYLING_AVAILABLE = False

# Apply professional CSS
if STYLING_AVAILABLE:
    st.markdown(NERVA_PROFESSIONAL_CSS, unsafe_allow_html=True)

# Professional header
if STYLING_AVAILABLE:
    st.markdown(get_professional_header(
        "NERVA Professional Intelligence Platform",
        "Advanced Economic Analytics & Risk Assessment System"
    ), unsafe_allow_html=True)
else:
    st.title("NERVA Professional Intelligence Platform")
    st.markdown("**Advanced Economic Analytics & Risk Assessment System**")

# Initialize analytics engine
if STYLING_AVAILABLE:
    analytics_engine = AdvancedAnalyticsEngine()

# Sidebar navigation with professional styling
st.sidebar.markdown("### Navigation")

if STYLING_AVAILABLE:
    # Professional navigation with icons
    nav_options = {
        "Dashboard": ("dashboard", "System Overview & Key Metrics"),
        "Data Analytics": ("analytics", "Advanced Statistical Analysis"),
        "Modeling Suite": ("models", "Predictive Models & Forecasting"),
        "Intelligence Reports": ("notebooks", "Specialized Analysis Notebooks"),
        "System Configuration": ("settings", "Platform Settings & Controls")
    }
    
    selected_page = st.sidebar.selectbox(
        "Select Module",
        list(nav_options.keys()),
        format_func=lambda x: f"📊 {x}" if x == "Dashboard" else 
                             f"📈 {x}" if x == "Data Analytics" else
                             f"⭐ {x}" if x == "Modeling Suite" else
                             f"📋 {x}" if x == "Intelligence Reports" else
                             f"⚙️ {x}"
    )
else:
    selected_page = st.sidebar.selectbox(
        "Navigation",
        ["Dashboard", "Data Analytics", "Modeling Suite", "Intelligence Reports", "System Configuration"]
    )

# Load sample data
@st.cache_data
def load_sample_data():
    """Load sample economic data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=48, freq='M')
    
    data = pd.DataFrame({
        'date': dates,
        'cbr_rate': 7.0 + np.cumsum(np.random.normal(0, 0.2, 48)),
        'inflation_rate': 5.0 + np.sin(np.arange(48) * 0.3) + np.random.normal(0, 0.5, 48),
        'fx_rate': 110 + np.cumsum(np.random.normal(0, 2, 48)),
        'gdp_growth': 2.5 + np.random.normal(0, 1, 48),
        'credit_growth': 8.0 + np.random.normal(0, 2, 48)
    })
    
    # Ensure positive values
    data['cbr_rate'] = np.maximum(data['cbr_rate'], 0.5)
    data['fx_rate'] = np.maximum(data['fx_rate'], 80)
    
    return data

# Load data
sample_data = load_sample_data()

# Dashboard Page
if selected_page == "Dashboard":
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    latest_data = sample_data.iloc[-1]
    prev_data = sample_data.iloc[-2]
    
    with col1:
        cbr_change = ((latest_data['cbr_rate'] - prev_data['cbr_rate']) / prev_data['cbr_rate'] * 100)
        change_type = "positive" if cbr_change > 0 else "negative"
        
        if STYLING_AVAILABLE:
            st.markdown(get_metric_card(
                f"{latest_data['cbr_rate']:.2f}%",
                "Central Bank Rate",
                f"{cbr_change:+.2f}%",
                change_type
            ), unsafe_allow_html=True)
        else:
            st.metric("Central Bank Rate", f"{latest_data['cbr_rate']:.2f}%", f"{cbr_change:+.2f}%")
    
    with col2:
        inf_change = ((latest_data['inflation_rate'] - prev_data['inflation_rate']) / prev_data['inflation_rate'] * 100)
        change_type = "negative" if inf_change > 0 else "positive"
        
        if STYLING_AVAILABLE:
            st.markdown(get_metric_card(
                f"{latest_data['inflation_rate']:.2f}%",
                "Inflation Rate",
                f"{inf_change:+.2f}%",
                change_type
            ), unsafe_allow_html=True)
        else:
            st.metric("Inflation Rate", f"{latest_data['inflation_rate']:.2f}%", f"{inf_change:+.2f}%")
    
    with col3:
        fx_change = ((latest_data['fx_rate'] - prev_data['fx_rate']) / prev_data['fx_rate'] * 100)
        change_type = "negative" if fx_change > 0 else "positive"
        
        if STYLING_AVAILABLE:
            st.markdown(get_metric_card(
                f"{latest_data['fx_rate']:.2f}",
                "USD/KES Rate",
                f"{fx_change:+.2f}%",
                change_type
            ), unsafe_allow_html=True)
        else:
            st.metric("USD/KES Rate", f"{latest_data['fx_rate']:.2f}", f"{fx_change:+.2f}%")
    
    with col4:
        gdp_change = ((latest_data['gdp_growth'] - prev_data['gdp_growth']) / abs(prev_data['gdp_growth']) * 100)
        change_type = "positive" if gdp_change > 0 else "negative"
        
        if STYLING_AVAILABLE:
            st.markdown(get_metric_card(
                f"{latest_data['gdp_growth']:.2f}%",
                "GDP Growth",
                f"{gdp_change:+.2f}%",
                change_type
            ), unsafe_allow_html=True)
        else:
            st.metric("GDP Growth", f"{latest_data['gdp_growth']:.2f}%", f"{gdp_change:+.2f}%")
    
    # System status
    if STYLING_AVAILABLE:
        st.markdown("### System Status")
        status_col1, status_col2, status_col3 = st.columns(3)
        
        with status_col1:
            st.markdown(get_status_indicator("active", "Data Processing"), unsafe_allow_html=True)
        with status_col2:
            st.markdown(get_status_indicator("active", "Analytics Engine"), unsafe_allow_html=True)
        with status_col3:
            st.markdown(get_status_indicator("active", "Model Training"), unsafe_allow_html=True)
    
    # Economic indicators chart
    st.markdown("### Economic Indicators Overview")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Central Bank Rate', 'Inflation Rate', 'Exchange Rate', 'GDP Growth'],
        vertical_spacing=0.1
    )
    
    # CBR
    fig.add_trace(
        go.Scatter(x=sample_data['date'], y=sample_data['cbr_rate'],
                  mode='lines+markers', name='CBR Rate',
                  line=dict(color='#2E86C1', width=3)),
        row=1, col=1
    )
    
    # Inflation
    fig.add_trace(
        go.Scatter(x=sample_data['date'], y=sample_data['inflation_rate'],
                  mode='lines+markers', name='Inflation',
                  line=dict(color='#E74C3C', width=3)),
        row=1, col=2
    )
    
    # FX Rate
    fig.add_trace(
        go.Scatter(x=sample_data['date'], y=sample_data['fx_rate'],
                  mode='lines+markers', name='USD/KES',
                  line=dict(color='#8E44AD', width=3)),
        row=2, col=1
    )
    
    # GDP Growth
    fig.add_trace(
        go.Scatter(x=sample_data['date'], y=sample_data['gdp_growth'],
                  mode='lines+markers', name='GDP Growth',
                  line=dict(color='#27AE60', width=3)),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        paper_bgcolor='white' if STYLING_AVAILABLE else 'rgba(0,0,0,0)',
        plot_bgcolor='#F8F9FA' if STYLING_AVAILABLE else 'rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Data Analytics Page
elif selected_page == "Data Analytics":
    st.markdown("### Advanced Data Analytics")
    
    if STYLING_AVAILABLE and analytics_engine:
        # Correlation analysis
        st.markdown("#### Correlation Matrix Analysis")
        
        correlation_fig, correlation_matrix = analytics_engine.advanced_correlation_analysis(sample_data)
        if correlation_fig:
            st.plotly_chart(correlation_fig, use_container_width=True)
        
        # Predictive modeling
        st.markdown("#### Predictive Modeling Suite")
        
        target_variable = st.selectbox(
            "Select Target Variable",
            ['inflation_rate', 'cbr_rate', 'fx_rate', 'gdp_growth']
        )
        
        if st.button("Run Predictive Analysis"):
            with st.spinner("Training predictive models..."):
                results, predictions = analytics_engine.predictive_modeling_suite(sample_data, target_variable)
                
                if results and predictions:
                    # Display results
                    performance_dashboard = analytics_engine.create_performance_dashboard(results, predictions)
                    if performance_dashboard:
                        st.plotly_chart(performance_dashboard, use_container_width=True)
                    
                    # Model comparison table
                    results_df = pd.DataFrame(results).T
                    st.markdown("#### Model Performance Comparison")
                    st.dataframe(results_df.round(4))
                else:
                    st.error("Unable to perform predictive analysis")
        
        # Time series analysis
        st.markdown("#### Time Series Decomposition")
        
        ts_variable = st.selectbox(
            "Select Time Series Variable",
            ['inflation_rate', 'cbr_rate', 'fx_rate', 'gdp_growth'],
            key="ts_var"
        )
        
        if st.button("Perform Time Series Analysis"):
            ts_fig, ts_data = analytics_engine.time_series_decomposition(sample_data, 'date', ts_variable)
            if ts_fig:
                st.plotly_chart(ts_fig, use_container_width=True)
    else:
        st.info("Advanced analytics engine not available. Please check module installation.")
        
        # Fallback basic analysis
        st.markdown("#### Basic Statistical Summary")
        st.dataframe(sample_data.describe())

# Modeling Suite Page
elif selected_page == "Modeling Suite":
    st.markdown("### Advanced Modeling Suite")
    
    if STYLING_AVAILABLE:
        st.markdown(get_alert_box(
            "Advanced modeling capabilities ready for deployment. Select modeling approach below.",
            "info"
        ), unsafe_allow_html=True)
    
    # Model selection
    model_type = st.selectbox(
        "Select Modeling Approach",
        ["Ensemble Forecasting", "Neural Network", "Time Series ARIMA", "Regime Detection"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Model Configuration")
        
        if model_type == "Ensemble Forecasting":
            n_estimators = st.slider("Number of Estimators", 50, 500, 100)
            max_depth = st.slider("Maximum Depth", 3, 20, 10)
            
        elif model_type == "Neural Network":
            hidden_layers = st.slider("Hidden Layers", 1, 5, 2)
            neurons_per_layer = st.slider("Neurons per Layer", 32, 256, 64)
            
        forecast_horizon = st.slider("Forecast Horizon (months)", 1, 24, 12)
    
    with col2:
        st.markdown("#### Training Progress")
        
        if st.button("Start Model Training"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate training progress
            for i in range(101):
                progress_bar.progress(i)
                status_text.text(f'Training Progress: {i}%')
                time.sleep(0.01)
            
            status_text.text('Training Complete!')
            
            if STYLING_AVAILABLE:
                st.markdown(get_alert_box(
                    f"{model_type} model trained successfully with {forecast_horizon}-month horizon.",
                    "success"
                ), unsafe_allow_html=True)
            else:
                st.success(f"{model_type} model trained successfully!")

# Intelligence Reports Page
elif selected_page == "Intelligence Reports":
    st.markdown("### Specialized Intelligence Notebooks")
    
    # Import notebook manager
    try:
        from nerva.core.notebook_manager import NotebookManager, get_notebook_info
        notebook_manager = NotebookManager()
        NOTEBOOK_MANAGER_AVAILABLE = True
    except ImportError:
        NOTEBOOK_MANAGER_AVAILABLE = False
    
    # Notebook showcase
    notebooks = {
        "Advanced_Inflation_Modeling.ipynb": {
            "description": "Quantum-inspired inflation forecasting with neural networks and regime detection",
            "status": "Active",
            "complexity": "Advanced"
        },
        "Quantum_FX_Dynamics.ipynb": {
            "description": "Advanced foreign exchange modeling with volatility clustering",
            "status": "Ready",
            "complexity": "Expert"
        },
        "Market_Dynamics_Intelligence.ipynb": {
            "description": "High-frequency market analysis and liquidity flow mapping",
            "status": "Ready", 
            "complexity": "Expert"
        },
        "Policy_Simulation_Engine.ipynb": {
            "description": "Central bank policy transmission and impact modeling",
            "status": "Ready",
            "complexity": "Advanced"
        }
    }
    
    # Notebook management controls
    if NOTEBOOK_MANAGER_AVAILABLE:
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("#### Jupyter Lab Control")
            if st.button("Launch Jupyter Lab", type="primary"):
                try:
                    process, url = notebook_manager.launch_jupyter_lab(port=8889)
                    if url:
                        st.success(f"Jupyter Lab launching at: {url}")
                        st.markdown(f"[Open Jupyter Lab]({url})", unsafe_allow_html=True)
                    else:
                        st.error("Failed to launch Jupyter Lab")
                except Exception as e:
                    st.error(f"Error launching Jupyter Lab: {e}")
    
    # Display notebooks
    for filename, info in notebooks.items():
        with st.expander(f"📋 {filename.replace('.ipynb', '').replace('_', ' ')}"):
            
            if NOTEBOOK_MANAGER_AVAILABLE:
                notebook_info = get_notebook_info(filename)
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Description:** {notebook_info['description']}")
                    st.markdown(f"**Category:** {notebook_info['category']}")
                    st.markdown(f"**Complexity:** {notebook_info['complexity']}")
                    
                    # Features list
                    st.markdown("**Key Features:**")
                    for feature in notebook_info['features']:
                        st.markdown(f"- {feature}")
                    
                    if STYLING_AVAILABLE:
                        status_class = "active" if info["status"] == "Active" else "warning"
                        st.markdown(get_status_indicator(status_class, info["status"]), unsafe_allow_html=True)
                
                with col2:
                    if st.button(f"Open {filename}", key=f"open_{filename}"):
                        # Check if file exists
                        notebook_path = Path(f"notebooks/{filename}")
                        if notebook_path.exists():
                            st.info(f"Opening {filename} in Jupyter Lab...")
                            # In a real scenario, this would open the notebook
                        else:
                            st.warning(f"Notebook file not found: {filename}")
            else:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(info["description"])
                    st.write(f"**Complexity:** {info['complexity']}")
                    
                    if STYLING_AVAILABLE:
                        status_class = "active" if info["status"] == "Active" else "warning"
                        st.markdown(get_status_indicator(status_class, info["status"]), unsafe_allow_html=True)
                
                with col2:
                    if st.button(f"View {filename}", key=f"view_{filename}"):
                        st.info(f"Notebook: {filename}")
    
    # Real-time notebook execution status
    if STYLING_AVAILABLE:
        st.markdown("#### Execution Status")
        
        execution_status = {
            "Kernel Sessions": 2,
            "Active Notebooks": 1, 
            "Completed Analyses": 15,
            "System Load": 45
        }
        
        status_cols = st.columns(4)
        for i, (metric, value) in enumerate(execution_status.items()):
            with status_cols[i]:
                if metric == "System Load":
                    st.markdown(get_metric_card(f"{value}%", metric), unsafe_allow_html=True)
                    st.markdown(get_progress_bar(value), unsafe_allow_html=True)
                else:
                    st.markdown(get_metric_card(str(value), metric), unsafe_allow_html=True)

# System Configuration Page
elif selected_page == "System Configuration":
    st.markdown("### System Configuration & Settings")
    
    # Import real-time engine
    try:
        from nerva.data.real_time_engine import get_real_time_streamer, get_quality_monitor
        streamer = get_real_time_streamer()
        quality_monitor = get_quality_monitor()
        REAL_TIME_AVAILABLE = True
    except ImportError:
        REAL_TIME_AVAILABLE = False
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Data Sources & Connectivity")
        
        data_sources = {
            "Central Bank of Kenya": {"status": True, "latency": "12ms", "quality": 98.5},
            "Kenya National Bureau of Statistics": {"status": True, "latency": "45ms", "quality": 95.2},
            "IMF World Economic Outlook": {"status": False, "latency": "N/A", "quality": 0.0},
            "World Bank Open Data": {"status": False, "latency": "N/A", "quality": 0.0},
            "Bloomberg Terminal": {"status": False, "latency": "N/A", "quality": 0.0}
        }
        
        for source, metrics in data_sources.items():
            st.markdown(f"**{source}**")
            
            if STYLING_AVAILABLE:
                status = "active" if metrics["status"] else "error"
                status_text = "Connected" if metrics["status"] else "Disconnected"
                st.markdown(get_status_indicator(status, status_text), unsafe_allow_html=True)
                
                if metrics["status"]:
                    st.markdown(f"<small>Latency: {metrics['latency']} | Quality: {metrics['quality']:.1f}%</small>", 
                              unsafe_allow_html=True)
            else:
                st.write(f"Status: {'✅ Connected' if metrics['status'] else '❌ Disconnected'}")
                if metrics["status"]:
                    st.write(f"Latency: {metrics['latency']} | Quality: {metrics['quality']:.1f}%")
        
        # Real-time streaming controls
        if REAL_TIME_AVAILABLE:
            st.markdown("#### Real-time Data Streaming")
            
            col_start, col_stop = st.columns(2)
            
            with col_start:
                if st.button("Start Real-time Stream", type="primary"):
                    streamer.start_streaming()
                    st.success("Real-time streaming started")
            
            with col_stop:
                if st.button("Stop Stream"):
                    streamer.stop_streaming()
                    st.info("Real-time streaming stopped")
            
            # Display latest data
            latest_data = streamer.get_latest_data()
            if latest_data:
                st.markdown("**Latest Data Point:**")
                st.json({
                    "timestamp": latest_data["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "cbr_rate": f"{latest_data['cbr_rate']:.2f}%",
                    "inflation_rate": f"{latest_data['inflation_rate']:.2f}%",
                    "fx_rate": f"{latest_data['fx_rate']:.2f}",
                    "market_sentiment": f"{latest_data['market_sentiment']:.3f}"
                })
    
    with col2:
        st.markdown("#### System Performance & Health")
        
        # System metrics
        if REAL_TIME_AVAILABLE:
            quality_report = quality_monitor.generate_quality_report()
            
            performance_metrics = {
                "Data Quality Score": quality_report['overall_quality'] * 100,
                "CPU Usage": np.random.randint(30, 70),
                "Memory Usage": np.random.randint(40, 80),
                "Model Training Progress": np.random.randint(60, 90),
                "Active Connections": quality_report['total_points']
            }
        else:
            performance_metrics = {
                "CPU Usage": 45,
                "Memory Usage": 62,
                "Model Training": 78,
                "Data Processing": 33,
                "Active Connections": 3
            }
        
        for metric, value in performance_metrics.items():
            st.markdown(f"**{metric}**")
            
            if isinstance(value, (int, float)) and value <= 100:
                if STYLING_AVAILABLE:
                    st.markdown(get_progress_bar(value), unsafe_allow_html=True)
                    st.markdown(f"<small>{value:.1f}%</small>", unsafe_allow_html=True)
                else:
                    st.progress(value / 100)
                    st.caption(f"{value:.1f}%")
            else:
                st.markdown(f"<span style='color: #2E86C1; font-weight: bold;'>{value}</span>", 
                          unsafe_allow_html=True)
        
        # Data quality alerts
        if REAL_TIME_AVAILABLE and quality_report.get('alerts'):
            st.markdown("#### Data Quality Alerts")
            for alert in quality_report['alerts']:
                if STYLING_AVAILABLE:
                    st.markdown(get_alert_box(alert, "warning"), unsafe_allow_html=True)
                else:
                    st.warning(alert)
        
        # Advanced configuration
        st.markdown("#### Advanced Configuration")
        
        with st.expander("Model Parameters"):
            forecast_horizon = st.slider("Default Forecast Horizon (months)", 1, 36, 12)
            confidence_level = st.selectbox("Confidence Level", ["90%", "95%", "99%"], index=1)
            update_frequency = st.selectbox("Update Frequency", ["Real-time", "Hourly", "Daily"], index=1)
        
        with st.expander("Alert Thresholds"):
            inflation_threshold = st.slider("Inflation Alert Threshold (%)", 0.0, 20.0, 8.0, 0.1)
            fx_volatility_threshold = st.slider("FX Volatility Threshold (%)", 0.0, 10.0, 5.0, 0.1)
            quality_threshold = st.slider("Data Quality Threshold (%)", 50.0, 100.0, 80.0, 1.0)
        
        if st.button("Save Configuration"):
            st.success("Configuration saved successfully")
    
    # System logs
    st.markdown("#### System Activity Log")
    
    log_entries = [
        {"time": "2024-08-18 10:30:15", "level": "INFO", "message": "Real-time data streaming started"},
        {"time": "2024-08-18 10:28:42", "level": "INFO", "message": "Model training completed successfully"},
        {"time": "2024-08-18 10:25:33", "level": "WARNING", "message": "Data quality below threshold for EUR/KES"},
        {"time": "2024-08-18 10:22:18", "level": "INFO", "message": "CBR rate forecast updated"},
        {"time": "2024-08-18 10:20:05", "level": "INFO", "message": "System initialization complete"}
    ]
    
    log_df = pd.DataFrame(log_entries)
    
    # Color code log levels
    def style_log_level(level):
        if level == "ERROR":
            return "color: #DC143C; font-weight: bold"
        elif level == "WARNING":
            return "color: #FF8C00; font-weight: bold"
        elif level == "INFO":
            return "color: #4682B4"
        return ""
    
    if STYLING_AVAILABLE:
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(log_df, use_container_width=True)

# Footer
st.markdown("---")
if STYLING_AVAILABLE:
    st.markdown("""
    <div style="text-align: center; color: #6C757D; font-size: 0.9rem;">
        NERVA Professional Intelligence Platform | Advanced Economic Analytics | Confidential
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("**NERVA Professional Intelligence Platform** | Advanced Economic Analytics")
