"""
NERVA Enhanced Dashboard
Professional UI with FontAwesome icons and advanced visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# NERVA imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config.settings import config
    from etl.processor import CBKDataProcessor, get_data_catalog
    from models.baseline import BaselineForecaster, train_baseline_forecaster
    from models.advanced import AdvancedForecaster, train_advanced_forecaster
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config.settings import config
    from etl.processor import CBKDataProcessor, get_data_catalog
    from models.baseline import BaselineForecaster, train_baseline_forecaster
    from models.advanced import AdvancedForecaster, train_advanced_forecaster

# Enhanced page configuration
st.set_page_config(
    page_title="NERVA | CBK Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with FontAwesome and professional styling
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main > div {
        padding-top: 2rem;
    }
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header Styles */
    .header-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        color: #6c757d;
        font-weight: 400;
    }
    
    /* Card Styles */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
    }
    
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: 500;
    }
    
    .metric-delta {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .delta-positive { color: #28a745; }
    .delta-negative { color: #dc3545; }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: white;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.2);
    }
    
    /* Chart Container */
    .chart-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Alert Styles */
    .alert-success {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-info {
        background: linear-gradient(135deg, #17a2b8 0%, #6f42c1 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Data Table Styles */
    .dataframe {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Selectbox and Input Styles */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 8px;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

class EnhancedNERVADashboard:
    """Enhanced NERVA Dashboard with professional UI"""
    
    def __init__(self):
        self.processor = CBKDataProcessor()
        self.datasets = {}
        self.forecasters = {}
        
        # Enhanced color scheme
        self.colors = {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'success': '#28a745',
            'danger': '#dc3545',
            'warning': '#ffc107',
            'info': '#17a2b8',
            'dark': '#343a40',
            'light': '#f8f9fa'
        }
        
        # Initialize session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize Streamlit session state"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'datasets' not in st.session_state:
            st.session_state.datasets = {}
        if 'selected_models' not in st.session_state:
            st.session_state.selected_models = ['ensemble']
        if 'advanced_mode' not in st.session_state:
            st.session_state.advanced_mode = False
    
    def render_header(self):
        """Render enhanced header with branding"""
        st.markdown("""
        <div class="header-container">
            <h1 class="header-title">
                <i class="fas fa-brain"></i> NERVA
            </h1>
            <p class="header-subtitle">
                <i class="fas fa-university"></i> National Economic & Risk Visual Analytics | Central Bank of Kenya
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def load_data(self):
        """Load and cache CBK data with progress tracking"""
        if not st.session_state.data_loaded:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Initializing data processor...")
                progress_bar.progress(10)
                
                status_text.text("Scanning CBK data archive...")
                progress_bar.progress(30)
                
                self.datasets = self.processor.scan_all_files()
                progress_bar.progress(70)
                
                status_text.text("Generating data catalog...")
                st.session_state.datasets = self.datasets
                st.session_state.data_loaded = True
                progress_bar.progress(100)
                
                status_text.empty()
                progress_bar.empty()
                
                st.markdown(f"""
                <div class="alert-success">
                    <i class="fas fa-check-circle"></i> Successfully loaded {len(self.datasets)} datasets
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                <div class="alert-warning">
                    <i class="fas fa-exclamation-triangle"></i> Failed to load data: {str(e)}
                </div>
                """, unsafe_allow_html=True)
                return False
        else:
            self.datasets = st.session_state.datasets
        return True
    
    def render_enhanced_sidebar(self):
        """Render enhanced sidebar with FontAwesome icons"""
        
        # Sidebar header
        st.sidebar.markdown("""
        ### <i class="fas fa-control-panel"></i> Control Center
        """, unsafe_allow_html=True)
        
        # Data refresh
        if st.sidebar.button("🔄 Refresh Data", help="Reload CBK data archive"):
            st.session_state.data_loaded = False
            st.experimental_rerun()
        
        st.sidebar.markdown("---")
        
        # Advanced mode toggle
        st.session_state.advanced_mode = st.sidebar.checkbox(
            "🚀 Advanced Analytics Mode",
            value=st.session_state.advanced_mode,
            help="Enable Transformer models and advanced features"
        )
        
        st.sidebar.markdown("---")
        
        # Dataset controls
        st.sidebar.markdown('### <i class="fas fa-database"></i> Data Selection')
        
        if self.datasets:
            dataset_names = list(self.datasets.keys())
            selected_dataset = st.sidebar.selectbox(
                "📊 Dataset",
                dataset_names,
                index=0,
                help="Choose dataset for analysis"
            )
            
            # Target variable selector
            if selected_dataset in self.datasets:
                df = self.datasets[selected_dataset]
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if numeric_cols:
                    target_variable = st.sidebar.selectbox(
                        "🎯 Target Variable",
                        numeric_cols,
                        help="Variable to forecast"
                    )
                else:
                    st.sidebar.warning("⚠️ No numeric columns found")
                    target_variable = None
            else:
                target_variable = None
        else:
            selected_dataset = None
            target_variable = None
        
        st.sidebar.markdown("---")
        
        # Forecasting controls
        st.sidebar.markdown('### <i class="fas fa-crystal-ball"></i> Forecast Settings')
        
        forecast_horizon = st.sidebar.slider(
            "📅 Horizon (months)",
            min_value=1,
            max_value=24,
            value=6,
            help="Number of months to forecast ahead"
        )
        
        if st.session_state.advanced_mode:
            model_options = ['lightgbm', 'random_forest', 'transformer', 'garch', 'var', 'ensemble']
        else:
            model_options = ['lightgbm', 'random_forest', 'linear', 'arima', 'ets', 'ensemble']
        
        model_selection = st.sidebar.multiselect(
            "🤖 Models",
            model_options,
            default=['ensemble'],
            help="Select forecasting models"
        )
        
        st.sidebar.markdown("---")
        
        # Scenario simulation
        st.sidebar.markdown('### <i class="fas fa-sliders-h"></i> Scenario Simulator')
        
        policy_rate_change = st.sidebar.slider(
            "💰 Policy Rate (bps)",
            min_value=-500,
            max_value=500,
            value=0,
            step=25,
            help="Simulate policy rate changes"
        )
        
        fx_shock = st.sidebar.slider(
            "💱 FX Shock (%)",
            min_value=-20.0,
            max_value=20.0,
            value=0.0,
            step=1.0,
            help="Simulate currency shock"
        )
        
        # Risk parameters
        st.sidebar.markdown('### <i class="fas fa-shield-alt"></i> Risk Parameters')
        
        confidence_level = st.sidebar.selectbox(
            "📊 Confidence Level",
            [0.90, 0.95, 0.99],
            index=1,
            format_func=lambda x: f"{x*100:.0f}%"
        )
        
        return {
            'selected_dataset': selected_dataset,
            'target_variable': target_variable,
            'forecast_horizon': forecast_horizon,
            'model_selection': model_selection,
            'policy_rate_change': policy_rate_change,
            'fx_shock': fx_shock,
            'confidence_level': confidence_level,
            'advanced_mode': st.session_state.advanced_mode
        }
    
    def render_enhanced_kpi_cards(self):
        """Render enhanced KPI cards with animations and icons"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-icon" style="color: #dc3545;">
                    <i class="fas fa-shield-alt"></i>
                </div>
                <div class="metric-value" style="color: #dc3545;">0.23</div>
                <div class="metric-label">Systemic Risk Index</div>
                <div class="metric-delta delta-negative">
                    <i class="fas fa-arrow-down"></i> -0.05 (Good)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-icon" style="color: #ffc107;">
                    <i class="fas fa-chart-line"></i>
                </div>
                <div class="metric-value" style="color: #ffc107;">5.2%</div>
                <div class="metric-label">1M Inflation Forecast</div>
                <div class="metric-delta delta-positive">
                    <i class="fas fa-arrow-up"></i> +0.3%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-icon" style="color: #17a2b8;">
                    <i class="fas fa-exchange-alt"></i>
                </div>
                <div class="metric-value" style="color: #17a2b8;">142.5</div>
                <div class="metric-label">3M FX Forecast (USD/KES)</div>
                <div class="metric-delta delta-positive">
                    <i class="fas fa-arrow-up"></i> +2.8
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-icon" style="color: #28a745;">
                    <i class="fas fa-piggy-bank"></i>
                </div>
                <div class="metric-value" style="color: #28a745;">4.2</div>
                <div class="metric-label">Reserve Adequacy (months)</div>
                <div class="metric-delta delta-negative">
                    <i class="fas fa-arrow-down"></i> -0.1
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def render_advanced_data_overview(self):
        """Render enhanced data overview with interactive charts"""
        
        st.markdown('## <i class="fas fa-table"></i> CBK Data Intelligence', unsafe_allow_html=True)
        
        try:
            catalog_df = self.processor.generate_data_catalog()
            
            # Create enhanced visualization
            fig = self.create_data_quality_dashboard(catalog_df)
            
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Interactive data table
            st.markdown("### <i class="fas fa-list"></i> Dataset Catalog", unsafe_allow_html=True)
            
            # Add quality filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_quality = st.slider("Minimum Quality Score", 0.0, 1.0, 0.0, 0.1)
            with col2:
                max_missing = st.slider("Max Missing %", 0.0, 100.0, 100.0, 5.0)
            with col3:
                min_rows = st.number_input("Minimum Rows", 0, int(catalog_df['rows'].max()), 0)
            
            # Filter data
            filtered_df = catalog_df[
                (catalog_df['quality_score'] >= min_quality) &
                (catalog_df['missing_percentage'] <= max_missing) &
                (catalog_df['rows'] >= min_rows)
            ]
            
            # Style the dataframe
            def style_quality_score(val):
                if val >= 0.8:
                    return 'background-color: #d4edda; color: #155724'
                elif val >= 0.6:
                    return 'background-color: #fff3cd; color: #856404'
                else:
                    return 'background-color: #f8d7da; color: #721c24'
            
            styled_df = filtered_df.style.applymap(
                style_quality_score, 
                subset=['quality_score']
            ).format({
                'quality_score': '{:.3f}',
                'missing_percentage': '{:.1f}%',
                'rows': '{:,}',
                'columns': '{:,}'
            })
            
            st.dataframe(styled_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Failed to generate data overview: {str(e)}")
    
    def create_data_quality_dashboard(self, catalog_df):
        """Create comprehensive data quality dashboard"""
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Quality Score Distribution',
                'Data Completeness vs Quality',
                'Dataset Size Analysis',
                'Missing Data Patterns',
                'Outlier Analysis',
                'Data Freshness'
            ],
            specs=[[{"type": "histogram"}, {"type": "scatter"}, {"type": "bubble"}],
                   [{"type": "bar"}, {"type": "box"}, {"type": "indicator"}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        # Quality distribution
        fig.add_trace(
            go.Histogram(
                x=catalog_df['quality_score'],
                nbinsx=15,
                name='Quality',
                marker_color=self.colors['primary'],
                opacity=0.8
            ),
            row=1, col=1
        )
        
        # Quality vs Completeness
        completeness = 100 - catalog_df['missing_percentage']
        fig.add_trace(
            go.Scatter(
                x=completeness,
                y=catalog_df['quality_score'],
                mode='markers+text',
                text=catalog_df['dataset_name'].str[:8],
                textposition='top center',
                marker=dict(
                    size=catalog_df['numeric_columns'] * 3,
                    color=catalog_df['rows'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Rows", x=0.65, len=0.4)
                ),
                name='Datasets'
            ),
            row=1, col=2
        )
        
        # Dataset size bubble chart
        fig.add_trace(
            go.Scatter(
                x=catalog_df['rows'],
                y=catalog_df['columns'],
                mode='markers',
                marker=dict(
                    size=catalog_df['quality_score'] * 30,
                    color=catalog_df['quality_score'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Quality", x=1.02, len=0.4)
                ),
                text=catalog_df['dataset_name'],
                name='Size Analysis'
            ),
            row=1, col=3
        )
        
        # Missing data patterns
        fig.add_trace(
            go.Bar(
                x=catalog_df['dataset_name'][:10],
                y=catalog_df['missing_percentage'][:10],
                marker_color=catalog_df['missing_percentage'][:10],
                colorscale='Reds',
                name='Missing %'
            ),
            row=2, col=1
        )
        
        # Outlier analysis
        outlier_ratio = catalog_df['outliers'] / catalog_df['rows'] * 100
        fig.add_trace(
            go.Box(
                y=outlier_ratio,
                name='Outlier Ratio',
                marker_color=self.colors['warning']
            ),
            row=2, col=2
        )
        
        # Data freshness indicator
        avg_quality = catalog_df['quality_score'].mean()
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=avg_quality,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall Quality"},
                delta={'reference': 0.8},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': self.colors['success']},
                    'steps': [
                        {'range': [0, 0.6], 'color': "lightgray"},
                        {'range': [0.6, 0.8], 'color': "yellow"},
                        {'range': [0.8, 1], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9
                    }
                }
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="<b>CBK Data Quality Intelligence Dashboard</b>",
            title_x=0.5,
            title_font_size=24,
            showlegend=False,
            template='plotly_white',
            font=dict(family="Inter, sans-serif")
        )
        
        # Update axes
        fig.update_xaxes(title_text="Quality Score", row=1, col=1)
        fig.update_xaxes(title_text="Completeness (%)", row=1, col=2)
        fig.update_xaxes(title_text="Number of Rows", row=1, col=3)
        fig.update_xaxes(title_text="Dataset", row=2, col=1, tickangle=45)
        
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Quality Score", row=1, col=2)
        fig.update_yaxes(title_text="Number of Columns", row=1, col=3)
        fig.update_yaxes(title_text="Missing %", row=2, col=1)
        fig.update_yaxes(title_text="Outlier Ratio %", row=2, col=2)
        
        return fig
    
    def render_main(self):
        """Main dashboard rendering with enhanced UI"""
        
        # Render header
        self.render_header()
        
        # Load data
        if not self.load_data():
            st.stop()
        
        # Sidebar controls
        controls = self.render_enhanced_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Executive Dashboard",
            "🔮 Advanced Forecasting", 
            "🕸️ Network Analysis",
            "📈 Market Intelligence",
            "📝 Policy Intelligence"
        ])
        
        with tab1:
            self.render_enhanced_kpi_cards()
            st.markdown("---")
            self.render_advanced_data_overview()
        
        with tab2:
            self.render_advanced_forecasting_panel(controls)
        
        with tab3:
            self.render_network_analysis_panel()
        
        with tab4:
            self.render_market_intelligence_panel()
        
        with tab5:
            self.render_policy_intelligence_panel()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #6c757d; padding: 2rem;">
            <i class="fas fa-brain"></i> <strong>NERVA v2.0</strong> | 
            <i class="fas fa-university"></i> Central Bank of Kenya | 
            <i class="fas fa-clock"></i> Last updated: {}
        </div>
        """.format(datetime.now().strftime('%Y-%m-%d %H:%M')), unsafe_allow_html=True)
    
    def render_advanced_forecasting_panel(self, controls):
        """Enhanced forecasting panel with advanced models"""
        
        st.markdown("## <i class="fas fa-crystal-ball"></i> Advanced Forecasting Engine", unsafe_allow_html=True)
        
        if not controls['selected_dataset'] or not controls['target_variable']:
            st.markdown("""
            <div class="alert-info">
                <i class="fas fa-info-circle"></i> Please select a dataset and target variable from the sidebar
            </div>
            """, unsafe_allow_html=True)
            return
        
        dataset_name = controls['selected_dataset']
        target_var = controls['target_variable']
        
        # Dataset information
        df = self.datasets[dataset_name]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Data Points", f"{len(df):,}")
        with col2:
            latest_value = df[target_var].iloc[-1] if target_var in df.columns else 0
            st.metric("📈 Latest Value", f"{latest_value:.4f}")
        with col3:
            mean_value = df[target_var].mean() if target_var in df.columns else 0
            st.metric("📊 Mean", f"{mean_value:.4f}")
        with col4:
            std_value = df[target_var].std() if target_var in df.columns else 0
            st.metric("📊 Volatility", f"{std_value:.4f}")
        
        # Model training section
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### <i class="fas fa-cogs"></i> Model Training", unsafe_allow_html=True)
        
        with col2:
            if st.button("🚀 Train Models", type="primary"):
                self.train_forecasting_models(df, target_var, controls)
    
    def train_forecasting_models(self, df, target_var, controls):
        """Train forecasting models with progress tracking"""
        
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Train baseline models
                status_text.text("Training baseline ensemble...")
                progress_bar.progress(20)
                
                baseline_forecaster = train_baseline_forecaster(df, target_var)
                st.session_state[f'baseline_{target_var}'] = baseline_forecaster
                
                progress_bar.progress(50)
                
                # Train advanced models if enabled
                if controls['advanced_mode']:
                    status_text.text("Training advanced models (Transformer, GARCH)...")
                    progress_bar.progress(70)
                    
                    advanced_forecaster = train_advanced_forecaster(df, target_var)
                    st.session_state[f'advanced_{target_var}'] = advanced_forecaster
                
                progress_bar.progress(100)
                status_text.text("Training completed!")
                
                # Display results
                self.display_model_results(baseline_forecaster, target_var, controls)
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
            finally:
                progress_bar.empty()
                status_text.empty()
    
    def display_model_results(self, forecaster, target_var, controls):
        """Display comprehensive model results"""
        
        st.markdown("### <i class="fas fa-chart-area"></i> Model Performance", unsafe_allow_html=True)
        
        # Performance metrics
        performance_df = forecaster.get_model_performance()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance table
            st.markdown("#### Model Comparison")
            styled_performance = performance_df.style.format({
                'rmse': '{:.4f}',
                'mae': '{:.4f}',
                'mape': '{:.2f}%'
            }).background_gradient(subset=['rmse', 'mae', 'mape'], cmap='RdYlGn_r')
            
            st.dataframe(styled_performance, use_container_width=True)
        
        with col2:
            # Performance visualization
            fig = go.Figure()
            
            models = performance_df['model'].unique()
            metrics = ['rmse', 'mae', 'mape']
            
            for i, metric in enumerate(metrics):
                fig.add_trace(go.Scatter(
                    x=models,
                    y=performance_df[performance_df['target'] == target_var][metric],
                    mode='markers+lines',
                    name=metric.upper(),
                    line=dict(width=3),
                    marker=dict(size=10)
                ))
            
            fig.update_layout(
                title="Model Performance Comparison",
                xaxis_title="Models",
                yaxis_title="Error Metrics",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_network_analysis_panel(self):
        """Network analysis panel placeholder"""
        st.markdown("## <i class="fas fa-project-diagram"></i> Network Analysis", unsafe_allow_html=True)
        st.markdown("""
        <div class="alert-info">
            <i class="fas fa-construction"></i> <strong>Coming in Sprint 2:</strong>
            <ul>
                <li>Dynamic institutional network mapping</li>
                <li>Systemic risk propagation analysis</li>
                <li>Contagion pathway detection</li>
                <li>Interactive network exploration</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    def render_market_intelligence_panel(self):
        """Market intelligence panel"""
        st.markdown("## <i class="fas fa-chart-line"></i> Market Intelligence", unsafe_allow_html=True)
        st.markdown("""
        <div class="alert-info">
            <i class="fas fa-tools"></i> <strong>In Development:</strong>
            <ul>
                <li>Real-time market sentiment analysis</li>
                <li>Cross-asset correlation matrices</li>
                <li>Volatility regime detection</li>
                <li>Stress testing scenarios</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    def render_policy_intelligence_panel(self):
        """Policy intelligence panel"""
        st.markdown("## <i class="fas fa-file-contract"></i> Policy Intelligence", unsafe_allow_html=True)
        st.markdown("""
        <div class="alert-info">
            <i class="fas fa-robot"></i> <strong>AI-Powered Features:</strong>
            <ul>
                <li>Automated policy brief generation</li>
                <li>Impact assessment modeling</li>
                <li>Prescriptive recommendations</li>
                <li>Regulatory compliance monitoring</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Main execution
def main():
    """Main application entry point"""
    dashboard = EnhancedNERVADashboard()
    dashboard.render_main()

if __name__ == "__main__":
    main()
