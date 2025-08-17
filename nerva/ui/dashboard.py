"""
NERVA: National Economic & Risk Visual Analytics
GODMODE_X: Streamlit Command Center
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# NERVA imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import config
from etl.processor import CBKDataProcessor, get_data_catalog
from models.baseline import BaselineForecaster, train_baseline_forecaster

# Page configuration
st.set_page_config(
    page_title=config.ui.page_title,
    page_icon=config.ui.page_icon,
    layout=config.ui.layout,
    initial_sidebar_state="expanded"
)

class NERVADashboard:
    """Main NERVA Dashboard Controller"""
    
    def __init__(self):
        self.processor = CBKDataProcessor()
        self.datasets = {}
        self.forecasters = {}
        
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
    
    def load_data(self):
        """Load and cache CBK data"""
        if not st.session_state.data_loaded:
            with st.spinner("🔄 Loading CBK Economic Data Archive..."):
                try:
                    self.datasets = self.processor.scan_all_files()
                    st.session_state.datasets = self.datasets
                    st.session_state.data_loaded = True
                    st.success(f"✅ Loaded {len(self.datasets)} datasets successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to load data: {str(e)}")
                    return False
        else:
            self.datasets = st.session_state.datasets
        return True
    
    def render_sidebar(self):
        """Render main control sidebar"""
        st.sidebar.title("🧠 NERVA Control Center")
        
        # Data refresh
        if st.sidebar.button("🔄 Refresh Data", help="Reload CBK data archive"):
            st.session_state.data_loaded = False
            st.experimental_rerun()
        
        st.sidebar.markdown("---")
        
        # Dataset selector
        if self.datasets:
            dataset_names = list(self.datasets.keys())
            selected_dataset = st.sidebar.selectbox(
                "📊 Select Dataset",
                dataset_names,
                index=0,
                help="Choose dataset for analysis"
            )
            
            # Target variable selector for selected dataset
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
                    st.sidebar.warning("No numeric columns found for forecasting")
                    target_variable = None
            else:
                target_variable = None
        else:
            selected_dataset = None
            target_variable = None
        
        st.sidebar.markdown("---")
        
        # Forecasting controls
        st.sidebar.subheader("🔮 Forecast Settings")
        
        forecast_horizon = st.sidebar.slider(
            "Forecast Horizon (months)",
            min_value=1,
            max_value=24,
            value=6,
            help="Number of months to forecast ahead"
        )
        
        model_selection = st.sidebar.multiselect(
            "🤖 Models to Use",
            ['lightgbm', 'random_forest', 'linear', 'arima', 'ets', 'ensemble'],
            default=['ensemble'],
            help="Select forecasting models"
        )
        
        st.sidebar.markdown("---")
        
        # Scenario simulation controls
        st.sidebar.subheader("🎭 Scenario Simulator")
        
        policy_rate_change = st.sidebar.slider(
            "Policy Rate Change (bps)",
            min_value=-500,
            max_value=500,
            value=0,
            step=25,
            help="Simulate policy rate changes"
        )
        
        fx_shock = st.sidebar.slider(
            "FX Shock (%)",
            min_value=-20.0,
            max_value=20.0,
            value=0.0,
            step=1.0,
            help="Simulate currency shock"
        )
        
        return {
            'selected_dataset': selected_dataset,
            'target_variable': target_variable,
            'forecast_horizon': forecast_horizon,
            'model_selection': model_selection,
            'policy_rate_change': policy_rate_change,
            'fx_shock': fx_shock
        }
    
    def render_kpi_cards(self):
        """Render top-level KPI cards"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📈 Systemic Risk Index",
                value="0.23",
                delta="-0.05",
                delta_color="inverse",
                help="Overall financial system risk score (0-1)"
            )
        
        with col2:
            st.metric(
                label="💹 1M Inflation Forecast",
                value="5.2%",
                delta="+0.3%",
                help="1-month ahead inflation prediction"
            )
        
        with col3:
            st.metric(
                label="💱 3M FX Forecast (USD/KES)",
                value="142.5",
                delta="+2.8",
                help="3-month USD/KES exchange rate forecast"
            )
        
        with col4:
            st.metric(
                label="🏦 Reserve Adequacy",
                value="4.2 months",
                delta="-0.1",
                delta_color="inverse",
                help="Import cover in months"
            )
    
    def render_data_overview(self):
        """Render data catalog and quality overview"""
        
        st.header("📋 CBK Data Catalog")
        
        # Generate data catalog
        try:
            catalog_df = self.processor.generate_data_catalog()
            
            # Quality score color coding
            def color_quality_score(val):
                if val >= 0.8:
                    return 'background-color: #d4edda'  # Green
                elif val >= 0.6:
                    return 'background-color: #fff3cd'  # Yellow
                else:
                    return 'background-color: #f8d7da'  # Red
            
            styled_catalog = catalog_df.style.applymap(
                color_quality_score, 
                subset=['quality_score']
            ).format({
                'quality_score': '{:.2f}',
                'missing_percentage': '{:.1f}%'
            })
            
            st.dataframe(styled_catalog, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Total Datasets",
                    len(catalog_df),
                    help="Number of processed datasets"
                )
            
            with col2:
                avg_quality = catalog_df['quality_score'].mean()
                st.metric(
                    "Average Quality Score",
                    f"{avg_quality:.2f}",
                    help="Mean data quality across all datasets"
                )
            
            with col3:
                total_rows = catalog_df['rows'].sum()
                st.metric(
                    "Total Data Points",
                    f"{total_rows:,}",
                    help="Total rows across all datasets"
                )
        
        except Exception as e:
            st.error(f"Failed to generate data catalog: {str(e)}")
    
    def render_forecasting_panel(self, controls):
        """Render forecasting analysis panel"""
        
        if not controls['selected_dataset'] or not controls['target_variable']:
            st.info("👆 Please select a dataset and target variable from the sidebar")
            return
        
        st.header(f"🔮 Forecasting: {controls['target_variable']}")
        
        dataset_name = controls['selected_dataset']
        target_var = controls['target_variable']
        
        if dataset_name not in self.datasets:
            st.error("Dataset not found")
            return
        
        df = self.datasets[dataset_name]
        
        # Show basic data info
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Data Points", len(df))
            st.metric("Date Range", f"{len(df)} periods")
        
        with col2:
            if target_var in df.columns:
                latest_value = df[target_var].iloc[-1]
                st.metric("Latest Value", f"{latest_value:.2f}")
                
                mean_value = df[target_var].mean()
                st.metric("Historical Mean", f"{mean_value:.2f}")
        
        # Train forecasting models
        if st.button("🚀 Train Forecasting Models", type="primary"):
            with st.spinner("Training baseline ensemble..."):
                try:
                    forecaster = train_baseline_forecaster(df, target_var)
                    
                    # Store in session state
                    st.session_state[f'forecaster_{dataset_name}_{target_var}'] = forecaster
                    
                    st.success("✅ Models trained successfully!")
                    
                    # Show model performance
                    performance_df = forecaster.get_model_performance()
                    
                    st.subheader("📊 Model Performance")
                    st.dataframe(performance_df.style.format({
                        'rmse': '{:.4f}',
                        'mae': '{:.4f}',
                        'mape': '{:.2f}%'
                    }), use_container_width=True)
                    
                    # Visualize predictions vs actual
                    self._plot_forecast_results(forecaster, df, target_var)
                    
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
        
        # If models are trained, show predictions
        forecaster_key = f'forecaster_{dataset_name}_{target_var}'
        if forecaster_key in st.session_state:
            forecaster = st.session_state[forecaster_key]
            
            st.subheader("🎯 Current Predictions")
            
            # Get predictions for different horizons
            predictions_data = []
            for horizon in [1, 3, 6, 12]:
                try:
                    preds = forecaster.predict(df, horizon)
                    for model_name, pred_value in preds.items():
                        predictions_data.append({
                            'Horizon (months)': horizon,
                            'Model': model_name,
                            'Prediction': pred_value
                        })
                except Exception as e:
                    st.warning(f"Prediction failed for horizon {horizon}: {str(e)}")
            
            if predictions_data:
                pred_df = pd.DataFrame(predictions_data)
                
                # Pivot table for better display
                pivot_pred = pred_df.pivot(index='Horizon (months)', columns='Model', values='Prediction')
                st.dataframe(pivot_pred.style.format('{:.4f}'), use_container_width=True)
                
                # Plot predictions
                self._plot_prediction_horizons(pivot_pred)
    
    def _plot_forecast_results(self, forecaster, df, target_var):
        """Plot forecasting results"""
        
        if target_var not in df.columns:
            return
        
        # Get historical data
        historical_data = df[target_var].dropna()
        
        # Create time-based plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=list(range(len(historical_data))),
            y=historical_data.values,
            mode='lines',
            name='Historical',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title=f"📈 Historical Data: {target_var}",
            xaxis_title="Time Period",
            yaxis_title="Value",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_prediction_horizons(self, pred_df):
        """Plot prediction horizons"""
        
        fig = go.Figure()
        
        for model in pred_df.columns:
            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df[model],
                mode='lines+markers',
                name=model,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="🎯 Multi-Horizon Forecasts",
            xaxis_title="Forecast Horizon (months)",
            yaxis_title="Predicted Value",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_main(self):
        """Main dashboard rendering"""
        
        # Header
        st.title("🧠 NERVA: National Economic & Risk Visual Analytics")
        st.markdown("**Central Bank of Kenya Decision Support System**")
        
        # Load data
        if not self.load_data():
            st.stop()
        
        # Sidebar controls
        controls = self.render_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", 
            "🔮 Forecasting", 
            "🕸️ Network Analysis", 
            "📝 Policy Brief"
        ])
        
        with tab1:
            self.render_kpi_cards()
            st.markdown("---")
            self.render_data_overview()
        
        with tab2:
            self.render_forecasting_panel(controls)
        
        with tab3:
            st.header("🕸️ Systemic Risk Network")
            st.info("🚧 Graph network analysis coming in Sprint 2")
            
            # Placeholder network visualization
            st.markdown("""
            **Coming Soon:**
            - Dynamic institutional network mapping
            - Contagion risk pathways
            - Systemic risk scoring
            - Interactive node exploration
            """)
        
        with tab4:
            st.header("📝 Automated Policy Brief")
            st.info("🚧 NLP-powered policy analysis coming in Sprint 2")
            
            # Placeholder policy brief
            st.markdown("""
            **Coming Soon:**
            - Auto-generated executive summaries
            - Policy impact assessments  
            - Prescriptive recommendations
            - PDF report export
            """)
        
        # Footer
        st.markdown("---")
        st.markdown(
            "**NERVA v1.0** | Built with ❤️ for Central Bank of Kenya | "
            f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )

# Main execution
def main():
    """Main application entry point"""
    dashboard = NERVADashboard()
    dashboard.render_main()

if __name__ == "__main__":
    main()
