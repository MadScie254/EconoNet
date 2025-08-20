"""
Predictive Models Dashboard
==========================

Streamlit page for advanced forecasting models with interactive model selection,
data upload, parameter tuning, and comprehensive visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import plotting utilities
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.plotting import (
    create_time_series_plot, create_forecast_plot, create_model_comparison_plot,
    create_distribution_plot, create_feature_importance_plot
)
from src.models.forecasting import (
    ARIMAForecaster, ProphetForecaster, XGBoostForecaster, 
    LSTMForecaster, EnsembleForecaster, create_forecasting_pipeline
)

# Page configuration
st.set_page_config(
    page_title="🔮 Predictive Models",
    page_icon="🔮",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .model-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .prediction-highlight {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

def render_model_selection_sidebar():
    """Render model selection and configuration sidebar"""
    st.sidebar.markdown("## 🔮 Model Configuration")
    
    # Model selection
    model_options = {
        "📊 ARIMA/SARIMA": "arima",
        "📈 Facebook Prophet": "prophet", 
        "🚀 XGBoost": "xgboost",
        "🧠 LSTM Neural Network": "lstm",
        "🏆 Ensemble (All Models)": "ensemble"
    }
    
    selected_model = st.sidebar.selectbox(
        "Select Forecasting Model",
        list(model_options.keys()),
        index=4  # Default to ensemble
    )
    
    model_type = model_options[selected_model]
    
    # Model parameters
    st.sidebar.markdown("### ⚙️ Model Parameters")
    
    forecast_horizon = st.sidebar.slider(
        "Forecast Horizon",
        min_value=1, max_value=36, value=12,
        help="Number of periods to forecast into the future"
    )
    
    confidence_level = st.sidebar.select_slider(
        "Confidence Level",
        options=[0.80, 0.85, 0.90, 0.95, 0.99],
        value=0.95,
        format_func=lambda x: f"{x*100:.0f}%"
    )
    
    # Advanced parameters based on model type
    advanced_params = {}
    
    if model_type == "arima":
        st.sidebar.markdown("#### ARIMA Parameters")
        auto_order = st.sidebar.checkbox("Auto-detect ARIMA order", value=True)
        if not auto_order:
            advanced_params['order'] = (
                st.sidebar.number_input("p (AR)", 0, 5, 1),
                st.sidebar.number_input("d (I)", 0, 2, 1),
                st.sidebar.number_input("q (MA)", 0, 5, 1)
            )
    
    elif model_type == "prophet":
        st.sidebar.markdown("#### Prophet Parameters")
        advanced_params['yearly_seasonality'] = st.sidebar.checkbox("Yearly Seasonality", True)
        advanced_params['weekly_seasonality'] = st.sidebar.checkbox("Weekly Seasonality", False)
        advanced_params['daily_seasonality'] = st.sidebar.checkbox("Daily Seasonality", False)
    
    elif model_type == "xgboost":
        st.sidebar.markdown("#### XGBoost Parameters")
        advanced_params['n_estimators'] = st.sidebar.slider("Number of Trees", 50, 500, 100)
        advanced_params['max_depth'] = st.sidebar.slider("Max Depth", 3, 15, 6)
        advanced_params['learning_rate'] = st.sidebar.slider("Learning Rate", 0.01, 0.3, 0.1)
        advanced_params['lag_features'] = st.sidebar.slider("Lag Features", 6, 24, 12)
    
    elif model_type == "lstm":
        st.sidebar.markdown("#### LSTM Parameters")
        advanced_params['sequence_length'] = st.sidebar.slider("Sequence Length", 6, 24, 12)
        advanced_params['hidden_units'] = st.sidebar.slider("Hidden Units", 32, 128, 50)
        advanced_params['epochs'] = st.sidebar.slider("Training Epochs", 50, 200, 100)
        advanced_params['dropout_rate'] = st.sidebar.slider("Dropout Rate", 0.1, 0.5, 0.2)
    
    elif model_type == "ensemble":
        st.sidebar.markdown("#### Ensemble Parameters")
        ensemble_method = st.sidebar.selectbox(
            "Ensemble Method",
            ["simple_average", "weighted_average", "stacking"],
            index=1
        )
        advanced_params['ensemble_method'] = ensemble_method
    
    return {
        'model_type': model_type,
        'forecast_horizon': forecast_horizon,
        'confidence_level': confidence_level,
        **advanced_params
    }

def load_sample_data():
    """Generate sample economic time series data"""
    dates = pd.date_range('2010-01-01', '2023-12-31', freq='M')
    
    # Generate synthetic economic data
    np.random.seed(42)
    trend = np.linspace(100, 150, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    noise = np.random.normal(0, 5, len(dates))
    
    gdp = trend + seasonal + noise
    inflation = 2 + 0.5 * np.sin(2 * np.pi * np.arange(len(dates)) / 12) + np.random.normal(0, 0.3, len(dates))
    unemployment = 5 + 2 * np.sin(2 * np.pi * np.arange(len(dates)) / 24) + np.random.normal(0, 0.5, len(dates))
    
    return pd.DataFrame({
        'date': dates,
        'gdp': gdp,
        'inflation': inflation,
        'unemployment': unemployment,
        'debt': gdp * 0.6 + np.random.normal(0, 3, len(dates))
    }).set_index('date')

def main():
    """Main predictive models page"""
    
    # Header
    st.markdown("# 🔮 Advanced Predictive Models")
    st.markdown("""
    <div class="model-card">
        <h3>🚀 State-of-the-Art Forecasting Platform</h3>
        <p>• Multiple algorithms: ARIMA, Prophet, XGBoost, LSTM, Ensemble methods</p>
        <p>• Automated hyperparameter tuning and model selection</p>
        <p>• Real-time prediction intervals and uncertainty quantification</p>
        <p>• Interactive visualizations and comprehensive model evaluation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get model configuration
    config = render_model_selection_sidebar()
    
    # Data upload section
    st.markdown("## 📊 Data Input")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your time series data",
            type=['csv', 'xlsx'],
            help="Upload a CSV or Excel file with time series data. Should contain a date column and one or more numeric columns."
        )
    
    with col2:
        use_sample = st.button("📈 Use Sample Data", help="Load sample economic time series data")
    
    # Load data
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            # Try to detect date column
            date_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                data[date_cols[0]] = pd.to_datetime(data[date_cols[0]])
                data = data.set_index(date_cols[0])
            
            st.success(f"✅ Data loaded successfully: {len(data)} rows, {len(data.columns)} columns")
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            data = None
    
    elif use_sample:
        data = load_sample_data()
        st.success("✅ Sample data loaded successfully")
    
    else:
        data = None
        st.info("👆 Please upload a dataset or use sample data to begin analysis")
    
    if data is not None:
        # Data preview and column selection
        st.markdown("## 🔍 Data Preview & Configuration")
        
        tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "📈 Data Visualization", "⚙️ Column Selection"])
        
        with tab1:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader("Dataset Preview")
                st.dataframe(data.head(10), use_container_width=True)
            
            with col2:
                st.subheader("Data Info")
                st.write(f"**Rows:** {len(data):,}")
                st.write(f"**Columns:** {len(data.columns)}")
                st.write(f"**Date Range:** {data.index.min()} to {data.index.max()}")
                
                # Data quality metrics
                missing_pct = (data.isnull().sum() / len(data) * 100).round(2)
                if missing_pct.max() > 0:
                    st.write("**Missing Data:**")
                    for col, pct in missing_pct[missing_pct > 0].items():
                        st.write(f"• {col}: {pct}%")
        
        with tab2:
            # Quick visualization
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 0:
                viz_cols = st.multiselect(
                    "Select columns to visualize",
                    numeric_cols,
                    default=numeric_cols[:min(3, len(numeric_cols))]
                )
                
                if viz_cols:
                    fig = create_time_series_plot(
                        data[viz_cols], 
                        viz_cols, 
                        title="Time Series Overview",
                        show_trend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Target variable selection
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) == 0:
                st.error("❌ No numeric columns found in the dataset")
                return
            
            target_column = st.selectbox(
                "Select target variable to forecast",
                numeric_cols,
                help="Choose the variable you want to predict"
            )
            
            # Feature columns (optional)
            feature_columns = st.multiselect(
                "Select additional feature columns (optional)",
                [col for col in numeric_cols if col != target_column],
                help="Additional variables that might help predict the target"
            )
        
        # Model training and prediction
        if 'target_column' in locals():
            st.markdown("## 🚀 Model Training & Prediction")
            
            if st.button("🔥 Train Model & Generate Forecasts", type="primary"):
                
                with st.spinner("🧠 Training advanced forecasting models..."):
                    
                    try:
                        # Prepare data
                        model_data = data[[target_column] + feature_columns].dropna()
                        
                        if len(model_data) < 10:
                            st.error("❌ Insufficient data points for modeling (need at least 10)")
                            return
                        
                        # Create forecasting pipeline
                        model = create_forecasting_pipeline(
                            data=model_data,
                            target_column=target_column,
                            model_type=config['model_type'],
                            forecast_horizon=config['forecast_horizon'],
                            confidence_level=config['confidence_level'],
                            **{k: v for k, v in config.items() if k not in ['model_type', 'forecast_horizon', 'confidence_level']}
                        )
                        
                        # Generate forecasts
                        forecasts, lower_bound, upper_bound = model.predict_with_intervals()
                        
                        # Create forecast dates
                        last_date = model_data.index[-1]
                        if hasattr(last_date, 'to_pydatetime'):
                            forecast_dates = pd.date_range(
                                start=last_date + pd.offsets.MonthBegin(1),
                                periods=config['forecast_horizon'],
                                freq='M'
                            )
                        else:
                            forecast_dates = pd.RangeIndex(
                                start=len(model_data),
                                stop=len(model_data) + config['forecast_horizon']
                            )
                        
                        st.success("✅ Model training completed successfully!")
                        
                        # Results visualization
                        st.markdown("### 📈 Forecast Results")
                        
                        # Create forecast plot
                        fig_forecast = create_forecast_plot(
                            historical_data=model_data[target_column],
                            forecasts=forecasts,
                            forecast_dates=forecast_dates,
                            confidence_intervals=(lower_bound, upper_bound),
                            title=f"{config['model_type'].upper()} Forecast for {target_column}",
                            model_name=config['model_type'].title()
                        )
                        
                        st.plotly_chart(fig_forecast, use_container_width=True)
                        
                        # Prediction summary
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="prediction-highlight">
                                Next Period Prediction<br>
                                <span style="font-size: 1.5rem;">{forecasts[0]:,.2f}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="prediction-highlight">
                                12-Month Average<br>
                                <span style="font-size: 1.5rem;">{np.mean(forecasts[:min(12, len(forecasts))]):,.2f}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            change_pct = ((forecasts[-1] / model_data[target_column].iloc[-1]) - 1) * 100
                            st.markdown(f"""
                            <div class="prediction-highlight">
                                Total Change<br>
                                <span style="font-size: 1.5rem;">{change_pct:+.1f}%</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Model evaluation metrics
                        if hasattr(model, 'calculate_metrics'):
                            st.markdown("### 📊 Model Performance")
                            
                            # In-sample evaluation
                            y_true = model_data[target_column].values
                            if hasattr(model, 'predict'):
                                try:
                                    y_pred = model.predict()
                                    if len(y_pred) != len(y_true):
                                        y_pred = y_pred[:len(y_true)]
                                    
                                    metrics = model.calculate_metrics(y_true, y_pred)
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        st.metric("R² Score", f"{metrics.get('R2', 0):.4f}")
                                    with col2:
                                        st.metric("RMSE", f"{metrics.get('RMSE', 0):.2f}")
                                    with col3:
                                        st.metric("MAE", f"{metrics.get('MAE', 0):.2f}")
                                    with col4:
                                        st.metric("MAPE", f"{metrics.get('MAPE', 0):.2f}%")
                                    
                                except Exception as e:
                                    st.warning(f"Could not calculate in-sample metrics: {e}")
                        
                        # Feature importance (if available)
                        if hasattr(model, 'get_feature_importance') and len(feature_columns) > 0:
                            try:
                                importance = model.get_feature_importance(model.feature_names)
                                if importance:
                                    st.markdown("### 🎯 Feature Importance")
                                    
                                    fig_importance = create_feature_importance_plot(
                                        importance[:15],  # Top 15 features
                                        title="Top Feature Importance"
                                    )
                                    st.plotly_chart(fig_importance, use_container_width=True)
                            except:
                                pass  # Skip if feature importance not available
                        
                        # Downloadable results
                        st.markdown("### 💾 Export Results")
                        
                        # Create results DataFrame
                        results_df = pd.DataFrame({
                            'Date': forecast_dates,
                            'Forecast': forecasts,
                            'Lower_Bound': lower_bound,
                            'Upper_Bound': upper_bound,
                            'Model': config['model_type']
                        })
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Forecast Results",
                            data=csv,
                            file_name=f"{target_column}_forecast_{config['model_type']}.csv",
                            mime="text/csv"
                        )
                        
                        # Model comparison (if ensemble)
                        if config['model_type'] == 'ensemble' and hasattr(model, 'individual_scores'):
                            st.markdown("### 🏆 Model Comparison")
                            
                            if model.individual_scores:
                                comparison_data = {
                                    f'Model_{i}': {'R2': score} 
                                    for i, score in enumerate(model.individual_scores)
                                }
                                
                                if comparison_data:
                                    fig_comparison = create_model_comparison_plot(
                                        comparison_data,
                                        metrics=['R2'],
                                        title="Ensemble Model Performance"
                                    )
                                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"❌ Model training failed: {str(e)}")
                        st.error("Please check your data format and try again.")
            
            # Model information
            st.markdown("### ℹ️ Model Information")
            
            model_info = {
                "arima": {
                    "name": "ARIMA/SARIMA",
                    "description": "AutoRegressive Integrated Moving Average model with seasonal components",
                    "strengths": "• Excellent for stationary time series\n• Handles seasonality well\n• Interpretable parameters",
                    "best_for": "Economic indicators, financial time series, seasonal data"
                },
                "prophet": {
                    "name": "Facebook Prophet",
                    "description": "Additive model with trend, seasonality, and holiday effects",
                    "strengths": "• Robust to missing data\n• Handles holidays and events\n• Automatic seasonality detection",
                    "best_for": "Business metrics, daily data, irregular patterns"
                },
                "xgboost": {
                    "name": "XGBoost",
                    "description": "Gradient boosting with advanced regularization and feature engineering",
                    "strengths": "• Handles non-linear patterns\n• Feature importance\n• High accuracy",
                    "best_for": "Complex datasets, multiple features, non-linear relationships"
                },
                "lstm": {
                    "name": "LSTM Neural Network",
                    "description": "Long Short-Term Memory network with bidirectional architecture",
                    "strengths": "• Captures long-term dependencies\n• Non-linear modeling\n• Deep learning power",
                    "best_for": "Complex patterns, long sequences, non-linear data"
                },
                "ensemble": {
                    "name": "Ensemble Methods",
                    "description": "Combines multiple models for robust predictions",
                    "strengths": "• Reduces overfitting\n• Combines model strengths\n• Better generalization",
                    "best_for": "Production forecasting, maximum accuracy, risk reduction"
                }
            }
            
            if config['model_type'] in model_info:
                info = model_info[config['model_type']]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>{info['name']}</h4>
                        <p>{info['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    **Key Strengths:**
                    {info['strengths']}
                    
                    **Best For:** {info['best_for']}
                    """)

if __name__ == "__main__":
    main()
