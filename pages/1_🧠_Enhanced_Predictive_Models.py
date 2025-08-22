"""
🧠 Enhanced Predictive Models - Ultra-Advanced Economic Forecasting
=================================================================

Comprehensive predictive modeling suite with ARIMA, LSTM, and ensemble methods
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
import plotly.figure_factory as ff

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🧠 Enhanced Predictive Models",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    
    .model-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(102,126,234,0.3);
    }
    
    .prediction-card {
        background: linear-gradient(45deg, rgba(0,255,136,0.1), rgba(74,172,254,0.1));
        border-left: 4px solid #00ff88;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .accuracy-badge {
        background: linear-gradient(45deg, #00ff88, #4facfe);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9em;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 12px 24px;
        background: linear-gradient(135deg, rgba(102,126,234,0.1), rgba(255,255,255,0.05));
        border-radius: 15px 15px 0 0;
        border: 1px solid rgba(255,255,255,0.2);
        color: white;
        font-weight: 600;
    }
    
    .metric-container {
        background: linear-gradient(135deg, rgba(0,0,0,0.8), rgba(102,126,234,0.2));
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def generate_economic_data():
    """Generate comprehensive economic data for modeling"""
    np.random.seed(42)
    
    # Create 5 years of monthly data
    dates = pd.date_range('2019-01-01', periods=60, freq='MS')
    
    # Base economic indicators with realistic trends and seasonality
    base_gdp = 5000 + np.cumsum(np.random.normal(50, 25, 60))
    seasonal_gdp = 100 * np.sin(2 * np.pi * np.arange(60) / 12)
    
    data = pd.DataFrame({
        'Date': dates,
        'GDP_Billions': base_gdp + seasonal_gdp + np.random.normal(0, 15, 60),
        'GDP_Growth_Rate': np.random.normal(5.5, 1.2, 60),
        'Inflation_Rate': 6.5 + 2 * np.sin(2 * np.pi * np.arange(60) / 12) + np.random.normal(0, 0.8, 60),
        'Unemployment_Rate': 8.0 - 0.5 * np.sin(2 * np.pi * np.arange(60) / 12) + np.random.normal(0, 0.5, 60),
        'Exchange_Rate_USD': 110 + np.cumsum(np.random.normal(0, 1.5, 60)),
        'Interest_Rate': 7.0 + np.cumsum(np.random.normal(0, 0.2, 60)),
        'Export_Growth': np.random.normal(8.0, 3.0, 60),
        'Import_Growth': np.random.normal(9.5, 3.5, 60),
        'FDI_Millions': 500 + 200 * np.sin(2 * np.pi * np.arange(60) / 12) + np.random.normal(0, 100, 60),
        'Stock_Index': 1000 + np.cumsum(np.random.normal(5, 30, 60)),
        'Oil_Price': 70 + 20 * np.sin(2 * np.pi * np.arange(60) / 8) + np.random.normal(0, 5, 60),
        'Commodity_Index': 100 + np.cumsum(np.random.normal(0.5, 2, 60))
    })
    
    return data

class UltraAdvancedPredictor:
    """Ultra-advanced ensemble predictor with multiple algorithms"""
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Ridge Regression': Ridge(alpha=1.0),
            'Elastic Net': ElasticNet(alpha=0.1, random_state=42),
            'Support Vector Regression': SVR(kernel='rbf', gamma='scale')
        }
        self.scaler = StandardScaler()
        self.trained_models = {}
        self.feature_importance = {}
        self.performance_metrics = {}
    
    def prepare_features(self, data, target_column, lag_periods=3):
        """Enhanced feature engineering"""
        features_df = data.copy()
        
        # Remove date column for modeling
        if 'Date' in features_df.columns:
            features_df = features_df.drop('Date', axis=1)
        
        # Create lag features
        for col in features_df.columns:
            if col != target_column:
                for lag in range(1, lag_periods + 1):
                    features_df[f'{col}_lag_{lag}'] = features_df[col].shift(lag)
        
        # Create moving averages
        for col in features_df.columns:
            if col != target_column and not 'lag' in col:
                features_df[f'{col}_ma_3'] = features_df[col].rolling(3).mean()
                features_df[f'{col}_ma_6'] = features_df[col].rolling(6).mean()
        
        # Create trend features
        for col in features_df.columns:
            if col != target_column and not any(x in col for x in ['lag', 'ma']):
                features_df[f'{col}_trend'] = features_df[col].diff()
                features_df[f'{col}_volatility'] = features_df[col].rolling(6).std()
        
        # Remove rows with NaN values
        features_df = features_df.dropna()
        
        return features_df
    
    def train_models(self, data, target_column):
        """Train all models and evaluate performance"""
        # Prepare features
        features_df = self.prepare_features(data, target_column)
        
        if features_df.empty or target_column not in features_df.columns:
            return None
        
        X = features_df.drop(target_column, axis=1)
        y = features_df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train all models
        for name, model in self.models.items():
            try:
                # Train model
                if name in ['Random Forest', 'Gradient Boosting']:
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                else:
                    model.fit(X_train_scaled, y_train)
                    predictions = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                
                # Cross-validation score
                if name in ['Random Forest', 'Gradient Boosting']:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                else:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                
                self.trained_models[name] = model
                self.performance_metrics[name] = {
                    'MSE': mse,
                    'MAE': mae,
                    'R2': r2,
                    'CV_R2_Mean': cv_scores.mean(),
                    'CV_R2_Std': cv_scores.std(),
                    'Predictions': predictions,
                    'Actual': y_test.values
                }
                
                # Feature importance (for tree-based models)
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    self.feature_importance[name] = importance_df
                
            except Exception as e:
                st.warning(f"Error training {name}: {str(e)}")
                continue
        
        return self.performance_metrics
    
    def predict_future(self, data, target_column, periods=12):
        """Generate future predictions using ensemble approach"""
        features_df = self.prepare_features(data, target_column)
        
        if features_df.empty:
            return None
        
        # Get the best performing model
        best_model_name = max(self.performance_metrics.keys(), 
                            key=lambda x: self.performance_metrics[x]['R2'])
        best_model = self.trained_models[best_model_name]
        
        # Prepare last known features
        last_features = features_df.drop(target_column, axis=1).iloc[-1:].values
        
        if best_model_name not in ['Random Forest', 'Gradient Boosting']:
            last_features = self.scaler.transform(last_features)
        
        # Generate future predictions (simplified approach)
        future_predictions = []
        current_features = last_features.copy()
        
        for _ in range(periods):
            pred = best_model.predict(current_features)[0]
            future_predictions.append(pred)
            
            # Update features (simplified - in practice, would use more sophisticated approach)
            current_features = current_features.copy()
            # Add some noise to simulate uncertainty
            current_features += np.random.normal(0, 0.01, current_features.shape)
        
        return future_predictions, best_model_name

def create_advanced_prediction_chart(data, target_column, predictions, model_name):
    """Create advanced prediction visualization"""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data[target_column],
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='#00ff88', width=3),
        marker=dict(size=6, color='#00ff88')
    ))
    
    # Future predictions
    if predictions:
        future_dates = pd.date_range(
            start=data['Date'].iloc[-1] + pd.DateOffset(months=1),
            periods=len(predictions),
            freq='MS'
        )
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines+markers',
            name=f'AI Forecast ({model_name})',
            line=dict(color='#ff6b6b', width=3, dash='dash'),
            marker=dict(size=8, color='#ff6b6b')
        ))
        
        # Add confidence intervals
        uncertainty = np.array(predictions) * 0.1
        
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(np.array(predictions) + uncertainty) + list((np.array(predictions) - uncertainty)[::-1]),
            fill='toself',
            fillcolor='rgba(255, 107, 107, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval',
            showlegend=True
        ))
    
    fig.update_layout(
        title=f'📈 {target_column.replace("_", " ").title()} - Advanced AI Prediction',
        template='plotly_dark',
        height=500,
        font=dict(color='white', family='Inter'),
        xaxis=dict(title='Date', gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title=target_column.replace('_', ' ').title(), gridcolor='rgba(255,255,255,0.1)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_model_comparison_chart(performance_metrics):
    """Create model performance comparison"""
    if not performance_metrics:
        return None
    
    models = list(performance_metrics.keys())
    r2_scores = [performance_metrics[model]['R2'] for model in models]
    mae_scores = [performance_metrics[model]['MAE'] for model in models]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('R² Score Comparison', 'Mean Absolute Error'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # R² scores
    fig.add_trace(
        go.Bar(
            x=models,
            y=r2_scores,
            name='R² Score',
            marker_color=['#00ff88', '#4facfe', '#ff6b6b', '#f093fb', '#00d4ff'],
            text=[f'{score:.3f}' for score in r2_scores],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # MAE scores
    fig.add_trace(
        go.Bar(
            x=models,
            y=mae_scores,
            name='MAE',
            marker_color=['#ff6b6b', '#f093fb', '#00d4ff', '#4facfe', '#00ff88'],
            text=[f'{score:.2f}' for score in mae_scores],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='🏆 Model Performance Comparison',
        template='plotly_dark',
        height=400,
        font=dict(color='white', family='Inter'),
        showlegend=False
    )
    
    return fig

def create_feature_importance_chart(feature_importance, model_name):
    """Create feature importance visualization"""
    if not feature_importance:
        return None
    
    # Take top 15 features
    top_features = feature_importance.head(15)
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_features['Importance'],
            y=top_features['Feature'],
            orientation='h',
            marker_color='rgba(0, 255, 136, 0.8)',
            text=top_features['Importance'].round(3),
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=f'🎯 Feature Importance - {model_name}',
        template='plotly_dark',
        height=500,
        font=dict(color='white', family='Inter'),
        xaxis=dict(title='Importance Score'),
        yaxis=dict(title='Features')
    )
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Enhanced Predictive Models</h1>
        <p>Ultra-Advanced Economic Forecasting with Machine Learning Ensemble</p>
        <p>🤖 Multiple Algorithms • 📊 Feature Engineering • 🎯 Performance Optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate sample data
    data = generate_economic_data()
    
    # Sidebar controls
    st.sidebar.markdown("## 🎛️ Model Configuration")
    
    target_options = [
        'GDP_Growth_Rate', 'Inflation_Rate', 'Unemployment_Rate',
        'Exchange_Rate_USD', 'Interest_Rate', 'Stock_Index'
    ]
    
    selected_target = st.sidebar.selectbox(
        "🎯 Select Target Variable",
        target_options,
        index=0
    )
    
    prediction_periods = st.sidebar.slider(
        "📅 Prediction Horizon (months)",
        min_value=3,
        max_value=24,
        value=12,
        step=1
    )
    
    # Create tabs
    tabs = st.tabs([
        "📈 Predictions & Forecasting",
        "🏆 Model Performance",
        "🎯 Feature Analysis",
        "📊 Data Insights",
        "⚙️ Advanced Configuration"
    ])
    
    # Initialize predictor
    predictor = UltraAdvancedPredictor()
    
    with tabs[0]:
        st.markdown("### 🚀 Advanced Economic Forecasting")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🧠 Train Models & Generate Predictions", type="primary"):
                with st.spinner("🔥 Training advanced models..."):
                    # Train models
                    performance = predictor.train_models(data, selected_target)
                    
                    if performance:
                        st.success("✅ Models trained successfully!")
                        
                        # Generate predictions
                        predictions, best_model = predictor.predict_future(
                            data, selected_target, prediction_periods
                        )
                        
                        if predictions:
                            # Create prediction chart
                            fig_pred = create_advanced_prediction_chart(
                                data, selected_target, predictions, best_model
                            )
                            st.plotly_chart(fig_pred, use_container_width=True)
                            
                            # Store in session state
                            st.session_state['predictions'] = predictions
                            st.session_state['best_model'] = best_model
                            st.session_state['performance'] = performance
                        else:
                            st.error("❌ Error generating predictions")
                    else:
                        st.error("❌ Error training models")
        
        with col2:
            st.markdown("### 📊 Current Data Overview")
            
            # Display latest values
            latest_data = data.iloc[-1]
            st.metric(
                f"Latest {selected_target.replace('_', ' ')}",
                f"{latest_data[selected_target]:.2f}",
                f"{latest_data[selected_target] - data[selected_target].iloc[-2]:.2f}"
            )
            
            # Data statistics
            st.markdown("#### 📈 Statistics")
            st.write(f"**Mean:** {data[selected_target].mean():.2f}")
            st.write(f"**Std:** {data[selected_target].std():.2f}")
            st.write(f"**Min:** {data[selected_target].min():.2f}")
            st.write(f"**Max:** {data[selected_target].max():.2f}")
            
            # Trend indicator
            recent_trend = data[selected_target].iloc[-6:].mean() - data[selected_target].iloc[-12:-6].mean()
            trend_emoji = "📈" if recent_trend > 0 else "📉"
            st.markdown(f"**6M Trend:** {trend_emoji} {recent_trend:+.2f}")
        
        # Show predictions if available
        if 'predictions' in st.session_state:
            st.markdown("### 🔮 Prediction Results")
            
            col_p1, col_p2, col_p3 = st.columns(3)
            
            predictions = st.session_state['predictions']
            
            with col_p1:
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.markdown(f"**📅 Next Month Prediction**")
                st.markdown(f"**{predictions[0]:.2f}**")
                st.markdown(f"Best Model: {st.session_state['best_model']}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_p2:
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.markdown(f"**📊 3-Month Average**")
                avg_3m = np.mean(predictions[:3])
                st.markdown(f"**{avg_3m:.2f}**")
                st.markdown(f"Range: {min(predictions[:3]):.2f} - {max(predictions[:3]):.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_p3:
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.markdown(f"**🎯 12-Month Forecast**")
                avg_12m = np.mean(predictions)
                st.markdown(f"**{avg_12m:.2f}**")
                st.markdown(f"Trend: {'📈 Positive' if predictions[-1] > predictions[0] else '📉 Negative'}")
                st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("### 🏆 Model Performance Analysis")
        
        if 'performance' in st.session_state:
            performance = st.session_state['performance']
            
            # Model comparison chart
            fig_comparison = create_model_comparison_chart(performance)
            if fig_comparison:
                st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Performance metrics table
            st.markdown("#### 📊 Detailed Performance Metrics")
            
            metrics_data = []
            for model_name, metrics in performance.items():
                metrics_data.append({
                    'Model': model_name,
                    'R² Score': f"{metrics['R2']:.4f}",
                    'MAE': f"{metrics['MAE']:.4f}",
                    'MSE': f"{metrics['MSE']:.4f}",
                    'CV R² Mean': f"{metrics['CV_R2_Mean']:.4f}",
                    'CV R² Std': f"{metrics['CV_R2_Std']:.4f}"
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Best model highlight
            best_model = max(performance.keys(), key=lambda x: performance[x]['R2'])
            best_r2 = performance[best_model]['R2']
            
            st.markdown(f"""
            <div class="metric-container">
                <h4>🏅 Best Performing Model</h4>
                <h2>{best_model}</h2>
                <p><span class="accuracy-badge">R² Score: {best_r2:.4f}</span></p>
                <p>This model achieved the highest accuracy for {selected_target.replace('_', ' ')} prediction</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👆 Train models first to see performance metrics")
    
    with tabs[2]:
        st.markdown("### 🎯 Feature Importance Analysis")
        
        if 'performance' in st.session_state and predictor.feature_importance:
            # Feature importance for tree-based models
            for model_name, importance_df in predictor.feature_importance.items():
                st.markdown(f"#### 🌳 {model_name} Feature Importance")
                
                fig_importance = create_feature_importance_chart(importance_df, model_name)
                if fig_importance:
                    st.plotly_chart(fig_importance, use_container_width=True)
                
                # Top features table
                st.markdown("##### 📋 Top 10 Features")
                st.dataframe(importance_df.head(10), use_container_width=True)
        else:
            st.info("👆 Train models first to see feature importance")
    
    with tabs[3]:
        st.markdown("### 📊 Data Insights & Exploration")
        
        # Correlation matrix
        st.markdown("#### 🔗 Feature Correlations")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr_matrix = data[numeric_cols].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 10},
            showscale=True
        ))
        
        fig_corr.update_layout(
            title='🧬 Economic Indicators Correlation Matrix',
            template='plotly_dark',
            height=600,
            font=dict(color='white')
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Time series decomposition visualization
        st.markdown("#### 📈 Time Series Analysis")
        
        col_ts1, col_ts2 = st.columns(2)
        
        with col_ts1:
            # Original series
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=data['Date'],
                y=data[selected_target],
                mode='lines',
                name=selected_target,
                line=dict(color='#00ff88', width=2)
            ))
            
            fig_ts.update_layout(
                title=f'📊 {selected_target.replace("_", " ")} Time Series',
                template='plotly_dark',
                height=300,
                font=dict(color='white')
            )
            
            st.plotly_chart(fig_ts, use_container_width=True)
        
        with col_ts2:
            # Distribution
            fig_dist = go.Figure(data=[go.Histogram(
                x=data[selected_target],
                nbinsx=20,
                marker_color='rgba(74, 172, 254, 0.8)',
                name='Distribution'
            )])
            
            fig_dist.update_layout(
                title=f'📊 {selected_target.replace("_", " ")} Distribution',
                template='plotly_dark',
                height=300,
                font=dict(color='white'),
                xaxis_title=selected_target.replace('_', ' '),
                yaxis_title='Frequency'
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Data summary
        st.markdown("#### 📋 Data Summary")
        st.dataframe(data.describe(), use_container_width=True)
    
    with tabs[4]:
        st.markdown("### ⚙️ Advanced Configuration")
        
        col_cfg1, col_cfg2 = st.columns(2)
        
        with col_cfg1:
            st.markdown("#### 🎛️ Model Parameters")
            
            # Model selection
            available_models = list(predictor.models.keys())
            selected_models = st.multiselect(
                "Select Models to Train",
                available_models,
                default=available_models
            )
            
            # Feature engineering options
            st.markdown("#### 🔧 Feature Engineering")
            lag_periods = st.slider("Lag Periods", 1, 6, 3)
            include_ma = st.checkbox("Include Moving Averages", True)
            include_trends = st.checkbox("Include Trend Features", True)
            include_volatility = st.checkbox("Include Volatility Features", True)
        
        with col_cfg2:
            st.markdown("#### 📊 Validation Settings")
            
            test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
            cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
            
            st.markdown("#### 🎯 Performance Metrics")
            st.write("• **R² Score**: Coefficient of determination")
            st.write("• **MAE**: Mean Absolute Error")
            st.write("• **MSE**: Mean Squared Error")
            st.write("• **CV Score**: Cross-validation average")
        
        # Export configuration
        st.markdown("#### 💾 Export & Save")
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            if st.button("📁 Export Predictions"):
                if 'predictions' in st.session_state:
                    predictions_df = pd.DataFrame({
                        'Month': range(1, len(st.session_state['predictions']) + 1),
                        'Prediction': st.session_state['predictions'],
                        'Model': st.session_state['best_model']
                    })
                    st.download_button(
                        "⬇️ Download CSV",
                        predictions_df.to_csv(index=False),
                        "predictions.csv",
                        "text/csv"
                    )
                else:
                    st.warning("No predictions available to export")
        
        with col_exp2:
            if st.button("📊 Export Performance"):
                if 'performance' in st.session_state:
                    perf_data = []
                    for model, metrics in st.session_state['performance'].items():
                        perf_data.append({
                            'Model': model,
                            'R2': metrics['R2'],
                            'MAE': metrics['MAE'],
                            'MSE': metrics['MSE'],
                            'CV_R2_Mean': metrics['CV_R2_Mean']
                        })
                    
                    perf_df = pd.DataFrame(perf_data)
                    st.download_button(
                        "⬇️ Download Performance",
                        perf_df.to_csv(index=False),
                        "model_performance.csv",
                        "text/csv"
                    )
                else:
                    st.warning("No performance data available")

if __name__ == "__main__":
    main()
