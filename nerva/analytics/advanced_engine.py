"""
NERVA Advanced Analytics Engine
Professional Economic Intelligence System
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnalyticsEngine:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
        
        # Professional color scheme
        self.colors = {
            'primary': '#2E86C1',
            'secondary': '#E74C3C', 
            'accent': '#8E44AD',
            'success': '#27AE60',
            'warning': '#F39C12',
            'info': '#17A2B8',
            'dark': '#2C3E50',
            'light': '#ECF0F1'
        }
        
        # Set professional template
        pio.templates['nerva_pro'] = go.layout.Template(
            layout=go.Layout(
                paper_bgcolor='white',
                plot_bgcolor='#F8F9FA',
                font=dict(family='Arial, sans-serif', size=12, color='#2C3E50'),
                colorway=[self.colors['primary'], self.colors['secondary'], 
                         self.colors['accent'], self.colors['success'], self.colors['warning']]
            )
        )
        pio.templates.default = 'nerva_pro'
    
    def advanced_correlation_analysis(self, data):
        """Perform advanced correlation analysis with clustering"""
        try:
            # Select numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return None
                
            correlation_matrix = data[numeric_cols].corr()
            
            # Create advanced correlation heatmap
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(correlation_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Advanced Correlation Matrix Analysis',
                xaxis_title='Variables',
                yaxis_title='Variables',
                height=600
            )
            
            return fig, correlation_matrix
            
        except Exception as e:
            print(f"Correlation analysis error: {e}")
            return None, None
    
    def predictive_modeling_suite(self, data, target_col):
        """Advanced predictive modeling with multiple algorithms"""
        try:
            # Prepare data
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if target_col not in numeric_cols:
                return None
                
            feature_cols = [col for col in numeric_cols if col != target_col]
            if len(feature_cols) < 1:
                return None
                
            X = data[feature_cols].fillna(data[feature_cols].mean())
            y = data[target_col].fillna(data[target_col].mean())
            
            # Train-test split
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Multiple models
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
            }
            
            results = {}
            predictions = {}
            
            for name, model in models.items():
                # Train model
                if name == 'Random Forest':
                    model.fit(X_train, y_train)
                    pred_train = model.predict(X_train)
                    pred_test = model.predict(X_test)
                else:
                    model.fit(X_train_scaled, y_train)
                    pred_train = model.predict(X_train_scaled)
                    pred_test = model.predict(X_test_scaled)
                
                # Calculate metrics
                train_rmse = np.sqrt(mean_squared_error(y_train, pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))
                test_r2 = r2_score(y_test, pred_test)
                
                results[name] = {
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'r2_score': test_r2
                }
                
                predictions[name] = {
                    'actual': y_test,
                    'predicted': pred_test
                }
            
            return results, predictions
            
        except Exception as e:
            print(f"Predictive modeling error: {e}")
            return None, None
    
    def create_performance_dashboard(self, results, predictions):
        """Create professional model performance dashboard"""
        if not results or not predictions:
            return None
            
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Model Performance Metrics', 'Prediction Accuracy', 
                          'Residual Analysis', 'Feature Importance'],
            specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'histogram'}, {'type': 'bar'}]]
        )
        
        # 1. Performance metrics
        models = list(results.keys())
        rmse_scores = [results[model]['test_rmse'] for model in models]
        r2_scores = [results[model]['r2_score'] for model in models]
        
        fig.add_trace(
            go.Bar(name='RMSE', x=models, y=rmse_scores, 
                   marker_color=self.colors['primary']),
            row=1, col=1
        )
        
        # 2. Prediction vs Actual (first model)
        first_model = models[0]
        actual = predictions[first_model]['actual']
        predicted = predictions[first_model]['predicted']
        
        fig.add_trace(
            go.Scatter(x=actual, y=predicted, mode='markers',
                      marker=dict(color=self.colors['secondary'], size=8),
                      name='Predictions'),
            row=1, col=2
        )
        
        # Perfect prediction line
        min_val, max_val = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', line=dict(dash='dash', color=self.colors['warning']),
                      name='Perfect Prediction'),
            row=1, col=2
        )
        
        # 3. Residuals
        residuals = actual - predicted
        fig.add_trace(
            go.Histogram(x=residuals, nbinsx=20, 
                        marker_color=self.colors['accent'], opacity=0.7,
                        name='Residuals'),
            row=2, col=1
        )
        
        # 4. R² scores
        fig.add_trace(
            go.Bar(x=models, y=r2_scores, 
                   marker_color=self.colors['success'],
                   name='R² Score'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="NERVA Advanced Analytics Performance Dashboard",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def time_series_decomposition(self, data, date_col, value_col):
        """Advanced time series decomposition and analysis"""
        try:
            # Prepare time series data
            ts_data = data[[date_col, value_col]].copy()
            ts_data[date_col] = pd.to_datetime(ts_data[date_col], errors='coerce')
            ts_data = ts_data.dropna().sort_values(date_col)
            
            if len(ts_data) < 12:
                print("Insufficient data for time series analysis")
                return None
                
            # Set date as index
            ts_data.set_index(date_col, inplace=True)
            
            # Calculate moving averages
            ts_data['MA_3'] = ts_data[value_col].rolling(window=3).mean()
            ts_data['MA_6'] = ts_data[value_col].rolling(window=6).mean()
            ts_data['MA_12'] = ts_data[value_col].rolling(window=12).mean()
            
            # Calculate volatility
            ts_data['Volatility'] = ts_data[value_col].rolling(window=6).std()
            
            # Create time series visualization
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=['Original Series with Moving Averages', 
                              'Trend Analysis', 'Volatility Analysis'],
                vertical_spacing=0.08
            )
            
            # Original series with moving averages
            fig.add_trace(
                go.Scatter(x=ts_data.index, y=ts_data[value_col],
                          mode='lines', name='Original',
                          line=dict(color=self.colors['primary'])),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=ts_data.index, y=ts_data['MA_6'],
                          mode='lines', name='6-Period MA',
                          line=dict(color=self.colors['secondary'])),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=ts_data.index, y=ts_data['MA_12'],
                          mode='lines', name='12-Period MA',
                          line=dict(color=self.colors['accent'])),
                row=1, col=1
            )
            
            # Trend analysis (first differences)
            ts_data['Diff'] = ts_data[value_col].diff()
            fig.add_trace(
                go.Scatter(x=ts_data.index, y=ts_data['Diff'],
                          mode='lines', name='First Difference',
                          line=dict(color=self.colors['warning'])),
                row=2, col=1
            )
            
            # Volatility
            fig.add_trace(
                go.Scatter(x=ts_data.index, y=ts_data['Volatility'],
                          mode='lines', name='Rolling Volatility',
                          line=dict(color=self.colors['info'])),
                row=3, col=1
            )
            
            fig.update_layout(
                title_text="Advanced Time Series Analysis",
                height=900,
                showlegend=True
            )
            
            return fig, ts_data
            
        except Exception as e:
            print(f"Time series analysis error: {e}")
            return None, None

# Initialize analytics engine
analytics_engine = AdvancedAnalyticsEngine()
