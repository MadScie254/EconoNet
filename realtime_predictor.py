"""
NERVA Real-Time Economic Prediction Engine
High-frequency prediction system for live economic forecasting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

warnings.filterwarnings('ignore')

class RealtimePredictionEngine:
    """Real-time economic prediction engine for NERVA system"""
    
    def __init__(self, data_path="data/raw/"):
        self.data_path = data_path
        self.models = {}
        self.scalers = {}
        self.last_predictions = {}
        self.prediction_history = {}
        self.data_cache = {}
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize prediction models for different economic indicators"""
        
        model_configs = {
            'fx_rate': {
                'model': RandomForestRegressor(n_estimators=50, random_state=42),
                'scaler': StandardScaler(),
                'features': ['rate_lag1', 'rate_lag2', 'rate_lag3', 'volatility', 'momentum'],
                'horizon': 1  # 1 month ahead
            },
            'inflation': {
                'model': GradientBoostingRegressor(n_estimators=50, random_state=42),
                'scaler': StandardScaler(),
                'features': ['cbr_rate', 'cbr_change', 'cbr_momentum', 'policy_stance'],
                'horizon': 3  # 3 months ahead
            },
            'liquidity': {
                'model': LinearRegression(),
                'scaler': StandardScaler(),
                'features': ['interbank_rate', 'rate_volatility', 'volume_trend'],
                'horizon': 1  # 1 month ahead
            }
        }
        
        for name, config in model_configs.items():
            self.models[name] = config['model']
            self.scalers[name] = config['scaler']
            self.prediction_history[name] = []
            
    def load_realtime_data(self):
        """Load latest economic data for real-time predictions"""
        try:
            # Load FX data
            fx_data = pd.read_csv(f"{self.data_path}Monthly exchange rate (end period).csv")
            fx_processed = self._process_fx_data(fx_data)
            self.data_cache['fx'] = fx_processed
            
            # Load CBR data
            cbr_data = pd.read_csv(f"{self.data_path}Central Bank Rate (CBR)  .csv")
            cbr_processed = self._process_cbr_data(cbr_data)
            self.data_cache['cbr'] = cbr_processed
            
            # Load interbank data
            try:
                interbank_data = pd.read_csv(f"{self.data_path}Interbank Rates  Volumes.csv")
                interbank_processed = self._process_interbank_data(interbank_data)
                self.data_cache['interbank'] = interbank_processed
            except:
                self.data_cache['interbank'] = None
            
            return True
            
        except Exception as e:
            print(f"Error loading realtime data: {str(e)}")
            return False
    
    def _process_fx_data(self, fx_data):
        """Process FX data for real-time prediction"""
        try:
            # Skip header and process
            df = fx_data.iloc[1:].copy()
            df.columns = ['Year', 'Month', 'USD_KES'] + list(df.columns[3:])
            
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
            df['USD_KES'] = pd.to_numeric(df['USD_KES'], errors='coerce')
            
            df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1), errors='coerce')
            df = df.dropna(subset=['Date', 'USD_KES']).sort_values('Date')
            
            # Create features
            df['rate_lag1'] = df['USD_KES'].shift(1)
            df['rate_lag2'] = df['USD_KES'].shift(2)
            df['rate_lag3'] = df['USD_KES'].shift(3)
            df['volatility'] = df['USD_KES'].rolling(6).std()
            df['momentum'] = df['USD_KES'].diff(3)
            
            return df.dropna()
            
        except Exception as e:
            print(f"Error processing FX data: {str(e)}")
            return None
    
    def _process_cbr_data(self, cbr_data):
        """Process CBR data for real-time prediction"""
        try:
            # Process CBR data structure
            df = cbr_data.copy()
            
            # Find appropriate columns
            date_cols = [col for col in df.columns if any(word in col.lower() for word in ['date', 'period', 'month'])]
            rate_cols = [col for col in df.columns if any(word in col.lower() for word in ['rate', 'cbr'])]
            
            if date_cols and rate_cols:
                df_clean = df[[date_cols[0], rate_cols[0]]].copy()
                df_clean.columns = ['date', 'cbr_rate']
                
                df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
                df_clean['cbr_rate'] = pd.to_numeric(df_clean['cbr_rate'], errors='coerce')
                df_clean = df_clean.dropna().sort_values('date')
                
                # Create features
                df_clean['cbr_change'] = df_clean['cbr_rate'].diff()
                df_clean['cbr_momentum'] = df_clean['cbr_rate'].diff(3)
                df_clean['policy_stance'] = (df_clean['cbr_rate'] > df_clean['cbr_rate'].rolling(6).mean()).astype(int)
                
                return df_clean.dropna()
            
            return None
            
        except Exception as e:
            print(f"Error processing CBR data: {str(e)}")
            return None
    
    def _process_interbank_data(self, interbank_data):
        """Process interbank data for real-time prediction"""
        try:
            # Basic processing for interbank data
            df = interbank_data.copy()
            
            # This would need to be customized based on actual data structure
            if len(df.columns) >= 3:
                df.columns = ['date', 'interbank_rate', 'volume'] + list(df.columns[3:])
                
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df['interbank_rate'] = pd.to_numeric(df['interbank_rate'], errors='coerce')
                df = df.dropna().sort_values('date')
                
                # Create features
                df['rate_volatility'] = df['interbank_rate'].rolling(5).std()
                df['volume_trend'] = df['volume'].rolling(5).mean() if 'volume' in df.columns else 0
                
                return df.dropna()
            
            return None
            
        except Exception as e:
            print(f"Error processing interbank data: {str(e)}")
            return None
    
    def train_models(self):
        """Train all prediction models with latest data"""
        if not self.data_cache:
            if not self.load_realtime_data():
                return False
        
        success_count = 0
        
        # Train FX model
        if self.data_cache.get('fx') is not None:
            if self._train_fx_model():
                success_count += 1
        
        # Train inflation model (CBR)
        if self.data_cache.get('cbr') is not None:
            if self._train_inflation_model():
                success_count += 1
        
        # Train liquidity model
        if self.data_cache.get('interbank') is not None:
            if self._train_liquidity_model():
                success_count += 1
        
        return success_count > 0
    
    def _train_fx_model(self):
        """Train FX prediction model"""
        try:
            df = self.data_cache['fx']
            
            # Prepare features and target
            features = ['rate_lag1', 'rate_lag2', 'rate_lag3', 'volatility', 'momentum']
            target = 'USD_KES'
            
            # Create future target
            df['future_rate'] = df[target].shift(-1)
            df_clean = df.dropna()
            
            if len(df_clean) < 10:
                return False
            
            X = df_clean[features]
            y = df_clean['future_rate']
            
            # Split for validation
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Scale and train
            X_train_scaled = self.scalers['fx_rate'].fit_transform(X_train)
            X_test_scaled = self.scalers['fx_rate'].transform(X_test)
            
            self.models['fx_rate'].fit(X_train_scaled, y_train)
            
            # Store model performance
            y_pred = self.models['fx_rate'].predict(X_test_scaled)
            score = r2_score(y_test, y_pred)
            
            print(f"FX Model R2 Score: {score:.4f}")
            return True
            
        except Exception as e:
            print(f"Error training FX model: {str(e)}")
            return False
    
    def _train_inflation_model(self):
        """Train inflation prediction model"""
        try:
            df = self.data_cache['cbr']
            
            # Prepare features and target
            features = ['cbr_change', 'cbr_momentum', 'policy_stance']
            target = 'cbr_rate'
            
            # Create future target
            df['future_cbr'] = df[target].shift(-3)  # 3 months ahead
            df_clean = df.dropna()
            
            if len(df_clean) < 10:
                return False
            
            X = df_clean[features]
            y = df_clean['future_cbr']
            
            # Scale and train
            X_scaled = self.scalers['inflation'].fit_transform(X)
            self.models['inflation'].fit(X_scaled, y)
            
            print("Inflation model trained successfully")
            return True
            
        except Exception as e:
            print(f"Error training inflation model: {str(e)}")
            return False
    
    def _train_liquidity_model(self):
        """Train liquidity prediction model"""
        try:
            df = self.data_cache['interbank']
            
            if df is None or len(df) < 10:
                return False
            
            # Prepare features and target
            features = ['rate_volatility']
            if 'volume_trend' in df.columns:
                features.append('volume_trend')
            
            target = 'interbank_rate'
            
            # Create future target
            df['future_rate'] = df[target].shift(-1)
            df_clean = df.dropna()
            
            X = df_clean[features]
            y = df_clean['future_rate']
            
            # Scale and train
            X_scaled = self.scalers['liquidity'].fit_transform(X)
            self.models['liquidity'].fit(X_scaled, y)
            
            print("Liquidity model trained successfully")
            return True
            
        except Exception as e:
            print(f"Error training liquidity model: {str(e)}")
            return False
    
    def generate_realtime_predictions(self):
        """Generate real-time predictions for all economic indicators"""
        predictions = {}
        timestamp = datetime.now()
        
        # FX predictions
        if 'fx' in self.data_cache and self.data_cache['fx'] is not None:
            fx_pred = self._predict_fx_rate()
            if fx_pred is not None:
                predictions['fx_rate'] = {
                    'value': fx_pred,
                    'timestamp': timestamp,
                    'confidence': 'High' if hasattr(self.models['fx_rate'], 'feature_importances_') else 'Medium'
                }
        
        # Inflation predictions
        if 'cbr' in self.data_cache and self.data_cache['cbr'] is not None:
            inflation_pred = self._predict_inflation()
            if inflation_pred is not None:
                predictions['inflation'] = {
                    'value': inflation_pred,
                    'timestamp': timestamp,
                    'confidence': 'High'
                }
        
        # Liquidity predictions
        if 'interbank' in self.data_cache and self.data_cache['interbank'] is not None:
            liquidity_pred = self._predict_liquidity()
            if liquidity_pred is not None:
                predictions['liquidity'] = {
                    'value': liquidity_pred,
                    'timestamp': timestamp,
                    'confidence': 'Medium'
                }
        
        # Store predictions
        self.last_predictions = predictions
        
        # Add to history
        for key, pred in predictions.items():
            self.prediction_history[key].append({
                'timestamp': timestamp,
                'prediction': pred['value'],
                'confidence': pred['confidence']
            })
            
            # Keep only last 100 predictions
            if len(self.prediction_history[key]) > 100:
                self.prediction_history[key] = self.prediction_history[key][-100:]
        
        return predictions
    
    def _predict_fx_rate(self):
        """Predict next FX rate"""
        try:
            df = self.data_cache['fx']
            latest = df.iloc[-1]
            
            features = [latest['rate_lag1'], latest['rate_lag2'], latest['rate_lag3'], 
                       latest['volatility'], latest['momentum']]
            
            features_scaled = self.scalers['fx_rate'].transform([features])
            prediction = self.models['fx_rate'].predict(features_scaled)[0]
            
            return round(prediction, 2)
            
        except Exception as e:
            print(f"Error predicting FX rate: {str(e)}")
            return None
    
    def _predict_inflation(self):
        """Predict inflation trend (CBR)"""
        try:
            df = self.data_cache['cbr']
            latest = df.iloc[-1]
            
            features = [latest['cbr_change'], latest['cbr_momentum'], latest['policy_stance']]
            
            features_scaled = self.scalers['inflation'].transform([features])
            prediction = self.models['inflation'].predict(features_scaled)[0]
            
            return round(prediction, 2)
            
        except Exception as e:
            print(f"Error predicting inflation: {str(e)}")
            return None
    
    def _predict_liquidity(self):
        """Predict liquidity conditions"""
        try:
            df = self.data_cache['interbank']
            latest = df.iloc[-1]
            
            features = [latest['rate_volatility']]
            if 'volume_trend' in latest:
                features.append(latest['volume_trend'])
            
            features_scaled = self.scalers['liquidity'].transform([features])
            prediction = self.models['liquidity'].predict(features_scaled)[0]
            
            return round(prediction, 2)
            
        except Exception as e:
            print(f"Error predicting liquidity: {str(e)}")
            return None
    
    def create_prediction_dashboard(self):
        """Create real-time prediction dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['FX Rate Predictions', 'Inflation Forecast', 
                           'Liquidity Conditions', 'Prediction Confidence'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # FX Rate predictions
        if 'fx_rate' in self.prediction_history and self.prediction_history['fx_rate']:
            fx_history = self.prediction_history['fx_rate']
            timestamps = [p['timestamp'] for p in fx_history]
            predictions = [p['prediction'] for p in fx_history]
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=predictions, mode='lines+markers',
                          name='FX Predictions', line=dict(color='blue')),
                row=1, col=1
            )
        
        # Inflation predictions
        if 'inflation' in self.prediction_history and self.prediction_history['inflation']:
            inf_history = self.prediction_history['inflation']
            timestamps = [p['timestamp'] for p in inf_history]
            predictions = [p['prediction'] for p in inf_history]
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=predictions, mode='lines+markers',
                          name='Inflation Forecast', line=dict(color='red')),
                row=1, col=2
            )
        
        # Liquidity predictions
        if 'liquidity' in self.prediction_history and self.prediction_history['liquidity']:
            liq_history = self.prediction_history['liquidity']
            timestamps = [p['timestamp'] for p in liq_history]
            predictions = [p['prediction'] for p in liq_history]
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=predictions, mode='lines+markers',
                          name='Liquidity Conditions', line=dict(color='green')),
                row=2, col=1
            )
        
        # Confidence indicators
        if self.last_predictions:
            indicators = list(self.last_predictions.keys())
            confidences = [1 if self.last_predictions[ind]['confidence'] == 'High' else 0.7 if self.last_predictions[ind]['confidence'] == 'Medium' else 0.4 for ind in indicators]
            
            fig.add_trace(
                go.Bar(x=indicators, y=confidences, name='Confidence Levels',
                      marker_color=['green' if c == 1 else 'orange' if c == 0.7 else 'red' for c in confidences]),
                row=2, col=2
            )
        
        fig.update_layout(
            title="NERVA Real-Time Economic Predictions",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def get_prediction_summary(self):
        """Get summary of latest predictions"""
        if not self.last_predictions:
            return "No predictions available"
        
        summary = "REAL-TIME ECONOMIC PREDICTIONS:\n"
        summary += "=" * 40 + "\n"
        
        for indicator, pred in self.last_predictions.items():
            summary += f"{indicator.upper()}: {pred['value']} (Confidence: {pred['confidence']})\n"
            summary += f"Timestamp: {pred['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return summary

# Global prediction engine instance
prediction_engine = RealtimePredictionEngine()
