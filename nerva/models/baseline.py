"""
NERVA Baseline Forecasting Models
GODMODE_X: Fast, robust baseline ensemble
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

import logging
try:
    from ..config.settings import config
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config.settings import config

logger = logging.getLogger(__name__)

class BaselineForecaster:
    """Multi-model ensemble for economic forecasting"""
    
    def __init__(self, target_column: str = None):
        self.target_column = target_column
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
        # Model configurations
        self.horizons = config.model.forecast_horizons
        self.train_size = config.model.train_test_split
    
    def prepare_features(self, df: pd.DataFrame, date_column: str = None) -> pd.DataFrame:
        """Feature engineering for time series forecasting"""
        
        df_features = df.copy()
        
        # Auto-detect date column if not provided
        if date_column is None:
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                date_column = date_cols[0]
        
        if date_column and date_column in df.columns:
            df_features[date_column] = pd.to_datetime(df_features[date_column])
            df_features = df_features.set_index(date_column).sort_index()
            
            # Time-based features
            df_features['year'] = df_features.index.year
            df_features['month'] = df_features.index.month
            df_features['quarter'] = df_features.index.quarter
            df_features['day_of_year'] = df_features.index.dayofyear
        
        # Technical indicators for numeric columns
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col != self.target_column:
                # Lagged features
                for lag in [1, 3, 6, 12]:
                    df_features[f'{col}_lag_{lag}'] = df_features[col].shift(lag)
                
                # Rolling statistics
                for window in [3, 6, 12]:
                    df_features[f'{col}_ma_{window}'] = df_features[col].rolling(window).mean()
                    df_features[f'{col}_std_{window}'] = df_features[col].rolling(window).std()
                
                # Rate of change
                df_features[f'{col}_pct_change'] = df_features[col].pct_change()
                df_features[f'{col}_diff'] = df_features[col].diff()
        
        # Target variable lags (for autoregressive features)
        if self.target_column and self.target_column in df_features.columns:
            for lag in [1, 2, 3, 6, 12]:
                df_features[f'target_lag_{lag}'] = df_features[self.target_column].shift(lag)
        
        return df_features.dropna()
    
    def train_ensemble(self, df: pd.DataFrame, target_column: str) -> Dict[str, any]:
        """Train ensemble of baseline models"""
        
        self.target_column = target_column
        logger.info(f"🚀 Training baseline ensemble for target: {target_column}")
        
        # Prepare features
        df_features = self.prepare_features(df)
        
        if target_column not in df_features.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Split data
        split_idx = int(len(df_features) * self.train_size)
        train_data = df_features.iloc[:split_idx]
        test_data = df_features.iloc[split_idx:]
        
        # Prepare training data
        feature_cols = [col for col in df_features.columns 
                       if col != target_column and not col.startswith('_')]
        
        X_train = train_data[feature_cols]
        y_train = train_data[target_column]
        X_test = test_data[feature_cols]
        y_test = test_data[target_column]
        
        # Handle missing values
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())  # Use train median for test
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[target_column] = scaler
        
        # Train multiple models
        models_to_train = {
            'lightgbm': self._train_lightgbm,
            'random_forest': self._train_random_forest,
            'linear': self._train_linear,
            'arima': self._train_arima,
            'ets': self._train_ets
        }
        
        model_predictions = {}
        model_scores = {}
        
        for model_name, train_func in models_to_train.items():
            try:
                logger.info(f"Training {model_name}...")
                
                if model_name in ['arima', 'ets']:
                    # Time series models use target only
                    model, predictions = train_func(y_train, y_test)
                else:
                    # ML models use features
                    model, predictions = train_func(
                        X_train_scaled if model_name == 'linear' else X_train,
                        y_train, 
                        X_test_scaled if model_name == 'linear' else X_test
                    )
                
                self.models[f"{target_column}_{model_name}"] = model
                model_predictions[model_name] = predictions
                
                # Calculate performance metrics
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                mae = mean_absolute_error(y_test, predictions)
                mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
                
                model_scores[model_name] = {
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape
                }
                
                logger.info(f"✅ {model_name}: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")
                
            except Exception as e:
                logger.warning(f"❌ Failed to train {model_name}: {str(e)}")
                continue
        
        # Ensemble prediction (simple average)
        if model_predictions:
            ensemble_pred = np.mean(list(model_predictions.values()), axis=0)
            ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
            ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
            ensemble_mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
            
            model_scores['ensemble'] = {
                'rmse': ensemble_rmse,
                'mae': ensemble_mae,
                'mape': ensemble_mape
            }
            
            logger.info(f"🎯 Ensemble: RMSE={ensemble_rmse:.4f}, MAE={ensemble_mae:.4f}, MAPE={ensemble_mape:.2f}%")
        
        # Store performance metrics
        self.performance_metrics[target_column] = model_scores
        
        return {
            'models': self.models,
            'performance': model_scores,
            'feature_columns': feature_cols,
            'test_data': test_data,
            'predictions': model_predictions
        }
    
    def _train_lightgbm(self, X_train, y_train, X_test):
        """Train LightGBM model"""
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[train_data])
        
        predictions = model.predict(X_test, num_iteration=model.best_iteration)
        
        # Store feature importance
        importance = model.feature_importance(importance_type='gain')
        self.feature_importance[f"{self.target_column}_lightgbm"] = dict(zip(X_train.columns, importance))
        
        return model, predictions
    
    def _train_random_forest(self, X_train, y_train, X_test):
        """Train Random Forest model"""
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Store feature importance
        importance = model.feature_importances_
        self.feature_importance[f"{self.target_column}_random_forest"] = dict(zip(X_train.columns, importance))
        
        return model, predictions
    
    def _train_linear(self, X_train, y_train, X_test):
        """Train Linear Regression model"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        return model, predictions
    
    def _train_arima(self, y_train, y_test):
        """Train ARIMA model"""
        try:
            # Auto ARIMA order selection (simple version)
            model = ARIMA(y_train, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Forecast
            forecast = fitted_model.forecast(steps=len(y_test))
            
            return fitted_model, forecast
        except:
            # Fallback to simple model
            model = ARIMA(y_train, order=(1, 0, 0))
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=len(y_test))
            return fitted_model, forecast
    
    def _train_ets(self, y_train, y_test):
        """Train Exponential Smoothing model"""
        try:
            model = ETSModel(y_train, error='add', trend='add', seasonal='add', seasonal_periods=12)
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=len(y_test))
            return fitted_model, forecast
        except:
            # Fallback to simple exponential smoothing
            model = ETSModel(y_train, error='add', trend=None, seasonal=None)
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=len(y_test))
            return fitted_model, forecast
    
    def predict(self, df: pd.DataFrame, horizon: int = 1) -> Dict[str, float]:
        """Make predictions using ensemble"""
        
        if not self.models:
            raise ValueError("No trained models available. Train models first.")
        
        # Prepare features
        df_features = self.prepare_features(df)
        feature_cols = [col for col in df_features.columns 
                       if col != self.target_column and not col.startswith('_')]
        
        X = df_features[feature_cols].tail(1)  # Latest data point
        X = X.fillna(X.median())
        
        predictions = {}
        
        # Get predictions from each model
        for model_key, model in self.models.items():
            try:
                target_col, model_name = model_key.rsplit('_', 1)
                
                if model_name in ['arima', 'ets']:
                    # Time series models
                    pred = model.forecast(steps=horizon)
                    predictions[model_name] = pred[-1] if hasattr(pred, '__len__') else pred
                else:
                    # ML models
                    if model_name == 'linear' and target_col in self.scalers:
                        X_scaled = self.scalers[target_col].transform(X)
                        pred = model.predict(X_scaled)[0]
                    else:
                        pred = model.predict(X)[0]
                    
                    predictions[model_name] = pred
                    
            except Exception as e:
                logger.warning(f"Prediction failed for {model_key}: {str(e)}")
                continue
        
        # Ensemble prediction
        if predictions:
            ensemble_pred = np.mean(list(predictions.values()))
            predictions['ensemble'] = ensemble_pred
        
        return predictions
    
    def get_model_performance(self) -> pd.DataFrame:
        """Get performance summary of all models"""
        
        performance_data = []
        
        for target, metrics in self.performance_metrics.items():
            for model_name, scores in metrics.items():
                performance_data.append({
                    'target': target,
                    'model': model_name,
                    'rmse': scores['rmse'],
                    'mae': scores['mae'],
                    'mape': scores['mape']
                })
        
        return pd.DataFrame(performance_data)

# Quick access functions
def train_baseline_forecaster(df: pd.DataFrame, target_column: str) -> BaselineForecaster:
    """Train baseline forecasting models"""
    forecaster = BaselineForecaster(target_column)
    forecaster.train_ensemble(df, target_column)
    return forecaster

def quick_forecast(df: pd.DataFrame, target_column: str, horizon: int = 1) -> Dict[str, float]:
    """Quick forecast using baseline ensemble"""
    forecaster = train_baseline_forecaster(df, target_column)
    return forecaster.predict(df, horizon)
