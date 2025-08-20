"""
Advanced Forecasting Models
==========================

Comprehensive forecasting module with multiple algorithms:
- ARIMA/SARIMA (statsmodels)
- Prophet (Facebook Prophet)
- Gradient Boosting (XGBoost, LightGBM)
- LSTM Neural Networks (TensorFlow/Keras)
- Ensemble Methods with meta-learning
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# Time series libraries
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

# Advanced ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class BaseForecaster(ABC, BaseEstimator, RegressorMixin):
    """Base class for all forecasting models"""
    
    def __init__(self, forecast_horizon: int = 12, confidence_level: float = 0.95):
        self.forecast_horizon = forecast_horizon
        self.confidence_level = confidence_level
        self.is_fitted = False
        self.feature_names = None
        self.scaler = None
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseForecaster':
        """Fit the forecasting model"""
        pass
    
    @abstractmethod
    def predict(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Generate point forecasts"""
        pass
    
    def predict_with_intervals(self, X: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecasts with confidence intervals"""
        forecasts = self.predict(X)
        # Default: use historical residuals for interval estimation
        if hasattr(self, 'residuals_'):
            residual_std = np.std(self.residuals_)
            from scipy.stats import norm
            z_score = norm.ppf((1 + self.confidence_level) / 2)
            margin = z_score * residual_std
            lower = forecasts - margin
            upper = forecasts + margin
        else:
            # Fallback: 10% interval
            margin = 0.1 * np.abs(forecasts)
            lower = forecasts - margin
            upper = forecasts + margin
            
        return forecasts, lower, upper
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (handling zero values)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1e-8))) * 100
        
        # Directional accuracy
        if len(y_true) > 1:
            true_directions = np.diff(y_true) > 0
            pred_directions = np.diff(y_pred) > 0
            directional_accuracy = np.mean(true_directions == pred_directions) * 100
        else:
            directional_accuracy = 0
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy
        }

class ARIMAForecaster(BaseForecaster):
    """ARIMA/SARIMA forecasting model with automatic order selection"""
    
    def __init__(self, order: Optional[Tuple[int, int, int]] = None, 
                 seasonal_order: Optional[Tuple[int, int, int, int]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        
    def _auto_arima_order(self, y: pd.Series) -> Tuple[int, int, int]:
        """Automatic ARIMA order selection using AIC"""
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        for p in range(0, 4):
            for d in range(0, 3):
                for q in range(0, 4):
                    try:
                        model = ARIMA(y, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        return best_order
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ARIMAForecaster':
        """Fit ARIMA model"""
        self.y_train = y.copy()
        
        # Auto-select order if not provided
        if self.order is None:
            self.order = self._auto_arima_order(y)
        
        # Fit model
        if self.seasonal_order is not None:
            self.model = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order)
        else:
            self.model = ARIMA(y, order=self.order)
        
        self.fitted_model = self.model.fit()
        
        # Calculate residuals
        self.residuals_ = self.fitted_model.resid
        
        self.is_fitted = True
        return self
    
    def predict(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Generate ARIMA forecasts"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        forecast = self.fitted_model.forecast(steps=self.forecast_horizon)
        return np.array(forecast)
    
    def predict_with_intervals(self, X: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecasts with confidence intervals"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        forecast_result = self.fitted_model.get_forecast(steps=self.forecast_horizon)
        forecasts = forecast_result.predicted_mean.values
        conf_int = forecast_result.conf_int(alpha=1-self.confidence_level)
        lower = conf_int.iloc[:, 0].values
        upper = conf_int.iloc[:, 1].values
        
        return forecasts, lower, upper

class ProphetForecaster(BaseForecaster):
    """Facebook Prophet forecasting model"""
    
    def __init__(self, yearly_seasonality: bool = True, weekly_seasonality: bool = False,
                 daily_seasonality: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.model = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ProphetForecaster':
        """Fit Prophet model"""
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not available. Install with: pip install prophet")
        
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': y.index,
            'y': y.values
        })
        
        # Initialize and fit Prophet model
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            confidence_interval=self.confidence_level
        )
        
        self.model.fit(df)
        
        # Calculate residuals for metrics
        in_sample_forecast = self.model.predict(df)
        self.residuals_ = y.values - in_sample_forecast['yhat'].values
        
        self.is_fitted = True
        return self
    
    def predict(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Generate Prophet forecasts"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=self.forecast_horizon, freq='M')
        forecast = self.model.predict(future)
        
        # Return only the forecasted values (last forecast_horizon periods)
        return forecast['yhat'].tail(self.forecast_horizon).values
    
    def predict_with_intervals(self, X: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecasts with confidence intervals"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        future = self.model.make_future_dataframe(periods=self.forecast_horizon, freq='M')
        forecast = self.model.predict(future)
        
        # Extract forecasted values and intervals
        forecasts = forecast['yhat'].tail(self.forecast_horizon).values
        lower = forecast['yhat_lower'].tail(self.forecast_horizon).values
        upper = forecast['yhat_upper'].tail(self.forecast_horizon).values
        
        return forecasts, lower, upper

class XGBoostForecaster(BaseForecaster):
    """XGBoost-based time series forecasting with feature engineering"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, 
                 learning_rate: float = 0.1, lag_features: int = 12, **kwargs):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.lag_features = lag_features
        self.model = None
        
    def _create_lag_features(self, y: pd.Series) -> pd.DataFrame:
        """Create lagged features for time series"""
        df = pd.DataFrame(index=y.index)
        
        # Lag features
        for lag in range(1, self.lag_features + 1):
            df[f'lag_{lag}'] = y.shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12]:
            if len(y) > window:
                df[f'rolling_mean_{window}'] = y.shift(1).rolling(window).mean()
                df[f'rolling_std_{window}'] = y.shift(1).rolling(window).std()
        
        # Trend and seasonality features
        df['trend'] = np.arange(len(y))
        df['month'] = y.index.month if hasattr(y.index, 'month') else 1
        df['quarter'] = y.index.quarter if hasattr(y.index, 'quarter') else 1
        
        return df.dropna()
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'XGBoostForecaster':
        """Fit XGBoost model"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Install with: pip install xgboost")
        
        # Create features
        features_df = self._create_lag_features(y)
        
        # Align target with features
        aligned_y = y.loc[features_df.index]
        
        # Initialize and fit model
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42
        )
        
        self.model.fit(features_df, aligned_y)
        
        # Store for prediction
        self.last_values = y.tail(self.lag_features).values
        self.last_index = y.index[-1]
        self.feature_columns = features_df.columns.tolist()
        
        # Calculate residuals
        predictions = self.model.predict(features_df)
        self.residuals_ = aligned_y.values - predictions
        
        self.is_fitted = True
        return self
    
    def predict(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Generate XGBoost forecasts"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        forecasts = []
        current_values = self.last_values.copy()
        
        for step in range(self.forecast_horizon):
            # Create features for current step
            features = []
            
            # Lag features
            for lag in range(1, self.lag_features + 1):
                if lag <= len(current_values):
                    features.append(current_values[-lag])
                else:
                    features.append(0)  # Padding
            
            # Rolling statistics (simplified)
            if len(current_values) >= 3:
                features.append(np.mean(current_values[-3:]))  # rolling_mean_3
                features.append(np.std(current_values[-3:]))   # rolling_std_3
            else:
                features.extend([0, 0])
            
            if len(current_values) >= 6:
                features.append(np.mean(current_values[-6:]))  # rolling_mean_6
                features.append(np.std(current_values[-6:]))   # rolling_std_6
            else:
                features.extend([0, 0])
            
            if len(current_values) >= 12:
                features.append(np.mean(current_values[-12:]))  # rolling_mean_12
                features.append(np.std(current_values[-12:]))   # rolling_std_12
            else:
                features.extend([0, 0])
            
            # Trend and seasonality
            features.append(step)  # trend
            features.append(((step + 1) % 12) + 1)  # month
            features.append(((step + 1) // 3) % 4 + 1)  # quarter
            
            # Ensure we have the right number of features
            while len(features) < len(self.feature_columns):
                features.append(0)
            features = features[:len(self.feature_columns)]
            
            # Predict
            pred = self.model.predict([features])[0]
            forecasts.append(pred)
            
            # Update current values
            current_values = np.append(current_values, pred)
        
        return np.array(forecasts)

class LSTMForecaster(BaseForecaster):
    """LSTM neural network for time series forecasting"""
    
    def __init__(self, sequence_length: int = 12, hidden_units: int = 50,
                 dropout_rate: float = 0.2, epochs: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.model = None
        self.scaler = MinMaxScaler()
        
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LSTMForecaster':
        """Fit LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available. Install with: pip install tensorflow")
        
        # Prepare data
        y_values = y.values.reshape(-1, 1)
        y_scaled = self.scaler.fit_transform(y_values)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(y_scaled.flatten())
        
        if len(X_seq) == 0:
            raise ValueError(f"Not enough data points. Need at least {self.sequence_length + 1} points.")
        
        # Reshape for LSTM
        X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))
        
        # Build model
        self.model = Sequential([
            Bidirectional(LSTM(self.hidden_units, return_sequences=True, 
                              input_shape=(self.sequence_length, 1))),
            Dropout(self.dropout_rate),
            Bidirectional(LSTM(self.hidden_units // 2, return_sequences=False)),
            Dropout(self.dropout_rate),
            Dense(25),
            Dense(1)
        ])
        
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Train model
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        
        self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=32,
            verbose=0,
            callbacks=[early_stopping]
        )
        
        # Store last sequence for prediction
        self.last_sequence = y_scaled[-self.sequence_length:].flatten()
        
        # Calculate residuals
        predictions_scaled = self.model.predict(X_seq, verbose=0)
        predictions = self.scaler.inverse_transform(predictions_scaled).flatten()
        actual = self.scaler.inverse_transform(y_seq.reshape(-1, 1)).flatten()
        self.residuals_ = actual - predictions
        
        self.is_fitted = True
        return self
    
    def predict(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Generate LSTM forecasts"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        forecasts = []
        current_sequence = self.last_sequence.copy()
        
        for _ in range(self.forecast_horizon):
            # Reshape for prediction
            X_pred = current_sequence.reshape(1, self.sequence_length, 1)
            
            # Predict next value
            pred_scaled = self.model.predict(X_pred, verbose=0)[0, 0]
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], pred_scaled)
            
            # Store prediction
            forecasts.append(pred_scaled)
        
        # Inverse transform
        forecasts_array = np.array(forecasts).reshape(-1, 1)
        forecasts_original = self.scaler.inverse_transform(forecasts_array).flatten()
        
        return forecasts_original

class EnsembleForecaster(BaseForecaster):
    """Ensemble forecaster combining multiple models with meta-learning"""
    
    def __init__(self, models: Optional[List[BaseForecaster]] = None, 
                 ensemble_method: str = 'weighted_average', **kwargs):
        super().__init__(**kwargs)
        self.ensemble_method = ensemble_method
        self.models = models or self._get_default_models()
        self.weights = None
        self.meta_model = None
        
    def _get_default_models(self) -> List[BaseForecaster]:
        """Get default ensemble of models"""
        models = [
            ARIMAForecaster(forecast_horizon=self.forecast_horizon, 
                           confidence_level=self.confidence_level),
            XGBoostForecaster(forecast_horizon=self.forecast_horizon,
                             confidence_level=self.confidence_level) if XGBOOST_AVAILABLE else None,
            ProphetForecaster(forecast_horizon=self.forecast_horizon,
                             confidence_level=self.confidence_level) if PROPHET_AVAILABLE else None
        ]
        return [m for m in models if m is not None]
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnsembleForecaster':
        """Fit ensemble of models"""
        self.individual_predictions = []
        self.individual_scores = []
        
        # Use time series cross-validation for ensemble weights
        tscv = TimeSeriesSplit(n_splits=3)
        
        for model in self.models:
            model_predictions = []
            model_scores = []
            
            for train_idx, val_idx in tscv.split(y):
                y_train = y.iloc[train_idx]
                y_val = y.iloc[val_idx]
                
                try:
                    # Fit model on training data
                    model_copy = type(model)(**model.get_params())
                    model_copy.fit(X.iloc[train_idx] if X is not None else None, y_train)
                    
                    # Predict on validation data (simplified for CV)
                    # For proper ensemble, we'd need more sophisticated prediction
                    pred = model_copy.predict()[:len(y_val)]
                    if len(pred) != len(y_val):
                        pred = np.resize(pred, len(y_val))
                    
                    model_predictions.append(pred)
                    
                    # Calculate score
                    score = r2_score(y_val, pred)
                    model_scores.append(score)
                    
                except Exception as e:
                    # If model fails, assign poor score
                    model_scores.append(-1.0)
                    model_predictions.append(np.zeros(len(y_val)))
            
            self.individual_predictions.append(model_predictions)
            avg_score = np.mean(model_scores)
            self.individual_scores.append(max(avg_score, 0))  # Ensure non-negative
        
        # Calculate ensemble weights based on performance
        total_score = sum(self.individual_scores) + 1e-8  # Avoid division by zero
        self.weights = [score / total_score for score in self.individual_scores]
        
        # Fit all models on full data
        self.fitted_models = []
        for model in self.models:
            try:
                fitted_model = model.fit(X, y)
                self.fitted_models.append(fitted_model)
            except Exception as e:
                print(f"Warning: Model {type(model).__name__} failed to fit: {e}")
                self.fitted_models.append(None)
        
        # Fit meta-model if using stacking
        if self.ensemble_method == 'stacking':
            self._fit_meta_model(X, y)
        
        self.is_fitted = True
        return self
    
    def _fit_meta_model(self, X: pd.DataFrame, y: pd.Series):
        """Fit meta-model for stacking ensemble"""
        # Generate meta-features using cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        meta_features = []
        meta_targets = []
        
        for train_idx, val_idx in tscv.split(y):
            fold_features = []
            
            for i, model in enumerate(self.models):
                if self.fitted_models[i] is not None:
                    try:
                        # Use simplified prediction for meta-features
                        pred = self.fitted_models[i].predict()[:len(val_idx)]
                        if len(pred) != len(val_idx):
                            pred = np.resize(pred, len(val_idx))
                        fold_features.append(pred)
                    except:
                        fold_features.append(np.zeros(len(val_idx)))
                else:
                    fold_features.append(np.zeros(len(val_idx)))
            
            if fold_features:
                meta_features.extend(np.array(fold_features).T)
                meta_targets.extend(y.iloc[val_idx].values)
        
        if meta_features:
            self.meta_model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.meta_model.fit(meta_features, meta_targets)
    
    def predict(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Generate ensemble forecasts"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        
        for i, model in enumerate(self.fitted_models):
            if model is not None:
                try:
                    pred = model.predict(X)
                    predictions.append(pred)
                except Exception as e:
                    # Fallback: use zeros
                    predictions.append(np.zeros(self.forecast_horizon))
            else:
                predictions.append(np.zeros(self.forecast_horizon))
        
        if not predictions:
            return np.zeros(self.forecast_horizon)
        
        # Ensure all predictions have the same length
        min_length = min(len(pred) for pred in predictions)
        predictions = [pred[:min_length] for pred in predictions]
        
        if self.ensemble_method == 'simple_average':
            return np.mean(predictions, axis=0)
        
        elif self.ensemble_method == 'weighted_average':
            weighted_pred = np.zeros(min_length)
            for i, pred in enumerate(predictions):
                weighted_pred += self.weights[i] * pred
            return weighted_pred
        
        elif self.ensemble_method == 'stacking' and self.meta_model is not None:
            # Use meta-model for final prediction
            meta_features = np.array(predictions).T
            return self.meta_model.predict(meta_features)
        
        else:
            # Fallback to simple average
            return np.mean(predictions, axis=0)
    
    def predict_with_intervals(self, X: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate ensemble forecasts with confidence intervals"""
        # Get individual model predictions with intervals
        all_forecasts = []
        all_lowers = []
        all_uppers = []
        
        for model in self.fitted_models:
            if model is not None:
                try:
                    forecast, lower, upper = model.predict_with_intervals(X)
                    all_forecasts.append(forecast)
                    all_lowers.append(lower)
                    all_uppers.append(upper)
                except:
                    # Fallback
                    fallback = np.zeros(self.forecast_horizon)
                    all_forecasts.append(fallback)
                    all_lowers.append(fallback)
                    all_uppers.append(fallback)
        
        if not all_forecasts:
            fallback = np.zeros(self.forecast_horizon)
            return fallback, fallback, fallback
        
        # Ensure consistent lengths
        min_length = min(len(f) for f in all_forecasts)
        all_forecasts = [f[:min_length] for f in all_forecasts]
        all_lowers = [l[:min_length] for l in all_lowers]
        all_uppers = [u[:min_length] for u in all_uppers]
        
        # Ensemble the forecasts
        ensemble_forecast = self.predict(X)[:min_length]
        
        # Ensemble the intervals (conservative approach)
        ensemble_lower = np.mean(all_lowers, axis=0)
        ensemble_upper = np.mean(all_uppers, axis=0)
        
        return ensemble_forecast, ensemble_lower, ensemble_upper

def create_forecasting_pipeline(data: pd.DataFrame, target_column: str,
                               model_type: str = 'ensemble',
                               forecast_horizon: int = 12,
                               **kwargs) -> BaseForecaster:
    """Factory function to create forecasting pipeline"""
    
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    y = data[target_column].dropna()
    X = data.drop(columns=[target_column]) if len(data.columns) > 1 else None
    
    if model_type == 'arima':
        model = ARIMAForecaster(forecast_horizon=forecast_horizon, **kwargs)
    elif model_type == 'prophet':
        model = ProphetForecaster(forecast_horizon=forecast_horizon, **kwargs)
    elif model_type == 'xgboost':
        model = XGBoostForecaster(forecast_horizon=forecast_horizon, **kwargs)
    elif model_type == 'lstm':
        model = LSTMForecaster(forecast_horizon=forecast_horizon, **kwargs)
    elif model_type == 'ensemble':
        model = EnsembleForecaster(forecast_horizon=forecast_horizon, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.fit(X, y)
