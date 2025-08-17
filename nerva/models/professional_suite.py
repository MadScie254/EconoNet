"""
NERVA Advanced Model Suite
Professional machine learning models for economic forecasting
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import warnings
warnings.filterwarnings('ignore')

class ProfessionalEnsembleForecaster:
    """Professional ensemble forecasting system"""
    
    def __init__(self, models=None):
        if models is None:
            self.models = {
                'Random_Forest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'Gradient_Boosting': GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                ),
                'Linear_Regression': LinearRegression(),
                'Ridge_Regression': Ridge(alpha=1.0),
                'Lasso_Regression': Lasso(alpha=0.1)
            }
        else:
            self.models = models
            
        self.scalers = {}
        self.trained_models = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
    def prepare_features(self, data, target_col, feature_cols=None):
        """Prepare features for modeling"""
        if feature_cols is None:
            feature_cols = [col for col in data.select_dtypes(include=[np.number]).columns 
                          if col != target_col]
        
        # Clean data
        clean_data = data[feature_cols + [target_col]].dropna()
        
        X = clean_data[feature_cols]
        y = clean_data[target_col]
        
        return X, y, feature_cols
    
    def train_ensemble(self, X, y, test_size=0.2):
        """Train ensemble of models"""
        
        # Train-test split (time series aware)
        split_idx = int((1 - test_size) * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train each model
        for name, model in self.models.items():
            try:
                # Scale data for linear models
                if 'Regression' in name:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    self.scalers[name] = scaler
                    
                    model.fit(X_train_scaled, y_train)
                    train_pred = model.predict(X_train_scaled)
                    test_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    train_pred = model.predict(X_train)
                    test_pred = model.predict(X_test)
                
                # Store trained model
                self.trained_models[name] = model
                
                # Calculate metrics
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                test_r2 = r2_score(y_test, test_pred)
                test_mae = mean_absolute_error(y_test, test_pred)
                
                self.performance_metrics[name] = {
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'test_r2': test_r2,
                    'test_mae': test_mae,
                    'predictions': test_pred,
                    'actual': y_test
                }
                
                # Feature importance (for tree-based models)
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    self.feature_importance[name] = np.abs(model.coef_)
                    
                print(f"{name}: Test RMSE={test_rmse:.4f}, R²={test_r2:.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                
        return self.performance_metrics
    
    def generate_ensemble_forecast(self, X, method='weighted_average'):
        """Generate ensemble forecast"""
        predictions = {}
        
        for name, model in self.trained_models.items():
            try:
                if name in self.scalers:
                    X_scaled = self.scalers[name].transform(X)
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X)
                
                predictions[name] = pred
                
            except Exception as e:
                print(f"Error generating prediction for {name}: {e}")
        
        if not predictions:
            return None
            
        # Combine predictions
        if method == 'simple_average':
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
        elif method == 'weighted_average':
            # Weight by inverse of test RMSE
            weights = []
            pred_values = []
            
            for name, pred in predictions.items():
                if name in self.performance_metrics:
                    rmse = self.performance_metrics[name]['test_rmse']
                    weight = 1 / (rmse + 1e-8)  # Add small value to avoid division by zero
                    weights.append(weight)
                    pred_values.append(pred)
            
            if weights:
                weights = np.array(weights) / np.sum(weights)  # Normalize
                ensemble_pred = np.average(pred_values, axis=0, weights=weights)
            else:
                ensemble_pred = np.mean(list(predictions.values()), axis=0)
        
        return ensemble_pred, predictions
    
    def cross_validate_models(self, X, y, cv_folds=5):
        """Perform cross-validation on all models"""
        cv_results = {}
        
        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        for name, model in self.models.items():
            try:
                if 'Regression' in name:
                    # For linear models, need to scale within each fold
                    scores = []
                    for train_idx, val_idx in tscv.split(X):
                        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                        
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train_fold)
                        X_val_scaled = scaler.transform(X_val_fold)
                        
                        model.fit(X_train_scaled, y_train_fold)
                        pred = model.predict(X_val_scaled)
                        score = r2_score(y_val_fold, pred)
                        scores.append(score)
                    
                    cv_scores = np.array(scores)
                else:
                    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
                
                cv_results[name] = {
                    'mean_score': cv_scores.mean(),
                    'std_score': cv_scores.std(),
                    'scores': cv_scores
                }
                
                print(f"{name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                print(f"Error in cross-validation for {name}: {e}")
        
        return cv_results

class AdvancedTimeSeriesAnalyzer:
    """Advanced time series analysis and decomposition"""
    
    def __init__(self):
        self.trend_model = None
        self.seasonal_model = None
        self.residual_stats = {}
        
    def decompose_series(self, data, period=12):
        """Decompose time series into trend, seasonal, and residual components"""
        try:
            from scipy import signal
            from sklearn.linear_model import LinearRegression
            
            # Ensure we have enough data
            if len(data) < period * 2:
                print("Insufficient data for seasonal decomposition")
                return None
            
            # Create time index
            time_idx = np.arange(len(data))
            
            # 1. Trend extraction using moving average
            trend = pd.Series(data).rolling(window=period, center=True).mean()
            
            # 2. Detrend the series
            detrended = data - trend
            
            # 3. Extract seasonal component
            seasonal_pattern = np.zeros(period)
            for i in range(period):
                seasonal_values = detrended[i::period].dropna()
                if len(seasonal_values) > 0:
                    seasonal_pattern[i] = seasonal_values.mean()
            
            # Repeat seasonal pattern
            seasonal = np.tile(seasonal_pattern, len(data) // period + 1)[:len(data)]
            
            # 4. Calculate residuals
            residual = data - trend - seasonal
            
            return {
                'original': data,
                'trend': trend,
                'seasonal': seasonal,
                'residual': residual,
                'seasonal_pattern': seasonal_pattern
            }
            
        except Exception as e:
            print(f"Error in time series decomposition: {e}")
            return None
    
    def detect_anomalies(self, data, method='zscore', threshold=3):
        """Detect anomalies in time series data"""
        try:
            if method == 'zscore':
                z_scores = np.abs((data - data.mean()) / data.std())
                anomalies = z_scores > threshold
                
            elif method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                anomalies = (data < lower_bound) | (data > upper_bound)
                
            elif method == 'isolation_forest':
                from sklearn.ensemble import IsolationForest
                
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                anomalies = iso_forest.fit_predict(data.values.reshape(-1, 1)) == -1
                
            return anomalies, data[anomalies]
            
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            return None, None
    
    def calculate_volatility_metrics(self, data, window=30):
        """Calculate various volatility metrics"""
        try:
            # Returns
            returns = data.pct_change().dropna()
            
            # Rolling volatility
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
            
            # GARCH-like volatility (simplified)
            squared_returns = returns ** 2
            ewm_vol = squared_returns.ewm(span=window).mean() ** 0.5 * np.sqrt(252)
            
            # Volatility clustering measure
            vol_clustering = returns.rolling(window=window).std().std()
            
            return {
                'returns': returns,
                'rolling_volatility': rolling_vol,
                'ewm_volatility': ewm_vol,
                'volatility_clustering': vol_clustering,
                'current_vol': rolling_vol.iloc[-1] if len(rolling_vol) > 0 else None
            }
            
        except Exception as e:
            print(f"Error calculating volatility metrics: {e}")
            return None

# Model registry for easy access
MODEL_REGISTRY = {
    'ensemble_forecaster': ProfessionalEnsembleForecaster,
    'time_series_analyzer': AdvancedTimeSeriesAnalyzer
}

def get_model(model_name, **kwargs):
    """Get model instance from registry"""
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name](**kwargs)
    else:
        raise ValueError(f"Model {model_name} not found in registry")

def list_available_models():
    """List all available models"""
    return list(MODEL_REGISTRY.keys())
