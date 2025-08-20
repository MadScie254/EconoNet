"""
Base Debt Predictor Model
========================

Refactored base class for all debt prediction models with proper inheritance,
type hints, and comprehensive documentation.
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# Time series libraries
import statsmodels.api as sm
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

class BaseDebtPredictor(ABC, BaseEstimator, RegressorMixin):
    """
    Abstract base class for all debt prediction models.
    
    Provides common functionality for data loading, preprocessing,
    feature engineering, and model evaluation.
    """
    
    def __init__(self, 
                 target_column: str = 'Total_Public_Debt',
                 forecast_horizon: int = 12,
                 confidence_level: float = 0.95,
                 random_state: int = 42):
        """
        Initialize base debt predictor.
        
        Parameters:
        -----------
        target_column : str
            Name of the target variable column
        forecast_horizon : int
            Number of periods to forecast
        confidence_level : float
            Confidence level for prediction intervals
        random_state : int
            Random seed for reproducibility
        """
        self.target_column = target_column
        self.forecast_horizon = forecast_horizon
        self.confidence_level = confidence_level
        self.random_state = random_state
        
        # Model state
        self.is_fitted = False
        self.models = {}
        self.scaler = None
        self.feature_names = None
        
        # Data containers
        self.debt_ts = None
        self.feature_data = None
        self.model_results = {}
        self.feature_importance = []
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseDebtPredictor':
        """Fit the debt prediction model"""
        pass
    
    @abstractmethod
    def predict(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Generate debt forecasts"""
        pass
    
    def load_debt_data(self, data_path: str) -> pd.DataFrame:
        """
        Load debt data from various sources.
        
        Parameters:
        -----------
        data_path : str
            Path to data directory or file
            
        Returns:
        --------
        pd.DataFrame
            Loaded debt data
        """
        import os
        import glob
        
        debt_files = []
        
        if os.path.isdir(data_path):
            # Look for debt-related files
            patterns = [
                '*debt*.csv', '*Debt*.csv',
                '*public*.csv', '*Public*.csv',
                '*treasury*.csv', '*Treasury*.csv'
            ]
            
            for pattern in patterns:
                debt_files.extend(glob.glob(os.path.join(data_path, pattern)))
        else:
            debt_files = [data_path]
        
        if not debt_files:
            raise FileNotFoundError(f"No debt data files found in {data_path}")
        
        # Load and combine data
        all_data = []
        
        for file_path in debt_files:
            try:
                df = pd.read_csv(file_path)
                df['source_file'] = os.path.basename(file_path)
                all_data.append(df)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
        
        if not all_data:
            raise ValueError("No data could be loaded successfully")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True, sort=False)
        
        return self._clean_debt_data(combined_data)
    
    def _clean_debt_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize debt data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw debt data
            
        Returns:
        --------
        pd.DataFrame
            Cleaned debt data
        """
        # Make a copy to avoid modifying original
        df = data.copy()
        
        # Standardize column names
        df.columns = df.columns.str.strip()
        
        # Handle common column name variations
        column_mapping = {
            'Total Public Debt': 'Total_Public_Debt',
            'Total_Debt': 'Total_Public_Debt',
            'Public_Debt': 'Total_Public_Debt',
            'Domestic Debt': 'Domestic_Debt',
            'External Debt': 'External_Debt',
            'Date': 'date',
            'Period': 'date',
            'Year': 'year',
            'Month': 'month'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Convert numeric columns
        numeric_columns = df.select_dtypes(include=['object']).columns
        for col in numeric_columns:
            if col not in ['date', 'source_file']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle date column
        date_columns = ['date', 'Date', 'Period', 'TIME_PERIOD']
        date_col = None
        
        for col in date_columns:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.sort_values(date_col)
            df = df.set_index(date_col)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Forward fill missing values for time series continuity
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def prepare_debt_time_series(self) -> pd.DataFrame:
        """
        Prepare time series data for modeling.
        
        Returns:
        --------
        pd.DataFrame
            Prepared time series data
        """
        if self.debt_ts is None:
            raise ValueError("No debt data loaded. Call load_debt_data() first.")
        
        # Ensure we have the target column
        if self.target_column not in self.debt_ts.columns:
            # Try to find the best target column
            debt_cols = [col for col in self.debt_ts.columns 
                        if 'debt' in col.lower() or 'total' in col.lower()]
            
            if debt_cols:
                self.target_column = debt_cols[0]
                print(f"Using {self.target_column} as target column")
            else:
                raise ValueError(f"Target column {self.target_column} not found")
        
        # Create time series features
        ts_data = self.debt_ts.copy()
        
        # Ensure numeric target
        ts_data[self.target_column] = pd.to_numeric(ts_data[self.target_column], errors='coerce')
        
        # Add time-based features
        if hasattr(ts_data.index, 'year'):
            ts_data['year'] = ts_data.index.year
            ts_data['month'] = ts_data.index.month
            ts_data['quarter'] = ts_data.index.quarter
        
        # Add derived debt features
        if len(ts_data) > 1:
            ts_data['debt_growth'] = ts_data[self.target_column].pct_change() * 100
            ts_data['debt_diff'] = ts_data[self.target_column].diff()
            
            # Rolling statistics
            for window in [3, 6, 12]:
                if len(ts_data) > window:
                    ts_data[f'debt_ma_{window}'] = ts_data[self.target_column].rolling(window).mean()
                    ts_data[f'debt_std_{window}'] = ts_data[self.target_column].rolling(window).std()
        
        # Calculate debt ratios if we have other debt components
        debt_components = ['Domestic_Debt', 'External_Debt']
        available_components = [col for col in debt_components if col in ts_data.columns]
        
        if len(available_components) >= 2:
            total_component_debt = ts_data[available_components].sum(axis=1)
            for component in available_components:
                ts_data[f'{component.lower()}_share'] = (ts_data[component] / total_component_debt) * 100
        
        self.debt_ts = ts_data
        return ts_data
    
    def create_features(self, external_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Create comprehensive feature set for debt modeling.
        
        Parameters:
        -----------
        external_data : Dict[str, pd.DataFrame], optional
            External economic data sources
            
        Returns:
        --------
        pd.DataFrame
            Feature matrix
        """
        if self.debt_ts is None:
            raise ValueError("No prepared time series data. Call prepare_debt_time_series() first.")
        
        # Start with time series features
        features_df = self.debt_ts.copy()
        
        # Lag features
        for lag in range(1, min(13, len(features_df))):
            features_df[f'{self.target_column}_lag_{lag}'] = features_df[self.target_column].shift(lag)
        
        # Seasonal decomposition features
        if len(features_df) >= 24:  # Need at least 2 years for seasonal decomposition
            try:
                decomposition = seasonal_decompose(
                    features_df[self.target_column].dropna(), 
                    model='additive', 
                    period=12
                )
                features_df['debt_trend'] = decomposition.trend
                features_df['debt_seasonal'] = decomposition.seasonal
                features_df['debt_residual'] = decomposition.resid
            except:
                pass  # Skip if decomposition fails
        
        # External economic indicators
        if external_data:
            for name, data in external_data.items():
                # Align dates and merge
                aligned_data = data.reindex(features_df.index, method='nearest')
                
                # Add prefix to avoid column conflicts
                aligned_data.columns = [f'{name}_{col}' for col in aligned_data.columns]
                features_df = pd.concat([features_df, aligned_data], axis=1)
        
        # Feature engineering
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        
        # Interaction features (debt with economic indicators)
        external_cols = [col for col in numeric_cols if any(prefix in col for prefix in ['gdp', 'inflation', 'rate'])]
        
        for ext_col in external_cols[:5]:  # Limit to top 5 to avoid explosion
            if ext_col in features_df.columns:
                features_df[f'debt_{ext_col}_ratio'] = features_df[self.target_column] / (features_df[ext_col] + 1e-8)
                features_df[f'debt_{ext_col}_product'] = features_df[self.target_column] * features_df[ext_col]
        
        # Polynomial features for main debt series
        features_df[f'{self.target_column}_squared'] = features_df[self.target_column] ** 2
        features_df[f'{self.target_column}_sqrt'] = np.sqrt(np.abs(features_df[self.target_column]))
        
        # Technical indicators
        if len(features_df) > 20:
            # Simple moving average convergence divergence (MACD-like)
            ema_12 = features_df[self.target_column].ewm(span=12).mean()
            ema_26 = features_df[self.target_column].ewm(span=26).mean()
            features_df['debt_macd'] = ema_12 - ema_26
            
            # Bollinger bands
            ma_20 = features_df[self.target_column].rolling(20).mean()
            std_20 = features_df[self.target_column].rolling(20).std()
            features_df['debt_bb_upper'] = ma_20 + (2 * std_20)
            features_df['debt_bb_lower'] = ma_20 - (2 * std_20)
            features_df['debt_bb_position'] = (features_df[self.target_column] - features_df['debt_bb_lower']) / (features_df['debt_bb_upper'] - features_df['debt_bb_lower'])
        
        # Clean features
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
        # Store feature data
        self.feature_data = features_df
        self.feature_names = features_df.columns.tolist()
        
        return features_df
    
    def prepare_training_data(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for model training.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature matrix
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series]
            Features (X) and target (y)
        """
        # Remove rows with missing target
        clean_data = features_df.dropna(subset=[self.target_column])
        
        if len(clean_data) == 0:
            raise ValueError("No valid data points after cleaning")
        
        # Separate features and target
        y = clean_data[self.target_column]
        X = clean_data.drop(columns=[self.target_column])
        
        # Remove features with too many missing values
        missing_threshold = 0.5
        X = X.loc[:, X.isnull().mean() < missing_threshold]
        
        # Fill remaining missing values
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Remove constant features
        constant_features = X.columns[X.var() == 0]
        X = X.drop(columns=constant_features)
        
        return X, y
    
    def evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                      cv_folds: int = 3) -> Dict[str, float]:
        """
        Evaluate model performance using time series cross-validation.
        
        Parameters:
        -----------
        model : sklearn estimator
            Fitted model to evaluate
        X : pd.DataFrame
            Features
        y : pd.Series
            Target values
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        Dict[str, float]
            Evaluation metrics
        """
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
        
        # Full dataset predictions for additional metrics
        y_pred = model.predict(X)
        
        # Calculate comprehensive metrics
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        # MAPE (handling zero values)
        mape = np.mean(np.abs((y - y_pred) / np.where(y != 0, y, 1e-8))) * 100
        
        # Directional accuracy
        if len(y) > 1:
            y_diff = np.diff(y)
            pred_diff = np.diff(y_pred)
            directional_accuracy = np.mean((y_diff > 0) == (pred_diff > 0)) * 100
        else:
            directional_accuracy = 0
        
        return {
            'CV_R2_Mean': cv_scores.mean(),
            'CV_R2_Std': cv_scores.std(),
            'Test_R2': r2,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy
        }
    
    def get_feature_importance(self, model: Any, feature_names: List[str]) -> List[Tuple[str, float]]:
        """
        Extract feature importance from fitted model.
        
        Parameters:
        -----------
        model : sklearn estimator
            Fitted model
        feature_names : List[str]
            Names of features
            
        Returns:
        --------
        List[Tuple[str, float]]
            Feature importance pairs sorted by importance
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            # Fallback: use random importances
            importances = np.random.random(len(feature_names))
        
        # Ensure we have the right number of importances
        if len(importances) != len(feature_names):
            min_len = min(len(importances), len(feature_names))
            importances = importances[:min_len]
            feature_names = feature_names[:min_len]
        
        # Create and sort feature importance pairs
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importance
    
    def generate_predictions(self, X: Optional[pd.DataFrame] = None) -> Dict[str, Union[float, np.ndarray]]:
        """
        Generate predictions from all fitted models.
        
        Parameters:
        -----------
        X : pd.DataFrame, optional
            Feature matrix for prediction
            
        Returns:
        --------
        Dict[str, Union[float, np.ndarray]]
            Predictions from each model
        """
        if not self.models:
            raise ValueError("No models trained. Call fit() first.")
        
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                if X is not None:
                    pred = model.predict(X)
                else:
                    # Use last known values for forecast
                    if hasattr(self, 'X_train') and self.X_train is not None:
                        pred = model.predict(self.X_train.tail(1))
                    else:
                        pred = np.array([0])  # Fallback
                
                # Store prediction (scalar if single value, array if multiple)
                predictions[model_name] = pred[0] if len(pred) == 1 else pred
                
            except Exception as e:
                print(f"Warning: Prediction failed for {model_name}: {e}")
                predictions[model_name] = 0
        
        # Ensemble prediction (simple average)
        if len(predictions) > 1:
            pred_values = [pred for pred in predictions.values() if isinstance(pred, (int, float, np.number))]
            if pred_values:
                predictions['Ensemble'] = np.mean(pred_values)
        
        return predictions
    
    def save_model(self, filepath: str):
        """
        Save the fitted model to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        import pickle
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model state
        model_state = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_column': self.target_column,
            'forecast_horizon': self.forecast_horizon,
            'model_results': self.model_results,
            'feature_importance': self.feature_importance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)
    
    def load_model(self, filepath: str):
        """
        Load a fitted model from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)
        
        # Restore model state
        self.models = model_state['models']
        self.scaler = model_state.get('scaler')
        self.feature_names = model_state.get('feature_names')
        self.target_column = model_state.get('target_column', self.target_column)
        self.forecast_horizon = model_state.get('forecast_horizon', self.forecast_horizon)
        self.model_results = model_state.get('model_results', {})
        self.feature_importance = model_state.get('feature_importance', [])
        
        self.is_fitted = True

class EnhancedDebtPredictor(BaseDebtPredictor):
    """
    Enhanced debt predictor with multiple ML algorithms.
    
    Implements Random Forest, Gradient Boosting, and optional XGBoost/LightGBM
    with automated hyperparameter tuning and ensemble methods.
    """
    
    def __init__(self, 
                 ensemble_method: str = 'weighted_average',
                 n_estimators: int = 100,
                 **kwargs):
        """
        Initialize enhanced debt predictor.
        
        Parameters:
        -----------
        ensemble_method : str
            Method for combining model predictions ('simple_average', 'weighted_average')
        n_estimators : int
            Number of estimators for ensemble models
        **kwargs
            Additional arguments passed to base class
        """
        super().__init__(**kwargs)
        self.ensemble_method = ensemble_method
        self.n_estimators = n_estimators
        self.ensemble_weights = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnhancedDebtPredictor':
        """
        Fit enhanced debt prediction models.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
            
        Returns:
        --------
        EnhancedDebtPredictor
            Fitted model instance
        """
        # Store training data
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        # Scale features
        self.scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Initialize models
        models_config = {
            'Random_Forest': RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=10,
                min_samples_split=5,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient_Boosting': GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models_config['XGBoost'] = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models_config['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        
        # Train models
        for name, model in models_config.items():
            try:
                print(f"Training {name}...")
                model.fit(X_scaled, y)
                self.models[name] = model
                
                # Evaluate model
                metrics = self.evaluate_model(model, X_scaled, y)
                self.model_results[name] = metrics
                
                print(f"{name} - R²: {metrics['Test_R2']:.4f}, RMSE: {metrics['RMSE']:.2f}")
                
            except Exception as e:
                print(f"Warning: {name} training failed: {e}")
        
        # Calculate ensemble weights based on performance
        if len(self.models) > 1:
            self._calculate_ensemble_weights()
        
        # Get feature importance from best model
        if self.models:
            best_model_name = max(self.model_results.items(), key=lambda x: x[1]['Test_R2'])[0]
            best_model = self.models[best_model_name]
            self.feature_importance = self.get_feature_importance(best_model, X.columns.tolist())
        
        self.is_fitted = True
        return self
    
    def _calculate_ensemble_weights(self):
        """Calculate weights for ensemble based on model performance"""
        if self.ensemble_method == 'simple_average':
            n_models = len(self.models)
            self.ensemble_weights = {name: 1/n_models for name in self.models.keys()}
        
        elif self.ensemble_method == 'weighted_average':
            # Weight by R² score
            r2_scores = {name: results['Test_R2'] for name, results in self.model_results.items()}
            total_r2 = sum(max(0, score) for score in r2_scores.values())  # Ensure non-negative
            
            if total_r2 > 0:
                self.ensemble_weights = {name: max(0, score)/total_r2 for name, score in r2_scores.items()}
            else:
                # Fallback to equal weights
                n_models = len(self.models)
                self.ensemble_weights = {name: 1/n_models for name in self.models.keys()}
    
    def predict(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate enhanced debt forecasts.
        
        Parameters:
        -----------
        X : pd.DataFrame, optional
            Feature matrix for prediction
            
        Returns:
        --------
        np.ndarray
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if X is None:
            # Use last training data for forecast
            X = self.X_train.tail(1)
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Get predictions from all models
        model_predictions = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)
                model_predictions[name] = pred
            except Exception as e:
                print(f"Warning: Prediction failed for {name}: {e}")
        
        if not model_predictions:
            return np.array([0])
        
        # Ensemble predictions
        if len(model_predictions) == 1:
            return list(model_predictions.values())[0]
        
        elif self.ensemble_method == 'simple_average':
            predictions_array = np.array(list(model_predictions.values()))
            return np.mean(predictions_array, axis=0)
        
        elif self.ensemble_method == 'weighted_average' and self.ensemble_weights:
            weighted_pred = np.zeros_like(list(model_predictions.values())[0])
            for name, pred in model_predictions.items():
                weight = self.ensemble_weights.get(name, 0)
                weighted_pred += weight * pred
            return weighted_pred
        
        else:
            # Fallback to simple average
            predictions_array = np.array(list(model_predictions.values()))
            return np.mean(predictions_array, axis=0)

class UltimateDebtPredictor(EnhancedDebtPredictor):
    """
    Ultimate debt predictor with advanced feature engineering,
    automated hyperparameter tuning, and sophisticated ensemble methods.
    """
    
    def __init__(self, 
                 auto_tune: bool = True,
                 advanced_features: bool = True,
                 **kwargs):
        """
        Initialize ultimate debt predictor.
        
        Parameters:
        -----------
        auto_tune : bool
            Whether to perform automated hyperparameter tuning
        advanced_features : bool
            Whether to create advanced feature engineering
        **kwargs
            Additional arguments passed to base class
        """
        super().__init__(**kwargs)
        self.auto_tune = auto_tune
        self.advanced_features = advanced_features
        self.tuned_models = {}
    
    def create_advanced_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced feature engineering.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Base feature matrix
            
        Returns:
        --------
        pd.DataFrame
            Enhanced feature matrix
        """
        if not self.advanced_features:
            return features_df
        
        enhanced_df = features_df.copy()
        
        # Fourier features for seasonality
        if len(enhanced_df) > 24:
            for period in [12, 6, 4, 3]:  # Annual, semi-annual, quarterly, monthly cycles
                for i in range(1, 3):  # First 2 harmonics
                    enhanced_df[f'fourier_cos_{period}_{i}'] = np.cos(2 * np.pi * i * enhanced_df.index.to_list() / period)
                    enhanced_df[f'fourier_sin_{period}_{i}'] = np.sin(2 * np.pi * i * enhanced_df.index.to_list() / period)
        
        # Advanced rolling statistics
        numeric_cols = enhanced_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:10]:  # Limit to avoid explosion
            if col in enhanced_df.columns:
                # Exponential moving averages
                enhanced_df[f'{col}_ema_3'] = enhanced_df[col].ewm(span=3).mean()
                enhanced_df[f'{col}_ema_12'] = enhanced_df[col].ewm(span=12).mean()
                
                # Rolling skewness and kurtosis
                if len(enhanced_df) > 12:
                    enhanced_df[f'{col}_skew_12'] = enhanced_df[col].rolling(12).skew()
                    enhanced_df[f'{col}_kurt_12'] = enhanced_df[col].rolling(12).kurt()
        
        # Principal component features (simplified)
        from sklearn.decomposition import PCA
        
        pca_data = enhanced_df.select_dtypes(include=[np.number]).fillna(0)
        if len(pca_data.columns) > 5:
            pca = PCA(n_components=min(5, len(pca_data.columns)))
            pca_features = pca.fit_transform(pca_data)
            
            for i in range(pca_features.shape[1]):
                enhanced_df[f'pca_component_{i}'] = pca_features[:, i]
        
        return enhanced_df
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'UltimateDebtPredictor':
        """
        Fit ultimate debt prediction models with advanced features and tuning.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
            
        Returns:
        --------
        UltimateDebtPredictor
            Fitted model instance
        """
        # Create advanced features
        X_enhanced = self.create_advanced_features(X)
        
        # Call parent fit method
        super().fit(X_enhanced, y)
        
        # Perform hyperparameter tuning if requested
        if self.auto_tune and len(X_enhanced) > 50:  # Only tune with sufficient data
            self._tune_hyperparameters(X_enhanced, y)
        
        return self
    
    def _tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series):
        """
        Perform automated hyperparameter tuning.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        """
        from sklearn.model_selection import RandomizedSearchCV
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Define parameter grids
        param_grids = {
            'Random_Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [6, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient_Boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
        if XGBOOST_AVAILABLE:
            param_grids['XGBoost'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        
        # Perform tuning for each model
        tscv = TimeSeriesSplit(n_splits=3)
        
        for model_name in self.models.keys():
            if model_name in param_grids:
                print(f"Tuning {model_name}...")
                
                try:
                    # Get base model
                    base_model = type(self.models[model_name])()
                    
                    # Random search
                    random_search = RandomizedSearchCV(
                        base_model,
                        param_grids[model_name],
                        n_iter=20,
                        cv=tscv,
                        scoring='r2',
                        random_state=self.random_state,
                        n_jobs=-1
                    )
                    
                    random_search.fit(X_scaled, y)
                    
                    # Store tuned model
                    self.tuned_models[model_name] = random_search.best_estimator_
                    
                    # Re-evaluate with tuned parameters
                    tuned_metrics = self.evaluate_model(random_search.best_estimator_, X_scaled, y)
                    self.model_results[f'{model_name}_Tuned'] = tuned_metrics
                    
                    print(f"{model_name} Tuned - R²: {tuned_metrics['Test_R2']:.4f}")
                    
                except Exception as e:
                    print(f"Warning: Tuning failed for {model_name}: {e}")
        
        # Update models dictionary with tuned versions if they perform better
        for name, tuned_model in self.tuned_models.items():
            original_score = self.model_results.get(name, {}).get('Test_R2', 0)
            tuned_score = self.model_results.get(f'{name}_Tuned', {}).get('Test_R2', 0)
            
            if tuned_score > original_score:
                self.models[name] = tuned_model
                self.model_results[name] = self.model_results[f'{name}_Tuned']
                print(f"Updated {name} with tuned version (R² improvement: {tuned_score - original_score:.4f})")
        
        # Recalculate ensemble weights with updated models
        if len(self.models) > 1:
            self._calculate_ensemble_weights()
