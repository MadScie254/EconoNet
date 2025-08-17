"""
Advanced Time Series Models
Sophisticated forecasting with uncertainty quantification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna

# Advanced Time Series Models
from arch import arch_model
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

import logging
from config.settings import config

logger = logging.getLogger(__name__)

class TransformerForecaster(nn.Module):
    """Transformer-based time series forecasting model"""
    
    def __init__(self, input_dim: int, d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 6, dropout: float = 0.1, seq_len: int = 60):
        super(TransformerForecaster, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with uncertainty estimation"""
        
        # Input projection and positional encoding
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)
        
        # Take the last timestamp representation
        last_encoded = encoded[:, -1, :]
        
        # Predictions and uncertainty
        predictions = self.output_projection(last_encoded)
        uncertainty = self.uncertainty_head(last_encoded)
        
        return predictions, uncertainty

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :].transpose(0, 1)
        return self.dropout(x)

class AdvancedForecaster:
    """Advanced forecasting system with multiple sophisticated models"""
    
    def __init__(self, target_column: str = None):
        self.target_column = target_column
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        self.uncertainty_estimates = {}
        
        # Model configurations
        self.seq_len = 60
        self.horizons = config.model.forecast_horizons
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prepare_advanced_features(self, df: pd.DataFrame, date_column: str = None) -> pd.DataFrame:
        """Advanced feature engineering with technical indicators"""
        
        df_features = df.copy()
        
        # Auto-detect date column
        if date_column is None:
            date_cols = [col for col in df.columns if any(
                word in col.lower() for word in ['date', 'time', 'period']
            )]
            if date_cols:
                date_column = date_cols[0]
        
        if date_column and date_column in df.columns:
            df_features[date_column] = pd.to_datetime(df_features[date_column])
            df_features = df_features.set_index(date_column).sort_index()
            
            # Advanced time features
            df_features['year'] = df_features.index.year
            df_features['month'] = df_features.index.month
            df_features['quarter'] = df_features.index.quarter
            df_features['day_of_year'] = df_features.index.dayofyear
            df_features['week_of_year'] = df_features.index.isocalendar().week
            df_features['is_month_end'] = df_features.index.is_month_end.astype(int)
            df_features['is_quarter_end'] = df_features.index.is_quarter_end.astype(int)
        
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col != self.target_column and not col.startswith('_'):
                # Multi-scale lagged features
                for lag in [1, 2, 3, 6, 12, 24]:
                    df_features[f'{col}_lag_{lag}'] = df_features[col].shift(lag)
                
                # Rolling statistics with multiple windows
                for window in [3, 6, 12, 24]:
                    df_features[f'{col}_ma_{window}'] = df_features[col].rolling(window).mean()
                    df_features[f'{col}_std_{window}'] = df_features[col].rolling(window).std()
                    df_features[f'{col}_min_{window}'] = df_features[col].rolling(window).min()
                    df_features[f'{col}_max_{window}'] = df_features[col].rolling(window).max()
                    df_features[f'{col}_median_{window}'] = df_features[col].rolling(window).median()
                
                # Technical indicators
                df_features[f'{col}_rsi'] = self._calculate_rsi(df_features[col])
                df_features[f'{col}_macd'] = self._calculate_macd(df_features[col])
                df_features[f'{col}_bollinger_upper'], df_features[f'{col}_bollinger_lower'] = \
                    self._calculate_bollinger_bands(df_features[col])
                
                # Momentum indicators
                df_features[f'{col}_momentum_3'] = df_features[col] / df_features[col].shift(3) - 1
                df_features[f'{col}_momentum_12'] = df_features[col] / df_features[col].shift(12) - 1
                
                # Volatility measures
                df_features[f'{col}_volatility_12'] = df_features[col].rolling(12).std()
                df_features[f'{col}_cv_12'] = (df_features[f'{col}_volatility_12'] / 
                                             df_features[f'{col}_ma_12'])
        
        # Cross-sectional features
        numeric_cols_clean = [col for col in numeric_cols 
                             if col != self.target_column and not col.startswith('_')]
        
        if len(numeric_cols_clean) > 1:
            # Principal components
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_features[numeric_cols_clean].fillna(0))
            
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(5, len(numeric_cols_clean)))
            pca_features = pca.fit_transform(scaled_data)
            
            for i in range(pca_features.shape[1]):
                df_features[f'pca_component_{i+1}'] = pca_features[:, i]
        
        return df_features.dropna()
    
    def _calculate_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD"""
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def _calculate_bollinger_bands(self, series: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        ma = series.rolling(window).mean()
        std = series.rolling(window).std()
        upper_band = ma + (std * num_std)
        lower_band = ma - (std * num_std)
        return upper_band, lower_band
    
    def train_transformer_model(self, df: pd.DataFrame, target_column: str) -> Dict:
        """Train Transformer-based forecasting model"""
        
        logger.info("Training Transformer forecasting model...")
        
        df_features = self.prepare_advanced_features(df, None)
        
        if target_column not in df_features.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Prepare features
        feature_cols = [col for col in df_features.columns 
                       if col != target_column and not col.startswith('_')]
        
        # Scale data
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        
        X_scaled = feature_scaler.fit_transform(df_features[feature_cols].fillna(0))
        y_scaled = target_scaler.fit_transform(df_features[[target_column]].fillna(0))
        
        self.scalers[f"{target_column}_feature"] = feature_scaler
        self.scalers[f"{target_column}_target"] = target_scaler
        
        # Create sequences for training
        X_sequences, y_sequences = self._create_sequences(X_scaled, y_scaled.flatten(), self.seq_len)
        
        # Split data
        split_idx = int(len(X_sequences) * 0.8)
        X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
        y_train, y_test = y_sequences[:split_idx], y_sequences[split_idx:]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        # Initialize model
        model = TransformerForecaster(
            input_dim=X_train.shape[2],
            d_model=config.model.hidden_dim,
            nhead=config.model.num_heads,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            seq_len=self.seq_len
        ).to(self.device)
        
        # Training setup
        optimizer = optim.AdamW(model.parameters(), lr=config.model.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        model.train()
        train_losses = []
        
        for epoch in range(50):  # Reduced for demo
            optimizer.zero_grad()
            
            predictions, uncertainty = model(X_train_tensor)
            
            # Loss combining prediction error and uncertainty calibration
            pred_loss = nn.MSELoss()(predictions.squeeze(), y_train_tensor)
            uncertainty_loss = torch.mean(uncertainty)  # Regularize uncertainty
            
            total_loss = pred_loss + 0.1 * uncertainty_loss
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(total_loss.item())
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/50, Loss: {total_loss.item():.6f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_predictions, test_uncertainty = model(X_test_tensor)
            test_loss = nn.MSELoss()(test_predictions.squeeze(), y_test_tensor)
        
        # Store model
        self.models[f"{target_column}_transformer"] = {
            'model': model,
            'feature_columns': feature_cols,
            'train_loss': train_losses[-1],
            'test_loss': test_loss.item()
        }
        
        return {
            'model': model,
            'train_losses': train_losses,
            'test_loss': test_loss.item(),
            'feature_columns': feature_cols
        }
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series modeling"""
        X_sequences, y_sequences = [], []
        
        for i in range(len(X) - seq_len):
            X_sequences.append(X[i:i+seq_len])
            y_sequences.append(y[i+seq_len])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train_var_model(self, df: pd.DataFrame, target_columns: List[str]) -> Dict:
        """Train Vector Autoregression model for multivariate forecasting"""
        
        logger.info("Training VAR model for multivariate forecasting...")
        
        # Prepare data
        df_clean = df[target_columns].dropna()
        
        # Fit VAR model
        model = VAR(df_clean)
        
        # Optimal lag selection
        lag_results = model.select_order(maxlags=12)
        optimal_lag = lag_results.aic
        
        # Fit model with optimal lag
        fitted_model = model.fit(optimal_lag)
        
        # Store model
        self.models[f"var_{'_'.join(target_columns)}"] = {
            'model': fitted_model,
            'target_columns': target_columns,
            'optimal_lag': optimal_lag
        }
        
        return {
            'model': fitted_model,
            'optimal_lag': optimal_lag,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic
        }
    
    def train_garch_model(self, df: pd.DataFrame, target_column: str) -> Dict:
        """Train GARCH model for volatility forecasting"""
        
        logger.info("Training GARCH model for volatility forecasting...")
        
        # Calculate returns
        returns = df[target_column].pct_change().dropna() * 100
        
        # Fit GARCH(1,1) model
        garch_model = arch_model(returns, vol='Garch', p=1, q=1)
        fitted_model = garch_model.fit(disp='off')
        
        # Store model
        self.models[f"{target_column}_garch"] = {
            'model': fitted_model,
            'returns_series': returns
        }
        
        return {
            'model': fitted_model,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'log_likelihood': fitted_model.loglikelihood
        }
    
    def predict_with_uncertainty(self, df: pd.DataFrame, target_column: str, 
                                horizon: int = 1) -> Dict[str, Union[float, Tuple[float, float]]]:
        """Make predictions with uncertainty quantification"""
        
        predictions = {}
        
        # Transformer prediction with uncertainty
        transformer_key = f"{target_column}_transformer"
        if transformer_key in self.models:
            model_info = self.models[transformer_key]
            model = model_info['model']
            feature_cols = model_info['feature_columns']
            
            # Prepare input
            df_features = self.prepare_advanced_features(df)
            X = df_features[feature_cols].tail(self.seq_len).fillna(0)
            
            # Scale input
            feature_scaler = self.scalers[f"{target_column}_feature"]
            target_scaler = self.scalers[f"{target_column}_target"]
            
            X_scaled = feature_scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device)
            
            # Predict
            model.eval()
            with torch.no_grad():
                pred, uncertainty = model(X_tensor)
                
                # Scale back
                pred_scaled = target_scaler.inverse_transform(pred.cpu().numpy())
                uncertainty_scaled = uncertainty.cpu().numpy().flatten()[0]
                
                # Confidence interval (approximate)
                pred_value = pred_scaled.flatten()[0]
                ci_lower = pred_value - 1.96 * uncertainty_scaled
                ci_upper = pred_value + 1.96 * uncertainty_scaled
                
                predictions['transformer'] = {
                    'prediction': pred_value,
                    'uncertainty': uncertainty_scaled,
                    'confidence_interval': (ci_lower, ci_upper)
                }
        
        # GARCH volatility prediction
        garch_key = f"{target_column}_garch"
        if garch_key in self.models:
            model_info = self.models[garch_key]
            fitted_model = model_info['model']
            
            # Forecast volatility
            vol_forecast = fitted_model.forecast(horizon=horizon)
            vol_mean = vol_forecast.variance.iloc[-1, 0]
            
            predictions['garch_volatility'] = {
                'prediction': vol_mean,
                'forecast_type': 'volatility'
            }
        
        return predictions

# Quick access functions
def train_advanced_forecaster(df: pd.DataFrame, target_column: str) -> AdvancedForecaster:
    """Train advanced forecasting models"""
    forecaster = AdvancedForecaster(target_column)
    
    # Train multiple advanced models
    forecaster.train_transformer_model(df, target_column)
    forecaster.train_garch_model(df, target_column)
    
    return forecaster
