"""
Foreign Exchange (FX) Model - Advanced Currency Analysis
======================================================

Comprehensive FX modeling with machine learning, technical analysis,
and economic fundamentals for KES/USD and other currency pairs.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf

warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """Technical analysis indicators for FX modeling"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()
        
        return {
            'middle': sma,
            'upper': sma + (std * num_std),
            'lower': sma - (std * num_std)
        }
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

class EconomicFactors:
    """Economic fundamentals affecting FX rates"""
    
    def __init__(self):
        self.factors = {
            'interest_rate_differential': 0.0,
            'inflation_differential': 0.0,
            'gdp_growth_differential': 0.0,
            'current_account_balance': 0.0,
            'foreign_reserves': 0.0,
            'political_stability_index': 0.0,
            'commodity_prices': 0.0,
            'risk_sentiment': 0.0
        }
    
    def calculate_purchasing_power_parity(self, domestic_inflation: float, 
                                        foreign_inflation: float, 
                                        initial_rate: float) -> float:
        """Calculate PPP-adjusted exchange rate"""
        inflation_differential = domestic_inflation - foreign_inflation
        return initial_rate * (1 + inflation_differential / 100)
    
    def calculate_interest_rate_parity(self, domestic_rate: float, 
                                     foreign_rate: float, 
                                     spot_rate: float, 
                                     time_period: float = 1.0) -> float:
        """Calculate forward rate using interest rate parity"""
        rate_differential = (domestic_rate - foreign_rate) / 100
        return spot_rate * (1 + rate_differential * time_period)
    
    def get_economic_score(self) -> float:
        """Calculate composite economic strength score"""
        weights = {
            'interest_rate_differential': 0.2,
            'inflation_differential': -0.15,  # Lower inflation is better
            'gdp_growth_differential': 0.25,
            'current_account_balance': 0.15,
            'foreign_reserves': 0.1,
            'political_stability_index': 0.1,
            'commodity_prices': 0.05
        }
        
        score = sum(self.factors[factor] * weight for factor, weight in weights.items())
        return score

class VolatilityModel:
    """Volatility modeling for FX risk assessment"""
    
    def __init__(self):
        self.volatility_data = {}
    
    def garch_volatility(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """Simple GARCH-like volatility estimation"""
        # Simplified GARCH approximation
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        ewm_vol = returns.ewm(halflife=window//4).std() * np.sqrt(252)
        
        # Combine rolling and exponentially weighted volatility
        combined_vol = 0.7 * ewm_vol + 0.3 * rolling_vol
        return combined_vol
    
    def realized_volatility(self, prices: pd.Series, window: int = 21) -> pd.Series:
        """Calculate realized volatility"""
        returns = prices.pct_change().dropna()
        realized_vol = returns.rolling(window=window).std() * np.sqrt(252)
        return realized_vol
    
    def volatility_forecast(self, returns: pd.Series, horizon: int = 30) -> float:
        """Forecast volatility for given horizon"""
        current_vol = self.garch_volatility(returns).iloc[-1]
        long_term_vol = returns.std() * np.sqrt(252)
        
        # Mean reversion model
        decay_factor = np.exp(-0.1 * horizon / 252)  # Half-life of ~7 days
        forecast_vol = long_term_vol + decay_factor * (current_vol - long_term_vol)
        
        return forecast_vol

class FXPredictor:
    """Advanced FX prediction model combining multiple approaches"""
    
    def __init__(self, currency_pair: str = "KES/USD"):
        self.currency_pair = currency_pair
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.technical_indicators = TechnicalIndicators()
        self.economic_factors = EconomicFactors()
        self.volatility_model = VolatilityModel()
        
        # Model ensemble
        self.ensemble_models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression()
        }
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set for FX prediction"""
        
        features = data.copy()
        
        # Technical indicators
        if 'close' in features.columns:
            close_prices = features['close']
            
            # Moving averages
            features['sma_5'] = self.technical_indicators.sma(close_prices, 5)
            features['sma_10'] = self.technical_indicators.sma(close_prices, 10)
            features['sma_20'] = self.technical_indicators.sma(close_prices, 20)
            features['ema_12'] = self.technical_indicators.ema(close_prices, 12)
            features['ema_26'] = self.technical_indicators.ema(close_prices, 26)
            
            # Bollinger Bands
            bb = self.technical_indicators.bollinger_bands(close_prices)
            features['bb_upper'] = bb['upper']
            features['bb_lower'] = bb['lower']
            features['bb_width'] = bb['upper'] - bb['lower']
            features['bb_position'] = (close_prices - bb['lower']) / (bb['upper'] - bb['lower'])
            
            # RSI
            features['rsi'] = self.technical_indicators.rsi(close_prices)
            
            # MACD
            macd = self.technical_indicators.macd(close_prices)
            features['macd'] = macd['macd']
            features['macd_signal'] = macd['signal']
            features['macd_histogram'] = macd['histogram']
            
            # Price momentum features
            features['returns_1d'] = close_prices.pct_change(1)
            features['returns_5d'] = close_prices.pct_change(5)
            features['returns_20d'] = close_prices.pct_change(20)
            
            # Volatility features
            features['volatility_10d'] = features['returns_1d'].rolling(10).std()
            features['volatility_30d'] = features['returns_1d'].rolling(30).std()
            
            # Support/Resistance levels
            features['high_20d'] = features['high'].rolling(20).max() if 'high' in features.columns else close_prices.rolling(20).max()
            features['low_20d'] = features['low'].rolling(20).min() if 'low' in features.columns else close_prices.rolling(20).min()
            
        # Economic calendar features (simulated)
        features['day_of_week'] = pd.to_datetime(features.index).dayofweek
        features['month'] = pd.to_datetime(features.index).month
        features['quarter'] = pd.to_datetime(features.index).quarter
        
        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            if 'close' in features.columns:
                features[f'close_lag_{lag}'] = features['close'].shift(lag)
                features[f'returns_lag_{lag}'] = features['returns_1d'].shift(lag)
        
        return features
    
    def prepare_training_data(self, data: pd.DataFrame, target_col: str = 'close', 
                            forecast_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data with features and targets"""
        
        # Create features
        feature_data = self.create_features(data)
        
        # Create target (future price movement)
        if forecast_horizon == 1:
            target = feature_data[target_col].shift(-1)  # Next day's price
        else:
            target = feature_data[target_col].pct_change(forecast_horizon).shift(-forecast_horizon)
        
        # Remove NaN values
        valid_idx = ~(feature_data.isnull().any(axis=1) | target.isnull())
        feature_data = feature_data[valid_idx]
        target = target[valid_idx]
        
        # Select only numeric features
        numeric_features = feature_data.select_dtypes(include=[np.number])
        
        return numeric_features, target
    
    def train_ensemble(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict:
        """Train ensemble of models"""
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['main'] = scaler
        
        # Train models
        model_results = {}
        predictions = {}
        
        for name, model in self.ensemble_models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            test_mae = mean_absolute_error(y_test, test_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            model_results[name] = {
                'model': model,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2
            }
            
            predictions[name] = test_pred
            
            # Feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(X_train.columns, model.feature_importances_))
        
        # Ensemble prediction (average of all models)
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        
        model_results['ensemble'] = {
            'test_rmse': ensemble_rmse,
            'test_mae': ensemble_mae,
            'test_r2': ensemble_r2,
            'predictions': ensemble_pred,
            'actual': y_test.values
        }
        
        self.models = model_results
        return model_results
    
    def predict_next_values(self, data: pd.DataFrame, n_periods: int = 30) -> Dict:
        """Predict future FX values"""
        
        if not self.models:
            raise ValueError("Models not trained. Call train_ensemble first.")
        
        # Prepare recent data
        feature_data = self.create_features(data)
        numeric_features = feature_data.select_dtypes(include=[np.number]).iloc[-1:] 
        
        # Scale features
        scaled_features = self.scalers['main'].transform(numeric_features.fillna(0))
        
        # Generate predictions
        predictions = {}
        
        for name, model_info in self.models.items():
            if name != 'ensemble':
                model = model_info['model']
                pred = model.predict(scaled_features)[0]
                predictions[name] = pred
        
        # Ensemble prediction
        ensemble_pred = np.mean(list(predictions.values()))
        
        # Generate forecast sequence (simplified)
        forecast_dates = pd.date_range(
            start=data.index[-1] + timedelta(days=1),
            periods=n_periods,
            freq='D'
        )
        
        # Simple trend projection (can be made more sophisticated)
        recent_trend = data['close'].pct_change(5).iloc[-1] if 'close' in data.columns else 0
        noise_factor = 0.02  # 2% daily volatility
        
        forecast_values = []
        current_price = data['close'].iloc[-1] if 'close' in data.columns else 100
        
        for i in range(n_periods):
            # Add trend, mean reversion, and random noise
            trend_component = recent_trend * np.exp(-i * 0.1)  # Decay trend
            noise_component = np.random.normal(0, noise_factor)
            
            price_change = trend_component + noise_component
            current_price *= (1 + price_change)
            forecast_values.append(current_price)
        
        forecast_df = pd.DataFrame({
            'forecast': forecast_values,
            'date': forecast_dates
        }).set_index('date')
        
        return {
            'next_prediction': ensemble_pred,
            'forecast_series': forecast_df,
            'model_predictions': predictions,
            'confidence_interval': {
                'lower': np.array(forecast_values) * 0.95,
                'upper': np.array(forecast_values) * 1.05
            }
        }
    
    def get_trading_signals(self, data: pd.DataFrame) -> Dict:
        """Generate trading signals based on model predictions and technical analysis"""
        
        features = self.create_features(data)
        signals = {}
        
        if 'close' in data.columns:
            close_prices = data['close']
            
            # Technical signals
            signals['sma_signal'] = 'BUY' if close_prices.iloc[-1] > features['sma_20'].iloc[-1] else 'SELL'
            signals['rsi_signal'] = 'BUY' if features['rsi'].iloc[-1] < 30 else 'SELL' if features['rsi'].iloc[-1] > 70 else 'HOLD'
            
            # Bollinger Bands signal
            bb_pos = features['bb_position'].iloc[-1]
            if bb_pos < 0.2:
                signals['bb_signal'] = 'BUY'
            elif bb_pos > 0.8:
                signals['bb_signal'] = 'SELL'
            else:
                signals['bb_signal'] = 'HOLD'
            
            # MACD signal
            macd_current = features['macd'].iloc[-1]
            macd_signal = features['macd_signal'].iloc[-1]
            signals['macd_signal'] = 'BUY' if macd_current > macd_signal else 'SELL'
            
            # Volatility-adjusted signal strength
            vol_current = features['volatility_30d'].iloc[-1]
            vol_avg = features['volatility_30d'].mean()
            vol_multiplier = min(vol_current / vol_avg, 2.0) if vol_avg > 0 else 1.0
            
            # Composite signal
            buy_signals = sum(1 for signal in signals.values() if signal == 'BUY')
            sell_signals = sum(1 for signal in signals.values() if signal == 'SELL')
            
            if buy_signals > sell_signals:
                signals['composite'] = 'BUY'
                signals['strength'] = (buy_signals / len(signals)) * vol_multiplier
            elif sell_signals > buy_signals:
                signals['composite'] = 'SELL'
                signals['strength'] = (sell_signals / len(signals)) * vol_multiplier
            else:
                signals['composite'] = 'HOLD'
                signals['strength'] = 0.5
        
        return signals
    
    def risk_assessment(self, data: pd.DataFrame) -> Dict:
        """Comprehensive FX risk assessment"""
        
        if 'close' not in data.columns:
            return {'error': 'Close prices required for risk assessment'}
        
        returns = data['close'].pct_change().dropna()
        
        # Basic risk metrics
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility %
        var_95 = np.percentile(returns, 5) * 100  # 5% VaR
        var_99 = np.percentile(returns, 1) * 100  # 1% VaR
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Sharpe ratio (assuming 3% risk-free rate)
        risk_free_rate = 0.03
        excess_returns = returns.mean() * 252 - risk_free_rate
        sharpe_ratio = excess_returns / (returns.std() * np.sqrt(252))
        
        # Volatility forecast
        vol_forecast = self.volatility_model.volatility_forecast(returns, horizon=30)
        
        return {
            'annualized_volatility': volatility,
            'var_95': var_95,
            'var_99': var_99,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'volatility_forecast_30d': vol_forecast * 100,
            'risk_level': 'HIGH' if volatility > 20 else 'MEDIUM' if volatility > 10 else 'LOW'
        }

def get_sample_fx_data() -> pd.DataFrame:
    """Generate sample FX data for testing"""
    
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    n_days = len(dates)
    
    # Simulate KES/USD exchange rate
    np.random.seed(42)
    
    # Base rate around 130 KES/USD
    base_rate = 130.0
    
    # Generate realistic price movements
    returns = np.random.normal(0, 0.015, n_days)  # 1.5% daily volatility
    
    # Add trends and seasonality
    trend = np.linspace(0, 0.2, n_days)  # Gradual weakening
    seasonal = 0.05 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)  # Annual cycle
    
    # Combine components
    log_prices = np.log(base_rate) + np.cumsum(returns) + trend + seasonal
    close_prices = np.exp(log_prices)
    
    # Generate OHLC data
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    open_prices = np.roll(close_prices, 1)  # Previous day's close as open
    open_prices[0] = close_prices[0]
    
    # Volume (simulated)
    volume = np.random.lognormal(10, 0.5, n_days)
    
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    return data

# Example usage and testing
if __name__ == "__main__":
    # Create FX predictor
    fx_model = FXPredictor("KES/USD")
    
    # Get sample data
    sample_data = get_sample_fx_data()
    
    print("FX Model Initialized")
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Date range: {sample_data.index[0]} to {sample_data.index[-1]}")
    
    # Prepare training data
    X, y = fx_model.prepare_training_data(sample_data)
    print(f"Training features shape: {X.shape}")
    
    # Train models
    print("\nTraining ensemble models...")
    results = fx_model.train_ensemble(X, y)
    
    # Display results
    print("\nModel Performance:")
    for name, metrics in results.items():
        if name != 'ensemble':
            print(f"{name}: RMSE={metrics['test_rmse']:.4f}, R²={metrics['test_r2']:.4f}")
        else:
            print(f"Ensemble: RMSE={metrics['test_rmse']:.4f}, R²={metrics['test_r2']:.4f}")
    
    # Generate predictions
    predictions = fx_model.predict_next_values(sample_data, n_periods=30)
    print(f"\nNext prediction: {predictions['next_prediction']:.4f}")
    
    # Trading signals
    signals = fx_model.get_trading_signals(sample_data)
    print(f"\nTrading signals: {signals}")
    
    # Risk assessment
    risk_metrics = fx_model.risk_assessment(sample_data)
    print(f"\nRisk Assessment: {risk_metrics}")
