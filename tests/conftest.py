"""
Test Configuration and Fixtures
===============================

Shared test configuration, fixtures, and utilities for the EconoNet test suite.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from typing import Generator, Dict, Any
import tempfile
import warnings

# Suppress warnings during testing
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture(scope="session")
def sample_economic_data():
    """
    Generate comprehensive sample economic data for testing
    
    Returns:
        pd.DataFrame: Multi-variate economic time series
    """
    np.random.seed(42)  # Ensure reproducible tests
    
    # Generate 5 years of monthly data
    dates = pd.date_range('2019-01-01', '2023-12-31', freq='M')
    n_periods = len(dates)
    
    # Create realistic economic data with trends and seasonality
    time_index = np.arange(n_periods)
    
    # GDP with trend and business cycle
    gdp_trend = 1000 + 20 * time_index  # Growing trend
    gdp_cycle = 50 * np.sin(2 * np.pi * time_index / 48)  # 4-year cycle
    gdp_seasonal = 10 * np.sin(2 * np.pi * time_index / 12)  # Annual seasonality
    gdp_noise = np.random.normal(0, 15, n_periods)
    gdp = gdp_trend + gdp_cycle + gdp_seasonal + gdp_noise
    
    # Inflation with autocorrelation
    inflation = np.zeros(n_periods)
    inflation[0] = 2.0
    for i in range(1, n_periods):
        inflation[i] = 0.8 * inflation[i-1] + np.random.normal(0.1, 0.3)
    
    # Interest rates correlated with inflation
    interest_rates = 1.5 + 0.5 * inflation + np.random.normal(0, 0.2, n_periods)
    
    # Exchange rate with volatility clustering
    fx_returns = np.random.normal(0, 0.02, n_periods)
    fx_returns[30:35] = np.random.normal(0, 0.05, 5)  # Volatility spike
    fx_rate = 100 * np.exp(np.cumsum(fx_returns))
    
    # Public debt with growing trend
    debt_growth = np.random.normal(0.01, 0.005, n_periods)
    public_debt = 500 * np.exp(np.cumsum(debt_growth))
    
    # Trade balance with seasonality
    trade_seasonal = 20 * np.sin(2 * np.pi * time_index / 12 + np.pi/4)
    trade_balance = -10 + trade_seasonal + np.random.normal(0, 8, n_periods)
    
    # Unemployment rate
    unemployment = 5 + 2 * np.sin(2 * np.pi * time_index / 48) + np.random.normal(0, 0.3, n_periods)
    unemployment = np.clip(unemployment, 2, 15)  # Realistic bounds
    
    data = pd.DataFrame({
        'GDP': gdp,
        'Inflation': inflation,
        'Interest_Rate': interest_rates,
        'Exchange_Rate': fx_rate,
        'Public_Debt': public_debt,
        'Trade_Balance': trade_balance,
        'Unemployment': unemployment,
        'GDP_Growth': np.concatenate([[np.nan], np.diff(gdp) / gdp[:-1] * 100]),
        'Debt_to_GDP': public_debt / gdp * 100
    }, index=dates)
    
    return data

@pytest.fixture
def sample_returns_data():
    """
    Generate sample financial returns data
    
    Returns:
        pd.DataFrame: Daily returns for multiple assets
    """
    np.random.seed(42)
    
    # Generate 2 years of daily data
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    n_periods = len(dates)
    
    # Create correlated returns
    correlation_matrix = np.array([
        [1.0, 0.6, 0.3],
        [0.6, 1.0, 0.4],
        [0.3, 0.4, 1.0]
    ])
    
    # Generate correlated random variables
    random_vars = np.random.multivariate_normal(
        mean=[0, 0, 0],
        cov=correlation_matrix,
        size=n_periods
    )
    
    # Scale to realistic return volatilities
    returns_data = pd.DataFrame({
        'Stock_Index': random_vars[:, 0] * 0.015,  # 1.5% daily vol
        'Bond_Index': random_vars[:, 1] * 0.008,   # 0.8% daily vol
        'Currency': random_vars[:, 2] * 0.012      # 1.2% daily vol
    }, index=dates)
    
    # Add some fat tails and volatility clustering
    for col in returns_data.columns:
        # Add occasional large moves
        shock_indices = np.random.choice(n_periods, size=10, replace=False)
        returns_data.loc[returns_data.index[shock_indices], col] *= 3
    
    return returns_data

@pytest.fixture
def sample_forecast_config():
    """
    Sample configuration for forecasting models
    
    Returns:
        dict: Configuration parameters
    """
    return {
        'forecast_horizon': 12,
        'test_size': 0.2,
        'cv_folds': 3,
        'confidence_level': 0.95,
        'models': {
            'arima': {
                'max_p': 3,
                'max_d': 2,
                'max_q': 3,
                'seasonal': True
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1
            },
            'ensemble': {
                'method': 'weighted_average',
                'weights': None
            }
        }
    }

@pytest.fixture
def sample_risk_config():
    """
    Sample configuration for risk analysis
    
    Returns:
        dict: Risk analysis parameters
    """
    return {
        'confidence_levels': [0.95, 0.99],
        'var_methods': ['historical', 'parametric', 'monte_carlo'],
        'monte_carlo': {
            'n_simulations': 1000,
            'time_horizon': 30
        },
        'stress_scenarios': {
            'recession': {'gdp_shock': -0.05, 'unemployment_shock': 0.03},
            'inflation_shock': {'inflation_shock': 0.02, 'interest_shock': 0.015},
            'fx_crisis': {'fx_shock': 0.15, 'trade_shock': -0.1}
        }
    }

@pytest.fixture
def mock_model_results():
    """
    Mock model results for testing visualization and reporting
    
    Returns:
        dict: Sample model evaluation results
    """
    return {
        'ARIMA': {
            'Test_R2': 0.85,
            'MAE': 12.5,
            'RMSE': 18.3,
            'MAPE': 2.1,
            'Training_Time': 5.2,
            'Hyperparameters': {'order': (2, 1, 1), 'seasonal_order': (1, 1, 1, 12)}
        },
        'XGBoost': {
            'Test_R2': 0.91,
            'MAE': 9.8,
            'RMSE': 14.1,
            'MAPE': 1.7,
            'Training_Time': 8.7,
            'Hyperparameters': {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05}
        },
        'Prophet': {
            'Test_R2': 0.88,
            'MAE': 11.2,
            'RMSE': 16.5,
            'MAPE': 1.9,
            'Training_Time': 12.4,
            'Hyperparameters': {'seasonality_mode': 'multiplicative', 'changepoint_prior_scale': 0.05}
        },
        'Ensemble': {
            'Test_R2': 0.93,
            'MAE': 8.9,
            'RMSE': 13.2,
            'MAPE': 1.5,
            'Training_Time': 26.3,
            'Hyperparameters': {'method': 'weighted_average', 'weights': [0.3, 0.5, 0.2]}
        }
    }

@pytest.fixture
def sample_stress_test_data():
    """
    Generate sample data for stress testing
    
    Returns:
        pd.DataFrame: Economic indicators for stress testing
    """
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='Q')
    n_periods = len(dates)
    
    data = pd.DataFrame({
        'Portfolio_Value': 1000 * np.exp(np.cumsum(np.random.normal(0.02, 0.05, n_periods))),
        'GDP_Growth': np.random.normal(0.02, 0.01, n_periods),
        'Inflation_Rate': np.random.normal(0.025, 0.005, n_periods),
        'Unemployment_Rate': np.random.normal(0.05, 0.01, n_periods),
        'Interest_Rate': np.random.normal(0.03, 0.005, n_periods),
        'Currency_Return': np.random.normal(0, 0.02, n_periods)
    }, index=dates)
    
    return data

class TestDataValidator:
    """Utility class for validating test data quality"""
    
    @staticmethod
    def validate_time_series(data: pd.DataFrame, required_columns: list = None) -> bool:
        """
        Validate time series data for testing
        
        Args:
            data: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            bool: True if data is valid
        """
        # Check index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            return False
        
        # Check for required columns
        if required_columns:
            if not all(col in data.columns for col in required_columns):
                return False
        
        # Check for excessive missing values
        if data.isnull().sum().sum() > len(data) * 0.1:  # >10% missing
            return False
        
        # Check for infinite values
        if np.isinf(data.select_dtypes(include=[np.number])).any().any():
            return False
        
        return True
    
    @staticmethod
    def validate_model_results(results: dict) -> bool:
        """
        Validate model evaluation results
        
        Args:
            results: Dictionary of model results
            
        Returns:
            bool: True if results are valid
        """
        required_metrics = ['Test_R2', 'MAE', 'RMSE']
        
        for model_name, metrics in results.items():
            if not all(metric in metrics for metric in required_metrics):
                return False
            
            # Check metric ranges
            if not (0 <= metrics.get('Test_R2', 0) <= 1):
                return False
            
            if metrics.get('MAE', 0) < 0 or metrics.get('RMSE', 0) < 0:
                return False
        
        return True

# Test markers for different test categories
pytest_plugins = ["pytest_html"]

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "external: mark test as requiring external dependencies"
    )

# Custom assertion helpers
def assert_forecast_quality(forecasts: np.ndarray, actuals: np.ndarray = None, 
                           tolerance: float = 0.1) -> None:
    """
    Assert forecast quality meets minimum standards
    
    Args:
        forecasts: Forecast values
        actuals: Actual values (optional)
        tolerance: Acceptable error tolerance
    """
    # Basic checks
    assert len(forecasts) > 0, "Forecasts should not be empty"
    assert not np.isnan(forecasts).any(), "Forecasts should not contain NaN"
    assert not np.isinf(forecasts).any(), "Forecasts should not contain infinite values"
    
    # Quality checks if actuals provided
    if actuals is not None:
        assert len(forecasts) == len(actuals), "Forecasts and actuals must have same length"
        
        # Calculate MAPE
        mape = np.mean(np.abs((actuals - forecasts) / actuals))
        assert mape <= tolerance, f"MAPE {mape:.3f} exceeds tolerance {tolerance}"

def assert_risk_metrics_valid(metrics: dict) -> None:
    """
    Assert risk metrics are valid
    
    Args:
        metrics: Dictionary of risk metrics
    """
    assert 'VaR' in metrics, "VaR should be in risk metrics"
    assert 'CVaR' in metrics, "CVaR should be in risk metrics"
    
    # VaR should be negative (representing loss)
    assert metrics['VaR'] <= 0, "VaR should be negative or zero"
    
    # CVaR should be more extreme than VaR
    assert metrics['CVaR'] <= metrics['VaR'], "CVaR should be <= VaR"
    
    # Volatility should be positive
    if 'Volatility' in metrics:
        assert metrics['Volatility'] >= 0, "Volatility should be non-negative"
