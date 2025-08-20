"""
Test Suite for EconoNet Models
==============================

Comprehensive test coverage for forecasting models, risk analysis,
and utility functions with pytest framework.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.forecasting import (
    ARIMAForecaster, XGBoostForecaster, EnsembleForecaster, 
    create_forecasting_pipeline
)
from models.risk import VaRCalculator, MonteCarloSimulator, StressTesting
from models.BaseDebtPredictor import BaseDebtPredictor, EnhancedDebtPredictor
from utils.plotting import create_time_series_plot, create_forecast_plot

class TestForecasting:
    """Test forecasting models"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data for testing"""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
        np.random.seed(42)
        
        # Generate synthetic economic data
        trend = np.linspace(100, 150, len(dates))
        seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
        noise = np.random.normal(0, 5, len(dates))
        
        data = pd.DataFrame({
            'target': trend + seasonal + noise,
            'feature1': np.random.normal(50, 10, len(dates)),
            'feature2': np.random.normal(25, 5, len(dates))
        }, index=dates)
        
        return data
    
    def test_arima_forecaster(self, sample_data):
        """Test ARIMA forecaster"""
        model = ARIMAForecaster(forecast_horizon=6)
        X = sample_data[['feature1', 'feature2']]
        y = sample_data['target']
        
        # Test fitting
        fitted_model = model.fit(X, y)
        assert fitted_model.is_fitted
        assert fitted_model.model is not None
        
        # Test prediction
        forecasts = model.predict()
        assert len(forecasts) == 6
        assert not np.isnan(forecasts).any()
        
        # Test prediction with intervals
        forecasts, lower, upper = model.predict_with_intervals()
        assert len(forecasts) == len(lower) == len(upper) == 6
        assert np.all(lower <= forecasts)
        assert np.all(forecasts <= upper)
    
    def test_xgboost_forecaster(self, sample_data):
        """Test XGBoost forecaster"""
        try:
            model = XGBoostForecaster(forecast_horizon=6, n_estimators=10)
            X = sample_data[['feature1', 'feature2']]
            y = sample_data['target']
            
            # Test fitting
            fitted_model = model.fit(X, y)
            assert fitted_model.is_fitted
            
            # Test prediction
            forecasts = model.predict()
            assert len(forecasts) == 6
            assert not np.isnan(forecasts).any()
            
        except ImportError:
            pytest.skip("XGBoost not available")
    
    def test_ensemble_forecaster(self, sample_data):
        """Test ensemble forecaster"""
        model = EnsembleForecaster(forecast_horizon=6, ensemble_method='simple_average')
        X = sample_data[['feature1', 'feature2']]
        y = sample_data['target']
        
        # Test fitting
        fitted_model = model.fit(X, y)
        assert fitted_model.is_fitted
        assert len(fitted_model.models) > 0
        
        # Test prediction
        forecasts = model.predict()
        assert len(forecasts) == 6
        assert not np.isnan(forecasts).any()
    
    def test_forecasting_pipeline(self, sample_data):
        """Test forecasting pipeline factory"""
        model = create_forecasting_pipeline(
            data=sample_data,
            target_column='target',
            model_type='ensemble',
            forecast_horizon=6
        )
        
        assert model.is_fitted
        
        # Test prediction
        forecasts = model.predict()
        assert len(forecasts) == 6
        assert not np.isnan(forecasts).any()

class TestRiskAnalysis:
    """Test risk analysis models"""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns data"""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
        return returns
    
    def test_var_calculator_historical(self, sample_returns):
        """Test VaR calculator with historical method"""
        var_calc = VaRCalculator(method='historical', confidence_level=0.95)
        
        metrics = var_calc.calculate_risk_metrics(sample_returns)
        
        assert 'VaR' in metrics
        assert 'CVaR' in metrics
        assert 'Volatility' in metrics
        assert 'Sharpe_Ratio' in metrics
        
        # VaR should be negative (loss)
        assert metrics['VaR'] < 0
        # CVaR should be more negative than VaR
        assert metrics['CVaR'] <= metrics['VaR']
    
    def test_var_calculator_parametric(self, sample_returns):
        """Test VaR calculator with parametric method"""
        var_calc = VaRCalculator(method='parametric', confidence_level=0.95)
        
        metrics = var_calc.calculate_risk_metrics(sample_returns)
        
        assert 'VaR' in metrics
        assert 'CVaR' in metrics
        assert metrics['VaR'] < 0
        assert metrics['CVaR'] <= metrics['VaR']
    
    def test_monte_carlo_simulator(self):
        """Test Monte Carlo simulator"""
        simulator = MonteCarloSimulator(n_simulations=100, time_horizon=10)
        
        # Test geometric Brownian motion
        results = simulator.geometric_brownian_motion(
            S0=100, mu=0.05, sigma=0.2
        )
        
        assert len(results) == 100  # Number of simulations
        assert len(results.columns) == 12  # 10 time steps + 1 initial + simulation column
        
        # Test statistics calculation
        stats = simulator.calculate_scenario_statistics()
        assert len(stats) > 0
    
    def test_stress_testing(self, sample_returns):
        """Test stress testing framework"""
        stress_tester = StressTesting()
        stress_tester.create_default_scenarios()
        
        # Create test data
        test_data = pd.DataFrame({
            'returns': sample_returns,
            'gdp_growth': np.random.normal(0.02, 0.01, len(sample_returns)),
            'unemployment': np.random.normal(0.05, 0.01, len(sample_returns))
        })
        
        # Test stress scenarios
        var_model = VaRCalculator(method='historical', confidence_level=0.95)
        results = stress_tester.run_stress_test(test_data, var_model)
        
        assert 'Baseline' in results
        assert len(results) > 1  # Should have baseline + stress scenarios

class TestBaseDebtPredictor:
    """Test base debt predictor models"""
    
    @pytest.fixture
    def sample_debt_data(self):
        """Generate sample debt data"""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'Total_Public_Debt': np.random.normal(1000, 100, len(dates)).cumsum(),
            'GDP': np.random.normal(500, 50, len(dates)).cumsum(),
            'Inflation': np.random.normal(2, 0.5, len(dates)),
            'Interest_Rate': np.random.normal(3, 0.3, len(dates))
        }, index=dates)
        
        return data
    
    def test_enhanced_debt_predictor(self, sample_debt_data):
        """Test enhanced debt predictor"""
        predictor = EnhancedDebtPredictor(
            target_column='Total_Public_Debt',
            forecast_horizon=6,
            n_estimators=10  # Small number for fast testing
        )
        
        # Test data preparation
        X = sample_debt_data.drop(columns=['Total_Public_Debt'])
        y = sample_debt_data['Total_Public_Debt']
        
        # Test fitting
        fitted_predictor = predictor.fit(X, y)
        assert fitted_predictor.is_fitted
        assert len(fitted_predictor.models) > 0
        
        # Test prediction
        predictions = fitted_predictor.predict()
        assert len(predictions) > 0
        assert not np.isnan(predictions).any()
    
    def test_debt_predictor_evaluation(self, sample_debt_data):
        """Test debt predictor evaluation methods"""
        predictor = EnhancedDebtPredictor(n_estimators=10)
        
        X = sample_debt_data.drop(columns=['Total_Public_Debt'])
        y = sample_debt_data['Total_Public_Debt']
        
        fitted_predictor = predictor.fit(X, y)
        
        # Test model results
        assert len(fitted_predictor.model_results) > 0
        
        # Check that all models have evaluation metrics
        for model_name, metrics in fitted_predictor.model_results.items():
            assert 'Test_R2' in metrics
            assert 'MAE' in metrics
            assert 'RMSE' in metrics

class TestUtilities:
    """Test utility functions"""
    
    @pytest.fixture
    def sample_plot_data(self):
        """Generate sample data for plotting tests"""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
        data = pd.DataFrame({
            'series1': np.random.randn(len(dates)).cumsum(),
            'series2': np.random.randn(len(dates)).cumsum()
        }, index=dates)
        return data
    
    def test_time_series_plot(self, sample_plot_data):
        """Test time series plotting function"""
        fig = create_time_series_plot(
            sample_plot_data,
            ['series1', 'series2'],
            title="Test Plot"
        )
        
        # Check that figure is created
        assert fig is not None
        assert len(fig.data) >= 2  # At least 2 series
    
    def test_forecast_plot(self, sample_plot_data):
        """Test forecast plotting function"""
        historical = sample_plot_data['series1']
        forecasts = np.random.randn(6)
        forecast_dates = pd.date_range(historical.index[-1], periods=7, freq='M')[1:]
        
        fig = create_forecast_plot(
            historical_data=historical,
            forecasts=forecasts,
            forecast_dates=forecast_dates,
            title="Test Forecast"
        )
        
        assert fig is not None
        assert len(fig.data) >= 2  # Historical + forecast

class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_complete_forecasting_workflow(self):
        """Test complete forecasting workflow from data to prediction"""
        # Generate test data
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'target': np.random.randn(len(dates)).cumsum() + 100,
            'feature1': np.random.randn(len(dates)).cumsum(),
            'feature2': np.random.randn(len(dates))
        }, index=dates)
        
        # Create and fit model
        model = create_forecasting_pipeline(
            data=data,
            target_column='target',
            model_type='ensemble',
            forecast_horizon=6
        )
        
        # Generate predictions
        forecasts, lower, upper = model.predict_with_intervals()
        
        # Verify results
        assert len(forecasts) == 6
        assert len(lower) == 6
        assert len(upper) == 6
        assert np.all(lower <= forecasts)
        assert np.all(forecasts <= upper)
        assert not np.isnan(forecasts).any()
    
    def test_complete_risk_workflow(self):
        """Test complete risk analysis workflow"""
        # Generate test returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        
        # Calculate VaR
        var_calc = VaRCalculator(method='historical', confidence_level=0.95)
        metrics = var_calc.calculate_risk_metrics(returns)
        
        # Run Monte Carlo simulation
        simulator = MonteCarloSimulator(n_simulations=100, time_horizon=30)
        mc_results = simulator.geometric_brownian_motion(100, 0.05, 0.2)
        
        # Verify complete workflow
        assert 'VaR' in metrics
        assert 'CVaR' in metrics
        assert len(mc_results) == 100
        assert not mc_results.empty

# Performance benchmarks
class TestPerformance:
    """Performance and benchmark tests"""
    
    def test_forecasting_performance(self):
        """Test forecasting model performance with larger dataset"""
        # Generate larger dataset
        dates = pd.date_range('2010-01-01', '2023-12-31', freq='M')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'target': np.random.randn(len(dates)).cumsum() + 1000,
            'feature1': np.random.randn(len(dates)).cumsum(),
            'feature2': np.random.randn(len(dates)),
            'feature3': np.random.randn(len(dates))
        }, index=dates)
        
        import time
        start_time = time.time()
        
        # Create and fit model
        model = create_forecasting_pipeline(
            data=data,
            target_column='target',
            model_type='ensemble',
            forecast_horizon=12
        )
        
        forecasts = model.predict()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Performance assertions
        assert len(forecasts) == 12
        assert execution_time < 30  # Should complete within 30 seconds
        print(f"Forecasting performance: {execution_time:.2f} seconds")
    
    def test_risk_calculation_performance(self):
        """Test risk calculation performance"""
        # Generate large returns dataset
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 5000)  # ~20 years daily data
        
        import time
        start_time = time.time()
        
        # Calculate multiple VaR methods
        for method in ['historical', 'parametric', 'monte_carlo']:
            var_calc = VaRCalculator(method=method, confidence_level=0.95)
            metrics = var_calc.calculate_risk_metrics(returns)
            assert 'VaR' in metrics
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Performance assertion
        assert execution_time < 10  # Should complete within 10 seconds
        print(f"Risk calculation performance: {execution_time:.2f} seconds")

if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html"
    ])
