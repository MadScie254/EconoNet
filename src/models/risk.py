"""
Advanced Risk Analysis Models
============================

Comprehensive risk assessment module including:
- Value at Risk (VaR) and Conditional VaR
- Monte Carlo simulations for scenario analysis
- Credit risk modeling with machine learning
- Stress testing frameworks
- Portfolio risk optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Core libraries
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Advanced statistics
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

class BaseRiskModel(ABC):
    """Base class for all risk models"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.is_fitted = False
        
    @abstractmethod
    def calculate_risk_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate risk metrics for given returns"""
        pass
    
    def validate_returns(self, returns: np.ndarray) -> np.ndarray:
        """Validate and clean returns data"""
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]
        returns = returns[~np.isinf(returns)]
        
        if len(returns) == 0:
            raise ValueError("No valid returns data provided")
        
        return returns

class VaRCalculator(BaseRiskModel):
    """Value at Risk and Conditional VaR calculator with multiple methods"""
    
    def __init__(self, method: str = 'historical', **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.distribution_params = None
        
    def historical_var(self, returns: np.ndarray) -> Tuple[float, float]:
        """Calculate VaR using historical simulation"""
        returns = self.validate_returns(returns)
        alpha = 1 - self.confidence_level
        
        var = np.percentile(returns, alpha * 100)
        cvar = returns[returns <= var].mean()
        
        return var, cvar
    
    def parametric_var(self, returns: np.ndarray, distribution: str = 'normal') -> Tuple[float, float]:
        """Calculate VaR using parametric approach"""
        returns = self.validate_returns(returns)
        alpha = 1 - self.confidence_level
        
        if distribution == 'normal':
            mean = np.mean(returns)
            std = np.std(returns)
            var = stats.norm.ppf(alpha, mean, std)
            
            # Conditional VaR for normal distribution
            phi = stats.norm.pdf(stats.norm.ppf(alpha))
            cvar = mean - std * phi / alpha
            
        elif distribution == 't':
            # Fit t-distribution
            params = stats.t.fit(returns)
            var = stats.t.ppf(alpha, *params)
            
            # Approximate CVaR for t-distribution
            df, loc, scale = params
            cvar = loc + scale * stats.t.pdf(stats.t.ppf(alpha, df), df) * (df + (stats.t.ppf(alpha, df))**2) / (df - 1) / alpha
            
        elif distribution == 'skewnorm':
            # Fit skewed normal distribution
            params = stats.skewnorm.fit(returns)
            var = stats.skewnorm.ppf(alpha, *params)
            
            # Approximate CVaR
            cvar = returns[returns <= var].mean()
            
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        self.distribution_params = params if distribution != 'normal' else (mean, std)
        return var, cvar
    
    def monte_carlo_var(self, returns: np.ndarray, n_simulations: int = 10000) -> Tuple[float, float]:
        """Calculate VaR using Monte Carlo simulation"""
        returns = self.validate_returns(returns)
        alpha = 1 - self.confidence_level
        
        # Fit distribution to historical returns
        mean = np.mean(returns)
        std = np.std(returns)
        
        # Bootstrap from historical data
        simulated_returns = np.random.choice(returns, size=n_simulations, replace=True)
        
        var = np.percentile(simulated_returns, alpha * 100)
        cvar = simulated_returns[simulated_returns <= var].mean()
        
        return var, cvar
    
    def calculate_risk_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive VaR metrics"""
        returns = self.validate_returns(returns)
        
        # Calculate VaR using specified method
        if self.method == 'historical':
            var, cvar = self.historical_var(returns)
        elif self.method == 'parametric':
            var, cvar = self.parametric_var(returns)
        elif self.method == 'monte_carlo':
            var, cvar = self.monte_carlo_var(returns)
        else:
            raise ValueError(f"Unknown VaR method: {self.method}")
        
        # Additional risk metrics
        volatility = np.std(returns)
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        sharpe_ratio = np.mean(returns) / volatility if volatility > 0 else 0
        max_drawdown = self._calculate_max_drawdown(returns)
        
        return {
            'VaR': var,
            'CVaR': cvar,
            'Volatility': volatility,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Confidence_Level': self.confidence_level
        }
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)

class MonteCarloSimulator:
    """Monte Carlo simulation engine for economic scenarios"""
    
    def __init__(self, n_simulations: int = 10000, time_horizon: int = 252):
        self.n_simulations = n_simulations
        self.time_horizon = time_horizon
        self.simulation_results = None
        
    def geometric_brownian_motion(self, S0: float, mu: float, sigma: float,
                                 correlated_assets: Optional[Dict[str, Tuple[float, float, float]]] = None) -> pd.DataFrame:
        """Simulate geometric Brownian motion for asset prices"""
        
        # Time grid
        dt = 1 / 252  # Daily steps assuming 252 trading days
        t = np.linspace(0, self.time_horizon * dt, self.time_horizon + 1)
        
        # Generate random shocks
        np.random.seed(42)  # For reproducibility
        dW = np.random.normal(0, np.sqrt(dt), (self.n_simulations, self.time_horizon))
        
        # Simulate primary asset
        S = np.zeros((self.n_simulations, self.time_horizon + 1))
        S[:, 0] = S0
        
        for i in range(1, self.time_horizon + 1):
            S[:, i] = S[:, i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[:, i-1])
        
        results = pd.DataFrame(S, columns=[f'Day_{i}' for i in range(self.time_horizon + 1)])
        results['Simulation'] = range(self.n_simulations)
        
        # Add correlated assets if specified
        if correlated_assets:
            for asset_name, (S0_asset, mu_asset, sigma_asset, correlation) in correlated_assets.items():
                # Generate correlated random shocks
                dW_corr = correlation * dW + np.sqrt(1 - correlation**2) * np.random.normal(0, np.sqrt(dt), dW.shape)
                
                S_asset = np.zeros((self.n_simulations, self.time_horizon + 1))
                S_asset[:, 0] = S0_asset
                
                for i in range(1, self.time_horizon + 1):
                    S_asset[:, i] = S_asset[:, i-1] * np.exp((mu_asset - 0.5 * sigma_asset**2) * dt + sigma_asset * dW_corr[:, i-1])
                
                # Add to results
                for day in range(self.time_horizon + 1):
                    results[f'{asset_name}_Day_{day}'] = S_asset[:, day]
        
        self.simulation_results = results
        return results
    
    def jump_diffusion(self, S0: float, mu: float, sigma: float, 
                      jump_intensity: float, jump_mean: float, jump_std: float) -> pd.DataFrame:
        """Simulate jump-diffusion process (Merton model)"""
        
        dt = 1 / 252
        t = np.linspace(0, self.time_horizon * dt, self.time_horizon + 1)
        
        np.random.seed(42)
        
        S = np.zeros((self.n_simulations, self.time_horizon + 1))
        S[:, 0] = S0
        
        for sim in range(self.n_simulations):
            for i in range(1, self.time_horizon + 1):
                # Diffusion component
                dW = np.random.normal(0, np.sqrt(dt))
                diffusion = (mu - 0.5 * sigma**2) * dt + sigma * dW
                
                # Jump component
                jump = 0
                if np.random.poisson(jump_intensity * dt) > 0:
                    jump = np.random.normal(jump_mean, jump_std)
                
                S[sim, i] = S[sim, i-1] * np.exp(diffusion + jump)
        
        results = pd.DataFrame(S, columns=[f'Day_{i}' for i in range(self.time_horizon + 1)])
        results['Simulation'] = range(self.n_simulations)
        
        self.simulation_results = results
        return results
    
    def economic_scenario_simulation(self, 
                                   gdp_params: Tuple[float, float],
                                   inflation_params: Tuple[float, float],
                                   interest_rate_params: Tuple[float, float],
                                   correlations: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Simulate correlated economic variables"""
        
        np.random.seed(42)
        
        # Default correlation matrix if not provided
        if correlations is None:
            correlations = np.array([
                [1.0, -0.3, -0.5],  # GDP vs Inflation, Interest Rate
                [-0.3, 1.0, 0.7],   # Inflation vs GDP, Interest Rate
                [-0.5, 0.7, 1.0]    # Interest Rate vs GDP, Inflation
            ])
        
        # Generate correlated random variables
        random_matrix = np.random.multivariate_normal(
            mean=[0, 0, 0],
            cov=correlations,
            size=(self.n_simulations, self.time_horizon)
        )
        
        # Initialize arrays
        gdp = np.zeros((self.n_simulations, self.time_horizon + 1))
        inflation = np.zeros((self.n_simulations, self.time_horizon + 1))
        interest_rate = np.zeros((self.n_simulations, self.time_horizon + 1))
        
        # Initial values
        gdp[:, 0] = gdp_params[0]
        inflation[:, 0] = inflation_params[0]
        interest_rate[:, 0] = interest_rate_params[0]
        
        dt = 1 / 12  # Monthly steps
        
        for i in range(1, self.time_horizon + 1):
            # GDP growth (mean-reverting)
            gdp[:, i] = gdp[:, i-1] + 0.1 * (gdp_params[0] - gdp[:, i-1]) * dt + gdp_params[1] * random_matrix[:, i-1, 0] * np.sqrt(dt)
            
            # Inflation (mean-reverting)
            inflation[:, i] = inflation[:, i-1] + 0.2 * (inflation_params[0] - inflation[:, i-1]) * dt + inflation_params[1] * random_matrix[:, i-1, 1] * np.sqrt(dt)
            
            # Interest rate (mean-reverting with policy response)
            target_rate = interest_rate_params[0] + 0.5 * (inflation[:, i-1] - inflation_params[0])  # Taylor rule
            interest_rate[:, i] = interest_rate[:, i-1] + 0.3 * (target_rate - interest_rate[:, i-1]) * dt + interest_rate_params[1] * random_matrix[:, i-1, 2] * np.sqrt(dt)
        
        # Create results DataFrame
        results = pd.DataFrame()
        
        for i in range(self.time_horizon + 1):
            results[f'GDP_Period_{i}'] = gdp[:, i]
            results[f'Inflation_Period_{i}'] = inflation[:, i]
            results[f'Interest_Rate_Period_{i}'] = interest_rate[:, i]
        
        results['Simulation'] = range(self.n_simulations)
        
        self.simulation_results = results
        return results
    
    def calculate_scenario_statistics(self) -> Dict[str, pd.DataFrame]:
        """Calculate statistics from simulation results"""
        if self.simulation_results is None:
            raise ValueError("No simulation results available. Run a simulation first.")
        
        stats_dict = {}
        
        # Identify variable columns (exclude 'Simulation' column)
        var_columns = [col for col in self.simulation_results.columns if col != 'Simulation']
        
        # Group by time periods
        variables = set([col.split('_Period_')[0] if '_Period_' in col else col.split('_Day_')[0] 
                        for col in var_columns])
        
        for var in variables:
            var_cols = [col for col in var_columns if col.startswith(var)]
            
            if var_cols:
                var_data = self.simulation_results[var_cols]
                
                stats_df = pd.DataFrame({
                    'Mean': var_data.mean(),
                    'Std': var_data.std(),
                    'Min': var_data.min(),
                    'Q25': var_data.quantile(0.25),
                    'Median': var_data.median(),
                    'Q75': var_data.quantile(0.75),
                    'Max': var_data.max(),
                    'VaR_5%': var_data.quantile(0.05),
                    'VaR_1%': var_data.quantile(0.01)
                })
                
                stats_dict[var] = stats_df
        
        return stats_dict

class CreditRiskModel:
    """Credit risk assessment using machine learning"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
    def prepare_features(self, data: pd.DataFrame, 
                        financial_ratios: List[str],
                        macro_variables: List[str]) -> pd.DataFrame:
        """Prepare features for credit risk modeling"""
        features = data[financial_ratios + macro_variables].copy()
        
        # Add derived features
        if 'debt_to_equity' in financial_ratios and 'current_ratio' in financial_ratios:
            features['leverage_liquidity'] = features['debt_to_equity'] / features['current_ratio']
        
        if 'revenue' in data.columns and 'total_assets' in data.columns:
            features['asset_turnover'] = data['revenue'] / data['total_assets']
        
        # Lag features for time series
        for col in financial_ratios:
            if col in features.columns:
                features[f'{col}_lag1'] = features[col].shift(1)
                features[f'{col}_change'] = features[col].pct_change()
        
        # Rolling statistics
        for col in financial_ratios:
            if col in features.columns:
                features[f'{col}_ma3'] = features[col].rolling(3).mean()
                features[f'{col}_vol3'] = features[col].rolling(3).std()
        
        return features.dropna()
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'CreditRiskModel':
        """Fit credit risk model"""
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize model
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Fit model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        return self
    
    def predict_default_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Predict default probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)[:, 1]  # Probability of default class
        else:
            return self.model.decision_function(X_scaled)
    
    def calculate_credit_metrics(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Calculate credit model performance metrics"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Predictions
        y_pred = self.model.predict(self.scaler.transform(X_test))
        y_proba = self.predict_default_probability(X_test)
        
        # Metrics
        auc_score = roc_auc_score(y_test, y_proba)
        
        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'AUC': auc_score,
            'Precision': precision,
            'Recall': recall,
            'Specificity': specificity,
            'Default_Rate': y_test.mean(),
            'Predicted_Default_Rate': y_pred.mean()
        }

class StressTesting:
    """Comprehensive stress testing framework"""
    
    def __init__(self):
        self.stress_scenarios = {}
        self.baseline_metrics = {}
        
    def define_scenario(self, name: str, shocks: Dict[str, float]):
        """Define a stress scenario with variable shocks"""
        self.stress_scenarios[name] = shocks
    
    def apply_shocks(self, data: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
        """Apply stress scenario shocks to data"""
        if scenario_name not in self.stress_scenarios:
            raise ValueError(f"Scenario {scenario_name} not defined")
        
        stressed_data = data.copy()
        shocks = self.stress_scenarios[scenario_name]
        
        for variable, shock in shocks.items():
            if variable in stressed_data.columns:
                if shock > 0:  # Multiplicative shock
                    stressed_data[variable] *= (1 + shock)
                else:  # Additive shock
                    stressed_data[variable] += shock
        
        return stressed_data
    
    def run_stress_test(self, data: pd.DataFrame, 
                       risk_model: BaseRiskModel,
                       scenarios: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """Run comprehensive stress tests"""
        
        if scenarios is None:
            scenarios = list(self.stress_scenarios.keys())
        
        results = {}
        
        # Baseline (no stress)
        if 'returns' in data.columns:
            baseline_metrics = risk_model.calculate_risk_metrics(data['returns'].values)
            results['Baseline'] = baseline_metrics
            self.baseline_metrics = baseline_metrics
        
        # Stress scenarios
        for scenario in scenarios:
            stressed_data = self.apply_shocks(data, scenario)
            
            if 'returns' in stressed_data.columns:
                stressed_metrics = risk_model.calculate_risk_metrics(stressed_data['returns'].values)
                results[scenario] = stressed_metrics
        
        return results
    
    def create_default_scenarios(self):
        """Create standard stress scenarios"""
        self.define_scenario('Mild_Recession', {
            'gdp_growth': -0.02,
            'unemployment': 0.02,
            'interest_rate': 0.01
        })
        
        self.define_scenario('Severe_Recession', {
            'gdp_growth': -0.05,
            'unemployment': 0.05,
            'interest_rate': 0.02
        })
        
        self.define_scenario('Financial_Crisis', {
            'credit_spread': 0.03,
            'volatility': 0.5,
            'liquidity_ratio': -0.3
        })
        
        self.define_scenario('Inflation_Shock', {
            'inflation': 0.03,
            'interest_rate': 0.025,
            'real_gdp': -0.015
        })
        
        self.define_scenario('Market_Crash', {
            'stock_returns': -0.3,
            'volatility': 1.0,
            'correlation': 0.2
        })

class PortfolioRiskOptimizer:
    """Portfolio risk optimization and allocation"""
    
    def __init__(self, risk_model: str = 'covariance'):
        self.risk_model = risk_model
        self.expected_returns = None
        self.cov_matrix = None
        self.optimal_weights = None
        
    def estimate_parameters(self, returns: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate expected returns and covariance matrix"""
        
        # Expected returns (using historical mean)
        self.expected_returns = returns.mean().values
        
        # Covariance matrix
        if self.risk_model == 'covariance':
            self.cov_matrix = returns.cov().values
        elif self.risk_model == 'shrinkage':
            # Ledoit-Wolf shrinkage estimator
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            self.cov_matrix = lw.fit(returns).covariance_
        else:
            raise ValueError(f"Unknown risk model: {self.risk_model}")
        
        return self.expected_returns, self.cov_matrix
    
    def optimize_portfolio(self, returns: pd.DataFrame, 
                          objective: str = 'sharpe',
                          constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """Optimize portfolio allocation"""
        
        self.estimate_parameters(returns)
        n_assets = len(returns.columns)
        
        # Constraints
        bounds = tuple((0, 1) for _ in range(n_assets))  # Long-only
        
        # Budget constraint (weights sum to 1)
        budget_constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        constraints_list = [budget_constraint]
        
        # Additional constraints
        if constraints:
            if 'max_weight' in constraints:
                for i in range(n_assets):
                    constraints_list.append({
                        'type': 'ineq', 
                        'fun': lambda x, i=i: constraints['max_weight'] - x[i]
                    })
            
            if 'min_weight' in constraints:
                for i in range(n_assets):
                    constraints_list.append({
                        'type': 'ineq', 
                        'fun': lambda x, i=i: x[i] - constraints['min_weight']
                    })
        
        # Objective function
        if objective == 'sharpe':
            def objective_func(weights):
                portfolio_return = np.dot(weights, self.expected_returns)
                portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
                return -(portfolio_return / np.sqrt(portfolio_variance))  # Negative for minimization
        
        elif objective == 'min_variance':
            def objective_func(weights):
                return np.dot(weights.T, np.dot(self.cov_matrix, weights))
        
        elif objective == 'max_return':
            def objective_func(weights):
                return -np.dot(weights, self.expected_returns)  # Negative for minimization
        
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Optimization
        initial_guess = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            objective_func,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if result.success:
            self.optimal_weights = result.x
            
            # Calculate portfolio metrics
            portfolio_return = np.dot(self.optimal_weights, self.expected_returns)
            portfolio_variance = np.dot(self.optimal_weights.T, np.dot(self.cov_matrix, self.optimal_weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
            
            return {
                'weights': self.optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'optimization_success': True
            }
        else:
            return {
                'weights': initial_guess,
                'expected_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'optimization_success': False,
                'error_message': result.message
            }

def create_comprehensive_risk_report(data: pd.DataFrame, 
                                   returns_column: str = 'returns',
                                   confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, Any]:
    """Generate comprehensive risk assessment report"""
    
    if returns_column not in data.columns:
        raise ValueError(f"Returns column '{returns_column}' not found in data")
    
    returns = data[returns_column].dropna().values
    report = {}
    
    # VaR Analysis
    for confidence in confidence_levels:
        var_calc = VaRCalculator(method='historical', confidence_level=confidence)
        var_metrics = var_calc.calculate_risk_metrics(returns)
        report[f'VaR_{int(confidence*100)}%'] = var_metrics
    
    # Monte Carlo Simulation
    mc_sim = MonteCarloSimulator(n_simulations=10000, time_horizon=252)
    
    # Estimate parameters for simulation
    returns_mean = np.mean(returns)
    returns_std = np.std(returns)
    S0 = 100  # Normalized starting value
    
    # Run simulation
    mc_results = mc_sim.geometric_brownian_motion(S0, returns_mean, returns_std)
    mc_stats = mc_sim.calculate_scenario_statistics()
    
    report['Monte_Carlo_Simulation'] = {
        'simulation_statistics': mc_stats,
        'final_value_distribution': {
            'mean': mc_results.iloc[:, -1].mean(),
            'std': mc_results.iloc[:, -1].std(),
            'var_95': mc_results.iloc[:, -1].quantile(0.05),
            'var_99': mc_results.iloc[:, -1].quantile(0.01)
        }
    }
    
    # Stress Testing
    stress_tester = StressTesting()
    stress_tester.create_default_scenarios()
    
    # Create synthetic stress data
    stress_data = pd.DataFrame({
        'returns': returns,
        'gdp_growth': np.random.normal(0.02, 0.01, len(returns)),
        'unemployment': np.random.normal(0.05, 0.01, len(returns)),
        'interest_rate': np.random.normal(0.03, 0.005, len(returns))
    })
    
    var_model = VaRCalculator(method='historical', confidence_level=0.95)
    stress_results = stress_tester.run_stress_test(stress_data, var_model)
    
    report['Stress_Testing'] = stress_results
    
    return report
