"""
Advanced Financial Derivatives and Economic Instruments Modeling
================================================================

Ultra-sophisticated financial engineering for complex economic analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedFinancialInstruments:
    """
    Advanced financial derivatives and instruments modeling system
    """
    
    def __init__(self):
        self.risk_free_rate = 0.05  # 5% risk-free rate
        self.volatility_surface = {}
        self.correlation_matrix = {}
        
    def black_scholes_option_pricing(self, S, K, T, r, sigma, option_type='call'):
        """
        Black-Scholes option pricing model with advanced Greeks calculation
        """
        from scipy.stats import norm
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:  # put option
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        
        # Calculate Greeks
        delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - 
                r*K*np.exp(-r*T)*norm.cdf(d2 if option_type == 'call' else -d2))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        rho = (K*T*np.exp(-r*T)*norm.cdf(d2) if option_type == 'call' 
               else -K*T*np.exp(-r*T)*norm.cdf(-d2))
        
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta / 365,  # Per day
            'vega': vega / 100,    # Per 1% vol change
            'rho': rho / 100       # Per 1% rate change
        }
    
    def monte_carlo_exotic_options(self, S0, K, T, r, sigma, option_type='asian', 
                                 n_simulations=10000, n_steps=252):
        """
        Monte Carlo pricing for exotic options
        """
        dt = T / n_steps
        prices = np.zeros((n_simulations, n_steps + 1))
        prices[:, 0] = S0
        
        # Generate random price paths
        for i in range(1, n_steps + 1):
            z = np.random.normal(0, 1, n_simulations)
            prices[:, i] = prices[:, i-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
        
        # Calculate payoffs based on option type
        if option_type == 'asian':
            # Asian option - payoff based on average price
            avg_prices = np.mean(prices, axis=1)
            payoffs = np.maximum(avg_prices - K, 0)
        elif option_type == 'barrier_up_out':
            # Barrier option - knock out if price hits barrier
            barrier = K * 1.2
            knocked_out = np.any(prices > barrier, axis=1)
            final_prices = prices[:, -1]
            payoffs = np.where(knocked_out, 0, np.maximum(final_prices - K, 0))
        elif option_type == 'lookback':
            # Lookback option - payoff based on maximum price
            max_prices = np.max(prices, axis=1)
            payoffs = max_prices - K
        else:  # European
            final_prices = prices[:, -1]
            payoffs = np.maximum(final_prices - K, 0)
        
        # Discount to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)
        confidence_interval = np.exp(-r * T) * np.percentile(payoffs, [2.5, 97.5])
        
        return {
            'price': option_price,
            'confidence_interval': confidence_interval,
            'price_paths': prices[:100],  # Return first 100 paths for visualization
            'payoffs': payoffs
        }
    
    def stochastic_volatility_heston(self, S0, V0, T, r, kappa, theta, sigma_v, rho, 
                                   n_simulations=1000, n_steps=252):
        """
        Heston stochastic volatility model
        """
        dt = T / n_steps
        
        # Initialize arrays
        S = np.zeros((n_simulations, n_steps + 1))
        V = np.zeros((n_simulations, n_steps + 1))
        S[:, 0] = S0
        V[:, 0] = V0
        
        # Generate correlated random numbers
        for i in range(1, n_steps + 1):
            z1 = np.random.normal(0, 1, n_simulations)
            z2 = np.random.normal(0, 1, n_simulations)
            z2_corr = rho * z1 + np.sqrt(1 - rho**2) * z2
            
            # Update variance (with Feller condition)
            V[:, i] = np.maximum(V[:, i-1] + kappa*(theta - V[:, i-1])*dt + 
                               sigma_v*np.sqrt(V[:, i-1])*np.sqrt(dt)*z2_corr, 0)
            
            # Update stock price
            S[:, i] = S[:, i-1] * np.exp((r - 0.5*V[:, i-1])*dt + 
                                       np.sqrt(V[:, i-1])*np.sqrt(dt)*z1)
        
        return S, V
    
    def interest_rate_derivatives_hull_white(self, r0, a, sigma, T, 
                                           n_simulations=1000, n_steps=252):
        """
        Hull-White interest rate model for bond and rate derivatives
        """
        dt = T / n_steps
        rates = np.zeros((n_simulations, n_steps + 1))
        rates[:, 0] = r0
        
        # Mean reversion level (can be time-dependent)
        theta_t = 0.03  # Long-term rate
        
        for i in range(1, n_steps + 1):
            dW = np.random.normal(0, np.sqrt(dt), n_simulations)
            rates[:, i] = (rates[:, i-1] + a*(theta_t - rates[:, i-1])*dt + 
                          sigma*dW)
        
        # Calculate bond prices
        bond_prices = np.exp(-np.cumsum(rates, axis=1) * dt)
        
        return rates, bond_prices
    
    def credit_risk_merton_model(self, asset_value, debt_value, asset_volatility, 
                                risk_free_rate, time_to_maturity):
        """
        Merton's structural credit risk model
        """
        from scipy.stats import norm
        
        # Distance to default
        d1 = (np.log(asset_value/debt_value) + 
              (risk_free_rate + 0.5*asset_volatility**2)*time_to_maturity) / \
             (asset_volatility*np.sqrt(time_to_maturity))
        d2 = d1 - asset_volatility*np.sqrt(time_to_maturity)
        
        # Probability of default
        prob_default = norm.cdf(-d2)
        
        # Credit spread
        risk_neutral_prob = norm.cdf(-d2)
        credit_spread = -np.log(1 - risk_neutral_prob) / time_to_maturity
        
        # Equity value (call option on assets)
        equity_value = (asset_value*norm.cdf(d1) - 
                       debt_value*np.exp(-risk_free_rate*time_to_maturity)*norm.cdf(d2))
        
        # Debt value
        debt_market_value = asset_value - equity_value
        
        return {
            'probability_of_default': prob_default,
            'credit_spread': credit_spread,
            'equity_value': equity_value,
            'debt_market_value': debt_market_value,
            'distance_to_default': d2
        }
    
    def portfolio_risk_metrics(self, returns_matrix, confidence_level=0.05):
        """
        Advanced portfolio risk analytics
        """
        # Portfolio returns (assuming equal weights for simplicity)
        n_assets = returns_matrix.shape[1]
        weights = np.ones(n_assets) / n_assets
        portfolio_returns = np.dot(returns_matrix, weights)
        
        # Value at Risk (VaR)
        var_parametric = np.percentile(portfolio_returns, confidence_level * 100)
        var_historical = np.percentile(portfolio_returns, confidence_level * 100)
        
        # Conditional VaR (Expected Shortfall)
        cvar = np.mean(portfolio_returns[portfolio_returns <= var_historical])
        
        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns)
        
        # Sharpe Ratio
        excess_returns = portfolio_returns - self.risk_free_rate/252
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
        # Sortino Ratio
        downside_returns = excess_returns[excess_returns < 0]
        sortino_ratio = (np.mean(excess_returns) / 
                        np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0)
        
        # Calmar Ratio
        calmar_ratio = np.mean(excess_returns) * 252 / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'var_5_percent': var_historical,
            'cvar_5_percent': cvar,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'volatility_annualized': np.std(portfolio_returns) * np.sqrt(252)
        }
    
    def regime_switching_model(self, returns, n_regimes=2):
        """
        Markov regime switching model for market states
        """
        from sklearn.mixture import GaussianMixture
        
        # Fit Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        regime_probs = gmm.fit_predict(returns.reshape(-1, 1))
        
        # Calculate regime statistics
        regimes = {}
        for i in range(n_regimes):
            regime_returns = returns[regime_probs == i]
            regimes[f'regime_{i}'] = {
                'mean_return': np.mean(regime_returns),
                'volatility': np.std(regime_returns),
                'probability': np.sum(regime_probs == i) / len(regime_probs),
                'duration': np.mean(np.diff(np.where(np.diff(regime_probs == i))[0])) if len(np.where(np.diff(regime_probs == i))[0]) > 1 else len(regime_returns)
            }
        
        # Transition matrix
        transition_matrix = np.zeros((n_regimes, n_regimes))
        for i in range(len(regime_probs) - 1):
            transition_matrix[regime_probs[i], regime_probs[i+1]] += 1
        
        # Normalize rows
        for i in range(n_regimes):
            if np.sum(transition_matrix[i, :]) > 0:
                transition_matrix[i, :] /= np.sum(transition_matrix[i, :])
        
        return {
            'regimes': regimes,
            'regime_probabilities': regime_probs,
            'transition_matrix': transition_matrix
        }
    
    def copula_risk_modeling(self, returns_matrix):
        """
        Copula-based dependency modeling for portfolio risk
        """
        from scipy.stats import spearmanr, kendalltau
        
        n_assets = returns_matrix.shape[1]
        n_obs = returns_matrix.shape[0]
        
        # Convert to uniform margins using empirical CDFs
        uniform_data = np.zeros_like(returns_matrix)
        for i in range(n_assets):
            ranks = stats.rankdata(returns_matrix[:, i])
            uniform_data[:, i] = ranks / (n_obs + 1)
        
        # Correlation measures
        pearson_corr = np.corrcoef(returns_matrix.T)
        spearman_corr, _ = spearmanr(returns_matrix)
        kendall_corr, _ = kendalltau(returns_matrix)
        
        # Tail dependence (simplified estimation)
        tail_dependence = {}
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                # Upper tail dependence
                threshold = 0.95
                upper_exceedances = ((uniform_data[:, i] > threshold) & 
                                   (uniform_data[:, j] > threshold))
                upper_tail_dep = np.sum(upper_exceedances) / np.sum(uniform_data[:, i] > threshold)
                
                # Lower tail dependence
                threshold = 0.05
                lower_exceedances = ((uniform_data[:, i] < threshold) & 
                                   (uniform_data[:, j] < threshold))
                lower_tail_dep = np.sum(lower_exceedances) / np.sum(uniform_data[:, i] < threshold)
                
                tail_dependence[f'assets_{i}_{j}'] = {
                    'upper_tail': upper_tail_dep,
                    'lower_tail': lower_tail_dep
                }
        
        return {
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr,
            'kendall_correlation': kendall_corr,
            'tail_dependence': tail_dependence,
            'uniform_margins': uniform_data
        }
    
    def stress_testing_scenarios(self, portfolio_value, scenario_shocks):
        """
        Advanced stress testing with multiple scenarios
        """
        base_value = portfolio_value
        stress_results = {}
        
        scenarios = {
            'financial_crisis': {
                'equity_shock': -0.35,
                'bond_shock': -0.15,
                'fx_shock': 0.25,
                'volatility_shock': 2.0
            },
            'inflation_surge': {
                'equity_shock': -0.20,
                'bond_shock': -0.30,
                'fx_shock': 0.15,
                'volatility_shock': 1.5
            },
            'geopolitical_crisis': {
                'equity_shock': -0.25,
                'bond_shock': -0.10,
                'fx_shock': 0.20,
                'volatility_shock': 1.8
            },
            'central_bank_shock': {
                'equity_shock': -0.15,
                'bond_shock': -0.25,
                'fx_shock': 0.10,
                'volatility_shock': 1.3
            }
        }
        
        for scenario_name, shocks in scenarios.items():
            # Apply shocks (simplified calculation)
            stressed_value = base_value * (1 + shocks['equity_shock'] * 0.6 +  # 60% equity
                                         shocks['bond_shock'] * 0.3 +        # 30% bonds
                                         shocks['fx_shock'] * 0.1)           # 10% FX
            
            value_change = stressed_value - base_value
            percentage_change = (value_change / base_value) * 100
            
            stress_results[scenario_name] = {
                'portfolio_value': stressed_value,
                'value_change': value_change,
                'percentage_change': percentage_change,
                'shocks_applied': shocks
            }
        
        return stress_results
    
    def economic_capital_calculation(self, portfolio_returns, confidence_level=0.999):
        """
        Economic capital calculation using various methods
        """
        # Parametric approach (assuming normal distribution)
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)
        var_normal = stats.norm.ppf(1 - confidence_level, mean_return, std_return)
        
        # Historical simulation
        var_historical = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        
        # Monte Carlo simulation
        simulated_returns = np.random.normal(mean_return, std_return, 10000)
        var_monte_carlo = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        
        # Expected return
        expected_return = mean_return
        
        # Economic capital = Unexpected loss at confidence level
        ec_parametric = abs(var_normal - expected_return)
        ec_historical = abs(var_historical - expected_return)
        ec_monte_carlo = abs(var_monte_carlo - expected_return)
        
        return {
            'economic_capital_parametric': ec_parametric,
            'economic_capital_historical': ec_historical,
            'economic_capital_monte_carlo': ec_monte_carlo,
            'var_parametric': var_normal,
            'var_historical': var_historical,
            'var_monte_carlo': var_monte_carlo
        }

class QuantumFinancialEngineering:
    """
    Quantum-inspired financial engineering and modeling
    """
    
    def __init__(self):
        self.quantum_states = {}
        self.entanglement_matrix = None
        
    def quantum_portfolio_optimization(self, expected_returns, covariance_matrix, 
                                     risk_aversion=1.0):
        """
        Quantum-inspired portfolio optimization using superposition principles
        """
        n_assets = len(expected_returns)
        
        # Create quantum superposition of portfolio weights
        quantum_portfolios = []
        for _ in range(1000):  # Generate quantum portfolio states
            # Random quantum state
            weights = np.random.dirichlet(np.ones(n_assets))
            
            # Apply quantum interference
            interference = np.sin(np.pi * weights) * 0.1
            weights = weights + interference
            weights = np.abs(weights) / np.sum(np.abs(weights))  # Normalize
            
            # Calculate portfolio metrics
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            utility = portfolio_return - 0.5 * risk_aversion * portfolio_risk**2
            
            quantum_portfolios.append({
                'weights': weights,
                'return': portfolio_return,
                'risk': portfolio_risk,
                'utility': utility
            })
        
        # Quantum collapse - select optimal portfolio
        best_portfolio = max(quantum_portfolios, key=lambda x: x['utility'])
        
        # Calculate quantum entanglement between assets
        correlation_matrix = covariance_matrix / np.sqrt(
            np.outer(np.diag(covariance_matrix), np.diag(covariance_matrix))
        )
        entanglement_strength = np.abs(correlation_matrix)
        
        return {
            'optimal_weights': best_portfolio['weights'],
            'expected_return': best_portfolio['return'],
            'expected_risk': best_portfolio['risk'],
            'utility': best_portfolio['utility'],
            'entanglement_matrix': entanglement_strength,
            'quantum_portfolios': quantum_portfolios[:100]  # Return sample for analysis
        }
    
    def quantum_option_pricing(self, spot_price, strike_price, time_to_expiry, 
                             risk_free_rate, volatility):
        """
        Quantum superposition approach to option pricing
        """
        # Create quantum superposition of price paths
        n_paths = 1000
        n_steps = 100
        dt = time_to_expiry / n_steps
        
        quantum_paths = []
        
        for _ in range(n_paths):
            path = [spot_price]
            
            for step in range(n_steps):
                # Quantum random walk with interference
                z1 = np.random.normal(0, 1)
                z2 = np.random.normal(0, 1)
                
                # Quantum interference between random variables
                quantum_z = (z1 + z2) / np.sqrt(2)  # Superposition
                
                # Add quantum tunneling effect
                tunneling_prob = 0.01
                if np.random.random() < tunneling_prob:
                    quantum_z *= 2  # Quantum tunneling through barriers
                
                # Price evolution with quantum effects
                next_price = path[-1] * np.exp(
                    (risk_free_rate - 0.5 * volatility**2) * dt + 
                    volatility * np.sqrt(dt) * quantum_z
                )
                path.append(next_price)
            
            quantum_paths.append(path)
        
        # Calculate quantum option values
        final_prices = [path[-1] for path in quantum_paths]
        call_payoffs = [max(price - strike_price, 0) for price in final_prices]
        put_payoffs = [max(strike_price - price, 0) for price in final_prices]
        
        # Quantum measurement (collapse to classical value)
        call_price = np.exp(-risk_free_rate * time_to_expiry) * np.mean(call_payoffs)
        put_price = np.exp(-risk_free_rate * time_to_expiry) * np.mean(put_payoffs)
        
        # Quantum uncertainty
        call_uncertainty = np.std(call_payoffs) / np.sqrt(n_paths)
        put_uncertainty = np.std(put_payoffs) / np.sqrt(n_paths)
        
        return {
            'call_price': call_price,
            'put_price': put_price,
            'call_uncertainty': call_uncertainty,
            'put_uncertainty': put_uncertainty,
            'quantum_paths': quantum_paths[:50],  # Sample paths
            'quantum_distribution': final_prices
        }
    
    def market_regime_quantum_detection(self, price_series):
        """
        Quantum-inspired market regime detection
        """
        # Create quantum states for different market regimes
        returns = np.diff(np.log(price_series))
        
        # Define quantum basis states
        bull_state = np.array([1, 0, 0])  # Bull market
        bear_state = np.array([0, 1, 0])  # Bear market
        sideways_state = np.array([0, 0, 1])  # Sideways market
        
        regime_probabilities = []
        
        for i in range(20, len(returns)):  # Use rolling window
            window_returns = returns[i-20:i]
            
            # Calculate quantum features
            mean_return = np.mean(window_returns)
            volatility = np.std(window_returns)
            skewness = stats.skew(window_returns)
            
            # Quantum measurement probabilities
            bull_prob = max(0, mean_return + 0.5) * (1 - volatility/0.1)
            bear_prob = max(0, -mean_return + 0.5) * (1 - volatility/0.1)
            sideways_prob = 1 - abs(mean_return) * 10
            
            # Normalize to quantum probabilities
            total_prob = bull_prob + bear_prob + sideways_prob
            if total_prob > 0:
                bull_prob /= total_prob
                bear_prob /= total_prob
                sideways_prob /= total_prob
            
            # Quantum superposition state
            quantum_state = (bull_prob * bull_state + 
                           bear_prob * bear_state + 
                           sideways_prob * sideways_state)
            
            regime_probabilities.append({
                'bull_prob': bull_prob,
                'bear_prob': bear_prob,
                'sideways_prob': sideways_prob,
                'quantum_state': quantum_state,
                'dominant_regime': ['bull', 'bear', 'sideways'][np.argmax([bull_prob, bear_prob, sideways_prob])]
            })
        
        return regime_probabilities

def generate_synthetic_economic_data():
    """
    Generate synthetic economic data for advanced modeling
    """
    np.random.seed(42)
    n_observations = 252 * 5  # 5 years of daily data
    
    # Generate correlated economic variables
    correlation_matrix = np.array([
        [1.00, 0.30, -0.60, 0.40],  # GDP growth
        [0.30, 1.00, -0.20, 0.70],  # Inflation
        [-0.60, -0.20, 1.00, -0.30], # Interest rates
        [0.40, 0.70, -0.30, 1.00]   # Exchange rate
    ])
    
    # Generate multivariate normal data
    mean_values = [0.05/252, 0.03/252, 0.02/252, 0.001]  # Daily values
    cov_matrix = correlation_matrix * 0.01  # Scale covariances
    
    economic_data = np.random.multivariate_normal(mean_values, cov_matrix, n_observations)
    
    # Convert to levels (cumulative)
    gdp_growth = np.cumsum(economic_data[:, 0]) + 1
    inflation = np.cumsum(economic_data[:, 1]) + 0.05
    interest_rates = np.cumsum(economic_data[:, 2]) + 0.03
    exchange_rates = np.cumsum(economic_data[:, 3]) + 100
    
    dates = pd.date_range(start='2019-01-01', periods=n_observations, freq='D')
    
    return pd.DataFrame({
        'Date': dates,
        'GDP_Growth': gdp_growth,
        'Inflation': inflation,
        'Interest_Rates': interest_rates,
        'Exchange_Rates': exchange_rates,
        'Stock_Index': 1000 * np.cumprod(1 + np.random.normal(0.0008, 0.02, n_observations))
    })

if __name__ == "__main__":
    # Example usage
    print("Advanced Financial Derivatives and Economic Instruments System")
    print("=" * 60)
    
    # Initialize the system
    financial_system = AdvancedFinancialInstruments()
    quantum_system = QuantumFinancialEngineering()
    
    # Generate sample data
    economic_data = generate_synthetic_economic_data()
    
    print(f"Generated {len(economic_data)} observations of economic data")
    print("System ready for advanced financial modeling and analysis.")
