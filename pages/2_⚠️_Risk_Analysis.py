"""
Risk Analysis Dashboard
======================

Comprehensive risk assessment page featuring VaR, CVaR, Monte Carlo simulations,
stress testing, and portfolio optimization with interactive controls.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import risk models and plotting utilities
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.risk import (
    VaRCalculator, MonteCarloSimulator, CreditRiskModel, 
    StressTesting, PortfolioRiskOptimizer, create_comprehensive_risk_report
)
from src.utils.plotting import (
    create_risk_dashboard, create_monte_carlo_paths, create_distribution_plot,
    create_correlation_heatmap, create_time_series_plot
)

# Page configuration
st.set_page_config(
    page_title="⚠️ Risk Analysis",
    page_icon="⚠️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .risk-metric {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(255,107,107,0.3);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
        border-left: 5px solid #f57c00;
    }
    
    .scenario-card {
        background: linear-gradient(135deg, #42a5f5 0%, #1e88e5 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 3px 10px rgba(66,165,245,0.3);
    }
    
    .monte-carlo-summary {
        background: linear-gradient(135deg, #ab47bc 0%, #8e24aa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    
    .risk-level-high {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    
    .risk-level-medium {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    
    .risk-level-low {
        background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def render_risk_sidebar():
    """Render risk analysis configuration sidebar"""
    st.sidebar.markdown("## ⚠️ Risk Analysis Configuration")
    
    # Risk analysis type
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        [
            "📊 Value at Risk (VaR)",
            "🎲 Monte Carlo Simulation", 
            "⚡ Stress Testing",
            "💼 Portfolio Optimization",
            "🔍 Credit Risk Assessment"
        ]
    )
    
    # Common parameters
    st.sidebar.markdown("### ⚙️ Parameters")
    
    confidence_levels = st.sidebar.multiselect(
        "Confidence Levels",
        [0.90, 0.95, 0.99],
        default=[0.95, 0.99],
        format_func=lambda x: f"{x*100:.0f}%"
    )
    
    time_horizon = st.sidebar.slider(
        "Time Horizon (days)",
        min_value=1, max_value=365, value=252,
        help="Time horizon for risk calculations"
    )
    
    # Method-specific parameters
    if "VaR" in analysis_type:
        var_method = st.sidebar.selectbox(
            "VaR Method",
            ["Historical Simulation", "Parametric (Normal)", "Monte Carlo"],
            help="Method for calculating Value at Risk"
        )
        
        if "Parametric" in var_method:
            distribution = st.sidebar.selectbox(
                "Distribution",
                ["normal", "t", "skewnorm"],
                help="Statistical distribution assumption"
            )
        else:
            distribution = "normal"
    
    elif "Monte Carlo" in analysis_type:
        n_simulations = st.sidebar.slider(
            "Number of Simulations",
            min_value=1000, max_value=50000, value=10000, step=1000,
            help="More simulations = higher accuracy but slower computation"
        )
        
        simulation_type = st.sidebar.selectbox(
            "Simulation Type",
            ["Geometric Brownian Motion", "Jump Diffusion", "Economic Scenarios"],
            help="Type of stochastic process to simulate"
        )
    
    elif "Stress" in analysis_type:
        stress_scenarios = st.sidebar.multiselect(
            "Stress Scenarios",
            [
                "Mild Recession",
                "Severe Recession", 
                "Financial Crisis",
                "Inflation Shock",
                "Market Crash"
            ],
            default=["Mild Recession", "Financial Crisis"]
        )
    
    elif "Portfolio" in analysis_type:
        optimization_objective = st.sidebar.selectbox(
            "Optimization Objective",
            ["Maximize Sharpe Ratio", "Minimize Variance", "Maximize Return"],
            help="Portfolio optimization goal"
        )
        
        max_weight = st.sidebar.slider(
            "Maximum Asset Weight",
            min_value=0.1, max_value=1.0, value=0.4, step=0.05,
            help="Maximum allocation to any single asset"
        )
    
    # Return configuration dictionary
    config = {
        'analysis_type': analysis_type,
        'confidence_levels': confidence_levels,
        'time_horizon': time_horizon
    }
    
    if "VaR" in analysis_type:
        config.update({
            'var_method': var_method.lower().replace(' ', '_').replace('(', '').replace(')', ''),
            'distribution': distribution if "Parametric" in var_method else None
        })
    elif "Monte Carlo" in analysis_type:
        config.update({
            'n_simulations': n_simulations,
            'simulation_type': simulation_type.lower().replace(' ', '_')
        })
    elif "Stress" in analysis_type:
        config['stress_scenarios'] = stress_scenarios
    elif "Portfolio" in analysis_type:
        config.update({
            'optimization_objective': optimization_objective.lower().replace(' ', '_'),
            'max_weight': max_weight
        })
    
    return config

def generate_sample_returns_data(n_assets=5, n_periods=252):
    """Generate sample financial returns data"""
    np.random.seed(42)
    
    # Asset names
    asset_names = [f"Asset_{i+1}" for i in range(n_assets)]
    
    # Generate correlated returns
    correlation_matrix = np.random.uniform(0.1, 0.6, (n_assets, n_assets))
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1.0)
    
    # Generate returns
    mean_returns = np.random.uniform(0.0005, 0.002, n_assets)  # Daily returns
    volatilities = np.random.uniform(0.01, 0.03, n_assets)  # Daily volatilities
    
    # Multivariate normal returns
    returns = np.random.multivariate_normal(
        mean_returns, 
        np.outer(volatilities, volatilities) * correlation_matrix,
        size=n_periods
    )
    
    # Create DataFrame
    dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='D')
    returns_df = pd.DataFrame(returns, index=dates, columns=asset_names)
    
    return returns_df

def main():
    """Main risk analysis page"""
    
    # Header
    st.markdown("# ⚠️ Advanced Risk Analysis")
    st.markdown("""
    <div class="warning-box">
        <h3>🚨 Comprehensive Risk Assessment Platform</h3>
        <p>• Value at Risk (VaR) and Conditional VaR calculations with multiple methods</p>
        <p>• Monte Carlo simulations for scenario analysis and stress testing</p>
        <p>• Portfolio optimization and credit risk modeling</p>
        <p>• Real-time risk monitoring and early warning systems</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get configuration
    config = render_risk_sidebar()
    
    # Data input section
    st.markdown("## 📊 Data Input")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload returns/price data",
            type=['csv', 'xlsx'],
            help="Upload a file with time series of returns or prices. Should contain date column and numeric return/price columns."
        )
    
    with col2:
        use_sample = st.button("📈 Generate Sample Data", help="Create sample financial returns data for demonstration")
    
    # Load data
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            # Try to detect date column
            date_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                data[date_cols[0]] = pd.to_datetime(data[date_cols[0]])
                data = data.set_index(date_cols[0])
            
            st.success(f"✅ Data loaded: {len(data)} periods, {len(data.columns)} assets")
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            data = None
    
    elif use_sample:
        data = generate_sample_returns_data()
        st.success("✅ Sample returns data generated")
    
    else:
        data = None
        st.info("👆 Please upload returns data or generate sample data")
    
    if data is not None:
        # Data preview
        st.markdown("### 📋 Data Preview")
        
        tab1, tab2 = st.tabs(["📊 Raw Data", "📈 Visualization"])
        
        with tab1:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.dataframe(data.head(10), use_container_width=True)
            
            with col2:
                st.write(f"**Periods:** {len(data):,}")
                st.write(f"**Assets:** {len(data.columns)}")
                st.write(f"**Date Range:** {data.index.min()} to {data.index.max()}")
                
                # Summary statistics
                st.write("**Mean Returns:**")
                for col in data.columns[:3]:  # Show first 3
                    mean_ret = data[col].mean() * 252 * 100  # Annualized %
                    st.write(f"• {col}: {mean_ret:.1f}%")
        
        with tab2:
            # Visualization
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 0:
                viz_cols = st.multiselect(
                    "Select assets to visualize",
                    numeric_cols,
                    default=numeric_cols[:min(3, len(numeric_cols))]
                )
                
                if viz_cols:
                    # Cumulative returns
                    cumulative_returns = (1 + data[viz_cols]).cumprod()
                    
                    fig = create_time_series_plot(
                        cumulative_returns,
                        viz_cols,
                        title="Cumulative Returns",
                        y_title="Cumulative Return",
                        show_trend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Risk Analysis
        st.markdown("## 🎯 Risk Analysis")
        
        if st.button("🚀 Run Risk Analysis", type="primary"):
            
            with st.spinner(f"🔄 Running {config['analysis_type']}..."):
                
                try:
                    numeric_data = data.select_dtypes(include=[np.number])
                    
                    if "VaR" in config['analysis_type']:
                        # Value at Risk Analysis
                        st.markdown("### 📊 Value at Risk (VaR) Analysis")
                        
                        var_results = {}
                        
                        for confidence in config['confidence_levels']:
                            if config['var_method'] == 'historical_simulation':
                                var_calc = VaRCalculator(method='historical', confidence_level=confidence)
                            elif config['var_method'] == 'parametric_normal':
                                var_calc = VaRCalculator(method='parametric', confidence_level=confidence)
                            else:  # Monte Carlo
                                var_calc = VaRCalculator(method='monte_carlo', confidence_level=confidence)
                            
                            # Calculate VaR for each asset
                            for col in numeric_data.columns[:5]:  # Limit to 5 assets
                                returns = numeric_data[col].dropna().values
                                if len(returns) > 10:
                                    metrics = var_calc.calculate_risk_metrics(returns)
                                    var_results[f'{col}_{int(confidence*100)}%'] = metrics
                        
                        # Display VaR results
                        if var_results:
                            # Risk metrics table
                            metrics_data = []
                            for key, metrics in var_results.items():
                                asset, conf = key.rsplit('_', 1)
                                metrics_data.append([
                                    asset, conf,
                                    f"{metrics['VaR']:.4f}",
                                    f"{metrics['CVaR']:.4f}",
                                    f"{metrics['Volatility']:.4f}",
                                    f"{metrics['Sharpe_Ratio']:.4f}"
                                ])
                            
                            metrics_df = pd.DataFrame(
                                metrics_data,
                                columns=['Asset', 'Confidence', 'VaR', 'CVaR', 'Volatility', 'Sharpe Ratio']
                            )
                            
                            st.dataframe(metrics_df, use_container_width=True)
                            
                            # Risk level assessment
                            first_asset = list(var_results.keys())[0]
                            first_var = var_results[first_asset]['VaR']
                            
                            if abs(first_var) > 0.05:  # 5% threshold
                                risk_class = "risk-level-high"
                                risk_text = "🔴 HIGH RISK DETECTED"
                            elif abs(first_var) > 0.02:  # 2% threshold
                                risk_class = "risk-level-medium"
                                risk_text = "🟡 MEDIUM RISK LEVEL"
                            else:
                                risk_class = "risk-level-low"
                                risk_text = "🟢 LOW RISK LEVEL"
                            
                            st.markdown(f'<div class="{risk_class}">{risk_text}</div>', unsafe_allow_html=True)
                    
                    elif "Monte Carlo" in config['analysis_type']:
                        # Monte Carlo Simulation
                        st.markdown("### 🎲 Monte Carlo Simulation")
                        
                        simulator = MonteCarloSimulator(
                            n_simulations=config['n_simulations'],
                            time_horizon=config['time_horizon']
                        )
                        
                        if config['simulation_type'] == 'geometric_brownian_motion':
                            # Use first asset for demonstration
                            first_asset = numeric_data.columns[0]
                            returns = numeric_data[first_asset].dropna()
                            
                            mu = returns.mean() * 252  # Annualized
                            sigma = returns.std() * np.sqrt(252)  # Annualized
                            S0 = 100  # Starting price
                            
                            simulation_results = simulator.geometric_brownian_motion(S0, mu, sigma)
                            
                        elif config['simulation_type'] == 'jump_diffusion':
                            first_asset = numeric_data.columns[0]
                            returns = numeric_data[first_asset].dropna()
                            
                            mu = returns.mean() * 252
                            sigma = returns.std() * np.sqrt(252)
                            S0 = 100
                            
                            simulation_results = simulator.jump_diffusion(
                                S0, mu, sigma,
                                jump_intensity=10,  # 10 jumps per year
                                jump_mean=-0.02,    # -2% average jump
                                jump_std=0.05       # 5% jump volatility
                            )
                        
                        else:  # Economic scenarios
                            simulation_results = simulator.economic_scenario_simulation(
                                gdp_params=(2.0, 0.5),      # 2% growth, 0.5% volatility
                                inflation_params=(2.5, 0.3), # 2.5% inflation, 0.3% volatility
                                interest_rate_params=(3.0, 0.2) # 3% rate, 0.2% volatility
                            )
                        
                        # Display simulation results
                        if simulation_results is not None and not simulation_results.empty:
                            
                            # Summary statistics
                            stats = simulator.calculate_scenario_statistics()
                            
                            if stats:
                                st.markdown("#### 📈 Simulation Summary")
                                
                                for var_name, var_stats in list(stats.items())[:3]:  # Show first 3 variables
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    final_stats = var_stats.iloc[-1]  # Final period stats
                                    
                                    with col1:
                                        st.metric(
                                            f"{var_name} - Mean",
                                            f"{final_stats['Mean']:.2f}"
                                        )
                                    with col2:
                                        st.metric(
                                            f"{var_name} - VaR 5%",
                                            f"{final_stats['VaR_5%']:.2f}"
                                        )
                                    with col3:
                                        st.metric(
                                            f"{var_name} - VaR 1%",
                                            f"{final_stats['VaR_1%']:.2f}"
                                        )
                                    with col4:
                                        st.metric(
                                            f"{var_name} - Volatility",
                                            f"{final_stats['Std']:.2f}"
                                        )
                            
                            # Monte Carlo paths visualization
                            fig_paths = create_monte_carlo_paths(
                                simulation_results,
                                n_paths_display=100,
                                title="Monte Carlo Simulation Paths"
                            )
                            st.plotly_chart(fig_paths, use_container_width=True)
                            
                            # Distribution of final values
                            final_values = simulation_results.iloc[:, -2]  # Second to last column (excluding 'Simulation')
                            
                            fig_dist = create_distribution_plot(
                                final_values,
                                title="Distribution of Final Values",
                                show_normal=True
                            )
                            st.plotly_chart(fig_dist, use_container_width=True)
                    
                    elif "Stress" in config['analysis_type']:
                        # Stress Testing
                        st.markdown("### ⚡ Stress Testing")
                        
                        stress_tester = StressTesting()
                        stress_tester.create_default_scenarios()
                        
                        # Create stress data
                        first_asset = numeric_data.columns[0]
                        stress_data = pd.DataFrame({
                            'returns': numeric_data[first_asset],
                            'gdp_growth': np.random.normal(0.02, 0.01, len(numeric_data)),
                            'inflation': np.random.normal(0.025, 0.005, len(numeric_data)),
                            'interest_rate': np.random.normal(0.03, 0.002, len(numeric_data))
                        })
                        
                        # Run stress tests
                        var_model = VaRCalculator(method='historical', confidence_level=0.95)
                        stress_results = stress_tester.run_stress_test(
                            stress_data, var_model, config['stress_scenarios']
                        )
                        
                        # Display stress test results
                        if stress_results:
                            st.markdown("#### 🎯 Stress Test Results")
                            
                            stress_data_display = []
                            for scenario, metrics in stress_results.items():
                                stress_data_display.append([
                                    scenario,
                                    f"{metrics.get('VaR', 0):.4f}",
                                    f"{metrics.get('CVaR', 0):.4f}",
                                    f"{metrics.get('Volatility', 0):.4f}",
                                    f"{metrics.get('Max_Drawdown', 0):.4f}"
                                ])
                            
                            stress_df = pd.DataFrame(
                                stress_data_display,
                                columns=['Scenario', 'VaR', 'CVaR', 'Volatility', 'Max Drawdown']
                            )
                            
                            st.dataframe(stress_df, use_container_width=True)
                            
                            # Highlight worst-case scenario
                            worst_var = min(stress_results.items(), key=lambda x: x[1].get('VaR', 0))
                            
                            st.markdown(f"""
                            <div class="scenario-card">
                                <h4>🚨 Worst-Case Scenario: {worst_var[0]}</h4>
                                <p>VaR: {worst_var[1].get('VaR', 0):.4f} | CVaR: {worst_var[1].get('CVaR', 0):.4f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    elif "Portfolio" in config['analysis_type']:
                        # Portfolio Optimization
                        st.markdown("### 💼 Portfolio Optimization")
                        
                        if len(numeric_data.columns) > 1:
                            optimizer = PortfolioRiskOptimizer()
                            
                            # Map optimization objective
                            objective_map = {
                                'maximize_sharpe_ratio': 'sharpe',
                                'minimize_variance': 'min_variance',
                                'maximize_return': 'max_return'
                            }
                            
                            objective = objective_map.get(config['optimization_objective'], 'sharpe')
                            
                            # Optimize portfolio
                            optimization_result = optimizer.optimize_portfolio(
                                numeric_data,
                                objective=objective,
                                constraints={'max_weight': config['max_weight']}
                            )
                            
                            if optimization_result['optimization_success']:
                                st.success("✅ Portfolio optimization completed successfully!")
                                
                                # Display optimal weights
                                weights_df = pd.DataFrame({
                                    'Asset': numeric_data.columns,
                                    'Weight': optimization_result['weights'],
                                    'Weight %': optimization_result['weights'] * 100
                                }).sort_values('Weight', ascending=False)
                                
                                st.markdown("#### 🎯 Optimal Portfolio Weights")
                                st.dataframe(weights_df, use_container_width=True)
                                
                                # Portfolio metrics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric(
                                        "Expected Return",
                                        f"{optimization_result['expected_return']*252*100:.2f}%"
                                    )
                                with col2:
                                    st.metric(
                                        "Volatility",
                                        f"{optimization_result['volatility']*np.sqrt(252)*100:.2f}%"
                                    )
                                with col3:
                                    st.metric(
                                        "Sharpe Ratio",
                                        f"{optimization_result['sharpe_ratio']:.3f}"
                                    )
                                with col4:
                                    diversification = 1 - np.max(optimization_result['weights'])
                                    st.metric(
                                        "Diversification",
                                        f"{diversification*100:.1f}%"
                                    )
                                
                                # Correlation heatmap
                                fig_corr = create_correlation_heatmap(
                                    numeric_data,
                                    title="Asset Correlation Matrix"
                                )
                                st.plotly_chart(fig_corr, use_container_width=True)
                            
                            else:
                                st.error(f"❌ Portfolio optimization failed: {optimization_result.get('error_message', 'Unknown error')}")
                        
                        else:
                            st.warning("⚠️ Portfolio optimization requires at least 2 assets")
                    
                    elif "Credit" in config['analysis_type']:
                        # Credit Risk Assessment
                        st.markdown("### 🔍 Credit Risk Assessment")
                        
                        st.info("""
                        📋 **Credit Risk Modeling Requirements:**
                        
                        For credit risk analysis, please upload a dataset with:
                        • Financial ratios (debt-to-equity, current ratio, etc.)
                        • Historical default indicators (0/1 binary)
                        • Macroeconomic variables
                        
                        This demo shows the framework structure.
                        """)
                        
                        # Demo credit risk framework
                        if len(numeric_data.columns) >= 3:
                            
                            # Create synthetic credit features
                            n_samples = min(len(numeric_data), 1000)
                            
                            synthetic_features = pd.DataFrame({
                                'debt_to_equity': np.random.lognormal(0, 0.5, n_samples),
                                'current_ratio': np.random.lognormal(0.5, 0.3, n_samples),
                                'revenue_growth': np.random.normal(0.05, 0.1, n_samples),
                                'profitability': np.random.normal(0.1, 0.05, n_samples)
                            })
                            
                            # Synthetic default probabilities
                            default_prob = 1 / (1 + np.exp(-(
                                synthetic_features['debt_to_equity'] * 0.5 -
                                synthetic_features['current_ratio'] * 0.3 -
                                synthetic_features['profitability'] * 2 + 
                                np.random.normal(0, 0.1, n_samples)
                            )))
                            
                            synthetic_defaults = (default_prob > 0.5).astype(int)
                            
                            # Display credit metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Default Rate",
                                    f"{synthetic_defaults.mean()*100:.1f}%"
                                )
                            with col2:
                                st.metric(
                                    "Avg Debt/Equity",
                                    f"{synthetic_features['debt_to_equity'].mean():.2f}"
                                )
                            with col3:
                                st.metric(
                                    "Avg Current Ratio",
                                    f"{synthetic_features['current_ratio'].mean():.2f}"
                                )
                            
                            st.markdown("""
                            <div class="monte-carlo-summary">
                                <h4>🎯 Credit Risk Framework</h4>
                                <p>• Machine learning models for default prediction</p>
                                <p>• Feature importance analysis for risk factors</p>
                                <p>• Portfolio-level credit risk aggregation</p>
                                <p>• Regulatory capital calculations</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        else:
                            st.warning("⚠️ Insufficient data for credit risk modeling demo")
                
                except Exception as e:
                    st.error(f"❌ Risk analysis failed: {str(e)}")
                    st.error("Please check your data format and configuration.")

if __name__ == "__main__":
    main()
