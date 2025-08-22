"""
🛡️ Enhanced Risk Analysis & Portfolio Optimization
=================================================

An advanced suite for quantitative risk analysis, stress testing, scenario modeling,
and sophisticated portfolio optimization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🛡️ Enhanced Risk Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional and modern look
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #434343 0%, #000000 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    
    .risk-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .risk-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(255,255,255,0.1);
        border-color: #00ff88;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(0,0,0,0.3);
        color: white;
        border-radius: 10px 10px 0 0;
    }
</style>
""", unsafe_allow_html=True)

def generate_asset_returns(num_assets=5, num_days=500):
    """Generate sample daily returns for multiple assets"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=num_days, freq='D')
    asset_names = [f'Asset_{chr(65+i)}' for i in range(num_assets)]
    
    returns = pd.DataFrame(
        np.random.normal(0.0005, 0.015, (num_days, num_assets)),
        columns=asset_names,
        index=dates
    )
    
    # Add some correlation
    returns['Asset_B'] += returns['Asset_A'] * 0.3
    returns['Asset_D'] -= returns['Asset_C'] * 0.2
    
    return returns

def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    """Calculate portfolio performance metrics"""
    returns = np.sum(mean_returns * weights) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std_dev

def optimize_portfolio(mean_returns, cov_matrix, target_return):
    """Find the optimal portfolio for a given target return (minimum variance)"""
    num_assets = len(mean_returns)
    
    def portfolio_variance(weights):
        return calculate_portfolio_performance(weights, mean_returns, cov_matrix)[1]**2
    
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: calculate_portfolio_performance(x, mean_returns, cov_matrix)[0] - target_return}
    )
    
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1. / num_assets,]
    
    result = minimize(portfolio_variance, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def create_efficient_frontier_chart(mean_returns, cov_matrix, risk_free_rate=0.02):
    """Generate and plot the efficient frontier and Capital Allocation Line (CAL)"""
    num_assets = len(mean_returns)
    
    # Generate random portfolios
    results = np.zeros((3, 10000))
    weights_record = []
    for i in range(10000):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        portfolio_return, portfolio_std_dev = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
        
        results[0,i] = portfolio_return
        results[1,i] = portfolio_std_dev
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev # Sharpe Ratio
    
    # Find the portfolio with the maximum Sharpe Ratio
    max_sharpe_idx = np.argmax(results[2])
    max_sharpe_return, max_sharpe_std = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_weights = weights_record[max_sharpe_idx]
    
    # Generate the efficient frontier
    target_returns = np.linspace(results[0].min(), results[0].max(), 100)
    efficient_portfolios = [optimize_portfolio(mean_returns, cov_matrix, tr) for tr in target_returns]
    efficient_returns = [calculate_portfolio_performance(w, mean_returns, cov_matrix)[0] for w in efficient_portfolios]
    efficient_risks = [calculate_portfolio_performance(w, mean_returns, cov_matrix)[1] for w in efficient_portfolios]
    
    fig = go.Figure()
    
    # Random portfolios
    fig.add_trace(go.Scatter(
        x=results[1,:], y=results[0,:], mode='markers',
        marker=dict(
            color=results[2,:], # Color by Sharpe Ratio
            showscale=True,
            colorscale='Viridis',
            size=7,
            colorbar=dict(title='Sharpe Ratio')
        ),
        name='Random Portfolios'
    ))
    
    # Efficient Frontier
    fig.add_trace(go.Scatter(
        x=efficient_risks, y=efficient_returns, mode='lines',
        line=dict(color='#00ff88', width=4),
        name='Efficient Frontier'
    ))
    
    # Max Sharpe Ratio portfolio
    fig.add_trace(go.Scatter(
        x=[max_sharpe_std], y=[max_sharpe_return], mode='markers',
        marker=dict(color='#ff6b6b', size=15, symbol='star'),
        name='Optimal Portfolio (Max Sharpe)'
    ))
    
    fig.update_layout(
        title='Investment Portfolio Efficient Frontier',
        xaxis_title='Volatility (Standard Deviation)',
        yaxis_title='Expected Annual Return',
        template='plotly_dark',
        height=600
    )
    
    return fig, max_sharpe_weights

def create_stress_test_chart(returns, scenarios):
    """Visualize the impact of stress test scenarios on the portfolio"""
    base_performance = returns.mean() * 100
    
    scenario_names = list(scenarios.keys())
    impacts = []
    
    for scenario, shock in scenarios.items():
        stressed_returns = returns + shock
        stressed_performance = stressed_returns.mean() * 100
        impacts.append(stressed_performance)
        
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=scenario_names,
        y=impacts,
        name='Stressed Performance',
        marker_color=['#ff6b6b', '#f093fb', '#00d4ff']
    ))
    
    fig.add_hline(y=base_performance, line_dash="dash", line_color="white",
                  annotation_text="Base Performance", annotation_position="bottom right")
    
    fig.update_layout(
        title='Portfolio Stress Test Results',
        yaxis_title='Average Daily Return (%)',
        template='plotly_dark'
    )
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Enhanced Risk Analysis & Portfolio Optimization</h1>
        <p>A Quantitative Approach to Managing Financial Risk</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    asset_returns = generate_asset_returns()
    
    # Sidebar controls
    st.sidebar.header("⚙️ Analysis Configuration")
    num_assets = st.sidebar.slider("Number of Assets", 3, 10, 5)
    if st.sidebar.button("🔄 Regenerate Asset Data"):
        asset_returns = generate_asset_returns(num_assets)
    
    risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", 0.0, 5.0, 2.0) / 100
    
    tabs = st.tabs(["📈 Portfolio Optimization", "📉 Risk Analysis", "🔬 Stress Testing"])
    
    # Calculations
    mean_returns = asset_returns.mean()
    cov_matrix = asset_returns.cov()
    
    with tabs[0]:
        st.markdown("### 🚀 Efficient Frontier & Optimal Portfolio")
        
        fig_frontier, optimal_weights = create_efficient_frontier_chart(mean_returns, cov_matrix, risk_free_rate)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(fig_frontier, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Optimal Portfolio Allocation")
            st.markdown("This portfolio maximizes the Sharpe Ratio, offering the best return for its level of risk.")
            
            optimal_df = pd.DataFrame({
                'Asset': asset_returns.columns,
                'Optimal Weight': [f'{w*100:.2f}%' for w in optimal_weights]
            })
            st.dataframe(optimal_df, use_container_width=True)
            
            # Pie chart of allocation
            fig_pie = px.pie(
                optimal_df, values=optimal_weights, names='Asset',
                title='Optimal Asset Allocation',
                template='plotly_dark',
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
    with tabs[1]:
        st.markdown("### 📊 Quantitative Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📉 Value at Risk (VaR) & CVaR")
            
            confidence_level = st.slider("Confidence Level (%)", 90, 99, 95)
            
            portfolio_returns = asset_returns.dot(optimal_weights)
            var = np.percentile(portfolio_returns, 100 - confidence_level)
            cvar = portfolio_returns[portfolio_returns <= var].mean()
            
            st.metric(f"Value at Risk (VaR) at {confidence_level}%", f"{var*100:.3f}%")
            st.metric(f"Conditional VaR (CVaR) at {confidence_level}%", f"{cvar*100:.3f}%")
            
            st.markdown("""
            <div class="risk-card">
                <p><strong>VaR:</strong> The maximum expected loss over a single day with a certain confidence level.</p>
                <p><strong>CVaR:</strong> The expected loss if that VaR threshold is crossed (average loss in the worst cases).</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 🔗 Asset Correlation Matrix")
            corr_matrix = asset_returns.corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title='Asset Correlation Heatmap'
            )
            fig_corr.update_layout(template='plotly_dark')
            st.plotly_chart(fig_corr, use_container_width=True)
            
    with tabs[2]:
        st.markdown("### 🔬 Scenario Analysis & Stress Testing")
        
        st.markdown("Simulate the impact of market shocks on your optimal portfolio.")
        
        # Define scenarios
        scenarios = {
            'Market Crash': -0.05, # 5% daily drop
            'Interest Rate Hike': 0.01, # 1% positive shock to less risky assets
            'Geopolitical Shock': -0.03 # 3% drop with increased volatility
        }
        
        # Apply shocks
        stressed_returns_data = {}
        stressed_returns_data['Base'] = asset_returns.dot(optimal_weights)
        stressed_returns_data['Market Crash'] = (asset_returns - 0.05).dot(optimal_weights)
        stressed_returns_data['Rate Hike'] = (asset_returns + pd.Series([0.01, 0.01, -0.005, -0.005, 0.015], index=asset_returns.columns)).dot(optimal_weights)
        stressed_returns_data['Geopolitical Shock'] = (asset_returns * 1.5 - 0.03).dot(optimal_weights)
        
        stressed_df = pd.DataFrame(stressed_returns_data)
        
        fig_stress = go.Figure()
        for col in stressed_df.columns:
            fig_stress.add_trace(go.Box(y=stressed_df[col], name=col))
        
        fig_stress.update_layout(
            title='Portfolio Returns Distribution Under Stress Scenarios',
            yaxis_title='Daily Portfolio Return',
            template='plotly_dark',
            height=500
        )
        st.plotly_chart(fig_stress, use_container_width=True)
        
        st.markdown("#### 📋 Scenario Impact Summary")
        summary = stressed_df.describe().transpose()
        st.dataframe(summary, use_container_width=True)

if __name__ == "__main__":
    main()
