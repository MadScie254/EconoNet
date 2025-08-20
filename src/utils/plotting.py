"""
Advanced Plotting Utilities
===========================

Comprehensive visualization functions for economic data, forecasts, and risk analysis.
Built with Plotly for interactive dashboards and publication-ready charts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
from plotly.colors import qualitative, sequential, diverging
import warnings
warnings.filterwarnings('ignore')

# Color schemes
ECONET_COLORS = {
    'primary': '#FF6B6B',
    'secondary': '#4ECDC4', 
    'accent': '#45B7D1',
    'warning': '#FFA726',
    'success': '#66BB6A',
    'danger': '#EF5350',
    'dark': '#2C3E50',
    'light': '#ECF0F1'
}

ECONET_PALETTE = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
    '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43'
]

def create_time_series_plot(data: pd.DataFrame, 
                           columns: List[str],
                           title: str = "Time Series Analysis",
                           y_title: str = "Value",
                           show_trend: bool = True,
                           show_volatility: bool = False,
                           height: int = 600) -> go.Figure:
    """Create interactive time series plot with optional trend and volatility"""
    
    fig = go.Figure()
    
    for i, col in enumerate(columns):
        if col not in data.columns:
            continue
            
        color = ECONET_PALETTE[i % len(ECONET_PALETTE)]
        
        # Main time series
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[col],
            mode='lines',
            name=col,
            line=dict(color=color, width=2),
            hovertemplate=f'<b>{col}</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: %{y:,.2f}<br>' +
                         '<extra></extra>'
        ))
        
        # Add trend line if requested
        if show_trend and len(data[col].dropna()) > 10:
            # Simple linear trend
            valid_data = data[col].dropna()
            x_numeric = np.arange(len(valid_data))
            coeffs = np.polyfit(x_numeric, valid_data.values, 1)
            trend_line = np.polyval(coeffs, x_numeric)
            
            fig.add_trace(go.Scatter(
                x=valid_data.index,
                y=trend_line,
                mode='lines',
                name=f'{col} Trend',
                line=dict(color=color, width=1, dash='dash'),
                opacity=0.7,
                showlegend=False,
                hovertemplate=f'<b>{col} Trend</b><br>' +
                             'Date: %{x}<br>' +
                             'Value: %{y:,.2f}<br>' +
                             '<extra></extra>'
            ))
        
        # Add volatility bands if requested
        if show_volatility and len(data[col].dropna()) > 20:
            rolling_mean = data[col].rolling(window=20).mean()
            rolling_std = data[col].rolling(window=20).std()
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=rolling_mean + 2*rolling_std,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=rolling_mean - 2*rolling_std,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor=f'rgba({color[1:]}, 0.1)',
                name=f'{col} ±2σ',
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20, color=ECONET_COLORS['dark'])
        ),
        xaxis_title="Date",
        yaxis_title=y_title,
        hovermode='x unified',
        showlegend=True,
        height=height,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def create_forecast_plot(historical_data: pd.Series,
                        forecasts: np.ndarray,
                        forecast_dates: pd.DatetimeIndex,
                        confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                        title: str = "Forecast Analysis",
                        model_name: str = "Model") -> go.Figure:
    """Create forecast visualization with confidence intervals"""
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data.values,
        mode='lines',
        name='Historical',
        line=dict(color=ECONET_COLORS['primary'], width=3),
        hovertemplate='<b>Historical</b><br>' +
                     'Date: %{x}<br>' +
                     'Value: %{y:,.2f}<br>' +
                     '<extra></extra>'
    ))
    
    # Forecasts
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecasts,
        mode='lines+markers',
        name=f'{model_name} Forecast',
        line=dict(color=ECONET_COLORS['accent'], width=3, dash='dash'),
        marker=dict(size=6, color=ECONET_COLORS['accent']),
        hovertemplate=f'<b>{model_name} Forecast</b><br>' +
                     'Date: %{x}<br>' +
                     'Value: %{y:,.2f}<br>' +
                     '<extra></extra>'
    ))
    
    # Confidence intervals
    if confidence_intervals is not None:
        lower_bound, upper_bound = confidence_intervals
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=lower_bound,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(69, 183, 209, 0.2)',
            name='95% Confidence',
            hovertemplate='<b>Confidence Interval</b><br>' +
                         'Date: %{x}<br>' +
                         'Upper: %{y:,.2f}<br>' +
                         '<extra></extra>'
        ))
    
    # Add vertical line at forecast start
    if len(historical_data) > 0:
        fig.add_vline(
            x=historical_data.index[-1],
            line_dash="dash",
            line_color="gray",
            annotation_text="Forecast Start",
            annotation_position="top"
        )
    
    # Layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20, color=ECONET_COLORS['dark'])
        ),
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        showlegend=True,
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def create_model_comparison_plot(results: Dict[str, Dict[str, float]],
                               metrics: List[str] = ['MAE', 'RMSE', 'R2', 'MAPE'],
                               title: str = "Model Comparison") -> go.Figure:
    """Create model performance comparison visualization"""
    
    # Prepare data
    models = list(results.keys())
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=metrics,
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    for i, metric in enumerate(metrics[:4]):  # Limit to 4 metrics for 2x2 grid
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        values = [results[model].get(metric, 0) for model in models]
        colors = ECONET_PALETTE[:len(models)]
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=values,
                name=metric,
                marker_color=colors,
                showlegend=False,
                hovertemplate=f'<b>{metric}</b><br>' +
                             'Model: %{x}<br>' +
                             'Value: %{y:.4f}<br>' +
                             '<extra></extra>'
            ),
            row=row, col=col
        )
    
    # Layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20, color=ECONET_COLORS['dark'])
        ),
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False
    )
    
    return fig

def create_correlation_heatmap(data: pd.DataFrame,
                              title: str = "Correlation Matrix",
                              method: str = 'pearson') -> go.Figure:
    """Create interactive correlation heatmap"""
    
    # Calculate correlation matrix
    if method == 'pearson':
        corr_matrix = data.corr()
    elif method == 'spearman':
        corr_matrix = data.corr(method='spearman')
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate='<b>Correlation</b><br>' +
                     'X: %{x}<br>' +
                     'Y: %{y}<br>' +
                     'Correlation: %{z:.3f}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20, color=ECONET_COLORS['dark'])
        ),
        width=600,
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_distribution_plot(data: pd.Series,
                           title: str = "Distribution Analysis",
                           show_normal: bool = True,
                           bins: int = 50) -> go.Figure:
    """Create distribution plot with optional normal overlay"""
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=bins,
        name='Distribution',
        marker_color=ECONET_COLORS['primary'],
        opacity=0.7,
        hovertemplate='<b>Distribution</b><br>' +
                     'Range: %{x}<br>' +
                     'Count: %{y}<br>' +
                     '<extra></extra>'
    ))
    
    # Add normal distribution overlay
    if show_normal:
        mean = data.mean()
        std = data.std()
        x_range = np.linspace(data.min(), data.max(), 100)
        normal_dist = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std) ** 2)
        
        # Scale to match histogram
        hist_counts, hist_bins = np.histogram(data, bins=bins)
        bin_width = hist_bins[1] - hist_bins[0]
        normal_scaled = normal_dist * len(data) * bin_width
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal_scaled,
            mode='lines',
            name='Normal Distribution',
            line=dict(color=ECONET_COLORS['accent'], width=3),
            hovertemplate='<b>Normal Distribution</b><br>' +
                         'Value: %{x:.2f}<br>' +
                         'Density: %{y:.2f}<br>' +
                         '<extra></extra>'
        ))
    
    # Add statistics annotations
    mean_val = data.mean()
    std_val = data.std()
    skew_val = data.skew()
    kurt_val = data.kurtosis()
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        text=f"Mean: {mean_val:.2f}<br>" +
             f"Std: {std_val:.2f}<br>" +
             f"Skewness: {skew_val:.2f}<br>" +
             f"Kurtosis: {kurt_val:.2f}",
        showarrow=False,
        font=dict(size=12),
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1
    )
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20, color=ECONET_COLORS['dark'])
        ),
        xaxis_title="Value",
        yaxis_title="Frequency",
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True
    )
    
    return fig

def create_risk_dashboard(var_results: Dict[str, float],
                         monte_carlo_results: pd.DataFrame,
                         stress_test_results: Dict[str, Dict[str, float]]) -> go.Figure:
    """Create comprehensive risk analysis dashboard"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Value at Risk (VaR)",
            "Monte Carlo Distribution",
            "Stress Test Results",
            "Risk Metrics Summary"
        ),
        specs=[[{"type": "bar"}, {"type": "histogram"}],
               [{"type": "bar"}, {"type": "table"}]]
    )
    
    # VaR comparison
    var_methods = ['Historical', 'Parametric', 'Monte Carlo']
    var_values = [var_results.get(f'{method}_VaR', 0) for method in var_methods]
    
    fig.add_trace(
        go.Bar(
            x=var_methods,
            y=var_values,
            name='VaR',
            marker_color=ECONET_COLORS['danger'],
            hovertemplate='<b>VaR</b><br>' +
                         'Method: %{x}<br>' +
                         'VaR: %{y:.4f}<br>' +
                         '<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Monte Carlo distribution
    if not monte_carlo_results.empty:
        final_values = monte_carlo_results.iloc[:, -1]
        fig.add_trace(
            go.Histogram(
                x=final_values,
                nbinsx=50,
                name='MC Distribution',
                marker_color=ECONET_COLORS['primary'],
                opacity=0.7
            ),
            row=1, col=2
        )
    
    # Stress test results
    scenarios = list(stress_test_results.keys())
    stress_vars = [stress_test_results[scenario].get('VaR', 0) for scenario in scenarios]
    
    fig.add_trace(
        go.Bar(
            x=scenarios,
            y=stress_vars,
            name='Stress VaR',
            marker_color=ECONET_COLORS['warning'],
            hovertemplate='<b>Stress VaR</b><br>' +
                         'Scenario: %{x}<br>' +
                         'VaR: %{y:.4f}<br>' +
                         '<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Risk metrics table
    metrics_data = [
        ['VaR (95%)', f"{var_results.get('VaR', 0):.4f}"],
        ['CVaR (95%)', f"{var_results.get('CVaR', 0):.4f}"],
        ['Volatility', f"{var_results.get('Volatility', 0):.4f}"],
        ['Sharpe Ratio', f"{var_results.get('Sharpe_Ratio', 0):.4f}"],
        ['Max Drawdown', f"{var_results.get('Max_Drawdown', 0):.4f}"]
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Metric', 'Value'],
                fill_color=ECONET_COLORS['primary'],
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=list(zip(*metrics_data)),
                fill_color='white',
                font=dict(size=11)
            )
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title=dict(
            text="Risk Analysis Dashboard",
            x=0.5,
            font=dict(size=24, color=ECONET_COLORS['dark'])
        ),
        height=800,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_monte_carlo_paths(simulation_results: pd.DataFrame,
                           n_paths_display: int = 100,
                           title: str = "Monte Carlo Simulation Paths") -> go.Figure:
    """Visualize Monte Carlo simulation paths"""
    
    fig = go.Figure()
    
    # Select subset of paths for display
    n_simulations = len(simulation_results)
    path_indices = np.linspace(0, n_simulations-1, min(n_paths_display, n_simulations), dtype=int)
    
    # Time columns (exclude 'Simulation' column if present)
    time_cols = [col for col in simulation_results.columns if col != 'Simulation']
    time_steps = range(len(time_cols))
    
    # Plot individual paths
    for idx in path_indices:
        path_values = simulation_results.iloc[idx][time_cols].values
        
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=path_values,
            mode='lines',
            line=dict(color='rgba(69, 183, 209, 0.1)', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add mean path
    mean_path = simulation_results[time_cols].mean()
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=mean_path.values,
        mode='lines',
        name='Mean Path',
        line=dict(color=ECONET_COLORS['primary'], width=3),
        hovertemplate='<b>Mean Path</b><br>' +
                     'Time: %{x}<br>' +
                     'Value: %{y:.2f}<br>' +
                     '<extra></extra>'
    ))
    
    # Add percentile bands
    p5 = simulation_results[time_cols].quantile(0.05)
    p95 = simulation_results[time_cols].quantile(0.95)
    
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=p95.values,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=p5.values,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 107, 107, 0.2)',
        name='90% Confidence Band',
        hovertemplate='<b>90% Confidence</b><br>' +
                     'Time: %{x}<br>' +
                     'Value: %{y:.2f}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20, color=ECONET_COLORS['dark'])
        ),
        xaxis_title="Time Steps",
        yaxis_title="Value",
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x'
    )
    
    return fig

def create_feature_importance_plot(feature_importance: List[Tuple[str, float]],
                                 top_n: int = 15,
                                 title: str = "Feature Importance") -> go.Figure:
    """Create feature importance visualization"""
    
    # Sort and take top N
    sorted_features = sorted(feature_importance, key=lambda x: x[1], reverse=True)[:top_n]
    
    features, importances = zip(*sorted_features)
    
    fig = go.Figure(go.Bar(
        x=list(importances),
        y=list(features),
        orientation='h',
        marker_color=ECONET_COLORS['accent'],
        hovertemplate='<b>Feature Importance</b><br>' +
                     'Feature: %{y}<br>' +
                     'Importance: %{x:.4f}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20, color=ECONET_COLORS['dark'])
        ),
        xaxis_title="Importance",
        yaxis_title="Features",
        height=max(400, len(features) * 25),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_economic_indicators_dashboard(data: pd.DataFrame) -> go.Figure:
    """Create comprehensive economic indicators dashboard"""
    
    # Identify key economic columns
    economic_cols = []
    col_patterns = ['gdp', 'inflation', 'unemployment', 'interest', 'debt', 'growth']
    
    for col in data.columns:
        if any(pattern in col.lower() for pattern in col_patterns):
            economic_cols.append(col)
    
    if len(economic_cols) < 2:
        economic_cols = data.select_dtypes(include=[np.number]).columns.tolist()[:4]
    
    n_cols = min(len(economic_cols), 4)
    if n_cols <= 2:
        rows, cols = 1, n_cols
    else:
        rows, cols = 2, 2
    
    subplot_titles = economic_cols[:n_cols]
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    for i, col in enumerate(economic_cols[:n_cols]):
        row = (i // cols) + 1
        col_pos = (i % cols) + 1
        
        if col in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[col],
                    mode='lines',
                    name=col,
                    line=dict(color=ECONET_PALETTE[i % len(ECONET_PALETTE)], width=2),
                    showlegend=False,
                    hovertemplate=f'<b>{col}</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Value: %{y:.2f}<br>' +
                                 '<extra></extra>'
                ),
                row=row, col=col_pos
            )
    
    fig.update_layout(
        title=dict(
            text="Economic Indicators Dashboard",
            x=0.5,
            font=dict(size=24, color=ECONET_COLORS['dark'])
        ),
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x'
    )
    
    return fig
