"""
NERVA Professional Styling System
HTML/CSS Components for Economic Intelligence Dashboard
"""

# Professional CSS styling without emojis
NERVA_PROFESSIONAL_CSS = """
<style>
/* NERVA Professional Theme */
:root {
    --nerva-primary: #2E86C1;
    --nerva-secondary: #E74C3C;
    --nerva-accent: #8E44AD;
    --nerva-success: #27AE60;
    --nerva-warning: #F39C12;
    --nerva-info: #17A2B8;
    --nerva-dark: #2C3E50;
    --nerva-light: #ECF0F1;
    --nerva-bg: #F8F9FA;
    --nerva-border: #DEE2E6;
}

/* Main App Container */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
}

/* Header Styling */
.nerva-header {
    background: linear-gradient(135deg, var(--nerva-dark) 0%, var(--nerva-primary) 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    text-align: center;
}

.nerva-header h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}

.nerva-header .subtitle {
    font-size: 1.1rem;
    opacity: 0.9;
    margin-top: 0.5rem;
}

/* Status Indicators */
.status-indicator {
    display: inline-flex;
    align-items: center;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    margin: 0.25rem;
}

.status-active {
    background-color: var(--nerva-success);
    color: white;
}

.status-warning {
    background-color: var(--nerva-warning);
    color: white;
}

.status-error {
    background-color: var(--nerva-secondary);
    color: white;
}

/* Metric Cards */
.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    border-left: 4px solid var(--nerva-primary);
    margin-bottom: 1rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--nerva-primary);
    margin: 0;
}

.metric-label {
    font-size: 0.9rem;
    color: var(--nerva-dark);
    opacity: 0.8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-change {
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 0.5rem;
}

.metric-change.positive {
    color: var(--nerva-success);
}

.metric-change.negative {
    color: var(--nerva-secondary);
}

/* Navigation Tabs */
.nav-tabs {
    background: white;
    border-radius: 10px;
    padding: 0.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.nav-tab {
    display: inline-block;
    padding: 0.75rem 1.5rem;
    margin: 0.25rem;
    border-radius: 8px;
    background: var(--nerva-light);
    color: var(--nerva-dark);
    text-decoration: none;
    font-weight: 600;
    transition: all 0.2s ease;
    border: none;
    cursor: pointer;
}

.nav-tab:hover {
    background: var(--nerva-primary);
    color: white;
    transform: translateY(-1px);
}

.nav-tab.active {
    background: var(--nerva-primary);
    color: white;
}

/* Data Tables */
.dataframe {
    border: none !important;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.dataframe thead th {
    background: var(--nerva-primary) !important;
    color: white !important;
    font-weight: 600;
    padding: 1rem;
    border: none !important;
}

.dataframe tbody td {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--nerva-border) !important;
}

.dataframe tbody tr:hover {
    background: var(--nerva-bg) !important;
}

/* Alert Boxes */
.alert {
    padding: 1rem 1.5rem;
    border-radius: 8px;
    margin: 1rem 0;
    border-left: 4px solid;
}

.alert-info {
    background: rgba(23, 162, 184, 0.1);
    border-left-color: var(--nerva-info);
    color: var(--nerva-info);
}

.alert-success {
    background: rgba(39, 174, 96, 0.1);
    border-left-color: var(--nerva-success);
    color: var(--nerva-success);
}

.alert-warning {
    background: rgba(243, 156, 18, 0.1);
    border-left-color: var(--nerva-warning);
    color: var(--nerva-warning);
}

.alert-error {
    background: rgba(231, 76, 60, 0.1);
    border-left-color: var(--nerva-secondary);
    color: var(--nerva-secondary);
}

/* Progress Bars */
.progress {
    background: var(--nerva-light);
    border-radius: 10px;
    height: 20px;
    overflow: hidden;
    margin: 0.5rem 0;
}

.progress-bar {
    background: linear-gradient(90deg, var(--nerva-primary), var(--nerva-accent));
    height: 100%;
    border-radius: 10px;
    transition: width 0.5s ease;
}

/* Icon Styles */
.icon {
    display: inline-block;
    width: 16px;
    height: 16px;
    margin-right: 0.5rem;
    vertical-align: middle;
}

.icon-dashboard {
    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%23ffffff'%3E%3Cpath d='M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z'/%3E%3C/svg%3E") no-repeat center;
    background-size: contain;
}

.icon-analytics {
    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%23ffffff'%3E%3Cpath d='M16 6l2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z'/%3E%3C/svg%3E") no-repeat center;
    background-size: contain;
}

.icon-models {
    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%23ffffff'%3E%3Cpath d='M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z'/%3E%3C/svg%3E") no-repeat center;
    background-size: contain;
}

.icon-notebooks {
    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%23ffffff'%3E%3Cpath d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z'/%3E%3Cpolyline points='14 2 14 8 20 8'/%3E%3Cline x1='16' y1='13' x2='8' y2='13'/%3E%3Cline x1='16' y1='17' x2='8' y2='17'/%3E%3Cpolyline points='10 9 9 9 8 9'/%3E%3C/svg%3E") no-repeat center;
    background-size: contain;
}

.icon-settings {
    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%23ffffff'%3E%3Cpath d='M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z'/%3E%3C/svg%3E") no-repeat center;
    background-size: contain;
}

/* Sidebar */
.css-1d391kg {
    background: linear-gradient(180deg, var(--nerva-dark) 0%, var(--nerva-primary) 100%);
}

.css-1d391kg .css-1544g2n {
    color: white;
}

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: var(--nerva-light);
}

::-webkit-scrollbar-thumb {
    background: var(--nerva-primary);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--nerva-accent);
}
</style>
"""

# HTML Icons (SVG-based)
NERVA_ICONS = {
    'dashboard': """
    <svg class="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
        <path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z"/>
    </svg>
    """,
    
    'analytics': """
    <svg class="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
        <path d="M16 6l2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z"/>
    </svg>
    """,
    
    'models': """
    <svg class="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
    </svg>
    """,
    
    'notebooks': """
    <svg class="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z"/>
        <polyline points="14 2 14 8 20 8"/>
        <line x1="16" y1="13" x2="8" y2="13"/>
        <line x1="16" y1="17" x2="8" y2="17"/>
        <polyline points="10 9 9 9 8 9"/>
    </svg>
    """,
    
    'settings': """
    <svg class="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
        <path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/>
    </svg>
    """,
    
    'analysis': """
    <svg class="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
        <path d="M7 12l3-3 3 3 4.5-4.5L19 9V5h-4l1.5 1.5L13 10l-3-3-4.5 4.5L7 12z"/>
    </svg>
    """,
    
    'data': """
    <svg class="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 3C7.58 3 4 4.79 4 7s3.58 4 8 4 8-1.79 8-4-3.58-4-8-4zM4 9v3c0 2.21 3.58 4 8 4s8-1.79 8-4V9c0 2.21-3.58 4-8 4s-8-1.79-8-4zM4 14v3c0 2.21 3.58 4 8 4s8-1.79 8-4v-3c0 2.21-3.58 4-8 4s-8-1.79-8-4z"/>
    </svg>
    """,
    
    'forecast': """
    <svg class="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
        <path d="M3.5 18.49l6-6.01 4 4L22 6.92l-1.41-1.41-7.09 7.97-4-4L2 16.99z"/>
    </svg>
    """,
    
    'risk': """
    <svg class="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z"/>
    </svg>
    """
}

def get_professional_header(title, subtitle="Economic Intelligence System"):
    """Generate professional header HTML"""
    return f"""
    <div class="nerva-header">
        <h1>{title}</h1>
        <div class="subtitle">{subtitle}</div>
    </div>
    """

def get_metric_card(value, label, change=None, change_type="neutral"):
    """Generate metric card HTML"""
    change_html = ""
    if change is not None:
        change_class = f"metric-change {change_type}"
        change_html = f'<div class="{change_class}">{change}</div>'
    
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {change_html}
    </div>
    """

def get_status_indicator(status, label):
    """Generate status indicator HTML"""
    status_map = {
        'active': 'status-active',
        'warning': 'status-warning', 
        'error': 'status-error'
    }
    
    status_class = status_map.get(status, 'status-active')
    
    return f"""
    <span class="status-indicator {status_class}">
        {label}
    </span>
    """

def get_alert_box(message, alert_type="info"):
    """Generate alert box HTML"""
    return f"""
    <div class="alert alert-{alert_type}">
        {message}
    </div>
    """

def get_progress_bar(percentage, height=20):
    """Generate progress bar HTML"""
    return f"""
    <div class="progress" style="height: {height}px;">
        <div class="progress-bar" style="width: {percentage}%;"></div>
    </div>
    """
