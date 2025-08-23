# EconoNet Usage Guide

This guide shows you how to use EconoNet's various features and components.

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/MadScie254/EconoNet.git
cd EconoNet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Run the dashboards (IMPORTANT - Use Streamlit, not Python):**

```bash
# Main ultra dashboard
streamlit run ultra_dashboard_enhanced.py

# Immersive dashboard
streamlit run immersive_dashboard.py

# Enhanced streamlit app
streamlit run enhanced_streamlit_app.py
```

⚠️ **Do NOT run dashboards with `python` directly:**
```bash
# ❌ WRONG - This will cause ScriptRunContext warnings
python ultra_dashboard_enhanced.py

# ✅ CORRECT - Always use streamlit run
streamlit run ultra_dashboard_enhanced.py
```

### Basic Usage

```python
import sys
sys.path.append('src')

from econonet import get_worldbank, set_mode, OperationMode

# Set operation mode
set_mode(OperationMode.LIVE)

# Fetch GDP data for Kenya
gdp_data = get_worldbank('KE', 'NY.GDP.MKTP.CD')
print(gdp_data.head())
```

## Operation Modes

### Offline Mode (Demo)
```python
from econonet import set_mode, OperationMode

set_mode(OperationMode.OFFLINE)
# All API calls return synthetic data
```

### Live Mode (Production)
```python
set_mode(OperationMode.LIVE)
# Real API calls with fallbacks
```

### Expert Mode (Advanced)
```python
set_mode(OperationMode.EXPERT)
# All features enabled
```

## API Usage Examples

### World Bank Data
```python
from econonet import get_worldbank

# GDP for multiple countries
countries = ['KE', 'UG', 'TZ', 'RW']
for country in countries:
    gdp = get_worldbank(country, 'NY.GDP.MKTP.CD', start='2020')
    print(f"{country}: {len(gdp)} data points")
```

### Cryptocurrency Data
```python
from econonet import get_coingecko

# Bitcoin and Ethereum data
crypto = get_coingecko(['bitcoin', 'ethereum'])
print(f"Crypto data: {len(crypto)} records")

# Access volatility from metadata
btc_data = crypto[crypto['series'] == 'BITCOIN-USD']
volatility = btc_data['metadata'].apply(lambda x: x.get('volatility', 0))
print(f"Average BTC volatility: {volatility.mean():.2f}%")
```

### Risk Monitoring
```python
from econonet import get_usgs

# Recent earthquakes
earthquakes = get_usgs(min_mag=5.0, window="7d")
if not earthquakes.empty:
    latest = earthquakes.sort_values('date').iloc[-1]
    print(f"Latest earthquake: M{latest['value']:.1f}")
```

## Dashboard Components

### Running Dashboards

1. **Ultra Dashboard** (Main):
```bash
streamlit run ultra_dashboard_enhanced.py
```

2. **Immersive Dashboard**:
```bash
streamlit run immersive_dashboard.py
```

3. **Enhanced App**:
```bash
streamlit run enhanced_streamlit_app.py
```

### Dashboard Features

- **Mode Selector**: Switch between offline/live/expert modes
- **Country Selector**: Choose focus country (default: Kenya)
- **Real-time Sync**: Manual refresh buttons for live data
- **Provenance Display**: See data sources and last refresh times
- **Fallback Indicators**: Visual indicators when using synthetic data

## Jupyter Notebook Integration

### Live Data in Notebooks

```python
# In your notebook
import sys
sys.path.append('../src')

from econonet import get_all_data, is_live_mode

# Get comprehensive data
data = get_all_data('KE')

if is_live_mode():
    print("Using live API data")
else:
    print("Using synthetic data")

# Access individual datasets
gdp = data.get('worldbank_gdp')
crypto = data.get('crypto')
```

### Notebook Integration System

The notebook integrator provides:
- Automatic live data injection
- Execution metadata tracking
- Performance monitoring
- Error handling

## Visualization Components

### Sentiment Radar
```python
from econonet.visual import create_sentiment_radar

sentiment_scores = {
    'Economic Attention': 0.7,
    'Crypto Fear/Greed': 0.3,
    'Global Risk': 0.6,
    'Market Volatility': 0.8
}

fig = create_sentiment_radar(sentiment_scores)
fig.show()
```

### Real vs Synthetic Overlay
```python
from econonet.visual import create_real_vs_synthetic_overlay
import pandas as pd
import numpy as np

# Synthetic data
synthetic = pd.Series(np.random.normal(100, 10, 30))

# Real data (example)
real_data = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=30),
    'value': np.random.normal(105, 8, 30)
})

fig = create_real_vs_synthetic_overlay(synthetic, real_data)
fig.show()
```

### Risk Alert Cards
```python
from econonet.visual import create_risk_alert_card

risk_html = create_risk_alert_card(
    risk_level='HIGH',
    risk_score=0.8,
    risk_factors=[
        'High cryptocurrency volatility',
        'Earthquake activity increased',
        'Economic attention spiking'
    ]
)
print(risk_html)
```

## Configuration Management

### Environment Variables
```bash
# Set operation mode
export ECONET_MODE=live

# Set default country
export ECONET_COUNTRY=NG

# Set cache directory
export ECONET_CACHE_DIR=/path/to/cache

# Disable caching
export ECONET_CACHE=false
```

### Programmatic Configuration
```python
from econonet import get_config

config = get_config()

# Modify API settings
config.apis['worldbank'].ttl_seconds = 7200  # 2 hours
config.apis['coingecko'].timeout_seconds = 10

# Change default country
config.default_country = 'NG'  # Nigeria
```

## Data Schema Reference

All API functions return DataFrames with this schema:

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Timestamp (UTC) |
| `country` | str | ISO 3-letter code or None |
| `series` | str | Data series identifier |
| `value` | float | Numeric value |
| `source` | str | Data source name |
| `metadata` | dict | Additional information |

### Metadata Structure
```python
{
    'endpoint': 'https://api.example.com/...',
    'last_refresh': '2024-08-23T10:30:00Z',
    'unit': 'USD',
    'fallback': False,  # True if synthetic
    # API-specific fields...
}
```

## Error Handling

### Common Patterns
```python
from econonet import get_worldbank

try:
    data = get_worldbank('INVALID', 'NY.GDP.MKTP.CD')
except Exception as e:
    print(f"Error: {e}")
    # Function will return synthetic data instead

# Check if data is synthetic
if data['metadata'].iloc[0].get('fallback', False):
    print("Using synthetic fallback data")
```

### Logging
```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# EconoNet operations will now log details
data = get_worldbank('KE', 'NY.GDP.MKTP.CD')
```

## Performance Optimization

### Caching Best Practices
```python
# Pre-warm cache for common data
from econonet import get_all_data

data = get_all_data('KE')  # Caches all major datasets

# Subsequent calls will be fast
gdp = get_worldbank('KE', 'NY.GDP.MKTP.CD')  # From cache
```

### Batch Operations
```python
# Instead of multiple calls
countries = ['KE', 'UG', 'TZ']
all_gdp = []

for country in countries:
    gdp = get_worldbank(country, 'NY.GDP.MKTP.CD')
    all_gdp.append(gdp)

combined = pd.concat(all_gdp, ignore_index=True)
```

## Testing

### Running Tests
```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run specific test file
pytest tests/test_live_apis.py

# Run with coverage
pytest --cov=src/econonet tests/
```

### Custom Tests
```python
from econonet import validate_data_schema
import pandas as pd

# Validate your data follows the schema
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=10),
    'country': 'KE',
    'series': 'TEST',
    'value': range(10),
    'source': 'Test',
    'metadata': [{}] * 10
})

assert validate_data_schema(df)
```

## Troubleshooting

### Common Issues

1. **Import Errors**:
```python
import sys
sys.path.append('src')  # Add to Python path
```

2. **API Timeouts**:
```python
from econonet import get_config

config = get_config()
config.apis['worldbank'].timeout_seconds = 60  # Increase timeout
```

3. **Cache Issues**:
```bash
# Clear cache
rm -rf .econet_cache/

# Or disable caching
export ECONET_CACHE=false
```

4. **Streamlit Port Conflicts**:
```bash
streamlit run ultra_dashboard_enhanced.py --server.port 8502
```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

from econonet import get_worldbank
data = get_worldbank('KE', 'NY.GDP.MKTP.CD')
# Will show detailed logs
```

## Advanced Usage

### Custom API Adapters
```python
from econonet.live_apis import api_fallback
import pandas as pd

@api_fallback
def get_custom_api(param1, param2):
    # Your custom API implementation
    data = fetch_from_custom_api(param1, param2)
    
    # Return standardized DataFrame
    return pd.DataFrame({
        'date': data['dates'],
        'country': None,
        'series': 'CUSTOM',
        'value': data['values'],
        'source': 'Custom API',
        'metadata': [{'custom': True}] * len(data['dates'])
    })
```

### Custom Visualizations
```python
import plotly.graph_objects as go

def create_custom_chart(data):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['value'],
        mode='lines+markers',
        name=data['series'].iloc[0]
    ))
    
    fig.update_layout(
        template='plotly_dark',
        font=dict(color='white')
    )
    
    return fig
```

---

*Last Updated: August 23, 2025*  
*EconoNet Usage Guide v1.0*
