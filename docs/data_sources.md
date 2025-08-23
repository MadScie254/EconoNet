# EconoNet Data Sources

This document provides comprehensive information about all data sources, APIs, schemas, TTLs (Time To Live), and fallback mechanisms used in EconoNet.

## Overview

EconoNet integrates with multiple **free, no-token APIs** to provide real-time economic intelligence. The system features automatic caching, graceful fallbacks, and standardized data schemas.

## Operation Modes

| Mode | Description | Data Sources | Use Case |
|------|-------------|--------------|----------|
| **Offline** | Demo mode with synthetic data | Quantum simulations only | Demonstrations, testing, offline work |
| **Live** | Real API integration with fallbacks | Live APIs + synthetic fallbacks | Production use, real analysis |
| **Expert** | Advanced features with live data | All APIs + advanced analytics | Research, professional analysis |

## Data Sources

### 1. World Bank Open Data API

**Purpose**: Macroeconomic indicators (GDP, inflation, trade, etc.)

| Parameter | Value |
|-----------|--------|
| **Base URL** | `https://api.worldbank.org/v2` |
| **Authentication** | None required |
| **Rate Limit** | Generous (no official limit) |
| **TTL** | 3600 seconds (1 hour) |
| **Timeout** | 30 seconds |
| **Coverage** | 200+ countries, 1960-present |

**Available Indicators**:
- `NY.GDP.MKTP.CD` - GDP (current US$)
- `FP.CPI.TOTL.ZG` - Inflation, consumer prices (annual %)
- `SL.UEM.TOTL.ZS` - Unemployment, total (% of total labor force)
- `NE.TRD.GNFS.ZS` - Trade (% of GDP)

**Example Request**:
```
GET https://api.worldbank.org/v2/country/KE/indicator/NY.GDP.MKTP.CD?date=2010:2025&format=json
```

**Data Schema Output**:
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| date | datetime | End of year date | 2023-12-31 |
| country | string | ISO 3-letter code | KEN |
| series | string | Indicator code | NY.GDP.MKTP.CD |
| value | float | Indicator value | 112500000000.0 |
| source | string | Data source | World Bank |
| metadata | dict | Additional info | {"unit": "USD", "last_refresh": "..."} |

---

### 2. European Central Bank (ECB) Statistical Data Warehouse

**Purpose**: Foreign exchange rates, monetary policy indicators

| Parameter | Value |
|-----------|--------|
| **Base URL** | `https://data-api.ecb.europa.eu/service/data` |
| **Authentication** | None required |
| **Rate Limit** | Reasonable usage |
| **TTL** | 1800 seconds (30 minutes) |
| **Timeout** | 20 seconds |
| **Coverage** | EUR exchange rates, ECB policy rates |

**Available Series**:
- `EXR.D.USD.EUR.SP00.A` - USD/EUR daily exchange rate
- `EXR.D.GBP.EUR.SP00.A` - GBP/EUR daily exchange rate
- `FM.B.U2.EUR.4F.KR.DFR.LEV` - ECB main refinancing rate

**Example Request**:
```
GET https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?format=jsondata&startPeriod=2024-01-01
```

---

### 3. Federal Reserve Economic Data (FRED) CSV Endpoint

**Purpose**: US economic indicators (no API key required via CSV)

| Parameter | Value |
|-----------|--------|
| **Base URL** | `https://fred.stlouisfed.org/graph/fredgraph.csv` |
| **Authentication** | None required |
| **Rate Limit** | Reasonable usage |
| **TTL** | 3600 seconds (1 hour) |
| **Timeout** | 25 seconds |
| **Coverage** | 800,000+ US economic time series |

**Popular Series**:
- `GDPC1` - Real Gross Domestic Product
- `CPIAUCSL` - Consumer Price Index for All Urban Consumers
- `UNRATE` - Unemployment Rate
- `FEDFUNDS` - Federal Funds Effective Rate

**Example Request**:
```
GET https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDPC1
```

---

### 4. CoinGecko API

**Purpose**: Cryptocurrency prices, market data, and volatility metrics

| Parameter | Value |
|-----------|--------|
| **Base URL** | `https://api.coingecko.com/api/v3` |
| **Authentication** | None required (free tier) |
| **Rate Limit** | 100 calls/minute |
| **TTL** | 300 seconds (5 minutes) |
| **Timeout** | 15 seconds |
| **Coverage** | 10,000+ cryptocurrencies |

**Endpoints Used**:
- `/coins/{id}/market_chart` - Historical prices and volumes
- `/simple/price` - Current prices

**Calculated Metrics**:
- 7-day rolling volatility
- Price change percentages
- Market sentiment indicators

**Example Request**:
```
GET https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=90&interval=daily
```

---

### 5. USGS Earthquake Hazards Program API

**Purpose**: Real-time earthquake data for economic risk assessment

| Parameter | Value |
|-----------|--------|
| **Base URL** | `https://earthquake.usgs.gov/fdsnws/event/1` |
| **Authentication** | None required |
| **Rate Limit** | None specified |
| **TTL** | 1800 seconds (30 minutes) |
| **Timeout** | 20 seconds |
| **Coverage** | Global, real-time and historical |

**Query Parameters**:
- `minmagnitude` - Minimum earthquake magnitude (default: 5.5)
- `starttime` / `endtime` - Date range
- `format=geojson` - Response format

**Example Request**:
```
GET https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2024-01-01&minmagnitude=5.5
```

---

### 6. Wikipedia Pageviews API

**Purpose**: Economic sentiment analysis via search attention

| Parameter | Value |
|-----------|--------|
| **Base URL** | `https://wikimedia.org/api/rest_v1/metrics/pageviews` |
| **Authentication** | None required |
| **Rate Limit** | 5000 requests/hour |
| **TTL** | 3600 seconds (1 hour) |
| **Timeout** | 25 seconds |
| **Coverage** | All Wikipedia articles, daily data |

**Tracked Economic Terms**:
- Economy, Recession, Inflation
- GDP, Unemployment, Interest Rate
- Stock Market, Currency Crisis

**Example Request**:
```
GET https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/Economy/daily/20240101/20240201
```

---

### 7. Open-Meteo Weather API

**Purpose**: Weather data for economic impact analysis (agriculture, energy)

| Parameter | Value |
|-----------|--------|
| **Base URL** | `https://api.open-meteo.com/v1` |
| **Authentication** | None required |
| **Rate Limit** | 10,000 calls/day |
| **TTL** | 1800 seconds (30 minutes) |
| **Timeout** | 20 seconds |
| **Coverage** | Global, 80+ weather variables |

**Variables Used**:
- Temperature extremes
- Precipitation
- Wind speed
- Weather anomalies

**Example Request**:
```
GET https://api.open-meteo.com/v1/forecast?latitude=-1.2864&longitude=36.8172&daily=temperature_2m_max,precipitation_sum&past_days=30
```

---

## Standardized Data Schema

All API adapters return pandas DataFrames with this consistent schema:

```python
df.columns = [
    'date',      # pd.Timestamp, UTC normalized
    'country',   # str, ISO alpha-3 code (if applicable, else None)  
    'series',    # str, identifier (e.g., "GDP", "BTC-USD", "Earthquake")
    'value',     # float, numeric value
    'source',    # str, data source name (e.g., "World Bank", "CoinGecko")
    'metadata'   # dict, additional information
]
```

**Metadata Structure**:
```python
metadata = {
    'endpoint': 'https://api.example.com/...',
    'last_refresh': '2024-08-23T10:30:00Z',
    'unit': 'USD',
    'fallback': False,  # True if synthetic data
    # API-specific fields...
}
```

## Caching Strategy

EconoNet uses `requests-cache` with SQLite backend for intelligent caching:

| Data Type | TTL | Rationale |
|-----------|-----|-----------|
| GDP/Macro Data | 1 hour | Updated quarterly/annually |
| Exchange Rates | 30 minutes | High volatility during trading hours |
| Crypto Prices | 5 minutes | Very high volatility |
| Earthquake Data | 30 minutes | Real-time monitoring needs |
| Weather Data | 30 minutes | Moderate update frequency |
| Wikipedia Views | 1 hour | Daily aggregation |

## Fallback Mechanisms

When API calls fail, EconoNet provides graceful fallbacks:

### 1. **Synthetic Data Generation**
- Mathematically realistic time series
- Appropriate noise and trends
- Marked with `metadata.fallback = True`

### 2. **UI Indicators**
- 🔄 Red badges for fallback data
- 🌐 Blue badges for live data  
- Banner notifications for users

### 3. **Error Handling**
- Exponential backoff retries (via `tenacity`)
- Timeout protection
- Comprehensive logging

## Rate Limiting & Best Practices

### API Usage Guidelines:
1. **Respect rate limits** - Use caching to minimize requests
2. **Handle failures gracefully** - Always provide fallbacks
3. **Monitor usage** - Log API call patterns
4. **User feedback** - Show data provenance clearly

### Implementation:
```python
from econonet import get_worldbank, get_config

# Configure for your use case
config = get_config()
config.mode = OperationMode.LIVE

# Fetch data with automatic caching and fallbacks
gdp_data = get_worldbank('KE', 'NY.GDP.MKTP.CD')
```

## Error Scenarios & Responses

| Error Type | API Response | EconoNet Response |
|------------|--------------|-------------------|
| **Network Timeout** | - | Retry 3x, then synthetic data |
| **Rate Limit (429)** | HTTP 429 | Exponential backoff, then cache |
| **Not Found (404)** | HTTP 404 | Log warning, synthetic data |
| **Server Error (5xx)** | HTTP 5xx | Retry with backoff, then fallback |
| **Invalid Data** | Malformed JSON | Parse what possible, fill gaps |

## Regional Data Coverage

### Africa Focus (Default: Kenya)
- **World Bank**: Full coverage for 54 African countries
- **ECB**: EUR exchange rates relevant to African economies
- **Weather**: Continental coverage including agricultural regions
- **Crypto**: Global data, increasing African adoption

### Extensibility
The system is designed for easy geographic expansion:
```python
# Switch to different country/region
config.default_country = 'NG'  # Nigeria
config.default_region = 'Africa/Lagos'
```

## Testing & Validation

### Data Quality Checks:
1. **Schema validation** - Ensure consistent column structure
2. **Range validation** - Check for realistic value ranges  
3. **Temporal validation** - Verify date sequences
4. **Source validation** - Confirm metadata integrity

### Integration Tests:
- Mock API responses for reliability testing
- Fallback mechanism validation
- Cache behavior verification
- Performance benchmarking

## Future Enhancements

### Planned Additions:
1. **IMF Data** - Enhanced SDMX integration
2. **Regional Central Banks** - Bank of England, Bank of Japan APIs
3. **Commodity Data** - Oil, gold, agricultural prices
4. **Satellite Data** - Economic activity indicators
5. **Social Media Sentiment** - Twitter/Reddit economic sentiment

### Performance Optimizations:
1. **Async requests** - Concurrent API calls
2. **Data compression** - Efficient storage
3. **Smart caching** - Predictive data fetching
4. **CDN integration** - Global data distribution

---

*Last Updated: August 23, 2025*  
*EconoNet Data Sources Documentation v1.0*
