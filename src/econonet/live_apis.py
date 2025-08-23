"""
EconoNet Live API Adapters
==========================

Unified API connectors with caching, retries, and standardized schemas.
All functions return pandas DataFrames with consistent columns:
[date, country, series, value, source, metadata]
"""

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from functools import wraps
import logging
from pathlib import Path

# Third-party imports for caching and retries
try:
    import requests_cache
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False
    # Fallback decorators
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    stop_after_attempt = wait_exponential = retry_if_exception_type = lambda *args, **kwargs: None

from .config import get_config, is_live_mode

# Setup logging
logger = logging.getLogger(__name__)

# Global session with caching
_session = None

def get_session():
    """Get cached requests session"""
    global _session
    if _session is None:
        config = get_config()
        if CACHING_AVAILABLE and config.cache_enabled:
            cache_dir = Path(config.cache_dir)
            cache_dir.mkdir(exist_ok=True)
            _session = requests_cache.CachedSession(
                cache_name=str(cache_dir / "econet_cache"),
                backend="sqlite",
                expire_after=timedelta(seconds=3600)  # Default 1 hour
            )
        else:
            _session = requests.Session()
    return _session

def api_fallback(func):
    """Decorator to provide synthetic fallback when API calls fail"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not is_live_mode():
            # Generate synthetic data in offline mode
            return _generate_synthetic_data(func.__name__, *args, **kwargs)
        
        try:
            result = func(*args, **kwargs)
            if result is not None and not result.empty:
                return result
        except Exception as e:
            logger.warning(f"API call failed for {func.__name__}: {e}")
        
        # Generate fallback data
        logger.info(f"Generating synthetic fallback for {func.__name__}")
        return _generate_synthetic_data(func.__name__, *args, **kwargs)
    
    return wrapper

def _generate_synthetic_data(func_name: str, *args, **kwargs) -> pd.DataFrame:
    """Generate synthetic data matching the expected schema"""
    np.random.seed(42)  # Reproducible data
    
    # Common time range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    config = get_config()
    
    if 'worldbank' in func_name:
        country = args[0] if args else config.default_country
        indicator = args[1] if len(args) > 1 else 'GDP'
        
        # Generate realistic GDP growth
        base_value = 100e9  # 100 billion baseline
        trend = np.linspace(0, 0.05 * len(dates), len(dates))  # 5% annual growth
        noise = np.random.normal(0, 0.02, len(dates))
        values = base_value * (1 + trend + noise)
        
        return pd.DataFrame({
            'date': dates,
            'country': country,
            'series': indicator,
            'value': values,
            'source': 'Synthetic',
            'metadata': [{'fallback': True, 'unit': 'USD'}] * len(dates)
        })
    
    elif 'coingecko' in func_name:
        # Crypto price simulation
        btc_prices = 50000 + np.cumsum(np.random.normal(0, 1000, len(dates)))
        btc_prices = np.clip(btc_prices, 10000, 200000)  # Reasonable bounds
        
        data = []
        for i, date in enumerate(dates):
            data.append({
                'date': date,
                'country': None,
                'series': 'BTC-USD',
                'value': btc_prices[i],
                'source': 'Synthetic',
                'metadata': {'fallback': True, 'unit': 'USD'}
            })
        
        return pd.DataFrame(data)
    
    elif 'usgs' in func_name:
        # Generate some earthquake events
        num_events = np.random.poisson(5)  # Average 5 events
        
        data = []
        for _ in range(num_events):
            event_date = start_date + timedelta(days=np.random.randint(0, 365))
            magnitude = np.random.exponential(2) + 4.5  # Exponential distribution, min 4.5
            
            data.append({
                'date': event_date,
                'country': None,
                'series': 'Earthquake',
                'value': magnitude,
                'source': 'Synthetic',
                'metadata': {
                    'fallback': True,
                    'location': 'Synthetic Location',
                    'latitude': np.random.uniform(-90, 90),
                    'longitude': np.random.uniform(-180, 180)
                }
            })
        
        return pd.DataFrame(data)
    
    else:
        # Generic synthetic data
        values = np.random.normal(100, 10, len(dates))
        return pd.DataFrame({
            'date': dates,
            'country': config.default_country,
            'series': 'Generic',
            'value': values,
            'source': 'Synthetic',
            'metadata': [{'fallback': True}] * len(dates)
        })

# ============================================================================
# WORLD BANK API
# ============================================================================

@api_fallback
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(requests.RequestException)
)
def get_worldbank(country: str, indicator: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch data from World Bank Open Data API
    
    Args:
        country: ISO country code (e.g., 'KE' for Kenya)
        indicator: World Bank indicator code (e.g., 'NY.GDP.MKTP.CD')
        start: Start year (YYYY format)
        end: End year (YYYY format)
    
    Returns:
        DataFrame with standardized schema
    """
    config = get_config()
    api_config = config.get_api_config('worldbank')
    
    if not start:
        start = str(datetime.now().year - 10)
    if not end:
        end = str(datetime.now().year)
    
    url = f"{api_config.base_url}/country/{country}/indicator/{indicator}"
    params = {
        'date': f'{start}:{end}',
        'format': 'json',
        'per_page': 1000
    }
    
    session = get_session()
    response = session.get(url, params=params, timeout=api_config.timeout_seconds)
    response.raise_for_status()
    
    data = response.json()
    if len(data) < 2 or not data[1]:
        return pd.DataFrame()
    
    records = []
    for item in data[1]:
        if item['value'] is not None:
            records.append({
                'date': pd.to_datetime(f"{item['date']}-12-31"),  # End of year
                'country': country,
                'series': indicator,
                'value': float(item['value']),
                'source': 'World Bank',
                'metadata': {
                    'endpoint': response.url,
                    'last_refresh': datetime.now().isoformat(),
                    'unit': item.get('unit', 'Unknown'),
                    'country_name': item.get('country', {}).get('value', ''),
                    'indicator_name': item.get('indicator', {}).get('value', '')
                }
            })
    
    return pd.DataFrame(records)

# ============================================================================
# ECB (European Central Bank) API
# ============================================================================

@api_fallback
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(requests.RequestException)
)
def get_ecb(series_key: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch data from ECB Statistical Data Warehouse
    
    Args:
        series_key: ECB series key (e.g., 'EXR.D.USD.EUR.SP00.A')
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with standardized schema
    """
    config = get_config()
    api_config = config.get_api_config('ecb')
    
    url = f"{api_config.base_url}/EXR/{series_key}"
    params = {'format': 'jsondata'}
    
    if start:
        params['startPeriod'] = start
    if end:
        params['endPeriod'] = end
    
    session = get_session()
    response = session.get(url, params=params, timeout=api_config.timeout_seconds)
    response.raise_for_status()
    
    data = response.json()
    
    # Parse ECB JSON structure
    observations = data.get('data', {}).get('dataSets', [{}])[0].get('observations', {})
    structure = data.get('data', {}).get('structure', {})
    
    records = []
    for obs_key, obs_data in observations.items():
        if obs_data and obs_data[0] is not None:
            # Extract date from structure
            time_idx = int(obs_key.split(':')[-1])
            time_values = structure.get('dimensions', {}).get('observation', [{}])[-1].get('values', [])
            
            if time_idx < len(time_values):
                date_str = time_values[time_idx].get('id', '')
                try:
                    date = pd.to_datetime(date_str)
                    records.append({
                        'date': date,
                        'country': None,  # ECB series may not have country
                        'series': series_key,
                        'value': float(obs_data[0]),
                        'source': 'ECB',
                        'metadata': {
                            'endpoint': response.url,
                            'last_refresh': datetime.now().isoformat(),
                            'series_key': series_key
                        }
                    })
                except (ValueError, TypeError):
                    continue
    
    return pd.DataFrame(records)

# ============================================================================
# FRED (Federal Reserve Economic Data) API
# ============================================================================

@api_fallback
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(requests.RequestException)
)
def get_fred_csv(series_id: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch data from FRED CSV endpoint (no API key required)
    
    Args:
        series_id: FRED series ID (e.g., 'GDPC1')
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with standardized schema
    """
    config = get_config()
    api_config = config.get_api_config('fred')
    
    url = api_config.base_url
    params = {'id': series_id}
    
    session = get_session()
    response = session.get(url, params=params, timeout=api_config.timeout_seconds)
    response.raise_for_status()
    
    # Parse CSV data
    df = pd.read_csv(StringIO(response.text))
    
    # Standardize columns
    if len(df.columns) >= 2:
        date_col = df.columns[0]
        value_col = df.columns[1]
        
        # Filter date range if specified
        df[date_col] = pd.to_datetime(df[date_col])
        if start:
            df = df[df[date_col] >= start]
        if end:
            df = df[df[date_col] <= end]
        
        # Remove missing values
        df = df.dropna(subset=[value_col])
        
        records = []
        for _, row in df.iterrows():
            records.append({
                'date': row[date_col],
                'country': 'US',  # FRED is US data
                'series': series_id,
                'value': float(row[value_col]),
                'source': 'FRED',
                'metadata': {
                    'endpoint': response.url,
                    'last_refresh': datetime.now().isoformat(),
                    'series_id': series_id
                }
            })
        
        return pd.DataFrame(records)
    
    return pd.DataFrame()

# ============================================================================
# COINGECKO API
# ============================================================================

@api_fallback
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(requests.RequestException)
)
def get_coingecko(ids: List[str] = None, vs_currencies: List[str] = None) -> pd.DataFrame:
    """
    Fetch cryptocurrency data from CoinGecko API
    
    Args:
        ids: List of coin IDs (default: ['bitcoin', 'ethereum'])
        vs_currencies: List of currencies (default: ['usd'])
    
    Returns:
        DataFrame with standardized schema including volatility
    """
    if ids is None:
        ids = ['bitcoin', 'ethereum']
    if vs_currencies is None:
        vs_currencies = ['usd']
    
    config = get_config()
    api_config = config.get_api_config('coingecko')
    
    records = []
    
    for coin_id in ids:
        for currency in vs_currencies:
            # Get historical prices (90 days)
            url = f"{api_config.base_url}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': currency,
                'days': '90',
                'interval': 'daily'
            }
            
            session = get_session()
            response = session.get(url, params=params, timeout=api_config.timeout_seconds)
            response.raise_for_status()
            
            data = response.json()
            prices = data.get('prices', [])
            
            # Convert to DataFrame and calculate volatility
            price_df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            price_df['date'] = pd.to_datetime(price_df['timestamp'], unit='ms')
            
            # Calculate rolling volatility (7-day)
            price_df['returns'] = price_df['price'].pct_change()
            price_df['volatility'] = price_df['returns'].rolling(7).std() * 100
            
            for _, row in price_df.iterrows():
                if not pd.isna(row['price']):
                    records.append({
                        'date': row['date'],
                        'country': None,
                        'series': f"{coin_id.upper()}-{currency.upper()}",
                        'value': row['price'],
                        'source': 'CoinGecko',
                        'metadata': {
                            'endpoint': response.url,
                            'last_refresh': datetime.now().isoformat(),
                            'coin_id': coin_id,
                            'currency': currency,
                            'volatility': row.get('volatility', np.nan)
                        }
                    })
    
    return pd.DataFrame(records)

# ============================================================================
# USGS EARTHQUAKE API
# ============================================================================

@api_fallback
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(requests.RequestException)
)
def get_usgs(min_mag: float = 5.5, window: str = "30d") -> pd.DataFrame:
    """
    Fetch earthquake data from USGS API
    
    Args:
        min_mag: Minimum magnitude
        window: Time window (e.g., "30d", "7d")
    
    Returns:
        DataFrame with standardized schema
    """
    config = get_config()
    api_config = config.get_api_config('usgs')
    
    # Calculate date range
    days = int(window.replace('d', ''))
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    url = f"{api_config.base_url}/query"
    params = {
        'format': 'geojson',
        'starttime': start_time.strftime('%Y-%m-%d'),
        'endtime': end_time.strftime('%Y-%m-%d'),
        'minmagnitude': min_mag
    }
    
    session = get_session()
    response = session.get(url, params=params, timeout=api_config.timeout_seconds)
    response.raise_for_status()
    
    data = response.json()
    
    records = []
    for feature in data.get('features', []):
        props = feature.get('properties', {})
        coords = feature.get('geometry', {}).get('coordinates', [])
        
        if props.get('mag') is not None:
            records.append({
                'date': pd.to_datetime(props['time'], unit='ms'),
                'country': None,
                'series': 'Earthquake',
                'value': float(props['mag']),
                'source': 'USGS',
                'metadata': {
                    'endpoint': response.url,
                    'last_refresh': datetime.now().isoformat(),
                    'place': props.get('place', ''),
                    'latitude': coords[1] if len(coords) > 1 else None,
                    'longitude': coords[0] if len(coords) > 0 else None,
                    'depth': coords[2] if len(coords) > 2 else None
                }
            })
    
    return pd.DataFrame(records)

# ============================================================================
# WIKIPEDIA PAGEVIEWS API
# ============================================================================

@api_fallback
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(requests.RequestException)
)
def get_wiki_views(term: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch Wikipedia pageview data
    
    Args:
        term: Wikipedia article title
        start: Start date (YYYYMMDD)
        end: End date (YYYYMMDD)
    
    Returns:
        DataFrame with standardized schema
    """
    config = get_config()
    api_config = config.get_api_config('wikipedia')
    
    url = f"{api_config.base_url}/per-article/en.wikipedia/all-access/user/{term}/daily/{start}/{end}"
    
    session = get_session()
    response = session.get(url, timeout=api_config.timeout_seconds)
    response.raise_for_status()
    
    data = response.json()
    
    records = []
    for item in data.get('items', []):
        records.append({
            'date': pd.to_datetime(item['timestamp'], format='%Y%m%d%H'),
            'country': None,
            'series': f"Wikipedia-{term}",
            'value': int(item['views']),
            'source': 'Wikipedia',
            'metadata': {
                'endpoint': response.url,
                'last_refresh': datetime.now().isoformat(),
                'article': term,
                'project': item.get('project', ''),
                'access': item.get('access', ''),
                'agent': item.get('agent', '')
            }
        })
    
    return pd.DataFrame(records)

# ============================================================================
# OPEN-METEO WEATHER API
# ============================================================================

@api_fallback
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(requests.RequestException)
)
def get_open_meteo(lat: float, lon: float, daily: List[str] = None, hourly: List[str] = None) -> pd.DataFrame:
    """
    Fetch weather data from Open-Meteo API
    
    Args:
        lat: Latitude
        lon: Longitude
        daily: List of daily variables
        hourly: List of hourly variables
    
    Returns:
        DataFrame with standardized schema
    """
    if daily is None:
        daily = ['temperature_2m_max', 'precipitation_sum']
    
    config = get_config()
    api_config = config.get_api_config('openmeteo')
    
    url = f"{api_config.base_url}/forecast"
    params = {
        'latitude': lat,
        'longitude': lon,
        'daily': ','.join(daily),
        'past_days': 30,
        'forecast_days': 7
    }
    
    if hourly:
        params['hourly'] = ','.join(hourly)
    
    session = get_session()
    response = session.get(url, params=params, timeout=api_config.timeout_seconds)
    response.raise_for_status()
    
    data = response.json()
    
    records = []
    daily_data = data.get('daily', {})
    
    if 'time' in daily_data:
        dates = [pd.to_datetime(d) for d in daily_data['time']]
        
        for var in daily:
            if var in daily_data:
                values = daily_data[var]
                for date, value in zip(dates, values):
                    if value is not None:
                        records.append({
                            'date': date,
                            'country': None,
                            'series': f"Weather-{var}",
                            'value': float(value),
                            'source': 'Open-Meteo',
                            'metadata': {
                                'endpoint': response.url,
                                'last_refresh': datetime.now().isoformat(),
                                'latitude': lat,
                                'longitude': lon,
                                'variable': var
                            }
                        })
    
    return pd.DataFrame(records)

# ============================================================================
# IMF API (Placeholder - Limited free access)
# ============================================================================

@api_fallback
def get_imf(db: str, key: str, params: Optional[Dict] = None) -> pd.DataFrame:
    """
    Fetch data from IMF API (limited free access)
    
    Args:
        db: Database name
        key: Series key
        params: Additional parameters
    
    Returns:
        DataFrame with standardized schema
    """
    # IMF API has limited free access, return synthetic data for now
    logger.info("IMF API has limited free access, returning synthetic data")
    return _generate_synthetic_data('imf', db, key, params)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_all_data(country: str = None) -> Dict[str, pd.DataFrame]:
    """
    Fetch data from all available APIs for a given country
    
    Args:
        country: ISO country code (defaults to config default)
    
    Returns:
        Dictionary with API names as keys and DataFrames as values
    """
    config = get_config()
    if country is None:
        country = config.default_country
    
    data = {}
    
    # World Bank GDP
    try:
        data['worldbank_gdp'] = get_worldbank(country, 'NY.GDP.MKTP.CD')
    except Exception as e:
        logger.error(f"Failed to fetch World Bank data: {e}")
    
    # CoinGecko crypto
    try:
        data['crypto'] = get_coingecko()
    except Exception as e:
        logger.error(f"Failed to fetch CoinGecko data: {e}")
    
    # USGS earthquakes
    try:
        data['earthquakes'] = get_usgs()
    except Exception as e:
        logger.error(f"Failed to fetch USGS data: {e}")
    
    # Wikipedia sentiment (example terms)
    try:
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        
        sentiment_data = []
        for term in ['Economy', 'Inflation', 'GDP']:
            term_data = get_wiki_views(term, start_date, end_date)
            sentiment_data.append(term_data)
        
        if sentiment_data:
            data['wikipedia_sentiment'] = pd.concat(sentiment_data, ignore_index=True)
    except Exception as e:
        logger.error(f"Failed to fetch Wikipedia data: {e}")
    
    return data

def validate_data_schema(df: pd.DataFrame) -> bool:
    """
    Validate that a DataFrame follows the standard schema
    
    Args:
        df: DataFrame to validate
    
    Returns:
        True if valid, False otherwise
    """
    required_columns = ['date', 'country', 'series', 'value', 'source', 'metadata']
    return all(col in df.columns for col in required_columns)

# Import StringIO for FRED CSV parsing
from io import StringIO
