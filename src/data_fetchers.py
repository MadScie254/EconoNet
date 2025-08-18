"""
Lightweight data fetchers for free macro sources with caching.
Provides: World Bank and FRED fetch utilities (HTTP + caching to data/fetched/).
This is minimal and uses requests. For heavy use, swap to wbdata or fredapi.
"""
import os
import json
import time
from pathlib import Path

import requests

CACHE_DIR = Path(__file__).resolve().parents[1] / 'data' / 'fetched'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(name: str):
    return CACHE_DIR / f"{name}.json"


def fetch_world_bank_indicator(country_code: str, indicator: str, start_year: int = 2000, end_year: int = 2023, force_refresh: bool = False):
    """Fetch World Bank indicator and cache result to data/fetched.
    Returns JSON records (list of dicts) or None on failure.
    """
    cache_file = _cache_path(f"wb_{country_code}_{indicator}_{start_year}_{end_year}")
    if cache_file.exists() and not force_refresh:
        try:
            with cache_file.open('r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass

    url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}?date={start_year}:{end_year}&format=json&per_page=1000"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        # Save cache
        try:
            with cache_file.open('w', encoding='utf-8') as f:
                json.dump(data, f)
        except Exception:
            pass
        return data
    except Exception as e:
        print(f"World Bank fetch error: {e}")
        return None


def fetch_fred_series(series_id: str, api_key: str = None, force_refresh: bool = False):
    """Fetch FRED series using fred.stlouisfed.org API with optional API key.
    Caches to data/fetched.
    """
    cache_file = _cache_path(f"fred_{series_id}")
    if cache_file.exists() and not force_refresh:
        try:
            with cache_file.open('r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass

    if api_key is None:
        print("Warning: No FRED API key provided. Rate limits may apply and some series may be unavailable.")
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key or ''}&file_type=json"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        try:
            with cache_file.open('w', encoding='utf-8') as f:
                json.dump(data, f)
        except Exception:
            pass
        return data
    except Exception as e:
        print(f"FRED fetch error: {e}")
        return None
