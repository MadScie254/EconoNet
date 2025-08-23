"""
Tests for EconoNet Live API Adapters
===================================

Comprehensive tests for all API functions using mocked responses.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import responses
import json

# Import the functions we're testing
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from econonet import (
    get_worldbank, get_coingecko, get_usgs, get_wiki_views,
    get_open_meteo, get_fred_csv, validate_data_schema,
    set_mode, OperationMode
)

class TestAPIAdapters:
    
    def setup_method(self):
        """Setup for each test method"""
        # Set to live mode for testing
        set_mode(OperationMode.LIVE)
    
    @responses.activate
    def test_get_worldbank_success(self):
        """Test successful World Bank API call"""
        # Mock successful response
        mock_data = [
            {"page": 1, "pages": 1, "per_page": 50, "total": 2},
            [
                {
                    "indicator": {"id": "NY.GDP.MKTP.CD", "value": "GDP (current US$)"},
                    "country": {"id": "KE", "value": "Kenya"},
                    "countryiso3code": "KEN",
                    "date": "2023",
                    "value": 112500000000,
                    "unit": "",
                    "obs_status": "",
                    "decimal": 0
                },
                {
                    "indicator": {"id": "NY.GDP.MKTP.CD", "value": "GDP (current US$)"},
                    "country": {"id": "KE", "value": "Kenya"},
                    "countryiso3code": "KEN", 
                    "date": "2022",
                    "value": 110000000000,
                    "unit": "",
                    "obs_status": "",
                    "decimal": 0
                }
            ]
        ]
        
        responses.add(
            responses.GET,
            "https://api.worldbank.org/v2/country/KE/indicator/NY.GDP.MKTP.CD",
            json=mock_data,
            status=200
        )
        
        # Test the function
        result = get_worldbank('KE', 'NY.GDP.MKTP.CD')
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert validate_data_schema(result)
        assert result['country'].iloc[0] == 'KE'
        assert result['series'].iloc[0] == 'NY.GDP.MKTP.CD'
        assert result['source'].iloc[0] == 'World Bank'
        assert not result['metadata'].iloc[0].get('fallback', False)
    
    @responses.activate 
    def test_get_worldbank_failure_fallback(self):
        """Test World Bank API failure with fallback"""
        # Mock API failure
        responses.add(
            responses.GET,
            "https://api.worldbank.org/v2/country/KE/indicator/NY.GDP.MKTP.CD",
            json={"error": "Service unavailable"},
            status=500
        )
        
        # Test the function
        result = get_worldbank('KE', 'NY.GDP.MKTP.CD')
        
        # Should return synthetic data
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert validate_data_schema(result)
        assert result['source'].iloc[0] == 'Synthetic'
        assert result['metadata'].iloc[0].get('fallback', False)
    
    @responses.activate
    def test_get_coingecko_success(self):
        """Test successful CoinGecko API call"""
        # Mock successful response
        mock_data = {
            "prices": [
                [1691020800000, 60000.0],  # timestamp, price
                [1691107200000, 61000.0],
                [1691193600000, 59500.0]
            ]
        }
        
        responses.add(
            responses.GET,
            "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
            json=mock_data,
            status=200
        )
        
        # Test the function
        result = get_coingecko(['bitcoin'])
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert validate_data_schema(result)
        assert result['series'].iloc[0] == 'BITCOIN-USD'
        assert result['source'].iloc[0] == 'CoinGecko'
        # Check volatility calculation exists in metadata
        assert 'volatility' in result['metadata'].iloc[-1]
    
    @responses.activate
    def test_get_usgs_success(self):
        """Test successful USGS API call"""
        # Mock successful response
        mock_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "mag": 6.2,
                        "place": "Test Location",
                        "time": 1691020800000,
                        "updated": 1691020800000,
                        "tz": None,
                        "url": "https://earthquake.usgs.gov/earthquakes/eventpage/test123",
                        "detail": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/detail/test123.geojson",
                        "felt": None,
                        "cdi": None,
                        "mmi": None,
                        "alert": None,
                        "status": "reviewed",
                        "tsunami": 0,
                        "sig": 585,
                        "net": "us",
                        "code": "test123",
                        "ids": ",us_test123,",
                        "sources": ",us,",
                        "types": ",origin,phase-data,",
                        "nst": None,
                        "dmin": None,
                        "rms": None,
                        "gap": None,
                        "magType": "mww",
                        "type": "earthquake",
                        "title": "M 6.2 - Test Location"
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [-120.0, 35.0, 10.0]
                    }
                }
            ]
        }
        
        responses.add(
            responses.GET,
            "https://earthquake.usgs.gov/fdsnws/event/1/query",
            json=mock_data,
            status=200
        )
        
        # Test the function
        result = get_usgs(min_mag=5.0)
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert validate_data_schema(result)
        assert result['series'].iloc[0] == 'Earthquake'
        assert result['value'].iloc[0] == 6.2
        assert result['source'].iloc[0] == 'USGS'
        assert 'latitude' in result['metadata'].iloc[0]
        assert 'longitude' in result['metadata'].iloc[0]
    
    @responses.activate
    def test_get_wiki_views_success(self):
        """Test successful Wikipedia pageviews API call"""
        # Mock successful response
        mock_data = {
            "items": [
                {
                    "project": "en.wikipedia",
                    "article": "Economy",
                    "granularity": "daily",
                    "timestamp": "2024080100",
                    "access": "all-access",
                    "agent": "user",
                    "views": 1500
                },
                {
                    "project": "en.wikipedia", 
                    "article": "Economy",
                    "granularity": "daily",
                    "timestamp": "2024080200",
                    "access": "all-access",
                    "agent": "user",
                    "views": 1600
                }
            ]
        }
        
        responses.add(
            responses.GET,
            "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/Economy/daily/20240801/20240802",
            json=mock_data,
            status=200
        )
        
        # Test the function  
        result = get_wiki_views('Economy', '20240801', '20240802')
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert validate_data_schema(result)
        assert result['series'].iloc[0] == 'Wikipedia-Economy'
        assert result['source'].iloc[0] == 'Wikipedia'
        assert result['value'].iloc[0] == 1500
    
    @responses.activate
    def test_get_open_meteo_success(self):
        """Test successful Open-Meteo API call"""
        # Mock successful response
        mock_data = {
            "latitude": -1.2864,
            "longitude": 36.8172,
            "generationtime_ms": 0.123,
            "utc_offset_seconds": 10800,
            "timezone": "Africa/Nairobi",
            "timezone_abbreviation": "EAT",
            "elevation": 1795.0,
            "daily_units": {
                "time": "iso8601",
                "temperature_2m_max": "°C",
                "precipitation_sum": "mm"
            },
            "daily": {
                "time": ["2024-08-01", "2024-08-02", "2024-08-03"],
                "temperature_2m_max": [25.5, 26.2, 24.8],
                "precipitation_sum": [0.0, 2.5, 1.2]
            }
        }
        
        responses.add(
            responses.GET,
            "https://api.open-meteo.com/v1/forecast",
            json=mock_data,
            status=200
        )
        
        # Test the function
        result = get_open_meteo(-1.2864, 36.8172, ['temperature_2m_max', 'precipitation_sum'])
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 6  # 3 days * 2 variables
        assert validate_data_schema(result)
        assert result['source'].iloc[0] == 'Open-Meteo'
        assert 'Weather-temperature_2m_max' in result['series'].values
        assert 'Weather-precipitation_sum' in result['series'].values
    
    def test_offline_mode_synthetic_data(self):
        """Test that offline mode returns synthetic data"""
        # Set to offline mode
        set_mode(OperationMode.OFFLINE)
        
        # Test World Bank function
        result = get_worldbank('KE', 'NY.GDP.MKTP.CD')
        
        # Should return synthetic data
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert validate_data_schema(result)
        assert result['source'].iloc[0] == 'Synthetic'
        assert result['metadata'].iloc[0].get('fallback', False)
        
        # Reset to live mode
        set_mode(OperationMode.LIVE)
    
    def test_validate_data_schema_valid(self):
        """Test schema validation with valid data"""
        valid_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'country': ['KE', 'KE', 'KE'],
            'series': ['TEST', 'TEST', 'TEST'],
            'value': [100.0, 101.0, 102.0],
            'source': ['Test', 'Test', 'Test'],
            'metadata': [{'test': True}] * 3
        })
        
        assert validate_data_schema(valid_df)
    
    def test_validate_data_schema_invalid(self):
        """Test schema validation with invalid data"""
        invalid_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'wrong_column': ['A', 'B', 'C'],
            'value': [100.0, 101.0, 102.0]
        })
        
        assert not validate_data_schema(invalid_df)
    
    @responses.activate
    def test_api_timeout_fallback(self):
        """Test API timeout handling with fallback"""
        import requests
        
        # Mock timeout exception
        def request_callback(request):
            raise requests.exceptions.Timeout("Request timed out")
        
        responses.add_callback(
            responses.GET,
            "https://api.worldbank.org/v2/country/KE/indicator/NY.GDP.MKTP.CD",
            callback=request_callback
        )
        
        # Test the function
        result = get_worldbank('KE', 'NY.GDP.MKTP.CD')
        
        # Should return synthetic fallback data
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert result['source'].iloc[0] == 'Synthetic'
        assert result['metadata'].iloc[0].get('fallback', False)
    
    @responses.activate 
    def test_fred_csv_success(self):
        """Test successful FRED CSV endpoint"""
        # Mock CSV response
        csv_data = """DATE,GDPC1
2023-01-01,20000.0
2023-04-01,20100.0
2023-07-01,20200.0"""
        
        responses.add(
            responses.GET,
            "https://fred.stlouisfed.org/graph/fredgraph.csv",
            body=csv_data,
            status=200,
            content_type='text/csv'
        )
        
        # Test the function
        result = get_fred_csv('GDPC1')
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert validate_data_schema(result)
        assert result['series'].iloc[0] == 'GDPC1'
        assert result['country'].iloc[0] == 'US'  # FRED is US data
        assert result['source'].iloc[0] == 'FRED'

class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_multiple_api_calls(self):
        """Test calling multiple APIs and combining data"""
        # Set to offline mode for predictable results
        set_mode(OperationMode.OFFLINE)
        
        # Call multiple APIs
        gdp_data = get_worldbank('KE', 'NY.GDP.MKTP.CD')
        crypto_data = get_coingecko(['bitcoin'])
        earthquake_data = get_usgs()
        
        # Verify all return valid data
        assert validate_data_schema(gdp_data)
        assert validate_data_schema(crypto_data)
        assert validate_data_schema(earthquake_data)
        
        # Verify they're all synthetic in offline mode
        assert gdp_data['source'].iloc[0] == 'Synthetic'
        assert crypto_data['source'].iloc[0] == 'Synthetic'
        assert earthquake_data['source'].iloc[0] == 'Synthetic'
        
        # Reset mode
        set_mode(OperationMode.LIVE)
    
    def test_data_consistency_across_calls(self):
        """Test that synthetic data is consistent across multiple calls"""
        set_mode(OperationMode.OFFLINE)
        
        # Call same function twice
        data1 = get_worldbank('KE', 'NY.GDP.MKTP.CD')
        data2 = get_worldbank('KE', 'NY.GDP.MKTP.CD')
        
        # Should be identical (same random seed)
        pd.testing.assert_frame_equal(data1, data2)
        
        set_mode(OperationMode.LIVE)

if __name__ == "__main__":
    pytest.main([__file__])
