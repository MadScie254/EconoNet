"""
Tests for EconoNet Configuration System
======================================

Tests for configuration management, environment variables, and mode switching.
"""

import pytest
import os
from unittest.mock import patch

# Import the functions we're testing
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from econonet.config import (
    OperationMode, APIConfig, EconoNetConfig,
    get_config, set_mode, is_live_mode, is_offline_mode, is_expert_mode
)

class TestOperationMode:
    """Test the OperationMode enum"""
    
    def test_operation_mode_values(self):
        """Test that operation modes have correct values"""
        assert OperationMode.OFFLINE.value == "offline"
        assert OperationMode.LIVE.value == "live"
        assert OperationMode.EXPERT.value == "expert"

class TestAPIConfig:
    """Test API configuration class"""
    
    def test_api_config_defaults(self):
        """Test default API configuration values"""
        config = APIConfig()
        
        assert config.enabled is True
        assert config.ttl_seconds == 3600
        assert config.timeout_seconds == 30
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.base_url == ""
    
    def test_api_config_custom_values(self):
        """Test custom API configuration values"""
        config = APIConfig(
            enabled=False,
            ttl_seconds=1800,
            timeout_seconds=60,
            max_retries=5,
            retry_delay=2.0,
            base_url="https://api.example.com"
        )
        
        assert config.enabled is False
        assert config.ttl_seconds == 1800
        assert config.timeout_seconds == 60
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.base_url == "https://api.example.com"

class TestEconoNetConfig:
    """Test main configuration class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = EconoNetConfig()
        
        assert config.mode == OperationMode.LIVE
        assert config.default_country == "KE"
        assert config.default_region == "Africa/Nairobi"
        assert config.cache_enabled is True
        assert config.show_provenance is True
        assert config.show_fallback_banners is True
        assert config.enable_sentiment_radar is True
        assert config.notebook_timeout == 300
        assert config.notebook_cache_results is True
    
    def test_api_configs_present(self):
        """Test that all expected API configurations are present"""
        config = EconoNetConfig()
        
        expected_apis = [
            'worldbank', 'ecb', 'fred', 'imf', 'coingecko',
            'openmeteo', 'wikipedia', 'usgs'
        ]
        
        for api_name in expected_apis:
            assert api_name in config.apis
            assert isinstance(config.apis[api_name], APIConfig)
    
    def test_get_api_config(self):
        """Test getting API configuration"""
        config = EconoNetConfig()
        
        # Test existing API
        wb_config = config.get_api_config('worldbank')
        assert isinstance(wb_config, APIConfig)
        assert wb_config.base_url == "https://api.worldbank.org/v2"
        
        # Test non-existing API (should return default)
        unknown_config = config.get_api_config('unknown_api')
        assert isinstance(unknown_config, APIConfig)
        assert unknown_config.base_url == ""  # Default value
    
    def test_is_api_enabled(self):
        """Test API enabled checking"""
        config = EconoNetConfig()
        
        # Test in live mode
        config.mode = OperationMode.LIVE
        assert config.is_api_enabled('worldbank') is True
        
        # Test in expert mode
        config.mode = OperationMode.EXPERT
        assert config.is_api_enabled('worldbank') is True
        
        # Test in offline mode
        config.mode = OperationMode.OFFLINE
        assert config.is_api_enabled('worldbank') is False
        
        # Test disabled API
        config.mode = OperationMode.LIVE
        config.apis['worldbank'].enabled = False
        assert config.is_api_enabled('worldbank') is False
    
    @patch.dict(os.environ, {
        'ECONET_MODE': 'expert',
        'ECONET_COUNTRY': 'NG',
        'ECONET_REGION': 'Africa/Lagos',
        'ECONET_CACHE': 'false',
        'ECONET_CACHE_DIR': '/custom/cache',
        'ECONET_SHOW_PROVENANCE': 'false',
        'ECONET_SHOW_FALLBACKS': 'false'
    })
    def test_from_env(self):
        """Test configuration from environment variables"""
        config = EconoNetConfig.from_env()
        
        assert config.mode == OperationMode.EXPERT
        assert config.default_country == 'NG'
        assert config.default_region == 'Africa/Lagos'
        assert config.cache_enabled is False
        assert config.cache_dir == '/custom/cache'
        assert config.show_provenance is False
        assert config.show_fallback_banners is False
    
    @patch.dict(os.environ, {'ECONET_MODE': 'invalid_mode'})
    def test_from_env_invalid_mode(self):
        """Test handling of invalid mode in environment"""
        config = EconoNetConfig.from_env()
        
        # Should fall back to default
        assert config.mode == OperationMode.LIVE
    
    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_no_vars(self):
        """Test configuration when no environment variables are set"""
        config = EconoNetConfig.from_env()
        
        # Should use defaults
        assert config.mode == OperationMode.LIVE
        assert config.default_country == "KE"
        assert config.cache_enabled is True

class TestGlobalFunctions:
    """Test global configuration functions"""
    
    def test_get_config(self):
        """Test getting global configuration"""
        config = get_config()
        assert isinstance(config, EconoNetConfig)
    
    def test_set_mode(self):
        """Test setting operation mode"""
        original_mode = get_config().mode
        
        try:
            # Test setting different modes
            set_mode(OperationMode.OFFLINE)
            assert get_config().mode == OperationMode.OFFLINE
            
            set_mode(OperationMode.EXPERT)
            assert get_config().mode == OperationMode.EXPERT
            
            set_mode(OperationMode.LIVE)
            assert get_config().mode == OperationMode.LIVE
        finally:
            # Restore original mode
            set_mode(original_mode)
    
    def test_mode_checking_functions(self):
        """Test mode checking utility functions"""
        original_mode = get_config().mode
        
        try:
            # Test offline mode
            set_mode(OperationMode.OFFLINE)
            assert is_offline_mode() is True
            assert is_live_mode() is False
            assert is_expert_mode() is False
            
            # Test live mode
            set_mode(OperationMode.LIVE)
            assert is_offline_mode() is False
            assert is_live_mode() is True
            assert is_expert_mode() is False
            
            # Test expert mode
            set_mode(OperationMode.EXPERT)
            assert is_offline_mode() is False
            assert is_live_mode() is True  # Expert mode includes live functionality
            assert is_expert_mode() is True
        finally:
            # Restore original mode
            set_mode(original_mode)

class TestConfigurationIntegration:
    """Integration tests for configuration system"""
    
    def test_api_config_modifications_persist(self):
        """Test that API configuration modifications persist"""
        config = get_config()
        original_ttl = config.apis['worldbank'].ttl_seconds
        
        try:
            # Modify configuration
            config.apis['worldbank'].ttl_seconds = 7200
            
            # Should persist in the same instance
            assert get_config().apis['worldbank'].ttl_seconds == 7200
        finally:
            # Restore original value
            config.apis['worldbank'].ttl_seconds = original_ttl
    
    def test_mode_affects_api_enabled_state(self):
        """Test that mode changes affect API enabled state"""
        config = get_config()
        original_mode = config.mode
        
        try:
            # Test that APIs are enabled in live mode
            config.mode = OperationMode.LIVE
            assert config.is_api_enabled('worldbank') is True
            assert config.is_api_enabled('coingecko') is True
            
            # Test that APIs are disabled in offline mode
            config.mode = OperationMode.OFFLINE
            assert config.is_api_enabled('worldbank') is False
            assert config.is_api_enabled('coingecko') is False
            
            # Test that APIs are enabled in expert mode
            config.mode = OperationMode.EXPERT
            assert config.is_api_enabled('worldbank') is True
            assert config.is_api_enabled('coingecko') is True
        finally:
            # Restore original mode
            config.mode = original_mode

if __name__ == "__main__":
    pytest.main([__file__])
