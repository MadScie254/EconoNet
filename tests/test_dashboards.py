"""
Tests for EconoNet Dashboard Components
=====================================

Tests for dashboard imports, fallback behaviors, and UI components.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from econonet import set_mode, OperationMode

class TestDashboardImports:
    """Test that dashboard components can be imported without errors"""
    
    def test_econonet_package_import(self):
        """Test that the main package imports correctly"""
        import econonet
        
        # Test that main components are available
        assert hasattr(econonet, 'get_worldbank')
        assert hasattr(econonet, 'get_coingecko') 
        assert hasattr(econonet, 'OperationMode')
        assert hasattr(econonet, 'set_mode')
    
    def test_config_import(self):
        """Test configuration system import"""
        from econonet.config import EconoNetConfig, OperationMode
        
        config = EconoNetConfig()
        assert config.mode in [OperationMode.OFFLINE, OperationMode.LIVE, OperationMode.EXPERT]
    
    def test_live_apis_import(self):
        """Test live APIs import"""
        from econonet.live_apis import (
            get_worldbank, get_coingecko, get_usgs,
            validate_data_schema
        )
        
        # Test functions are callable
        assert callable(get_worldbank)
        assert callable(get_coingecko)
        assert callable(get_usgs)
        assert callable(validate_data_schema)
    
    def test_visual_components_import(self):
        """Test visual components import"""
        from econonet.visual import (
            create_sentiment_radar,
            create_provenance_footer,
            create_real_vs_synthetic_overlay
        )
        
        # Test functions are callable
        assert callable(create_sentiment_radar)
        assert callable(create_provenance_footer)
        assert callable(create_real_vs_synthetic_overlay)

class TestDashboardFallbacks:
    """Test dashboard behavior with API failures"""
    
    def setup_method(self):
        """Setup for each test"""
        self.original_mode = None
        try:
            from econonet import get_config
            self.original_mode = get_config().mode
        except:
            pass
    
    def teardown_method(self):
        """Cleanup after each test"""
        if self.original_mode:
            set_mode(self.original_mode)
    
    def test_offline_mode_fallbacks(self):
        """Test that offline mode uses synthetic data"""
        from econonet import get_worldbank, set_mode, OperationMode
        
        # Set to offline mode
        set_mode(OperationMode.OFFLINE)
        
        # Should return synthetic data
        result = get_worldbank('KE', 'NY.GDP.MKTP.CD')
        
        assert not result.empty
        assert result['source'].iloc[0] == 'Synthetic'
        assert result['metadata'].iloc[0].get('fallback', False)
    
    def test_live_mode_with_mock_failure(self):
        """Test live mode behavior when APIs fail"""
        from econonet import get_worldbank, set_mode, OperationMode
        
        # Set to live mode
        set_mode(OperationMode.LIVE)
        
        # Mock requests to fail
        with patch('requests.Session.get') as mock_get:
            mock_get.side_effect = Exception("Network error")
            
            # Should fall back to synthetic data
            result = get_worldbank('KE', 'NY.GDP.MKTP.CD')
            
            assert not result.empty
            assert result['source'].iloc[0] == 'Synthetic'
            assert result['metadata'].iloc[0].get('fallback', False)

class TestVisualComponents:
    """Test visual component creation"""
    
    def test_sentiment_radar_creation(self):
        """Test sentiment radar chart creation"""
        from econonet.visual import create_sentiment_radar
        import plotly.graph_objects as go
        
        sentiment_scores = {
            'Economic Attention': 0.7,
            'Crypto Fear/Greed': 0.3,
            'Global Risk': 0.6,
            'Market Volatility': 0.8
        }
        
        fig = create_sentiment_radar(sentiment_scores)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1  # Should have at least one trace
    
    def test_provenance_footer_creation(self):
        """Test provenance footer creation"""
        from econonet.visual import create_provenance_footer
        from datetime import datetime
        
        data_sources = [
            {'name': 'World Bank', 'url': 'https://api.worldbank.org', 'fallback': False},
            {'name': 'CoinGecko', 'url': 'https://api.coingecko.com', 'fallback': False}
        ]
        
        footer_html = create_provenance_footer(data_sources, datetime.now())
        
        assert isinstance(footer_html, str)
        assert 'World Bank' in footer_html
        assert 'CoinGecko' in footer_html
        assert 'Data Sources' in footer_html
    
    def test_real_vs_synthetic_overlay(self):
        """Test real vs synthetic data overlay"""
        from econonet.visual import create_real_vs_synthetic_overlay
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        
        # Create test data
        synthetic_data = pd.Series(np.random.normal(100, 10, 30))
        real_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30),
            'value': np.random.normal(105, 8, 30)
        })
        
        fig = create_real_vs_synthetic_overlay(synthetic_data, real_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Should have both synthetic and real traces

class TestDataValidation:
    """Test data validation and schema compliance"""
    
    def test_valid_schema_validation(self):
        """Test validation of correctly formatted data"""
        from econonet import validate_data_schema
        import pandas as pd
        
        valid_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'country': ['KE', 'KE', 'KE'],
            'series': ['TEST', 'TEST', 'TEST'],
            'value': [100.0, 101.0, 102.0],
            'source': ['Test', 'Test', 'Test'],
            'metadata': [{'test': True}] * 3
        })
        
        assert validate_data_schema(valid_df) is True
    
    def test_invalid_schema_validation(self):
        """Test validation of incorrectly formatted data"""
        from econonet import validate_data_schema
        import pandas as pd
        
        invalid_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'wrong_column': ['A', 'B', 'C'],
            'value': [100.0, 101.0, 102.0]
        })
        
        assert validate_data_schema(invalid_df) is False

class TestErrorHandling:
    """Test error handling in dashboard components"""
    
    def test_import_error_handling(self):
        """Test graceful handling of missing optional dependencies"""
        # Test that core functionality works even if optional packages are missing
        try:
            from econonet import get_worldbank, OperationMode, set_mode
            
            # Should not raise import errors
            set_mode(OperationMode.OFFLINE)
            result = get_worldbank('KE', 'NY.GDP.MKTP.CD')
            assert not result.empty
            
        except ImportError as e:
            pytest.fail(f"Core functionality should not require optional dependencies: {e}")
    
    def test_configuration_error_handling(self):
        """Test handling of configuration errors"""
        from econonet.config import EconoNetConfig
        
        # Should handle missing environment variables gracefully
        config = EconoNetConfig.from_env()
        assert config.mode in [OperationMode.OFFLINE, OperationMode.LIVE, OperationMode.EXPERT]

if __name__ == "__main__":
    pytest.main([__file__])
