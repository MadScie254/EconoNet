"""
EconoNet Configuration Management
================================

Centralized configuration for API modes, caching, and data sources.
Supports three modes: offline (demo), live (APIs), and expert (advanced features).
"""

import os
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import timedelta

class OperationMode(Enum):
    """Operation modes for EconoNet"""
    OFFLINE = "offline"  # Demo mode with synthetic data
    LIVE = "live"        # Live API data with fallbacks
    EXPERT = "expert"    # Advanced features + live data

@dataclass
class APIConfig:
    """Configuration for a specific API endpoint"""
    enabled: bool = True
    ttl_seconds: int = 3600  # 1 hour default
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    base_url: str = ""
    
@dataclass
class EconoNetConfig:
    """Main configuration class for EconoNet"""
    
    # Operation mode
    mode: OperationMode = OperationMode.LIVE
    
    # Default country/region settings
    default_country: str = "KE"  # Kenya ISO code
    default_region: str = "Africa/Nairobi"
    
    # Cache settings
    cache_enabled: bool = True
    cache_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), ".econet_cache"))
    
    # API configurations
    apis: Dict[str, APIConfig] = field(default_factory=lambda: {
        "worldbank": APIConfig(
            base_url="https://api.worldbank.org/v2",
            ttl_seconds=3600,  # 1 hour for macro data
            timeout_seconds=30
        ),
        "ecb": APIConfig(
            base_url="https://data-api.ecb.europa.eu/service/data",
            ttl_seconds=1800,  # 30 min for FX data
            timeout_seconds=20
        ),
        "fred": APIConfig(
            base_url="https://fred.stlouisfed.org/graph/fredgraph.csv",
            ttl_seconds=3600,  # 1 hour for economic indicators
            timeout_seconds=25
        ),
        "imf": APIConfig(
            base_url="https://www.imf.org/external/datamapper/api/v1",
            ttl_seconds=7200,  # 2 hours for IMF data
            timeout_seconds=30
        ),
        "coingecko": APIConfig(
            base_url="https://api.coingecko.com/api/v3",
            ttl_seconds=300,   # 5 min for crypto data
            timeout_seconds=15
        ),
        "openmeteo": APIConfig(
            base_url="https://api.open-meteo.com/v1",
            ttl_seconds=1800,  # 30 min for weather data
            timeout_seconds=20
        ),
        "wikipedia": APIConfig(
            base_url="https://wikimedia.org/api/rest_v1/metrics/pageviews",
            ttl_seconds=3600,  # 1 hour for pageview data
            timeout_seconds=25
        ),
        "usgs": APIConfig(
            base_url="https://earthquake.usgs.gov/fdsnws/event/1",
            ttl_seconds=1800,  # 30 min for earthquake data
            timeout_seconds=20
        )
    })
    
    # UI settings
    show_provenance: bool = True
    show_fallback_banners: bool = True
    enable_sentiment_radar: bool = True
    
    # Notebook integration
    notebook_timeout: int = 300  # 5 minutes
    notebook_cache_results: bool = True
    
    @classmethod
    def from_env(cls) -> 'EconoNetConfig':
        """Create configuration from environment variables"""
        config = cls()
        
        # Mode from environment
        mode_str = os.getenv("ECONET_MODE", "live").lower()
        if mode_str in [m.value for m in OperationMode]:
            config.mode = OperationMode(mode_str)
        
        # Country/region settings
        config.default_country = os.getenv("ECONET_COUNTRY", "KE")
        config.default_region = os.getenv("ECONET_REGION", "Africa/Nairobi")
        
        # Cache settings
        config.cache_enabled = os.getenv("ECONET_CACHE", "true").lower() == "true"
        config.cache_dir = os.getenv("ECONET_CACHE_DIR", config.cache_dir)
        
        # UI settings
        config.show_provenance = os.getenv("ECONET_SHOW_PROVENANCE", "true").lower() == "true"
        config.show_fallback_banners = os.getenv("ECONET_SHOW_FALLBACKS", "true").lower() == "true"
        
        return config
    
    def get_api_config(self, api_name: str) -> APIConfig:
        """Get configuration for a specific API"""
        return self.apis.get(api_name, APIConfig())
    
    def is_api_enabled(self, api_name: str) -> bool:
        """Check if an API is enabled"""
        if self.mode == OperationMode.OFFLINE:
            return False
        return self.get_api_config(api_name).enabled

# Global configuration instance
config = EconoNetConfig.from_env()

# Convenience functions
def get_config() -> EconoNetConfig:
    """Get the global configuration instance"""
    return config

def set_mode(mode: OperationMode) -> None:
    """Set the operation mode"""
    global config
    config.mode = mode

def is_live_mode() -> bool:
    """Check if we're in live API mode"""
    return config.mode in [OperationMode.LIVE, OperationMode.EXPERT]

def is_offline_mode() -> bool:
    """Check if we're in offline demo mode"""
    return config.mode == OperationMode.OFFLINE

def is_expert_mode() -> bool:
    """Check if we're in expert mode"""
    return config.mode == OperationMode.EXPERT
