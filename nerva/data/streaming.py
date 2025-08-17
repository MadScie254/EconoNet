"""
NERVA Real-time Data Streaming and Updates
GODMODE_X: Live data integration pipeline
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import logging
from pathlib import Path
import json
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Configuration for a data source"""
    name: str
    url: str
    update_frequency: int  # minutes
    parser: str  # 'json', 'csv', 'xml'
    auth_required: bool = False
    headers: Dict[str, str] = None
    params: Dict[str, str] = None

class RealTimeDataStreamer:
    """
    Real-time data streaming and update manager
    """
    
    def __init__(self, update_callback: Optional[Callable] = None):
        self.data_sources = {}
        self.update_callback = update_callback
        self.running = False
        self.last_updates = {}
        self.data_cache = {}
        
        # Initialize data sources
        self._initialize_data_sources()
    
    def _initialize_data_sources(self):
        """Initialize available data sources"""
        
        # CBK Real-time data sources (simulated endpoints)
        self.data_sources = {
            'cbk_rates': DataSource(
                name='CBK Interest Rates',
                url='https://api.cbk.go.ke/rates/current',
                update_frequency=60,  # 1 hour
                parser='json',
                headers={'Accept': 'application/json'}
            ),
            'fx_rates': DataSource(
                name='Foreign Exchange Rates',
                url='https://api.cbk.go.ke/fx/current',
                update_frequency=30,  # 30 minutes
                parser='json'
            ),
            'interbank_rates': DataSource(
                name='Interbank Rates',
                url='https://api.cbk.go.ke/interbank/current',
                update_frequency=15,  # 15 minutes
                parser='json'
            ),
            'market_indicators': DataSource(
                name='Market Indicators',
                url='https://api.cbk.go.ke/market/indicators',
                update_frequency=60,  # 1 hour
                parser='json'
            )
        }
    
    async def fetch_data_source(self, source: DataSource) -> Optional[Dict]:
        """Fetch data from a single source"""
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                headers = source.headers or {}
                params = source.params or {}
                
                async with session.get(source.url, headers=headers, params=params) as response:
                    if response.status == 200:
                        if source.parser == 'json':
                            data = await response.json()
                        elif source.parser == 'csv':
                            text = await response.text()
                            # Convert CSV text to dict (simplified)
                            data = {'csv_data': text}
                        else:
                            data = {'raw_data': await response.text()}
                        
                        # Add metadata
                        data['_source'] = source.name
                        data['_timestamp'] = datetime.now().isoformat()
                        data['_status'] = 'success'
                        
                        return data
                    else:
                        logger.warning(f"HTTP {response.status} for {source.name}")
                        return None
                        
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching {source.name}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {source.name}: {e}")
            # Return simulated data for demonstration
            return self._generate_simulated_data(source)
    
    def _generate_simulated_data(self, source: DataSource) -> Dict:
        """Generate simulated data when real endpoints aren't available"""
        
        current_time = datetime.now()
        base_data = {
            '_source': source.name,
            '_timestamp': current_time.isoformat(),
            '_status': 'simulated'
        }
        
        if 'rates' in source.name.lower():
            # Simulate interest rate data
            base_rate = 7.0  # Base rate around 7%
            noise = np.random.normal(0, 0.1)  # Small random variation
            
            base_data.update({
                'central_bank_rate': base_rate + noise,
                'interbank_rate': base_rate + 0.5 + noise,
                'lending_rate': base_rate + 2.0 + noise,
                'deposit_rate': base_rate - 1.0 + noise,
                'last_updated': current_time.isoformat()
            })
            
        elif 'fx' in source.name.lower():
            # Simulate FX rate data
            base_usd_kes = 150.0  # Base USD/KES rate
            fx_noise = np.random.normal(0, 2.0)
            
            base_data.update({
                'USD_KES': base_usd_kes + fx_noise,
                'EUR_KES': (base_usd_kes + fx_noise) * 1.1,
                'GBP_KES': (base_usd_kes + fx_noise) * 1.25,
                'last_updated': current_time.isoformat()
            })
            
        elif 'market' in source.name.lower():
            # Simulate market indicators
            base_data.update({
                'nse_20_share_index': 1800 + np.random.normal(0, 50),
                'bond_yield_10yr': 12.5 + np.random.normal(0, 0.5),
                'treasury_bill_91day': 7.2 + np.random.normal(0, 0.2),
                'last_updated': current_time.isoformat()
            })
        
        return base_data
    
    async def update_all_sources(self):
        """Update all data sources that are due for refresh"""
        
        current_time = datetime.now()
        update_tasks = []
        
        for source_id, source in self.data_sources.items():
            last_update = self.last_updates.get(source_id, datetime.min)
            minutes_since_update = (current_time - last_update).total_seconds() / 60
            
            if minutes_since_update >= source.update_frequency:
                logger.info(f"Updating {source.name}...")
                task = self.fetch_data_source(source)
                update_tasks.append((source_id, task))
        
        if update_tasks:
            # Execute all updates concurrently
            results = await asyncio.gather(*[task for _, task in update_tasks], return_exceptions=True)
            
            # Process results
            for i, (source_id, _) in enumerate(update_tasks):
                result = results[i]
                
                if isinstance(result, Exception):
                    logger.error(f"Update failed for {source_id}: {result}")
                elif result is not None:
                    self.data_cache[source_id] = result
                    self.last_updates[source_id] = current_time
                    
                    # Trigger callback if provided
                    if self.update_callback:
                        try:
                            await self.update_callback(source_id, result)
                        except Exception as e:
                            logger.error(f"Update callback failed: {e}")
    
    async def start_streaming(self):
        """Start the real-time data streaming"""
        
        self.running = True
        logger.info("🚀 Starting real-time data streaming...")
        
        while self.running:
            try:
                await self.update_all_sources()
                
                # Wait before next update cycle
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in streaming loop: {e}")
                await asyncio.sleep(10)  # Brief pause before retry
    
    def stop_streaming(self):
        """Stop the real-time data streaming"""
        self.running = False
        logger.info("⏹️ Stopped real-time data streaming")
    
    def get_latest_data(self, source_id: Optional[str] = None) -> Dict:
        """Get the latest data from cache"""
        
        if source_id:
            return self.data_cache.get(source_id, {})
        else:
            return dict(self.data_cache)
    
    def get_data_freshness(self) -> Dict[str, Dict]:
        """Get information about data freshness"""
        
        current_time = datetime.now()
        freshness_info = {}
        
        for source_id, source in self.data_sources.items():
            last_update = self.last_updates.get(source_id)
            
            if last_update:
                age_minutes = (current_time - last_update).total_seconds() / 60
                is_stale = age_minutes > source.update_frequency * 1.5  # 50% tolerance
                
                freshness_info[source_id] = {
                    'source_name': source.name,
                    'last_update': last_update.isoformat(),
                    'age_minutes': age_minutes,
                    'is_stale': is_stale,
                    'expected_frequency': source.update_frequency
                }
            else:
                freshness_info[source_id] = {
                    'source_name': source.name,
                    'last_update': None,
                    'age_minutes': None,
                    'is_stale': True,
                    'expected_frequency': source.update_frequency
                }
        
        return freshness_info

class DataQualityMonitor:
    """
    Monitor data quality in real-time
    """
    
    def __init__(self):
        self.quality_metrics = {}
        self.alerts = []
        self.thresholds = {
            'missing_data_threshold': 0.1,  # 10% missing data threshold
            'outlier_threshold': 3.0,       # 3 standard deviations
            'staleness_threshold': 120      # 2 hours
        }
    
    def assess_data_quality(self, source_id: str, data: Dict) -> Dict[str, Any]:
        """Assess the quality of incoming data"""
        
        quality_report = {
            'source_id': source_id,
            'timestamp': datetime.now().isoformat(),
            'issues': [],
            'quality_score': 1.0
        }
        
        # Check for completeness
        if '_status' in data and data['_status'] == 'simulated':
            quality_report['issues'].append('Data is simulated, not real')
            quality_report['quality_score'] *= 0.8
        
        # Check for required fields
        numeric_fields = [k for k, v in data.items() 
                         if isinstance(v, (int, float)) and not k.startswith('_')]
        
        if len(numeric_fields) == 0:
            quality_report['issues'].append('No numeric data fields found')
            quality_report['quality_score'] *= 0.5
        
        # Check for outliers in numeric fields
        for field in numeric_fields:
            value = data[field]
            
            # Simple outlier detection (could be more sophisticated)
            if abs(value) > 1000000:  # Arbitrary large number check
                quality_report['issues'].append(f'Potential outlier in {field}: {value}')
                quality_report['quality_score'] *= 0.9
        
        # Check timestamp freshness
        if '_timestamp' in data:
            try:
                data_time = datetime.fromisoformat(data['_timestamp'])
                age_minutes = (datetime.now() - data_time).total_seconds() / 60
                
                if age_minutes > self.thresholds['staleness_threshold']:
                    quality_report['issues'].append(f'Data is stale: {age_minutes:.1f} minutes old')
                    quality_report['quality_score'] *= 0.7
                    
            except Exception as e:
                quality_report['issues'].append(f'Invalid timestamp format: {e}')
                quality_report['quality_score'] *= 0.8
        
        # Store quality metrics
        self.quality_metrics[source_id] = quality_report
        
        # Generate alerts for significant issues
        if quality_report['quality_score'] < 0.7:
            alert = {
                'severity': 'high' if quality_report['quality_score'] < 0.5 else 'medium',
                'source_id': source_id,
                'message': f"Data quality issues detected: {', '.join(quality_report['issues'])}",
                'timestamp': quality_report['timestamp'],
                'quality_score': quality_report['quality_score']
            }
            self.alerts.append(alert)
            
            # Keep only recent alerts
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.alerts = [a for a in self.alerts 
                          if datetime.fromisoformat(a['timestamp']) > cutoff_time]
        
        return quality_report
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get overall data quality summary"""
        
        if not self.quality_metrics:
            return {'status': 'no_data', 'overall_score': 0.0}
        
        # Calculate overall quality score
        scores = [metrics['quality_score'] for metrics in self.quality_metrics.values()]
        overall_score = np.mean(scores)
        
        # Count issues by severity
        recent_alerts = [a for a in self.alerts 
                        if datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=1)]
        
        high_severity_count = len([a for a in recent_alerts if a['severity'] == 'high'])
        medium_severity_count = len([a for a in recent_alerts if a['severity'] == 'medium'])
        
        return {
            'status': 'healthy' if overall_score > 0.8 else 'degraded' if overall_score > 0.6 else 'poor',
            'overall_score': overall_score,
            'sources_monitored': len(self.quality_metrics),
            'recent_alerts': len(recent_alerts),
            'high_severity_alerts': high_severity_count,
            'medium_severity_alerts': medium_severity_count
        }

# Factory functions
def create_data_streamer(update_callback: Optional[Callable] = None) -> RealTimeDataStreamer:
    """Create real-time data streamer"""
    return RealTimeDataStreamer(update_callback)

def create_quality_monitor() -> DataQualityMonitor:
    """Create data quality monitor"""
    return DataQualityMonitor()
