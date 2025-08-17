"""
NERVA Real-time Data Streaming Engine
Professional-grade real-time data processing and monitoring
"""

import asyncio
import threading
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import queue
import logging

class RealTimeDataStreamer:
    """Professional real-time data streaming system"""
    
    def __init__(self, update_interval=1.0):
        self.update_interval = update_interval
        self.data_queue = queue.Queue(maxsize=1000)
        self.subscribers = []
        self.is_streaming = False
        self.stream_thread = None
        self.latest_data = {}
        
        # Data generators
        self.data_generators = {
            'cbr_rate': self._generate_cbr_data,
            'inflation_rate': self._generate_inflation_data,
            'fx_rate': self._generate_fx_data,
            'market_sentiment': self._generate_sentiment_data,
            'liquidity_index': self._generate_liquidity_data
        }
        
        # Initialize baseline values
        self.baseline_values = {
            'cbr_rate': 7.5,
            'inflation_rate': 5.2,
            'fx_rate': 115.0,
            'market_sentiment': 0.0,
            'liquidity_index': 1.0
        }
        
        self.current_values = self.baseline_values.copy()
        
    def _generate_cbr_data(self):
        """Generate realistic CBR rate changes"""
        # CBR changes are infrequent and deliberate
        if np.random.random() < 0.001:  # 0.1% chance of change
            change = np.random.choice([-0.25, 0.25], p=[0.6, 0.4])  # 25bps steps
            self.current_values['cbr_rate'] = max(0.5, 
                min(15.0, self.current_values['cbr_rate'] + change))
        
        return self.current_values['cbr_rate']
    
    def _generate_inflation_data(self):
        """Generate inflation rate updates"""
        # Monthly inflation with some daily noise
        daily_noise = np.random.normal(0, 0.01)
        trend = np.sin(time.time() / 86400 / 30) * 0.5  # Monthly seasonality
        
        self.current_values['inflation_rate'] = max(0, 
            self.baseline_values['inflation_rate'] + trend + daily_noise)
        
        return self.current_values['inflation_rate']
    
    def _generate_fx_data(self):
        """Generate real-time FX rate updates"""
        # FX rates have higher frequency changes
        volatility = 0.1  # Daily volatility
        dt = self.update_interval / 86400  # Convert to day fraction
        
        change = np.random.normal(0, volatility * np.sqrt(dt))
        self.current_values['fx_rate'] = max(80, 
            min(150, self.current_values['fx_rate'] * (1 + change)))
        
        return self.current_values['fx_rate']
    
    def _generate_sentiment_data(self):
        """Generate market sentiment indicator"""
        # Mean-reverting sentiment
        mean_reversion = -0.1 * self.current_values['market_sentiment']
        noise = np.random.normal(0, 0.05)
        
        self.current_values['market_sentiment'] = np.clip(
            self.current_values['market_sentiment'] + mean_reversion + noise,
            -1.0, 1.0
        )
        
        return self.current_values['market_sentiment']
    
    def _generate_liquidity_data(self):
        """Generate liquidity index"""
        # Liquidity varies with market hours and sentiment
        base_liquidity = 1.0
        sentiment_impact = 0.2 * self.current_values['market_sentiment']
        noise = np.random.normal(0, 0.05)
        
        self.current_values['liquidity_index'] = max(0.1,
            base_liquidity + sentiment_impact + noise)
        
        return self.current_values['liquidity_index']
    
    def _generate_data_point(self):
        """Generate a complete data point"""
        timestamp = datetime.now()
        
        data_point = {
            'timestamp': timestamp,
            'cbr_rate': self._generate_cbr_data(),
            'inflation_rate': self._generate_inflation_data(),
            'fx_rate': self._generate_fx_data(),
            'market_sentiment': self._generate_sentiment_data(),
            'liquidity_index': self._generate_liquidity_data()
        }
        
        # Add derived metrics
        data_point['fx_change'] = ((data_point['fx_rate'] - self.baseline_values['fx_rate']) / 
                                  self.baseline_values['fx_rate'] * 100)
        
        data_point['policy_pressure'] = (data_point['inflation_rate'] - 5.0) * 0.5  # Target inflation 5%
        
        return data_point
    
    def _stream_worker(self):
        """Background worker for data streaming"""
        while self.is_streaming:
            try:
                data_point = self._generate_data_point()
                
                # Update latest data
                self.latest_data = data_point
                
                # Add to queue
                if not self.data_queue.full():
                    self.data_queue.put(data_point)
                
                # Notify subscribers
                for callback in self.subscribers:
                    try:
                        callback(data_point)
                    except Exception as e:
                        logging.error(f"Error in subscriber callback: {e}")
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logging.error(f"Error in stream worker: {e}")
                time.sleep(1)
    
    def start_streaming(self):
        """Start real-time data streaming"""
        if self.is_streaming:
            return
            
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.stream_thread.start()
        
        logging.info("Real-time data streaming started")
    
    def stop_streaming(self):
        """Stop real-time data streaming"""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2)
        
        logging.info("Real-time data streaming stopped")
    
    def subscribe(self, callback):
        """Subscribe to real-time data updates"""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback):
        """Unsubscribe from real-time data updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def get_latest_data(self):
        """Get the most recent data point"""
        return self.latest_data.copy() if self.latest_data else None
    
    def get_historical_buffer(self, n_points=100):
        """Get recent historical data from buffer"""
        data_points = []
        temp_queue = queue.Queue()
        
        # Extract data from queue
        while not self.data_queue.empty() and len(data_points) < n_points:
            try:
                item = self.data_queue.get_nowait()
                data_points.append(item)
                temp_queue.put(item)
            except queue.Empty:
                break
        
        # Put data back in queue
        while not temp_queue.empty():
            self.data_queue.put(temp_queue.get_nowait())
        
        return pd.DataFrame(data_points) if data_points else pd.DataFrame()

class DataQualityMonitor:
    """Monitor data quality and system health"""
    
    def __init__(self, streamer):
        self.streamer = streamer
        self.quality_metrics = {}
        self.alerts = []
        
    def check_data_quality(self, data_point):
        """Check data quality for a single data point"""
        quality_score = 1.0
        issues = []
        
        # Check for missing values
        missing_count = sum(1 for v in data_point.values() if v is None or pd.isna(v))
        if missing_count > 0:
            quality_score -= 0.2 * missing_count
            issues.append(f"Missing values: {missing_count}")
        
        # Check for outliers
        if data_point.get('fx_rate', 0) > 200 or data_point.get('fx_rate', 0) < 50:
            quality_score -= 0.3
            issues.append("FX rate outlier detected")
        
        if data_point.get('inflation_rate', 0) > 50 or data_point.get('inflation_rate', 0) < -10:
            quality_score -= 0.3
            issues.append("Inflation rate outlier detected")
        
        # Check timestamp freshness
        if 'timestamp' in data_point:
            age_seconds = (datetime.now() - data_point['timestamp']).total_seconds()
            if age_seconds > 10:  # Data older than 10 seconds
                quality_score -= 0.2
                issues.append(f"Stale data: {age_seconds:.1f}s old")
        
        return {
            'quality_score': max(0, quality_score),
            'issues': issues,
            'timestamp': datetime.now()
        }
    
    def generate_quality_report(self):
        """Generate data quality report"""
        recent_data = self.streamer.get_historical_buffer(50)
        
        if recent_data.empty:
            return {
                'overall_quality': 0.0,
                'total_points': 0,
                'avg_quality': 0.0,
                'alerts': ["No data available"]
            }
        
        # Calculate quality metrics
        quality_scores = []
        all_issues = []
        
        for _, row in recent_data.iterrows():
            quality_check = self.check_data_quality(row.to_dict())
            quality_scores.append(quality_check['quality_score'])
            all_issues.extend(quality_check['issues'])
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        # Generate alerts for low quality
        alerts = []
        if avg_quality < 0.8:
            alerts.append(f"Low data quality detected: {avg_quality:.2f}")
        
        if len(set(all_issues)) > 3:
            alerts.append(f"Multiple data issues: {len(set(all_issues))} types")
        
        return {
            'overall_quality': avg_quality,
            'total_points': len(recent_data),
            'quality_scores': quality_scores,
            'common_issues': list(set(all_issues)),
            'alerts': alerts,
            'last_updated': datetime.now()
        }

# Global instances
_streamer_instance = None
_quality_monitor_instance = None

def get_real_time_streamer():
    """Get singleton real-time data streamer"""
    global _streamer_instance
    if _streamer_instance is None:
        _streamer_instance = RealTimeDataStreamer()
    return _streamer_instance

def get_quality_monitor():
    """Get singleton quality monitor"""
    global _quality_monitor_instance, _streamer_instance
    if _quality_monitor_instance is None:
        if _streamer_instance is None:
            _streamer_instance = RealTimeDataStreamer()
        _quality_monitor_instance = DataQualityMonitor(_streamer_instance)
    return _quality_monitor_instance
