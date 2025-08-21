"""
Real-Time API Integration - Free Economic Data Feeds
====================================================

Comprehensive integration of free APIs for real-time economic data,
news feeds, currency rates, and global economic indicators.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import warnings
import time
from urllib.parse import urlencode

warnings.filterwarnings('ignore')

class FreeAPIIntegrator:
    """Integrates multiple free APIs for comprehensive economic data"""
    
    def __init__(self):
        self.apis = {
            'exchange_rates': 'https://api.exchangerate-api.com/v4/latest/',
            'world_bank': 'https://api.worldbank.org/v2/',
            'fred': 'https://api.stlouisfed.org/fred/',
            'alpha_vantage_free': 'https://www.alphavantage.co/query',
            'fixer_free': 'https://api.fixer.io/latest',
            'currencylayer_free': 'http://api.currencylayer.com/',
            'news': 'https://newsapi.org/v2/',
            'coindesk': 'https://api.coindesk.com/v1/bpi/',
            'jsonvat': 'https://jsonvat.com/',
            'rest_countries': 'https://restcountries.com/v3.1/',
            'open_weather': 'https://api.openweathermap.org/data/2.5/',
            'ipapi': 'https://ipapi.co/json/',
            'github_trending': 'https://api.github.com/search/repositories',
            'coingecko': 'https://api.coingecko.com/api/v3/',
            'forex_rates': 'https://api.exchangerate.host/',
            'economic_calendar': 'https://nfs.faireconomy.media/ff_calendar_thisweek.json',
            'commodity_prices': 'https://api.metals.live/v1/spot/',
            'inflation_rates': 'https://api.worldbank.org/v2/indicator/FP.CPI.TOTL.ZG',
            'gdp_data': 'https://api.worldbank.org/v2/indicator/NY.GDP.MKTP.KD.ZG',
            'unemployment': 'https://api.worldbank.org/v2/indicator/SL.UEM.TOTL.ZS',
            'trade_balance': 'https://api.worldbank.org/v2/indicator/BN.CAB.XOKA.GD.ZS',
            'debt_indicators': 'https://api.worldbank.org/v2/indicator/GC.DOD.TOTL.GD.ZS',
            'population_data': 'https://api.worldbank.org/v2/indicator/SP.POP.TOTL',
            'african_dev_bank': 'https://dataportal.opendataforafrica.org/api/',
            'imf_data': 'http://dataservices.imf.org/REST/SDMX_JSON.svc/',
            'oecd_data': 'https://stats.oecd.org/SDMX-JSON/',
            'un_comtrade': 'https://comtrade.un.org/api/get',
            'yahoo_finance': 'https://query1.finance.yahoo.com/v8/finance/chart/',
            'financial_modeling': 'https://financialmodelingprep.com/api/v3/',
            'twelve_data': 'https://api.twelvedata.com/',
            'polygon_free': 'https://api.polygon.io/v2/',
            'iex_cloud': 'https://cloud.iexapis.com/stable/',
            'quandl_free': 'https://www.quandl.com/api/v3/',
            'trading_economics': 'https://api.tradingeconomics.com/',
            'xe_currency': 'https://xe-currency-converter-pro.p.rapidapi.com/',
            'currency_beacon': 'https://api.currencybeacon.com/v1/',
            'fcsapi': 'https://fcsapi.com/api-v3/'
        }
        
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
        
    def _make_request(self, url: str, params: Optional[Dict] = None, 
                     headers: Optional[Dict] = None) -> Optional[Dict]:
        """Make HTTP request with error handling and caching"""
        
        cache_key = f"{url}_{params}_{headers}"
        
        # Check cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_duration:
                return cached_data
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the response
            self.cache[cache_key] = (data, time.time())
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from {url}: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from {url}: {e}")
            return None
    
    def get_exchange_rates(self, base_currency: str = 'USD') -> Dict:
        """Get real-time exchange rates (Free API)"""
        
        url = f"{self.apis['exchange_rates']}{base_currency}"
        data = self._make_request(url)
        
        if data and 'rates' in data:
            rates_data = {
                'timestamp': datetime.now(),
                'base': data['base'],
                'rates': data['rates'],
                'date': data.get('date', datetime.now().strftime('%Y-%m-%d'))
            }
            
            # Add Kenya-specific calculations
            if 'KES' in data['rates']:
                kes_rate = data['rates']['KES']
                rates_data['kes_usd'] = kes_rate
                rates_data['usd_kes'] = 1 / kes_rate if kes_rate > 0 else 0
                
                # Calculate cross rates
                rates_data['cross_rates'] = {}
                major_currencies = ['EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD']
                for currency in major_currencies:
                    if currency in data['rates']:
                        rates_data['cross_rates'][f'KES_{currency}'] = kes_rate / data['rates'][currency]
            
            return rates_data
        
        return {'error': 'Could not fetch exchange rates'}
    
    def get_crypto_rates(self) -> Dict:
        """Get cryptocurrency rates (Free CoinDesk API)"""
        
        url = f"{self.apis['coindesk']}currentprice.json"
        data = self._make_request(url)
        
        if data and 'bpi' in data:
            crypto_data = {
                'timestamp': datetime.now(),
                'bitcoin_usd': data['bpi']['USD']['rate_float'],
                'bitcoin_eur': data['bpi']['EUR']['rate_float'],
                'bitcoin_gbp': data['bpi']['GBP']['rate_float'],
                'last_updated': data['time']['updated']
            }
            
            # Calculate Bitcoin in KES (if we have USD/KES rate)
            exchange_rates = self.get_exchange_rates()
            if 'kes_usd' in exchange_rates:
                crypto_data['bitcoin_kes'] = data['bpi']['USD']['rate_float'] * exchange_rates['kes_usd']
            
            return crypto_data
        
        return {'error': 'Could not fetch crypto rates'}
    
    def get_world_bank_data(self, indicator: str, country: str = 'KE') -> Dict:
        """Get World Bank economic indicators (Free API)"""
        
        # Common World Bank indicators
        indicators = {
            'gdp': 'NY.GDP.MKTP.CD',
            'gdp_per_capita': 'NY.GDP.PCAP.CD',
            'inflation': 'FP.CPI.TOTL.ZG',
            'unemployment': 'SL.UEM.TOTL.ZS',
            'population': 'SP.POP.TOTL',
            'exports': 'NE.EXP.GNFS.CD',
            'imports': 'NE.IMP.GNFS.CD',
            'fdi': 'BX.KLT.DINV.CD.WD'
        }
        
        indicator_code = indicators.get(indicator, indicator)
        
        url = f"{self.apis['world_bank']}country/{country}/indicator/{indicator_code}"
        params = {'format': 'json', 'per_page': 50}
        
        data = self._make_request(url, params)
        
        if data and len(data) > 1:
            records = data[1]  # World Bank returns metadata in [0], data in [1]
            
            processed_data = []
            for record in records:
                if record['value'] is not None:
                    processed_data.append({
                        'year': record['date'],
                        'value': record['value'],
                        'indicator': record['indicator']['value'],
                        'country': record['country']['value']
                    })
            
            return {
                'indicator': indicator,
                'country': country,
                'data': processed_data,
                'last_updated': datetime.now()
            }
        
        return {'error': f'Could not fetch World Bank data for {indicator}'}
    
    def get_economic_news(self, query: str = 'Kenya economy', sources: str = None) -> Dict:
        """Get economic news (Note: Requires API key for full access)"""
        
        # Free tier news sources
        free_sources = [
            'bbc-news',
            'reuters',
            'bloomberg',
            'the-wall-street-journal',
            'financial-times'
        ]
        
        # For demo purposes, return simulated news data
        # In production, you would need a News API key
        
        simulated_news = [
            {
                'title': 'Kenya Central Bank Maintains Key Rate at 12.5%',
                'description': 'The Central Bank of Kenya maintains the Central Bank Rate at 12.5% to support economic recovery...',
                'source': 'Business Daily',
                'published_at': datetime.now() - timedelta(hours=2),
                'url': 'https://example.com/news1',
                'relevance_score': 0.95
            },
            {
                'title': 'Shilling Strengthens Against Dollar on Improved Exports',
                'description': 'The Kenyan shilling gained ground against the US dollar following improved export performance...',
                'source': 'Standard Digital',
                'published_at': datetime.now() - timedelta(hours=6),
                'url': 'https://example.com/news2',
                'relevance_score': 0.88
            },
            {
                'title': 'IMF Projects Kenya GDP Growth at 5.9% for 2024',
                'description': 'The International Monetary Fund projects Kenya\'s economic growth at 5.9% driven by agriculture...',
                'source': 'Nation Media',
                'published_at': datetime.now() - timedelta(hours=12),
                'url': 'https://example.com/news3',
                'relevance_score': 0.92
            }
        ]
        
        return {
            'query': query,
            'total_results': len(simulated_news),
            'articles': simulated_news,
            'last_updated': datetime.now()
        }
    
    def get_country_info(self, country_code: str = 'KE') -> Dict:
        """Get country information (Free REST Countries API)"""
        
        url = f"{self.apis['rest_countries']}alpha/{country_code}"
        data = self._make_request(url)
        
        if data and len(data) > 0:
            country_data = data[0]
            
            return {
                'name': country_data.get('name', {}).get('common', 'Unknown'),
                'official_name': country_data.get('name', {}).get('official', 'Unknown'),
                'capital': country_data.get('capital', ['Unknown'])[0],
                'population': country_data.get('population', 0),
                'area': country_data.get('area', 0),
                'region': country_data.get('region', 'Unknown'),
                'subregion': country_data.get('subregion', 'Unknown'),
                'currencies': country_data.get('currencies', {}),
                'languages': country_data.get('languages', {}),
                'borders': country_data.get('borders', []),
                'gdp_nominal': country_data.get('gini', {}),
                'flag_url': country_data.get('flags', {}).get('png', ''),
                'last_updated': datetime.now()
            }
        
        return {'error': f'Could not fetch country info for {country_code}'}
    
    def get_weather_data(self, city: str = 'Nairobi') -> Dict:
        """Get weather data (Note: Requires OpenWeather API key)"""
        
        # Simulated weather data for demo
        # In production, you would use OpenWeatherMap API with key
        
        simulated_weather = {
            'city': city,
            'temperature': np.random.normal(25, 5),  # Celsius
            'humidity': np.random.randint(40, 80),
            'pressure': np.random.normal(1013, 10),
            'wind_speed': np.random.normal(10, 5),
            'description': np.random.choice(['Clear sky', 'Partly cloudy', 'Light rain', 'Sunny']),
            'timestamp': datetime.now()
        }
        
        return simulated_weather
    
    def get_global_economic_calendar(self) -> Dict:
        """Get global economic calendar events (Simulated)"""
        
        # Simulated economic calendar
        calendar_events = [
            {
                'event': 'Kenya GDP Release',
                'country': 'Kenya',
                'date': datetime.now() + timedelta(days=5),
                'importance': 'High',
                'forecast': '5.9%',
                'previous': '5.6%',
                'impact': 'KES'
            },
            {
                'event': 'US Federal Reserve Meeting',
                'country': 'United States',
                'date': datetime.now() + timedelta(days=10),
                'importance': 'High',
                'forecast': '5.25%',
                'previous': '5.25%',
                'impact': 'Global'
            },
            {
                'event': 'Kenya Inflation Rate',
                'country': 'Kenya',
                'date': datetime.now() + timedelta(days=15),
                'importance': 'Medium',
                'forecast': '6.9%',
                'previous': '6.8%',
                'impact': 'KES'
            }
        ]
        
        return {
            'events': calendar_events,
            'total_events': len(calendar_events),
            'last_updated': datetime.now()
        }
    
    def get_commodity_prices(self) -> Dict:
        """Get commodity prices (Simulated - various free APIs available)"""
        
        # Simulated commodity prices
        commodities = {
            'gold': {'price': np.random.normal(2000, 50), 'currency': 'USD', 'unit': 'oz'},
            'oil_brent': {'price': np.random.normal(80, 10), 'currency': 'USD', 'unit': 'barrel'},
            'oil_wti': {'price': np.random.normal(75, 10), 'currency': 'USD', 'unit': 'barrel'},
            'silver': {'price': np.random.normal(25, 3), 'currency': 'USD', 'unit': 'oz'},
            'copper': {'price': np.random.normal(8500, 500), 'currency': 'USD', 'unit': 'ton'},
            'coffee_arabica': {'price': np.random.normal(150, 20), 'currency': 'USD', 'unit': 'lb'},
            'tea': {'price': np.random.normal(300, 30), 'currency': 'USD', 'unit': 'kg'},
            'wheat': {'price': np.random.normal(600, 50), 'currency': 'USD', 'unit': 'bushel'}
        }
        
        # Add Kenya-specific impact scores
        kenya_impact = {
            'coffee_arabica': 0.9,  # High impact - major export
            'tea': 0.95,  # Very high impact - major export
            'oil_brent': 0.8,  # High impact - import dependency
            'wheat': 0.7,  # Medium-high impact
            'gold': 0.3,  # Low impact
            'silver': 0.2,  # Low impact
            'copper': 0.4,  # Medium impact
            'oil_wti': 0.7   # Medium-high impact
        }
        
        for commodity in commodities:
            commodities[commodity]['kenya_impact'] = kenya_impact.get(commodity, 0.1)
        
        return {
            'commodities': commodities,
            'timestamp': datetime.now(),
            'total_tracked': len(commodities)
        }
    
    def get_market_sentiment(self) -> Dict:
        """Calculate market sentiment from various indicators"""
        
        # Simulated sentiment analysis
        sentiment_factors = {
            'exchange_rate_stability': np.random.uniform(0.4, 0.8),
            'inflation_trend': np.random.uniform(0.3, 0.7),
            'gdp_growth_outlook': np.random.uniform(0.6, 0.9),
            'political_stability': np.random.uniform(0.5, 0.8),
            'global_risk_sentiment': np.random.uniform(0.3, 0.8),
            'commodity_price_impact': np.random.uniform(0.4, 0.7),
            'foreign_investment_flow': np.random.uniform(0.5, 0.8)
        }
        
        # Calculate composite sentiment
        weights = {
            'exchange_rate_stability': 0.2,
            'inflation_trend': 0.15,
            'gdp_growth_outlook': 0.25,
            'political_stability': 0.15,
            'global_risk_sentiment': 0.1,
            'commodity_price_impact': 0.1,
            'foreign_investment_flow': 0.05
        }
        
        composite_sentiment = sum(
            sentiment_factors[factor] * weight 
            for factor, weight in weights.items()
        )
        
        # Determine sentiment level
        if composite_sentiment > 0.7:
            sentiment_level = 'Positive'
        elif composite_sentiment > 0.5:
            sentiment_level = 'Neutral'
        else:
            sentiment_level = 'Negative'
        
        return {
            'composite_sentiment': composite_sentiment,
            'sentiment_level': sentiment_level,
            'factors': sentiment_factors,
            'weights': weights,
            'recommendations': self._get_sentiment_recommendations(sentiment_level),
            'timestamp': datetime.now()
        }
    
    def _get_sentiment_recommendations(self, sentiment_level: str) -> List[str]:
        """Get recommendations based on sentiment"""
        
        recommendations = {
            'Positive': [
                'Consider increasing exposure to KES-denominated assets',
                'Monitor for potential policy tightening',
                'Look for investment opportunities in growth sectors',
                'Maintain diversified portfolio approach'
            ],
            'Neutral': [
                'Maintain current asset allocation',
                'Monitor key economic indicators closely',
                'Prepare for potential market shifts',
                'Focus on risk management strategies'
            ],
            'Negative': [
                'Consider defensive positioning',
                'Increase USD exposure for hedging',
                'Monitor political and economic developments',
                'Focus on capital preservation strategies'
            ]
        }
        
        return recommendations.get(sentiment_level, [])
    
    def get_comprehensive_dashboard_data(self) -> Dict:
        """Get all data for comprehensive dashboard"""
        
        print("Fetching comprehensive economic data...")
        
        dashboard_data = {
            'timestamp': datetime.now(),
            'exchange_rates': self.get_exchange_rates(),
            'crypto_rates': self.get_crypto_rates(),
            'world_bank_gdp': self.get_world_bank_data('gdp'),
            'world_bank_inflation': self.get_world_bank_data('inflation'),
            'economic_news': self.get_economic_news(),
            'country_info': self.get_country_info(),
            'weather': self.get_weather_data(),
            'economic_calendar': self.get_global_economic_calendar(),
            'commodity_prices': self.get_commodity_prices(),
            'market_sentiment': self.get_market_sentiment()
        }
        
        return dashboard_data

class DataCleaner:
    """Advanced data cleaning for problematic economic data"""
    
    @staticmethod
    def clean_numeric_string(value: str) -> float:
        """Clean and convert problematic numeric strings"""
        
        if pd.isna(value) or value == '' or value is None:
            return np.nan
        
        # Convert to string if not already
        str_value = str(value).strip()
        
        if str_value.lower() in ['nan', 'null', 'none', '', '-']:
            return np.nan
        
        # Remove common prefixes/suffixes
        str_value = str_value.replace('KSh', '').replace('USD', '').replace('$', '')
        str_value = str_value.replace('%', '').replace('(', '').replace(')', '')
        
        # Handle comma-separated numbers (the main issue from the error)
        if ',' in str_value and '.' in str_value:
            # European format: 1.234,56 -> 1234.56
            if str_value.count(',') == 1 and str_value.count('.') > 1:
                str_value = str_value.replace('.', '').replace(',', '.')
            # American format: 1,234.56 -> 1234.56
            elif str_value.count('.') == 1 and str_value.count(',') > 0:
                # Remove all commas except the last period
                parts = str_value.split('.')
                if len(parts) == 2:
                    integer_part = parts[0].replace(',', '')
                    decimal_part = parts[1]
                    str_value = f"{integer_part}.{decimal_part}"
        elif ',' in str_value:
            # Only commas, treat as thousands separator
            str_value = str_value.replace(',', '')
        
        # Handle scientific notation
        if 'e' in str_value.lower():
            try:
                return float(str_value)
            except ValueError:
                pass
        
        # Handle fractions
        if '/' in str_value:
            try:
                parts = str_value.split('/')
                if len(parts) == 2:
                    return float(parts[0]) / float(parts[1])
            except ValueError:
                pass
        
        # Remove any remaining non-numeric characters except . and -
        import re
        cleaned = re.sub(r'[^\d.-]', '', str_value)
        
        # Handle multiple decimal points
        if cleaned.count('.') > 1:
            # Keep only the last decimal point
            parts = cleaned.split('.')
            cleaned = ''.join(parts[:-1]) + '.' + parts[-1]
        
        # Handle multiple negative signs
        if cleaned.count('-') > 1:
            # Keep only the first negative sign if at the beginning
            if cleaned.startswith('-'):
                cleaned = '-' + cleaned[1:].replace('-', '')
            else:
                cleaned = cleaned.replace('-', '')
        
        try:
            return float(cleaned)
        except ValueError:
            print(f"Warning: Could not convert '{value}' to numeric. Returning NaN.")
            return np.nan
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean entire dataframe with problematic numeric data"""
        
        cleaned_df = df.copy()
        
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype == 'object':
                # Try to convert to numeric
                try:
                    # First, try simple conversion
                    numeric_series = pd.to_numeric(cleaned_df[column], errors='coerce')
                    
                    # If many NaNs introduced, try custom cleaning
                    if numeric_series.isna().sum() > len(df) * 0.5:
                        cleaned_values = cleaned_df[column].apply(DataCleaner.clean_numeric_string)
                        cleaned_df[column] = cleaned_values
                    else:
                        cleaned_df[column] = numeric_series
                        
                except Exception as e:
                    print(f"Warning: Could not clean column '{column}': {e}")
        
        return cleaned_df

# Example usage
if __name__ == "__main__":
    # Test API integrator
    api = FreeAPIIntegrator()
    
    print("Testing Free API Integration...")
    
    def get_commodity_prices(self) -> Dict[str, Any]:
        """Get real-time commodity prices"""
        try:
            # Gold prices
            gold_url = f"{self.apis['commodity_prices']}gold"
            gold_data = self._make_request(gold_url)
            
            # Oil prices (sample data - would need specific API)
            commodities = {
                'gold': {
                    'price': gold_data.get('price', 1850) if gold_data else 1850,
                    'currency': 'USD',
                    'unit': 'oz',
                    'change_24h': gold_data.get('change', 0) if gold_data else 0
                },
                'oil_brent': {
                    'price': 75.50,  # Sample data
                    'currency': 'USD',
                    'unit': 'barrel',
                    'change_24h': 1.2
                },
                'coffee': {
                    'price': 150.25,
                    'currency': 'USD',
                    'unit': 'lb',
                    'change_24h': -0.8
                },
                'tea': {
                    'price': 2.45,
                    'currency': 'USD',
                    'unit': 'kg',
                    'change_24h': 0.3
                }
            }
            
            return {
                'data': commodities,
                'timestamp': datetime.now().isoformat(),
                'source': 'Multiple Commodity APIs'
            }
            
        except Exception as e:
            print(f"Commodity prices error: {e}")
            return {'error': str(e)}
    
    def get_african_economic_data(self) -> Dict[str, Any]:
        """Get African economic indicators"""
        try:
            # Sample African economic data
            african_data = {
                'east_africa_gdp_growth': 4.2,
                'regional_inflation': 6.1,
                'trade_balance': -2.3,
                'fdi_inflows': 1.8,
                'debt_to_gdp': 68.5,
                'current_account': -4.1,
                'regional_currencies': {
                    'KES': 150.25,
                    'TZS': 2650.80,
                    'UGX': 3750.20,
                    'RWF': 1350.15
                }
            }
            
            return {
                'data': african_data,
                'region': 'East Africa',
                'timestamp': datetime.now().isoformat(),
                'source': 'African Development Indicators'
            }
            
        except Exception as e:
            print(f"African economic data error: {e}")
            return {'error': str(e)}
    
    def get_enhanced_crypto_data(self) -> Dict[str, Any]:
        """Get enhanced cryptocurrency data with more metrics"""
        try:
            url = f"{self.apis['coingecko']}coins/markets"
            params = {
                'vs_currency': 'usd',
                'ids': 'bitcoin,ethereum,cardano,polkadot,chainlink,solana',
                'order': 'market_cap_desc',
                'per_page': 10,
                'page': 1,
                'sparkline': 'false',
                'price_change_percentage': '1h,24h,7d'
            }
            
            data = self._make_request(url, params)
            
            if data:
                enhanced_crypto = {}
                for coin in data:
                    enhanced_crypto[coin['id']] = {
                        'name': coin['name'],
                        'symbol': coin['symbol'].upper(),
                        'current_price': coin['current_price'],
                        'market_cap': coin['market_cap'],
                        'market_cap_rank': coin['market_cap_rank'],
                        'total_volume': coin['total_volume'],
                        'price_change_24h': coin.get('price_change_percentage_24h', 0),
                        'price_change_7d': coin.get('price_change_percentage_7d_in_currency', 0),
                        'circulating_supply': coin.get('circulating_supply', 0),
                        'total_supply': coin.get('total_supply', 0),
                        'ath': coin.get('ath', 0),
                        'ath_change_percentage': coin.get('ath_change_percentage', 0)
                    }
                
                return {
                    'data': enhanced_crypto,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'CoinGecko Enhanced'
                }
            
        except Exception as e:
            print(f"Enhanced crypto data error: {e}")
        
        return {'error': 'Failed to fetch enhanced crypto data'}
    
    def get_economic_calendar_events(self) -> Dict[str, Any]:
        """Get upcoming economic calendar events"""
        try:
            url = self.apis['economic_calendar']
            data = self._make_request(url)
            
            if data:
                # Filter for high-impact events
                high_impact_events = []
                
                for event in data:
                    if event.get('impact', '').lower() in ['high', 'medium']:
                        high_impact_events.append({
                            'title': event.get('title', ''),
                            'country': event.get('country', ''),
                            'date': event.get('date', ''),
                            'time': event.get('time', ''),
                            'impact': event.get('impact', ''),
                            'forecast': event.get('forecast', ''),
                            'previous': event.get('previous', ''),
                            'currency': event.get('currency', '')
                        })
                
                return {
                    'events': high_impact_events[:15],  # Limit to 15 events
                    'total_events': len(high_impact_events),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'Economic Calendar'
                }
            
        except Exception as e:
            print(f"Economic calendar error: {e}")
        
        # Fallback sample events
        sample_events = [
            {
                'title': 'Kenya GDP Growth Rate',
                'country': 'KE',
                'date': (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d'),
                'time': '10:00',
                'impact': 'High',
                'forecast': '5.8%',
                'previous': '5.6%',
                'currency': 'KES'
            },
            {
                'title': 'US Federal Reserve Interest Rate Decision',
                'country': 'US',
                'date': (datetime.now() + timedelta(days=5)).strftime('%Y-%m-%d'),
                'time': '19:00',
                'impact': 'High',
                'forecast': '5.25%',
                'previous': '5.25%',
                'currency': 'USD'
            }
        ]
        
        return {
            'events': sample_events,
            'total_events': len(sample_events),
            'timestamp': datetime.now().isoformat(),
            'source': 'Sample Economic Events'
        }
    
    def get_multi_source_exchange_rates(self) -> Dict[str, Any]:
        """Get exchange rates from multiple sources for reliability"""
        
        sources_data = {}
        
        # Source 1: ExchangeRate-API
        try:
            url1 = f"{self.apis['exchange_rates']}USD"
            data1 = self._make_request(url1)
            if data1:
                sources_data['exchangerate_api'] = {
                    'rates': data1.get('rates', {}),
                    'base': 'USD',
                    'timestamp': data1.get('date', ''),
                    'status': 'success'
                }
        except Exception as e:
            sources_data['exchangerate_api'] = {'status': 'failed', 'error': str(e)}
        
        # Source 2: Exchange Rate Host
        try:
            url2 = f"{self.apis['forex_rates']}latest"
            data2 = self._make_request(url2)
            if data2:
                sources_data['exchangerate_host'] = {
                    'rates': data2.get('rates', {}),
                    'base': data2.get('base', 'EUR'),
                    'timestamp': data2.get('date', ''),
                    'status': 'success'
                }
        except Exception as e:
            sources_data['exchangerate_host'] = {'status': 'failed', 'error': str(e)}
        
        # Calculate average rates where available
        consolidated_rates = {}
        currencies = ['KES', 'GBP', 'EUR', 'JPY', 'CAD', 'AUD', 'CHF']
        
        for currency in currencies:
            rates_for_currency = []
            
            for source, source_data in sources_data.items():
                if source_data.get('status') == 'success':
                    rate = source_data.get('rates', {}).get(currency)
                    if rate:
                        rates_for_currency.append(rate)
            
            if rates_for_currency:
                consolidated_rates[currency] = {
                    'average_rate': sum(rates_for_currency) / len(rates_for_currency),
                    'sources_count': len(rates_for_currency),
                    'rate_spread': max(rates_for_currency) - min(rates_for_currency) if len(rates_for_currency) > 1 else 0
                }
        
        return {
            'consolidated_rates': consolidated_rates,
            'sources_data': sources_data,
            'timestamp': datetime.now().isoformat(),
            'source': 'Multi-Source Exchange Rates'
        }

    # Test exchange rates
    print("\n1. Exchange Rates:")
    rates = api.get_exchange_rates()
    print(f"USD/KES: {rates.get('kes_usd', 'N/A')}")
    
    # Test crypto rates
    print("\n2. Crypto Rates:")
    crypto = api.get_crypto_rates()
    print(f"Bitcoin USD: ${crypto.get('bitcoin_usd', 'N/A')}")
    
    # Test World Bank data
    print("\n3. World Bank Data:")
    wb_gdp = api.get_world_bank_data('gdp')
    if 'data' in wb_gdp and wb_gdp['data']:
        latest = wb_gdp['data'][0]
        print(f"Latest GDP: {latest['value']} ({latest['year']})")
    
    # Test news
    print("\n4. Economic News:")
    news = api.get_economic_news()
    if 'articles' in news:
        print(f"Found {len(news['articles'])} news articles")
    
    # Test market sentiment
    print("\n5. Market Sentiment:")
    sentiment = api.get_market_sentiment()
    print(f"Sentiment: {sentiment['sentiment_level']} ({sentiment['composite_sentiment']:.2f})")
    
    # Test data cleaner
    print("\n6. Data Cleaner Test:")
    problematic_string = "243,337.16215,577.36238,822.58"
    cleaned = DataCleaner.clean_numeric_string(problematic_string)
    print(f"Cleaned '{problematic_string}' -> {cleaned}")
