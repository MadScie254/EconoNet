"""
NERVA Advanced Feature Engineering
GODMODE_X: Automated feature generation pipeline
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import logging

logger = logging.getLogger(__name__)

class AdvancedFeatureEngine:
    """
    Automated feature engineering with temporal, cross-sectional, and derived features
    """
    
    def __init__(self):
        self.scalers = {}
        self.feature_importance = {}
        self.generated_features = {}
    
    def create_temporal_features(self, df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
        """Create comprehensive temporal features"""
        
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        
        # Lag features
        for lag in [1, 3, 6, 12]:
            df[f'{value_col}_lag_{lag}'] = df[value_col].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12, 24]:
            df[f'{value_col}_ma_{window}'] = df[value_col].rolling(window).mean()
            df[f'{value_col}_std_{window}'] = df[value_col].rolling(window).std()
            df[f'{value_col}_min_{window}'] = df[value_col].rolling(window).min()
            df[f'{value_col}_max_{window}'] = df[value_col].rolling(window).max()
        
        # Momentum features
        df[f'{value_col}_momentum_3'] = df[value_col] / df[value_col].shift(3) - 1
        df[f'{value_col}_momentum_12'] = df[value_col] / df[value_col].shift(12) - 1
        
        # Volatility features
        df[f'{value_col}_volatility_12'] = df[value_col].rolling(12).std()
        df[f'{value_col}_volatility_24'] = df[value_col].rolling(24).std()
        
        # Trend features
        for window in [6, 12]:
            x = np.arange(window)
            df[f'{value_col}_trend_{window}'] = df[value_col].rolling(window).apply(
                lambda y: np.polyfit(x, y, 1)[0] if len(y) == window else np.nan, raw=False
            )
        
        # Seasonal decomposition features
        if len(df) >= 24:
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                decomp = seasonal_decompose(df[value_col].dropna(), period=12, extrapolate_trend='freq')
                df[f'{value_col}_trend'] = decomp.trend
                df[f'{value_col}_seasonal'] = decomp.seasonal
                df[f'{value_col}_residual'] = decomp.resid
            except Exception as e:
                logger.warning(f"Seasonal decomposition failed: {e}")
        
        # Cyclical features
        df[f'{value_col}_month'] = df.index.month
        df[f'{value_col}_quarter'] = df.index.quarter
        df[f'{value_col}_year'] = df.index.year
        
        # Month and quarter cyclical encoding
        df[f'{value_col}_month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df[f'{value_col}_month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        df[f'{value_col}_quarter_sin'] = np.sin(2 * np.pi * df.index.quarter / 4)
        df[f'{value_col}_quarter_cos'] = np.cos(2 * np.pi * df.index.quarter / 4)
        
        return df
    
    def create_cross_sectional_features(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create features from multiple datasets"""
        
        combined_features = pd.DataFrame()
        
        # Collect all numeric features
        for dataset_name, df in datasets.items():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if not col.startswith('_') and df[col].notna().sum() > 10:
                    series = df[col].dropna()
                    
                    # Basic statistics
                    feature_name = f"{dataset_name}_{col}"
                    if len(combined_features) == 0:
                        combined_features = pd.DataFrame(index=range(len(series)))
                    
                    # Ensure same length
                    target_length = len(combined_features)
                    if len(series) < target_length:
                        series = series.reindex(range(target_length), method='ffill')
                    else:
                        series = series.iloc[:target_length]
                    
                    combined_features[feature_name] = series.values
        
        # Cross-dataset ratios
        feature_names = list(combined_features.columns)
        for i, col1 in enumerate(feature_names[:10]):  # Limit to avoid explosion
            for j, col2 in enumerate(feature_names[i+1:11]):  # Limit pairs
                if combined_features[col2].std() > 0:
                    ratio_name = f"ratio_{col1}_to_{col2}"
                    combined_features[ratio_name] = combined_features[col1] / (combined_features[col2] + 1e-8)
        
        # PCA features
        if len(combined_features.columns) > 5:
            # Clean data for PCA
            clean_data = combined_features.fillna(combined_features.mean())
            clean_data = clean_data.replace([np.inf, -np.inf], np.nan).fillna(clean_data.mean())
            
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(clean_data)
            
            pca = PCA(n_components=min(5, len(clean_data.columns)))
            pca_features = pca.fit_transform(scaled_data)
            
            for i in range(pca_features.shape[1]):
                combined_features[f'pca_component_{i}'] = pca_features[:, i]
        
        return combined_features
    
    def create_economic_indicators(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create derived economic indicators"""
        
        indicators = pd.DataFrame()
        
        # Key economic relationships
        economic_formulas = {
            'fx_volatility': lambda: self._calculate_fx_volatility(datasets),
            'yield_spread': lambda: self._calculate_yield_spread(datasets),
            'liquidity_ratio': lambda: self._calculate_liquidity_ratio(datasets),
            'trade_balance_ratio': lambda: self._calculate_trade_balance(datasets),
            'inflation_proxy': lambda: self._calculate_inflation_proxy(datasets),
        }
        
        for indicator_name, calc_func in economic_formulas.items():
            try:
                indicator_value = calc_func()
                if indicator_value is not None:
                    indicators[indicator_name] = indicator_value
            except Exception as e:
                logger.warning(f"Failed to calculate {indicator_name}: {e}")
        
        return indicators
    
    def _calculate_fx_volatility(self, datasets: Dict[str, pd.DataFrame]) -> Optional[np.ndarray]:
        """Calculate FX volatility from exchange rate data"""
        for key in ['monthly_exchange_rate_end_period', 'fx_rate', 'exchange_rate']:
            if key in datasets:
                df = datasets[key]
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    rates = df[numeric_cols[0]].dropna()
                    returns = rates.pct_change().dropna()
                    volatility = returns.rolling(12).std() * np.sqrt(12)
                    return volatility.values
        return None
    
    def _calculate_yield_spread(self, datasets: Dict[str, pd.DataFrame]) -> Optional[np.ndarray]:
        """Calculate yield spread between different instruments"""
        tbill_data = None
        tbond_data = None
        
        for key in ['issues_of_treasury_bills', 'treasury_bills']:
            if key in datasets:
                df = datasets[key]
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    tbill_data = df[numeric_cols[0]].dropna()
                    break
        
        for key in ['issues_of_treasury_bonds', 'treasury_bonds']:
            if key in datasets:
                df = datasets[key]
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    tbond_data = df[numeric_cols[0]].dropna()
                    break
        
        if tbill_data is not None and tbond_data is not None:
            min_len = min(len(tbill_data), len(tbond_data))
            spread = tbond_data.iloc[:min_len] - tbill_data.iloc[:min_len]
            return spread.values
        
        return None
    
    def _calculate_liquidity_ratio(self, datasets: Dict[str, pd.DataFrame]) -> Optional[np.ndarray]:
        """Calculate banking system liquidity ratio"""
        for key in ['repo_and_reverse_repo_', 'liquidity', 'interbank']:
            if key in datasets:
                df = datasets[key]
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    # Ratio of two related liquidity measures
                    ratio = df[numeric_cols[0]] / (df[numeric_cols[1]] + 1e-8)
                    return ratio.dropna().values
        return None
    
    def _calculate_trade_balance(self, datasets: Dict[str, pd.DataFrame]) -> Optional[np.ndarray]:
        """Calculate trade balance proxy"""
        for key in ['foreign_trade_summary_ksh_million', 'trade_summary', 'trade']:
            if key in datasets:
                df = datasets[key]
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    # Assume first two columns are exports and imports
                    balance = df[numeric_cols[0]] - df[numeric_cols[1]]
                    return balance.dropna().values
        return None
    
    def _calculate_inflation_proxy(self, datasets: Dict[str, pd.DataFrame]) -> Optional[np.ndarray]:
        """Calculate inflation proxy from interest rates"""
        for key in ['commercial_banks_weighted_average_rates_', 'commercial_rates']:
            if key in datasets:
                df = datasets[key]
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    rates = df[numeric_cols[0]].dropna()
                    # Year-over-year change as inflation proxy
                    inflation_proxy = rates.pct_change(12)
                    return inflation_proxy.dropna().values
        return None
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'mutual_info', k: int = 20) -> List[str]:
        """Feature selection using various methods"""
        
        X_clean = X.fillna(X.mean()).replace([np.inf, -np.inf], np.nan).fillna(X.mean())
        y_clean = y.fillna(y.mean())
        
        # Align X and y
        min_len = min(len(X_clean), len(y_clean))
        X_clean = X_clean.iloc[:min_len]
        y_clean = y_clean.iloc[:min_len]
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=min(k, X_clean.shape[1]))
        else:
            selector = SelectKBest(score_func=f_regression, k=min(k, X_clean.shape[1]))
        
        try:
            selector.fit(X_clean, y_clean)
            selected_features = X_clean.columns[selector.get_support()].tolist()
            
            # Store feature importance
            feature_scores = dict(zip(X_clean.columns, selector.scores_))
            self.feature_importance[method] = feature_scores
            
            return selected_features
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}")
            return list(X_clean.columns[:k])
    
    def generate_all_features(self, datasets: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict]:
        """Generate comprehensive feature set"""
        
        logger.info("🔧 Generating comprehensive feature set...")
        
        all_features = pd.DataFrame()
        feature_metadata = {}
        
        # 1. Cross-sectional features
        cross_features = self.create_cross_sectional_features(datasets)
        if not cross_features.empty:
            all_features = pd.concat([all_features, cross_features], axis=1)
            feature_metadata['cross_sectional'] = list(cross_features.columns)
        
        # 2. Economic indicators
        econ_indicators = self.create_economic_indicators(datasets)
        if not econ_indicators.empty:
            all_features = pd.concat([all_features, econ_indicators], axis=1)
            feature_metadata['economic_indicators'] = list(econ_indicators.columns)
        
        # 3. Temporal features for key series
        temporal_features = pd.DataFrame()
        key_series = ['monthly_exchange_rate_end_period', 'central_bank_rate_cbr_']
        
        for series_key in key_series:
            if series_key in datasets:
                df = datasets[series_key]
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0 and len(df) > 24:
                    # Create synthetic date column if needed
                    df_temp = df.copy()
                    if 'date' not in df_temp.columns:
                        df_temp['date'] = pd.date_range(start='2010-01-01', periods=len(df_temp), freq='M')
                    
                    temp_features = self.create_temporal_features(df_temp, 'date', numeric_cols[0])
                    
                    # Remove original columns and date
                    feature_cols = [col for col in temp_features.columns 
                                  if col not in [numeric_cols[0], 'date'] and not pd.isna(temp_features[col].iloc[-1])]
                    
                    if feature_cols:
                        temp_subset = temp_features[feature_cols].iloc[:len(all_features) if len(all_features) > 0 else len(temp_features)]
                        temporal_features = pd.concat([temporal_features, temp_subset], axis=1)
        
        if not temporal_features.empty:
            # Align with existing features
            if len(all_features) > 0:
                min_len = min(len(all_features), len(temporal_features))
                all_features = all_features.iloc[:min_len]
                temporal_features = temporal_features.iloc[:min_len]
            
            all_features = pd.concat([all_features, temporal_features], axis=1)
            feature_metadata['temporal'] = list(temporal_features.columns)
        
        # Clean final feature set
        all_features = all_features.replace([np.inf, -np.inf], np.nan)
        all_features = all_features.fillna(all_features.mean())
        
        logger.info(f"✅ Generated {len(all_features.columns)} features across {len(all_features)} observations")
        
        return all_features, feature_metadata

def create_feature_pipeline(datasets: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict]:
    """Main feature engineering pipeline"""
    
    engine = AdvancedFeatureEngine()
    features, metadata = engine.generate_all_features(datasets)
    
    return features, metadata
