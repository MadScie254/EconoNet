"""
EconoNet - Advanced Data Cleaner with Complex Processing
=======================================================

Enhanced data cleaning and preprocessing with advanced error handling,
intelligent type conversion, and complex feature engineering.
"""

import pandas as pd
import numpy as np
import re
import warnings
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class AdvancedDataProcessor:
    """
    Advanced data processing engine with intelligent cleaning,
    feature engineering, and anomaly detection capabilities.
    """
    
    def __init__(self):
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        self.imputers = {
            'knn': KNNImputer(n_neighbors=5),
            'iterative': IterativeImputer(random_state=42)
        }
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.processing_log = []
        
    def intelligent_numeric_conversion(self, series: pd.Series, column_name: str = "") -> pd.Series:
        """
        Intelligently convert series to numeric with advanced error handling
        """
        if series.dtype in ['int64', 'float64']:
            return series
            
        # Create copy to avoid modifying original
        clean_series = series.copy()
        
        # Handle common string patterns
        if clean_series.dtype == 'object':
            # Remove leading/trailing whitespace
            clean_series = clean_series.astype(str).str.strip()
            
            # Remove currency symbols and commas
            clean_series = clean_series.str.replace(r'[KSh$,€£¥₹]', '', regex=True)
            
            # Remove percentage signs
            clean_series = clean_series.str.replace('%', '', regex=True)
            
            # Handle parentheses (negative numbers)
            clean_series = clean_series.str.replace(r'\((.*?)\)', r'-\1', regex=True)
            
            # Remove multiple spaces
            clean_series = clean_series.str.replace(r'\s+', ' ', regex=True)
            
            # Handle dash/hyphen as zero or missing
            clean_series = clean_series.replace(['-', '--', '---', 'N/A', 'NA', 'n/a'], np.nan)
            
            # Handle specific problematic patterns
            clean_series = clean_series.str.replace(r'^\s*$', '', regex=True)  # Empty strings
            clean_series = clean_series.replace(['', ' ', '  '], np.nan)
            
        # Attempt conversion
        try:
            numeric_series = pd.to_numeric(clean_series, errors='coerce')
            
            # Log conversion results
            original_nulls = series.isnull().sum()
            new_nulls = numeric_series.isnull().sum()
            converted_count = len(series) - new_nulls
            
            if new_nulls > original_nulls:
                failed_conversions = new_nulls - original_nulls
                self.processing_log.append({
                    'column': column_name,
                    'action': 'numeric_conversion',
                    'converted': converted_count,
                    'failed': failed_conversions,
                    'success_rate': (converted_count / len(series)) * 100
                })
                
                # Only warn if significant failures
                if failed_conversions > len(series) * 0.1:  # More than 10% failures
                    warnings.warn(f"Column '{column_name}': {failed_conversions} values could not be converted to numeric")
            
            return numeric_series
            
        except Exception as e:
            warnings.warn(f"Error converting column '{column_name}' to numeric: {str(e)}")
            return series
    
    def advanced_outlier_detection(self, df: pd.DataFrame, method: str = 'isolation_forest') -> Dict[str, Any]:
        """
        Advanced outlier detection using multiple methods
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_results = {}
        
        for col in numeric_cols:
            if df[col].isnull().all():
                continue
                
            col_data = df[col].dropna()
            
            if method == 'isolation_forest':
                outliers = self.anomaly_detector.fit_predict(col_data.values.reshape(-1, 1))
                outlier_indices = df.index[df[col].notnull()][outliers == -1]
                
            elif method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_indices = df.index[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                outlier_indices = df.index[df[col].notnull()][z_scores > 3]
            
            outlier_results[col] = {
                'indices': outlier_indices,
                'count': len(outlier_indices),
                'percentage': (len(outlier_indices) / len(df)) * 100,
                'values': df.loc[outlier_indices, col].tolist() if len(outlier_indices) > 0 else []
            }
        
        return outlier_results
    
    def feature_engineering_suite(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive feature engineering with economic domain knowledge
        """
        df_engineered = df.copy()
        
        # Date feature engineering if date columns exist
        date_cols = df.select_dtypes(include=['datetime64']).columns
        for col in date_cols:
            df_engineered[f'{col}_year'] = df[col].dt.year
            df_engineered[f'{col}_month'] = df[col].dt.month
            df_engineered[f'{col}_quarter'] = df[col].dt.quarter
            df_engineered[f'{col}_day_of_year'] = df[col].dt.dayofyear
            df_engineered[f'{col}_weekday'] = df[col].dt.weekday
        
        # Economic indicators feature engineering
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Growth rates
        for col in numeric_cols:
            if 'GDP' in col.upper() or 'EXPORT' in col.upper() or 'IMPORT' in col.upper():
                df_engineered[f'{col}_growth_rate'] = df[col].pct_change() * 100
                df_engineered[f'{col}_moving_avg_3'] = df[col].rolling(window=3).mean()
                df_engineered[f'{col}_moving_avg_12'] = df[col].rolling(window=12).mean()
        
        # Ratios and derived indicators
        debt_cols = [col for col in numeric_cols if 'DEBT' in col.upper()]
        if len(debt_cols) >= 2:
            for i, col1 in enumerate(debt_cols):
                for col2 in debt_cols[i+1:]:
                    df_engineered[f'{col1}_to_{col2}_ratio'] = df[col1] / df[col2]
        
        # Exchange rate features
        fx_cols = [col for col in numeric_cols if any(curr in col.upper() for curr in ['USD', 'EUR', 'GBP', 'EXCHANGE', 'RATE'])]
        for col in fx_cols:
            df_engineered[f'{col}_volatility'] = df[col].rolling(window=30).std()
            df_engineered[f'{col}_rsi'] = self.calculate_rsi(df[col])
        
        # Interaction features
        if len(numeric_cols) >= 2:
            # Create polynomial features for key economic indicators
            key_indicators = [col for col in numeric_cols if any(term in col.upper() 
                            for term in ['GDP', 'INFLATION', 'RATE', 'DEBT', 'EXPORT', 'IMPORT'])]
            
            for i, col1 in enumerate(key_indicators[:5]):  # Limit to avoid explosion
                for col2 in key_indicators[i+1:6]:
                    df_engineered[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        
        return df_engineered
    
    def calculate_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def comprehensive_data_cleaning(self, df: pd.DataFrame, 
                                  config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Comprehensive data cleaning pipeline with detailed reporting
        """
        if config is None:
            config = {
                'handle_outliers': True,
                'impute_missing': True,
                'feature_engineering': True,
                'normalize_data': False,
                'outlier_method': 'isolation_forest'
            }
        
        df_clean = df.copy()
        cleaning_report = {
            'original_shape': df.shape,
            'processing_steps': [],
            'data_quality_metrics': {}
        }
        
        # Step 1: Intelligent numeric conversion
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                original_type = df_clean[col].dtype
                df_clean[col] = self.intelligent_numeric_conversion(df_clean[col], col)
                if df_clean[col].dtype != original_type:
                    cleaning_report['processing_steps'].append(f"Converted {col} to numeric")
        
        # Step 2: Handle missing values
        if config.get('impute_missing', True):
            missing_before = df_clean.isnull().sum().sum()
            
            # Use different strategies based on data type and missingness pattern
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                missing_pct = df_clean[col].isnull().mean()
                
                if 0 < missing_pct < 0.1:  # Less than 10% missing - use interpolation
                    df_clean[col] = df_clean[col].interpolate(method='linear')
                elif 0.1 <= missing_pct < 0.3:  # 10-30% missing - use KNN imputation
                    if len(numeric_cols) > 1:
                        imputer = self.imputers['knn']
                        df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
                elif missing_pct >= 0.3:  # More than 30% missing - use median/mode
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
            missing_after = df_clean.isnull().sum().sum()
            cleaning_report['processing_steps'].append(f"Imputed {missing_before - missing_after} missing values")
        
        # Step 3: Outlier detection and handling
        if config.get('handle_outliers', True):
            outlier_results = self.advanced_outlier_detection(df_clean, config.get('outlier_method', 'isolation_forest'))
            cleaning_report['outlier_analysis'] = outlier_results
            
            # Cap extreme outliers
            for col, results in outlier_results.items():
                if results['percentage'] > 0 and results['percentage'] < 5:  # Cap if less than 5% outliers
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                    cleaning_report['processing_steps'].append(f"Capped outliers in {col}")
        
        # Step 4: Feature engineering
        if config.get('feature_engineering', True):
            original_cols = len(df_clean.columns)
            df_clean = self.feature_engineering_suite(df_clean)
            new_features = len(df_clean.columns) - original_cols
            cleaning_report['processing_steps'].append(f"Created {new_features} engineered features")
        
        # Step 5: Data normalization
        if config.get('normalize_data', False):
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            scaler_type = config.get('scaler_type', 'robust')
            scaler = self.scalers[scaler_type]
            
            df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])
            cleaning_report['processing_steps'].append(f"Applied {scaler_type} scaling")
        
        # Step 6: Data quality metrics
        cleaning_report['data_quality_metrics'] = {
            'final_shape': df_clean.shape,
            'missing_values': df_clean.isnull().sum().sum(),
            'duplicate_rows': df_clean.duplicated().sum(),
            'numeric_columns': len(df_clean.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df_clean.select_dtypes(include=['object']).columns),
            'data_completeness': (1 - df_clean.isnull().sum().sum() / df_clean.size) * 100
        }
        
        return df_clean, cleaning_report

# Enhanced data loading with intelligent processing
def load_and_process_economic_data(file_path: str, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and intelligently process economic data files
    """
    processor = AdvancedDataProcessor()
    
    try:
        # Load data based on file extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, **kwargs)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Apply comprehensive cleaning
        df_processed, report = processor.comprehensive_data_cleaning(df)
        
        print(f"✅ Successfully processed {file_path}")
        print(f"📊 Original shape: {report['original_shape']}")
        print(f"📈 Final shape: {report['data_quality_metrics']['final_shape']}")
        print(f"🎯 Data completeness: {report['data_quality_metrics']['data_completeness']:.1f}%")
        
        return df_processed, report
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")
        return pd.DataFrame(), {}

# Create visualizations for data quality
def create_data_quality_dashboard(df: pd.DataFrame, report: Dict[str, Any]) -> go.Figure:
    """
    Create comprehensive data quality visualization dashboard
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Missing Values by Column',
            'Data Type Distribution', 
            'Outlier Detection Results',
            'Data Completeness Over Time'
        ),
        specs=[[{"type": "bar"}, {"type": "pie"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Missing values by column
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    fig.add_trace(
        go.Bar(x=missing_data.index, y=missing_data.values, 
               name="Missing Values", marker_color='#ff6b6b'),
        row=1, col=1
    )
    
    # Data type distribution
    dtype_counts = df.dtypes.value_counts()
    fig.add_trace(
        go.Pie(labels=dtype_counts.index.astype(str), values=dtype_counts.values,
               name="Data Types"),
        row=1, col=2
    )
    
    # Outlier detection (if available in report)
    if 'outlier_analysis' in report:
        outlier_cols = []
        outlier_counts = []
        for col, results in report['outlier_analysis'].items():
            outlier_cols.append(col)
            outlier_counts.append(results['count'])
        
        fig.add_trace(
            go.Scatter(x=outlier_cols, y=outlier_counts, mode='markers+lines',
                      name="Outliers", marker=dict(size=10, color='#4ecdc4')),
            row=2, col=1
        )
    
    # Data completeness trend (if time-based data)
    date_cols = df.select_dtypes(include=['datetime64']).columns
    if len(date_cols) > 0:
        df_sorted = df.sort_values(date_cols[0])
        completeness = df_sorted.notna().mean(axis=1).rolling(window=30).mean() * 100
        
        fig.add_trace(
            go.Scatter(x=df_sorted[date_cols[0]], y=completeness,
                      mode='lines', name="Completeness %",
                      line=dict(color='#45b7d1')),
            row=2, col=2
        )
    
    fig.update_layout(
        title="Advanced Data Quality Analysis Dashboard",
        showlegend=True,
        height=600,
        template="plotly_white"
    )
    
    return fig

if __name__ == "__main__":
    # Test the advanced data processor
    print("🧪 Testing Advanced Data Processor...")
    
    # Create sample problematic data
    sample_data = {
        'GDP_Growth': ['5.2%', '4.8%', ' 6.1% ', '-1.2%', 'N/A'],
        'Exchange_Rate': ['132.45', '131.80', ' 133.20 ', '(134.10)', '132.90'],
        'Debt_Domestic': [' 2,500,000 ', '2,400,000', '2,600,000', '  ', '2,550,000'],
        'Debt_External': ['1,800,000', ' 1,750,000 ', '1,900,000', 'NA', '1,850,000'],
        'Total_Debt': [' 4,300,000 ', '4,150,000', '4,500,000', '--', '4,400,000']
    }
    
    df = pd.DataFrame(sample_data)
    processor = AdvancedDataProcessor()
    
    print("📊 Original Data:")
    print(df)
    print(f"Data types: {df.dtypes.to_dict()}")
    
    df_processed, report = processor.comprehensive_data_cleaning(df)
    
    print("\n📈 Processed Data:")
    print(df_processed)
    print(f"Data types: {df_processed.dtypes.to_dict()}")
    print(f"\n📋 Processing Report: {len(report['processing_steps'])} steps completed")
    for step in report['processing_steps']:
        print(f"  ✓ {step}")
