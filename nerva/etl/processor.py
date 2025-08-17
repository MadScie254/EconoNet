"""
NERVA ETL Engine
GODMODE_X: Brutal efficiency data pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from ..config.settings import config
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config.settings import config

# Setup logging
logging.basicConfig(level=getattr(logging, config.system.log_level))
logger = logging.getLogger(__name__)

@dataclass
class DataQualityReport:
    """Data quality assessment results"""
    file_name: str
    total_rows: int
    total_columns: int
    missing_percentage: float
    duplicate_rows: int
    date_columns: List[str]
    numeric_columns: List[str]
    outliers_count: int
    quality_score: float  # 0-1
    recommendations: List[str]

class CBKDataProcessor:
    """Central Bank of Kenya data processor with automated schema detection"""
    
    def __init__(self):
        self.raw_path = config.data.raw_data_path
        self.processed_path = config.data.processed_data_path
        self.parquet_path = config.data.parquet_path
        
        # Data catalog
        self.catalog = {}
        self.quality_reports = {}
    
    def scan_all_files(self) -> Dict[str, pd.DataFrame]:
        """Scan and load all CBK data files"""
        logger.info("🔍 Scanning CBK data archive...")
        
        datasets = {}
        file_paths = list(self.raw_path.glob("*.csv")) + list(self.raw_path.glob("*.xlsx"))
        
        for file_path in file_paths:
            try:
                dataset_name = file_path.stem.lower().replace(" ", "_").replace("(", "").replace(")", "")
                
                logger.info(f"📊 Processing: {file_path.name}")
                df = self._load_file(file_path)
                
                if df is not None and not df.empty:
                    # Clean and standardize
                    df_clean = self._clean_dataframe(df, dataset_name)
                    datasets[dataset_name] = df_clean
                    
                    # Generate quality report
                    quality = self._assess_data_quality(df_clean, file_path.name)
                    self.quality_reports[dataset_name] = quality
                    
                    # Save to parquet
                    self._save_parquet(df_clean, dataset_name)
                    
                    logger.info(f"✅ {dataset_name}: {len(df_clean)} rows, Quality Score: {quality.quality_score:.2f}")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {file_path.name}: {str(e)}")
                continue
        
        logger.info(f"🎯 Successfully processed {len(datasets)} datasets")
        return datasets
    
    def _load_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Smart file loader with format detection"""
        try:
            if file_path.suffix.lower() == '.csv':
                # Try different encodings and separators
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        if len(df.columns) > 1:
                            return df
                    except:
                        continue
                
                # Try semicolon separator
                try:
                    df = pd.read_csv(file_path, sep=';', encoding='utf-8')
                    return df
                except:
                    pass
                    
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
                return df
                
        except Exception as e:
            logger.warning(f"Failed to load {file_path.name}: {str(e)}")
            return None
    
    def _clean_dataframe(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Automated data cleaning and standardization"""
        df_clean = df.copy()
        
        # 1. Column name standardization
        df_clean.columns = [
            col.strip().lower()
            .replace(' ', '_')
            .replace('(', '')
            .replace(')', '')
            .replace('-', '_')
            .replace('.', '_')
            for col in df_clean.columns
        ]
        
        # 2. Date column detection and parsing
        date_columns = self._detect_date_columns(df_clean)
        for col in date_columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce', infer_datetime_format=True)
        
        # 3. Numeric column detection and cleaning
        numeric_columns = self._detect_numeric_columns(df_clean)
        for col in numeric_columns:
            # Remove common currency symbols and commas
            if df_clean[col].dtype == 'object':
                df_clean[col] = (df_clean[col].astype(str)
                               .str.replace(',', '')
                               .str.replace('KSh', '')
                               .str.replace('$', '')
                               .str.replace('%', ''))
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # 4. Remove completely empty rows/columns
        df_clean = df_clean.dropna(how='all')
        df_clean = df_clean.loc[:, df_clean.notna().any()]
        
        # 5. Add metadata columns
        df_clean['_dataset_name'] = dataset_name
        df_clean['_processed_timestamp'] = datetime.now()
        
        return df_clean
    
    def _detect_date_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect date columns using heuristics"""
        date_keywords = ['date', 'time', 'year', 'month', 'period', 'quarter']
        date_columns = []
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check for date keywords in column name
            if any(keyword in col_lower for keyword in date_keywords):
                date_columns.append(col)
                continue
            
            # Check data patterns for first few non-null values
            sample_values = df[col].dropna().head(10).astype(str)
            if len(sample_values) > 0:
                # Simple date pattern detection
                date_patterns = [
                    r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD
                    r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
                    r'\d{1,2}-\d{1,2}-\d{4}',  # MM-DD-YYYY
                    r'\d{4}',                   # YYYY only
                ]
                
                for pattern in date_patterns:
                    if sample_values.str.contains(pattern, regex=True).any():
                        date_columns.append(col)
                        break
        
        return date_columns
    
    def _detect_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect numeric columns that might be stored as strings"""
        numeric_columns = []
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                numeric_columns.append(col)
                continue
            
            # Check if string column contains mostly numeric data
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(100).astype(str)
                if len(sample_values) > 0:
                    # Remove common non-numeric characters and test
                    cleaned_sample = (sample_values.str.replace(',', '')
                                    .str.replace('KSh', '')
                                    .str.replace('$', '')
                                    .str.replace('%', '')
                                    .str.strip())
                    
                    # Count how many can be converted to numeric
                    numeric_count = pd.to_numeric(cleaned_sample, errors='coerce').notna().sum()
                    if numeric_count / len(cleaned_sample) > 0.7:  # 70% threshold
                        numeric_columns.append(col)
        
        return numeric_columns
    
    def _assess_data_quality(self, df: pd.DataFrame, file_name: str) -> DataQualityReport:
        """Comprehensive data quality assessment"""
        
        # Basic stats
        total_rows = len(df)
        total_columns = len(df.columns)
        missing_percentage = (df.isnull().sum().sum() / (total_rows * total_columns)) * 100
        duplicate_rows = df.duplicated().sum()
        
        # Column type detection
        date_columns = self._detect_date_columns(df)
        numeric_columns = self._detect_numeric_columns(df)
        
        # Outlier detection for numeric columns
        outliers_count = 0
        for col in numeric_columns:
            if df[col].dtype in ['int64', 'float64']:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers_count += (z_scores > config.data.max_outlier_zscore).sum()
        
        # Quality score calculation (0-1)
        completeness_score = 1 - (missing_percentage / 100)
        duplicate_penalty = min(duplicate_rows / total_rows, 0.3)  # Cap at 30% penalty
        outlier_penalty = min(outliers_count / (total_rows * len(numeric_columns)), 0.2) if numeric_columns else 0
        
        quality_score = max(0, completeness_score - duplicate_penalty - outlier_penalty)
        
        # Recommendations
        recommendations = []
        if missing_percentage > 30:
            recommendations.append("High missing data rate - consider imputation strategies")
        if duplicate_rows > total_rows * 0.05:
            recommendations.append("Significant duplicate rows detected")
        if outliers_count > total_rows * 0.1:
            recommendations.append("High outlier count - review data collection")
        if not date_columns:
            recommendations.append("No date columns detected - may affect time series analysis")
        
        return DataQualityReport(
            file_name=file_name,
            total_rows=total_rows,
            total_columns=total_columns,
            missing_percentage=missing_percentage,
            duplicate_rows=duplicate_rows,
            date_columns=date_columns,
            numeric_columns=numeric_columns,
            outliers_count=outliers_count,
            quality_score=quality_score,
            recommendations=recommendations
        )
    
    def _save_parquet(self, df: pd.DataFrame, dataset_name: str):
        """Save cleaned dataset to parquet format"""
        parquet_file = self.parquet_path / f"{dataset_name}.parquet"
        df.to_parquet(parquet_file, compression='snappy', index=False)
        
        # Update catalog
        self.catalog[dataset_name] = {
            'file_path': str(parquet_file),
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'last_updated': datetime.now().isoformat()
        }
    
    def generate_data_catalog(self) -> pd.DataFrame:
        """Generate comprehensive data catalog"""
        catalog_data = []
        
        for dataset_name, quality_report in self.quality_reports.items():
            catalog_data.append({
                'dataset_name': dataset_name,
                'file_name': quality_report.file_name,
                'rows': quality_report.total_rows,
                'columns': quality_report.total_columns,
                'missing_percentage': quality_report.missing_percentage,
                'quality_score': quality_report.quality_score,
                'date_columns': len(quality_report.date_columns),
                'numeric_columns': len(quality_report.numeric_columns),
                'outliers': quality_report.outliers_count,
                'recommendations': '; '.join(quality_report.recommendations[:3])  # Top 3
            })
        
        catalog_df = pd.DataFrame(catalog_data)
        
        # Save catalog
        catalog_file = self.processed_path / "data_catalog.csv"
        catalog_df.to_csv(catalog_file, index=False)
        
        return catalog_df
    
    def get_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """Load a specific dataset from parquet"""
        parquet_file = self.parquet_path / f"{dataset_name}.parquet"
        
        if parquet_file.exists():
            return pd.read_parquet(parquet_file)
        else:
            logger.warning(f"Dataset '{dataset_name}' not found")
            return None

# Quick access functions
def load_cbk_data() -> Dict[str, pd.DataFrame]:
    """Load all CBK datasets"""
    processor = CBKDataProcessor()
    return processor.scan_all_files()

def get_data_catalog() -> pd.DataFrame:
    """Get the data catalog"""
    processor = CBKDataProcessor()
    processor.scan_all_files()  # Ensure data is processed
    return processor.generate_data_catalog()
