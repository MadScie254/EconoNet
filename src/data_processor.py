"""
Enhanced Data Processing Module for EconoNet
==========================================

Comprehensive data ingestion and preprocessing for all Kenya economic data sources.
Handles data cleaning, type conversion, and preparation for modeling.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class KenyaEconomicDataProcessor:
    """
    Advanced data processor for Kenya economic datasets
    """
    
    def __init__(self, data_path: str = "data/raw"):
        self.data_path = Path(data_path)
        self.processed_data = {}
        self.metadata = {}
        
    def clean_numeric_string(self, value: str) -> float:
        """Clean numeric strings with commas and convert to float"""
        if pd.isna(value) or value == '' or value == '-':
            return np.nan
        
        # Convert to string if not already
        value = str(value).strip()
        
        # Remove common non-numeric characters
        value = re.sub(r'[^\d.,\-+]', '', value)
        
        # Handle multiple comma-separated numbers (take the first one)
        if ',' in value and value.count(',') > 1:
            # Split by comma and take the first valid number
            parts = value.split(',')
            for part in parts:
                if part and part.replace('.', '').replace('-', '').isdigit():
                    value = part
                    break
        
        # Remove commas used as thousands separators
        value = value.replace(',', '')
        
        try:
            return float(value)
        except (ValueError, TypeError):
            return np.nan
    
    def process_csv_file(self, filename: str, date_columns: List[str] = None, 
                        numeric_columns: List[str] = None) -> pd.DataFrame:
        """Process individual CSV file with proper type conversion"""
        
        filepath = self.data_path / filename
        
        if not filepath.exists():
            print(f"Warning: File {filename} not found")
            return pd.DataFrame()
        
        try:
            # Read CSV with multiple encoding attempts
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(filepath, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                df = pd.read_csv(filepath, encoding='utf-8', errors='ignore')
            
            # Clean column names
            df.columns = df.columns.str.strip().str.replace('\n', ' ').str.replace('\r', '')
            
            # Auto-detect date columns if not specified
            if date_columns is None:
                date_columns = []
                for col in df.columns:
                    if any(word in col.lower() for word in ['date', 'month', 'year', 'period']):
                        date_columns.append(col)
            
            # Auto-detect numeric columns if not specified
            if numeric_columns is None:
                numeric_columns = []
                for col in df.columns:
                    if col not in date_columns:
                        # Check if column contains mostly numeric-like values
                        sample = df[col].dropna().head(10).astype(str)
                        numeric_count = sum(1 for val in sample if re.search(r'[\d,.]', val))
                        if numeric_count > len(sample) * 0.7:  # 70% numeric-like
                            numeric_columns.append(col)
            
            # Process date columns
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Process numeric columns
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].apply(self.clean_numeric_string)
            
            # Store metadata
            self.metadata[filename] = {
                'shape': df.shape,
                'date_columns': date_columns,
                'numeric_columns': numeric_columns,
                'processed_date': pd.Timestamp.now()
            }
            
            return df
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            return pd.DataFrame()
    
    def load_gdp_data(self) -> pd.DataFrame:
        """Load and process GDP data"""
        return self.process_csv_file("Annual GDP.csv")
    
    def load_inflation_data(self) -> pd.DataFrame:
        """Load and process inflation data"""
        # Try multiple potential inflation files
        files = ["Central Bank Rate (CBR)  .csv", "Central Bank Rates ().csv"]
        for file in files:
            df = self.process_csv_file(file)
            if not df.empty:
                return df
        return pd.DataFrame()
    
    def load_exchange_rate_data(self) -> pd.DataFrame:
        """Load and process exchange rate data"""
        monthly_end = self.process_csv_file("Monthly exchange rate (end period).csv")
        monthly_avg = self.process_csv_file("Monthly Exchange rate (period average).csv")
        
        # Merge if both exist
        if not monthly_end.empty and not monthly_avg.empty:
            # Merge on common columns (usually date)
            common_cols = set(monthly_end.columns) & set(monthly_avg.columns)
            if common_cols:
                return pd.merge(monthly_end, monthly_avg, on=list(common_cols), how='outer', suffixes=('_end', '_avg'))
        
        return monthly_end if not monthly_end.empty else monthly_avg
    
    def load_trade_data(self) -> pd.DataFrame:
        """Load and process trade data"""
        trade_files = [
            "Foreign Trade Summary (Ksh Million).csv",
            "Value of Exports to Selected African Countries (Ksh Million).csv",
            "Value of Exports to Selected Rest of World Countries (Ksh Million).csv",
            "Value of Direct Imports from Selected African Countries (Ksh. Million).xlsx",
            "Value of Direct Imports from Selected Rest of World Countries  (Kshs. Millions).csv"
        ]
        
        trade_data = {}
        for file in trade_files:
            if file.endswith('.xlsx'):
                # Handle Excel files
                filepath = self.data_path / file
                if filepath.exists():
                    try:
                        df = pd.read_excel(filepath)
                        trade_data[file] = df
                    except Exception as e:
                        print(f"Error reading Excel file {file}: {e}")
            else:
                df = self.process_csv_file(file)
                if not df.empty:
                    trade_data[file] = df
        
        return trade_data
    
    def load_financial_data(self) -> Dict[str, pd.DataFrame]:
        """Load and process financial sector data"""
        financial_files = [
            "Issues of Treasury Bills.csv",
            "Issues of Treasury Bonds.csv",
            "Public Debt.csv",
            "Domestic Debt by Instrument.csv",
            "Government Securities Auction and Maturities Schedule.csv",
            "Commercial Banks Weighted Average Rates ().csv"
        ]
        
        financial_data = {}
        for file in financial_files:
            df = self.process_csv_file(file)
            if not df.empty:
                financial_data[file.replace('.csv', '')] = df
        
        return financial_data
    
    def load_payments_data(self) -> Dict[str, pd.DataFrame]:
        """Load and process payments and banking data"""
        payments_files = [
            "Mobile Payments.csv",
            "KEPSSRTGS.csv",
            "Cheques  EFTs.csv",
            "Number of Transactions.csv",
            "Value of Transactions (Kshs. Millions).csv",
            "Number of ATMs, ATM Cards,  POS Machines.csv"
        ]
        
        payments_data = {}
        for file in payments_files:
            df = self.process_csv_file(file)
            if not df.empty:
                payments_data[file.replace('.csv', '')] = df
        
        return payments_data
    
    def load_monetary_data(self) -> Dict[str, pd.DataFrame]:
        """Load and process monetary policy data"""
        monetary_files = [
            "Interbank Rates  Volumes.csv",
            "Interbank Rates () .csv",
            "Repo and Reverse Repo .csv",
            "Horizontal Repo Market.csv",
            "Discount Window.csv",
            "Daily KES Interbank Activity Report.csv"
        ]
        
        monetary_data = {}
        for file in monetary_files:
            df = self.process_csv_file(file)
            if not df.empty:
                monetary_data[file.replace('.csv', '')] = df
        
        return monetary_data
    
    def load_external_data(self) -> Dict[str, pd.DataFrame]:
        """Load and process external sector data"""
        external_files = [
            "Diaspora Remittances.csv",
            "TRADE WEIGHTED AVERAGE INDICATIVE RATES.csv",
            "Principal Exports Volume, Value and Unit Prices (Ksh Million).csv",
            "Value of Selected Domestic Exports (Ksh Million).csv"
        ]
        
        external_data = {}
        for file in external_files:
            df = self.process_csv_file(file)
            if not df.empty:
                external_data[file.replace('.csv', '')] = df
        
        return external_data
    
    def load_fiscal_data(self) -> pd.DataFrame:
        """Load and process fiscal data"""
        return self.process_csv_file("Revenue and Expenditure.csv")
    
    def create_comprehensive_dataset(self) -> pd.DataFrame:
        """Create a comprehensive dataset combining all sources"""
        
        print("Loading all economic datasets...")
        
        # Load all data sources
        gdp_data = self.load_gdp_data()
        inflation_data = self.load_inflation_data()
        fx_data = self.load_exchange_rate_data()
        trade_data = self.load_trade_data()
        financial_data = self.load_financial_data()
        payments_data = self.load_payments_data()
        monetary_data = self.load_monetary_data()
        external_data = self.load_external_data()
        fiscal_data = self.load_fiscal_data()
        
        # Create master dataset
        master_data = pd.DataFrame()
        
        # Add key economic indicators
        datasets = [
            ('GDP', gdp_data),
            ('Inflation', inflation_data),
            ('Exchange_Rate', fx_data),
            ('Fiscal', fiscal_data)
        ]
        
        for name, data in datasets:
            if not data.empty:
                # Add prefix to avoid column conflicts
                data_prefixed = data.add_prefix(f'{name}_')
                
                # Try to identify date column for merging
                date_col = None
                for col in data_prefixed.columns:
                    if any(word in col.lower() for word in ['date', 'month', 'year', 'period']):
                        date_col = col
                        break
                
                if master_data.empty:
                    master_data = data_prefixed
                elif date_col:
                    # Merge on date column
                    master_data = pd.merge(master_data, data_prefixed, 
                                         left_index=True, right_index=True, how='outer')
        
        # Store comprehensive data
        self.processed_data['comprehensive'] = master_data
        
        print(f"Comprehensive dataset created with shape: {master_data.shape}")
        return master_data
    
    def get_sample_data_for_modeling(self, n_periods: int = 60) -> pd.DataFrame:
        """Generate sample data for modeling when real data is unavailable"""
        
        dates = pd.date_range(start='2019-01-01', periods=n_periods, freq='M')
        
        # Generate realistic economic indicators for Kenya
        data = pd.DataFrame({
            'Date': dates,
            'GDP_Growth': np.random.normal(5.5, 1.8, n_periods),
            'Inflation_Rate': np.random.normal(6.2, 2.1, n_periods),
            'Exchange_Rate_KES_USD': np.random.normal(135, 12, n_periods),
            'CBR_Rate': np.random.normal(7.5, 1.2, n_periods),
            'Treasury_Bills_91_Day': np.random.normal(7.8, 1.5, n_periods),
            'Current_Account_Balance': np.random.normal(-4.2, 1.8, n_periods),
            'Foreign_Reserves_Months': np.random.normal(5.2, 0.8, n_periods),
            'Public_Debt_GDP_Ratio': np.random.normal(62, 5, n_periods),
            'Credit_Growth': np.random.normal(8.5, 3.2, n_periods),
            'Mobile_Money_Transactions': np.random.normal(15000, 2000, n_periods),
            'Export_Value_Million_KES': np.random.normal(55000, 8000, n_periods),
            'Import_Value_Million_KES': np.random.normal(85000, 12000, n_periods)
        })
        
        # Add some realistic correlations and trends
        data['Trade_Balance'] = data['Export_Value_Million_KES'] - data['Import_Value_Million_KES']
        data['Real_Interest_Rate'] = data['CBR_Rate'] - data['Inflation_Rate']
        
        # Ensure realistic bounds
        data['GDP_Growth'] = np.clip(data['GDP_Growth'], -5, 15)
        data['Inflation_Rate'] = np.clip(data['Inflation_Rate'], 0, 20)
        data['Exchange_Rate_KES_USD'] = np.clip(data['Exchange_Rate_KES_USD'], 100, 180)
        data['CBR_Rate'] = np.clip(data['CBR_Rate'], 4, 12)
        data['Foreign_Reserves_Months'] = np.clip(data['Foreign_Reserves_Months'], 3, 8)
        data['Public_Debt_GDP_Ratio'] = np.clip(data['Public_Debt_GDP_Ratio'], 40, 80)
        
        return data.set_index('Date')
    
    def get_metadata_summary(self) -> Dict[str, Any]:
        """Get summary of all processed datasets"""
        
        summary = {
            'total_files_processed': len(self.metadata),
            'processing_date': pd.Timestamp.now(),
            'file_details': self.metadata
        }
        
        if self.processed_data:
            summary['dataset_shapes'] = {
                name: data.shape for name, data in self.processed_data.items()
            }
        
        return summary


# Global instance for easy access
data_processor = KenyaEconomicDataProcessor()
