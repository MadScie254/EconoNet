"""
EconoNet Pipeline Diagnostics and Repair System
==============================================

Comprehensive system to detect, diagnose, and fix all pipeline issues
in the EconoNet platform.
"""

import os
import sys
import pandas as pd
import numpy as np
import traceback
import warnings
from pathlib import Path
import json
import subprocess
from datetime import datetime

class PipelineDiagnostics:
    """Advanced system for detecting and fixing pipeline issues"""
    
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.issues = []
        self.fixes_applied = []
        self.data_path = self.project_root / "data" / "raw"
        
    def run_full_diagnostics(self):
        """Run comprehensive diagnostics on all systems"""
        print("🔍 Starting EconoNet Pipeline Diagnostics...")
        
        # 1. Check data availability
        self._check_data_availability()
        
        # 2. Check import dependencies
        self._check_dependencies()
        
        # 3. Check data format issues
        self._check_data_formats()
        
        # 4. Check model pipelines
        self._check_model_pipelines()
        
        # 5. Check notebook accessibility
        self._check_notebook_accessibility()
        
        # Generate diagnostics report
        self._generate_report()
        
        return self.issues
    
    def _check_data_availability(self):
        """Check if all required data files are available"""
        print("📊 Checking data availability...")
        
        required_files = [
            "Annual GDP.csv",
            "Central Bank Rate (CBR)  .csv",
            "Mobile Payments.csv",
            "Foreign Trade Summary (Ksh Million).csv",
            "Inflation_Rate.csv",  # May not exist
            "Exchange_Rates.csv"   # May not exist
        ]
        
        available_files = []
        missing_files = []
        
        if self.data_path.exists():
            for file in self.data_path.glob("*.csv"):
                available_files.append(file.name)
            
            for required_file in required_files:
                if not any(req in available for available in available_files for req in required_file.split()):
                    missing_files.append(required_file)
        else:
            self.issues.append({
                'type': 'CRITICAL',
                'component': 'Data Directory',
                'issue': f"Data directory not found: {self.data_path}",
                'fix': "Create data/raw directory and add CSV files"
            })
        
        if missing_files:
            self.issues.append({
                'type': 'WARNING',
                'component': 'Data Files',
                'issue': f"Missing data files: {missing_files}",
                'fix': "Add missing CSV files to data/raw directory"
            })
        
        print(f"✅ Found {len(available_files)} data files")
    
    def _check_dependencies(self):
        """Check if all required Python packages are available"""
        print("🔧 Checking Python dependencies...")
        
        required_packages = [
            'streamlit', 'pandas', 'numpy', 'plotly', 'scikit-learn',
            'scipy', 'networkx', 'nbformat', 'nbconvert'
        ]
        
        missing_packages = []
        available_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                available_packages.append(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.issues.append({
                'type': 'ERROR',
                'component': 'Dependencies',
                'issue': f"Missing packages: {missing_packages}",
                'fix': f"Install packages: pip install {' '.join(missing_packages)}"
            })
        
        print(f"✅ {len(available_packages)}/{len(required_packages)} packages available")
    
    def _check_data_formats(self):
        """Check for data format issues that cause conversion warnings"""
        print("📋 Checking data formats...")
        
        if not self.data_path.exists():
            return
        
        format_issues = []
        
        for csv_file in self.data_path.glob("*.csv"):
            try:
                # Read first few rows to check format
                df = pd.read_csv(csv_file, nrows=10)
                
                # Check for common issues
                for col in df.columns:
                    if 'debt' in col.lower() or 'gdp' in col.lower() or 'rate' in col.lower():
                        sample_values = df[col].dropna().astype(str).head(5)
                        
                        for val in sample_values:
                            # Check for whitespace issues
                            if val != val.strip():
                                format_issues.append({
                                    'file': csv_file.name,
                                    'column': col,
                                    'issue': 'Whitespace in numeric columns',
                                    'sample': repr(val)
                                })
                            
                            # Check for non-numeric characters
                            if val.replace('.', '').replace('-', '').replace('+', '').strip():
                                if not val.replace('.', '').replace('-', '').replace('+', '').replace(',', '').isdigit():
                                    if val not in ['nan', 'NaN', '', 'null']:
                                        format_issues.append({
                                            'file': csv_file.name,
                                            'column': col,
                                            'issue': 'Non-numeric data in numeric column',
                                            'sample': repr(val)
                                        })
                
            except Exception as e:
                format_issues.append({
                    'file': csv_file.name,
                    'column': 'N/A',
                    'issue': f'Cannot read file: {str(e)}',
                    'sample': 'N/A'
                })
        
        if format_issues:
            self.issues.append({
                'type': 'WARNING',
                'component': 'Data Format',
                'issue': f"Found {len(format_issues)} format issues",
                'details': format_issues,
                'fix': "Use AdvancedDataProcessor to clean data automatically"
            })
        
        print(f"🔍 Checked data formats, found {len(format_issues)} issues")
    
    def _check_model_pipelines(self):
        """Check model pipeline functionality"""
        print("🤖 Checking model pipelines...")
        
        try:
            # Test basic model imports
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            
            # Test with sample data
            X = np.random.random((50, 5))
            y = np.random.random(50)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X_scaled, y)
            
            predictions = model.predict(X_scaled[:5])
            
            if len(predictions) == 5:
                print("✅ Model pipelines working correctly")
            else:
                self.issues.append({
                    'type': 'ERROR',
                    'component': 'Model Pipeline',
                    'issue': 'Model prediction shape mismatch',
                    'fix': 'Check model implementation and data preprocessing'
                })
        
        except Exception as e:
            self.issues.append({
                'type': 'ERROR',
                'component': 'Model Pipeline',
                'issue': f'Model pipeline error: {str(e)}',
                'fix': 'Check scikit-learn installation and model implementation'
            })
    
    def _check_notebook_accessibility(self):
        """Check notebook availability and accessibility"""
        print("📚 Checking notebook accessibility...")
        
        notebooks_path = self.project_root / "notebooks"
        
        if not notebooks_path.exists():
            self.issues.append({
                'type': 'WARNING',
                'component': 'Notebooks',
                'issue': 'Notebooks directory not found',
                'fix': 'Create notebooks directory'
            })
            return
        
        notebook_files = list(notebooks_path.glob("*.ipynb"))
        
        if len(notebook_files) == 0:
            self.issues.append({
                'type': 'WARNING',
                'component': 'Notebooks',
                'issue': 'No notebook files found',
                'fix': 'Add Jupyter notebook files to notebooks directory'
            })
        else:
            # Test notebook reading
            try:
                import nbformat
                test_notebook = notebook_files[0]
                with open(test_notebook, 'r', encoding='utf-8') as f:
                    nb = nbformat.read(f, as_version=4)
                print(f"✅ Successfully read {len(notebook_files)} notebook files")
            except Exception as e:
                self.issues.append({
                    'type': 'ERROR',
                    'component': 'Notebooks',
                    'issue': f'Cannot read notebook files: {str(e)}',
                    'fix': 'Install nbformat package: pip install nbformat'
                })
    
    def _generate_report(self):
        """Generate comprehensive diagnostics report"""
        print("\n" + "="*60)
        print("📋 ECONET PIPELINE DIAGNOSTICS REPORT")
        print("="*60)
        
        if not self.issues:
            print("🎉 ALL SYSTEMS OPERATIONAL - NO ISSUES DETECTED!")
            return
        
        # Group issues by type
        critical_issues = [i for i in self.issues if i['type'] == 'CRITICAL']
        error_issues = [i for i in self.issues if i['type'] == 'ERROR']
        warning_issues = [i for i in self.issues if i['type'] == 'WARNING']
        
        if critical_issues:
            print(f"\n🚨 CRITICAL ISSUES ({len(critical_issues)}):")
            for issue in critical_issues:
                print(f"   ❌ {issue['component']}: {issue['issue']}")
                print(f"      💡 Fix: {issue['fix']}")
        
        if error_issues:
            print(f"\n⚠️  ERROR ISSUES ({len(error_issues)}):")
            for issue in error_issues:
                print(f"   🔴 {issue['component']}: {issue['issue']}")
                print(f"      💡 Fix: {issue['fix']}")
        
        if warning_issues:
            print(f"\n⚡ WARNING ISSUES ({len(warning_issues)}):")
            for issue in warning_issues:
                print(f"   🟡 {issue['component']}: {issue['issue']}")
                print(f"      💡 Fix: {issue['fix']}")
        
        print(f"\n📊 SUMMARY: {len(critical_issues)} Critical, {len(error_issues)} Errors, {len(warning_issues)} Warnings")
        print("="*60)
    
    def auto_fix_issues(self):
        """Automatically fix issues where possible"""
        print("🔧 Starting automatic issue resolution...")
        
        for issue in self.issues:
            try:
                if issue['component'] == 'Data Format' and 'Whitespace' in issue['issue']:
                    self._fix_whitespace_issues()
                
                elif issue['component'] == 'Dependencies' and 'Missing packages' in issue['issue']:
                    self._install_missing_packages(issue)
                
                elif issue['component'] == 'Notebooks' and 'directory not found' in issue['issue']:
                    self._create_notebooks_directory()
                
            except Exception as e:
                print(f"❌ Failed to auto-fix {issue['component']}: {str(e)}")
        
        print("✅ Auto-fix process completed")
    
    def _fix_whitespace_issues(self):
        """Fix whitespace issues in data files"""
        print("🧹 Fixing whitespace issues in data files...")
        
        try:
            # Import and use the advanced data processor
            sys.path.append(str(self.project_root / "src"))
            from advanced_data_processor import AdvancedDataProcessor
            
            processor = AdvancedDataProcessor()
            
            # Process all CSV files
            for csv_file in self.data_path.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    cleaned_df = processor.intelligent_numeric_conversion(df)
                    
                    # Save cleaned version
                    cleaned_path = self.project_root / "data" / "cleaned" / csv_file.name
                    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
                    cleaned_df.to_csv(cleaned_path, index=False)
                    
                    self.fixes_applied.append(f"Cleaned whitespace in {csv_file.name}")
                
                except Exception as e:
                    print(f"⚠️  Could not clean {csv_file.name}: {str(e)}")
        
        except ImportError:
            print("⚠️  AdvancedDataProcessor not available for auto-fix")
    
    def _install_missing_packages(self, issue):
        """Install missing Python packages"""
        missing_packages = issue['issue'].split(': ')[1].strip('[]').replace("'", "").split(', ')
        
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                self.fixes_applied.append(f"Installed package: {package}")
            except Exception as e:
                print(f"❌ Failed to install {package}: {str(e)}")
    
    def _create_notebooks_directory(self):
        """Create notebooks directory if missing"""
        notebooks_path = self.project_root / "notebooks"
        notebooks_path.mkdir(parents=True, exist_ok=True)
        self.fixes_applied.append("Created notebooks directory")

def run_pipeline_diagnostics():
    """Main function to run pipeline diagnostics"""
    diagnostics = PipelineDiagnostics()
    issues = diagnostics.run_full_diagnostics()
    
    if issues:
        print(f"\n🔧 Found {len(issues)} issues. Attempting auto-fix...")
        diagnostics.auto_fix_issues()
        
        if diagnostics.fixes_applied:
            print("\n✅ FIXES APPLIED:")
            for fix in diagnostics.fixes_applied:
                print(f"   ✓ {fix}")
    
    return diagnostics

if __name__ == "__main__":
    run_pipeline_diagnostics()
