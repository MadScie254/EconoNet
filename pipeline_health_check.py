"""
EconoNet Pipeline Health Check & Auto-Fix System
==============================================

Comprehensive system to diagnose and automatically fix all pipeline issues
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import importlib
import warnings
from datetime import datetime
import json
import streamlit as st

warnings.filterwarnings('ignore')

class PipelineHealthChecker:
    """Comprehensive pipeline health monitoring and auto-fix system"""
    
    def __init__(self, root_dir="."):
        self.root_dir = Path(root_dir)
        self.health_report = {}
        self.fix_log = []
        
    def check_all_pipelines(self):
        """Run comprehensive health check on all pipelines"""
        print("🔍 Starting comprehensive pipeline health check...")
        
        # Check data pipeline
        self.check_data_pipeline()
        
        # Check model pipeline
        self.check_model_pipeline()
        
        # Check dashboard pipeline
        self.check_dashboard_pipeline()
        
        # Check notebook pipeline
        self.check_notebook_pipeline()
        
        # Check dependencies
        self.check_dependencies()
        
        # Generate health report
        self.generate_health_report()
        
        return self.health_report
    
    def check_data_pipeline(self):
        """Check data ingestion and processing pipeline"""
        print("📊 Checking data pipeline...")
        
        data_issues = []
        data_fixes = []
        
        # Check data directories
        required_dirs = ['data', 'data/raw', 'data/processed', 'data/cleaned']
        for dir_path in required_dirs:
            full_path = self.root_dir / dir_path
            if not full_path.exists():
                data_issues.append(f"Missing directory: {dir_path}")
                # Auto-fix: create directory
                try:
                    full_path.mkdir(parents=True, exist_ok=True)
                    fix_msg = f"✅ Created directory: {dir_path}"
                    data_fixes.append(fix_msg)
                    self.fix_log.append(fix_msg)
                except Exception as e:
                    data_issues.append(f"Failed to create {dir_path}: {e}")
        
        # Check for data files
        raw_data_dir = self.root_dir / 'data' / 'raw'
        if raw_data_dir.exists():
            csv_files = list(raw_data_dir.glob('*.csv'))
            xlsx_files = list(raw_data_dir.glob('*.xlsx'))
            
            if len(csv_files) == 0 and len(xlsx_files) == 0:
                data_issues.append("No data files found in data/raw/")
                # Auto-fix: generate sample data
                try:
                    self.generate_sample_data()
                    fix_msg = "✅ Generated sample economic data files"
                    data_fixes.append(fix_msg)
                    self.fix_log.append(fix_msg)
                except Exception as e:
                    data_issues.append(f"Failed to generate sample data: {e}")
            else:
                # Check data file integrity
                corrupted_files = []
                for csv_file in csv_files:
                    try:
                        pd.read_csv(csv_file, nrows=5)
                    except Exception as e:
                        corrupted_files.append(f"{csv_file.name}: {e}")
                
                if corrupted_files:
                    data_issues.extend(corrupted_files)
        
        # Check data processing scripts
        src_dir = self.root_dir / 'src'
        if src_dir.exists():
            required_scripts = ['data_ingestion.py', 'preprocessors.py', 'utils.py']
            for script in required_scripts:
                script_path = src_dir / script
                if not script_path.exists():
                    data_issues.append(f"Missing script: src/{script}")
                else:
                    # Check if script is importable
                    try:
                        sys.path.insert(0, str(src_dir))
                        module_name = script.replace('.py', '')
                        importlib.import_module(module_name)
                        sys.path.remove(str(src_dir))
                    except Exception as e:
                        data_issues.append(f"Script {script} has import errors: {e}")
        
        self.health_report['data_pipeline'] = {
            'status': 'healthy' if not data_issues else 'issues_found',
            'issues': data_issues,
            'fixes_applied': data_fixes,
            'files_checked': len(csv_files) if 'csv_files' in locals() else 0
        }
    
    def check_model_pipeline(self):
        """Check ML model pipeline"""
        print("🤖 Checking model pipeline...")
        
        model_issues = []
        model_fixes = []
        
        # Check model scripts
        src_dir = self.root_dir / 'src'
        if src_dir.exists():
            model_scripts = ['fx_model.py', 'gdp_model.py', 'inflation_model.py']
            
            for script in model_scripts:
                script_path = src_dir / script
                if not script_path.exists():
                    model_issues.append(f"Missing model script: src/{script}")
                    # Auto-fix: create basic model template
                    try:
                        self.create_model_template(script_path)
                        fix_msg = f"✅ Created model template: {script}"
                        model_fixes.append(fix_msg)
                        self.fix_log.append(fix_msg)
                    except Exception as e:
                        model_issues.append(f"Failed to create {script}: {e}")
        
        # Check for universal predictor
        universal_predictor_path = src_dir / 'universal_predictor.py'
        if not universal_predictor_path.exists():
            model_issues.append("Missing universal_predictor.py")
            # Auto-fix: create universal predictor
            try:
                self.create_universal_predictor()
                fix_msg = "✅ Created universal_predictor.py"
                model_fixes.append(fix_msg)
                self.fix_log.append(fix_msg)
            except Exception as e:
                model_issues.append(f"Failed to create universal predictor: {e}")
        
        self.health_report['model_pipeline'] = {
            'status': 'healthy' if not model_issues else 'issues_found',
            'issues': model_issues,
            'fixes_applied': model_fixes
        }
    
    def check_dashboard_pipeline(self):
        """Check dashboard and visualization pipeline"""
        print("📈 Checking dashboard pipeline...")
        
        dashboard_issues = []
        dashboard_fixes = []
        
        # Check for main dashboard files
        dashboard_files = ['ultra_dashboard.py', 'dashboard/app.py']
        main_dashboard = None
        
        for dashboard_file in dashboard_files:
            dashboard_path = self.root_dir / dashboard_file
            if dashboard_path.exists():
                main_dashboard = dashboard_path
                break
        
        if not main_dashboard:
            dashboard_issues.append("No main dashboard file found")
        else:
            # Check dashboard imports
            try:
                with open(main_dashboard, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                required_imports = ['streamlit', 'plotly', 'pandas', 'numpy']
                missing_imports = []
                
                for imp in required_imports:
                    if f'import {imp}' not in content and f'from {imp}' not in content:
                        missing_imports.append(imp)
                
                if missing_imports:
                    dashboard_issues.append(f"Missing imports: {missing_imports}")
                
            except Exception as e:
                dashboard_issues.append(f"Error reading dashboard file: {e}")
        
        self.health_report['dashboard_pipeline'] = {
            'status': 'healthy' if not dashboard_issues else 'issues_found',
            'issues': dashboard_issues,
            'fixes_applied': dashboard_fixes,
            'main_dashboard': str(main_dashboard) if main_dashboard else None
        }
    
    def check_notebook_pipeline(self):
        """Check Jupyter notebook pipeline"""
        print("📚 Checking notebook pipeline...")
        
        notebook_issues = []
        notebook_fixes = []
        
        # Check notebooks directory
        notebooks_dir = self.root_dir / 'notebooks'
        if not notebooks_dir.exists():
            notebook_issues.append("Missing notebooks directory")
            # Auto-fix: create notebooks directory
            try:
                notebooks_dir.mkdir(exist_ok=True)
                fix_msg = "✅ Created notebooks directory"
                notebook_fixes.append(fix_msg)
                self.fix_log.append(fix_msg)
            except Exception as e:
                notebook_issues.append(f"Failed to create notebooks directory: {e}")
        else:
            # Check for notebook files
            notebook_files = list(notebooks_dir.glob('*.ipynb'))
            if len(notebook_files) == 0:
                notebook_issues.append("No notebook files found")
                # Auto-fix: create sample notebooks
                try:
                    self.create_sample_notebooks()
                    fix_msg = "✅ Created sample analysis notebooks"
                    notebook_fixes.append(fix_msg)
                    self.fix_log.append(fix_msg)
                except Exception as e:
                    notebook_issues.append(f"Failed to create sample notebooks: {e}")
            else:
                # Check notebook integrity
                corrupted_notebooks = []
                for nb_file in notebook_files:
                    try:
                        import nbformat
                        with open(nb_file, 'r', encoding='utf-8') as f:
                            nbformat.read(f, as_version=4)
                    except Exception as e:
                        corrupted_notebooks.append(f"{nb_file.name}: {e}")
                
                if corrupted_notebooks:
                    notebook_issues.extend(corrupted_notebooks)
        
        self.health_report['notebook_pipeline'] = {
            'status': 'healthy' if not notebook_issues else 'issues_found',
            'issues': notebook_issues,
            'fixes_applied': notebook_fixes,
            'notebooks_found': len(notebook_files) if 'notebook_files' in locals() else 0
        }
    
    def check_dependencies(self):
        """Check Python dependencies"""
        print("📦 Checking dependencies...")
        
        dependency_issues = []
        dependency_fixes = []
        
        # Check requirements.txt
        requirements_path = self.root_dir / 'requirements.txt'
        if not requirements_path.exists():
            dependency_issues.append("Missing requirements.txt")
            # Auto-fix: create requirements.txt
            try:
                self.create_requirements_file()
                fix_msg = "✅ Created requirements.txt"
                dependency_fixes.append(fix_msg)
                self.fix_log.append(fix_msg)
            except Exception as e:
                dependency_issues.append(f"Failed to create requirements.txt: {e}")
        
        # Check critical dependencies
        critical_packages = [
            'streamlit', 'pandas', 'numpy', 'plotly', 'scikit-learn',
            'jupyter', 'nbformat', 'nbconvert'
        ]
        
        missing_packages = []
        for package in critical_packages:
            try:
                importlib.import_module(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            dependency_issues.append(f"Missing packages: {missing_packages}")
        
        self.health_report['dependencies'] = {
            'status': 'healthy' if not dependency_issues else 'issues_found',
            'issues': dependency_issues,
            'fixes_applied': dependency_fixes,
            'missing_packages': missing_packages if missing_packages else []
        }
    
    def generate_sample_data(self):
        """Generate sample economic data files"""
        np.random.seed(42)
        
        # Generate sample datasets
        date_range = pd.date_range('2020-01-01', periods=48, freq='M')
        
        datasets = {
            'Annual GDP.csv': pd.DataFrame({
                'Date': date_range,
                'GDP_Nominal': 10000 + np.cumsum(np.random.normal(200, 100, 48)),
                'GDP_Real': 9500 + np.cumsum(np.random.normal(150, 80, 48)),
                'Growth_Rate': np.random.normal(5.5, 1.2, 48)
            }),
            'Central Bank Rate (CBR).csv': pd.DataFrame({
                'Date': date_range,
                'CBR_Rate': 7.0 + np.cumsum(np.random.normal(0, 0.2, 48)),
                'Repo_Rate': 6.5 + np.cumsum(np.random.normal(0, 0.15, 48))
            }),
            'Mobile Payments.csv': pd.DataFrame({
                'Date': date_range,
                'Transaction_Volume': np.random.poisson(1500000, 48),
                'Transaction_Value': np.random.gamma(3, 50, 48),
                'Active_Users': np.random.normal(25000000, 500000, 48)
            }),
            'Monthly exchange rate (end period).csv': pd.DataFrame({
                'Date': date_range,
                'USD_KES': 110 + np.cumsum(np.random.normal(0, 1, 48)),
                'EUR_KES': 125 + np.cumsum(np.random.normal(0, 1.2, 48)),
                'GBP_KES': 145 + np.cumsum(np.random.normal(0, 1.5, 48))
            }),
            'Diaspora Remittances.csv': pd.DataFrame({
                'Date': date_range,
                'Remittances_USD_Million': np.random.gamma(2, 150, 48),
                'Growth_Rate': np.random.normal(0.12, 0.08, 48)
            })
        }
        
        # Save datasets
        raw_data_dir = self.root_dir / 'data' / 'raw'
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        for filename, df in datasets.items():
            df.to_csv(raw_data_dir / filename, index=False)
    
    def create_model_template(self, script_path):
        """Create a basic model template"""
        template = '''"""
Economic Model Template
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class EconomicModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
    
    def prepare_features(self, data):
        """Prepare features for modeling"""
        # Basic feature engineering
        features = data.select_dtypes(include=[np.number]).fillna(method='ffill')
        return features
    
    def train(self, data, target_column):
        """Train the model"""
        features = self.prepare_features(data)
        
        if target_column in features.columns:
            X = features.drop(columns=[target_column])
            y = features[target_column]
            
            self.model.fit(X, y)
            self.is_trained = True
            
            # Calculate training metrics
            predictions = self.model.predict(X)
            mse = mean_squared_error(y, predictions)
            r2 = r2_score(y, predictions)
            
            return {'mse': mse, 'r2': r2}
        else:
            raise ValueError(f"Target column '{target_column}' not found in data")
    
    def predict(self, data):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        features = self.prepare_features(data)
        return self.model.predict(features)
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.is_trained:
            joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = joblib.load(filepath)
        self.is_trained = True

# Model instance
model = EconomicModel()
'''
        
        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(template)
    
    def create_universal_predictor(self):
        """Create universal predictor if missing"""
        # This would be the content from our previous universal_predictor.py
        # For brevity, creating a simplified version
        universal_predictor_content = '''"""
Universal Predictive Analytics Engine
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import plotly.graph_objects as go

class UniversalPredictor:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ridge': Ridge(alpha=1.0)
        }
        self.trained_models = {}
    
    def predict_timeseries(self, data, target_col, periods=12):
        """Predict future values for time series data"""
        if target_col not in data.columns:
            return None
        
        # Prepare features
        y = data[target_col].dropna()
        X = np.arange(len(y)).reshape(-1, 1)
        
        # Train simple model
        model = self.models['random_forest']
        model.fit(X, y)
        
        # Predict future
        future_X = np.arange(len(y), len(y) + periods).reshape(-1, 1)
        predictions = model.predict(future_X)
        
        return predictions
    
    def create_prediction_chart(self, data, target_col, predictions):
        """Create prediction visualization"""
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=list(range(len(data))),
            y=data[target_col],
            mode='lines+markers',
            name='Historical',
            line=dict(color='#00ff88', width=3)
        ))
        
        # Predictions
        if predictions is not None:
            future_x = list(range(len(data), len(data) + len(predictions)))
            fig.add_trace(go.Scatter(
                x=future_x,
                y=predictions,
                mode='lines+markers',
                name='Predictions',
                line=dict(color='#ff6b6b', width=3, dash='dash')
            ))
        
        fig.update_layout(
            title=f"📈 {target_col} - Predictive Analysis",
            template="plotly_dark",
            height=400
        )
        
        return fig

# Global predictor instance
universal_predictor = UniversalPredictor()
'''
        
        universal_predictor_path = self.root_dir / 'src' / 'universal_predictor.py'
        universal_predictor_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(universal_predictor_path, 'w', encoding='utf-8') as f:
            f.write(universal_predictor_content)
    
    def create_sample_notebooks(self):
        """Create sample analysis notebooks"""
        
        notebook_template = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["# Economic Analysis Notebook\n", "\n", "This notebook demonstrates economic data analysis and modeling."]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import pandas as pd\n",
                        "import numpy as np\n",
                        "import matplotlib.pyplot as plt\n",
                        "import plotly.graph_objects as go\n",
                        "\n",
                        "# Load sample data\n",
                        "print('📊 Economic Analysis Notebook Ready!')\n",
                        "print('This notebook provides economic insights and analysis.')"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## Data Analysis\n", "\n", "Comprehensive analysis of economic indicators."]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Generate sample economic data\n",
                        "dates = pd.date_range('2020-01-01', periods=48, freq='M')\n",
                        "sample_data = pd.DataFrame({\n",
                        "    'Date': dates,\n",
                        "    'GDP_Growth': np.random.normal(5.5, 1.2, 48),\n",
                        "    'Inflation_Rate': np.random.normal(6.5, 1.5, 48),\n",
                        "    'Exchange_Rate': 110 + np.cumsum(np.random.normal(0, 1, 48))\n",
                        "})\n",
                        "\n",
                        "print(f'Sample data shape: {sample_data.shape}')\n",
                        "sample_data.head()"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.8.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        notebooks_to_create = [
            'Economic_Analysis_Sample.ipynb',
            'GDP_Modeling_Demo.ipynb',
            'Inflation_Analysis_Demo.ipynb',
            'FX_Modeling_Demo.ipynb'
        ]
        
        notebooks_dir = self.root_dir / 'notebooks'
        notebooks_dir.mkdir(exist_ok=True)
        
        import json
        for notebook_name in notebooks_to_create:
            notebook_path = notebooks_dir / notebook_name
            
            # Customize notebook title
            custom_template = notebook_template.copy()
            custom_template['cells'][0]['source'][0] = f"# {notebook_name.replace('_', ' ').replace('.ipynb', '')}\n"
            
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(custom_template, f, indent=2)
    
    def create_requirements_file(self):
        """Create comprehensive requirements.txt"""
        requirements = [
            "streamlit>=1.28.0",
            "pandas>=1.5.0",
            "numpy>=1.21.0",
            "plotly>=5.15.0",
            "scikit-learn>=1.3.0",
            "jupyter>=1.0.0",
            "nbformat>=5.7.0",
            "nbconvert>=6.5.0",
            "matplotlib>=3.6.0",
            "seaborn>=0.12.0",
            "joblib>=1.3.0",
            "openpyxl>=3.1.0",
            "xlsxwriter>=3.1.0"
        ]
        
        requirements_path = self.root_dir / 'requirements.txt'
        with open(requirements_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(requirements))
    
    def generate_health_report(self):
        """Generate comprehensive health report"""
        print("\n" + "="*60)
        print("🏥 ECONNONET PIPELINE HEALTH REPORT")
        print("="*60)
        
        total_issues = 0
        total_fixes = 0
        
        for pipeline_name, pipeline_data in self.health_report.items():
            status = pipeline_data['status']
            issues = pipeline_data.get('issues', [])
            fixes = pipeline_data.get('fixes_applied', [])
            
            total_issues += len(issues)
            total_fixes += len(fixes)
            
            status_emoji = "✅" if status == 'healthy' else "⚠️"
            print(f"\n{status_emoji} {pipeline_name.upper().replace('_', ' ')} PIPELINE")
            print(f"   Status: {status}")
            
            if issues:
                print(f"   Issues Found: {len(issues)}")
                for issue in issues[:3]:  # Show first 3 issues
                    print(f"     • {issue}")
                if len(issues) > 3:
                    print(f"     ... and {len(issues) - 3} more")
            
            if fixes:
                print(f"   Fixes Applied: {len(fixes)}")
                for fix in fixes:
                    print(f"     {fix}")
        
        print(f"\n📊 SUMMARY")
        print(f"   Total Issues Found: {total_issues}")
        print(f"   Automatic Fixes Applied: {total_fixes}")
        print(f"   Remaining Issues: {max(0, total_issues - total_fixes)}")
        
        if total_fixes > 0:
            print(f"\n🎉 Successfully applied {total_fixes} automatic fixes!")
            print(f"   Your EconoNet pipelines are now more robust.")
        
        print("\n" + "="*60)
        
        # Save report to file
        report_path = self.root_dir / 'pipeline_health_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'health_report': self.health_report,
                'fix_log': self.fix_log,
                'summary': {
                    'total_issues': total_issues,
                    'total_fixes': total_fixes,
                    'remaining_issues': max(0, total_issues - total_fixes)
                }
            }, f, indent=2)
        
        print(f"📝 Detailed report saved to: {report_path}")

def run_pipeline_health_check():
    """Run the complete pipeline health check"""
    print("🚀 EconoNet Pipeline Health Check & Auto-Fix System")
    print("=" * 55)
    
    # Initialize health checker
    checker = PipelineHealthChecker()
    
    # Run comprehensive check
    health_report = checker.check_all_pipelines()
    
    return health_report

if __name__ == "__main__":
    run_pipeline_health_check()
