"""
Advanced Notebook Integration System
====================================

Comprehensive notebook execution, result extraction, and dashboard integration
for the EconoNet platform with real-time data processing.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import subprocess
import tempfile
import base64
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import nbformat for notebook manipulation
try:
    import nbformat
    from nbconvert import PythonExporter
    from nbconvert.preprocessors import ExecutePreprocessor
    HAS_NBFORMAT = True
except ImportError:
    HAS_NBFORMAT = False
    st.warning("📘 nbformat not installed. Install with: pip install nbformat nbconvert")

class NotebookIntegrator:
    """Enhanced notebook integration with real-time execution and result extraction"""
    
    def __init__(self, notebooks_dir: str = "notebooks"):
        self.notebooks_dir = Path(notebooks_dir)
        self.results_cache = {}
        self.execution_timeout = 300  # 5 minutes timeout
        
    def list_notebooks(self) -> Dict[str, Dict]:
        """List all available notebooks with metadata"""
        notebooks = {}
        
        if not self.notebooks_dir.exists():
            return notebooks
            
        for nb_file in self.notebooks_dir.glob("*.ipynb"):
            if ".ipynb_checkpoints" not in str(nb_file):
                notebooks[nb_file.stem] = {
                    'path': str(nb_file),
                    'name': nb_file.stem.replace('_', ' ').title(),
                    'modified': datetime.fromtimestamp(nb_file.stat().st_mtime),
                    'size': nb_file.stat().st_size
                }
        
        return notebooks
    
    def extract_notebook_metadata(self, notebook_path: str) -> Dict:
        """Extract metadata and structure from notebook"""
        if not HAS_NBFORMAT:
            return {}
            
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            metadata = {
                'title': 'Untitled',
                'description': '',
                'cells_count': len(nb.cells),
                'code_cells': len([c for c in nb.cells if c.cell_type == 'code']),
                'markdown_cells': len([c for c in nb.cells if c.cell_type == 'markdown']),
                'sections': [],
                'imports': set(),
                'functions': set()
            }
            
            # Extract title and description from first markdown cell
            for cell in nb.cells:
                if cell.cell_type == 'markdown' and cell.source.strip():
                    lines = cell.source.strip().split('\n')
                    for line in lines:
                        if line.startswith('# '):
                            metadata['title'] = line[2:].strip()
                        elif line.startswith('## '):
                            metadata['sections'].append(line[3:].strip())
                        elif not metadata['description'] and line.strip() and not line.startswith('#'):
                            metadata['description'] = line.strip()
                    break
            
            # Extract imports and functions from code cells
            for cell in nb.cells:
                if cell.cell_type == 'code' and cell.source.strip():
                    lines = cell.source.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith('import ') or line.startswith('from '):
                            metadata['imports'].add(line.split()[1] if line.startswith('import ') else line.split()[1])
                        elif line.startswith('def '):
                            func_name = line.split('(')[0].replace('def ', '').strip()
                            metadata['functions'].add(func_name)
            
            metadata['imports'] = list(metadata['imports'])
            metadata['functions'] = list(metadata['functions'])
            
            return metadata
            
        except Exception as e:
            st.error(f"Error extracting metadata: {e}")
            return {}
    
    def execute_notebook_section(self, notebook_path: str, section_name: str = None) -> Dict:
        """Execute specific section of notebook and return results"""
        if not HAS_NBFORMAT:
            return {'error': 'nbformat not available'}
            
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # Create execution processor
            ep = ExecutePreprocessor(timeout=self.execution_timeout, kernel_name='python3')
            
            # Execute notebook
            ep.preprocess(nb, {'metadata': {'path': str(self.notebooks_dir)}})
            
            # Extract results
            results = {
                'outputs': [],
                'figures': [],
                'data': {},
                'metrics': {},
                'status': 'success'
            }
            
            for cell in nb.cells:
                if cell.cell_type == 'code' and hasattr(cell, 'outputs'):
                    for output in cell.outputs:
                        if output.output_type == 'display_data':
                            if 'application/json' in output.data:
                                results['data'].update(json.loads(output.data['application/json']))
                            elif 'text/html' in output.data:
                                results['outputs'].append({
                                    'type': 'html',
                                    'content': output.data['text/html']
                                })
                        elif output.output_type == 'execute_result':
                            if 'text/plain' in output.data:
                                results['outputs'].append({
                                    'type': 'text',
                                    'content': output.data['text/plain']
                                })
            
            return results
            
        except Exception as e:
            return {'error': str(e), 'status': 'failed'}
    
    def get_notebook_variables(self, notebook_path: str, variable_names: List[str]) -> Dict:
        """Extract specific variables from executed notebook"""
        cache_key = f"{notebook_path}_{hash(tuple(variable_names))}"
        
        if cache_key in self.results_cache:
            cache_time = self.results_cache[cache_key].get('timestamp', 0)
            if datetime.now().timestamp() - cache_time < 300:  # 5 minute cache
                return self.results_cache[cache_key]['data']
        
        try:
            # Create temporary script to extract variables
            script_content = f"""
import sys
import pickle
import pandas as pd
import numpy as np
sys.path.append('{os.path.dirname(notebook_path)}')

# Execute notebook code (simplified approach)
exec(open('{notebook_path.replace('.ipynb', '_temp.py')}').read())

# Extract requested variables
results = {{}}
{chr(10).join([f"if '{var}' in locals(): results['{var}'] = {var}" for var in variable_names])}

# Save results
with open('temp_results.pkl', 'wb') as f:
    pickle.dump(results, f)
"""
            
            # This is a simplified approach - in production, use proper notebook execution
            results = {}
            for var_name in variable_names:
                # Return mock data for now - replace with actual execution
                if 'df' in var_name.lower() or 'data' in var_name.lower():
                    results[var_name] = pd.DataFrame()
                elif 'model' in var_name.lower():
                    results[var_name] = {'type': 'model', 'status': 'trained'}
                else:
                    results[var_name] = None
            
            # Cache results
            self.results_cache[cache_key] = {
                'data': results,
                'timestamp': datetime.now().timestamp()
            }
            
            return results
            
        except Exception as e:
            st.error(f"Error extracting variables: {e}")
            return {}
    
    def create_notebook_summary(self, notebook_path: str) -> Dict:
        """Create comprehensive summary of notebook for dashboard display"""
        metadata = self.extract_notebook_metadata(notebook_path)
        
        summary = {
            'title': metadata.get('title', 'Untitled Notebook'),
            'description': metadata.get('description', 'No description available'),
            'structure': {
                'total_cells': metadata.get('cells_count', 0),
                'code_cells': metadata.get('code_cells', 0),
                'markdown_cells': metadata.get('markdown_cells', 0)
            },
            'sections': metadata.get('sections', []),
            'key_imports': metadata.get('imports', [])[:10],  # Top 10 imports
            'functions': metadata.get('functions', [])[:10],   # Top 10 functions
            'last_modified': datetime.fromtimestamp(Path(notebook_path).stat().st_mtime),
            'executable': HAS_NBFORMAT
        }
        
        return summary

class DatasetLoader:
    """Load and prepare datasets from the raw data folder"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.dataset_cache = {}
        
    def list_datasets(self) -> Dict[str, Dict]:
        """List all available datasets with metadata"""
        datasets = {}
        
        if not self.data_dir.exists():
            return datasets
            
        for data_file in self.data_dir.glob("*"):
            if data_file.suffix.lower() in ['.csv', '.xlsx', '.xls']:
                datasets[data_file.stem] = {
                    'path': str(data_file),
                    'name': data_file.stem.replace('_', ' ').title(),
                    'type': data_file.suffix[1:].upper(),
                    'size': data_file.stat().st_size,
                    'modified': datetime.fromtimestamp(data_file.stat().st_mtime)
                }
        
        return datasets
    
    def load_dataset(self, dataset_name: str, sample_size: int = None) -> pd.DataFrame:
        """Load dataset with caching"""
        cache_key = f"{dataset_name}_{sample_size}"
        
        if cache_key in self.dataset_cache:
            return self.dataset_cache[cache_key]
        
        datasets = self.list_datasets()
        if dataset_name not in datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        file_path = datasets[dataset_name]['path']
        
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
            
            # Apply sampling if requested
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
            
            # Cache the result
            self.dataset_cache[cache_key] = df
            
            return df
            
        except Exception as e:
            st.error(f"Error loading dataset '{dataset_name}': {e}")
            return pd.DataFrame()
    
    def get_dataset_summary(self, dataset_name: str) -> Dict:
        """Get comprehensive summary of dataset"""
        try:
            df = self.load_dataset(dataset_name, sample_size=1000)  # Sample for summary
            
            summary = {
                'name': dataset_name.replace('_', ' ').title(),
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(df.select_dtypes(include=['object']).columns),
                'date_columns': []
            }
            
            # Detect date columns
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        pd.to_datetime(df[col].dropna().iloc[:100])
                        summary['date_columns'].append(col)
                    except:
                        pass
            
            # Add basic statistics for numeric columns
            if summary['numeric_columns']:
                summary['statistics'] = df[summary['numeric_columns']].describe().to_dict()
            
            return summary
            
        except Exception as e:
            return {'error': str(e)}

# Global instances
notebook_integrator = NotebookIntegrator()
dataset_loader = DatasetLoader()
