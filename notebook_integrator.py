"""
EconoNet - Notebook Integration System
=====================================

Advanced system for seamless notebook integration with error handling,
data fixes, and interactive execution capabilities.
"""

import os
import json
import nbformat
import pandas as pd
import numpy as np
from nbconvert.preprocessors import ExecutePreprocessor
import subprocess
import warnings
warnings.filterwarnings('ignore')

class NotebookIntegrator:
    """Enhanced notebook integration with automatic error fixing"""
    
    def __init__(self, notebooks_dir="notebooks"):
        self.notebooks_dir = notebooks_dir
        self.fix_registry = {}
        
    def fix_plotly_data(self, data):
        """Convert range/arange objects to lists for Plotly compatibility"""
        if isinstance(data, range):
            return list(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif hasattr(data, 'tolist'):
            return data.tolist()
        return data
    
    def apply_range_fixes(self, notebook_path):
        """Apply range/arange fixes to notebook cells"""
        
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        fixes_applied = 0
        
        # Iterate through cells
        for cell in nb.cells:
            if cell.cell_type == 'code':
                source = cell.source
                
                # Fix common range issues
                if 'x=range(' in source or 'x=np.arange(' in source:
                    # Add fix_plotly_data wrapper
                    source = source.replace('x=range(', 'x=fix_plotly_data(range(')
                    source = source.replace('x=np.arange(', 'x=fix_plotly_data(np.arange(')
                    cell.source = source
                    fixes_applied += 1
                
                if 'y=range(' in source or 'y=np.arange(' in source:
                    source = source.replace('y=range(', 'y=fix_plotly_data(range(')
                    source = source.replace('y=np.arange(', 'y=fix_plotly_data(np.arange(')
                    cell.source = source
                    fixes_applied += 1
        
        # Add fix function to first cell if fixes were applied
        if fixes_applied > 0:
            fix_cell = nbformat.v4.new_code_cell('''
# Plotly compatibility fix
def fix_plotly_data(data):
    """Convert range/arange objects to lists for Plotly compatibility"""
    if isinstance(data, range):
        return list(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif hasattr(data, 'tolist'):
        return data.tolist()
    return data
''')
            nb.cells.insert(0, fix_cell)
        
        # Save fixed notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        return fixes_applied
    
    def get_available_notebooks(self):
        """Get list of available notebooks with metadata"""
        notebooks = []
        
        if not os.path.exists(self.notebooks_dir):
            return notebooks
            
        for file in os.listdir(self.notebooks_dir):
            if file.endswith('.ipynb'):
                notebook_path = os.path.join(self.notebooks_dir, file)
                
                # Get basic metadata
                try:
                    with open(notebook_path, 'r', encoding='utf-8') as f:
                        nb = nbformat.read(f, as_version=4)
                    
                    # Extract title from first markdown cell
                    title = file.replace('.ipynb', '').replace('_', ' ')
                    description = "Economic analysis notebook"
                    
                    for cell in nb.cells:
                        if cell.cell_type == 'markdown' and len(cell.source) > 10:
                            lines = cell.source.split('\n')
                            for line in lines:
                                if line.startswith('#'):
                                    title = line.replace('#', '').strip()
                                    break
                            if len(lines) > 1:
                                description = lines[1].strip()
                            break
                    
                    notebooks.append({
                        'filename': file,
                        'title': title,
                        'description': description,
                        'path': notebook_path,
                        'cell_count': len(nb.cells)
                    })
                    
                except Exception as e:
                    print(f"Error reading {file}: {e}")
        
        return notebooks
    
    def execute_notebook_section(self, notebook_path, section_name=None):
        """Execute specific section of notebook"""
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # Apply fixes before execution
            self.apply_range_fixes(notebook_path)
            
            # Execute notebook
            ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
            ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
            
            return {
                'status': 'success',
                'cells_executed': len(nb.cells),
                'outputs': self._extract_outputs(nb)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'cells_executed': 0
            }
    
    def _extract_outputs(self, nb):
        """Extract outputs from executed notebook"""
        outputs = []
        
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code' and cell.outputs:
                cell_outputs = []
                for output in cell.outputs:
                    if output.output_type == 'execute_result':
                        if 'data' in output:
                            cell_outputs.append({
                                'type': 'result',
                                'data': output.data
                            })
                    elif output.output_type == 'display_data':
                        if 'data' in output:
                            cell_outputs.append({
                                'type': 'display',
                                'data': output.data
                            })
                
                if cell_outputs:
                    outputs.append({
                        'cell_index': i,
                        'outputs': cell_outputs
                    })
        
        return outputs

# Global integrator instance
notebook_integrator = NotebookIntegrator()

def get_notebook_data(notebook_name):
    """Get processed data from notebook execution"""
    
    # Simulated notebook data based on name
    if "Risk" in notebook_name:
        return {
            'var_95': -0.0341,
            'var_99': -0.0523,
            'cvar_95': -0.0456,
            'monte_carlo_paths': np.random.normal(0, 0.02, (1000, 252)),
            'risk_metrics': {
                'volatility': 0.021,
                'sharpe_ratio': 0.84,
                'max_drawdown': -0.123
            }
        }
    
    elif "GDP" in notebook_name:
        return {
            'gdp_growth_rate': 5.8,
            'gdp_forecast': [5.5, 5.7, 5.9, 6.1, 5.8, 5.6],
            'gdp_components': {
                'agriculture': 22.3,
                'manufacturing': 15.8,
                'services': 51.2,
                'other': 10.7
            },
            'quarterly_data': pd.DataFrame({
                'quarter': pd.date_range('2023-Q1', periods=8, freq='Q'),
                'gdp_growth': [5.1, 5.4, 5.8, 6.0, 5.7, 5.5, 5.8, 5.9]
            })
        }
    
    elif "Exchange" in notebook_name or "FX" in notebook_name:
        return {
            'current_rate': 132.45,
            'volatility': 0.021,
            'forecast': pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=12, freq='M'),
                'rate': 132.45 + np.cumsum(np.random.normal(0, 0.5, 12)),
                'upper_bound': 132.45 + np.cumsum(np.random.normal(0.5, 0.5, 12)),
                'lower_bound': 132.45 + np.cumsum(np.random.normal(-0.5, 0.5, 12))
            }),
            'technical_indicators': {
                'rsi': 68.4,
                'macd': 0.234,
                'bollinger_position': 0.73
            }
        }
    
    elif "Inflation" in notebook_name:
        return {
            'current_inflation': 6.8,
            'core_inflation': 5.9,
            'food_inflation': 8.2,
            'forecast': pd.DataFrame({
                'month': pd.date_range('2024-01-01', periods=12, freq='M'),
                'headline': 6.8 + np.random.normal(0, 0.3, 12),
                'core': 5.9 + np.random.normal(0, 0.2, 12)
            }),
            'components': {
                'food': 35.2,
                'transport': 12.8,
                'housing': 18.4,
                'other': 33.6
            }
        }
    
    else:
        return {
            'status': 'No specific data available',
            'generic_metrics': {
                'data_points': 1000,
                'time_range': '2020-2024',
                'variables': 15
            }
        }

def fix_all_notebooks():
    """Apply range fixes to all notebooks in the directory"""
    integrator = NotebookIntegrator()
    notebooks_fixed = 0
    total_fixes = 0
    
    for notebook in integrator.get_available_notebooks():
        fixes = integrator.apply_range_fixes(notebook['path'])
        if fixes > 0:
            notebooks_fixed += 1
            total_fixes += fixes
            print(f"✅ Fixed {fixes} range issues in {notebook['filename']}")
    
    print(f"\n🎯 Summary: Fixed {total_fixes} issues across {notebooks_fixed} notebooks")
    return notebooks_fixed, total_fixes

if __name__ == "__main__":
    # Fix all notebooks when run directly
    fix_all_notebooks()
