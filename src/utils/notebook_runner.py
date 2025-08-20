"""
Notebook Runner Utility
======================

Utility functions to execute Jupyter notebooks and display results
in Streamlit applications. Supports both direct execution and
result caching for performance.
"""

import os
import sys
import tempfile
import subprocess
import json
import base64
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import nbformat
from nbconvert import HTMLExporter, PythonExporter
from nbconvert.preprocessors import ExecutePreprocessor
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import warnings

warnings.filterwarnings('ignore')

class NotebookRunner:
    """
    Handles execution and display of Jupyter notebooks within Streamlit
    """
    
    def __init__(self, notebooks_dir: str = "notebooks"):
        """
        Initialize notebook runner
        
        Args:
            notebooks_dir: Directory containing notebooks
        """
        self.notebooks_dir = Path(notebooks_dir)
        self.cache_dir = Path("artifacts/notebook_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def list_notebooks(self) -> List[Dict[str, str]]:
        """
        List all available notebooks
        
        Returns:
            List of notebook metadata
        """
        notebooks = []
        
        if not self.notebooks_dir.exists():
            return notebooks
            
        for notebook_path in self.notebooks_dir.glob("*.ipynb"):
            # Skip checkpoint files
            if ".ipynb_checkpoints" in str(notebook_path):
                continue
                
            try:
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    nb = nbformat.read(f, as_version=4)
                
                # Extract title from first markdown cell or filename
                title = notebook_path.stem.replace('_', ' ').title()
                description = "No description available"
                
                for cell in nb.cells:
                    if cell.cell_type == 'markdown' and cell.source.strip():
                        lines = cell.source.split('\n')
                        if lines[0].startswith('#'):
                            title = lines[0].strip('# ')
                        if len(lines) > 1:
                            description = lines[1].strip()
                        break
                
                notebooks.append({
                    'name': notebook_path.name,
                    'path': str(notebook_path),
                    'title': title,
                    'description': description,
                    'size': f"{notebook_path.stat().st_size / 1024:.1f} KB"
                })
                
            except Exception as e:
                st.warning(f"Could not read notebook {notebook_path.name}: {e}")
                
        return sorted(notebooks, key=lambda x: x['name'])
    
    @st.cache_data
    def execute_notebook(_self, notebook_path: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a notebook and return results
        
        Args:
            notebook_path: Path to notebook file
            parameters: Parameters to inject into notebook
            
        Returns:
            Dictionary containing execution results
        """
        try:
            # Read notebook
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # Inject parameters if provided
            if parameters:
                _self._inject_parameters(nb, parameters)
            
            # Execute notebook
            ep = ExecutePreprocessor(
                timeout=600,  # 10 minutes timeout
                kernel_name='python3',
                allow_errors=True
            )
            
            ep.preprocess(nb, {'metadata': {'path': str(Path(notebook_path).parent)}})
            
            # Extract results
            results = _self._extract_results(nb)
            
            return {
                'status': 'success',
                'results': results,
                'notebook': nb
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'results': {}
            }
    
    def _inject_parameters(self, notebook: nbformat.NotebookNode, parameters: Dict[str, Any]) -> None:
        """Inject parameters into notebook"""
        # Create parameter cell
        param_code = "\n".join([f"{k} = {repr(v)}" for k, v in parameters.items()])
        
        param_cell = nbformat.v4.new_code_cell(
            source=f"# Injected parameters\n{param_code}",
            metadata={"tags": ["parameters"]}
        )
        
        # Insert as second cell (after imports)
        if len(notebook.cells) > 1:
            notebook.cells.insert(1, param_cell)
        else:
            notebook.cells.insert(0, param_cell)
    
    def _extract_results(self, notebook: nbformat.NotebookNode) -> Dict[str, Any]:
        """Extract outputs and results from executed notebook"""
        results = {
            'outputs': [],
            'plots': [],
            'dataframes': [],
            'metrics': {},
            'errors': []
        }
        
        for cell_idx, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code' and hasattr(cell, 'outputs'):
                for output in cell.outputs:
                    try:
                        # Handle different output types
                        if output.output_type == 'display_data':
                            self._extract_display_data(output, results)
                        elif output.output_type == 'execute_result':
                            self._extract_execute_result(output, results)
                        elif output.output_type == 'stream':
                            self._extract_stream_output(output, results)
                        elif output.output_type == 'error':
                            results['errors'].append({
                                'cell': cell_idx,
                                'error': output.traceback
                            })
                    except Exception as e:
                        results['errors'].append({
                            'cell': cell_idx,
                            'error': f"Error extracting output: {e}"
                        })
        
        return results
    
    def _extract_display_data(self, output: Dict, results: Dict) -> None:
        """Extract display data (plots, images, etc.)"""
        data = output.get('data', {})
        
        # Handle Plotly plots
        if 'application/vnd.plotly.v1+json' in data:
            results['plots'].append({
                'type': 'plotly',
                'data': data['application/vnd.plotly.v1+json']
            })
        
        # Handle images
        elif 'image/png' in data:
            results['plots'].append({
                'type': 'image',
                'data': data['image/png']
            })
        
        # Handle HTML
        elif 'text/html' in data:
            html_content = data['text/html']
            # Check if it's a DataFrame
            if 'dataframe' in html_content.lower():
                results['dataframes'].append(html_content)
            else:
                results['outputs'].append({
                    'type': 'html',
                    'content': html_content
                })
    
    def _extract_execute_result(self, output: Dict, results: Dict) -> None:
        """Extract execution results"""
        data = output.get('data', {})
        
        if 'text/plain' in data:
            text_output = data['text/plain']
            # Try to parse as metrics
            if any(keyword in text_output.lower() for keyword in ['accuracy', 'mse', 'rmse', 'r2', 'score']):
                results['metrics'].update(self._parse_metrics(text_output))
            else:
                results['outputs'].append({
                    'type': 'text',
                    'content': text_output
                })
    
    def _extract_stream_output(self, output: Dict, results: Dict) -> None:
        """Extract stream output (print statements)"""
        text = output.get('text', '')
        if text.strip():
            results['outputs'].append({
                'type': 'stream',
                'content': text.strip()
            })
    
    def _parse_metrics(self, text: str) -> Dict[str, float]:
        """Parse metrics from text output"""
        metrics = {}
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if ':' in line:
                try:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value_str = value.strip()
                    
                    # Try to extract numeric value
                    import re
                    numbers = re.findall(r'-?\d+\.?\d*', value_str)
                    if numbers:
                        metrics[key] = float(numbers[0])
                except:
                    continue
        
        return metrics
    
    def display_notebook_results(self, results: Dict[str, Any]) -> None:
        """
        Display notebook execution results in Streamlit
        
        Args:
            results: Results from execute_notebook
        """
        if results['status'] == 'error':
            st.error(f"Notebook execution failed: {results['error']}")
            return
        
        notebook_results = results['results']
        
        # Display metrics
        if notebook_results.get('metrics'):
            st.subheader("📊 Key Metrics")
            
            metrics = notebook_results['metrics']
            cols = st.columns(min(len(metrics), 4))
            
            for idx, (key, value) in enumerate(metrics.items()):
                with cols[idx % len(cols)]:
                    st.metric(key, f"{value:.4f}" if isinstance(value, float) else str(value))
        
        # Display plots
        if notebook_results.get('plots'):
            st.subheader("📈 Visualizations")
            
            for idx, plot_data in enumerate(notebook_results['plots']):
                if plot_data['type'] == 'plotly':
                    try:
                        fig = go.Figure(plot_data['data'])
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error displaying plot {idx + 1}: {e}")
                
                elif plot_data['type'] == 'image':
                    try:
                        import base64
                        from PIL import Image
                        from io import BytesIO
                        
                        image_data = base64.b64decode(plot_data['data'])
                        image = Image.open(BytesIO(image_data))
                        st.image(image, use_column_width=True)
                    except Exception as e:
                        st.error(f"Error displaying image {idx + 1}: {e}")
        
        # Display DataFrames
        if notebook_results.get('dataframes'):
            st.subheader("📋 Data Tables")
            
            for idx, df_html in enumerate(notebook_results['dataframes']):
                st.markdown(f"**Table {idx + 1}:**")
                st.components.v1.html(df_html, height=400, scrolling=True)
        
        # Display text outputs
        if notebook_results.get('outputs'):
            with st.expander("📝 Detailed Output", expanded=False):
                for output in notebook_results['outputs']:
                    if output['type'] == 'text':
                        st.code(output['content'], language='text')
                    elif output['type'] == 'stream':
                        st.text(output['content'])
                    elif output['type'] == 'html':
                        st.components.v1.html(output['content'], height=300)
        
        # Display errors if any
        if notebook_results.get('errors'):
            with st.expander("⚠️ Errors and Warnings", expanded=False):
                for error in notebook_results['errors']:
                    st.error(f"Cell {error['cell']}: {error['error']}")
    
    def convert_to_html(self, notebook_path: str) -> str:
        """
        Convert notebook to HTML for display
        
        Args:
            notebook_path: Path to notebook file
            
        Returns:
            HTML content
        """
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            html_exporter = HTMLExporter()
            html_exporter.template_name = 'classic'
            
            (body, resources) = html_exporter.from_notebook_node(nb)
            return body
            
        except Exception as e:
            return f"<p>Error converting notebook to HTML: {e}</p>"
    
    def get_notebook_preview(self, notebook_path: str, max_cells: int = 5) -> str:
        """
        Get a preview of notebook content
        
        Args:
            notebook_path: Path to notebook file
            max_cells: Maximum number of cells to preview
            
        Returns:
            Preview text
        """
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            preview_parts = []
            cell_count = 0
            
            for cell in nb.cells:
                if cell_count >= max_cells:
                    break
                
                if cell.cell_type == 'markdown':
                    preview_parts.append(f"**Markdown:** {cell.source[:200]}...")
                elif cell.cell_type == 'code':
                    preview_parts.append(f"**Code:** {cell.source[:200]}...")
                
                cell_count += 1
            
            if len(nb.cells) > max_cells:
                preview_parts.append(f"... and {len(nb.cells) - max_cells} more cells")
            
            return "\n\n".join(preview_parts)
            
        except Exception as e:
            return f"Error reading notebook: {e}"

def create_parameter_form(notebook_name: str) -> Dict[str, Any]:
    """
    Create a parameter input form for notebook execution
    
    Args:
        notebook_name: Name of the notebook
        
    Returns:
        Dictionary of parameters
    """
    parameters = {}
    
    st.subheader("⚙️ Notebook Parameters")
    
    # Common parameters based on notebook type
    if "predictive" in notebook_name.lower() or "forecast" in notebook_name.lower():
        parameters['forecast_horizon'] = st.slider(
            "Forecast Horizon (months)", 
            min_value=3, 
            max_value=36, 
            value=12
        )
        parameters['confidence_level'] = st.slider(
            "Confidence Level", 
            min_value=0.8, 
            max_value=0.99, 
            value=0.95, 
            step=0.01
        )
    
    elif "risk" in notebook_name.lower():
        parameters['var_confidence'] = st.slider(
            "VaR Confidence Level", 
            min_value=0.90, 
            max_value=0.99, 
            value=0.95, 
            step=0.01
        )
        parameters['n_simulations'] = st.slider(
            "Monte Carlo Simulations", 
            min_value=100, 
            max_value=10000, 
            value=1000, 
            step=100
        )
    
    elif "eda" in notebook_name.lower() or "exploration" in notebook_name.lower():
        parameters['correlation_threshold'] = st.slider(
            "Correlation Threshold", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.5, 
            step=0.1
        )
        parameters['outlier_threshold'] = st.slider(
            "Outlier Detection Threshold", 
            min_value=1.0, 
            max_value=3.0, 
            value=2.0, 
            step=0.1
        )
    
    # Allow custom parameters
    with st.expander("🔧 Custom Parameters", expanded=False):
        st.markdown("Add custom parameters as key=value pairs (one per line):")
        custom_params = st.text_area(
            "Custom Parameters",
            placeholder="variable_name=value\nanother_param=123",
            help="Enter parameters in the format: variable_name=value"
        )
        
        if custom_params.strip():
            for line in custom_params.strip().split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Try to parse as number, otherwise keep as string
                    try:
                        if '.' in value:
                            parameters[key] = float(value)
                        else:
                            parameters[key] = int(value)
                    except ValueError:
                        parameters[key] = value
    
    return parameters

# Export main classes and functions
__all__ = ['NotebookRunner', 'create_parameter_form']
