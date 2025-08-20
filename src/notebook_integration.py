"""
Advanced Notebook Integration System for EconoNet
===============================================

Comprehensive system for executing and integrating all Jupyter notebooks
into the Streamlit application with proper error handling and results display.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import subprocess
import nbformat
from nbconvert import PythonExporter
from nbconvert.preprocessors import ExecutePreprocessor
import io
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Any, Optional, Tuple
import warnings
import traceback
from pathlib import Path

warnings.filterwarnings('ignore')

class NotebookExecutor:
    """Execute Jupyter notebooks and capture results"""
    
    def __init__(self, notebooks_path: str = "notebooks"):
        self.notebooks_path = Path(notebooks_path)
        self.execution_results = {}
        self.available_notebooks = self._scan_notebooks()
        
    def _scan_notebooks(self) -> Dict[str, Dict[str, Any]]:
        """Scan and catalog available notebooks"""
        
        notebooks = {}
        
        if not self.notebooks_path.exists():
            return notebooks
        
        notebook_files = list(self.notebooks_path.glob("*.ipynb"))
        
        for nb_file in notebook_files:
            try:
                with open(nb_file, 'r', encoding='utf-8') as f:
                    nb = nbformat.read(f, as_version=4)
                
                # Extract metadata
                title = nb_file.stem.replace('_', ' ').title()
                description = self._extract_description(nb)
                cell_count = len(nb.cells)
                
                notebooks[nb_file.stem] = {
                    'title': title,
                    'description': description,
                    'file_path': str(nb_file),
                    'cell_count': cell_count,
                    'last_modified': os.path.getmtime(nb_file),
                    'category': self._categorize_notebook(nb_file.stem)
                }
                
            except Exception as e:
                print(f"Error scanning notebook {nb_file}: {e}")
        
        return notebooks
    
    def _extract_description(self, notebook: nbformat.NotebookNode) -> str:
        """Extract description from notebook markdown cells"""
        
        for cell in notebook.cells[:3]:  # Check first 3 cells
            if cell.cell_type == 'markdown':
                content = cell.source
                if len(content) > 50:
                    # Take first meaningful line
                    lines = content.split('\n')
                    for line in lines:
                        if line.strip() and not line.startswith('#'):
                            return line.strip()[:200] + "..."
        
        return "Economic analysis notebook"
    
    def _categorize_notebook(self, notebook_name: str) -> str:
        """Categorize notebook based on name"""
        
        name_lower = notebook_name.lower()
        
        if any(word in name_lower for word in ['inflation', 'price']):
            return 'Inflation Analysis'
        elif any(word in name_lower for word in ['fx', 'exchange', 'currency']):
            return 'Foreign Exchange'
        elif any(word in name_lower for word in ['gdp', 'growth', 'economic']):
            return 'GDP & Growth'
        elif any(word in name_lower for word in ['risk', 'var', 'stress']):
            return 'Risk Analysis'
        elif any(word in name_lower for word in ['trade', 'export', 'import']):
            return 'Trade Analysis'
        elif any(word in name_lower for word in ['debt', 'fiscal', 'government']):
            return 'Fiscal Analysis'
        elif any(word in name_lower for word in ['monetary', 'policy', 'cbr']):
            return 'Monetary Policy'
        elif any(word in name_lower for word in ['neural', 'ai', 'ml', 'prophet']):
            return 'AI/ML Models'
        elif 'eda' in name_lower or 'exploration' in name_lower:
            return 'Data Exploration'
        else:
            return 'General Analysis'
    
    def execute_notebook(self, notebook_name: str, timeout: int = 300) -> Dict[str, Any]:
        """Execute a notebook and return results"""
        
        if notebook_name not in self.available_notebooks:
            return {'error': f"Notebook '{notebook_name}' not found"}
        
        notebook_info = self.available_notebooks[notebook_name]
        notebook_path = notebook_info['file_path']
        
        try:
            # Read notebook
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # Create executor
            ep = ExecutePreprocessor(
                timeout=timeout,
                kernel_name='python3',
                allow_errors=True
            )
            
            # Execute notebook
            executed_nb, resources = ep.preprocess(nb, {'metadata': {'path': str(self.notebooks_path)}})
            
            # Extract results
            results = self._extract_execution_results(executed_nb)
            
            execution_result = {
                'status': 'success',
                'notebook_name': notebook_name,
                'title': notebook_info['title'],
                'category': notebook_info['category'],
                'execution_time': pd.Timestamp.now(),
                'results': results,
                'error': None
            }
            
            self.execution_results[notebook_name] = execution_result
            return execution_result
            
        except Exception as e:
            error_result = {
                'status': 'error',
                'notebook_name': notebook_name,
                'title': notebook_info['title'],
                'category': notebook_info['category'],
                'execution_time': pd.Timestamp.now(),
                'results': {},
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            
            self.execution_results[notebook_name] = error_result
            return error_result
    
    def _extract_execution_results(self, notebook: nbformat.NotebookNode) -> Dict[str, Any]:
        """Extract results from executed notebook"""
        
        results = {
            'outputs': [],
            'plots': [],
            'dataframes': [],
            'metrics': {},
            'summary': ""
        }
        
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code' and hasattr(cell, 'outputs'):
                for output in cell.outputs:
                    if output.output_type == 'display_data':
                        # Handle plots and visualizations
                        if 'image/png' in output.data:
                            results['plots'].append({
                                'cell_index': i,
                                'type': 'plot',
                                'data': output.data['image/png']
                            })
                    
                    elif output.output_type == 'execute_result':
                        # Handle dataframes and metrics
                        if 'text/html' in output.data:
                            # Likely a DataFrame
                            results['dataframes'].append({
                                'cell_index': i,
                                'html': output.data['text/html']
                            })
                        elif 'text/plain' in output.data:
                            text_output = output.data['text/plain']
                            results['outputs'].append({
                                'cell_index': i,
                                'type': 'text',
                                'content': text_output
                            })
                    
                    elif output.output_type == 'stream':
                        # Handle print outputs
                        results['outputs'].append({
                            'cell_index': i,
                            'type': 'stream',
                            'content': output.text
                        })
        
        return results
    
    def get_notebook_categories(self) -> Dict[str, List[str]]:
        """Get notebooks organized by category"""
        
        categories = {}
        
        for notebook_name, info in self.available_notebooks.items():
            category = info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(notebook_name)
        
        return categories
    
    def get_notebook_summary(self) -> Dict[str, Any]:
        """Get summary of all available notebooks"""
        
        categories = self.get_notebook_categories()
        
        summary = {
            'total_notebooks': len(self.available_notebooks),
            'categories': list(categories.keys()),
            'notebooks_by_category': categories,
            'recently_executed': [],
            'execution_stats': {
                'successful': 0,
                'failed': 0,
                'total_executed': len(self.execution_results)
            }
        }
        
        # Add execution statistics
        for result in self.execution_results.values():
            if result['status'] == 'success':
                summary['execution_stats']['successful'] += 1
            else:
                summary['execution_stats']['failed'] += 1
        
        # Add recently executed notebooks
        recent_executions = sorted(
            self.execution_results.values(),
            key=lambda x: x['execution_time'],
            reverse=True
        )[:5]
        
        summary['recently_executed'] = [
            {
                'name': r['notebook_name'],
                'title': r['title'],
                'status': r['status'],
                'time': r['execution_time']
            }
            for r in recent_executions
        ]
        
        return summary


class StreamlitNotebookInterface:
    """Streamlit interface for notebook execution and display"""
    
    def __init__(self, executor: NotebookExecutor):
        self.executor = executor
    
    def render_notebook_selector(self) -> Optional[str]:
        """Render notebook selection interface"""
        
        st.markdown("""
        <div style="background: #f8fafc; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <h3><i class="fas fa-book"></i> Available Analysis Notebooks</h3>
        </div>
        """, unsafe_allow_html=True)
        
        categories = self.executor.get_notebook_categories()
        
        if not categories:
            st.warning("No notebooks found in the notebooks directory")
            return None
        
        # Category selection
        selected_category = st.selectbox(
            "Select Analysis Category:",
            options=list(categories.keys()),
            help="Choose the type of analysis you want to run"
        )
        
        if selected_category:
            notebooks_in_category = categories[selected_category]
            
            # Notebook selection
            notebook_options = {}
            for nb_name in notebooks_in_category:
                nb_info = self.executor.available_notebooks[nb_name]
                notebook_options[nb_info['title']] = nb_name
            
            selected_title = st.selectbox(
                "Select Notebook:",
                options=list(notebook_options.keys()),
                help="Choose the specific notebook to execute"
            )
            
            if selected_title:
                selected_notebook = notebook_options[selected_title]
                
                # Display notebook info
                nb_info = self.executor.available_notebooks[selected_notebook]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Cells", nb_info['cell_count'])
                
                with col2:
                    st.metric("Category", nb_info['category'])
                
                with col3:
                    last_modified = pd.Timestamp.fromtimestamp(nb_info['last_modified'])
                    st.metric("Last Modified", last_modified.strftime('%Y-%m-%d'))
                
                st.write(f"**Description:** {nb_info['description']}")
                
                return selected_notebook
        
        return None
    
    def execute_and_display_notebook(self, notebook_name: str):
        """Execute notebook and display results"""
        
        if st.button(f"🚀 Execute {self.executor.available_notebooks[notebook_name]['title']}", type="primary"):
            
            with st.spinner("Executing notebook... This may take several minutes"):
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Update progress
                progress_bar.progress(25)
                status_text.text("Reading notebook...")
                
                # Execute notebook
                result = self.executor.execute_notebook(notebook_name, timeout=600)
                
                progress_bar.progress(100)
                status_text.text("Execution complete!")
                
                # Display results
                self.display_execution_results(result)
    
    def display_execution_results(self, result: Dict[str, Any]):
        """Display notebook execution results"""
        
        if result['status'] == 'error':
            st.error(f"❌ Notebook execution failed: {result['error']}")
            
            with st.expander("Error Details"):
                st.code(result.get('traceback', 'No traceback available'))
            
            return
        
        # Success message
        st.success(f"✅ Notebook '{result['title']}' executed successfully!")
        
        # Display execution info
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Category:** {result['category']}")
        
        with col2:
            st.info(f"**Executed:** {result['execution_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Display results
        results = result['results']
        
        # Text outputs
        if results['outputs']:
            st.subheader("📄 Analysis Output")
            
            for i, output in enumerate(results['outputs'][:5]):  # Limit to first 5 outputs
                with st.expander(f"Output {i+1} (Cell {output['cell_index']})"):
                    if output['type'] == 'text':
                        st.code(output['content'])
                    else:
                        st.text(output['content'])
        
        # DataFrames
        if results['dataframes']:
            st.subheader("📊 Data Tables")
            
            for i, df_output in enumerate(results['dataframes'][:3]):  # Limit to first 3 tables
                with st.expander(f"Table {i+1} (Cell {df_output['cell_index']})"):
                    st.components.v1.html(df_output['html'], height=400, scrolling=True)
        
        # Plots (would need additional processing for display)
        if results['plots']:
            st.subheader("📈 Visualizations")
            st.info(f"Generated {len(results['plots'])} visualizations")
        
        # Download results
        self.render_download_options(result)
    
    def render_download_options(self, result: Dict[str, Any]):
        """Render download options for results"""
        
        st.subheader("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create summary report
            report_content = self.create_text_report(result)
            
            st.download_button(
                label="📋 Download Analysis Report",
                data=report_content,
                file_name=f"{result['notebook_name']}_report.txt",
                mime="text/plain"
            )
        
        with col2:
            # Create JSON results
            import json
            
            json_results = {
                'notebook': result['notebook_name'],
                'execution_time': result['execution_time'].isoformat(),
                'status': result['status'],
                'category': result['category'],
                'outputs_count': len(result['results']['outputs']),
                'dataframes_count': len(result['results']['dataframes']),
                'plots_count': len(result['results']['plots'])
            }
            
            st.download_button(
                label="📊 Download Results (JSON)",
                data=json.dumps(json_results, indent=2),
                file_name=f"{result['notebook_name']}_results.json",
                mime="application/json"
            )
    
    def create_text_report(self, result: Dict[str, Any]) -> str:
        """Create text report from execution results"""
        
        report = []
        report.append(f"NOTEBOOK ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Notebook: {result['title']}")
        report.append(f"Category: {result['category']}")
        report.append(f"Executed: {result['execution_time']}")
        report.append(f"Status: {result['status']}")
        report.append("")
        
        results = result['results']
        
        report.append("SUMMARY")
        report.append("-" * 20)
        report.append(f"Text Outputs: {len(results['outputs'])}")
        report.append(f"Data Tables: {len(results['dataframes'])}")
        report.append(f"Visualizations: {len(results['plots'])}")
        report.append("")
        
        if results['outputs']:
            report.append("KEY OUTPUTS")
            report.append("-" * 20)
            
            for i, output in enumerate(results['outputs'][:3]):
                report.append(f"Output {i+1}:")
                content = output['content'][:500] + "..." if len(output['content']) > 500 else output['content']
                report.append(content)
                report.append("")
        
        return "\n".join(report)
    
    def render_notebook_dashboard(self):
        """Render comprehensive notebook dashboard"""
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
            <h1><i class="fas fa-book"></i> Advanced Notebook Analytics</h1>
            <p style="font-size: 1.2rem; margin: 0; opacity: 0.9;">
                Execute comprehensive economic analysis notebooks
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Summary statistics
        summary = self.executor.get_notebook_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Notebooks", summary['total_notebooks'])
        
        with col2:
            st.metric("Categories", len(summary['categories']))
        
        with col3:
            st.metric("Executed", summary['execution_stats']['total_executed'])
        
        with col4:
            success_rate = (summary['execution_stats']['successful'] / 
                          max(summary['execution_stats']['total_executed'], 1) * 100)
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Recent executions
        if summary['recently_executed']:
            st.subheader("🕒 Recent Executions")
            
            for execution in summary['recently_executed']:
                status_icon = "✅" if execution['status'] == 'success' else "❌"
                st.write(f"{status_icon} **{execution['title']}** - {execution['time'].strftime('%Y-%m-%d %H:%M')}")


# Global notebook interface
notebook_executor = NotebookExecutor()
notebook_interface = StreamlitNotebookInterface(notebook_executor)
