"""
NERVA Notebook Integration System
Professional notebook management and execution
"""

import os
import subprocess
import json
from pathlib import Path
import streamlit as st

class NotebookManager:
    def __init__(self, notebooks_dir="notebooks"):
        self.notebooks_dir = Path(notebooks_dir)
        self.notebooks_dir.mkdir(exist_ok=True)
        
    def get_available_notebooks(self):
        """Get list of available notebooks"""
        notebooks = []
        for nb_file in self.notebooks_dir.glob("*.ipynb"):
            try:
                with open(nb_file, 'r', encoding='utf-8') as f:
                    nb_content = json.load(f)
                    
                # Extract metadata
                title = nb_file.stem.replace('_', ' ').title()
                
                # Get first markdown cell for description
                description = "Advanced analytics notebook"
                for cell in nb_content.get('cells', []):
                    if cell.get('cell_type') == 'markdown':
                        source = ''.join(cell.get('source', []))
                        if len(source) > 50:
                            description = source[:200] + "..."
                            break
                
                notebooks.append({
                    'title': title,
                    'filename': nb_file.name,
                    'path': str(nb_file),
                    'description': description,
                    'size': nb_file.stat().st_size,
                    'modified': nb_file.stat().st_mtime
                })
                
            except Exception as e:
                print(f"Error reading notebook {nb_file}: {e}")
                
        return sorted(notebooks, key=lambda x: x['modified'], reverse=True)
    
    def launch_jupyter_lab(self, port=8888):
        """Launch Jupyter Lab server"""
        try:
            cmd = [
                "python", "-m", "jupyter", "lab",
                f"--port={port}",
                "--no-browser",
                f"--notebook-dir={self.notebooks_dir}",
                "--allow-root"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            return process, f"http://localhost:{port}"
            
        except Exception as e:
            print(f"Error launching Jupyter Lab: {e}")
            return None, None
    
    def create_notebook_link(self, notebook_path, jupyter_url):
        """Create direct link to notebook"""
        notebook_name = Path(notebook_path).name
        return f"{jupyter_url}/lab/tree/{notebook_name}"

# Notebook metadata
NOTEBOOK_CATALOG = {
    "Advanced_Inflation_Modeling.ipynb": {
        "title": "Advanced Inflation Modeling Engine",
        "description": "Quantum-inspired inflation forecasting with neural networks, regime detection, and stress testing capabilities",
        "category": "Forecasting",
        "complexity": "Advanced",
        "features": [
            "Neural Network Forecasting",
            "Hidden Markov Regime Detection", 
            "Stress Testing Scenarios",
            "Real-time Visualization",
            "Confidence Intervals"
        ]
    },
    "Quantum_FX_Dynamics.ipynb": {
        "title": "Quantum FX Dynamics Engine", 
        "description": "Advanced foreign exchange modeling with volatility clustering and intervention point detection",
        "category": "FX Markets",
        "complexity": "Expert",
        "features": [
            "Currency Pair Modeling",
            "Volatility Clustering",
            "Cross-Currency Correlations",
            "Intervention Detection",
            "Market Microstructure"
        ]
    },
    "Market_Dynamics_Intelligence.ipynb": {
        "title": "Market Dynamics Intelligence",
        "description": "High-frequency market analysis, liquidity flow mapping, and systematic risk detection",
        "category": "Market Analysis", 
        "complexity": "Expert",
        "features": [
            "High-Frequency Analysis",
            "Liquidity Flow Mapping",
            "Market Maker Behavior",
            "Risk Detection",
            "Sentiment Analysis"
        ]
    },
    "Policy_Simulation_Engine.ipynb": {
        "title": "Policy Simulation Engine",
        "description": "Central bank policy transmission mechanisms and quantitative easing impact assessment",
        "category": "Policy Analysis",
        "complexity": "Advanced", 
        "features": [
            "Policy Transmission",
            "Interest Rate Optimization",
            "QE Impact Assessment",
            "Stability Testing",
            "Multi-Scenario Modeling"
        ]
    }
}

def get_notebook_info(filename):
    """Get notebook information from catalog"""
    return NOTEBOOK_CATALOG.get(filename, {
        "title": filename.replace('.ipynb', '').replace('_', ' ').title(),
        "description": "Professional analytics notebook",
        "category": "Analysis",
        "complexity": "Intermediate",
        "features": ["Data Analysis", "Visualization", "Modeling"]
    })
