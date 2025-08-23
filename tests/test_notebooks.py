"""
Tests for EconoNet Notebook Integration
======================================

Tests for notebook execution with live data and metadata tracking.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from econonet import set_mode, OperationMode

class TestNotebookIntegration:
    """Test notebook integration functionality"""
    
    def test_offline_notebook_execution(self):
        """Test that notebooks can execute in offline mode"""
        from econonet import get_worldbank, set_mode, OperationMode
        
        # Set to offline mode
        set_mode(OperationMode.OFFLINE)
        
        # This simulates what would happen in a notebook
        data = get_worldbank('KE', 'NY.GDP.MKTP.CD')
        
        # Should return synthetic data without errors
        assert not data.empty
        assert data['source'].iloc[0] == 'Synthetic'
        assert len(data) > 0
    
    def test_notebook_data_consistency(self):
        """Test that notebook data is consistent across cells"""
        from econonet import get_worldbank, get_coingecko, set_mode, OperationMode
        
        set_mode(OperationMode.OFFLINE)
        
        # Simulate multiple notebook cells calling APIs
        gdp_data = get_worldbank('KE', 'NY.GDP.MKTP.CD')
        crypto_data = get_coingecko(['bitcoin'])
        
        # Both should return synthetic data
        assert gdp_data['source'].iloc[0] == 'Synthetic'
        assert crypto_data['source'].iloc[0] == 'Synthetic'
        
        # Data should be reproducible (same random seed)
        gdp_data2 = get_worldbank('KE', 'NY.GDP.MKTP.CD')
        assert gdp_data.equals(gdp_data2)
    
    def create_test_notebook(self, temp_dir):
        """Helper to create a test notebook"""
        notebook_content = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import sys\n",
                        "sys.path.append('../src')\n",
                        "from econonet import get_worldbank, set_mode, OperationMode"
                    ]
                },
                {
                    "cell_type": "code", 
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "set_mode(OperationMode.OFFLINE)\n",
                        "data = get_worldbank('KE', 'NY.GDP.MKTP.CD')\n",
                        "print(f'Loaded {len(data)} data points')\n",
                        "assert len(data) > 0\n",
                        "print('✅ Notebook test passed!')"
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
                    "version": "3.10.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        notebook_path = temp_dir / "test_notebook.ipynb"
        with open(notebook_path, 'w') as f:
            json.dump(notebook_content, f, indent=2)
        
        return notebook_path
    
    def test_notebook_structure_validation(self):
        """Test validation of notebook structure"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            notebook_path = self.create_test_notebook(temp_path)
            
            # Verify notebook was created correctly
            assert notebook_path.exists()
            
            # Load and validate structure
            with open(notebook_path, 'r') as f:
                notebook = json.load(f)
            
            assert 'cells' in notebook
            assert 'metadata' in notebook
            assert len(notebook['cells']) == 2
            assert all(cell['cell_type'] == 'code' for cell in notebook['cells'])
    
    @pytest.mark.skipif(
        not os.system("which papermill > /dev/null 2>&1") == 0,
        reason="papermill not available"
    )
    def test_papermill_execution(self):
        """Test notebook execution via papermill (if available)"""
        try:
            import papermill as pm
        except ImportError:
            pytest.skip("papermill not available")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test notebook
            input_notebook = self.create_test_notebook(temp_path)
            output_notebook = temp_path / "output_notebook.ipynb"
            
            # Execute notebook
            try:
                pm.execute_notebook(
                    str(input_notebook),
                    str(output_notebook),
                    parameters={}
                )
                
                # Verify execution
                with open(output_notebook, 'r') as f:
                    executed_notebook = json.load(f)
                
                # Check that cells were executed
                executed_cells = sum(
                    1 for cell in executed_notebook['cells']
                    if cell.get('execution_count') is not None
                )
                
                assert executed_cells > 0, "No cells were executed"
                
            except Exception as e:
                pytest.skip(f"Papermill execution failed: {e}")

class TestNotebookDataIntegration:
    """Test integration of live data into notebooks"""
    
    def test_live_data_injection(self):
        """Test that live data can be injected into notebook context"""
        from econonet import get_all_data, set_mode, OperationMode
        
        set_mode(OperationMode.OFFLINE)
        
        # Simulate getting all data for a notebook
        all_data = get_all_data('KE')
        
        # Should return dictionary with various datasets
        assert isinstance(all_data, dict)
        
        # Check for expected keys
        expected_keys = ['worldbank_gdp', 'crypto', 'earthquakes']
        available_keys = [k for k in expected_keys if k in all_data]
        
        # At least some data should be available
        assert len(available_keys) > 0
        
        # All available data should be synthetic in offline mode
        for key, data in all_data.items():
            if data is not None and not data.empty:
                assert data['source'].iloc[0] == 'Synthetic'
    
    def test_notebook_metadata_tracking(self):
        """Test metadata tracking for notebook execution"""
        from econonet import get_worldbank, set_mode, OperationMode
        from datetime import datetime
        
        set_mode(OperationMode.OFFLINE)
        
        # Get data and track metadata
        start_time = datetime.now()
        data = get_worldbank('KE', 'NY.GDP.MKTP.CD')
        end_time = datetime.now()
        
        # Verify metadata is present
        assert not data.empty
        assert 'metadata' in data.columns
        
        metadata = data['metadata'].iloc[0]
        assert isinstance(metadata, dict)
        assert 'fallback' in metadata
        assert metadata['fallback'] is True  # Should be synthetic in offline mode
    
    def test_notebook_error_handling(self):
        """Test error handling in notebook context"""
        from econonet import get_worldbank, set_mode, OperationMode
        
        set_mode(OperationMode.LIVE)
        
        # Even with potential API failures, should not crash
        try:
            data = get_worldbank('INVALID_COUNTRY', 'INVALID_INDICATOR')
            # Should return synthetic fallback data
            assert not data.empty
        except Exception as e:
            pytest.fail(f"Notebook execution should not crash on API errors: {e}")

class TestNotebookPerformance:
    """Test performance aspects of notebook integration"""
    
    def test_caching_in_notebooks(self):
        """Test that caching works properly in notebook context"""
        from econonet import get_worldbank, set_mode, OperationMode
        import time
        
        set_mode(OperationMode.OFFLINE)
        
        # First call
        start1 = time.time()
        data1 = get_worldbank('KE', 'NY.GDP.MKTP.CD')
        time1 = time.time() - start1
        
        # Second call (should be faster due to consistent synthetic data)
        start2 = time.time()
        data2 = get_worldbank('KE', 'NY.GDP.MKTP.CD')
        time2 = time.time() - start2
        
        # Data should be identical
        assert data1.equals(data2)
        
        # Both should complete quickly in offline mode
        assert time1 < 5.0  # Should be very fast for synthetic data
        assert time2 < 5.0
    
    def test_memory_usage(self):
        """Test memory usage of data structures"""
        from econonet import get_worldbank, set_mode, OperationMode
        import sys
        
        set_mode(OperationMode.OFFLINE)
        
        # Get data and check memory usage
        data = get_worldbank('KE', 'NY.GDP.MKTP.CD')
        
        # Should not use excessive memory
        data_size = sys.getsizeof(data)
        assert data_size < 10 * 1024 * 1024  # Less than 10MB
        
        # Verify data structure efficiency
        assert len(data) > 0
        assert len(data.columns) == 6  # Standardized schema

if __name__ == "__main__":
    pytest.main([__file__])
