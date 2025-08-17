"""
NERVA Model Registry and Ensemble Management
GODMODE_X: Automated model orchestration
"""

import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Model metadata for registry"""
    model_id: str
    model_type: str
    created_at: datetime
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    feature_names: List[str]
    target_variable: str
    training_data_info: Dict[str, Any]
    model_version: str = "1.0"
    is_active: bool = True

class ModelRegistry:
    """
    Centralized model registry with versioning and metadata management
    """
    
    def __init__(self, registry_path: Path = Path("models_registry")):
        self.registry_path = registry_path
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_path / "model_metadata.json"
        self.models_dir = self.registry_path / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Load existing metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Dict]:
        """Load model metadata from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    raw_metadata = json.load(f)
                
                # Convert to ModelMetadata objects
                metadata = {}
                for model_id, meta_dict in raw_metadata.items():
                    meta_dict['created_at'] = datetime.fromisoformat(meta_dict['created_at'])
                    metadata[model_id] = ModelMetadata(**meta_dict)
                
                return metadata
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        
        return {}
    
    def _save_metadata(self):
        """Save model metadata to disk"""
        try:
            # Convert ModelMetadata to dict for JSON serialization
            serializable_metadata = {}
            for model_id, metadata in self.metadata.items():
                meta_dict = asdict(metadata)
                meta_dict['created_at'] = metadata.created_at.isoformat()
                serializable_metadata[model_id] = meta_dict
            
            with open(self.metadata_file, 'w') as f:
                json.dump(serializable_metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def register_model(self, 
                      model: Any,
                      model_id: str,
                      model_type: str,
                      performance_metrics: Dict[str, float],
                      hyperparameters: Dict[str, Any],
                      feature_names: List[str],
                      target_variable: str,
                      training_data_info: Dict[str, Any]) -> str:
        """Register a new model in the registry"""
        
        # Create unique model ID if collision
        original_id = model_id
        counter = 1
        while model_id in self.metadata:
            model_id = f"{original_id}_v{counter}"
            counter += 1
        
        # Save model
        model_path = self.models_dir / f"{model_id}.pkl"
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {e}")
            return None
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_type=model_type,
            created_at=datetime.now(),
            performance_metrics=performance_metrics,
            hyperparameters=hyperparameters,
            feature_names=feature_names,
            target_variable=target_variable,
            training_data_info=training_data_info
        )
        
        # Register metadata
        self.metadata[model_id] = metadata
        self._save_metadata()
        
        logger.info(f"✅ Registered model: {model_id}")
        return model_id
    
    def load_model(self, model_id: str) -> Optional[Any]:
        """Load a model from the registry"""
        if model_id not in self.metadata:
            logger.warning(f"Model {model_id} not found in registry")
            return None
        
        model_path = self.models_dir / f"{model_id}.pkl"
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"✅ Loaded model: {model_id}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return None
    
    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get metadata for a specific model"""
        return self.metadata.get(model_id)
    
    def list_models(self, model_type: Optional[str] = None, active_only: bool = True) -> List[ModelMetadata]:
        """List all models with optional filtering"""
        models = list(self.metadata.values())
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if active_only:
            models = [m for m in models if m.is_active]
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.created_at, reverse=True)
        
        return models
    
    def get_best_model(self, model_type: str, metric: str = 'mse') -> Optional[str]:
        """Get the best performing model of a given type"""
        models = self.list_models(model_type=model_type)
        
        if not models:
            return None
        
        # Find model with best performance on specified metric
        best_model = None
        best_score = float('inf') if 'mse' in metric.lower() or 'error' in metric.lower() else float('-inf')
        
        for model in models:
            if metric in model.performance_metrics:
                score = model.performance_metrics[metric]
                
                # Lower is better for error metrics
                if 'mse' in metric.lower() or 'error' in metric.lower() or 'loss' in metric.lower():
                    if score < best_score:
                        best_score = score
                        best_model = model.model_id
                else:
                    # Higher is better for accuracy, R2, etc.
                    if score > best_score:
                        best_score = score
                        best_model = model.model_id
        
        return best_model
    
    def deactivate_model(self, model_id: str):
        """Deactivate a model (mark as inactive)"""
        if model_id in self.metadata:
            self.metadata[model_id].is_active = False
            self._save_metadata()
            logger.info(f"Deactivated model: {model_id}")
    
    def delete_model(self, model_id: str):
        """Completely remove a model from registry"""
        if model_id in self.metadata:
            # Delete model file
            model_path = self.models_dir / f"{model_id}.pkl"
            if model_path.exists():
                model_path.unlink()
            
            # Remove from metadata
            del self.metadata[model_id]
            self._save_metadata()
            
            logger.info(f"Deleted model: {model_id}")

class EnsembleManager:
    """
    Manage ensemble models with dynamic weighting
    """
    
    def __init__(self, model_registry: ModelRegistry):
        self.registry = model_registry
        self.ensemble_weights = {}
    
    def create_ensemble(self, 
                       model_ids: List[str], 
                       weighting_method: str = 'performance',
                       performance_metric: str = 'mse') -> Dict[str, float]:
        """Create ensemble with weighted models"""
        
        weights = {}
        
        if weighting_method == 'equal':
            # Equal weighting
            weight = 1.0 / len(model_ids)
            weights = {model_id: weight for model_id in model_ids}
        
        elif weighting_method == 'performance':
            # Performance-based weighting
            performances = []
            valid_models = []
            
            for model_id in model_ids:
                metadata = self.registry.get_model_metadata(model_id)
                if metadata and performance_metric in metadata.performance_metrics:
                    perf = metadata.performance_metrics[performance_metric]
                    performances.append(perf)
                    valid_models.append(model_id)
            
            if performances:
                # For error metrics, invert so better models get higher weights
                if 'mse' in performance_metric.lower() or 'error' in performance_metric.lower():
                    # Convert to inverse weights (smaller error = higher weight)
                    inverse_perfs = [1.0 / (p + 1e-8) for p in performances]
                    total_weight = sum(inverse_perfs)
                    weights = {model_id: w/total_weight for model_id, w in zip(valid_models, inverse_perfs)}
                else:
                    # For accuracy metrics, use direct weights
                    total_perf = sum(performances)
                    weights = {model_id: p/total_perf for model_id, p in zip(valid_models, performances)}
        
        return weights
    
    def predict_ensemble(self, 
                        model_ids: List[str], 
                        X: pd.DataFrame,
                        weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Generate ensemble predictions"""
        
        if weights is None:
            weights = self.create_ensemble(model_ids)
        
        predictions = []
        model_weights = []
        
        for model_id in model_ids:
            if model_id in weights:
                model = self.registry.load_model(model_id)
                if model is not None:
                    try:
                        pred = model.predict(X)
                        predictions.append(pred)
                        model_weights.append(weights[model_id])
                    except Exception as e:
                        logger.warning(f"Prediction failed for model {model_id}: {e}")
        
        if predictions:
            # Weighted average of predictions
            predictions = np.array(predictions)
            model_weights = np.array(model_weights)
            
            # Normalize weights
            model_weights = model_weights / model_weights.sum()
            
            ensemble_pred = np.average(predictions, axis=0, weights=model_weights)
            return ensemble_pred
        
        return np.array([])
    
    def evaluate_ensemble(self, 
                         model_ids: List[str], 
                         X: pd.DataFrame, 
                         y: pd.Series,
                         weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Evaluate ensemble performance"""
        
        predictions = self.predict_ensemble(model_ids, X, weights)
        
        if len(predictions) == 0:
            return {}
        
        # Align predictions and targets
        min_len = min(len(predictions), len(y))
        predictions = predictions[:min_len]
        y_aligned = y.iloc[:min_len]
        
        # Calculate metrics
        mse = np.mean((predictions - y_aligned) ** 2)
        mae = np.mean(np.abs(predictions - y_aligned))
        
        # R-squared
        ss_res = np.sum((y_aligned - predictions) ** 2)
        ss_tot = np.sum((y_aligned - np.mean(y_aligned)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'r2': r2
        }

def create_model_registry() -> ModelRegistry:
    """Factory function to create model registry"""
    return ModelRegistry()

def create_ensemble_manager(registry: ModelRegistry) -> EnsembleManager:
    """Factory function to create ensemble manager"""
    return EnsembleManager(registry)
