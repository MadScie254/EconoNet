"""
Ultra-Advanced Machine Learning Ensemble for Economic Prediction
================================================================

State-of-the-art ensemble methods combining classical ML, deep learning,
and quantum-inspired algorithms for economic forecasting
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                             ExtraTreesRegressor, VotingRegressor, BaggingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import (Ridge, Lasso, ElasticNet, BayesianRidge, 
                                 ARDRegression, HuberRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class QuantumInspiredEnsemble:
    """
    Quantum-inspired ensemble learning for economic prediction
    """
    
    def __init__(self, n_quantum_states=50):
        self.n_quantum_states = n_quantum_states
        self.quantum_models = []
        self.ensemble_weights = None
        self.feature_importance_quantum = None
        
    def create_quantum_features(self, X):
        """
        Create quantum-inspired features using superposition and entanglement
        """
        quantum_features = []
        
        # Original features
        quantum_features.append(X)
        
        # Quantum superposition features (linear combinations)
        n_features = X.shape[1]
        for _ in range(self.n_quantum_states):
            # Random quantum coefficients
            coeffs = np.random.normal(0, 1, n_features)
            coeffs = coeffs / np.linalg.norm(coeffs)  # Normalize
            
            # Superposition feature
            superposition = np.dot(X, coeffs).reshape(-1, 1)
            quantum_features.append(superposition)
        
        # Quantum entanglement features (pairwise interactions)
        for i in range(min(n_features, 10)):  # Limit for computational efficiency
            for j in range(i+1, min(n_features, 10)):
                entangled = (X[:, i] * X[:, j]).reshape(-1, 1)
                quantum_features.append(entangled)
        
        # Quantum interference patterns
        for i in range(min(n_features, 5)):
            interference = np.sin(X[:, i] * np.pi).reshape(-1, 1)
            quantum_features.append(interference)
            
            coherence = np.cos(X[:, i] * np.pi).reshape(-1, 1)
            quantum_features.append(coherence)
        
        return np.hstack(quantum_features)
    
    def quantum_ensemble_weights(self, predictions_matrix, y_true):
        """
        Calculate quantum ensemble weights using optimization
        """
        from scipy.optimize import minimize
        
        n_models = predictions_matrix.shape[1]
        
        def quantum_loss(weights):
            # Ensure weights sum to 1 (quantum normalization)
            weights = np.abs(weights) / np.sum(np.abs(weights))
            
            # Ensemble prediction
            ensemble_pred = np.dot(predictions_matrix, weights)
            
            # Quantum loss function
            mse = mean_squared_error(y_true, ensemble_pred)
            
            # Add quantum regularization (entropy-like term)
            entropy_reg = -np.sum(weights * np.log(weights + 1e-10))
            
            return mse - 0.01 * entropy_reg  # Encourage diversity
        
        # Initial weights (equal)
        initial_weights = np.ones(n_models) / n_models
        
        # Optimization constraints
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Optimize
        result = minimize(quantum_loss, initial_weights, 
                         constraints=constraints, bounds=bounds)
        
        return np.abs(result.x) / np.sum(np.abs(result.x))

class UltraAdvancedEnsemble:
    """
    Ultra-advanced ensemble system combining multiple ML paradigms
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.meta_learner = None
        self.feature_selectors = {}
        self.quantum_ensemble = QuantumInspiredEnsemble()
        
    def initialize_models(self):
        """
        Initialize all base models and meta-learners
        """
        # Tree-based models
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        
        self.models['gradient_boost'] = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=8,
            subsample=0.8, random_state=42
        )
        
        self.models['extra_trees'] = ExtraTreesRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            random_state=42, n_jobs=-1
        )
        
        # Neural networks with different architectures
        self.models['neural_deep'] = MLPRegressor(
            hidden_layer_sizes=(200, 100, 50), activation='relu',
            solver='adam', alpha=0.001, learning_rate='adaptive',
            max_iter=1000, random_state=42
        )
        
        self.models['neural_wide'] = MLPRegressor(
            hidden_layer_sizes=(500, 200), activation='tanh',
            solver='adam', alpha=0.01, learning_rate='adaptive',
            max_iter=1000, random_state=42
        )
        
        # Support Vector Machines
        self.models['svr_rbf'] = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.01)
        self.models['svr_poly'] = SVR(kernel='poly', degree=3, C=100, epsilon=0.01)
        
        # Linear models with regularization
        self.models['ridge'] = Ridge(alpha=1.0)
        self.models['lasso'] = Lasso(alpha=0.1)
        self.models['elastic_net'] = ElasticNet(alpha=0.1, l1_ratio=0.5)
        self.models['bayesian_ridge'] = BayesianRidge()
        self.models['ard'] = ARDRegression()
        self.models['huber'] = HuberRegressor(epsilon=1.35)
        
        # Nearest neighbors
        self.models['knn'] = KNeighborsRegressor(n_neighbors=10, weights='distance')
        
        # Gaussian Process
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        self.models['gaussian_process'] = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, random_state=42
        )
        
        # Advanced ensemble methods
        self.models['voting_ensemble'] = VotingRegressor([
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)),
            ('nn', MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42))
        ])
        
        self.models['bagging_ensemble'] = BaggingRegressor(
            base_estimator=DecisionTreeRegressor(max_depth=10),
            n_estimators=100, random_state=42, n_jobs=-1
        )
        
        # Initialize scalers
        self.scalers['standard'] = StandardScaler()
        self.scalers['robust'] = RobustScaler()
        self.scalers['minmax'] = MinMaxScaler()
        
        print(f"Initialized {len(self.models)} models and {len(self.scalers)} scalers")
    
    def advanced_feature_engineering(self, X, y=None):
        """
        Advanced feature engineering with domain knowledge
        """
        features = []
        feature_names = []
        
        # Original features
        features.append(X)
        feature_names.extend([f'original_{i}' for i in range(X.shape[1])])
        
        # Polynomial features (degree 2)
        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                poly_feature = (X[:, i] * X[:, j]).reshape(-1, 1)
                features.append(poly_feature)
                feature_names.append(f'poly_{i}_{j}')
        
        # Logarithmic transformations
        X_positive = np.abs(X) + 1e-8  # Ensure positive values
        log_features = np.log(X_positive)
        features.append(log_features)
        feature_names.extend([f'log_{i}' for i in range(X.shape[1])])
        
        # Square root transformations
        sqrt_features = np.sqrt(X_positive)
        features.append(sqrt_features)
        feature_names.extend([f'sqrt_{i}' for i in range(X.shape[1])])
        
        # Exponential decay features
        exp_features = np.exp(-np.abs(X))
        features.append(exp_features)
        feature_names.extend([f'exp_{i}' for i in range(X.shape[1])])
        
        # Rolling statistics (if time series)
        if len(X) > 10:
            rolling_mean = np.array([np.mean(X[max(0, i-5):i+1], axis=0) 
                                   for i in range(len(X))])
            rolling_std = np.array([np.std(X[max(0, i-5):i+1], axis=0) 
                                  for i in range(len(X))])
            features.extend([rolling_mean, rolling_std])
            feature_names.extend([f'rolling_mean_{i}' for i in range(X.shape[1])])
            feature_names.extend([f'rolling_std_{i}' for i in range(X.shape[1])])
        
        # Fourier features for cyclical patterns
        for freq in [1, 2, 3]:  # Different frequencies
            for i in range(min(X.shape[1], 3)):  # Limit for efficiency
                sin_feat = np.sin(2 * np.pi * freq * X[:, i] / X.shape[0])
                cos_feat = np.cos(2 * np.pi * freq * X[:, i] / X.shape[0])
                features.extend([sin_feat.reshape(-1, 1), cos_feat.reshape(-1, 1)])
                feature_names.extend([f'sin_{freq}_{i}', f'cos_{freq}_{i}'])
        
        # Combine all features
        engineered_features = np.hstack(features)
        
        return engineered_features, feature_names
    
    def train_ensemble(self, X, y, use_quantum=True, use_meta_learning=True):
        """
        Train the ultra-advanced ensemble
        """
        print("Starting ultra-advanced ensemble training...")
        
        # Feature engineering
        X_engineered, feature_names = self.advanced_feature_engineering(X)
        print(f"Engineered {X_engineered.shape[1]} features from {X.shape[1]} original features")
        
        # Add quantum features if enabled
        if use_quantum:
            X_quantum = self.quantum_ensemble.create_quantum_features(X_engineered)
            print(f"Added quantum features: {X_quantum.shape[1]} total features")
        else:
            X_quantum = X_engineered
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train all base models
        model_predictions = {}
        model_scores = {}
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            # Select appropriate scaler
            if 'neural' in model_name or 'svr' in model_name or 'gaussian' in model_name:
                scaler_name = 'standard'
            elif 'tree' in model_name or 'forest' in model_name or 'boost' in model_name:
                scaler_name = None  # Tree models don't need scaling
            else:
                scaler_name = 'robust'
            
            try:
                if scaler_name:
                    # Scale features
                    X_scaled = self.scalers[scaler_name].fit_transform(X_quantum)
                else:
                    X_scaled = X_quantum
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, 
                                          scoring='neg_mean_squared_error', n_jobs=-1)
                model_scores[model_name] = -cv_scores.mean()
                
                # Train on full data
                model.fit(X_scaled, y)
                
                # Store predictions for meta-learning
                predictions = model.predict(X_scaled)
                model_predictions[model_name] = predictions
                
                print(f"  {model_name}: CV RMSE = {np.sqrt(model_scores[model_name]):.4f}")
                
            except Exception as e:
                print(f"  Error training {model_name}: {e}")
                continue
        
        # Meta-learning (stacking)
        if use_meta_learning and len(model_predictions) > 1:
            print("Training meta-learner...")
            
            # Create meta-features matrix
            meta_X = np.column_stack(list(model_predictions.values()))
            
            # Meta-learner (simple ridge regression)
            from sklearn.linear_model import Ridge
            self.meta_learner = Ridge(alpha=1.0)
            self.meta_learner.fit(meta_X, y)
            
            # Meta-learner prediction
            meta_pred = self.meta_learner.predict(meta_X)
            meta_score = mean_squared_error(y, meta_pred)
            print(f"Meta-learner RMSE: {np.sqrt(meta_score):.4f}")
        
        # Quantum ensemble weights
        if use_quantum and len(model_predictions) > 1:
            print("Computing quantum ensemble weights...")
            pred_matrix = np.column_stack(list(model_predictions.values()))
            quantum_weights = self.quantum_ensemble.quantum_ensemble_weights(pred_matrix, y)
            self.quantum_ensemble.ensemble_weights = quantum_weights
            print(f"Quantum weights: {dict(zip(model_predictions.keys(), quantum_weights))}")
        
        # Store training data characteristics
        self.training_stats = {
            'n_samples': len(X),
            'n_features_original': X.shape[1],
            'n_features_engineered': X_engineered.shape[1],
            'n_features_quantum': X_quantum.shape[1] if use_quantum else X_engineered.shape[1],
            'model_scores': model_scores,
            'feature_names': feature_names
        }
        
        print("Ensemble training completed!")
        return model_scores
    
    def predict_ensemble(self, X, use_quantum=True, use_meta_learning=True):
        """
        Make predictions using the ensemble
        """
        # Feature engineering
        X_engineered, _ = self.advanced_feature_engineering(X)
        
        # Add quantum features if enabled
        if use_quantum:
            X_quantum = self.quantum_ensemble.create_quantum_features(X_engineered)
        else:
            X_quantum = X_engineered
        
        # Get predictions from all models
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                # Select appropriate scaler
                if 'neural' in model_name or 'svr' in model_name or 'gaussian' in model_name:
                    scaler_name = 'standard'
                elif 'tree' in model_name or 'forest' in model_name or 'boost' in model_name:
                    scaler_name = None
                else:
                    scaler_name = 'robust'
                
                if scaler_name:
                    X_scaled = self.scalers[scaler_name].transform(X_quantum)
                else:
                    X_scaled = X_quantum
                
                pred = model.predict(X_scaled)
                predictions[model_name] = pred
                
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No models could make predictions")
        
        # Meta-learning prediction
        if use_meta_learning and self.meta_learner is not None:
            meta_X = np.column_stack(list(predictions.values()))
            meta_prediction = self.meta_learner.predict(meta_X)
        else:
            meta_prediction = None
        
        # Quantum ensemble prediction
        if use_quantum and self.quantum_ensemble.ensemble_weights is not None:
            pred_matrix = np.column_stack(list(predictions.values()))
            quantum_prediction = np.dot(pred_matrix, self.quantum_ensemble.ensemble_weights)
        else:
            # Simple average
            quantum_prediction = np.mean(list(predictions.values()), axis=0)
        
        return {
            'individual_predictions': predictions,
            'meta_prediction': meta_prediction,
            'quantum_prediction': quantum_prediction,
            'ensemble_prediction': quantum_prediction if use_quantum else meta_prediction
        }
    
    def feature_importance_analysis(self):
        """
        Analyze feature importance across all models
        """
        importance_dict = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[model_name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance_dict[model_name] = np.abs(model.coef_)
        
        if importance_dict:
            # Average importance across models
            avg_importance = np.mean(list(importance_dict.values()), axis=0)
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature_index': range(len(avg_importance)),
                'importance': avg_importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return None
    
    def model_performance_analysis(self, X_test, y_test):
        """
        Comprehensive performance analysis
        """
        predictions = self.predict_ensemble(X_test)
        
        performance = {}
        
        # Individual model performance
        for model_name, pred in predictions['individual_predictions'].items():
            mse = mean_squared_error(y_test, pred)
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            
            performance[model_name] = {
                'rmse': np.sqrt(mse),
                'mae': mae,
                'r2': r2
            }
        
        # Ensemble performance
        if predictions['ensemble_prediction'] is not None:
            ensemble_pred = predictions['ensemble_prediction']
            mse = mean_squared_error(y_test, ensemble_pred)
            mae = mean_absolute_error(y_test, ensemble_pred)
            r2 = r2_score(y_test, ensemble_pred)
            
            performance['ensemble'] = {
                'rmse': np.sqrt(mse),
                'mae': mae,
                'r2': r2
            }
        
        return performance

def create_synthetic_economic_dataset(n_samples=1000, n_features=10, noise_level=0.1):
    """
    Create synthetic economic dataset for testing
    """
    np.random.seed(42)
    
    # Generate correlated features
    correlation_matrix = np.random.rand(n_features, n_features)
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1)
    
    # Generate features
    X = np.random.multivariate_normal(np.zeros(n_features), correlation_matrix, n_samples)
    
    # Create complex target variable
    # Linear combination
    linear_coeffs = np.random.normal(0, 1, n_features)
    y_linear = np.dot(X, linear_coeffs)
    
    # Non-linear components
    y_nonlinear = np.sum(X[:, :3]**2, axis=1) - np.sum(X[:, 3:6]**3, axis=1)
    
    # Interaction terms
    y_interaction = X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3]
    
    # Combine components
    y = y_linear + 0.5 * y_nonlinear + 0.3 * y_interaction
    
    # Add noise
    y += np.random.normal(0, noise_level * np.std(y), n_samples)
    
    return X, y

if __name__ == "__main__":
    print("Ultra-Advanced Machine Learning Ensemble System")
    print("=" * 50)
    
    # Create synthetic dataset
    X, y = create_synthetic_economic_dataset(n_samples=500, n_features=8)
    print(f"Created dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Initialize ensemble
    ensemble = UltraAdvancedEnsemble()
    ensemble.initialize_models()
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train ensemble
    scores = ensemble.train_ensemble(X_train, y_train, use_quantum=True, use_meta_learning=True)
    
    # Test predictions
    predictions = ensemble.predict_ensemble(X_test, use_quantum=True, use_meta_learning=True)
    
    # Performance analysis
    performance = ensemble.model_performance_analysis(X_test, y_test)
    
    print("\nPerformance Summary:")
    for model_name, metrics in performance.items():
        print(f"{model_name}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
    
    print("\nUltra-Advanced Ensemble System Ready!")
