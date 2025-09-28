#!/usr/bin/env python3
"""
🚀 OPTIMIZED MODEL TRAINING SYSTEM
Ultra-efficient training with memory optimization, GPU acceleration, and advanced techniques
"""

import numpy as np
import pandas as pd
import os
import gc
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
from datetime import datetime
import joblib
import json

# Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb

# Deep Learning (with error handling)
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
    
    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    
    # Set device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
except ImportError:
    TORCH_AVAILABLE = False

# Hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

class OptimizedModelTrainer:
    """
    🎯 Ultra-efficient model training system with advanced optimizations
    """
    
    def __init__(self, 
                 memory_limit_gb: float = 8.0,
                 use_gpu: bool = True,
                 n_jobs: int = -1,
                 random_state: int = 42):
        
        self.memory_limit_gb = memory_limit_gb
        self.use_gpu = use_gpu
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.training_metrics = {
            'model_performances': {},
            'training_times': {},
            'memory_usage': {},
            'optimization_history': []
        }
        
        # Create output directories
        self.models_dir = Path("models/optimized")
        self.results_dir = Path("results/optimized")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print("🚀 Optimized Model Training System Initialized")
        print(f"   💾 Memory limit: {memory_limit_gb} GB")
        print(f"   🔧 GPU enabled: {use_gpu and (TF_AVAILABLE or TORCH_AVAILABLE)}")
        print(f"   ⚡ CPU cores: {os.cpu_count()}")
        
    def check_memory_usage(self) -> Dict[str, float]:
        """Monitor memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        usage = {
            'process_memory_gb': memory_info.rss / 1024 / 1024 / 1024,
            'system_memory_percent': system_memory.percent,
            'available_memory_gb': system_memory.available / 1024 / 1024 / 1024
        }
        
        # Trigger garbage collection if memory usage is high
        if usage['process_memory_gb'] > self.memory_limit_gb * 0.8:
            self.logger.warning(f"High memory usage: {usage['process_memory_gb']:.1f}GB")
            gc.collect()
            
        return usage
    
    def optimize_sklearn_model(self, model_class, param_grid: Dict, 
                             X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Optimize scikit-learn model with efficient hyperparameter search"""
        
        from sklearn.model_selection import RandomizedSearchCV
        
        best_score = 0
        best_params = None
        best_model = None
        
        # Use RandomizedSearchCV for efficiency
        model = model_class(random_state=self.random_state, n_jobs=self.n_jobs)
        
        # Efficient parameter search
        random_search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=20,  # Limit iterations for efficiency
            cv=3,       # Reduced CV folds for speed
            scoring='accuracy',
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=0
        )
        
        # Fit with memory monitoring
        start_time = time.time()
        random_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Get best model
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        
        # Validate on holdout set
        val_predictions = best_model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)
        
        return {
            'model': best_model,
            'params': best_params,
            'val_accuracy': val_accuracy,
            'training_time': training_time,
            'cv_score': random_search.best_score_
        }
    
    def train_tree_based_models(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train optimized tree-based models"""
        
        print("🌳 Training optimized tree-based models...")
        
        models = {}
        
        # 1. Optimized Random Forest
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        rf_result = self.optimize_sklearn_model(
            RandomForestClassifier, rf_params, X_train, y_train, X_val, y_val
        )
        models['RandomForest_Optimized'] = rf_result
        
        # 2. Optimized Extra Trees
        et_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [15, 20, 25],
            'min_samples_split': [2, 3],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        et_result = self.optimize_sklearn_model(
            ExtraTreesClassifier, et_params, X_train, y_train, X_val, y_val
        )
        models['ExtraTrees_Optimized'] = et_result
        
        # 3. Memory-efficient XGBoost
        print("🚀 Training XGBoost with memory optimization...")
        
        xgb_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'tree_method': 'hist',  # Memory efficient
            'max_bin': 256         # Reduce memory usage
        }
        
        start_time = time.time()
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train, y_train)
        xgb_time = time.time() - start_time
        
        xgb_pred = xgb_model.predict(X_val)
        xgb_accuracy = accuracy_score(y_val, xgb_pred)
        
        models['XGBoost_Optimized'] = {
            'model': xgb_model,
            'params': xgb_params,
            'val_accuracy': xgb_accuracy,
            'training_time': xgb_time
        }
        
        # 4. Memory-efficient LightGBM
        print("💡 Training LightGBM with memory optimization...")
        
        lgb_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbosity': -1,
            'max_bin': 255        # Memory efficient
        }
        
        start_time = time.time()
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(X_train, y_train)
        lgb_time = time.time() - start_time
        
        lgb_pred = lgb_model.predict(X_val)
        lgb_accuracy = accuracy_score(y_val, lgb_pred)
        
        models['LightGBM_Optimized'] = {
            'model': lgb_model,
            'params': lgb_params,
            'val_accuracy': lgb_accuracy,
            'training_time': lgb_time
        }
        
        # Track memory usage
        memory_usage = self.check_memory_usage()
        for model_name in models:
            self.training_metrics['memory_usage'][model_name] = memory_usage
        
        return models
    
    def create_neural_network(self, input_dim: int, n_classes: int) -> Optional[Any]:
        """Create optimized neural network"""
        
        if not TF_AVAILABLE:
            self.logger.warning("TensorFlow not available, skipping neural network")
            return None
        
        # Memory-efficient neural network architecture
        model = keras.Sequential([
            # Input layer with batch normalization
            keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            # Hidden layers with progressive size reduction
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            
            # Output layer
            keras.layers.Dense(n_classes, activation='softmax')
        ])
        
        # Memory-efficient optimizer
        optimizer = keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> Optional[Dict]:
        """Train memory-efficient neural network"""
        
        if not TF_AVAILABLE:
            return None
        
        print("🧠 Training optimized neural network...")
        
        n_classes = len(np.unique(y_train))
        input_dim = X_train.shape[1]
        
        # Create model
        model = self.create_neural_network(input_dim, n_classes)
        
        if model is None:
            return None
        
        # Memory-efficient callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train with memory monitoring
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=128,  # Memory-efficient batch size
            callbacks=callbacks,
            verbose=0
        )
        
        training_time = time.time() - start_time
        
        # Get best validation accuracy
        best_val_accuracy = max(history.history['val_accuracy'])
        
        return {
            'model': model,
            'history': history,
            'val_accuracy': best_val_accuracy,
            'training_time': training_time
        }
    
    def create_ensemble(self, models: Dict, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Create memory-efficient ensemble"""
        
        print("🤝 Creating optimized ensemble...")
        
        # Select best models for ensemble (top 3 to avoid overfitting)
        model_scores = [(name, result['val_accuracy']) for name, result in models.items()]
        model_scores.sort(key=lambda x: x[1], reverse=True)
        top_models = model_scores[:3]
        
        print(f"📊 Top models for ensemble: {[name for name, _ in top_models]}")
        
        # Create voting ensemble
        estimators = [(name, models[name]['model']) for name, _ in top_models]
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probabilities
            n_jobs=self.n_jobs
        )
        
        start_time = time.time()
        ensemble.fit(X_train, y_train)
        ensemble_time = time.time() - start_time
        
        return {
            'model': ensemble,
            'component_models': [name for name, _ in top_models],
            'training_time': ensemble_time
        }
    
    def evaluate_models(self, models: Dict, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Comprehensive model evaluation"""
        
        print("📊 Evaluating models...")
        
        results = {}
        
        for model_name, model_info in models.items():
            if 'model' not in model_info:
                continue
                
            model = model_info['model']
            
            try:
                # Predictions
                start_time = time.time()
                y_pred = model.predict(X_test)
                prediction_time = time.time() - start_time
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'prediction_time': prediction_time,
                    'training_time': model_info.get('training_time', 0),
                    'val_accuracy': model_info.get('val_accuracy', 0)
                }
                
                print(f"✅ {model_name:<25} {accuracy:.4f} accuracy")
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {e}")
                
        return results
    
    def save_optimized_models(self, models: Dict, results: Dict):
        """Save models with compression"""
        
        print("💾 Saving optimized models...")
        
        # Sort by performance
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        # Save top 3 models
        for i, (model_name, metrics) in enumerate(sorted_results[:3]):
            if model_name in models and 'model' in models[model_name]:
                
                model = models[model_name]['model']
                
                # Save with compression
                model_path = self.models_dir / f"rank_{i+1}_{model_name.lower()}.joblib"
                
                try:
                    if hasattr(model, 'save') and TF_AVAILABLE:  # TensorFlow model
                        keras_path = self.models_dir / f"rank_{i+1}_{model_name.lower()}.keras"
                        model.save(keras_path)
                        print(f"💾 Saved Keras model: {keras_path}")
                    else:  # Scikit-learn model
                        joblib.dump(model, model_path, compress=3)
                        print(f"💾 Saved model: {model_path}")
                        
                except Exception as e:
                    self.logger.error(f"Error saving {model_name}: {e}")
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'model_results': results,
            'training_metrics': self.training_metrics,
            'system_info': {
                'cpu_count': os.cpu_count(),
                'memory_limit_gb': self.memory_limit_gb,
                'gpu_available': self.use_gpu and (TF_AVAILABLE or TORCH_AVAILABLE)
            }
        }
        
        metadata_path = self.results_dir / f"training_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"📄 Saved metadata: {metadata_path}")
    
    def run_optimized_training(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Run complete optimized training pipeline"""
        
        print("\n🚀 STARTING OPTIMIZED TRAINING PIPELINE")
        print("=" * 60)
        
        # Check initial memory
        initial_memory = self.check_memory_usage()
        print(f"🧠 Initial memory: {initial_memory['process_memory_gb']:.1f}GB")
        
        # 1. Train tree-based models
        tree_models = self.train_tree_based_models(X_train, y_train, X_val, y_val)
        
        # 2. Train neural network (if available)
        neural_result = self.train_neural_network(X_train, y_train, X_val, y_val)
        
        all_models = tree_models.copy()
        if neural_result:
            all_models['NeuralNetwork_Optimized'] = neural_result
        
        # 3. Create ensemble
        ensemble_result = self.create_ensemble(all_models, X_train, y_train)
        all_models['Ensemble_Optimized'] = ensemble_result
        
        # 4. Final evaluation
        results = self.evaluate_models(all_models, X_test, y_test)
        
        # 5. Save models
        self.save_optimized_models(all_models, results)
        
        # Final memory check
        final_memory = self.check_memory_usage()
        
        print(f"\n🎉 OPTIMIZED TRAINING COMPLETE!")
        print("=" * 50)
        print(f"🧠 Memory usage: {initial_memory['process_memory_gb']:.1f}GB → {final_memory['process_memory_gb']:.1f}GB")
        print(f"🏆 Best model: {max(results.items(), key=lambda x: x[1]['accuracy'])[0]}")
        print(f"🎯 Best accuracy: {max(results.values(), key=lambda x: x['accuracy'])['accuracy']:.4f}")
        
        return {
            'models': all_models,
            'results': results,
            'memory_usage': {
                'initial': initial_memory,
                'final': final_memory
            }
        }

def main():
    """Demonstrate optimized training system"""
    
    print("🚀 OPTIMIZED MODEL TRAINING DEMO")
    print("=" * 40)
    
    # Import the optimized data pipeline
    from optimized_data_pipeline import OptimizedExoplanetDataset
    
    try:
        # Load data
        dataset = OptimizedExoplanetDataset()
        features, labels = dataset.load_exoplanet_data(enhanced=True)
        splits = dataset.create_train_test_splits(features, labels)
        
        # Initialize trainer
        trainer = OptimizedModelTrainer()
        
        # Prepare data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        X_train_scaled = scaler.fit_transform(splits['X_train'])
        X_test_scaled = scaler.transform(splits['X_test'])
        
        # Split training data for validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_scaled, splits['y_train'], 
            test_size=0.2, stratify=splits['y_train'], random_state=42
        )
        
        # Run training
        training_results = trainer.run_optimized_training(
            X_train_final, y_train_final,
            X_val, y_val,
            X_test_scaled, splits['y_test']
        )
        
        return training_results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()