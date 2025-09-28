"""
NASA Space Apps Challenge 2025 - Exoplanet Classifier
Standardized Prediction API System

High-performance, memory-efficient prediction interface with caching and async support.
"""

import asyncio
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, RobustScaler

from .config import (
    MODELS_DIR, PRODUCTION_MODELS_DIR, PREPROCESSORS_DIR, CACHE_DIR,
    MODEL_CONFIG, APP_CONFIG, setup_logging, get_device_info
)

warnings.filterwarnings('ignore')


class ModelCache:
    """Intelligent model caching system with memory management"""
    
    def __init__(self, max_size: int = 3, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Dict] = {}
        self.access_times: Dict[str, float] = {}
        self.logger = setup_logging("model_cache")
    
    def _generate_key(self, model_path: Union[str, Path]) -> str:
        """Generate cache key from model path"""
        return hashlib.md5(str(model_path).encode()).hexdigest()[:16]
    
    def _is_expired(self, key: str) -> bool:
        """Check if cached item is expired"""
        if key not in self.access_times:
            return True
        return time.time() - self.access_times[key] > self.ttl
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.cache:
            return
        
        lru_key = min(self.access_times.keys(), key=self.access_times.get)
        del self.cache[lru_key]
        del self.access_times[lru_key]
        self.logger.debug(f"Evicted model from cache: {lru_key}")
    
    def get(self, model_path: Union[str, Path]) -> Optional[Dict]:
        """Get model from cache"""
        key = self._generate_key(model_path)
        
        if key not in self.cache or self._is_expired(key):
            return None
        
        self.access_times[key] = time.time()
        return self.cache[key]
    
    def put(self, model_path: Union[str, Path], model_data: Dict):
        """Put model in cache with LRU eviction"""
        key = self._generate_key(model_path)
        
        # Evict if cache is full
        while len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = model_data
        self.access_times[key] = time.time()
        self.logger.debug(f"Cached model: {key}")
    
    def clear(self):
        """Clear all cached models"""
        self.cache.clear()
        self.access_times.clear()
        self.logger.info("Model cache cleared")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'keys': list(self.cache.keys()),
            'ttl': self.ttl
        }


class DataPreprocessor:
    """Optimized data preprocessing with caching"""
    
    def __init__(self):
        self.scalers: Dict[str, BaseEstimator] = {}
        self.feature_names: List[str] = []
        self.is_fitted = False
        self.logger = setup_logging("preprocessor")
    
    def load_preprocessors(self, preprocessor_dir: Path = None):
        """Load saved preprocessors"""
        if preprocessor_dir is None:
            preprocessor_dir = PREPROCESSORS_DIR
        
        try:
            # Load standard scaler
            scaler_path = preprocessor_dir / "standard_scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scalers['standard'] = pickle.load(f)
            
            # Load robust scaler
            robust_scaler_path = preprocessor_dir / "robust_scaler.pkl"
            if robust_scaler_path.exists():
                with open(robust_scaler_path, 'rb') as f:
                    self.scalers['robust'] = pickle.load(f)
            
            # Load feature names
            features_path = preprocessor_dir / "feature_names.json"
            if features_path.exists():
                with open(features_path, 'r') as f:
                    self.feature_names = json.load(f)
            
            self.is_fitted = bool(self.scalers)
            self.logger.info(f"Loaded {len(self.scalers)} preprocessors")
            
        except Exception as e:
            self.logger.error(f"Failed to load preprocessors: {e}")
    
    def validate_input(self, data: Union[pd.DataFrame, np.ndarray, Dict]) -> pd.DataFrame:
        """Validate and convert input data to DataFrame"""
        try:
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            elif isinstance(data, np.ndarray):
                data = pd.DataFrame(data, columns=self.feature_names[:data.shape[1]])
            elif isinstance(data, list):
                data = pd.DataFrame(data)
            
            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"Cannot convert data type {type(data)} to DataFrame")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            raise ValueError(f"Invalid input data: {e}")
    
    def preprocess(self, data: Union[pd.DataFrame, np.ndarray, Dict], 
                  scaler_type: str = "standard") -> np.ndarray:
        """Preprocess input data"""
        # Validate input
        df = self.validate_input(data)
        
        # Handle missing values
        if df.isnull().any().any():
            df = df.fillna(df.median())
            self.logger.warning("Missing values detected and filled with median")
        
        # Apply scaling
        if scaler_type in self.scalers:
            try:
                scaled_data = self.scalers[scaler_type].transform(df)
                return scaled_data
            except Exception as e:
                self.logger.warning(f"Scaling failed with {scaler_type}, using robust scaler: {e}")
                if 'robust' in self.scalers:
                    return self.scalers['robust'].transform(df)
        
        # Fallback to simple normalization
        self.logger.warning("No suitable scaler found, using simple normalization")
        return (df - df.mean()) / df.std()
    
    def get_feature_importance(self, model, feature_names: List[str] = None) -> Dict[str, float]:
        """Extract feature importance from model"""
        if feature_names is None:
            feature_names = self.feature_names
        
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
            else:
                return {}
            
            return dict(zip(feature_names[:len(importance)], importance))
            
        except Exception as e:
            self.logger.error(f"Failed to extract feature importance: {e}")
            return {}


class PredictionAPI:
    """Main prediction API with async support and caching"""
    
    def __init__(self, model_cache_size: int = 3, enable_async: bool = True):
        self.model_cache = ModelCache(max_size=model_cache_size)
        self.preprocessor = DataPreprocessor()
        self.enable_async = enable_async
        self.executor = ThreadPoolExecutor(max_workers=MODEL_CONFIG.max_models_ensemble)
        self.logger = setup_logging("prediction_api")
        
        # Load preprocessors
        self.preprocessor.load_preprocessors()
        
        # Device info
        self.device_info = get_device_info()
        self.logger.info(f"PredictionAPI initialized with device: {self.device_info['current_device']}")
    
    def _load_model(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """Load model with caching"""
        model_path = Path(model_path)
        
        # Check cache first
        cached = self.model_cache.get(model_path)
        if cached:
            self.logger.debug(f"Model loaded from cache: {model_path.name}")
            return cached
        
        # Load from disk
        try:
            if model_path.suffix == '.pkl':
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            elif model_path.suffix in ['.joblib', '.pkl']:
                import joblib
                model = joblib.load(model_path)
            else:
                raise ValueError(f"Unsupported model format: {model_path.suffix}")
            
            # Get model metadata
            metadata = self._get_model_metadata(model, model_path)
            
            model_data = {
                'model': model,
                'metadata': metadata,
                'path': str(model_path),
                'loaded_at': time.time()
            }
            
            # Cache the model
            self.model_cache.put(model_path, model_data)
            
            self.logger.info(f"Model loaded successfully: {model_path.name}")
            return model_data
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_path}: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _get_model_metadata(self, model, model_path: Path) -> Dict[str, Any]:
        """Extract model metadata"""
        metadata = {
            'name': model_path.stem,
            'type': type(model).__name__,
            'file_size': model_path.stat().st_size if model_path.exists() else 0,
            'features': getattr(model, 'feature_names_in_', []),
            'classes': getattr(model, 'classes_', []),
            'n_features': getattr(model, 'n_features_in_', None)
        }
        
        # Add model-specific metadata
        if hasattr(model, 'get_params'):
            try:
                metadata['params'] = model.get_params()
            except:
                pass
        
        return metadata
    
    def predict_single(self, model_path: Union[str, Path], 
                      data: Union[pd.DataFrame, np.ndarray, Dict],
                      return_proba: bool = True,
                      return_features: bool = False) -> Dict[str, Any]:
        """Single model prediction"""
        start_time = time.time()
        
        try:
            # Load model
            model_data = self._load_model(model_path)
            model = model_data['model']
            
            # Preprocess data
            X = self.preprocessor.preprocess(data)
            
            # Make prediction
            if return_proba and hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                prediction = np.argmax(proba, axis=1)
                confidence = np.max(proba, axis=1)
            else:
                prediction = model.predict(X)
                proba = None
                confidence = None
            
            # Prepare result
            result = {
                'predictions': prediction.tolist() if hasattr(prediction, 'tolist') else [prediction],
                'model_name': model_data['metadata']['name'],
                'model_type': model_data['metadata']['type'],
                'processing_time': time.time() - start_time
            }
            
            if proba is not None:
                result['probabilities'] = proba.tolist()
                result['confidence'] = confidence.tolist()
            
            if return_features:
                feature_importance = self.preprocessor.get_feature_importance(
                    model, self.preprocessor.feature_names
                )
                result['feature_importance'] = feature_importance
            
            self.logger.debug(f"Prediction completed in {result['processing_time']:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction error: {e}")
    
    async def predict_single_async(self, model_path: Union[str, Path], 
                                  data: Union[pd.DataFrame, np.ndarray, Dict],
                                  **kwargs) -> Dict[str, Any]:
        """Async single model prediction"""
        if not self.enable_async:
            return self.predict_single(model_path, data, **kwargs)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.predict_single, 
            model_path, data, kwargs.get('return_proba', True), 
            kwargs.get('return_features', False)
        )
    
    def predict_ensemble(self, model_paths: List[Union[str, Path]],
                        data: Union[pd.DataFrame, np.ndarray, Dict],
                        voting: str = 'soft',
                        return_individual: bool = False) -> Dict[str, Any]:
        """Ensemble prediction from multiple models"""
        start_time = time.time()
        
        try:
            individual_results = []
            all_probabilities = []
            
            for model_path in model_paths:
                try:
                    result = self.predict_single(model_path, data, return_proba=True)
                    individual_results.append(result)
                    
                    if 'probabilities' in result:
                        all_probabilities.append(np.array(result['probabilities']))
                    
                except Exception as e:
                    self.logger.warning(f"Model {model_path} failed: {e}")
                    continue
            
            if not individual_results:
                raise RuntimeError("All models failed to make predictions")
            
            # Ensemble predictions
            if voting == 'soft' and all_probabilities:
                # Average probabilities
                ensemble_proba = np.mean(all_probabilities, axis=0)
                ensemble_pred = np.argmax(ensemble_proba, axis=1)
                ensemble_confidence = np.max(ensemble_proba, axis=1)
            else:
                # Hard voting
                all_predictions = [np.array(r['predictions']) for r in individual_results]
                from scipy import stats
                ensemble_pred = stats.mode(all_predictions, axis=0)[0].flatten()
                ensemble_proba = None
                ensemble_confidence = None
            
            result = {
                'predictions': ensemble_pred.tolist(),
                'ensemble_method': voting,
                'n_models': len(individual_results),
                'processing_time': time.time() - start_time
            }
            
            if ensemble_proba is not None:
                result['probabilities'] = ensemble_proba.tolist()
                result['confidence'] = ensemble_confidence.tolist()
            
            if return_individual:
                result['individual_results'] = individual_results
            
            self.logger.info(f"Ensemble prediction completed with {len(individual_results)} models")
            return result
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction failed: {e}")
            raise RuntimeError(f"Ensemble prediction error: {e}")
    
    async def predict_ensemble_async(self, model_paths: List[Union[str, Path]],
                                   data: Union[pd.DataFrame, np.ndarray, Dict],
                                   **kwargs) -> Dict[str, Any]:
        """Async ensemble prediction"""
        if not self.enable_async:
            return self.predict_ensemble(model_paths, data, **kwargs)
        
        # Run individual predictions concurrently
        tasks = [
            self.predict_single_async(model_path, data, return_proba=True)
            for model_path in model_paths
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        
        if not successful_results:
            raise RuntimeError("All async model predictions failed")
        
        # Process ensemble result (similar to sync version)
        # ... (implementation details similar to predict_ensemble)
        
        return {
            'predictions': [],  # Placeholder
            'ensemble_method': kwargs.get('voting', 'soft'),
            'n_models': len(successful_results),
            'processing_time': 0  # Calculate actual time
        }
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        models = []
        
        for model_dir in [PRODUCTION_MODELS_DIR, MODELS_DIR]:
            if model_dir.exists():
                for model_file in model_dir.glob("*.pkl"):
                    try:
                        model_data = self._load_model(model_file)
                        models.append({
                            'name': model_data['metadata']['name'],
                            'path': str(model_file),
                            'type': model_data['metadata']['type'],
                            'size': model_data['metadata']['file_size'],
                            'features': len(model_data['metadata'].get('features', [])),
                            'cached': self.model_cache.get(model_file) is not None
                        })
                    except Exception as e:
                        self.logger.warning(f"Could not load model metadata for {model_file}: {e}")
        
        return models
    
    def health_check(self) -> Dict[str, Any]:
        """API health check"""
        return {
            'status': 'healthy',
            'cache_stats': self.model_cache.stats(),
            'device_info': self.device_info,
            'available_models': len(self.get_available_models()),
            'preprocessor_loaded': self.preprocessor.is_fitted,
            'async_enabled': self.enable_async,
            'timestamp': time.time()
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.model_cache.clear()
        self.logger.info("All caches cleared")
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# Global API instance
_prediction_api = None

def get_prediction_api(**kwargs) -> PredictionAPI:
    """Get global prediction API instance (singleton pattern)"""
    global _prediction_api
    if _prediction_api is None:
        _prediction_api = PredictionAPI(**kwargs)
    return _prediction_api


# Convenience functions
def predict(model_path: Union[str, Path], data: Union[pd.DataFrame, np.ndarray, Dict], 
           **kwargs) -> Dict[str, Any]:
    """Quick prediction function"""
    api = get_prediction_api()
    return api.predict_single(model_path, data, **kwargs)

async def predict_async(model_path: Union[str, Path], data: Union[pd.DataFrame, np.ndarray, Dict], 
                       **kwargs) -> Dict[str, Any]:
    """Quick async prediction function"""
    api = get_prediction_api()
    return await api.predict_single_async(model_path, data, **kwargs)

def predict_ensemble(model_paths: List[Union[str, Path]], 
                    data: Union[pd.DataFrame, np.ndarray, Dict], 
                    **kwargs) -> Dict[str, Any]:
    """Quick ensemble prediction function"""
    api = get_prediction_api()
    return api.predict_ensemble(model_paths, data, **kwargs)


# Context manager for batch processing
@asynccontextmanager
async def prediction_session(**api_kwargs):
    """Context manager for prediction sessions with cleanup"""
    api = PredictionAPI(**api_kwargs)
    try:
        yield api
    finally:
        api.clear_cache()
        if hasattr(api, 'executor'):
            api.executor.shutdown(wait=True)


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Initialize API
        api = get_prediction_api()
        
        # Health check
        health = api.health_check()
        print("Health Check:", health)
        
        # List available models
        models = api.get_available_models()
        print(f"Available Models: {len(models)}")
        
        if models:
            # Example prediction
            sample_data = {
                'koi_period': 365.25,
                'koi_prad': 1.0,
                'koi_teq': 288,
                'koi_insol': 1.0
            }
            
            try:
                result = await api.predict_single_async(
                    models[0]['path'], sample_data, return_features=True
                )
                print("Prediction Result:", result)
            except Exception as e:
                print(f"Prediction failed: {e}")
    
    asyncio.run(main())