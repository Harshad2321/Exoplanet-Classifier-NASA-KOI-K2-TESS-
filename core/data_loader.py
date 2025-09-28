"""
NASA Space Apps Challenge 2025 - Exoplanet Classifier
Memory-Optimized Data Loading System

Efficient data loading with chunking, caching, and memory management.
"""

import gc
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Iterator, Any
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from .config import (
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR, CACHE_DIR,
    DATA_CONFIG, ALL_FEATURES, TARGET_MAPPING, setup_logging
)

warnings.filterwarnings('ignore')


class MemoryMonitor:
    """Monitor and manage memory usage during data operations"""
    
    def __init__(self, max_memory_gb: float = 4.0):
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.logger = setup_logging("memory_monitor")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_gb': memory_info.rss / 1024**3,
                'vms_gb': memory_info.vms / 1024**3,
                'percent': process.memory_percent(),
                'available_gb': psutil.virtual_memory().available / 1024**3
            }
        except ImportError:
            # Fallback without psutil
            import sys
            return {
                'python_objects_mb': sys.getsizeof(gc.get_objects()) / 1024**2,
                'available_gb': 'unknown'
            }
    
    def should_trigger_gc(self) -> bool:
        """Check if garbage collection should be triggered"""
        memory_info = self.get_memory_usage()
        if 'rss_gb' in memory_info:
            return memory_info['rss_gb'] > self.max_memory_bytes / 1024**3 * 0.8
        return False
    
    def cleanup_memory(self):
        """Force garbage collection and memory cleanup"""
        gc.collect()
        self.logger.debug("Memory cleanup performed")


class DataCache:
    """Smart caching system for processed data chunks"""
    
    def __init__(self, cache_dir: Path = None, max_cache_size_gb: float = 1.0):
        self.cache_dir = cache_dir or CACHE_DIR / "data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size_gb * 1024**3
        self.logger = setup_logging("data_cache")
        self._cache_index_file = self.cache_dir / "cache_index.json"
        self._load_cache_index()
    
    def _load_cache_index(self):
        """Load cache index from disk"""
        try:
            if self._cache_index_file.exists():
                with open(self._cache_index_file, 'r') as f:
                    self.cache_index = json.load(f)
            else:
                self.cache_index = {}
        except Exception as e:
            self.logger.warning(f"Could not load cache index: {e}")
            self.cache_index = {}
    
    def _save_cache_index(self):
        """Save cache index to disk"""
        try:
            with open(self._cache_index_file, 'w') as f:
                json.dump(self.cache_index, f)
        except Exception as e:
            self.logger.error(f"Could not save cache index: {e}")
    
    def _generate_key(self, data_source: str, params: Dict) -> str:
        """Generate cache key from data source and parameters"""
        key_string = f"{data_source}_{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def _cleanup_old_files(self):
        """Remove old cache files to maintain size limit"""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            if total_size > self.max_cache_size:
                # Sort by access time (oldest first)
                cache_files.sort(key=lambda x: self.cache_index.get(x.stem, {}).get('last_accessed', 0))
                
                while total_size > self.max_cache_size * 0.8 and cache_files:
                    file_to_remove = cache_files.pop(0)
                    file_size = file_to_remove.stat().st_size
                    file_to_remove.unlink()
                    
                    # Remove from index
                    if file_to_remove.stem in self.cache_index:
                        del self.cache_index[file_to_remove.stem]
                    
                    total_size -= file_size
                    self.logger.debug(f"Removed cache file: {file_to_remove.name}")
                
                self._save_cache_index()
                
        except Exception as e:
            self.logger.error(f"Cache cleanup failed: {e}")
    
    def get(self, data_source: str, params: Dict) -> Optional[Any]:
        """Get data from cache"""
        key = self._generate_key(data_source, params)
        cache_file = self.cache_dir / f"{key}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            # Update access time
            self.cache_index[key] = {
                'last_accessed': time.time(),
                'size': cache_file.stat().st_size,
                'source': data_source
            }
            self._save_cache_index()
            
            self.logger.debug(f"Cache hit: {key}")
            return data
            
        except Exception as e:
            self.logger.warning(f"Cache read failed for {key}: {e}")
            # Remove corrupted cache file
            cache_file.unlink(missing_ok=True)
            return None
    
    def put(self, data_source: str, params: Dict, data: Any):
        """Put data in cache"""
        key = self._generate_key(data_source, params)
        cache_file = self.cache_dir / f"{key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            # Update index
            self.cache_index[key] = {
                'last_accessed': time.time(),
                'size': cache_file.stat().st_size,
                'source': data_source
            }
            self._save_cache_index()
            
            # Cleanup if needed
            self._cleanup_old_files()
            
            self.logger.debug(f"Cached data: {key}")
            
        except Exception as e:
            self.logger.error(f"Cache write failed for {key}: {e}")
    
    def clear(self, pattern: str = None):
        """Clear cache files"""
        try:
            if pattern:
                files = list(self.cache_dir.glob(f"*{pattern}*.pkl"))
            else:
                files = list(self.cache_dir.glob("*.pkl"))
            
            for file in files:
                file.unlink()
                if file.stem in self.cache_index:
                    del self.cache_index[file.stem]
            
            self._save_cache_index()
            self.logger.info(f"Cleared {len(files)} cache files")
            
        except Exception as e:
            self.logger.error(f"Cache clear failed: {e}")


class ChunkedDataLoader:
    """Memory-efficient data loader with chunking support"""
    
    def __init__(self, chunk_size: int = None, memory_monitor: MemoryMonitor = None):
        self.chunk_size = chunk_size or DATA_CONFIG.chunk_size
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.data_cache = DataCache()
        self.logger = setup_logging("data_loader")
        
        # Feature processing components
        self.feature_encoders = {}
        self.scalers = {}
        self.feature_names = []
    
    def load_csv_chunks(self, file_path: Union[str, Path], 
                       **kwargs) -> Iterator[pd.DataFrame]:
        """Load CSV file in chunks"""
        file_path = Path(file_path)
        
        try:
            chunk_params = {
                'chunksize': self.chunk_size,
                'dtype': kwargs.get('dtype', None),
                'usecols': kwargs.get('usecols', None),
                'low_memory': True
            }
            
            # Remove None values
            chunk_params = {k: v for k, v in chunk_params.items() if v is not None}
            
            self.logger.info(f"Loading {file_path.name} in chunks of {self.chunk_size}")
            
            for i, chunk in enumerate(pd.read_csv(file_path, **chunk_params)):
                self.logger.debug(f"Loaded chunk {i+1}, shape: {chunk.shape}")
                
                # Memory check
                if self.memory_monitor.should_trigger_gc():
                    self.memory_monitor.cleanup_memory()
                
                yield chunk
                
        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {e}")
            raise
    
    def load_with_cache(self, file_path: Union[str, Path], 
                       processing_params: Dict = None) -> pd.DataFrame:
        """Load data with caching support"""
        file_path = Path(file_path)
        processing_params = processing_params or {}
        
        # Check cache first
        cache_key_params = {
            'file_path': str(file_path),
            'file_mtime': file_path.stat().st_mtime,
            **processing_params
        }
        
        cached_data = self.data_cache.get(str(file_path), cache_key_params)
        if cached_data is not None:
            self.logger.info(f"Loaded {file_path.name} from cache")
            return cached_data
        
        # Load from disk
        try:
            if file_path.suffix.lower() == '.csv':
                data = self._load_csv_optimized(file_path, processing_params)
            elif file_path.suffix.lower() in ['.pkl', '.pickle']:
                data = pd.read_pickle(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # Cache the processed data
            self.data_cache.put(str(file_path), cache_key_params, data)
            
            self.logger.info(f"Loaded {file_path.name}, shape: {data.shape}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {e}")
            raise
    
    def _load_csv_optimized(self, file_path: Path, params: Dict) -> pd.DataFrame:
        """Load CSV with memory optimizations"""
        dtype_optimizations = {
            'float64': 'float32',
            'int64': 'int32'
        }
        
        # First pass: determine optimal dtypes
        sample = pd.read_csv(file_path, nrows=1000)
        optimized_dtypes = {}
        
        for col, dtype in sample.dtypes.items():
            if dtype.name in dtype_optimizations:
                # Check if we can safely downcast
                try:
                    sample_col = sample[col].dropna()
                    if dtype.name == 'float64':
                        if sample_col.between(-3.4e38, 3.4e38).all():
                            optimized_dtypes[col] = 'float32'
                    elif dtype.name == 'int64':
                        if sample_col.between(-2147483648, 2147483647).all():
                            optimized_dtypes[col] = 'int32'
                except:
                    pass  # Keep original dtype if conversion fails
        
        # Load with optimized dtypes
        data = pd.read_csv(file_path, dtype=optimized_dtypes, low_memory=True)
        
        self.logger.debug(f"Optimized dtypes for {len(optimized_dtypes)} columns")
        return data
    
    def preprocess_features(self, data: pd.DataFrame, 
                           fit_transforms: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess features with memory optimization"""
        try:
            # Handle missing values
            data = data.copy()
            
            # Separate features and target
            target_col = None
            for col in data.columns:
                if col.lower() in ['disposition', 'target', 'label', 'class']:
                    target_col = col
                    break
            
            if target_col:
                X = data.drop(columns=[target_col])
                y = data[target_col]
            else:
                X = data
                y = None
            
            # Encode target variable
            if y is not None:
                if fit_transforms:
                    self.target_encoder = LabelEncoder()
                    y_encoded = self.target_encoder.fit_transform(y)
                elif hasattr(self, 'target_encoder'):
                    y_encoded = self.target_encoder.transform(y)
                else:
                    # Use TARGET_MAPPING
                    y_encoded = y.map(TARGET_MAPPING).fillna(0).astype(int)
            else:
                y_encoded = None
            
            # Handle categorical features
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0 and fit_transforms:
                for col in categorical_cols:
                    self.feature_encoders[col] = LabelEncoder()
                    X[col] = self.feature_encoders[col].fit_transform(X[col].astype(str))
            elif hasattr(self, 'feature_encoders'):
                for col in categorical_cols:
                    if col in self.feature_encoders:
                        X[col] = self.feature_encoders[col].transform(X[col].astype(str))
            
            # Handle missing values
            X = X.fillna(X.median())
            
            # Scale features
            if fit_transforms:
                self.scalers['standard'] = StandardScaler()
                self.scalers['robust'] = RobustScaler()
                
                X_scaled = self.scalers['standard'].fit_transform(X)
                self.feature_names = list(X.columns)
            elif 'standard' in self.scalers:
                X_scaled = self.scalers['standard'].transform(X)
            else:
                X_scaled = X.values
            
            self.logger.debug(f"Preprocessed data: X{X_scaled.shape}, y{y_encoded.shape if y_encoded is not None else 'None'}")
            
            return X_scaled, y_encoded
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            raise
    
    def load_training_data(self, data_sources: List[Union[str, Path]], 
                          test_size: float = None, 
                          validation_size: float = None,
                          random_state: int = None) -> Dict[str, np.ndarray]:
        """Load and split training data efficiently"""
        test_size = test_size or DATA_CONFIG.train_test_split
        validation_size = validation_size or DATA_CONFIG.validation_split
        random_state = random_state or DATA_CONFIG.random_state
        
        all_data = []
        
        # Load all data sources
        for source in data_sources:
            source_path = Path(source)
            if not source_path.exists():
                # Try in data directories
                for data_dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, DATA_DIR]:
                    potential_path = data_dir / source_path.name
                    if potential_path.exists():
                        source_path = potential_path
                        break
            
            if source_path.exists():
                data = self.load_with_cache(source_path)
                all_data.append(data)
                self.logger.info(f"Loaded {source_path.name}: {data.shape}")
            else:
                self.logger.warning(f"Data source not found: {source}")
        
        if not all_data:
            raise RuntimeError("No data sources could be loaded")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True, sort=False)
        self.logger.info(f"Combined dataset shape: {combined_data.shape}")
        
        # Preprocess
        X, y = self.preprocess_features(combined_data, fit_transforms=True)
        
        # Split data
        if y is not None:
            # Train/test split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state,
                stratify=y if DATA_CONFIG.stratify else None
            )
            
            # Train/validation split
            if validation_size > 0:
                val_size = validation_size / (1 - test_size)  # Adjust for already split data
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=val_size, random_state=random_state,
                    stratify=y_temp if DATA_CONFIG.stratify else None
                )
            else:
                X_train, X_val, y_train, y_val = X_temp, None, y_temp, None
            
            result = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
            
            if X_val is not None:
                result.update({
                    'X_val': X_val,
                    'y_val': y_val
                })
            
            self.logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}" + 
                           (f", Val: {X_val.shape}" if X_val is not None else ""))
            
            return result
        else:
            return {'X': X}
    
    def save_preprocessors(self, save_dir: Path = None):
        """Save preprocessing components"""
        save_dir = save_dir or PROCESSED_DATA_DIR / "preprocessors"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save scalers
            if self.scalers:
                with open(save_dir / "scalers.pkl", 'wb') as f:
                    pickle.dump(self.scalers, f)
            
            # Save encoders
            if self.feature_encoders:
                with open(save_dir / "encoders.pkl", 'wb') as f:
                    pickle.dump(self.feature_encoders, f)
            
            # Save feature names
            if self.feature_names:
                with open(save_dir / "feature_names.json", 'w') as f:
                    json.dump(self.feature_names, f)
            
            # Save target encoder
            if hasattr(self, 'target_encoder'):
                with open(save_dir / "target_encoder.pkl", 'wb') as f:
                    pickle.dump(self.target_encoder, f)
            
            self.logger.info(f"Preprocessors saved to {save_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save preprocessors: {e}")
    
    def get_memory_usage_report(self) -> Dict[str, Any]:
        """Get detailed memory usage report"""
        memory_info = self.memory_monitor.get_memory_usage()
        
        return {
            'memory_usage': memory_info,
            'cache_stats': {
                'size': len(self.data_cache.cache_index),
                'cache_dir_size_mb': sum(
                    f.stat().st_size for f in self.data_cache.cache_dir.glob('*.pkl')
                ) / 1024**2 if self.data_cache.cache_dir.exists() else 0
            },
            'loader_config': {
                'chunk_size': self.chunk_size,
                'scalers_loaded': len(self.scalers),
                'encoders_loaded': len(self.feature_encoders),
                'feature_count': len(self.feature_names)
            }
        }


# Global loader instance
_data_loader = None

def get_data_loader(**kwargs) -> ChunkedDataLoader:
    """Get global data loader instance"""
    global _data_loader
    if _data_loader is None:
        _data_loader = ChunkedDataLoader(**kwargs)
    return _data_loader


# Convenience functions
def load_exoplanet_data(data_sources: List[str] = None) -> Dict[str, np.ndarray]:
    """Quick function to load exoplanet training data"""
    if data_sources is None:
        # Default data sources
        data_sources = []
        for data_dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, DATA_DIR]:
            for pattern in ["*.csv", "exoplanet*.csv", "kepler*.csv", "k2*.csv", "tess*.csv"]:
                data_sources.extend([str(f) for f in data_dir.glob(pattern)])
    
    loader = get_data_loader()
    return loader.load_training_data(data_sources)


if __name__ == "__main__":
    # Example usage
    loader = ChunkedDataLoader()
    
    # Memory report
    report = loader.get_memory_usage_report()
    print("Memory Usage Report:", json.dumps(report, indent=2))
    
    # Try to load data if available
    try:
        data = load_exoplanet_data()
        print(f"Loaded data with keys: {list(data.keys())}")
        for key, value in data.items():
            if hasattr(value, 'shape'):
                print(f"{key}: {value.shape}")
    except Exception as e:
        print(f"Could not load data: {e}")