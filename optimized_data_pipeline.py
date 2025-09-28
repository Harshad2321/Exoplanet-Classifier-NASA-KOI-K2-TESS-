#!/usr/bin/env python3
"""
🚀 OPTIMIZED DATA PIPELINE
Memory-efficient, fast, and scalable data loading and processing
"""

import numpy as np
import pandas as pd
import os
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import psutil

# For data compression and optimization
import joblib
try:
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

class OptimizedDataPipeline:
    """
    🎯 Ultra-efficient data pipeline with memory optimization and performance tuning
    """
    
    def __init__(self, 
                 cache_size: int = 128,
                 batch_size: int = 1000,
                 n_workers: int = None,
                 memory_threshold: float = 0.8):
        
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.n_workers = n_workers or min(4, os.cpu_count())
        self.memory_threshold = memory_threshold
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics
        self.metrics = {
            'load_times': [],
            'memory_usage': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        print("🚀 Optimized Data Pipeline Initialized")
        print(f"   💾 Cache size: {cache_size}")
        print(f"   📦 Batch size: {batch_size}")
        print(f"   🔧 Workers: {self.n_workers}")
        print(f"   🧠 Memory threshold: {memory_threshold*100}%")
    
    def check_memory_usage(self) -> Dict[str, float]:
        """Check current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        usage = {
            'process_memory_mb': memory_info.rss / 1024 / 1024,
            'process_memory_percent': process.memory_percent(),
            'system_memory_percent': system_memory.percent / 100,
            'available_memory_gb': system_memory.available / 1024 / 1024 / 1024
        }
        
        self.metrics['memory_usage'].append(usage)
        return usage
    
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage by downcasting numeric types"""
        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Downcast integers
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Downcast floats
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert object columns to category if beneficial
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
                df[col] = df[col].astype('category')
        
        optimized_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        memory_reduction = (original_memory - optimized_memory) / original_memory * 100
        
        self.logger.info(f"Memory optimization: {original_memory:.1f}MB → {optimized_memory:.1f}MB "
                        f"({memory_reduction:.1f}% reduction)")
        
        return df
    
    @lru_cache(maxsize=128)
    def load_cached_data(self, file_path: str) -> pd.DataFrame:
        """Load data with caching for repeated access"""
        self.metrics['cache_hits'] += 1
        return self._load_data_internal(file_path)
    
    def _load_data_internal(self, file_path: str) -> pd.DataFrame:
        """Internal data loading with format detection and optimization"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load based on file type
        if path.suffix.lower() == '.csv':
            # Optimized CSV loading
            df = pd.read_csv(
                file_path,
                low_memory=False,
                dtype_backend='numpy_nullable'  # Use nullable dtypes
            )
        elif path.suffix.lower() == '.parquet' and PARQUET_AVAILABLE:
            df = pd.read_parquet(file_path)
        elif path.suffix.lower() in ['.pkl', '.pickle']:
            df = pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Optimize memory usage
        df = self.optimize_dataframe_memory(df)
        
        return df
    
    def load_data_streaming(self, file_path: str, chunk_size: int = 10000):
        """Load data in streaming fashion for large files"""
        path = Path(file_path)
        
        if path.suffix.lower() == '.csv':
            chunk_reader = pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)
            
            for chunk in chunk_reader:
                # Optimize each chunk
                chunk = self.optimize_dataframe_memory(chunk)
                yield chunk
                
                # Check memory usage
                memory_usage = self.check_memory_usage()
                if memory_usage['system_memory_percent'] > self.memory_threshold:
                    self.logger.warning("Memory usage high, forcing garbage collection")
                    gc.collect()
        else:
            # For non-CSV files, load normally but in batches
            df = self._load_data_internal(file_path)
            
            for start_idx in range(0, len(df), chunk_size):
                end_idx = min(start_idx + chunk_size, len(df))
                yield df.iloc[start_idx:end_idx].copy()
    
    def parallel_feature_engineering(self, df: pd.DataFrame, feature_functions: List) -> pd.DataFrame:
        """Apply feature engineering functions in parallel"""
        
        def apply_feature_function(args):
            func, data_chunk = args
            return func(data_chunk)
        
        # Split data into chunks for parallel processing
        chunk_size = len(df) // self.n_workers
        chunks = [df.iloc[i:i+chunk_size].copy() for i in range(0, len(df), chunk_size)]
        
        enhanced_chunks = []
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            for func in feature_functions:
                # Apply each function to all chunks in parallel
                futures = [executor.submit(apply_feature_function, (func, chunk)) for chunk in chunks]
                
                processed_chunks = []
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        processed_chunks.append(result)
                    except Exception as e:
                        self.logger.error(f"Feature engineering error: {e}")
                
                # Update chunks with processed results
                chunks = processed_chunks
        
        # Combine all chunks
        result = pd.concat(chunks, ignore_index=True)
        
        # Force garbage collection
        gc.collect()
        
        return result
    
    def create_optimized_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create optimized feature set with memory efficiency"""
        
        print("🔧 Creating optimized features...")
        
        # Start with original features
        features = df.copy()
        
        # Efficient numerical transformations
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        
        # Vectorized transformations (more efficient than loops)
        for col in numeric_cols:
            if col in features.columns:
                # Log and square root transforms
                features[f'{col}_log1p'] = np.log1p(np.abs(features[col]))
                features[f'{col}_sqrt'] = np.sqrt(np.abs(features[col]))
                
        # Physics-inspired features (vectorized)
        if all(col in features.columns for col in ['period', 'radius']):
            features['kepler_ratio'] = np.power(features['period'], 2/3) / (features['radius'] + 1e-8)
            features['orbital_velocity'] = 2 * np.pi * features['radius'] / (features['period'] + 1e-8)
            
        if all(col in features.columns for col in ['temperature', 'insolation']):
            features['luminosity_proxy'] = np.power(features['temperature'], 4) * features['insolation']
            features['habitable_zone'] = features['temperature'] / np.sqrt(features['insolation'] + 1e-8)
            
        # Clean up infinite and NaN values efficiently
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill NaN with median (computed once and reused)
        numeric_medians = features.select_dtypes(include=[np.number]).median()
        features.fillna(numeric_medians, inplace=True)
        
        # Optimize memory after feature creation
        features = self.optimize_dataframe_memory(features)
        
        print(f"✅ Feature engineering complete: {features.shape[1]} features")
        return features
    
    def save_optimized_data(self, df: pd.DataFrame, output_path: str, format: str = 'parquet'):
        """Save data in optimized format"""
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'parquet' and PARQUET_AVAILABLE:
            # Parquet is highly optimized for analytics
            df.to_parquet(path, compression='snappy', index=False)
            self.logger.info(f"Data saved as Parquet: {path}")
            
        elif format == 'feather':
            # Feather is fast for read/write
            df.to_feather(path)
            self.logger.info(f"Data saved as Feather: {path}")
            
        elif format == 'pickle_compressed':
            # Compressed pickle for Python objects
            joblib.dump(df, path, compress=3)
            self.logger.info(f"Data saved as compressed pickle: {path}")
            
        else:
            # Fallback to CSV with optimization
            df.to_csv(path, index=False, float_format='%.6f')
            self.logger.info(f"Data saved as CSV: {path}")
    
    def create_data_summary(self, df: pd.DataFrame) -> Dict:
        """Create comprehensive data summary"""
        
        summary = {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'dtypes': df.dtypes.value_counts().to_dict(),
            'missing_values': df.isnull().sum().sum(),
            'numeric_stats': {},
            'categorical_stats': {}
        }
        
        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_stats = df[numeric_cols].describe()
            summary['numeric_stats'] = {
                'mean': numeric_stats.loc['mean'].to_dict(),
                'std': numeric_stats.loc['std'].to_dict(),
                'min': numeric_stats.loc['min'].to_dict(),
                'max': numeric_stats.loc['max'].to_dict()
            }
        
        # Categorical column statistics
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            summary['categorical_stats'] = {
                col: {
                    'unique_values': df[col].nunique(),
                    'most_common': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None
                } for col in categorical_cols
            }
        
        return summary
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report"""
        
        memory_stats = self.metrics['memory_usage']
        
        if memory_stats:
            avg_memory = np.mean([m['process_memory_mb'] for m in memory_stats])
            peak_memory = np.max([m['process_memory_mb'] for m in memory_stats])
        else:
            avg_memory = peak_memory = 0
        
        return {
            'cache_efficiency': {
                'hits': self.metrics['cache_hits'],
                'misses': self.metrics['cache_misses'],
                'hit_rate': self.metrics['cache_hits'] / max(1, self.metrics['cache_hits'] + self.metrics['cache_misses'])
            },
            'memory_usage': {
                'average_mb': avg_memory,
                'peak_mb': peak_memory,
                'current_mb': self.check_memory_usage()['process_memory_mb']
            },
            'performance': {
                'avg_load_time': np.mean(self.metrics['load_times']) if self.metrics['load_times'] else 0,
                'total_operations': len(self.metrics['load_times'])
            }
        }

class OptimizedExoplanetDataset:
    """
    🌟 Optimized Exoplanet Dataset Handler
    Specialized for NASA Kepler/K2/TESS data with domain-specific optimizations
    """
    
    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.pipeline = OptimizedDataPipeline()
        
        # Domain-specific configurations
        self.exoplanet_features = [
            'period', 'radius', 'temperature', 'insolation', 
            'depth', 'ra', 'dec'
        ]
        
        self.target_classes = ['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE']
        
    def load_exoplanet_data(self, enhanced: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and optimize exoplanet data"""
        
        print("🌟 Loading Exoplanet Data")
        print("-" * 30)
        
        # Try to load from different possible locations
        possible_paths = [
            self.data_path / "processed" / "features.csv",
            self.data_path / "features.csv",
            "data/processed/features.csv"
        ]
        
        features_path = None
        for path in possible_paths:
            if Path(path).exists():
                features_path = path
                break
        
        if features_path is None:
            raise FileNotFoundError("Could not find features.csv in any expected location")
        
        # Load features
        features = self.pipeline.load_cached_data(str(features_path))
        print(f"📊 Loaded features: {features.shape}")
        
        # Load labels
        labels_path = features_path.parent / "labels.csv"
        if labels_path.exists():
            labels = self.pipeline.load_cached_data(str(labels_path))
        else:
            # Try alternative label file location
            alt_labels_path = self.data_path / "labels.csv"
            if alt_labels_path.exists():
                labels = self.pipeline.load_cached_data(str(alt_labels_path))
            else:
                raise FileNotFoundError("Could not find labels.csv")
        
        print(f"🎯 Loaded labels: {labels.shape}")
        
        if enhanced:
            # Apply optimized feature engineering
            features = self.pipeline.create_optimized_features(features)
            print(f"✨ Enhanced features: {features.shape}")
        
        # Create data summary
        summary = self.pipeline.create_data_summary(features)
        print(f"💾 Total memory usage: {summary['memory_usage_mb']:.1f} MB")
        
        return features, labels
    
    def create_train_test_splits(self, features: pd.DataFrame, labels: pd.DataFrame, 
                                test_size: float = 0.2, random_state: int = 42) -> Dict:
        """Create optimized train/test splits with memory efficiency"""
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        
        print("🎯 Creating optimized train/test splits...")
        
        # Encode labels efficiently
        le = LabelEncoder()
        y_encoded = le.fit_transform(labels.iloc[:, 0])
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            features, y_encoded, 
            test_size=test_size, 
            stratify=y_encoded,
            random_state=random_state
        )
        
        # Optimize memory for each split
        X_train = self.pipeline.optimize_dataframe_memory(X_train)
        X_test = self.pipeline.optimize_dataframe_memory(X_test)
        
        splits = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'label_encoder': le,
            'feature_names': list(features.columns)
        }
        
        print(f"✅ Splits created:")
        print(f"   📊 Train: {X_train.shape}")
        print(f"   📊 Test: {X_test.shape}")
        
        return splits

def main():
    """Demonstrate optimized data pipeline"""
    
    print("🚀 OPTIMIZED DATA PIPELINE DEMO")
    print("=" * 40)
    
    # Initialize dataset handler
    dataset = OptimizedExoplanetDataset()
    
    try:
        # Load and process data
        features, labels = dataset.load_exoplanet_data(enhanced=True)
        
        # Create train/test splits
        splits = dataset.create_train_test_splits(features, labels)
        
        # Get performance report
        performance = dataset.pipeline.get_performance_report()
        
        print("\n📊 PERFORMANCE REPORT")
        print("-" * 25)
        print(f"💾 Peak memory usage: {performance['memory_usage']['peak_mb']:.1f} MB")
        print(f"🎯 Cache hit rate: {performance['cache_efficiency']['hit_rate']:.2%}")
        print(f"⚡ Processing efficiency: OPTIMIZED")
        
        return splits, performance
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

if __name__ == "__main__":
    results = main()