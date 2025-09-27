"""
Utility Functions for NASA Exoplanet Classification Project

This module contains helper functions and utilities used across the project:
- Data validation and quality checks
- Configuration management
- Logging setup
- Common transformations
- File I/O helpers

Author: NASA Space Apps Challenge 2025 Team
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import joblib
import json
from datetime import datetime
import warnings
import os

# Configure warnings
warnings.filterwarnings('ignore')

def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger('exoplanet_classifier')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def validate_dataframe(df: pd.DataFrame, 
                      required_columns: List[str] = None,
                      min_rows: int = 1) -> Dict[str, Any]:
    """
    Validate DataFrame structure and content
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        
    Returns:
        Validation results dictionary
    """
    results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Basic checks
    if df is None:
        results['is_valid'] = False
        results['errors'].append("DataFrame is None")
        return results
    
    if df.empty:
        results['is_valid'] = False
        results['errors'].append("DataFrame is empty")
        return results
    
    if len(df) < min_rows:
        results['is_valid'] = False
        results['errors'].append(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")
    
    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            results['is_valid'] = False
            results['errors'].append(f"Missing required columns: {list(missing_cols)}")
    
    # Generate statistics
    results['stats'] = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
    }
    
    # Check for high missing value columns
    high_missing = [col for col, pct in results['stats']['missing_percentage'].items() if pct > 50]
    if high_missing:
        results['warnings'].append(f"Columns with >50% missing values: {high_missing}")
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        results['warnings'].append(f"Found {duplicates} duplicate rows")
    
    return results

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize column names
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with cleaned column names
    """
    df_clean = df.copy()
    
    # Convert to lowercase and replace spaces/special characters
    df_clean.columns = (df_clean.columns
                       .str.lower()
                       .str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
                       .str.replace(r'_+', '_', regex=True)
                       .str.strip('_'))
    
    return df_clean

def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """
    Detect outliers using Interquartile Range (IQR) method
    
    Args:
        series: Pandas Series to analyze
        k: IQR multiplier (1.5 = mild outliers, 3.0 = extreme outliers)
        
    Returns:
        Boolean Series indicating outliers
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    
    return (series < lower_bound) | (series > upper_bound)

def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers using Z-score method
    
    Args:
        series: Pandas Series to analyze
        threshold: Z-score threshold for outlier detection
        
    Returns:
        Boolean Series indicating outliers
    """
    z_scores = np.abs((series - series.mean()) / series.std())
    return z_scores > threshold

def calculate_data_quality_score(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate overall data quality score
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with quality metrics
    """
    scores = {}
    
    # Completeness (1 - missing value ratio)
    missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    scores['completeness'] = 1 - missing_ratio
    
    # Uniqueness (based on duplicate rows)
    duplicate_ratio = df.duplicated().sum() / len(df)
    scores['uniqueness'] = 1 - duplicate_ratio
    
    # Consistency (based on data type consistency)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    consistency_issues = 0
    
    for col in numeric_cols:
        # Check for infinite values
        if np.isinf(df[col]).any():
            consistency_issues += 1
        
        # Check for extreme outliers (Z-score > 5)
        outliers = detect_outliers_zscore(df[col], threshold=5)
        if outliers.sum() / len(df) > 0.05:  # >5% outliers
            consistency_issues += 1
    
    scores['consistency'] = 1 - (consistency_issues / len(numeric_cols)) if len(numeric_cols) > 0 else 1.0
    
    # Overall score (weighted average)
    scores['overall'] = (scores['completeness'] * 0.4 + 
                        scores['uniqueness'] * 0.3 + 
                        scores['consistency'] * 0.3)
    
    return scores

def save_model_artifact(obj: Any, filepath: str, metadata: Dict = None):
    """
    Save model artifact with metadata
    
    Args:
        obj: Object to save
        filepath: Path to save file
        metadata: Optional metadata dictionary
    """
    # Save main object
    joblib.dump(obj, filepath)
    
    # Save metadata if provided
    if metadata:
        metadata_path = str(filepath).replace('.pkl', '_metadata.json')
        metadata['saved_at'] = datetime.now().isoformat()
        metadata['file_size_mb'] = os.path.getsize(filepath) / (1024 * 1024)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

def load_model_artifact(filepath: str) -> Tuple[Any, Dict]:
    """
    Load model artifact with metadata
    
    Args:
        filepath: Path to load file from
        
    Returns:
        Tuple of (loaded object, metadata dict)
    """
    # Load main object
    obj = joblib.load(filepath)
    
    # Load metadata if exists
    metadata_path = str(filepath).replace('.pkl', '_metadata.json')
    metadata = {}
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return obj, metadata

def create_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive feature summary
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Summary DataFrame
    """
    summary_data = []
    
    for col in df.columns:
        col_data = {
            'feature': col,
            'dtype': str(df[col].dtype),
            'missing_count': df[col].isnull().sum(),
            'missing_percentage': df[col].isnull().sum() / len(df) * 100,
            'unique_count': df[col].nunique(),
            'unique_percentage': df[col].nunique() / len(df) * 100
        }
        
        if df[col].dtype in ['int64', 'float64']:
            # Numeric column statistics
            col_data.update({
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'q25': df[col].quantile(0.25),
                'median': df[col].median(),
                'q75': df[col].quantile(0.75),
                'max': df[col].max(),
                'outliers_iqr': detect_outliers_iqr(df[col]).sum(),
                'outliers_zscore': detect_outliers_zscore(df[col]).sum()
            })
        else:
            # Categorical column statistics
            col_data.update({
                'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'most_frequent_count': df[col].value_counts().iloc[0] if not df[col].empty else 0,
                'mean': None,
                'std': None,
                'min': None,
                'q25': None,
                'median': None,
                'q75': None,
                'max': None,
                'outliers_iqr': None,
                'outliers_zscore': None
            })
        
        summary_data.append(col_data)
    
    return pd.DataFrame(summary_data)

def compare_datasets(df1: pd.DataFrame, df2: pd.DataFrame, 
                    name1: str = "Dataset 1", name2: str = "Dataset 2") -> Dict:
    """
    Compare two datasets and highlight differences
    
    Args:
        df1: First dataset
        df2: Second dataset
        name1: Name of first dataset
        name2: Name of second dataset
        
    Returns:
        Comparison results dictionary
    """
    comparison = {
        'basic_stats': {
            name1: {'rows': len(df1), 'columns': len(df1.columns)},
            name2: {'rows': len(df2), 'columns': len(df2.columns)}
        },
        'common_columns': list(set(df1.columns) & set(df2.columns)),
        'unique_to_dataset1': list(set(df1.columns) - set(df2.columns)),
        'unique_to_dataset2': list(set(df2.columns) - set(df1.columns))
    }
    
    # Compare common columns
    if comparison['common_columns']:
        column_comparison = []
        
        for col in comparison['common_columns']:
            col_comp = {
                'column': col,
                'dtype_match': df1[col].dtype == df2[col].dtype,
                'missing_values': {
                    name1: df1[col].isnull().sum(),
                    name2: df2[col].isnull().sum()
                }
            }
            
            if df1[col].dtype in ['int64', 'float64'] and df2[col].dtype in ['int64', 'float64']:
                col_comp['stats_comparison'] = {
                    'mean_diff': abs(df1[col].mean() - df2[col].mean()),
                    'std_diff': abs(df1[col].std() - df2[col].std())
                }
            
            column_comparison.append(col_comp)
        
        comparison['column_comparison'] = column_comparison
    
    return comparison

def validate_model_inputs(X: pd.DataFrame, feature_names: List[str]) -> Dict:
    """
    Validate model input data against expected features
    
    Args:
        X: Input features DataFrame
        feature_names: Expected feature names
        
    Returns:
        Validation results
    """
    results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check for missing features
    missing_features = set(feature_names) - set(X.columns)
    if missing_features:
        results['is_valid'] = False
        results['errors'].append(f"Missing features: {list(missing_features)}")
    
    # Check for extra features
    extra_features = set(X.columns) - set(feature_names)
    if extra_features:
        results['warnings'].append(f"Extra features will be ignored: {list(extra_features)}")
    
    # Check for missing values
    if X.isnull().any().any():
        missing_cols = X.columns[X.isnull().any()].tolist()
        results['errors'].append(f"Missing values found in columns: {missing_cols}")
        results['is_valid'] = False
    
    # Check for infinite values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(X[col]).any():
            results['errors'].append(f"Infinite values found in column: {col}")
            results['is_valid'] = False
    
    return results

def create_project_structure(base_dir: str = "exoplanet_classifier") -> bool:
    """
    Create standard project directory structure
    
    Args:
        base_dir: Base directory name
        
    Returns:
        Success status
    """
    try:
        base_path = Path(base_dir)
        
        directories = [
            "data/raw",
            "data/processed", 
            "data/splits",
            "notebooks",
            "src",
            "models",
            "logs",
            "outputs"
        ]
        
        for directory in directories:
            (base_path / directory).mkdir(parents=True, exist_ok=True)
        
        return True
    except Exception as e:
        print(f"Failed to create project structure: {e}")
        return False

def print_dataset_summary(df: pd.DataFrame, name: str = "Dataset"):
    """
    Print comprehensive dataset summary
    
    Args:
        df: DataFrame to summarize
        name: Dataset name for display
    """
    print(f"\n{'='*50}")
    print(f"{name} Summary")
    print(f"{'='*50}")
    
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    
    # Data types
    print(f"\nData Types:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # Missing values
    total_missing = df.isnull().sum().sum()
    missing_pct = total_missing / (df.shape[0] * df.shape[1]) * 100
    print(f"\nMissing Values: {total_missing:,} ({missing_pct:.2f}%)")
    
    if total_missing > 0:
        missing_by_col = df.isnull().sum()
        missing_by_col = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
        print("  Top columns with missing values:")
        for col, count in missing_by_col.head().items():
            pct = count / len(df) * 100
            print(f"    {col}: {count:,} ({pct:.1f}%)")
    
    # Data quality score
    quality_scores = calculate_data_quality_score(df)
    print(f"\nData Quality Score: {quality_scores['overall']:.2f}/1.00")
    print(f"  Completeness: {quality_scores['completeness']:.2f}")
    print(f"  Uniqueness: {quality_scores['uniqueness']:.2f}")
    print(f"  Consistency: {quality_scores['consistency']:.2f}")


# Configuration management functions
class Config:
    """Configuration management class"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            return self.get_default_config()
    
    def save_config(self):
        """Save current configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'data': {
                'missing_threshold': 0.7,
                'imputation_method': 'median',
                'balance_method': 'smote',
                'test_size': 0.2,
                'random_state': 42
            },
            'models': {
                'tune_hyperparameters': True,
                'cv_folds': 5,
                'n_trials': 100
            },
            'logging': {
                'level': 'INFO',
                'save_logs': True
            },
            'paths': {
                'data_dir': 'data',
                'models_dir': 'models',
                'logs_dir': 'logs'
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value


if __name__ == "__main__":
    print("🔧 NASA Exoplanet Classification Utilities")
    print("=" * 50)
    print("This module contains utility functions for the project.")
    print("Import and use individual functions as needed.")