"""
NASA Space Apps Challenge 2025 - Exoplanet Classifier
Core Module Initialization

This module provides the core functionality for the exoplanet classification system:
- Configuration management
- Optimized data loading with memory management
- Standardized prediction API with caching
- GPU acceleration support
"""

from .config import (
    # Project metadata
    PROJECT_NAME, PROJECT_VERSION, CHALLENGE_NAME, TEAM_CHALLENGE,
    
    # Directory paths
    BASE_DIR, DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR, CACHE_DIR,
    CORE_DIR, APP_DIR, API_DIR, DEPLOYMENT_DIR, TESTS_DIR,
    RAW_DATA_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR,
    PRODUCTION_MODELS_DIR, EXPERIMENTAL_MODELS_DIR, PREPROCESSORS_DIR,
    PERFORMANCE_DIR, VISUALIZATIONS_DIR, REPORTS_DIR,
    
    # Configuration instances
    DATA_CONFIG, MODEL_CONFIG, APP_CONFIG, LOG_CONFIG, GPU_CONFIG,
    
    # Feature definitions
    KEPLER_FEATURES, K2_FEATURES, TESS_FEATURES, ALL_FEATURES, TARGET_MAPPING,
    
    # Utility functions
    ensure_directories, setup_logging, get_device_info, validate_config
)

from .data_loader import (
    # Main classes
    ChunkedDataLoader, DataCache, MemoryMonitor,
    
    # Global instances and convenience functions
    get_data_loader, load_exoplanet_data
)

from .prediction import (
    # Main classes
    PredictionAPI, ModelCache, DataPreprocessor,
    
    # Global instances and convenience functions
    get_prediction_api, predict, predict_async, predict_ensemble,
    
    # Context manager
    prediction_session
)

# Version information
__version__ = PROJECT_VERSION
__author__ = "NASA Space Apps Challenge Team"
__description__ = "AI-powered exoplanet classification system for NASA Space Apps Challenge 2025"

# Initialize core system
def initialize_core_system():
    """Initialize the core system with all configurations"""
    try:
        # Validate configuration
        if not validate_config():
            raise RuntimeError("Core system configuration validation failed")
        
        # Ensure all directories exist
        ensure_directories()
        
        # Setup logging
        logger = setup_logging("core")
        logger.info(f"Core system initialized for {PROJECT_NAME} v{PROJECT_VERSION}")
        logger.info(f"Challenge: {CHALLENGE_NAME}")
        
        # Log device information
        device_info = get_device_info()
        logger.info(f"Device: {device_info['current_device']} "
                   f"({device_info['available_gpus']} GPUs available)")
        
        return True
        
    except Exception as e:
        print(f"Core system initialization failed: {e}")
        return False

# Lazy initialization
_system_initialized = False

def ensure_initialized():
    """Ensure core system is initialized"""
    global _system_initialized
    if not _system_initialized:
        _system_initialized = initialize_core_system()
    return _system_initialized

# Auto-initialize on import
ensure_initialized()

# Export main classes for easy access
__all__ = [
    # Project metadata
    'PROJECT_NAME', 'PROJECT_VERSION', 'CHALLENGE_NAME', 'TEAM_CHALLENGE',
    
    # Configuration classes
    'DATA_CONFIG', 'MODEL_CONFIG', 'APP_CONFIG', 'LOG_CONFIG', 'GPU_CONFIG',
    
    # Main functional classes
    'ChunkedDataLoader', 'PredictionAPI', 'DataCache', 'ModelCache',
    'MemoryMonitor', 'DataPreprocessor',
    
    # Convenience functions
    'get_data_loader', 'get_prediction_api', 'load_exoplanet_data',
    'predict', 'predict_async', 'predict_ensemble', 'prediction_session',
    
    # Utility functions
    'setup_logging', 'get_device_info', 'ensure_directories', 'validate_config',
    
    # Path constants
    'BASE_DIR', 'DATA_DIR', 'MODELS_DIR', 'RESULTS_DIR', 'CACHE_DIR',
    
    # Feature definitions
    'ALL_FEATURES', 'TARGET_MAPPING',
    
    # System functions
    'initialize_core_system', 'ensure_initialized'
]