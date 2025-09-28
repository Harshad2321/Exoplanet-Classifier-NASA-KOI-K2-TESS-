"""
NASA Space Apps Challenge 2025 - Exoplanet Classifier
Core Configuration and Constants
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import logging

# ================== PROJECT METADATA ==================
PROJECT_NAME = "Exoplanet-Classifier-NASA-KOI-K2-TESS"
PROJECT_VERSION = "2.0.0"
CHALLENGE_NAME = "NASA Space Apps Challenge 2025"
TEAM_CHALLENGE = "A World Away: Hunting for Exoplanets with AI"

# ================== DIRECTORY PATHS ==================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / ".cache"

# Core subdirectories
CORE_DIR = BASE_DIR / "core"
APP_DIR = BASE_DIR / "app"
API_DIR = BASE_DIR / "api"
DEPLOYMENT_DIR = BASE_DIR / "deployment"
TESTS_DIR = BASE_DIR / "tests"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"

# Model subdirectories
PRODUCTION_MODELS_DIR = MODELS_DIR / "production"
EXPERIMENTAL_MODELS_DIR = MODELS_DIR / "experimental"
PREPROCESSORS_DIR = MODELS_DIR / "preprocessors"

# Results subdirectories
PERFORMANCE_DIR = RESULTS_DIR / "performance"
VISUALIZATIONS_DIR = RESULTS_DIR / "visualizations"
REPORTS_DIR = RESULTS_DIR / "reports"

# ================== DATA CONFIGURATION ==================
@dataclass
class DataConfig:
    """Configuration for data processing and loading"""
    
    # Dataset parameters
    train_test_split: float = 0.2
    validation_split: float = 0.15
    random_state: int = 42
    stratify: bool = True
    
    # Memory optimization
    chunk_size: int = 10000
    max_memory_usage: str = "4GB"
    use_memory_mapping: bool = True
    
    # Feature engineering
    missing_threshold: float = 0.1
    correlation_threshold: float = 0.95
    variance_threshold: float = 0.01
    
    # Data validation
    outlier_method: str = "iqr"
    outlier_threshold: float = 3.0
    validate_schema: bool = True

# ================== MODEL CONFIGURATION ==================
@dataclass
class ModelConfig:
    """Configuration for model training and evaluation"""
    
    # Training parameters
    cv_folds: int = 5
    scoring_metric: str = "accuracy"
    optimization_metric: str = "roc_auc"
    early_stopping_rounds: int = 50
    
    # Model selection
    ensemble_methods: List[str] = None
    max_models_ensemble: int = 5
    meta_learner: str = "logistic_regression"
    
    # Performance thresholds
    min_accuracy: float = 0.65
    min_precision: float = 0.60
    min_recall: float = 0.60
    min_f1_score: float = 0.60
    
    # GPU configuration
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    mixed_precision: bool = True
    
    def __post_init__(self):
        if self.ensemble_methods is None:
            self.ensemble_methods = [
                "random_forest", "xgboost", "lightgbm", 
                "extra_trees", "gradient_boosting"
            ]

# ================== APPLICATION CONFIGURATION ==================
@dataclass
class AppConfig:
    """Configuration for web application and API"""
    
    # Streamlit settings
    page_title: str = "NASA Exoplanet Hunter"
    page_icon: str = "🌌"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    
    # API settings
    api_title: str = "NASA Exoplanet Classifier API"
    api_version: str = "v1"
    api_description: str = "AI-powered exoplanet classification system"
    max_request_size: int = 10_000_000  # 10MB
    rate_limit: str = "100/minute"
    
    # Caching
    cache_ttl: int = 3600  # 1 hour
    max_cache_size: str = "500MB"
    enable_model_caching: bool = True
    
    # Performance
    max_concurrent_predictions: int = 10
    prediction_timeout: int = 30  # seconds
    enable_async: bool = True

# ================== LOGGING CONFIGURATION ==================
@dataclass
class LogConfig:
    """Configuration for logging system"""
    
    # Basic settings
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # File logging
    log_to_file: bool = True
    log_rotation: str = "midnight"
    backup_count: int = 7
    max_file_size: str = "10MB"
    
    # Console logging
    log_to_console: bool = True
    console_level: str = "INFO"
    
    # Component-specific levels
    component_levels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.component_levels is None:
            self.component_levels = {
                "model": "DEBUG",
                "data": "INFO", 
                "api": "INFO",
                "app": "WARNING"
            }

# ================== FEATURE COLUMNS ==================
# NASA Kepler/K2/TESS feature columns
KEPLER_FEATURES = [
    'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration', 'koi_depth',
    'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_tce_plnt_num',
    'koi_steff', 'koi_slogg', 'koi_srad', 'ra', 'dec', 'koi_kepmag'
]

K2_FEATURES = [
    'k2_period', 'k2_time0', 'k2_impact', 'k2_duration', 'k2_depth',
    'k2_prad', 'k2_teq', 'k2_insol', 'k2_snr', 'k2_campaign',
    'k2_steff', 'k2_slogg', 'k2_srad', 'k2_ra', 'k2_dec', 'k2_kepmag'
]

TESS_FEATURES = [
    'tess_period', 'tess_time0', 'tess_impact', 'tess_duration', 'tess_depth',
    'tess_prad', 'tess_teq', 'tess_insol', 'tess_snr', 'tess_sector',
    'tess_steff', 'tess_slogg', 'tess_srad', 'tess_ra', 'tess_dec', 'tess_tmag'
]

# Combined feature set for unified model
ALL_FEATURES = list(set(KEPLER_FEATURES + K2_FEATURES + TESS_FEATURES))

# Target variable mapping
TARGET_MAPPING = {
    'CONFIRMED': 1,
    'CANDIDATE': 0,
    'FALSE POSITIVE': 0
}

# ================== GPU CONFIGURATION ==================
@dataclass
class GPUConfig:
    """Configuration for GPU acceleration"""
    
    # Detection
    auto_detect: bool = True
    preferred_device: str = "cuda"  # cuda, mps, cpu
    
    # Memory management
    memory_growth: bool = True
    memory_limit_mb: Optional[int] = None
    clear_cache_after_training: bool = True
    
    # Training optimization
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    
    # Model specific
    tensorflow_gpu_options: Dict[str, Any] = None
    pytorch_device_kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tensorflow_gpu_options is None:
            self.tensorflow_gpu_options = {
                'allow_growth': True,
                'memory_limit': 6144  # 6GB for RTX 4060
            }
        
        if self.pytorch_device_kwargs is None:
            self.pytorch_device_kwargs = {
                'non_blocking': True,
                'dtype': 'float16' if self.use_mixed_precision else 'float32'
            }

# ================== GLOBAL INSTANCES ==================
# Create global configuration instances
DATA_CONFIG = DataConfig()
MODEL_CONFIG = ModelConfig()
APP_CONFIG = AppConfig()
LOG_CONFIG = LogConfig()
GPU_CONFIG = GPUConfig()

# ================== UTILITY FUNCTIONS ==================
def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR,
        MODELS_DIR, PRODUCTION_MODELS_DIR, EXPERIMENTAL_MODELS_DIR, PREPROCESSORS_DIR,
        RESULTS_DIR, PERFORMANCE_DIR, VISUALIZATIONS_DIR, REPORTS_DIR,
        LOGS_DIR, CACHE_DIR,
        CORE_DIR, APP_DIR, API_DIR, DEPLOYMENT_DIR, TESTS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py for Python packages
        if directory.name in ['core', 'app', 'api', 'tests']:
            init_file = directory / "__init__.py"
            if not init_file.exists():
                init_file.touch()

def setup_logging(component: str = "main") -> logging.Logger:
    """Setup logging for a specific component"""
    logger = logging.getLogger(component)
    
    if not logger.handlers:
        # Console handler
        if LOG_CONFIG.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, LOG_CONFIG.console_level))
            console_formatter = logging.Formatter(LOG_CONFIG.format, LOG_CONFIG.date_format)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if LOG_CONFIG.log_to_file:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            log_file = LOGS_DIR / f"{component}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=LOG_CONFIG.backup_count
            )
            file_handler.setLevel(getattr(logging, LOG_CONFIG.level))
            file_formatter = logging.Formatter(LOG_CONFIG.format, LOG_CONFIG.date_format)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
    
    # Set component-specific level
    component_level = LOG_CONFIG.component_levels.get(component, LOG_CONFIG.level)
    logger.setLevel(getattr(logging, component_level))
    
    return logger

def get_device_info() -> Dict[str, Any]:
    """Get GPU/device information"""
    device_info = {
        'available_gpus': 0,
        'gpu_names': [],
        'current_device': 'cpu',
        'memory_total': 0,
        'memory_available': 0
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            device_info['available_gpus'] = torch.cuda.device_count()
            device_info['gpu_names'] = [torch.cuda.get_device_name(i) 
                                       for i in range(torch.cuda.device_count())]
            device_info['current_device'] = 'cuda'
            device_info['memory_total'] = torch.cuda.get_device_properties(0).total_memory
            device_info['memory_available'] = torch.cuda.memory_reserved(0)
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus and device_info['available_gpus'] == 0:
            device_info['available_gpus'] = len(gpus)
            device_info['current_device'] = 'gpu'
    except ImportError:
        pass
    
    return device_info

def validate_config() -> bool:
    """Validate configuration settings"""
    try:
        # Ensure directories
        ensure_directories()
        
        # Validate data config
        assert 0 < DATA_CONFIG.train_test_split < 1
        assert 0 < DATA_CONFIG.validation_split < 1
        assert DATA_CONFIG.chunk_size > 0
        
        # Validate model config
        assert MODEL_CONFIG.cv_folds > 1
        assert 0 < MODEL_CONFIG.min_accuracy <= 1
        
        # Validate app config
        assert APP_CONFIG.max_request_size > 0
        assert APP_CONFIG.cache_ttl > 0
        
        return True
    except Exception as e:
        logger = setup_logging("config")
        logger.error(f"Configuration validation failed: {e}")
        return False

# ================== INITIALIZATION ==================
if __name__ == "__main__":
    # Initialize configuration on import
    if validate_config():
        logger = setup_logging("config")
        logger.info(f"Configuration initialized successfully for {PROJECT_NAME} v{PROJECT_VERSION}")
        logger.info(f"GPU Device Info: {get_device_info()}")
    else:
        raise RuntimeError("Configuration validation failed")