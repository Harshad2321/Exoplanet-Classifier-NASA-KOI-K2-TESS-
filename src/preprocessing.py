"""
Data Preprocessing Module for NASA Exoplanet Classification

This module handles:
- Data cleaning and missing value imputation
- Feature engineering from astronomical parameters
- Data normalization and scaling
- Class balancing and train/test splitting

Author: NASA Space Apps Challenge 2025 Team
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import logging
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExoplanetPreprocessor:
    """
    Comprehensive preprocessing pipeline for exoplanet classification
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize preprocessor
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.splits_dir = self.data_dir / "splits"
        
        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize preprocessing components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = None
        self.feature_names = None
        self.target_mapping = None
        
        # Define feature mappings for different datasets
        self.feature_mappings = {
            'koi': {
                # Planetary parameters
                'period': 'koi_period',
                'period_err1': 'koi_period_err1',
                'period_err2': 'koi_period_err2', 
                'time0': 'koi_time0',
                'impact': 'koi_impact',
                'impact_err1': 'koi_impact_err1',
                'impact_err2': 'koi_impact_err2',
                'duration': 'koi_duration',
                'duration_err1': 'koi_duration_err1',
                'duration_err2': 'koi_duration_err2',
                'depth': 'koi_depth',
                'depth_err1': 'koi_depth_err1', 
                'depth_err2': 'koi_depth_err2',
                'prad': 'koi_prad',
                'prad_err1': 'koi_prad_err1',
                'prad_err2': 'koi_prad_err2',
                'sma': 'koi_sma',
                'sma_err1': 'koi_sma_err1',
                'sma_err2': 'koi_sma_err2',
                'teq': 'koi_teq',
                'teq_err1': 'koi_teq_err1',
                'teq_err2': 'koi_teq_err2',
                'insol': 'koi_insol',
                'insol_err1': 'koi_insol_err1',
                'insol_err2': 'koi_insol_err2',
                
                # Stellar parameters  
                'slogg': 'koi_slogg',
                'slogg_err1': 'koi_slogg_err1',
                'slogg_err2': 'koi_slogg_err2',
                'srad': 'koi_srad',
                'srad_err1': 'koi_srad_err1', 
                'srad_err2': 'koi_srad_err2',
                'smass': 'koi_smass',
                'smass_err1': 'koi_smass_err1',
                'smass_err2': 'koi_smass_err2',
                'steff': 'koi_steff',
                'steff_err1': 'koi_steff_err1',
                'steff_err2': 'koi_steff_err2',
                'smet': 'koi_smet',
                'smet_err1': 'koi_smet_err1',
                'smet_err2': 'koi_smet_err2',
                
                # Data quality
                'max_sngle_ev': 'koi_max_sngle_ev',
                'max_mult_ev': 'koi_max_mult_ev',
                'model_snr': 'koi_model_snr'
            }
        }
        
        # Standard class mappings
        self.standard_class_mapping = {
            # Confirmed planets
            'CONFIRMED': 'CONFIRMED',
            'CANDIDATE': 'CANDIDATE', 
            'FALSE POSITIVE': 'FALSE_POSITIVE',
            
            # Variations in naming
            'PC': 'CANDIDATE',  # Planetary Candidate
            'CP': 'CONFIRMED',  # Confirmed Planet
            'FP': 'FALSE_POSITIVE',  # False Positive
            'FA': 'FALSE_POSITIVE',  # False Alarm
        }
    
    def load_and_combine_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Load and combine multiple exoplanet datasets
        
        Args:
            datasets: Dictionary of dataset name -> DataFrame
            
        Returns:
            Combined and standardized DataFrame
        """
        combined_dfs = []
        
        for dataset_name, df in datasets.items():
            logger.info(f"Processing {dataset_name} dataset with {len(df)} records")
            
            # Add source column
            df_copy = df.copy()
            df_copy['source_dataset'] = dataset_name
            
            # Standardize target column based on dataset
            target_col = self._get_target_column(dataset_name)
            if target_col and target_col in df_copy.columns:
                df_copy['target'] = df_copy[target_col]
                
                # Clean and standardize target values
                df_copy['target'] = df_copy['target'].astype(str).str.upper().str.strip()
                df_copy['target'] = df_copy['target'].replace(self.standard_class_mapping)
                
                # Filter valid targets
                valid_targets = ['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE']
                df_copy = df_copy[df_copy['target'].isin(valid_targets)]
                
                logger.info(f"After filtering: {len(df_copy)} records with valid targets")
                combined_dfs.append(df_copy)
            else:
                logger.warning(f"Target column not found for {dataset_name}")
        
        if not combined_dfs:
            raise ValueError("No datasets with valid target columns found")
        
        # Combine all datasets
        combined_df = pd.concat(combined_dfs, ignore_index=True, sort=False)
        logger.info(f"Combined dataset: {len(combined_df)} total records")
        
        return combined_df
    
    def _get_target_column(self, dataset_name: str) -> Optional[str]:
        """Get target column name for dataset"""
        target_columns = {
            'koi': 'disposition',  # Updated to match NASA archive format
            'k2': 'disposition',   # Updated to match NASA archive format 
            'toi': 'disposition'   # Updated to match NASA archive format
        }
        return target_columns.get(dataset_name)
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and engineer features from raw data
        
        Args:
            df: Raw combined dataset
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature extraction...")
        
        # Start with core astronomical features that are commonly available
        feature_df = pd.DataFrame()
        
        # Helper function to find column variants
        def find_column(df, patterns):
            for pattern in patterns:
                matches = [col for col in df.columns if pattern.lower() in col.lower()]
                if matches:
                    return matches[0]
            return None
        
        # Orbital period (days)
        period_col = find_column(df, ['period', 'per'])
        if period_col:
            feature_df['orbital_period'] = pd.to_numeric(df[period_col], errors='coerce')
        
        # Transit duration (hours)  
        duration_col = find_column(df, ['duration', 'dur'])
        if duration_col:
            feature_df['transit_duration'] = pd.to_numeric(df[duration_col], errors='coerce')
        
        # Transit depth (ppm)
        depth_col = find_column(df, ['depth', 'dep'])
        if depth_col:
            feature_df['transit_depth'] = pd.to_numeric(df[depth_col], errors='coerce')
        
        # Planetary radius (Earth radii)
        prad_col = find_column(df, ['prad', 'rad', 'rp'])
        if prad_col:
            feature_df['planet_radius'] = pd.to_numeric(df[prad_col], errors='coerce')
        
        # Stellar temperature (K)
        teff_col = find_column(df, ['teff', 'temp', 'st_teff'])
        if teff_col:
            feature_df['stellar_temp'] = pd.to_numeric(df[teff_col], errors='coerce')
        
        # Stellar radius (Solar radii)
        srad_col = find_column(df, ['srad', 'st_rad'])
        if srad_col:
            feature_df['stellar_radius'] = pd.to_numeric(df[srad_col], errors='coerce')
        
        # Stellar mass (Solar masses)
        smass_col = find_column(df, ['smass', 'st_mass'])
        if smass_col:
            feature_df['stellar_mass'] = pd.to_numeric(df[smass_col], errors='coerce')
        
        # Stellar metallicity
        smet_col = find_column(df, ['smet', 'feh', 'st_met'])
        if smet_col:
            feature_df['stellar_metallicity'] = pd.to_numeric(df[smet_col], errors='coerce')
        
        # Semi-major axis (AU)
        sma_col = find_column(df, ['sma', 'a'])
        if sma_col:
            feature_df['semi_major_axis'] = pd.to_numeric(df[sma_col], errors='coerce')
        
        # Equilibrium temperature (K)
        teq_col = find_column(df, ['teq', 'temp_eq'])
        if teq_col:
            feature_df['equilibrium_temp'] = pd.to_numeric(df[teq_col], errors='coerce')
        
        # Impact parameter
        impact_col = find_column(df, ['impact', 'b'])
        if impact_col:
            feature_df['impact_parameter'] = pd.to_numeric(df[impact_col], errors='coerce')
        
        # Signal-to-noise ratio
        snr_col = find_column(df, ['snr', 'model_snr'])
        if snr_col:
            feature_df['signal_to_noise'] = pd.to_numeric(df[snr_col], errors='coerce')
        
        # Insolation flux (Earth flux)
        insol_col = find_column(df, ['insol', 'flux'])
        if insol_col:
            feature_df['insolation_flux'] = pd.to_numeric(df[insol_col], errors='coerce')
        
        # Add target and source info
        feature_df['target'] = df['target']
        feature_df['source_dataset'] = df['source_dataset']
        
        logger.info(f"Extracted {len(feature_df.columns)-2} base features")
        
        # Feature engineering - derived features
        feature_df = self._engineer_derived_features(feature_df)
        
        return feature_df
    
    def _engineer_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from base astronomical parameters
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with additional engineered features
        """
        logger.info("Engineering derived features...")
        
        # Transit-to-period ratio
        if 'transit_duration' in df.columns and 'orbital_period' in df.columns:
            df['duration_period_ratio'] = df['transit_duration'] / (df['orbital_period'] * 24)
        
        # Planet-to-star radius ratio
        if 'planet_radius' in df.columns and 'stellar_radius' in df.columns:
            df['radius_ratio'] = df['planet_radius'] / (df['stellar_radius'] * 109.2)  # Convert to Earth radii
        
        # Scaled planet radius (accounting for stellar size)
        if 'planet_radius' in df.columns and 'stellar_radius' in df.columns:
            df['scaled_planet_radius'] = df['planet_radius'] / df['stellar_radius']
        
        # Orbital velocity (circular assumption)
        if 'semi_major_axis' in df.columns and 'orbital_period' in df.columns:
            df['orbital_velocity'] = 2 * np.pi * df['semi_major_axis'] / (df['orbital_period'] / 365.25)
        
        # Stellar density proxy
        if 'stellar_mass' in df.columns and 'stellar_radius' in df.columns:
            df['stellar_density'] = df['stellar_mass'] / (df['stellar_radius'] ** 3)
        
        # Temperature contrast
        if 'equilibrium_temp' in df.columns and 'stellar_temp' in df.columns:
            df['temp_contrast'] = df['equilibrium_temp'] / df['stellar_temp']
        
        # Habitable zone indicator
        if 'equilibrium_temp' in df.columns:
            df['in_habitable_zone'] = ((df['equilibrium_temp'] >= 200) & 
                                     (df['equilibrium_temp'] <= 350)).astype(int)
        
        # Transit depth consistency check
        if 'transit_depth' in df.columns and 'planet_radius' in df.columns and 'stellar_radius' in df.columns:
            expected_depth = (df['planet_radius'] / (df['stellar_radius'] * 109.2)) ** 2 * 1e6
            df['depth_consistency'] = np.abs(df['transit_depth'] - expected_depth) / expected_depth
        
        # Signal strength indicators
        if 'signal_to_noise' in df.columns:
            df['high_snr'] = (df['signal_to_noise'] > 10).astype(int)
        
        if 'transit_depth' in df.columns:
            df['significant_depth'] = (df['transit_depth'] > 50).astype(int)  # > 50 ppm
        
        # Planetary characteristics
        if 'planet_radius' in df.columns:
            df['rocky_planet'] = (df['planet_radius'] < 2.0).astype(int)  # < 2 Earth radii
            df['super_earth'] = ((df['planet_radius'] >= 1.0) & 
                               (df['planet_radius'] <= 1.75)).astype(int)
            df['neptune_like'] = ((df['planet_radius'] > 2.0) & 
                                (df['planet_radius'] < 6.0)).astype(int)
            df['jupiter_like'] = (df['planet_radius'] >= 6.0).astype(int)
        
        # Stellar type indicators
        if 'stellar_temp' in df.columns:
            df['main_sequence'] = ((df['stellar_temp'] >= 3000) & 
                                 (df['stellar_temp'] <= 8000)).astype(int)
            df['solar_like'] = ((df['stellar_temp'] >= 5000) & 
                              (df['stellar_temp'] <= 6500)).astype(int)
        
        logger.info(f"Added {len([col for col in df.columns if col not in ['target', 'source_dataset']])} total features")
        
        return df
    
    def clean_data(self, df: pd.DataFrame, missing_threshold: float = 0.7) -> pd.DataFrame:
        """
        Clean data by handling missing values and outliers
        
        Args:
            df: Input DataFrame
            missing_threshold: Drop columns with missing values above this fraction
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        
        initial_shape = df.shape
        feature_cols = [col for col in df.columns if col not in ['target', 'source_dataset']]
        
        # Remove columns with too many missing values
        missing_pcts = df[feature_cols].isnull().sum() / len(df)
        cols_to_drop = missing_pcts[missing_pcts > missing_threshold].index.tolist()
        
        if cols_to_drop:
            logger.info(f"Dropping {len(cols_to_drop)} columns with >{missing_threshold*100}% missing values")
            df = df.drop(columns=cols_to_drop)
        
        # Remove rows with all missing features
        feature_cols = [col for col in df.columns if col not in ['target', 'source_dataset']]
        df = df.dropna(subset=feature_cols, how='all')
        
        # Handle infinite values
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Remove obvious outliers using IQR method for key features
        key_features = ['orbital_period', 'planet_radius', 'stellar_temp', 'transit_depth']
        for feature in key_features:
            if feature in df.columns:
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR  # Using 3*IQR for less aggressive filtering
                upper_bound = Q3 + 3 * IQR
                
                outliers = ((df[feature] < lower_bound) | (df[feature] > upper_bound))
                outlier_count = outliers.sum()
                if outlier_count > 0:
                    logger.info(f"Removing {outlier_count} outliers from {feature}")
                    df = df[~outliers]
        
        logger.info(f"Data cleaning: {initial_shape} -> {df.shape}")
        return df
    
    def impute_missing_values(self, X: pd.DataFrame, method: str = 'median') -> pd.DataFrame:
        """
        Impute missing values using specified method
        
        Args:
            X: Feature DataFrame
            method: Imputation method ('mean', 'median', 'mode', 'knn')
            
        Returns:
            DataFrame with imputed values
        """
        logger.info(f"Imputing missing values using {method} method...")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        
        X_imputed = X.copy()
        
        # Handle numeric columns
        if len(numeric_cols) > 0:
            if method in ['mean', 'median']:
                self.imputer = SimpleImputer(strategy=method)
                X_imputed[numeric_cols] = self.imputer.fit_transform(X_imputed[numeric_cols])
            elif method == 'knn':
                self.imputer = KNNImputer(n_neighbors=5)
                X_imputed[numeric_cols] = self.imputer.fit_transform(X_imputed[numeric_cols])
        
        # Handle categorical columns  
        if len(categorical_cols) > 0:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            X_imputed[categorical_cols] = categorical_imputer.fit_transform(X_imputed[categorical_cols])
        
        missing_after = X_imputed.isnull().sum().sum()
        logger.info(f"Missing values after imputation: {missing_after}")
        
        return X_imputed
    
    def prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target, encode target labels
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['target', 'source_dataset']]
        X = df[feature_cols].copy()
        y = df['target'].copy()
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Store target mapping for later reference
        self.target_mapping = dict(zip(
            self.label_encoder.classes_, 
            self.label_encoder.transform(self.label_encoder.classes_)
        ))
        
        logger.info(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
        logger.info(f"Encoded mapping: {self.target_mapping}")
        
        return X, pd.Series(y_encoded, index=X.index)
    
    def split_and_balance_data(self, X: pd.DataFrame, y: pd.Series, 
                              test_size: float = 0.2, balance_method: str = 'smote') -> Dict:
        """
        Split data and apply class balancing
        
        Args:
            X: Features DataFrame
            y: Target Series (encoded)
            test_size: Fraction for test set
            balance_method: Balancing method ('smote', 'undersample', 'none')
            
        Returns:
            Dictionary with train/test splits
        """
        logger.info("Splitting and balancing data...")
        
        # Initial train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Initial split: Train={len(X_train)}, Test={len(X_test)}")
        logger.info(f"Train class distribution: {y_train.value_counts().to_dict()}")
        
        # Apply balancing to training set only
        if balance_method == 'smote':
            balancer = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = balancer.fit_resample(X_train, y_train)
        elif balance_method == 'undersample':
            balancer = RandomUnderSampler(random_state=42)
            X_train_balanced, y_train_balanced = balancer.fit_resample(X_train, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        if balance_method != 'none':
            logger.info(f"After balancing: {pd.Series(y_train_balanced).value_counts().to_dict()}")
        
        return {
            'X_train': X_train_balanced,
            'X_test': X_test,
            'y_train': y_train_balanced,
            'y_test': y_test,
            'feature_names': X.columns.tolist()
        }
    
    def scale_features(self, data_splits: Dict) -> Dict:
        """
        Scale features using StandardScaler
        
        Args:
            data_splits: Dictionary with train/test data
            
        Returns:
            Dictionary with scaled data
        """
        logger.info("Scaling features...")
        
        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(data_splits['X_train'])
        X_test_scaled = self.scaler.transform(data_splits['X_test'])
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(
            X_train_scaled, 
            columns=data_splits['feature_names'],
            index=data_splits['X_train'].index
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled,
            columns=data_splits['feature_names'], 
            index=data_splits['X_test'].index
        )
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': data_splits['y_train'],
            'y_test': data_splits['y_test'],
            'feature_names': data_splits['feature_names']
        }
    
    def save_processed_data(self, data_splits: Dict, processed_df: pd.DataFrame = None):
        """
        Save processed data and preprocessing components
        
        Args:
            data_splits: Dictionary with processed train/test data
            processed_df: Full processed DataFrame (optional)
        """
        logger.info("Saving processed data...")
        
        # Save train/test splits
        pd.concat([data_splits['X_train'], data_splits['y_train']], axis=1).to_csv(
            self.splits_dir / 'train.csv', index=False
        )
        pd.concat([data_splits['X_test'], data_splits['y_test']], axis=1).to_csv(
            self.splits_dir / 'test.csv', index=False
        )
        
        # Save full processed dataset if provided
        if processed_df is not None:
            processed_df.to_csv(self.processed_dir / 'processed_data.csv', index=False)
        
        # Save preprocessing components
        joblib.dump(self.scaler, self.processed_dir / 'scaler.pkl')
        joblib.dump(self.label_encoder, self.processed_dir / 'label_encoder.pkl')
        joblib.dump(data_splits['feature_names'], self.processed_dir / 'feature_names.pkl')
        joblib.dump(self.target_mapping, self.processed_dir / 'target_mapping.pkl')
        
        if self.imputer:
            joblib.dump(self.imputer, self.processed_dir / 'imputer.pkl')
        
        logger.info("✅ Processed data saved successfully")
    
    def load_preprocessing_components(self):
        """Load saved preprocessing components"""
        try:
            self.scaler = joblib.load(self.processed_dir / 'scaler.pkl')
            self.label_encoder = joblib.load(self.processed_dir / 'label_encoder.pkl')
            self.feature_names = joblib.load(self.processed_dir / 'feature_names.pkl')
            self.target_mapping = joblib.load(self.processed_dir / 'target_mapping.pkl')
            
            if (self.processed_dir / 'imputer.pkl').exists():
                self.imputer = joblib.load(self.processed_dir / 'imputer.pkl')
            
            logger.info("✅ Preprocessing components loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load preprocessing components: {e}")
            return False
    
    def full_preprocessing_pipeline(self, datasets: Dict[str, pd.DataFrame], 
                                   missing_threshold: float = 0.7,
                                   imputation_method: str = 'median',
                                   balance_method: str = 'smote',
                                   test_size: float = 0.2) -> Dict:
        """
        Complete preprocessing pipeline
        
        Args:
            datasets: Dictionary of dataset name -> DataFrame
            missing_threshold: Threshold for dropping columns with missing values
            imputation_method: Method for missing value imputation
            balance_method: Method for class balancing
            test_size: Fraction for test set
            
        Returns:
            Dictionary with processed train/test data
        """
        logger.info("🚀 Starting full preprocessing pipeline...")
        
        # Step 1: Load and combine datasets
        combined_df = self.load_and_combine_datasets(datasets)
        
        # Step 2: Extract and engineer features
        feature_df = self.extract_features(combined_df)
        
        # Step 3: Clean data
        cleaned_df = self.clean_data(feature_df, missing_threshold)
        
        # Step 4: Prepare features and target
        X, y = self.prepare_features_and_target(cleaned_df)
        
        # Step 5: Impute missing values
        X_imputed = self.impute_missing_values(X, imputation_method)
        
        # Step 6: Split and balance data
        data_splits = self.split_and_balance_data(X_imputed, y, test_size, balance_method)
        
        # Step 7: Scale features
        final_data = self.scale_features(data_splits)
        
        # Step 8: Save processed data
        self.save_processed_data(final_data, cleaned_df)
        
        logger.info("✅ Preprocessing pipeline completed successfully!")
        
        return final_data


def main():
    """
    Main function to demonstrate preprocessing functionality
    """
    print("🧪 NASA Exoplanet Data Preprocessing")
    print("=" * 50)
    
    # This would typically be called with actual data
    # preprocessor = ExoplanetPreprocessor()
    # datasets = {...}  # Load from data_loader
    # processed_data = preprocessor.full_preprocessing_pipeline(datasets)
    
    print("Preprocessing module ready!")
    print("Use ExoplanetPreprocessor.full_preprocessing_pipeline() to process your data")


if __name__ == "__main__":
    main()