"""
Data Preprocessing Pipeline for NASA Space Apps Challenge 2025
"A World Away: Hunting for Exoplanets with AI"

This module handles data preprocessing for the exoplanet classification task,
including data loading, cleaning, feature engineering, and preparation for ML models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExoplanetDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for NASA exoplanet datasets.
    
    Handles data loading, cleaning, normalization, and feature engineering
    for Kepler, K2, and TESS datasets.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the preprocessor.
        
        Args:
            data_dir: Directory containing the NASA dataset files
        """
        self.data_dir = Path(data_dir)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = None
        
        # Dataset configuration
        self.dataset_configs = {
            'kepler': {
                'file': 'raw/koi.csv',
                'target_col': 'koi_disposition',
                'id_col': 'kepid',
                'key_features': [
                    'koi_period', 'koi_prad', 'koi_teq', 'koi_insol',
                    'koi_dor', 'koi_duration', 'koi_depth', 'koi_impact',
                    'ra', 'dec', 'koi_kepmag'
                ]
            },
            'k2': {
                'file': 'raw/k2.csv', 
                'target_col': 'disposition',
                'id_col': 'epic_name',
                'key_features': [
                    'period', 'prad', 'teq', 'insol', 'dor',
                    'duration', 'depth', 'impact', 'ra', 'dec', 'kepmag'
                ]
            },
            'tess': {
                'file': 'raw/toi.csv',
                'target_col': 'tfopwg_disp',  
                'id_col': 'tic_id',
                'key_features': [
                    'pl_orbper', 'pl_rade', 'pl_eqt', 'pl_insol',
                    'pl_ratdor', 'pl_trandur', 'pl_trandep', 'pl_imppar',
                    'ra', 'dec', 'sy_tmag'
                ]
            }
        }
        
        # Label normalization mapping
        self.label_mapping = {
            # Confirmed planets
            'CONFIRMED': 'CONFIRMED',
            'confirmed': 'CONFIRMED', 
            'PLANET': 'CONFIRMED',
            'planet': 'CONFIRMED',
            'PC': 'CONFIRMED',
            
            # Candidates
            'CANDIDATE': 'CANDIDATE',
            'candidate': 'CANDIDATE',
            'CAND': 'CANDIDATE',
            'KOI': 'CANDIDATE',
            
            # False positives
            'FALSE POSITIVE': 'FALSE_POSITIVE',
            'false positive': 'FALSE_POSITIVE',
            'FALSE_POSITIVE': 'FALSE_POSITIVE',
            'FP': 'FALSE_POSITIVE',
            'EB': 'FALSE_POSITIVE',  # Eclipsing binary
            'V*': 'FALSE_POSITIVE',  # Variable star
            'NTP': 'FALSE_POSITIVE', # No Transit-like Planet
        }
        
    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all NASA exoplanet datasets.
        
        Returns:
            Dictionary containing loaded datasets
        """
        datasets = {}
        
        for name, config in self.dataset_configs.items():
            file_path = self.data_dir / config['file']
            
            try:
                if file_path.exists():
                    df = pd.read_csv(file_path, comment='#', low_memory=False)
                    logger.info(f"✅ Loaded {name.upper()}: {len(df):,} objects, {len(df.columns)} features")
                    datasets[name] = df
                else:
                    logger.warning(f"❌ File not found: {file_path}")
                    datasets[name] = pd.DataFrame()
                    
            except Exception as e:
                logger.error(f"❌ Error loading {name}: {e}")
                datasets[name] = pd.DataFrame()
        
        return datasets
    
    def normalize_labels(self, df: pd.DataFrame, target_col: str) -> pd.Series:
        """
        Normalize class labels across datasets.
        
        Args:
            df: DataFrame containing the target column
            target_col: Name of the target column
            
        Returns:
            Normalized labels
        """
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found")
            return pd.Series(dtype='object')
        
        labels = df[target_col].astype(str).str.strip()
        normalized = labels.map(self.label_mapping)
        
        # Handle unmapped labels
        unmapped = normalized.isna()
        if unmapped.sum() > 0:
            unique_unmapped = labels[unmapped].unique()
            logger.warning(f"Unmapped labels: {list(unique_unmapped)}")
            
            # Try to infer based on keywords
            for label in unique_unmapped:
                label_lower = label.lower()
                if any(kw in label_lower for kw in ['confirm', 'planet']):
                    normalized.loc[labels == label] = 'CONFIRMED'
                elif any(kw in label_lower for kw in ['candidate', 'cand']):
                    normalized.loc[labels == label] = 'CANDIDATE'
                elif any(kw in label_lower for kw in ['false', 'fp', 'eb', 'binary']):
                    normalized.loc[labels == label] = 'FALSE_POSITIVE'
                else:
                    # Default to candidate for unknown
                    normalized.loc[labels == label] = 'CANDIDATE'
        
        return normalized
    
    def select_common_features(self, datasets: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Identify and select common features across datasets.
        
        Args:
            datasets: Dictionary of datasets
            
        Returns:
            List of common feature names
        """
        feature_mapping = {
            # Orbital period
            'period': ['koi_period', 'period', 'pl_orbper'],
            # Planet radius  
            'radius': ['koi_prad', 'prad', 'pl_rade'],
            # Equilibrium temperature
            'temperature': ['koi_teq', 'teq', 'pl_eqt'],
            # Insolation flux
            'insolation': ['koi_insol', 'insol', 'pl_insol'],
            # Semi-major axis ratio
            'a_over_rstar': ['koi_dor', 'dor', 'pl_ratdor'],
            # Transit duration
            'duration': ['koi_duration', 'duration', 'pl_trandur'],
            # Transit depth
            'depth': ['koi_depth', 'depth', 'pl_trandep'],
            # Impact parameter
            'impact': ['koi_impact', 'impact', 'pl_imppar'],
            # Coordinates
            'ra': ['ra', 'ra', 'ra'],
            'dec': ['dec', 'dec', 'dec'],
            # Stellar magnitude
            'magnitude': ['koi_kepmag', 'kepmag', 'sy_tmag']
        }
        
        common_features = []
        available_datasets = [name for name, df in datasets.items() if not df.empty]
        
        for feature_name, column_names in feature_mapping.items():
            availability = []
            
            for i, dataset_name in enumerate(['kepler', 'k2', 'tess']):
                if dataset_name in available_datasets:
                    col_name = column_names[i]
                    if col_name in datasets[dataset_name].columns:
                        availability.append(True)
                    else:
                        availability.append(False)
                else:
                    availability.append(False)
            
            # Include feature if available in at least 2 datasets
            if sum(availability) >= 2:
                common_features.append(feature_name)
        
        logger.info(f"Selected {len(common_features)} common features: {common_features}")
        return common_features
    
    def engineer_features(self, df: pd.DataFrame, feature_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Engineer additional features from the base measurements.
        
        Args:
            df: Input DataFrame
            feature_mapping: Mapping of common features to dataset-specific columns
            
        Returns:
            DataFrame with engineered features
        """
        engineered_df = df.copy()
        
        # Create derived features
        if 'period' in feature_mapping and 'radius' in feature_mapping:
            period_col = feature_mapping['period']
            radius_col = feature_mapping['radius']
            
            if period_col in df.columns and radius_col in df.columns:
                # Planet density proxy (radius^3 / period^2)
                engineered_df['density_proxy'] = (
                    df[radius_col] ** 3 / df[period_col] ** 2
                )
        
        if 'temperature' in feature_mapping and 'insolation' in feature_mapping:
            temp_col = feature_mapping['temperature']
            insol_col = feature_mapping['insolation']
            
            if temp_col in df.columns and insol_col in df.columns:
                # Habitability proxy (temperature range and insolation)
                temp_mask = (df[temp_col] >= 200) & (df[temp_col] <= 400)
                insol_mask = (df[insol_col] >= 0.5) & (df[insol_col] <= 2.0)
                engineered_df['habitability_proxy'] = (temp_mask & insol_mask).astype(int)
        
        if 'duration' in feature_mapping and 'period' in feature_mapping:
            duration_col = feature_mapping['duration']
            period_col = feature_mapping['period']
            
            if duration_col in df.columns and period_col in df.columns:
                # Transit duty cycle (duration / period)
                engineered_df['duty_cycle'] = df[duration_col] / df[period_col]
        
        return engineered_df
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'knn') -> pd.DataFrame:
        """
        Handle missing values using specified strategy.
        
        Args:
            df: Input DataFrame
            strategy: Imputation strategy ('mean', 'median', 'knn')
            
        Returns:
            DataFrame with imputed values
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return df
        
        df_imputed = df.copy()
        
        if strategy == 'knn':
            self.imputer = KNNImputer(n_neighbors=5)
        else:
            self.imputer = SimpleImputer(strategy=strategy)
        
        try:
            df_imputed[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
            logger.info(f"✅ Applied {strategy} imputation to {len(numeric_cols)} features")
        except Exception as e:
            logger.warning(f"❌ Imputation failed: {e}. Using median fallback.")
            self.imputer = SimpleImputer(strategy='median')
            df_imputed[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        
        return df_imputed
    
    def remove_outliers(self, df: pd.DataFrame, columns: List[str], 
                       method: str = 'iqr', factor: float = 1.5) -> pd.DataFrame:
        """
        Remove statistical outliers from specified columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            method: Outlier detection method ('iqr' or 'zscore')
            factor: Multiplier for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        for col in columns:
            if col not in df_clean.columns:
                continue
                
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                df_clean = df_clean[mask | df_clean[col].isna()]
                
            elif method == 'zscore':
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                mask = z_scores <= factor
                df_clean = df_clean[mask | df_clean[col].isna()]
        
        removed_rows = initial_rows - len(df_clean)
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows:,} outlier rows ({removed_rows/initial_rows*100:.1f}%)")
        
        return df_clean
    
    def prepare_dataset(self, dataset_name: str, df: pd.DataFrame,
                       common_features: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare a single dataset for machine learning.
        
        Args:
            dataset_name: Name of the dataset
            df: Input DataFrame
            common_features: List of common features to extract
            
        Returns:
            Tuple of (features_df, labels_series)
        """
        config = self.dataset_configs[dataset_name]
        
        # Create feature mapping for this dataset
        feature_mapping = {}
        feature_map_dict = {
            'period': ['koi_period', 'period', 'pl_orbper'],
            'radius': ['koi_prad', 'prad', 'pl_rade'], 
            'temperature': ['koi_teq', 'teq', 'pl_eqt'],
            'insolation': ['koi_insol', 'insol', 'pl_insol'],
            'a_over_rstar': ['koi_dor', 'dor', 'pl_ratdor'],
            'duration': ['koi_duration', 'duration', 'pl_trandur'],
            'depth': ['koi_depth', 'depth', 'pl_trandep'],
            'impact': ['koi_impact', 'impact', 'pl_imppar'],
            'ra': ['ra', 'ra', 'ra'],
            'dec': ['dec', 'dec', 'dec'],
            'magnitude': ['koi_kepmag', 'kepmag', 'sy_tmag']
        }
        
        dataset_idx = ['kepler', 'k2', 'tess'].index(dataset_name)
        
        for common_feat in common_features:
            if common_feat in feature_map_dict:
                dataset_col = feature_map_dict[common_feat][dataset_idx]
                if dataset_col in df.columns:
                    feature_mapping[common_feat] = dataset_col
        
        # Extract and rename features
        features_df = pd.DataFrame()
        for common_name, original_name in feature_mapping.items():
            features_df[common_name] = df[original_name]
        
        # Engineer additional features
        features_df = self.engineer_features(features_df, feature_mapping)
        
        # Get normalized labels
        labels = self.normalize_labels(df, config['target_col'])
        
        # Filter out rows with missing labels
        valid_labels = labels.notna()
        features_df = features_df[valid_labels]
        labels = labels[valid_labels]
        
        logger.info(f"✅ Prepared {dataset_name.upper()}: {len(features_df)} samples, {len(features_df.columns)} features")
        
        return features_df, labels
    
    def combine_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Combine all datasets into a unified training set.
        
        Args:
            datasets: Dictionary of loaded datasets
            
        Returns:
            Tuple of (combined_features, combined_labels, dataset_sources)
        """
        common_features = self.select_common_features(datasets)
        
        all_features = []
        all_labels = []
        all_sources = []
        
        for dataset_name, df in datasets.items():
            if df.empty:
                continue
                
            features, labels = self.prepare_dataset(dataset_name, df, common_features)
            
            if len(features) > 0:
                all_features.append(features)
                all_labels.append(labels)
                all_sources.extend([dataset_name] * len(features))
        
        if not all_features:
            logger.error("❌ No valid datasets to combine")
            return pd.DataFrame(), pd.Series(dtype='object'), pd.Series(dtype='object')
        
        # Combine all datasets
        combined_features = pd.concat(all_features, ignore_index=True, sort=False)
        combined_labels = pd.concat(all_labels, ignore_index=True)
        dataset_sources = pd.Series(all_sources)
        
        # Handle missing values
        combined_features = self.handle_missing_values(combined_features)
        
        # Remove extreme outliers
        numeric_cols = combined_features.select_dtypes(include=[np.number]).columns
        combined_features_clean = self.remove_outliers(combined_features, numeric_cols.tolist())
        
        # Align labels with cleaned features
        clean_indices = combined_features_clean.index
        combined_labels_clean = combined_labels.iloc[clean_indices]
        dataset_sources_clean = dataset_sources.iloc[clean_indices]
        
        logger.info(f"🎯 Combined dataset: {len(combined_features_clean):,} samples, {len(combined_features_clean.columns)} features")
        
        # Print class distribution
        class_counts = combined_labels_clean.value_counts()
        logger.info("📊 Class distribution:")
        for class_name, count in class_counts.items():
            percentage = count / len(combined_labels_clean) * 100
            logger.info(f"   {class_name}: {count:,} ({percentage:.1f}%)")
        
        return combined_features_clean, combined_labels_clean, dataset_sources_clean
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            
        Returns:
            Scaled features
        """
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        X_train_scaled = X_train.copy()
        X_train_scaled[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def encode_labels(self, y_train: pd.Series, y_test: pd.Series = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Encode string labels to integers.
        
        Args:
            y_train: Training labels
            y_test: Test labels (optional)
            
        Returns:
            Encoded labels
        """
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        if y_test is not None:
            y_test_encoded = self.label_encoder.transform(y_test)
            return y_train_encoded, y_test_encoded
        
        return y_train_encoded
    
    def get_feature_info(self) -> Dict[str, str]:
        """
        Get information about the engineered features.
        
        Returns:
            Dictionary with feature descriptions
        """
        feature_info = {
            'period': 'Orbital period in days',
            'radius': 'Planet radius in Earth radii',
            'temperature': 'Equilibrium temperature in Kelvin',
            'insolation': 'Insolation flux relative to Earth',
            'a_over_rstar': 'Semi-major axis to stellar radius ratio',
            'duration': 'Transit duration in hours',
            'depth': 'Transit depth in parts per million',
            'impact': 'Impact parameter',
            'ra': 'Right ascension in degrees',
            'dec': 'Declination in degrees', 
            'magnitude': 'Stellar magnitude',
            'density_proxy': 'Planet density proxy (radius³/period²)',
            'habitability_proxy': 'Binary habitability indicator',
            'duty_cycle': 'Transit duty cycle (duration/period)'
        }
        return feature_info

def main():
    """
    Example usage of the preprocessing pipeline.
    """
    print("🚀 NASA Space Apps Challenge 2025 - Data Preprocessing")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = ExoplanetDataPreprocessor()
    
    # Load datasets
    print("\n📁 Loading NASA exoplanet datasets...")
    datasets = preprocessor.load_datasets()
    
    # Combine and preprocess
    print("\n🔄 Combining and preprocessing datasets...")
    X, y, sources = preprocessor.combine_datasets(datasets)
    
    if len(X) > 0:
        print(f"\n✅ Preprocessing complete!")
        print(f"   Features shape: {X.shape}")
        print(f"   Labels shape: {y.shape}")
        print(f"   Classes: {sorted(y.unique())}")
        
        # Feature information
        print(f"\n📋 Feature Information:")
        feature_info = preprocessor.get_feature_info()
        for feature, description in feature_info.items():
            if feature in X.columns:
                print(f"   • {feature}: {description}")
        
        # Save preprocessed data
        output_dir = Path("data/processed")
        output_dir.mkdir(exist_ok=True)
        
        X.to_csv(output_dir / "features.csv", index=False)
        pd.Series(y).to_csv(output_dir / "labels.csv", index=False, header=['label'])
        sources.to_csv(output_dir / "sources.csv", index=False, header=['source'])
        
        print(f"\n💾 Saved preprocessed data to {output_dir}/")
    else:
        print("\n❌ No data to preprocess. Check your data files.")

if __name__ == "__main__":
    main()