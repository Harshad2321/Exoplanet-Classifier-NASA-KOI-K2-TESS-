"""
Data Loader Module for NASA Exoplanet Datasets

This module handles downloading and loading datasets from NASA Exoplanet Archive:
- Kepler Objects of Interest (KOI)
- K2 Planets and Candidates  
- TESS Objects of Interest (TOI)

Author: NASA Space Apps Challenge 2025 Team
"""

import os
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExoplanetDataLoader:
    """
    Handles downloading and loading of NASA exoplanet datasets
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize data loader
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset URLs and configurations
        self.dataset_configs = {
            'koi': {
                'url': 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv',
                'filename': 'koi.csv',
                'target_column': 'disposition',  # Updated to match NASA archive format
                'description': 'Kepler Objects of Interest'
            },
            'k2': {
                'url': 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+k2candidates&format=csv',
                'filename': 'k2.csv',  # Updated filename
                'target_column': 'disposition',  # Updated to match NASA archive format
                'description': 'K2 Candidates'
            },
            'toi': {
                'url': 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+toi&format=csv',
                'filename': 'toi.csv',
                'target_column': 'disposition',  # Updated to match NASA archive format
                'description': 'TESS Objects of Interest'
            }
        }
    
    def download_dataset(self, dataset_name: str, force_redownload: bool = False) -> bool:
        """
        Download a specific dataset
        
        Args:
            dataset_name: Name of dataset ('koi', 'k2', 'toi')
            force_redownload: Whether to redownload existing files
            
        Returns:
            bool: Success status
        """
        if dataset_name not in self.dataset_configs:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
        
        config = self.dataset_configs[dataset_name]
        filepath = self.raw_dir / config['filename']
        
        # Check if file exists and skip if not forcing redownload
        if filepath.exists() and not force_redownload:
            logger.info(f"{config['description']} already exists, skipping download")
            return True
        
        try:
            logger.info(f"Downloading {config['description']}...")
            
            response = requests.get(config['url'], stream=True)
            response.raise_for_status()
            
            # Get total file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as file, tqdm(
                desc=config['description'],
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Successfully downloaded {config['description']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {config['description']}: {str(e)}")
            return False
    
    def download_all_datasets(self, force_redownload: bool = False) -> Dict[str, bool]:
        """
        Download all datasets
        
        Args:
            force_redownload: Whether to redownload existing files
            
        Returns:
            Dict mapping dataset names to success status
        """
        results = {}
        for dataset_name in self.dataset_configs:
            results[dataset_name] = self.download_dataset(dataset_name, force_redownload)
        return results
    
    def load_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        Load a specific dataset
        
        Args:
            dataset_name: Name of dataset to load
            
        Returns:
            DataFrame or None if loading fails
        """
        if dataset_name not in self.dataset_configs:
            logger.error(f"Unknown dataset: {dataset_name}")
            return None
        
        config = self.dataset_configs[dataset_name]
        filepath = self.raw_dir / config['filename']
        
        if not filepath.exists():
            logger.error(f"Dataset file not found: {filepath}")
            logger.info(f"Try running download_dataset('{dataset_name}') first")
            return None
        
        try:
            logger.info(f"Loading {config['description']}...")
            # Skip comment lines starting with # (NASA archive format)
            df = pd.read_csv(filepath, low_memory=False, comment='#')
            logger.info(f"Loaded {len(df)} records from {config['description']}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load {config['description']}: {str(e)}")
            return None
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available datasets
        
        Returns:
            Dictionary mapping dataset names to DataFrames
        """
        datasets = {}
        for dataset_name in self.dataset_configs:
            df = self.load_dataset(dataset_name)
            if df is not None:
                datasets[dataset_name] = df
        return datasets
    
    def get_dataset_info(self) -> pd.DataFrame:
        """
        Get information about available datasets
        
        Returns:
            DataFrame with dataset information
        """
        info_data = []
        for name, config in self.dataset_configs.items():
            filepath = self.raw_dir / config['filename']
            exists = filepath.exists()
            size_mb = filepath.stat().st_size / (1024*1024) if exists else 0
            
            # Try to get record count if file exists
            record_count = 0
            if exists:
                try:
                    df = pd.read_csv(filepath, nrows=0)  # Just get header
                    full_df = pd.read_csv(filepath)
                    record_count = len(full_df)
                except:
                    record_count = "Error"
            
            info_data.append({
                'Dataset': name.upper(),
                'Description': config['description'],
                'Target Column': config['target_column'],
                'File Exists': exists,
                'Size (MB)': round(size_mb, 2) if exists else 0,
                'Records': record_count
            })
        
        return pd.DataFrame(info_data)
    
    def validate_datasets(self) -> Dict[str, Dict[str, any]]:
        """
        Validate downloaded datasets and check for required columns
        
        Returns:
            Validation results for each dataset
        """
        validation_results = {}
        
        for dataset_name in self.dataset_configs:
            config = self.dataset_configs[dataset_name]
            result = {
                'exists': False,
                'loadable': False,
                'has_target': False,
                'record_count': 0,
                'column_count': 0,
                'errors': []
            }
            
            filepath = self.raw_dir / config['filename']
            
            # Check if file exists
            if not filepath.exists():
                result['errors'].append(f"File not found: {filepath}")
                validation_results[dataset_name] = result
                continue
            
            result['exists'] = True
            
            # Try to load dataset
            try:
                df = self.load_dataset(dataset_name)
                if df is not None:
                    result['loadable'] = True
                    result['record_count'] = len(df)
                    result['column_count'] = len(df.columns)
                    
                    # Check for target column
                    if config['target_column'] in df.columns:
                        result['has_target'] = True
                    else:
                        result['errors'].append(f"Target column '{config['target_column']}' not found")
                        # Show available columns that might be targets
                        target_like_cols = [col for col in df.columns if 'disp' in col.lower()]
                        if target_like_cols:
                            result['errors'].append(f"Possible target columns: {target_like_cols}")
                else:
                    result['errors'].append("Failed to load dataset")
                    
            except Exception as e:
                result['errors'].append(f"Loading error: {str(e)}")
            
            validation_results[dataset_name] = result
        
        return validation_results


def main():
    """
    Main function to demonstrate data loader functionality
    """
    print("🚀 NASA Exoplanet Data Loader")
    print("=" * 50)
    
    # Initialize data loader
    loader = ExoplanetDataLoader()
    
    # Show dataset info
    print("\n📊 Dataset Information:")
    info_df = loader.get_dataset_info()
    print(info_df.to_string(index=False))
    
    # Download datasets
    print("\n📥 Downloading datasets...")
    download_results = loader.download_all_datasets()
    
    for dataset, success in download_results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {dataset.upper()}: {status}")
    
    # Validate datasets
    print("\n🔍 Validating datasets...")
    validation_results = loader.validate_datasets()
    
    for dataset, result in validation_results.items():
        print(f"\n{dataset.upper()}:")
        print(f"  Exists: {'✅' if result['exists'] else '❌'}")
        print(f"  Loadable: {'✅' if result['loadable'] else '❌'}")
        print(f"  Has Target: {'✅' if result['has_target'] else '❌'}")
        print(f"  Records: {result['record_count']}")
        print(f"  Columns: {result['column_count']}")
        
        if result['errors']:
            print(f"  Errors:")
            for error in result['errors']:
                print(f"    - {error}")
    
    print("\n✅ Data loading complete!")


if __name__ == "__main__":
    main()