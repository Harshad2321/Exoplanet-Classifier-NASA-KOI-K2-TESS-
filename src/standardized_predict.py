"""
Standardized Prediction System for NASA Space Apps Challenge 2025
Exoplanet Classifier - Unified Input/Output Format

This module provides standardized prediction functions for both single
and batch predictions with consistent error handling and format validation.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StandardizedExoplanetPredictor:
    """
    Standardized prediction system for exoplanet classification.
    
    Supports both single predictions (dict input/output) and batch predictions (CSV).
    All inputs and outputs follow NASA Space Apps Challenge 2025 specifications.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the standardized predictor.
        
        Args:
            models_dir (str): Directory containing model files
        """
        self.models_dir = Path(models_dir)
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.class_labels = ["FALSE_POSITIVE", "CANDIDATE", "CONFIRMED"]
        
        # Required input fields for validation
        self.required_fields = [
            "orbital_period",
            "transit_duration", 
            "planet_radius",
            "stellar_radius",
            "stellar_temp"
        ]
        
        # Optional fields
        self.optional_fields = ["mission"]
        
        # Load model components
        self._load_models()
    
    def _load_models(self):
        """Load trained model and preprocessor."""
        try:
            # Try to load best model
            model_path = self.models_dir / "best_model.pkl"
            if not model_path.exists():
                # Fallback to other model files
                model_files = list(self.models_dir.glob("*.pkl"))
                if not model_files:
                    raise FileNotFoundError("No model files found in models directory")
                
                # Use the first available model
                model_path = model_files[0]
                logger.warning(f"best_model.pkl not found, using {model_path.name}")
            
            self.model = joblib.load(model_path)
            logger.info(f"✅ Loaded model: {model_path.name}")
            
            # Try to load preprocessor
            preprocessor_path = self.models_dir / "preprocessor.pkl"
            if preprocessor_path.exists():
                self.preprocessor = joblib.load(preprocessor_path)
                logger.info("✅ Loaded preprocessor")
            else:
                logger.warning("⚠️ Preprocessor not found, using raw features")
            
            # Try to load feature names
            feature_path = self.models_dir / "feature_names.pkl"
            if feature_path.exists():
                self.feature_names = joblib.load(feature_path)
                logger.info(f"✅ Loaded {len(self.feature_names)} feature names")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            raise
    
    def validate_input(self, input_dict: Dict[str, Any]) -> Dict[str, str]:
        """
        Validate input dictionary for required fields.
        
        Args:
            input_dict (dict): Input dictionary to validate
            
        Returns:
            dict: Error dictionary if validation fails, empty dict if valid
        """
        errors = {}
        
        # Check for missing required fields
        for field in self.required_fields:
            if field not in input_dict or input_dict[field] is None:
                errors["error"] = f"Missing required field: {field}"
                return errors
        
        # Validate data types and ranges
        try:
            # Orbital period should be positive
            if float(input_dict["orbital_period"]) <= 0:
                errors["error"] = "orbital_period must be positive"
                return errors
            
            # Transit duration should be positive
            if float(input_dict["transit_duration"]) <= 0:
                errors["error"] = "transit_duration must be positive"
                return errors
            
            # Planet radius should be positive
            if float(input_dict["planet_radius"]) <= 0:
                errors["error"] = "planet_radius must be positive"
                return errors
            
            # Stellar radius should be positive
            if float(input_dict["stellar_radius"]) <= 0:
                errors["error"] = "stellar_radius must be positive"
                return errors
            
            # Stellar temperature should be reasonable (1000-50000 K)
            stellar_temp = float(input_dict["stellar_temp"])
            if stellar_temp < 1000 or stellar_temp > 50000:
                errors["error"] = "stellar_temp must be between 1000 and 50000 K"
                return errors
                
        except (ValueError, TypeError):
            errors["error"] = "All numeric fields must be valid numbers"
            return errors
        
        return errors
    
    def preprocess_input(self, input_dict: Dict[str, Any]) -> np.ndarray:
        """
        Convert input dictionary to model-ready features.
        
        Args:
            input_dict (dict): Validated input dictionary
            
        Returns:
            np.ndarray: Preprocessed features for model
        """
        # Create feature vector based on standard format
        features = [
            float(input_dict["orbital_period"]),
            float(input_dict["transit_duration"]),
            float(input_dict["planet_radius"]),
            float(input_dict["stellar_radius"]),
            float(input_dict["stellar_temp"])
        ]
        
        # Add mission encoding if provided
        if "mission" in input_dict:
            mission = input_dict["mission"].lower()
            # One-hot encode mission (kepler=1,0,0; k2=0,1,0; tess=0,0,1)
            if mission == "kepler":
                features.extend([1, 0, 0])
            elif mission == "k2":
                features.extend([0, 1, 0])
            elif mission == "tess":
                features.extend([0, 0, 1])
            else:
                # Default to kepler if unknown mission
                features.extend([1, 0, 0])
        else:
            # Default to kepler if no mission specified
            features.extend([1, 0, 0])
        
        # Convert to numpy array
        features_array = np.array(features).reshape(1, -1)
        
        # Apply preprocessor if available
        if self.preprocessor:
            try:
                features_array = self.preprocessor.transform(features_array)
            except Exception as e:
                logger.warning(f"Preprocessor failed, using raw features: {e}")
        
        return features_array
    
    def predict(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a single prediction with standardized input/output format.
        
        Args:
            input_dict (dict): Input dictionary with exoplanet features
            
        Returns:
            dict: Prediction result or error message
            
        Example:
            >>> predictor = StandardizedExoplanetPredictor()
            >>> result = predictor.predict({
            ...     "orbital_period": 10.5,
            ...     "transit_duration": 2.0,
            ...     "planet_radius": 2.1,
            ...     "stellar_radius": 1.0,
            ...     "stellar_temp": 5800,
            ...     "mission": "tess"
            ... })
            >>> print(result)
            {
                "predicted_label": "CANDIDATE",
                "probabilities": {
                    "CONFIRMED": 0.42,
                    "CANDIDATE": 0.48,
                    "FALSE_POSITIVE": 0.10
                }
            }
        """
        try:
            # Validate input
            validation_errors = self.validate_input(input_dict)
            if validation_errors:
                return validation_errors
            
            # Preprocess input
            features = self.preprocess_input(input_dict)
            
            # Make prediction
            probabilities = self.model.predict_proba(features)[0]
            predicted_class_idx = np.argmax(probabilities)
            predicted_label = self.class_labels[predicted_class_idx]
            
            # Format output
            result = {
                "predicted_label": predicted_label,
                "probabilities": {
                    "CONFIRMED": round(float(probabilities[2]), 4),
                    "CANDIDATE": round(float(probabilities[1]), 4),
                    "FALSE_POSITIVE": round(float(probabilities[0]), 4)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": f"Prediction failed: {str(e)}"}
    
    def predict_batch(self, csv_file: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Make batch predictions on CSV data.
        
        Args:
            csv_file (str or DataFrame): Path to CSV file or DataFrame
            
        Returns:
            DataFrame: Original data with added prediction columns
            
        Example:
            >>> predictor = StandardizedExoplanetPredictor()
            >>> results_df = predictor.predict_batch("exoplanet_data.csv")
            >>> print(results_df.columns)
            ['orbital_period', 'transit_duration', 'planet_radius', 
             'stellar_radius', 'stellar_temp', 'mission', 'predicted_label',
             'prob_CONFIRMED', 'prob_CANDIDATE', 'prob_FALSE_POSITIVE']
        """
        try:
            # Load data if file path provided
            if isinstance(csv_file, str):
                df = pd.read_csv(csv_file)
            else:
                df = csv_file.copy()
            
            logger.info(f"Processing batch of {len(df)} samples")
            
            # Initialize prediction columns
            df['predicted_label'] = ''
            df['prob_CONFIRMED'] = 0.0
            df['prob_CANDIDATE'] = 0.0
            df['prob_FALSE_POSITIVE'] = 0.0
            
            # Process each row
            successful_predictions = 0
            for idx, row in df.iterrows():
                # Convert row to input dictionary
                input_dict = {
                    "orbital_period": row.get("orbital_period", None),
                    "transit_duration": row.get("transit_duration", None),
                    "planet_radius": row.get("planet_radius", None),
                    "stellar_radius": row.get("stellar_radius", None),
                    "stellar_temp": row.get("stellar_temp", None),
                    "mission": row.get("mission", "kepler")
                }
                
                # Make prediction
                result = self.predict(input_dict)
                
                if "error" not in result:
                    # Success - store results
                    df.loc[idx, 'predicted_label'] = result["predicted_label"]
                    df.loc[idx, 'prob_CONFIRMED'] = result["probabilities"]["CONFIRMED"]
                    df.loc[idx, 'prob_CANDIDATE'] = result["probabilities"]["CANDIDATE"]
                    df.loc[idx, 'prob_FALSE_POSITIVE'] = result["probabilities"]["FALSE_POSITIVE"]
                    successful_predictions += 1
                else:
                    # Error - store error info
                    df.loc[idx, 'predicted_label'] = f"ERROR: {result['error']}"
            
            logger.info(f"✅ Successfully predicted {successful_predictions}/{len(df)} samples")
            
            return df
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            # Return empty DataFrame with error info
            error_df = pd.DataFrame([{
                "error": f"Batch prediction failed: {str(e)}"
            }])
            return error_df

# Convenience functions for easy import
def predict_single(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function for single prediction.
    
    Args:
        input_dict (dict): Input features dictionary
        
    Returns:
        dict: Prediction result
    """
    predictor = StandardizedExoplanetPredictor()
    return predictor.predict(input_dict)

def predict_csv(csv_file: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Convenience function for batch prediction.
    
    Args:
        csv_file (str or DataFrame): CSV file path or DataFrame
        
    Returns:
        DataFrame: Results with prediction columns
    """
    predictor = StandardizedExoplanetPredictor()
    return predictor.predict_batch(csv_file)

if __name__ == "__main__":
    # Test the standardized predictor
    print("🧪 Testing Standardized Exoplanet Predictor")
    print("=" * 60)
    
    # Test single prediction
    test_input = {
        "orbital_period": 10.5,
        "transit_duration": 2.0,
        "planet_radius": 2.1,
        "stellar_radius": 1.0,
        "stellar_temp": 5800,
        "mission": "tess"
    }
    
    try:
        predictor = StandardizedExoplanetPredictor()
        result = predictor.predict(test_input)
        print("✅ Single Prediction Test:")
        print(f"Input: {test_input}")
        print(f"Output: {result}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")