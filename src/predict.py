"""
Prediction Interface for NASA Space Apps Challenge 2025
"A World Away: Hunting for Exoplanets with AI"

This module provides easy-to-use interfaces for making predictions
with the trained exoplanet classification model.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExoplanetPredictor:
    """
    Easy-to-use interface for exoplanet classification predictions.
    
    Handles model loading, feature preprocessing, and provides both
    single sample and batch prediction capabilities.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the predictor.
        
        Args:
            model_dir: Directory containing the saved model files
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.preprocessor = None
        self.metadata = None
        self.feature_names = None
        self.class_names = None
        
        # Load model and preprocessor
        self._load_model_and_preprocessor()
    
    def _load_model_and_preprocessor(self):
        """Load the trained model and preprocessor."""
        try:
            # Find model files
            model_files = list(self.model_dir.glob("best_model_*.joblib"))
            metadata_files = list(self.model_dir.glob("model_metadata_*.json"))
            preprocessor_files = list(self.model_dir.glob("preprocessor.joblib"))
            
            if not model_files:
                raise FileNotFoundError(f"No model files found in {self.model_dir}")
            
            # Load model
            model_file = model_files[0]  # Use first model found
            self.model = joblib.load(model_file)
            logger.info(f"✅ Loaded model from {model_file}")
            
            # Load metadata
            if metadata_files:
                metadata_file = metadata_files[0]
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                    
                self.class_names = self.metadata.get('classes', ['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE'])
                self.feature_names = self.metadata.get('feature_names', [])
                
                logger.info(f"✅ Loaded metadata from {metadata_file}")
                logger.info(f"   Model: {self.metadata.get('description', 'Unknown')}")
                logger.info(f"   Classes: {self.class_names}")
            
            # Load preprocessor
            if preprocessor_files:
                preprocessor_file = preprocessor_files[0]
                self.preprocessor = joblib.load(preprocessor_file)
                logger.info(f"✅ Loaded preprocessor from {preprocessor_file}")
            else:
                logger.warning("❌ No preprocessor found. Using minimal preprocessing.")
                
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def get_feature_template(self) -> Dict[str, str]:
        """
        Get a template dictionary with all required features and their descriptions.
        
        Returns:
            Dictionary with feature names and descriptions
        """
        feature_descriptions = {
            'period': 'Orbital period in days (e.g., 365.25 for Earth-like)',
            'radius': 'Planet radius in Earth radii (e.g., 1.0 for Earth-like)',
            'temperature': 'Equilibrium temperature in Kelvin (e.g., 288 for Earth-like)',
            'insolation': 'Insolation flux relative to Earth (e.g., 1.0 for Earth-like)',
            'a_over_rstar': 'Semi-major axis to stellar radius ratio (e.g., 215 for Earth-Sun)',
            'duration': 'Transit duration in hours (e.g., 13 for Earth-Sun)',
            'depth': 'Transit depth in parts per million (e.g., 84 for Earth-Sun)',
            'impact': 'Impact parameter (0 = center, 1 = edge)',
            'ra': 'Right ascension in degrees (0-360)',
            'dec': 'Declination in degrees (-90 to +90)',
            'magnitude': 'Stellar magnitude (brightness, e.g., 4.83 for Sun)',
            'density_proxy': 'Derived: Planet density proxy (auto-calculated)',
            'habitability_proxy': 'Derived: Habitability indicator (auto-calculated)',
            'duty_cycle': 'Derived: Transit duty cycle (auto-calculated)'
        }
        
        if self.feature_names:
            # Use actual feature names from trained model
            template = {name: feature_descriptions.get(name, f'Value for {name}') 
                       for name in self.feature_names}
        else:
            # Use default feature set
            template = feature_descriptions
            
        return template
    
    def create_sample_input(self, object_type: str = 'earth_like') -> Dict[str, float]:
        """
        Create sample input for different types of objects.
        
        Args:
            object_type: Type of object ('earth_like', 'hot_jupiter', 'super_earth', 'false_positive')
            
        Returns:
            Dictionary with sample feature values
        """
        samples = {
            'earth_like': {
                'period': 365.25,
                'radius': 1.0,
                'temperature': 288.0,
                'insolation': 1.0,
                'a_over_rstar': 215.0,
                'duration': 13.0,
                'depth': 84.0,
                'impact': 0.5,
                'ra': 180.0,
                'dec': 0.0,
                'magnitude': 10.0
            },
            'hot_jupiter': {
                'period': 3.5,
                'radius': 11.0,
                'temperature': 1500.0,
                'insolation': 1000.0,
                'a_over_rstar': 5.0,
                'duration': 3.0,
                'depth': 12000.0,
                'impact': 0.2,
                'ra': 45.0,
                'dec': 30.0,
                'magnitude': 12.0
            },
            'super_earth': {
                'period': 50.0,
                'radius': 1.8,
                'temperature': 350.0,
                'insolation': 5.0,
                'a_over_rstar': 25.0,
                'duration': 8.0,
                'depth': 324.0,
                'impact': 0.3,
                'ra': 270.0,
                'dec': -15.0,
                'magnitude': 11.5
            },
            'false_positive': {
                'period': 2.1,
                'radius': 0.5,
                'temperature': 2000.0,
                'insolation': 2000.0,
                'a_over_rstar': 3.0,
                'duration': 1.5,
                'depth': 25.0,
                'impact': 0.9,
                'ra': 90.0,
                'dec': 60.0,
                'magnitude': 15.0
            }
        }
        
        return samples.get(object_type, samples['earth_like'])
    
    def _preprocess_features(self, features: Dict[str, float]) -> pd.DataFrame:
        """
        Preprocess features for prediction.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Preprocessed feature DataFrame
        """
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Engineer derived features
        if 'period' in df.columns and 'radius' in df.columns:
            df['density_proxy'] = df['radius'] ** 3 / df['period'] ** 2
        
        if 'temperature' in df.columns and 'insolation' in df.columns:
            temp_mask = (df['temperature'] >= 200) & (df['temperature'] <= 400)
            insol_mask = (df['insolation'] >= 0.5) & (df['insolation'] <= 2.0)
            df['habitability_proxy'] = (temp_mask & insol_mask).astype(int)
        
        if 'duration' in df.columns and 'period' in df.columns:
            df['duty_cycle'] = df['duration'] / df['period']
        
        # Handle missing values with median (simple approach)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Scale features if preprocessor is available
        if self.preprocessor and hasattr(self.preprocessor, 'scaler'):
            try:
                # Get expected features from the model
                expected_features = self.feature_names if self.feature_names else df.columns
                
                # Align features
                for feature in expected_features:
                    if feature not in df.columns:
                        df[feature] = 0.0  # Default value for missing features
                
                # Select and order features
                df = df[expected_features]
                
                # Scale features
                df_scaled = df.copy()
                df_scaled[numeric_cols] = self.preprocessor.scaler.transform(df[numeric_cols])
                return df_scaled
                
            except Exception as e:
                logger.warning(f"⚠️  Preprocessing failed: {e}. Using raw features.")
        
        return df
    
    def predict_single(self, features: Dict[str, float], 
                      return_probabilities: bool = True) -> Dict[str, Union[str, Dict[str, float]]]:
        """
        Predict the class of a single exoplanet candidate.
        
        Args:
            features: Dictionary of feature values
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Check model directory.")
        
        # Preprocess features
        X = self._preprocess_features(features)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        result = {
            'predicted_class': prediction,
            'confidence': None,
            'probabilities': None
        }
        
        if return_probabilities:
            probabilities = self.model.predict_proba(X)[0]
            prob_dict = {class_name: float(prob) 
                        for class_name, prob in zip(self.class_names, probabilities)}
            
            result['probabilities'] = prob_dict
            result['confidence'] = float(max(probabilities))
        
        return result
    
    def predict_batch(self, features_list: List[Dict[str, float]], 
                     return_probabilities: bool = True) -> List[Dict[str, Union[str, Dict[str, float]]]]:
        """
        Predict the classes of multiple exoplanet candidates.
        
        Args:
            features_list: List of feature dictionaries
            return_probabilities: Whether to return class probabilities
            
        Returns:
            List of prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Check model directory.")
        
        results = []
        
        for features in features_list:
            try:
                result = self.predict_single(features, return_probabilities)
                results.append(result)
            except Exception as e:
                logger.error(f"❌ Failed to predict sample: {e}")
                results.append({
                    'predicted_class': 'ERROR',
                    'confidence': 0.0,
                    'probabilities': None,
                    'error': str(e)
                })
        
        return results
    
    def predict_from_csv(self, csv_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Predict from a CSV file containing multiple samples.
        
        Args:
            csv_path: Path to input CSV file
            output_path: Path to save results CSV (optional)
            
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Check model directory.")
        
        try:
            # Load data
            df = pd.read_csv(csv_path)
            logger.info(f"📁 Loaded {len(df)} samples from {csv_path}")
            
            # Convert DataFrame rows to feature dictionaries
            features_list = df.to_dict('records')
            
            # Make predictions
            results = self.predict_batch(features_list, return_probabilities=True)
            
            # Create results DataFrame
            predictions_df = pd.DataFrame()
            predictions_df['predicted_class'] = [r['predicted_class'] for r in results]
            predictions_df['confidence'] = [r['confidence'] for r in results]
            
            # Add probability columns
            for class_name in self.class_names:
                prob_col = f'prob_{class_name.lower()}'
                predictions_df[prob_col] = [
                    r['probabilities'].get(class_name, 0.0) if r['probabilities'] else 0.0 
                    for r in results
                ]
            
            # Combine with original data
            results_df = pd.concat([df, predictions_df], axis=1)
            
            if output_path:
                results_df.to_csv(output_path, index=False)
                logger.info(f"💾 Saved predictions to {output_path}")
            
            return results_df
            
        except Exception as e:
            logger.error(f"❌ Failed to predict from CSV: {e}")
            raise
    
    def interpret_prediction(self, prediction_result: Dict) -> str:
        """
        Provide human-readable interpretation of prediction results.
        
        Args:
            prediction_result: Result from predict_single()
            
        Returns:
            Human-readable interpretation
        """
        predicted_class = prediction_result['predicted_class']
        confidence = prediction_result.get('confidence', 0)
        probabilities = prediction_result.get('probabilities', {})
        
        # Main interpretation
        interpretations = {
            'CONFIRMED': "🪐 **CONFIRMED EXOPLANET** - This object shows strong evidence of being a genuine exoplanet.",
            'CANDIDATE': "🔍 **PLANET CANDIDATE** - This object shows promising signs but requires further validation.",
            'FALSE_POSITIVE': "❌ **FALSE POSITIVE** - This object is likely not a planet (possibly eclipsing binary, stellar variability, or instrumental noise)."
        }
        
        interpretation = interpretations.get(predicted_class, f"❓ Unknown class: {predicted_class}")
        
        # Add confidence information
        if confidence:
            confidence_level = "Very High" if confidence > 0.9 else \
                             "High" if confidence > 0.8 else \
                             "Moderate" if confidence > 0.6 else \
                             "Low"
            
            interpretation += f"\n\n**Confidence:** {confidence_level} ({confidence:.1%})"
        
        # Add probability breakdown
        if probabilities:
            interpretation += "\n\n**Class Probabilities:**"
            for class_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                emoji = {"CONFIRMED": "🪐", "CANDIDATE": "🔍", "FALSE_POSITIVE": "❌"}.get(class_name, "•")
                interpretation += f"\n{emoji} {class_name}: {prob:.1%}"
        
        # Add recommendations
        if predicted_class == 'CONFIRMED':
            interpretation += "\n\n**Recommendation:** This object warrants detailed follow-up observations for characterization studies."
        elif predicted_class == 'CANDIDATE':
            interpretation += "\n\n**Recommendation:** Additional observations and analysis needed to confirm planetary nature."
        else:
            interpretation += "\n\n**Recommendation:** This object can likely be deprioritized for planet hunting surveys."
        
        return interpretation
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.metadata:
            info = {
                'model_name': self.metadata.get('model_name', 'Unknown'),
                'description': self.metadata.get('description', 'Unknown'),
                'classes': self.class_names,
                'feature_count': len(self.feature_names) if self.feature_names else 'Unknown',
                'training_accuracy': self.metadata.get('train_metrics', {}).get('accuracy', 'Unknown'),
                'validation_accuracy': self.metadata.get('val_metrics', {}).get('accuracy', 'Unknown'),
                'test_accuracy': self.metadata.get('test_metrics', {}).get('accuracy', 'Unknown'),
                'training_time': self.metadata.get('training_time', 'Unknown'),
                'timestamp': self.metadata.get('timestamp', 'Unknown')
            }
        else:
            info = {
                'model_name': 'Unknown',
                'description': 'Model metadata not available',
                'classes': self.class_names or ['Unknown'],
                'feature_count': 'Unknown'
            }
        
        return info

def main():
    """
    Example usage of the prediction interface.
    """
    print("🚀 NASA Space Apps Challenge 2025 - Exoplanet Prediction Interface")
    print("=" * 70)
    
    try:
        # Initialize predictor
        print("\n🔄 Loading model...")
        predictor = ExoplanetPredictor()
        
        # Show model info
        model_info = predictor.get_model_info()
        print(f"\n🤖 Model Information:")
        print(f"   Model: {model_info['description']}")
        print(f"   Classes: {model_info['classes']}")
        print(f"   Features: {model_info['feature_count']}")
        if model_info.get('test_accuracy') != 'Unknown':
            print(f"   Test Accuracy: {model_info['test_accuracy']:.1%}")
        
        # Show feature template
        print(f"\n📋 Required Features:")
        template = predictor.get_feature_template()
        for feature, description in list(template.items())[:5]:  # Show first 5
            print(f"   • {feature}: {description}")
        print(f"   ... and {len(template) - 5} more features")
        
        # Example predictions
        print(f"\n🌟 Example Predictions:")
        print("=" * 50)
        
        # Predict Earth-like planet
        print("\n1️⃣  Earth-like Planet:")
        earth_features = predictor.create_sample_input('earth_like')
        earth_result = predictor.predict_single(earth_features)
        print(f"   Prediction: {earth_result['predicted_class']}")
        print(f"   Confidence: {earth_result['confidence']:.1%}")
        
        # Predict Hot Jupiter
        print("\n2️⃣  Hot Jupiter:")
        jupiter_features = predictor.create_sample_input('hot_jupiter')
        jupiter_result = predictor.predict_single(jupiter_features)
        print(f"   Prediction: {jupiter_result['predicted_class']}")
        print(f"   Confidence: {jupiter_result['confidence']:.1%}")
        
        # Predict False Positive
        print("\n3️⃣  False Positive:")
        fp_features = predictor.create_sample_input('false_positive')
        fp_result = predictor.predict_single(fp_features)
        print(f"   Prediction: {fp_result['predicted_class']}")
        print(f"   Confidence: {fp_result['confidence']:.1%}")
        
        # Detailed interpretation
        print(f"\n📖 Detailed Interpretation (Earth-like):")
        interpretation = predictor.interpret_prediction(earth_result)
        print(interpretation)
        
        print(f"\n✅ Prediction interface ready!")
        print(f"💡 Use predictor.predict_single(features) for single predictions")
        print(f"💡 Use predictor.predict_from_csv(csv_path) for batch predictions")
        
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        print("\n💡 Make sure you have trained a model first by running:")
        print("   python src/train.py")

if __name__ == "__main__":
    main()