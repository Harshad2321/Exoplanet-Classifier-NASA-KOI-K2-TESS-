"""
🚀 Enhanced Exoplanet Prediction Pipeline
NASA Space Apps Challenge 2025

This module provides an advanced prediction pipeline supporting multiple ML models,
ensemble voting, uncertainty estimation, and comprehensive prediction analysis.
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class EnhancedExoplanetPredictor:
    """
    Advanced exoplanet prediction system with multiple models and ensemble support.
    """
    
    def __init__(self, models_dir: str = "models", reports_dir: str = "reports"):
        """
        Initialize the enhanced predictor.
        
        Args:
            models_dir: Directory containing trained models
            reports_dir: Directory for saving reports and visualizations
        """
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.figures_dir = self.reports_dir / "figures"
        
        # Create directories if they don't exist
        self.models_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True)
        
        # Model storage
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_names: List[str] = []
        self.class_names: List[str] = []
        self.metadata: Dict = {}
        
        logger.info("🚀 Enhanced Exoplanet Predictor initialized")
    
    def load_models(self) -> int:
        """
        Load all available trained models and associated components.
        
        Returns:
            Number of models loaded successfully
        """
        logger.info("📥 Loading trained models...")
        
        models_loaded = 0
        
        # Load models
        for model_file in self.models_dir.glob("*_model.joblib"):
            model_name = model_file.stem.replace('_model', '')
            try:
                self.models[model_name] = joblib.load(model_file)
                logger.info(f"✅ Loaded {model_name} model")
                models_loaded += 1
            except Exception as e:
                logger.error(f"❌ Failed to load {model_name}: {e}")
        
        # Load scalers
        for scaler_file in self.models_dir.glob("*_scaler.joblib"):
            scaler_name = scaler_file.stem.replace('_scaler', '')
            try:
                self.scalers[scaler_name] = joblib.load(scaler_file)
                logger.info(f"📊 Loaded {scaler_name} scaler")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load {scaler_name} scaler: {e}")
        
        # Load label encoder
        label_encoder_file = self.models_dir / "label_encoder.joblib"
        if label_encoder_file.exists():
            try:
                self.label_encoder = joblib.load(label_encoder_file)
                self.class_names = list(self.label_encoder.classes_)
                logger.info(f"🏷️  Loaded label encoder with classes: {self.class_names}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load label encoder: {e}")
        
        # Load metadata
        metadata_files = list(self.models_dir.glob("*metadata*.json"))
        if metadata_files:
            try:
                with open(metadata_files[0], 'r') as f:
                    self.metadata = json.load(f)
                self.feature_names = self.metadata.get('feature_names', [])
                logger.info(f"📋 Loaded metadata with {len(self.feature_names)} features")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load metadata: {e}")
        
        if models_loaded == 0:
            logger.warning("⚠️  No models loaded! Please train models first.")
        else:
            logger.info(f"🎯 Successfully loaded {models_loaded} models")
        
        return models_loaded
    
    def preprocess_input(self, data: pd.DataFrame, scaler_name: str = 'main') -> np.ndarray:
        """
        Preprocess input data using the appropriate scaler.
        
        Args:
            data: Input DataFrame with features
            scaler_name: Name of the scaler to use
            
        Returns:
            Preprocessed numpy array
        """
        if scaler_name in self.scalers:
            return self.scalers[scaler_name].transform(data)
        else:
            logger.warning(f"⚠️  Scaler '{scaler_name}' not found, using raw data")
            return data.values
    
    def predict_single_model(self, 
                           data: pd.DataFrame, 
                           model_name: str,
                           return_probabilities: bool = True) -> Dict[str, Any]:
        """
        Make predictions using a single model.
        
        Args:
            data: Input data
            model_name: Name of the model to use
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with predictions and metadata
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Preprocess data
        X = self.preprocess_input(data, scaler_name=model_name)
        
        # Make predictions
        predictions = model.predict(X)
        
        result = {
            'model': model_name,
            'predictions': predictions,
            'prediction_labels': None,
            'probabilities': None,
            'confidence': None
        }
        
        # Convert to labels if label encoder available
        if self.label_encoder is not None:
            result['prediction_labels'] = self.label_encoder.inverse_transform(predictions)
        
        # Get probabilities if available and requested
        if return_probabilities and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            result['probabilities'] = probabilities
            result['confidence'] = np.max(probabilities, axis=1)
        
        return result
    
    def predict_ensemble(self, 
                        data: pd.DataFrame, 
                        voting: str = 'soft',
                        models_to_use: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Make ensemble predictions using multiple models.
        
        Args:
            data: Input data
            voting: 'hard' or 'soft' voting
            models_to_use: List of model names to use (None for all)
            
        Returns:
            Dictionary with ensemble predictions and metadata
        """
        if not self.models:
            raise ValueError("No models loaded")
        
        models_to_use = models_to_use or list(self.models.keys())
        models_to_use = [m for m in models_to_use if m in self.models]
        
        if not models_to_use:
            raise ValueError("No valid models specified")
        
        logger.info(f"🤝 Making ensemble predictions with {len(models_to_use)} models")
        
        # Collect predictions from all models
        all_predictions = []
        all_probabilities = []
        model_confidences = {}
        
        for model_name in models_to_use:
            result = self.predict_single_model(data, model_name, return_probabilities=True)
            all_predictions.append(result['predictions'])
            
            if result['probabilities'] is not None:
                all_probabilities.append(result['probabilities'])
                model_confidences[model_name] = result['confidence']
            else:
                # For models without probabilities, create dummy probabilities
                n_samples = len(result['predictions'])
                n_classes = len(self.class_names) if self.class_names else 3
                dummy_probs = np.zeros((n_samples, n_classes))
                for i, pred in enumerate(result['predictions']):
                    dummy_probs[i, pred] = 1.0
                all_probabilities.append(dummy_probs)
                model_confidences[model_name] = np.ones(n_samples)
        
        # Ensemble voting
        if voting == 'hard':
            # Hard voting: majority vote
            predictions_array = np.array(all_predictions).T
            ensemble_predictions = []
            for i in range(len(data)):
                unique, counts = np.unique(predictions_array[i], return_counts=True)
                ensemble_predictions.append(unique[np.argmax(counts)])
            ensemble_predictions = np.array(ensemble_predictions)
            
        else:  # soft voting
            # Soft voting: average probabilities
            if all_probabilities:
                avg_probabilities = np.mean(all_probabilities, axis=0)
                ensemble_predictions = np.argmax(avg_probabilities, axis=1)
            else:
                # Fallback to hard voting if no probabilities
                predictions_array = np.array(all_predictions).T
                ensemble_predictions = []
                for i in range(len(data)):
                    unique, counts = np.unique(predictions_array[i], return_counts=True)
                    ensemble_predictions.append(unique[np.argmax(counts)])
                ensemble_predictions = np.array(ensemble_predictions)
                avg_probabilities = None
        
        # Calculate ensemble confidence
        if avg_probabilities is not None:
            ensemble_confidence = np.max(avg_probabilities, axis=1)
        else:
            ensemble_confidence = np.ones(len(data)) * 0.5  # Default confidence
        
        # Calculate prediction agreement
        predictions_array = np.array(all_predictions).T
        agreement = []
        for i in range(len(data)):
            unique, counts = np.unique(predictions_array[i], return_counts=True)
            agreement.append(np.max(counts) / len(models_to_use))
        
        result = {
            'model': f'ensemble_{voting}_voting',
            'predictions': ensemble_predictions,
            'prediction_labels': None,
            'probabilities': avg_probabilities,
            'confidence': ensemble_confidence,
            'agreement': np.array(agreement),
            'individual_predictions': dict(zip(models_to_use, all_predictions)),
            'individual_confidences': model_confidences,
            'models_used': models_to_use
        }
        
        # Convert to labels if label encoder available
        if self.label_encoder is not None:
            result['prediction_labels'] = self.label_encoder.inverse_transform(ensemble_predictions)
        
        return result
    
    def predict_with_uncertainty(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions with uncertainty estimation using multiple models.
        
        Args:
            data: Input data
            
        Returns:
            Dictionary with predictions and uncertainty metrics
        """
        logger.info("🎲 Making predictions with uncertainty estimation...")
        
        # Get ensemble predictions
        ensemble_result = self.predict_ensemble(data, voting='soft')
        
        # Calculate uncertainty metrics
        if ensemble_result['probabilities'] is not None:
            probabilities = ensemble_result['probabilities']
            
            # Entropy-based uncertainty
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
            normalized_entropy = entropy / np.log(len(self.class_names)) if self.class_names else entropy
            
            # Variation ratio (1 - max probability)
            variation_ratio = 1 - np.max(probabilities, axis=1)
            
            # Mutual information approximation
            mean_entropy = np.mean(entropy)
            epistemic_uncertainty = mean_entropy - entropy
            
        else:
            # Fallback uncertainty metrics based on agreement
            normalized_entropy = 1 - ensemble_result['agreement']
            variation_ratio = 1 - ensemble_result['agreement']
            epistemic_uncertainty = variation_ratio
        
        result = ensemble_result.copy()
        result.update({
            'uncertainty_entropy': normalized_entropy,
            'uncertainty_variation_ratio': variation_ratio,
            'epistemic_uncertainty': epistemic_uncertainty,
            'model_agreement': ensemble_result['agreement']
        })
        
        return result
    
    def batch_predict(self, 
                     data: pd.DataFrame,
                     method: str = 'ensemble',
                     save_results: bool = True) -> pd.DataFrame:
        """
        Make batch predictions on a dataset.
        
        Args:
            data: Input DataFrame
            method: Prediction method ('ensemble', 'best_model', or specific model name)
            save_results: Whether to save results to file
            
        Returns:
            DataFrame with original data and predictions
        """
        logger.info(f"📊 Making batch predictions using {method} method...")
        
        if method == 'ensemble':
            result = self.predict_with_uncertainty(data)
        elif method == 'best_model':
            # Use the best model based on metadata
            best_model = self._get_best_model()
            result = self.predict_single_model(data, best_model)
        else:
            # Use specific model
            result = self.predict_single_model(data, method)
        
        # Create results DataFrame
        results_df = data.copy()
        results_df['predicted_class'] = result['predictions']
        
        if result['prediction_labels'] is not None:
            results_df['predicted_label'] = result['prediction_labels']
        
        if result['confidence'] is not None:
            results_df['confidence'] = result['confidence']
        
        # Add uncertainty metrics if available
        if 'uncertainty_entropy' in result:
            results_df['uncertainty_entropy'] = result['uncertainty_entropy']
            results_df['uncertainty_variation_ratio'] = result['uncertainty_variation_ratio']
            results_df['model_agreement'] = result['model_agreement']
        
        # Add probabilities if available
        if result['probabilities'] is not None:
            for i, class_name in enumerate(self.class_names):
                results_df[f'prob_{class_name}'] = result['probabilities'][:, i]
        
        # Save results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions_{method}_{timestamp}.csv"
            filepath = self.reports_dir / filename
            results_df.to_csv(filepath, index=False)
            logger.info(f"💾 Predictions saved to {filepath}")
        
        return results_df
    
    def _get_best_model(self) -> str:
        """Get the best performing model based on metadata."""
        if 'model_scores' in self.metadata:
            scores = self.metadata['model_scores']
            best_model = max(scores.keys(), key=lambda x: scores[x].get('f1_score', 0))
            return best_model
        elif self.models:
            return list(self.models.keys())[0]  # Return first model as fallback
        else:
            raise ValueError("No models available")
    
    def create_prediction_report(self, 
                               results_df: pd.DataFrame,
                               ground_truth: Optional[pd.Series] = None) -> None:
        """
        Create a comprehensive prediction report with visualizations.
        
        Args:
            results_df: DataFrame with predictions
            ground_truth: Optional ground truth labels for evaluation
        """
        logger.info("📈 Creating prediction report...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Prediction Distribution',
                'Confidence Distribution',
                'Uncertainty Analysis',
                'Model Performance (if ground truth available)'
            ],
            specs=[
                [{"type": "bar"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "table" if ground_truth is not None else "bar"}]
            ]
        )
        
        # 1. Prediction distribution
        if 'predicted_label' in results_df.columns:
            pred_counts = results_df['predicted_label'].value_counts()
            fig.add_trace(
                go.Bar(x=pred_counts.index, y=pred_counts.values, name='Predictions',
                       marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']),
                row=1, col=1
            )
        
        # 2. Confidence distribution
        if 'confidence' in results_df.columns:
            fig.add_trace(
                go.Histogram(x=results_df['confidence'], nbinsx=30, name='Confidence',
                           marker_color='lightblue', opacity=0.7),
                row=1, col=2
            )
        
        # 3. Uncertainty analysis
        if 'uncertainty_entropy' in results_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=results_df['confidence'] if 'confidence' in results_df.columns else range(len(results_df)),
                    y=results_df['uncertainty_entropy'],
                    mode='markers',
                    name='Uncertainty vs Confidence',
                    marker=dict(size=6, color=results_df['model_agreement'] if 'model_agreement' in results_df.columns else 'blue',
                              colorscale='Viridis', showscale=True, colorbar=dict(title="Agreement"))
                ),
                row=2, col=1
            )
        
        # 4. Performance metrics (if ground truth available)
        if ground_truth is not None and 'predicted_label' in results_df.columns:
            # Calculate metrics
            accuracy = accuracy_score(ground_truth, results_df['predicted_label'])
            report = classification_report(ground_truth, results_df['predicted_label'], output_dict=True)
            
            # Create confusion matrix
            cm = confusion_matrix(ground_truth, results_df['predicted_label'])
            
            # Add performance table
            metrics_data = [
                ['Accuracy', f'{accuracy:.4f}'],
                ['Precision (macro)', f'{report["macro avg"]["precision"]:.4f}'],
                ['Recall (macro)', f'{report["macro avg"]["recall"]:.4f}'],
                ['F1-Score (macro)', f'{report["macro avg"]["f1-score"]:.4f}']
            ]
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Metric', 'Value'], fill_color='lightblue'),
                    cells=dict(values=list(zip(*metrics_data)), fill_color='white')
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="🎯 Comprehensive Prediction Analysis Report",
            showlegend=False,
            template='plotly_white'
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Predicted Class", row=1, col=1)
        fig.update_xaxes(title_text="Confidence", row=1, col=2)
        fig.update_xaxes(title_text="Confidence" if 'confidence' in results_df.columns else "Sample Index", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Uncertainty (Entropy)", row=2, col=1)
        
        # Save and show
        fig.write_html(self.figures_dir / "prediction_report.html")
        fig.show()
        
        # Additional detailed analysis
        self._create_detailed_analysis(results_df, ground_truth)
        
        logger.info("✅ Prediction report created successfully!")
    
    def _create_detailed_analysis(self, 
                                 results_df: pd.DataFrame,
                                 ground_truth: Optional[pd.Series] = None) -> None:
        """Create detailed analysis plots."""
        
        # Class probability distribution
        if any(col.startswith('prob_') for col in results_df.columns):
            prob_cols = [col for col in results_df.columns if col.startswith('prob_')]
            
            fig = go.Figure()
            for col in prob_cols:
                class_name = col.replace('prob_', '')
                fig.add_trace(go.Histogram(
                    x=results_df[col],
                    name=class_name,
                    opacity=0.7,
                    nbinsx=30
                ))
            
            fig.update_layout(
                title="📊 Class Probability Distributions",
                xaxis_title="Probability",
                yaxis_title="Frequency",
                template='plotly_white',
                barmode='overlay'
            )
            
            fig.write_html(self.figures_dir / "probability_distributions.html")
            fig.show()
        
        # Uncertainty vs confidence scatter with more details
        if all(col in results_df.columns for col in ['confidence', 'uncertainty_entropy', 'model_agreement']):
            fig = go.Figure()
            
            # Color by predicted class
            if 'predicted_label' in results_df.columns:
                colors = results_df['predicted_label']
                color_discrete_map = {
                    'CONFIRMED': '#FF6B6B',
                    'CANDIDATE': '#4ECDC4', 
                    'FALSE_POSITIVE': '#45B7D1'
                }
            else:
                colors = 'blue'
                color_discrete_map = None
            
            fig.add_trace(go.Scatter(
                x=results_df['confidence'],
                y=results_df['uncertainty_entropy'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors,
                    opacity=0.6,
                    colorscale='Viridis' if color_discrete_map is None else None,
                    showscale=color_discrete_map is None
                ),
                text=results_df['predicted_label'] if 'predicted_label' in results_df.columns else None,
                name='Predictions'
            ))
            
            fig.update_layout(
                title="🎲 Prediction Confidence vs Uncertainty Analysis",
                xaxis_title="Confidence",
                yaxis_title="Uncertainty (Entropy)",
                template='plotly_white'
            )
            
            fig.write_html(self.figures_dir / "confidence_vs_uncertainty.html")
            fig.show()

def main():
    """
    Main function demonstrating the enhanced prediction pipeline.
    """
    print("🚀 Enhanced Exoplanet Prediction Pipeline Demo")
    print("=" * 50)
    
    # Initialize predictor
    predictor = EnhancedExoplanetPredictor()
    
    # Load models
    models_loaded = predictor.load_models()
    
    if models_loaded == 0:
        print("⚠️  No models found. Please train models first using enhanced_train.py")
        return
    
    # Create sample data for demonstration
    print("\n📊 Creating sample prediction data...")
    np.random.seed(42)
    
    # Generate realistic exoplanet features
    n_samples = 100
    sample_data = pd.DataFrame({
        'period': np.random.lognormal(2, 1, n_samples),  # Orbital period
        'radius': np.random.lognormal(0, 0.5, n_samples),  # Planet radius  
        'temperature': np.random.normal(5778, 1000, n_samples),  # Stellar temperature
        'insolation': np.random.lognormal(0, 1, n_samples),  # Insolation flux
        'depth': np.random.lognormal(-6, 1, n_samples),  # Transit depth
        'ra': np.random.uniform(0, 360, n_samples),  # Right ascension
        'dec': np.random.uniform(-90, 90, n_samples)  # Declination
    })
    
    print(f"✅ Sample data created: {len(sample_data)} samples")
    
    # Make predictions using different methods
    print("\n🎯 Making predictions...")
    
    try:
        # Ensemble predictions with uncertainty
        results = predictor.batch_predict(sample_data, method='ensemble')
        print(f"✅ Ensemble predictions completed: {len(results)} samples")
        
        # Display sample results
        print("\n📋 Sample Prediction Results:")
        columns_to_show = ['predicted_label', 'confidence', 'uncertainty_entropy', 'model_agreement']
        available_columns = [col for col in columns_to_show if col in results.columns]
        
        if available_columns:
            print(results[available_columns].head(10).round(4))
        
        # Create prediction report
        print("\n📈 Creating prediction report...")
        predictor.create_prediction_report(results)
        
        # Summary statistics
        print("\n📊 Prediction Summary:")
        if 'predicted_label' in results.columns:
            print("Class Distribution:")
            print(results['predicted_label'].value_counts())
        
        if 'confidence' in results.columns:
            print(f"\nConfidence Statistics:")
            print(f"Mean: {results['confidence'].mean():.4f}")
            print(f"Std:  {results['confidence'].std():.4f}")
            print(f"Min:  {results['confidence'].min():.4f}")
            print(f"Max:  {results['confidence'].max():.4f}")
        
        print("✅ Enhanced prediction pipeline demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()