"""
🧪 Model Explainability Module - SHAP & LIME Integration
NASA Space Apps Challenge 2025

Provides interpretability and explanation capabilities for the exoplanet classification models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExoplanetExplainer:
    """
    Advanced model explainability system for exoplanet classification.
    Integrates SHAP and LIME for comprehensive model interpretation.
    """
    
    def __init__(self, models_dir: str = "models", data_dir: str = "data"):
        """
        Initialize the explainability system.
        
        Args:
            models_dir: Directory containing trained models
            data_dir: Directory containing training data
        """
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        
        # Model storage
        self.models = {}
        self.feature_names = ['period', 'radius', 'temperature', 'insolation', 'depth', 'ra', 'dec']
        self.class_names = ['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE']
        
        # Explainer objects
        self.shap_explainers = {}
        self.lime_explainer = None
        
        # Training data for explainers
        self.X_background = None
        self.y_background = None
        
        logger.info("🧪 ExoplanetExplainer initialized")
    
    def load_models(self):
        """Load trained models for explanation"""
        logger.info("📥 Loading models for explainability analysis...")
        
        model_files = [
            "best_model.pkl", "random_forest_model.pkl", 
            "xgboost_model.pkl", "lightgbm_model.pkl"
        ]
        
        for model_file in model_files:
            model_path = self.models_dir / model_file
            if model_path.exists():
                try:
                    model_name = model_file.replace('_model.pkl', '').replace('.pkl', '')
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"✅ Loaded {model_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {model_file}: {e}")
        
        return len(self.models)
    
    def load_training_data(self):
        """Load training data for background/reference"""
        try:
            # Try to load processed training data
            train_file = self.data_dir / "splits" / "train.csv"
            if train_file.exists():
                train_data = pd.read_csv(train_file)
                self.y_background = train_data['label']
                self.X_background = train_data.drop('label', axis=1)
            else:
                # Fallback: load from processed features
                features_file = self.data_dir / "processed" / "features.csv"
                labels_file = self.data_dir / "processed" / "labels.csv"
                
                if features_file.exists() and labels_file.exists():
                    self.X_background = pd.read_csv(features_file)
                    labels_df = pd.read_csv(labels_file)
                    self.y_background = labels_df['label']
                    
                    # Take a sample for efficiency
                    if len(self.X_background) > 1000:
                        sample_idx = np.random.choice(len(self.X_background), 1000, replace=False)
                        self.X_background = self.X_background.iloc[sample_idx]
                        self.y_background = self.y_background.iloc[sample_idx]
                else:
                    logger.warning("⚠️ No training data found, using synthetic data")
                    self._create_synthetic_background()
            
            logger.info(f"✅ Loaded background data: {len(self.X_background)} samples")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load training data: {e}")
            self._create_synthetic_background()
            return False
    
    def _create_synthetic_background(self):
        """Create synthetic background data for demonstration"""
        np.random.seed(42)
        n_samples = 500
        
        self.X_background = pd.DataFrame({
            'period': np.random.lognormal(2, 1, n_samples),
            'radius': np.random.lognormal(0, 0.5, n_samples),
            'temperature': np.random.normal(5778, 1000, n_samples),
            'insolation': np.random.lognormal(0, 1, n_samples),
            'depth': np.random.lognormal(-6, 1, n_samples),
            'ra': np.random.uniform(0, 360, n_samples),
            'dec': np.random.uniform(-90, 90, n_samples)
        })
        
        self.y_background = np.random.choice([0, 1, 2], n_samples)
        logger.info("🔧 Created synthetic background data")
    
    def initialize_shap_explainers(self):
        """Initialize SHAP explainers for loaded models"""
        if self.X_background is None:
            self.load_training_data()
        
        logger.info("🔍 Initializing SHAP explainers...")
        
        # Take a smaller background sample for SHAP (for efficiency)
        background_sample = shap.sample(self.X_background, min(100, len(self.X_background)))
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    # For tree-based models, use TreeExplainer if possible
                    if hasattr(model, 'estimators_') or 'forest' in str(type(model)).lower():
                        self.shap_explainers[model_name] = shap.TreeExplainer(model)
                    else:
                        # Use KernelExplainer for other models
                        self.shap_explainers[model_name] = shap.KernelExplainer(
                            model.predict_proba, background_sample
                        )
                    logger.info(f"✅ SHAP explainer ready for {model_name}")
                else:
                    logger.warning(f"⚠️ Model {model_name} doesn't support probability predictions")
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to create SHAP explainer for {model_name}: {e}")
        
        return len(self.shap_explainers)
    
    def initialize_lime_explainer(self):
        """Initialize LIME explainer"""
        if self.X_background is None:
            self.load_training_data()
        
        try:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_background.values,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode='classification',
                discretize_continuous=True
            )
            logger.info("✅ LIME explainer initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize LIME: {e}")
            return False
    
    def explain_prediction_shap(self, model_name: str, input_data: pd.DataFrame, 
                              plot_type: str = 'waterfall') -> dict:
        """
        Generate SHAP explanation for a specific prediction.
        
        Args:
            model_name: Name of the model to explain
            input_data: Input data to explain (single row)
            plot_type: Type of SHAP plot ('waterfall', 'force', 'bar')
            
        Returns:
            Dictionary containing SHAP values and visualizations
        """
        if model_name not in self.shap_explainers:
            logger.error(f"❌ No SHAP explainer for model: {model_name}")
            return {}
        
        try:
            explainer = self.shap_explainers[model_name]
            shap_values = explainer.shap_values(input_data)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class for binary, or adjust for multi-class
            
            result = {
                'shap_values': shap_values,
                'feature_names': self.feature_names,
                'base_value': explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                'input_values': input_data.iloc[0].values
            }
            
            # Create visualizations
            result['plots'] = self._create_shap_plots(result, plot_type)
            
            logger.info(f"✅ SHAP explanation generated for {model_name}")
            return result
            
        except Exception as e:
            logger.error(f"❌ SHAP explanation failed: {e}")
            return {}
    
    def explain_prediction_lime(self, model_name: str, input_data: pd.DataFrame) -> dict:
        """
        Generate LIME explanation for a specific prediction.
        
        Args:
            model_name: Name of the model to explain
            input_data: Input data to explain (single row)
            
        Returns:
            Dictionary containing LIME explanation
        """
        if self.lime_explainer is None:
            logger.error("❌ LIME explainer not initialized")
            return {}
        
        if model_name not in self.models:
            logger.error(f"❌ Model not found: {model_name}")
            return {}
        
        try:
            model = self.models[model_name]
            
            # Generate LIME explanation
            explanation = self.lime_explainer.explain_instance(
                input_data.iloc[0].values,
                model.predict_proba,
                num_features=len(self.feature_names),
                top_labels=len(self.class_names)
            )
            
            # Extract explanation data
            result = {
                'explanation': explanation,
                'feature_importance': explanation.as_list(),
                'prediction_proba': explanation.predict_proba,
                'intercept': explanation.intercept[1] if len(explanation.intercept) > 1 else explanation.intercept[0]
            }
            
            # Create visualization
            result['plot'] = self._create_lime_plot(result)
            
            logger.info(f"✅ LIME explanation generated for {model_name}")
            return result
            
        except Exception as e:
            logger.error(f"❌ LIME explanation failed: {e}")
            return {}
    
    def _create_shap_plots(self, shap_result: dict, plot_type: str = 'waterfall'):
        """Create SHAP visualization plots"""
        try:
            import matplotlib.pyplot as plt
            plt.style.use('default')
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            if plot_type == 'waterfall':
                # Create waterfall plot data
                shap_values = shap_result['shap_values'][0] if len(shap_result['shap_values'].shape) > 1 else shap_result['shap_values']
                feature_names = shap_result['feature_names']
                
                # Sort by absolute SHAP value
                sorted_idx = np.argsort(np.abs(shap_values))[::-1]
                
                ax.barh(range(len(sorted_idx)), shap_values[sorted_idx], 
                       color=['red' if x < 0 else 'blue' for x in shap_values[sorted_idx]])
                ax.set_yticks(range(len(sorted_idx)))
                ax.set_yticklabels([feature_names[i] for i in sorted_idx])
                ax.set_xlabel('SHAP Value (impact on model output)')
                ax.set_title('SHAP Feature Importance')
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
            plt.tight_layout()
            
            # Convert to plotly for better integration
            return self._matplotlib_to_plotly(fig)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to create SHAP plot: {e}")
            return None
    
    def _create_lime_plot(self, lime_result: dict):
        """Create LIME visualization plot"""
        try:
            feature_importance = lime_result['feature_importance']
            
            # Extract feature names and values
            features = [item[0] for item in feature_importance]
            values = [item[1] for item in feature_importance]
            
            # Create plotly bar chart
            fig = go.Figure(data=[
                go.Bar(
                    y=features,
                    x=values,
                    orientation='h',
                    marker_color=['red' if x < 0 else 'blue' for x in values]
                )
            ])
            
            fig.update_layout(
                title='LIME Feature Importance',
                xaxis_title='Importance Score',
                yaxis_title='Features',
                height=400,
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to create LIME plot: {e}")
            return None
    
    def _matplotlib_to_plotly(self, fig):
        """Convert matplotlib figure to plotly (simplified)"""
        # This is a placeholder - in practice, you might use plotly.tools.mpl_to_plotly
        # or recreate the plot directly in plotly
        return None
    
    def generate_global_explanations(self, model_name: str, sample_size: int = 100):
        """
        Generate global model explanations using SHAP.
        
        Args:
            model_name: Name of the model to explain
            sample_size: Number of samples to use for global explanation
            
        Returns:
            Dictionary containing global explanation data
        """
        if model_name not in self.shap_explainers:
            logger.error(f"❌ No SHAP explainer for model: {model_name}")
            return {}
        
        try:
            # Sample data for global explanation
            sample_data = self.X_background.sample(min(sample_size, len(self.X_background)))
            
            explainer = self.shap_explainers[model_name]
            shap_values = explainer.shap_values(sample_data)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Adjust as needed
            
            # Calculate global feature importance
            global_importance = np.mean(np.abs(shap_values), axis=0)
            
            result = {
                'global_importance': global_importance,
                'feature_names': self.feature_names,
                'sample_data': sample_data,
                'shap_values': shap_values
            }
            
            # Create summary plots
            result['summary_plot'] = self._create_summary_plot(result)
            
            logger.info(f"✅ Global explanation generated for {model_name}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Global explanation failed: {e}")
            return {}
    
    def _create_summary_plot(self, global_result: dict):
        """Create SHAP summary plot"""
        try:
            importance = global_result['global_importance']
            features = global_result['feature_names']
            
            # Create plotly bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=features,
                    y=importance,
                    marker_color='skyblue',
                    text=[f'{x:.3f}' for x in importance],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title='Global Feature Importance (SHAP)',
                xaxis_title='Features',
                yaxis_title='Mean |SHAP Value|',
                height=400,
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to create summary plot: {e}")
            return None
    
    def compare_model_explanations(self, input_data: pd.DataFrame):
        """
        Compare explanations across multiple models for the same input.
        
        Args:
            input_data: Input data to explain (single row)
            
        Returns:
            Dictionary containing comparison data
        """
        logger.info("🔄 Comparing explanations across models...")
        
        results = {}
        
        for model_name in self.models.keys():
            if model_name in self.shap_explainers:
                shap_result = self.explain_prediction_shap(model_name, input_data, 'bar')
                if shap_result:
                    results[model_name] = {
                        'shap_values': shap_result['shap_values'],
                        'model_prediction': self.models[model_name].predict_proba(input_data)[0]
                    }
        
        # Create comparison visualization
        if results:
            comparison_plot = self._create_comparison_plot(results, input_data)
            return {
                'model_results': results,
                'comparison_plot': comparison_plot
            }
        
        return {}
    
    def _create_comparison_plot(self, results: dict, input_data: pd.DataFrame):
        """Create model explanation comparison plot"""
        try:
            # Prepare data for comparison
            models = list(results.keys())
            n_features = len(self.feature_names)
            
            fig = make_subplots(
                rows=len(models), cols=1,
                subplot_titles=[f'{model} Explanations' for model in models],
                vertical_spacing=0.1
            )
            
            for i, (model_name, result) in enumerate(results.items(), 1):
                shap_values = result['shap_values'][0] if len(result['shap_values'].shape) > 1 else result['shap_values']
                
                fig.add_trace(
                    go.Bar(
                        x=self.feature_names,
                        y=shap_values,
                        name=model_name,
                        marker_color=['red' if x < 0 else 'blue' for x in shap_values],
                        showlegend=False
                    ),
                    row=i, col=1
                )
            
            fig.update_layout(
                title='Model Explanation Comparison',
                height=200 * len(models),
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to create comparison plot: {e}")
            return None

def main():
    """Demonstration of explainability system"""
    print("🧪 ExoplanetExplainer Demonstration")
    print("=" * 50)
    
    # Initialize explainer
    explainer = ExoplanetExplainer()
    
    # Load models and data
    n_models = explainer.load_models()
    print(f"✅ Loaded {n_models} models")
    
    explainer.load_training_data()
    print(f"✅ Loaded background data: {len(explainer.X_background)} samples")
    
    # Initialize explainers
    n_shap = explainer.initialize_shap_explainers()
    print(f"✅ Initialized {n_shap} SHAP explainers")
    
    lime_ready = explainer.initialize_lime_explainer()
    print(f"✅ LIME explainer: {'Ready' if lime_ready else 'Failed'}")
    
    # Create sample input for explanation
    sample_input = pd.DataFrame({
        'period': [365.25],
        'radius': [1.0],
        'temperature': [5778],
        'insolation': [1.0],
        'depth': [100],
        'ra': [45.0],
        'dec': [30.0]
    })
    
    print("\n🔍 Generating explanations for sample input...")
    
    # Generate SHAP explanation
    if explainer.shap_explainers:
        model_name = list(explainer.shap_explainers.keys())[0]
        shap_result = explainer.explain_prediction_shap(model_name, sample_input)
        if shap_result:
            print(f"✅ SHAP explanation generated for {model_name}")
    
    # Generate LIME explanation
    if lime_ready and explainer.models:
        model_name = list(explainer.models.keys())[0]
        lime_result = explainer.explain_prediction_lime(model_name, sample_input)
        if lime_result:
            print(f"✅ LIME explanation generated for {model_name}")
    
    print("\n🎯 Explainability system demonstration complete!")

if __name__ == "__main__":
    main()