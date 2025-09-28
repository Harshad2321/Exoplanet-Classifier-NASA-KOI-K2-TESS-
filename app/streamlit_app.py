"""
NASA Space Apps Challenge 2025 - Exoplanet Classifier
Enhanced Streamlit Web Application

Modern, responsive web interface for exoplanet classification with improved UX.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt

# Import core functionality
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core import (
    PROJECT_NAME, PROJECT_VERSION, CHALLENGE_NAME, APP_CONFIG,
    get_prediction_api, get_data_loader, setup_logging,
    ALL_FEATURES, TARGET_MAPPING, MODELS_DIR
)

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title=APP_CONFIG.page_title,
    page_icon=APP_CONFIG.page_icon,
    layout=APP_CONFIG.layout,
    initial_sidebar_state=APP_CONFIG.initial_sidebar_state,
    menu_items={
        'Get Help': 'https://github.com/your-repo/help',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': f'{PROJECT_NAME} v{PROJECT_VERSION} - {CHALLENGE_NAME}'
    }
)

# Initialize logging
logger = setup_logging("streamlit_app")


class ExoplanetApp:
    """Main application class with enhanced features"""
    
    def __init__(self):
        self.prediction_api = get_prediction_api()
        self.data_loader = get_data_loader()
        self.logger = setup_logging("app")
        
        # Initialize session state
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        if 'current_data' not in st.session_state:
            st.session_state.current_data = None
        if 'models_loaded' not in st.session_state:
            st.session_state.models_loaded = False
    
    def render_header(self):
        """Render application header with branding"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <h1 style="color: #1f4e79; margin-bottom: 0;">🌌 {APP_CONFIG.page_title}</h1>
                    <p style="color: #666; font-size: 1.1em; margin-top: 0;">
                        <strong>{CHALLENGE_NAME}</strong><br>
                        <em>"A World Away: Hunting for Exoplanets with AI"</em>
                    </p>
                    <p style="color: #888; font-size: 0.9em;">
                        Powered by Advanced Machine Learning | Version {PROJECT_VERSION}
                    </p>
                </div>
                """, unsafe_allow_html=True
            )
        
        st.markdown("---")
    
    def render_sidebar(self):
        """Render enhanced sidebar with navigation and controls"""
        with st.sidebar:
            st.markdown("## 🚀 Mission Control")
            
            # Navigation
            page = st.selectbox(
                "Choose Your Mission",
                ["🔭 Single Prediction", "📊 Batch Analysis", "🎯 Model Comparison", "📈 Data Explorer"],
                key="navigation"
            )
            
            st.markdown("---")
            
            # System status
            st.markdown("### 🖥️ System Status")
            
            # API health check
            try:
                health = self.prediction_api.health_check()
                status_color = "🟢" if health['status'] == 'healthy' else "🔴"
                st.markdown(f"{status_color} **API Status**: {health['status'].title()}")
                
                # Available models
                available_models = health.get('available_models', 0)
                st.markdown(f"🤖 **Models**: {available_models} available")
                
                # Device info
                device = health.get('device_info', {}).get('current_device', 'unknown')
                gpu_count = health.get('device_info', {}).get('available_gpus', 0)
                device_emoji = "🚀" if device == 'cuda' else "💻"
                st.markdown(f"{device_emoji} **Device**: {device.upper()}")
                if gpu_count > 0:
                    st.markdown(f"⚡ **GPUs**: {gpu_count}")
                
            except Exception as e:
                st.markdown("🔴 **API Status**: Error")
                st.error(f"System check failed: {str(e)}")
            
            st.markdown("---")
            
            # Model selection
            st.markdown("### 🎯 Model Selection")
            models = self._get_available_models()
            if models:
                selected_models = st.multiselect(
                    "Choose Models for Prediction",
                    options=[m['name'] for m in models],
                    default=[models[0]['name']] if models else [],
                    help="Select one or more models for ensemble prediction"
                )
                st.session_state.selected_models = selected_models
            else:
                st.warning("No models available")
                st.session_state.selected_models = []
            
            # Advanced settings
            with st.expander("⚙️ Advanced Settings"):
                st.session_state.confidence_threshold = st.slider(
                    "Confidence Threshold", 0.0, 1.0, 0.5, 0.05
                )
                st.session_state.ensemble_method = st.selectbox(
                    "Ensemble Method", ["soft", "hard"], index=0
                )
                st.session_state.show_probabilities = st.checkbox(
                    "Show Prediction Probabilities", True
                )
                st.session_state.show_feature_importance = st.checkbox(
                    "Show Feature Importance", False
                )
            
            return page
    
    def _get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        try:
            return self.prediction_api.get_available_models()
        except Exception as e:
            self.logger.error(f"Failed to get available models: {e}")
            return []
    
    def render_single_prediction(self):
        """Render single prediction interface"""
        st.header("🔭 Single Exoplanet Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📋 Input Parameters")
            
            # Input method selection
            input_method = st.radio(
                "Choose Input Method",
                ["Manual Entry", "Upload CSV", "Use Example"],
                horizontal=True
            )
            
            if input_method == "Manual Entry":
                input_data = self._render_manual_input()
            elif input_method == "Upload CSV":
                input_data = self._render_csv_upload()
            else:
                input_data = self._render_example_data()
            
            # Prediction button
            if st.button("🚀 Classify Exoplanet", type="primary", key="predict_single"):
                if input_data and st.session_state.selected_models:
                    with st.spinner("🔍 Analyzing celestial data..."):
                        result = self._make_prediction(input_data)
                        if result:
                            st.session_state.current_result = result
                            st.session_state.prediction_history.append({
                                'timestamp': time.time(),
                                'data': input_data,
                                'result': result
                            })
                else:
                    st.error("Please provide input data and select at least one model.")
        
        with col2:
            st.markdown("### 📊 Live Prediction")
            if 'current_result' in st.session_state:
                self._render_prediction_result(st.session_state.current_result)
    
    def _render_manual_input(self) -> Dict[str, float]:
        """Render manual input form"""
        input_data = {}
        
        # Key exoplanet parameters
        key_features = [
            ('koi_period', 'Orbital Period (days)', 365.25, 0.1, 10000.0),
            ('koi_prad', 'Planet Radius (Earth radii)', 1.0, 0.1, 20.0),
            ('koi_teq', 'Equilibrium Temperature (K)', 288, 100, 3000),
            ('koi_insol', 'Insolation (Earth flux)', 1.0, 0.01, 1000.0),
            ('koi_depth', 'Transit Depth (ppm)', 1000, 1, 100000),
            ('koi_duration', 'Transit Duration (hours)', 6.0, 0.1, 24.0)
        ]
        
        for key, label, default, min_val, max_val in key_features:
            input_data[key] = st.number_input(
                label, value=default, min_value=min_val, max_value=max_val,
                help=f"Enter the {label.lower()} for the exoplanet candidate"
            )
        
        # Additional stellar parameters
        with st.expander("⭐ Stellar Parameters (Optional)"):
            stellar_features = [
                ('koi_steff', 'Stellar Effective Temperature (K)', 5778, 2000, 10000),
                ('koi_slogg', 'Stellar Surface Gravity (log g)', 4.44, 3.0, 5.5),
                ('koi_srad', 'Stellar Radius (Solar radii)', 1.0, 0.1, 10.0),
                ('koi_kepmag', 'Kepler Magnitude', 12.0, 5.0, 20.0)
            ]
            
            for key, label, default, min_val, max_val in stellar_features:
                input_data[key] = st.number_input(
                    label, value=default, min_value=min_val, max_value=max_val
                )
        
        return input_data
    
    def _render_csv_upload(self) -> Optional[pd.DataFrame]:
        """Render CSV upload interface"""
        uploaded_file = st.file_uploader(
            "Upload CSV file with exoplanet data",
            type=['csv'],
            help="Upload a CSV file with exoplanet parameters. The first row will be used for prediction."
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(df)} rows with {len(df.columns)} columns")
                
                # Show preview
                st.markdown("**Data Preview:**")
                st.dataframe(df.head())
                
                # Use first row for prediction
                return df.iloc[0].to_dict()
                
            except Exception as e:
                st.error(f"Failed to read CSV file: {str(e)}")
        
        return None
    
    def _render_example_data(self) -> Dict[str, float]:
        """Render example data selection"""
        examples = {
            "Earth-like": {
                'koi_period': 365.25, 'koi_prad': 1.0, 'koi_teq': 288,
                'koi_insol': 1.0, 'koi_depth': 84, 'koi_duration': 13.0,
                'koi_steff': 5778, 'koi_slogg': 4.44, 'koi_srad': 1.0
            },
            "Hot Jupiter": {
                'koi_period': 3.5, 'koi_prad': 11.2, 'koi_teq': 1500,
                'koi_insol': 2000, 'koi_depth': 15000, 'koi_duration': 3.2,
                'koi_steff': 6000, 'koi_slogg': 4.2, 'koi_srad': 1.2
            },
            "Super Earth": {
                'koi_period': 50.0, 'koi_prad': 1.8, 'koi_teq': 400,
                'koi_insol': 5.0, 'koi_depth': 500, 'koi_duration': 8.0,
                'koi_steff': 5200, 'koi_slogg': 4.5, 'koi_srad': 0.8
            }
        }
        
        selected_example = st.selectbox(
            "Choose Example Exoplanet",
            list(examples.keys()),
            help="Select a predefined exoplanet type to see example parameters"
        )
        
        example_data = examples[selected_example]
        
        # Display example parameters
        st.markdown(f"**{selected_example} Parameters:**")
        for key, value in example_data.items():
            st.markdown(f"- **{key}**: {value}")
        
        return example_data
    
    def _make_prediction(self, input_data: Union[Dict, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """Make prediction using selected models"""
        try:
            if len(st.session_state.selected_models) == 1:
                # Single model prediction
                model_path = self._get_model_path(st.session_state.selected_models[0])
                result = self.prediction_api.predict_single(
                    model_path, input_data,
                    return_proba=st.session_state.show_probabilities,
                    return_features=st.session_state.show_feature_importance
                )
            else:
                # Ensemble prediction
                model_paths = [self._get_model_path(name) for name in st.session_state.selected_models]
                result = self.prediction_api.predict_ensemble(
                    model_paths, input_data,
                    voting=st.session_state.ensemble_method,
                    return_individual=True
                )
            
            return result
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            self.logger.error(f"Prediction error: {e}")
            return None
    
    def _get_model_path(self, model_name: str) -> str:
        """Get full path for model name"""
        models = self._get_available_models()
        for model in models:
            if model['name'] == model_name:
                return model['path']
        raise ValueError(f"Model not found: {model_name}")
    
    def _render_prediction_result(self, result: Dict[str, Any]):
        """Render prediction result with visualizations"""
        # Main prediction
        prediction = result['predictions'][0] if result['predictions'] else 0
        prediction_label = "Exoplanet" if prediction == 1 else "Not an Exoplanet"
        
        # Color coding
        color = "green" if prediction == 1 else "red"
        icon = "✅" if prediction == 1 else "❌"
        
        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px; 
                        border: 2px solid {color}; border-radius: 10px; 
                        background-color: {'#d4edda' if prediction == 1 else '#f8d7da'};">
                <h2 style="color: {color}; margin: 0;">{icon} {prediction_label}</h2>
                <p style="margin: 5px 0; font-size: 1.1em;">
                    Model: {result.get('model_name', 'Ensemble')}
                </p>
            </div>
            """, unsafe_allow_html=True
        )
        
        # Confidence and probabilities
        if 'confidence' in result:
            confidence = result['confidence'][0]
            st.metric("Confidence", f"{confidence:.1%}")
            
            # Confidence bar
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=confidence * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Confidence"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': st.session_state.confidence_threshold * 100
                    }
                }
            ))
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        # Probabilities
        if 'probabilities' in result and st.session_state.show_probabilities:
            probs = result['probabilities'][0]
            prob_df = pd.DataFrame({
                'Class': ['Not Exoplanet', 'Exoplanet'],
                'Probability': probs
            })
            
            fig = px.bar(prob_df, x='Class', y='Probability',
                        title="Class Probabilities",
                        color='Probability',
                        color_continuous_scale='viridis')
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        if 'feature_importance' in result and st.session_state.show_feature_importance:
            feat_imp = result['feature_importance']
            if feat_imp:
                feat_df = pd.DataFrame(list(feat_imp.items()), 
                                     columns=['Feature', 'Importance'])
                feat_df = feat_df.sort_values('Importance', ascending=True).tail(10)
                
                fig = px.bar(feat_df, x='Importance', y='Feature', 
                           orientation='h', title="Top 10 Feature Importance")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Processing time
        if 'processing_time' in result:
            st.caption(f"⏱️ Processing time: {result['processing_time']:.3f} seconds")
    
    def render_batch_analysis(self):
        """Render batch analysis interface"""
        st.header("📊 Batch Analysis")
        st.markdown("Upload multiple exoplanet candidates for batch classification")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file for batch analysis",
            type=['csv'],
            help="Upload a CSV file with multiple exoplanet candidates"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(df)} candidates for analysis")
                
                # Show data preview
                with st.expander("📋 Data Preview"):
                    st.dataframe(df.head(10))
                
                # Analysis configuration
                col1, col2 = st.columns(2)
                with col1:
                    batch_size = st.slider("Batch Size", 1, min(100, len(df)), 10)
                with col2:
                    analysis_models = st.multiselect(
                        "Models for Analysis",
                        options=[m['name'] for m in self._get_available_models()],
                        default=st.session_state.selected_models
                    )
                
                if st.button("🚀 Start Batch Analysis", type="primary"):
                    if analysis_models:
                        self._run_batch_analysis(df, analysis_models, batch_size)
                    else:
                        st.error("Please select at least one model for analysis")
                        
            except Exception as e:
                st.error(f"Failed to load data: {str(e)}")
    
    def _run_batch_analysis(self, df: pd.DataFrame, models: List[str], batch_size: int):
        """Run batch analysis on uploaded data"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.empty()
        
        results = []
        
        try:
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                status_text.text(f"Processing batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}...")
                
                for idx, row in batch.iterrows():
                    try:
                        # Make prediction
                        result = self._make_prediction(row.to_dict())
                        if result:
                            results.append({
                                'Index': idx,
                                'Prediction': result['predictions'][0],
                                'Confidence': result.get('confidence', [0])[0],
                                'Model': result.get('model_name', 'Ensemble')
                            })
                    except Exception as e:
                        self.logger.error(f"Batch prediction failed for row {idx}: {e}")
                
                progress_bar.progress(min(1.0, (i + batch_size) / len(df)))
            
            # Display results
            if results:
                results_df = pd.DataFrame(results)
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    exoplanet_count = (results_df['Prediction'] == 1).sum()
                    st.metric("Exoplanets Found", exoplanet_count)
                with col2:
                    avg_confidence = results_df['Confidence'].mean()
                    st.metric("Average Confidence", f"{avg_confidence:.1%}")
                with col3:
                    success_rate = len(results) / len(df)
                    st.metric("Processing Success", f"{success_rate:.1%}")
                
                # Results table
                st.markdown("### 📋 Analysis Results")
                
                # Add original data
                display_df = df.copy()
                display_df['Prediction'] = results_df['Prediction'].map({1: 'Exoplanet', 0: 'Not Exoplanet'})
                display_df['Confidence'] = results_df['Confidence'].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(display_df, use_container_width=True)
                
                # Visualization
                fig = px.histogram(results_df, x='Prediction', 
                                 title="Prediction Distribution",
                                 labels={'Prediction': 'Classification', 'count': 'Number of Candidates'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name=f"exoplanet_analysis_{int(time.time())}.csv",
                    mime="text/csv"
                )
            
            status_text.success("✅ Batch analysis completed!")
            
        except Exception as e:
            st.error(f"Batch analysis failed: {str(e)}")
            self.logger.error(f"Batch analysis error: {e}")
    
    def render_model_comparison(self):
        """Render model comparison interface"""
        st.header("🎯 Model Comparison")
        st.markdown("Compare performance of different models on the same data")
        
        # Model selection for comparison
        models = self._get_available_models()
        if len(models) < 2:
            st.warning("Need at least 2 models for comparison")
            return
        
        comparison_models = st.multiselect(
            "Select Models to Compare",
            options=[m['name'] for m in models],
            default=[models[0]['name'], models[1]['name']] if len(models) >= 2 else [],
            help="Choose 2 or more models to compare their predictions"
        )
        
        if len(comparison_models) >= 2:
            # Test data input
            test_method = st.radio(
                "Test Data Source",
                ["Use Example Data", "Upload Test Set"],
                horizontal=True
            )
            
            if test_method == "Use Example Data":
                test_data = self._get_comparison_test_data()
            else:
                uploaded_file = st.file_uploader("Upload test dataset", type=['csv'])
                if uploaded_file:
                    test_data = pd.read_csv(uploaded_file)
                else:
                    test_data = None
            
            if test_data is not None and st.button("🔍 Compare Models"):
                self._run_model_comparison(comparison_models, test_data)
    
    def _get_comparison_test_data(self) -> pd.DataFrame:
        """Generate test data for model comparison"""
        test_cases = [
            {"name": "Earth-like", "koi_period": 365, "koi_prad": 1.0, "koi_teq": 288, "expected": 1},
            {"name": "Hot Jupiter", "koi_period": 3.5, "koi_prad": 11.2, "koi_teq": 1500, "expected": 1},
            {"name": "False Positive 1", "koi_period": 0.5, "koi_prad": 0.1, "koi_teq": 2000, "expected": 0},
            {"name": "Super Earth", "koi_period": 50, "koi_prad": 1.8, "koi_teq": 400, "expected": 1},
            {"name": "False Positive 2", "koi_period": 1000, "koi_prad": 20, "koi_teq": 100, "expected": 0},
        ]
        
        # Fill in missing features with defaults
        for case in test_cases:
            case.update({
                "koi_insol": 1.0,
                "koi_depth": 1000,
                "koi_duration": 6.0,
                "koi_steff": 5778,
                "koi_slogg": 4.44,
                "koi_srad": 1.0
            })
        
        return pd.DataFrame(test_cases)
    
    def _run_model_comparison(self, models: List[str], test_data: pd.DataFrame):
        """Run comparison between selected models"""
        comparison_results = []
        
        progress_bar = st.progress(0)
        
        for i, model_name in enumerate(models):
            try:
                model_path = self._get_model_path(model_name)
                
                for idx, row in test_data.iterrows():
                    result = self.prediction_api.predict_single(
                        model_path, row.to_dict(), return_proba=True
                    )
                    
                    comparison_results.append({
                        'Model': model_name,
                        'Test_Case': row.get('name', f'Case_{idx}'),
                        'Prediction': result['predictions'][0],
                        'Confidence': result.get('confidence', [0])[0],
                        'Expected': row.get('expected', None)
                    })
                
                progress_bar.progress((i + 1) / len(models))
                
            except Exception as e:
                st.error(f"Failed to test model {model_name}: {str(e)}")
        
        if comparison_results:
            results_df = pd.DataFrame(comparison_results)
            
            # Pivot table for comparison
            comparison_pivot = results_df.pivot_table(
                index='Test_Case', 
                columns='Model', 
                values=['Prediction', 'Confidence'],
                aggfunc='first'
            )
            
            st.markdown("### 📊 Model Comparison Results")
            st.dataframe(comparison_pivot, use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy comparison (if expected values available)
                if 'Expected' in test_data.columns:
                    accuracy_data = []
                    for model in models:
                        model_results = results_df[results_df['Model'] == model]
                        correct = (model_results['Prediction'] == model_results['Expected']).sum()
                        accuracy = correct / len(model_results)
                        accuracy_data.append({'Model': model, 'Accuracy': accuracy})
                    
                    acc_df = pd.DataFrame(accuracy_data)
                    fig = px.bar(acc_df, x='Model', y='Accuracy', 
                               title="Model Accuracy Comparison")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence distribution
                fig = px.box(results_df, x='Model', y='Confidence',
                           title="Confidence Distribution by Model")
                st.plotly_chart(fig, use_container_width=True)
    
    def render_data_explorer(self):
        """Render data exploration interface"""
        st.header("📈 Data Explorer")
        st.markdown("Explore the exoplanet dataset and understand feature distributions")
        
        # Load sample data for exploration
        try:
            # Try to load from data directory
            sample_data = self._load_sample_data()
            if sample_data is not None:
                self._render_data_analysis(sample_data)
            else:
                st.info("No sample data available for exploration. Upload a dataset to explore.")
                
                uploaded_file = st.file_uploader("Upload dataset for exploration", type=['csv'])
                if uploaded_file:
                    sample_data = pd.read_csv(uploaded_file)
                    self._render_data_analysis(sample_data)
                    
        except Exception as e:
            st.error(f"Data exploration failed: {str(e)}")
    
    def _load_sample_data(self) -> Optional[pd.DataFrame]:
        """Load sample data for exploration"""
        # Try to find existing data files
        data_files = list(Path("data").glob("*.csv")) if Path("data").exists() else []
        if not data_files:
            return None
        
        try:
            # Load first available data file
            return pd.read_csv(data_files[0]).head(1000)  # Sample for performance
        except:
            return None
    
    def _render_data_analysis(self, data: pd.DataFrame):
        """Render data analysis visualizations"""
        st.success(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")
        
        # Basic statistics
        st.markdown("### 📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", data.shape[0])
        with col2:
            st.metric("Features", data.shape[1])
        with col3:
            numeric_cols = data.select_dtypes(include=[np.number]).shape[1]
            st.metric("Numeric Features", numeric_cols)
        with col4:
            missing_values = data.isnull().sum().sum()
            st.metric("Missing Values", missing_values)
        
        # Feature selection for analysis
        st.markdown("### 🔍 Feature Analysis")
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_feature = st.selectbox(
                    "Select Feature for Distribution Analysis",
                    numeric_columns
                )
            
            with col2:
                chart_type = st.selectbox(
                    "Chart Type",
                    ["Histogram", "Box Plot", "Violin Plot"]
                )
            
            # Feature distribution
            if chart_type == "Histogram":
                fig = px.histogram(data, x=selected_feature, 
                                 title=f"{selected_feature} Distribution")
            elif chart_type == "Box Plot":
                fig = px.box(data, y=selected_feature,
                           title=f"{selected_feature} Box Plot")
            else:
                fig = px.violin(data, y=selected_feature,
                              title=f"{selected_feature} Violin Plot")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation analysis
            if len(numeric_columns) > 1:
                st.markdown("### 🔗 Feature Correlations")
                
                corr_features = st.multiselect(
                    "Select Features for Correlation Analysis",
                    numeric_columns,
                    default=numeric_columns[:5] if len(numeric_columns) >= 5 else numeric_columns
                )
                
                if len(corr_features) >= 2:
                    corr_matrix = data[corr_features].corr()
                    
                    fig = px.imshow(corr_matrix, 
                                  title="Feature Correlation Matrix",
                                  color_continuous_scale='RdBu',
                                  aspect="auto")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Data quality report
        with st.expander("📋 Data Quality Report"):
            quality_report = {
                'Feature': [],
                'Data Type': [],
                'Missing Values': [],
                'Unique Values': [],
                'Memory Usage (KB)': []
            }
            
            for col in data.columns:
                quality_report['Feature'].append(col)
                quality_report['Data Type'].append(str(data[col].dtype))
                quality_report['Missing Values'].append(data[col].isnull().sum())
                quality_report['Unique Values'].append(data[col].nunique())
                quality_report['Memory Usage (KB)'].append(round(data[col].memory_usage(deep=True) / 1024, 2))
            
            quality_df = pd.DataFrame(quality_report)
            st.dataframe(quality_df, use_container_width=True)
    
    def render_prediction_history(self):
        """Render prediction history in sidebar"""
        if st.session_state.prediction_history:
            with st.sidebar:
                st.markdown("---")
                st.markdown("### 📜 Recent Predictions")
                
                for i, entry in enumerate(reversed(st.session_state.prediction_history[-5:])):
                    timestamp = time.strftime('%H:%M:%S', time.localtime(entry['timestamp']))
                    prediction = entry['result']['predictions'][0] if entry['result']['predictions'] else 0
                    result_text = "Exoplanet" if prediction == 1 else "Not Exoplanet"
                    confidence = entry['result'].get('confidence', [0])[0]
                    
                    st.markdown(f"""
                    **{timestamp}**  
                    {result_text} ({confidence:.1%})
                    """)
                    
                    if i < 4:  # Don't add separator after last item
                        st.markdown("---")
                
                if st.button("🗑️ Clear History"):
                    st.session_state.prediction_history = []
                    st.experimental_rerun()
    
    def run(self):
        """Main application runner"""
        try:
            # Render header
            self.render_header()
            
            # Render sidebar and get selected page
            selected_page = self.render_sidebar()
            
            # Render prediction history
            self.render_prediction_history()
            
            # Render main content based on selected page
            if selected_page == "🔭 Single Prediction":
                self.render_single_prediction()
            elif selected_page == "📊 Batch Analysis":
                self.render_batch_analysis()
            elif selected_page == "🎯 Model Comparison":
                self.render_model_comparison()
            elif selected_page == "📈 Data Explorer":
                self.render_data_explorer()
            
            # Footer
            st.markdown("---")
            st.markdown(
                """
                <div style="text-align: center; color: #666; font-size: 0.8em;">
                    Developed for NASA Space Apps Challenge 2025 | 
                    Powered by Advanced Machine Learning & Streamlit
                </div>
                """, unsafe_allow_html=True
            )
            
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            self.logger.error(f"App error: {e}")
            
            # Show error details in expander for debugging
            with st.expander("🔧 Error Details"):
                st.exception(e)


# Application entry point
def main():
    """Main entry point for the Streamlit application"""
    try:
        app = ExoplanetApp()
        app.run()
    except Exception as e:
        st.error(f"Failed to initialize application: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()