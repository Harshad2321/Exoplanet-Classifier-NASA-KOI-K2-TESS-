#!/usr/bin/env python3
"""
🌐 NASA Exoplanet Hunter - Deployment Interface
NASA Space Apps Challenge 2025 - Production Deployment

This is the main deployment interface that loads pre-trained models
for instant exoplanet classification without requiring training.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io

# Configure Streamlit page
st.set_page_config(
    page_title="NASA Exoplanet Hunter",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ProductionPredictor:
    """
    🚀 Production-ready exoplanet classifier
    Loads pre-trained models for instant predictions
    """
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.metadata = None
        self.feature_names = []
        self.is_loaded = False
        
    def load_models(self):
        """Load all pre-trained components"""
        try:
            # Load scaler
            scaler_path = self.models_dir / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            
            # Load label encoder
            encoder_path = self.models_dir / "label_encoder.pkl"
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
            
            # Load metadata
            metadata_path = self.models_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.feature_names = self.metadata.get('feature_names', [])
            
            # Load all models
            for model_file in self.models_dir.glob("model_*.pkl"):
                model_name = model_file.stem.replace("model_", "")
                self.models[model_name] = joblib.load(model_file)
            
            self.is_loaded = len(self.models) > 0
            return self.is_loaded
            
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False
    
    def predict(self, data):
        """Make prediction on input data"""
        if not self.is_loaded:
            return None
        
        try:
            # Ensure data has the right features
            df = pd.DataFrame([data] if isinstance(data, dict) else data)
            
            # Select and order features
            missing_features = set(self.feature_names) - set(df.columns)
            for feature in missing_features:
                df[feature] = 0  # Default value for missing features
            
            # Select only required features in correct order
            X = df[self.feature_names]
            
            # Get best model
            best_model_name = self.metadata.get('best_model', list(self.models.keys())[0])
            model = self.models[best_model_name]
            
            # Scale data if needed
            if best_model_name in ['logistic_regression', 'svm'] and self.scaler:
                X_processed = self.scaler.transform(X)
            else:
                X_processed = X
            
            # Make prediction
            prediction_encoded = model.predict(X_processed)[0]
            probabilities = model.predict_proba(X_processed)[0]
            
            # Decode prediction
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
            
            # Create probability dict
            prob_dict = {}
            for i, class_name in enumerate(self.label_encoder.classes_):
                prob_dict[class_name] = probabilities[i]
            
            return {
                'prediction': prediction,
                'confidence': max(probabilities),
                'probabilities': prob_dict,
                'model_used': best_model_name
            }
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None

# Initialize predictor
@st.cache_resource
def load_predictor():
    """Load the production predictor (cached)"""
    predictor = ProductionPredictor()
    if predictor.load_models():
        return predictor
    return None

def main():
    """Main Streamlit interface"""
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(90deg, #1e3c72, #2a5298); border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0;">🌌 NASA Exoplanet Hunter</h1>
        <h3 style="color: #e0e0e0; margin: 0;">NASA Space Apps Challenge 2025</h3>
        <p style="color: #b0b0b0; margin: 0;">AI-Powered Exoplanet Classification System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load predictor
    predictor = load_predictor()
    
    if not predictor:
        st.error("🚫 **Models Not Found!**")
        st.markdown("""
        ### 🔧 Setup Required
        
        To use this interface, you need to train the production model first:
        
        ```bash
        python train_production_model.py
        ```
        
        This will:
        1. 📊 Load NASA datasets from `data/raw/`
        2. 🤖 Train ensemble models
        3. 💾 Save production-ready models
        4. ✅ Make this interface ready to use
        """)
        
        if st.button("🔄 Refresh Page"):
            st.experimental_rerun()
        
        return
    
    # Model info sidebar
    with st.sidebar:
        st.markdown("### 🤖 Model Information")
        
        if predictor.metadata:
            st.success(f"✅ **Status**: Ready")
            st.info(f"🏆 **Best Model**: {predictor.metadata['best_model']}")
            st.info(f"📊 **Accuracy**: {predictor.metadata['best_accuracy']:.1%}")
            st.info(f"📅 **Trained**: {predictor.metadata['training_date'][:10]}")
            st.info(f"🔢 **Features**: {predictor.metadata['n_features']}")
            
            # Model performance
            st.markdown("### 📈 Model Performance")
            scores_df = pd.DataFrame([
                {"Model": name.title(), "Accuracy": f"{score:.1%}"}
                for name, score in predictor.metadata['all_scores'].items()
            ])
            st.dataframe(scores_df, hide_index=True)
        
        st.markdown("---")
        st.markdown("### 🌟 Features")
        st.markdown("""
        - 🔭 **Single Prediction**
        - 📊 **Batch Processing** 
        - 💾 **Instant Results**
        - 🎯 **NASA Datasets**
        """)
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["🔭 Single Prediction", "📊 Batch Processing", "📈 About Model"])
    
    with tab1:
        st.markdown("### 🔭 Single Exoplanet Classification")
        st.markdown("Enter the parameters of your astronomical object below:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🪐 Planetary Parameters")
            koi_period = st.number_input("Orbital Period (days)", value=365.25, min_value=0.1, help="Time for one complete orbit")
            koi_prad = st.number_input("Planet Radius (Earth radii)", value=1.0, min_value=0.1, help="Radius compared to Earth")
            koi_teq = st.number_input("Equilibrium Temperature (K)", value=288.0, min_value=0.1, help="Planet's equilibrium temperature")
            koi_insol = st.number_input("Insolation Flux (Earth flux)", value=1.0, min_value=0.1, help="Stellar flux received")
            
        with col2:
            st.markdown("#### ⭐ Stellar Parameters")
            koi_srad = st.number_input("Stellar Radius (Solar radii)", value=1.0, min_value=0.1, help="Star radius compared to Sun")
            koi_dor = st.number_input("Distance/Stellar Radius Ratio", value=215.0, min_value=1.0, help="Semi-major axis to stellar radius ratio")
            ra = st.number_input("Right Ascension (deg)", value=290.0, min_value=0.0, max_value=360.0, help="Sky position coordinate")
            dec = st.number_input("Declination (deg)", value=42.0, min_value=-90.0, max_value=90.0, help="Sky position coordinate")
        
        # Additional parameters
        with st.expander("🔬 Advanced Parameters (Optional)"):
            col3, col4 = st.columns(2)
            with col3:
                koi_score = st.slider("KOI Score", 0.0, 1.0, 0.5, help="Kepler Object of Interest score")
                koi_smass = st.number_input("Stellar Mass (Solar masses)", value=1.0, min_value=0.1, help="Star mass compared to Sun")
            with col4:
                koi_sage = st.number_input("Stellar Age (Gyr)", value=4.5, min_value=0.1, help="Age of the host star")
                koi_steff = st.number_input("Stellar Temperature (K)", value=5778.0, min_value=1000.0, help="Surface temperature of host star")
        
        # Predict button
        if st.button("🚀 Classify Exoplanet", type="primary", use_container_width=True):
            # Prepare input data
            input_data = {
                'koi_period': koi_period,
                'koi_prad': koi_prad,
                'koi_teq': koi_teq,
                'koi_insol': koi_insol,
                'koi_srad': koi_srad,
                'koi_dor': koi_dor,
                'ra': ra,
                'dec': dec,
                'koi_score': koi_score,
                'koi_smass': koi_smass,
                'koi_sage': koi_sage,
                'koi_steff': koi_steff
            }
            
            # Make prediction
            with st.spinner("🤖 Analyzing astronomical data..."):
                result = predictor.predict(input_data)
            
            if result:
                # Display result
                st.markdown("---")
                st.markdown("### 🎯 Classification Result")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    prediction = result['prediction']
                    if prediction == 'CONFIRMED':
                        st.success(f"🪐 **{prediction}**")
                        st.balloons()
                    elif prediction == 'CANDIDATE':
                        st.warning(f"🔍 **{prediction}**")
                    else:
                        st.error(f"❌ **{prediction}**")
                
                with col2:
                    confidence = result['confidence']
                    st.metric("Confidence", f"{confidence:.1%}")
                
                with col3:
                    st.info(f"🤖 Model: {result['model_used'].title()}")
                
                # Probability visualization
                st.markdown("#### 📊 Classification Probabilities")
                prob_df = pd.DataFrame([
                    {"Class": k, "Probability": v}
                    for k, v in result['probabilities'].items()
                ])
                
                fig = px.bar(prob_df, x='Class', y='Probability', 
                           title="Classification Confidence by Class",
                           color='Probability',
                           color_continuous_scale='viridis')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 📊 Batch Exoplanet Classification")
        st.markdown("Upload a CSV file with multiple astronomical objects for bulk classification:")
        
        # File upload
        uploaded_file = st.file_uploader(
            "📁 Choose CSV file", 
            type=['csv'],
            help="Upload a CSV file with exoplanet parameters"
        )
        
        if uploaded_file:
            try:
                # Load data
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ Loaded {len(df)} objects from file")
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("🚀 Classify All Objects", type="primary"):
                    with st.spinner("🤖 Processing batch predictions..."):
                        # Process predictions
                        results = []
                        progress_bar = st.progress(0)
                        
                        for i, row in df.iterrows():
                            result = predictor.predict(row.to_dict())
                            if result:
                                results.append({
                                    'index': i,
                                    'prediction': result['prediction'],
                                    'confidence': result['confidence']
                                })
                            progress_bar.progress((i + 1) / len(df))
                        
                        # Create results dataframe
                        results_df = pd.DataFrame(results)
                        df_with_results = df.copy()
                        df_with_results['prediction'] = results_df['prediction'].values
                        df_with_results['confidence'] = results_df['confidence'].values
                        
                        st.markdown("---")
                        st.markdown("### 🎯 Batch Classification Results")
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            confirmed = len(results_df[results_df['prediction'] == 'CONFIRMED'])
                            st.metric("🪐 Confirmed Exoplanets", confirmed)
                        
                        with col2:
                            candidates = len(results_df[results_df['prediction'] == 'CANDIDATE'])
                            st.metric("🔍 Candidates", candidates)
                        
                        with col3:
                            false_pos = len(results_df[results_df['prediction'] == 'FALSE_POSITIVE'])
                            st.metric("❌ False Positives", false_pos)
                        
                        # Results visualization
                        st.markdown("#### 📈 Classification Distribution")
                        pred_counts = results_df['prediction'].value_counts()
                        
                        fig = px.pie(values=pred_counts.values, names=pred_counts.index,
                                   title="Distribution of Classifications")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display results table
                        st.markdown("#### 📋 Detailed Results")
                        st.dataframe(df_with_results, use_container_width=True)
                        
                        # Download results
                        csv_buffer = io.StringIO()
                        df_with_results.to_csv(csv_buffer, index=False)
                        
                        st.download_button(
                            label="💾 Download Results CSV",
                            data=csv_buffer.getvalue(),
                            file_name=f"exoplanet_classifications_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
        
        else:
            st.info("👆 Upload a CSV file to begin batch classification")
            
            # Sample CSV format
            st.markdown("#### 📝 Expected CSV Format")
            sample_data = {
                'koi_period': [365.25, 87.97, 225.0],
                'koi_prad': [1.0, 0.38, 0.95], 
                'koi_teq': [288, 700, 462],
                'koi_insol': [1.0, 6.67, 1.91],
                'ra': [290.0, 45.2, 180.5],
                'dec': [42.0, -16.7, 23.1]
            }
            sample_df = pd.DataFrame(sample_data)
            st.dataframe(sample_df, use_container_width=True)
    
    with tab3:
        st.markdown("### 📈 About the Model")
        
        if predictor.metadata:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Model Details")
                st.info(f"**Training Date**: {predictor.metadata['training_date'][:10]}")
                st.info(f"**Dataset Size**: {predictor.metadata['dataset_size']:,} objects")
                st.info(f"**Number of Features**: {predictor.metadata['n_features']}")
                st.info(f"**Classes**: {', '.join(predictor.metadata['classes'])}")
                
                st.markdown("#### 🏆 Performance Metrics")
                for model, score in predictor.metadata['all_scores'].items():
                    st.metric(f"{model.title()}", f"{score:.1%}")
            
            with col2:
                st.markdown("#### 🔬 Features Used")
                features_text = "\n".join([f"• {feature}" for feature in predictor.feature_names[:15]])
                st.text(features_text)
                if len(predictor.feature_names) > 15:
                    st.text(f"... and {len(predictor.feature_names) - 15} more")
        
        st.markdown("---")
        st.markdown("""
        #### 🌌 NASA Space Apps Challenge 2025
        
        **Challenge**: "A World Away: Hunting for Exoplanets with AI"
        
        This system classifies astronomical objects into:
        - **🪐 CONFIRMED**: Verified exoplanets
        - **🔍 CANDIDATE**: Potential exoplanets requiring further study  
        - **❌ FALSE_POSITIVE**: Objects that mimic planetary signals
        
        **Data Sources**: NASA's Kepler, K2, and TESS missions
        
        **Impact**: Accelerate exoplanet discovery and help NASA scientists 
        identify potentially habitable worlds beyond our solar system.
        """)

if __name__ == "__main__":
    main()