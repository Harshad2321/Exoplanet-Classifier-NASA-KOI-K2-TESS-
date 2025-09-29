#!/usr/bin/env python3
"""
🌐 NASA Exoplanet Hunter - Simple Deployment Interface
NASA Space Apps Challenge 2025 - Production Ready
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
    page_title="🌌 NASA Exoplanet Hunter",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for NASA theme
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 1rem;
    background: linear-gradient(90deg, #1e3c72, #2a5298);
    border-radius: 10px;
    margin-bottom: 2rem;
    color: white;
}
.metric-card {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 5px;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

class SimplePredictor:
    """Simple, robust predictor for NASA Space Apps Challenge"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.metadata = None
        self.feature_names = []
        self.is_loaded = False
        
    def load_models(self):
        """Load pre-trained models"""
        models_dir = Path("models")
        
        try:
            # Load the best model (Random Forest)
            model_path = models_dir / "model_random_forest.pkl"
            if model_path.exists():
                self.model = joblib.load(model_path)
                st.success("✅ Model loaded successfully!")
            else:
                st.error("❌ Model file not found!")
                return False
            
            # Load label encoder
            encoder_path = models_dir / "label_encoder.pkl"
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
            
            # Load metadata
            metadata_path = models_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.feature_names = self.metadata.get('feature_names', [])
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False
    
    def prepare_features(self, data):
        """Prepare features for prediction"""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Core features that the model expects
        expected_features = [
            'period', 'prad', 'teq', 'insol', 'dor',
            'srad', 'smass', 'sage', 'steff', 'slogg', 'smet',
            'ra', 'dec', 'score', 'fpflag_nt', 'fpflag_ss', 'fpflag_co'
        ]
        
        # Add missing features with default values
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0.5  # Default neutral value
        
        # Calculate derived features (as in training)
        df['habitable_zone'] = ((df['teq'] >= 200) & (df['teq'] <= 400)).astype(int)
        df['earth_like_size'] = ((df['prad'] >= 0.5) & (df['prad'] <= 2.0)).astype(int)
        df['reasonable_period'] = ((df['period'] >= 10) & (df['period'] <= 500)).astype(int)
        df['high_score'] = (df['score'] > 0.7).astype(int)
        df['no_flags'] = ((df['fpflag_nt'] == 0) & (df['fpflag_ss'] == 0) & (df['fpflag_co'] == 0)).astype(int)
        df['sun_like_star'] = ((df['steff'] >= 5000) & (df['steff'] <= 6000) & 
                              (df['srad'] >= 0.8) & (df['srad'] <= 1.2)).astype(int)
        
        # All features in the expected order
        all_features = expected_features + [
            'habitable_zone', 'earth_like_size', 'reasonable_period',
            'high_score', 'no_flags', 'sun_like_star'
        ]
        
        return df[all_features]
    
    def predict(self, data):
        """Make prediction"""
        if not self.is_loaded:
            return None
        
        try:
            # Prepare features
            X = self.prepare_features(data)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            # Get class names
            if self.label_encoder:
                classes = self.label_encoder.classes_
            else:
                classes = ['CANDIDATE', 'CONFIRMED', 'FALSE_POSITIVE']
            
            # Create result
            prob_dict = {classes[i]: prob for i, prob in enumerate(probabilities)}
            
            return {
                'prediction': prediction,
                'confidence': max(probabilities),
                'probabilities': prob_dict,
                'model_used': 'random_forest'
            }
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None

@st.cache_resource
def get_predictor():
    """Get cached predictor instance"""
    predictor = SimplePredictor()
    predictor.load_models()
    return predictor

def main():
    """Main Streamlit interface"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌌 NASA Exoplanet Hunter</h1>
        <h3>NASA Space Apps Challenge 2025</h3>
        <p>"A World Away: Hunting for Exoplanets with AI"</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load predictor
    predictor = get_predictor()
    
    if not predictor.is_loaded:
        st.error("🚫 **Models not loaded!** Please train the model first by running:")
        st.code("python complete_nasa_training.py")
        return
    
    # Sidebar with model info
    with st.sidebar:
        st.markdown("### 🤖 Model Status")
        st.success("✅ **Ready for Predictions**")
        
        if predictor.metadata:
            st.info(f"🏆 **Best Model**: {predictor.metadata['best_model']}")
            st.info(f"📊 **Accuracy**: {predictor.metadata['best_accuracy']:.1%}")
            st.info(f"🔢 **Features**: {predictor.metadata['n_features']}")
        
        st.markdown("---")
        st.markdown("### 🌟 NASA Challenge")
        st.markdown("""
        **Challenge**: A World Away: Hunting for Exoplanets with AI
        
        **Mission**: Help NASA classify objects detected by Kepler, K2, and TESS missions
        
        **Impact**: Accelerate discovery of potentially habitable exoplanets! 🪐
        """)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔭 Single Classification", "📊 Batch Processing", "ℹ️ About"])
    
    with tab1:
        st.markdown("### 🔭 Classify an Exoplanet Candidate")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🪐 Planetary Parameters")
            period = st.number_input("**Orbital Period** (days)", value=365.25, min_value=0.1, 
                                   help="Time for one orbit around the star")
            prad = st.number_input("**Planet Radius** (Earth radii)", value=1.0, min_value=0.1,
                                 help="Size compared to Earth")
            teq = st.number_input("**Equilibrium Temperature** (K)", value=288.0, min_value=0.1,
                                help="Surface temperature of the planet")
            insol = st.number_input("**Insolation Flux** (Earth flux)", value=1.0, min_value=0.1,
                                  help="Amount of stellar radiation received")
            
        with col2:
            st.markdown("#### ⭐ Stellar Parameters")
            srad = st.number_input("**Stellar Radius** (Solar radii)", value=1.0, min_value=0.1,
                                 help="Size of the host star compared to Sun")
            smass = st.number_input("**Stellar Mass** (Solar masses)", value=1.0, min_value=0.1,
                                  help="Mass of the host star compared to Sun")
            steff = st.number_input("**Stellar Temperature** (K)", value=5778.0, min_value=1000.0,
                                  help="Surface temperature of the host star")
            
        # Additional parameters in expander
        with st.expander("🔬 Additional Parameters"):
            col3, col4 = st.columns(2)
            with col3:
                dor = st.number_input("Distance/Radius Ratio", value=215.0, min_value=1.0)
                sage = st.number_input("Stellar Age (Gyr)", value=4.5, min_value=0.1)
                slogg = st.number_input("Stellar Surface Gravity", value=4.4, min_value=0.0)
                ra = st.number_input("Right Ascension (deg)", value=290.0, min_value=0.0, max_value=360.0)
            with col4:
                smet = st.number_input("Stellar Metallicity", value=0.0, min_value=-2.0, max_value=1.0)
                dec = st.number_input("Declination (deg)", value=42.0, min_value=-90.0, max_value=90.0)
                score = st.slider("Detection Score", 0.0, 1.0, 0.8)
                
        # Classification button
        if st.button("🚀 **Classify Exoplanet**", type="primary"):
            input_data = {
                'period': period, 'prad': prad, 'teq': teq, 'insol': insol,
                'dor': dor, 'srad': srad, 'smass': smass, 'sage': sage,
                'steff': steff, 'slogg': slogg, 'smet': smet,
                'ra': ra, 'dec': dec, 'score': score,
                'fpflag_nt': 0, 'fpflag_ss': 0, 'fpflag_co': 0
            }
            
            with st.spinner("🤖 Analyzing astronomical data..."):
                result = predictor.predict(input_data)
            
            if result:
                st.markdown("---")
                st.markdown("### 🎯 **Classification Result**")
                
                # Result display
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    prediction = result['prediction']
                    if prediction == 'CONFIRMED':
                        st.success(f"🪐 **{prediction}**")
                        st.balloons()
                        st.markdown("*Verified exoplanet!*")
                    elif prediction == 'CANDIDATE':
                        st.warning(f"🔍 **{prediction}**") 
                        st.markdown("*Needs further study*")
                    else:
                        st.error(f"❌ **FALSE POSITIVE**")
                        st.markdown("*Not a real planet*")
                
                with col2:
                    confidence = result['confidence']
                    st.metric("**Confidence**", f"{confidence:.1%}")
                    
                with col3:
                    st.metric("**Model**", "Random Forest")
                
                # Probability chart
                st.markdown("#### 📊 **Classification Probabilities**")
                prob_data = pd.DataFrame([
                    {"Class": k, "Probability": v, "Color": 
                     "green" if k == "CONFIRMED" else "orange" if k == "CANDIDATE" else "red"}
                    for k, v in result['probabilities'].items()
                ])
                
                fig = px.bar(prob_data, x='Class', y='Probability', 
                           title="Confidence by Classification",
                           color='Color', color_discrete_map={
                               "green": "#00ff00", "orange": "#ff8c00", "red": "#ff4444"
                           })
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 📊 Batch Exoplanet Classification")
        st.markdown("Upload a CSV file with multiple candidates for bulk analysis:")
        
        uploaded_file = st.file_uploader("📁 **Choose CSV File**", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded **{len(df)}** objects from file")
                
                # Show preview
                st.markdown("#### 📋 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                if st.button("🚀 **Classify All Objects**", type="primary"):
                    with st.spinner("🤖 Processing batch predictions..."):
                        results = []
                        progress = st.progress(0)
                        
                        for i, row in df.iterrows():
                            result = predictor.predict(row.to_dict())
                            if result:
                                results.append({
                                    'Index': i,
                                    'Prediction': result['prediction'],
                                    'Confidence': result['confidence']
                                })
                            progress.progress((i + 1) / len(df))
                        
                        # Results summary
                        results_df = pd.DataFrame(results)
                        
                        st.markdown("---")
                        st.markdown("### 🎯 **Batch Results Summary**")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            confirmed = len(results_df[results_df['Prediction'] == 'CONFIRMED'])
                            st.metric("🪐 **Confirmed**", confirmed)
                        with col2:
                            candidates = len(results_df[results_df['Prediction'] == 'CANDIDATE'])
                            st.metric("🔍 **Candidates**", candidates)
                        with col3:
                            false_pos = len(results_df[results_df['Prediction'] == 'FALSE_POSITIVE'])
                            st.metric("❌ **False Positives**", false_pos)
                        
                        # Results chart
                        pred_counts = results_df['Prediction'].value_counts()
                        fig = px.pie(values=pred_counts.values, names=pred_counts.index,
                                   title="Distribution of Classifications")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        combined_results = df.copy()
                        combined_results['NASA_Prediction'] = results_df['Prediction'].values
                        combined_results['NASA_Confidence'] = results_df['Confidence'].values
                        
                        csv_buffer = io.StringIO()
                        combined_results.to_csv(csv_buffer, index=False)
                        
                        st.download_button(
                            "💾 **Download Results**",
                            csv_buffer.getvalue(),
                            file_name=f"exoplanet_classifications_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Error processing file: {e}")
        
        else:
            st.info("👆 **Upload a CSV file to begin batch classification**")
    
    with tab3:
        st.markdown("### ℹ️ About NASA Exoplanet Hunter")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 **NASA Space Apps Challenge 2025**")
            st.markdown("""
            **Challenge**: "A World Away: Hunting for Exoplanets with AI"
            
            **Mission**: Develop AI systems to help NASA scientists classify objects 
            detected by space telescopes as confirmed exoplanets, planet candidates, 
            or false positives.
            
            **Impact**: Accelerate the discovery of potentially habitable worlds 
            beyond our solar system! 🌌
            """)
            
            st.markdown("#### 🚀 **How It Works**")
            st.markdown("""
            1. **Input**: Astronomical parameters from NASA missions
            2. **Analysis**: AI model trained on exoplanet characteristics
            3. **Classification**: CONFIRMED, CANDIDATE, or FALSE_POSITIVE
            4. **Result**: Confidence score and detailed analysis
            """)
        
        with col2:
            st.markdown("#### 📊 **Model Performance**")
            if predictor.metadata:
                st.info(f"**Accuracy**: {predictor.metadata['best_accuracy']:.1%}")
                st.info(f"**Model Type**: Random Forest Ensemble")
                st.info(f"**Features**: {predictor.metadata['n_features']} parameters")
                st.info(f"**Training Date**: {predictor.metadata['training_date'][:10]}")
            
            st.markdown("#### 🌟 **Classifications**")
            st.markdown("""
            - **🪐 CONFIRMED**: Verified exoplanets with strong evidence
            - **🔍 CANDIDATE**: Potential exoplanets needing further study  
            - **❌ FALSE_POSITIVE**: Signals that mimic planetary transits
            """)
            
            st.markdown("#### 📡 **Data Sources**")
            st.markdown("""
            - **Kepler Mission**: Original exoplanet hunter
            - **K2 Mission**: Extended Kepler observations
            - **TESS Mission**: All-sky exoplanet survey
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        🌌 NASA Space Apps Challenge 2025 | Made with ❤️ for exoplanet discovery
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()