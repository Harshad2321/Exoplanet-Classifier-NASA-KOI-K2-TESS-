"""
Streamlit Web Application for NASA Space Apps Challenge 2025
"A World Away: Hunting for Exoplanets with AI"

Interactive web interface for exoplanet classification using trained ML models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import io
from pathlib import Path
import sys
import warnings

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

warnings.filterwarnings('ignore')

# Import our prediction module
try:
    from src.predict import ExoplanetPredictor
    PREDICTOR_AVAILABLE = True
except ImportError:
    PREDICTOR_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="🌌 Exoplanet Classifier - NASA Space Apps 2025",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .prediction-result {
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f0f8ff;
        border-radius: 0 10px 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

def load_predictor():
    """Load the exoplanet predictor."""
    if not PREDICTOR_AVAILABLE:
        st.error("❌ Prediction module not available. Please ensure the model is trained and src/predict.py exists.")
        return None
    
    try:
        with st.spinner("🔄 Loading AI model..."):
            predictor = ExoplanetPredictor()
            st.success("✅ Model loaded successfully!")
            return predictor
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.info("💡 Make sure you have trained a model first by running: `python src/train.py`")
        return None

def create_feature_input_form():
    """Create input form for exoplanet features."""
    st.subheader("🔧 Object Parameters")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🌍 Orbital Characteristics**")
        period = st.number_input(
            "Orbital Period (days)", 
            min_value=0.1, max_value=10000.0, value=365.25,
            help="Time it takes for the planet to orbit its star"
        )
        
        radius = st.number_input(
            "Planet Radius (Earth radii)", 
            min_value=0.1, max_value=50.0, value=1.0,
            help="Size of the planet compared to Earth"
        )
        
        temperature = st.number_input(
            "Equilibrium Temperature (K)", 
            min_value=50.0, max_value=3000.0, value=288.0,
            help="Expected temperature of the planet"
        )
        
        insolation = st.number_input(
            "Insolation Flux (Earth = 1.0)", 
            min_value=0.001, max_value=10000.0, value=1.0,
            help="Amount of stellar energy received relative to Earth"
        )
        
        a_over_rstar = st.number_input(
            "a/R* (Semi-major axis / Stellar radius)", 
            min_value=1.0, max_value=1000.0, value=215.0,
            help="Orbital distance relative to star size"
        )
        
        magnitude = st.number_input(
            "Stellar Magnitude", 
            min_value=1.0, max_value=20.0, value=10.0,
            help="Brightness of the host star (lower = brighter)"
        )
    
    with col2:
        st.markdown("**🔭 Transit Characteristics**")
        duration = st.number_input(
            "Transit Duration (hours)", 
            min_value=0.1, max_value=100.0, value=13.0,
            help="How long the planet blocks the star's light"
        )
        
        depth = st.number_input(
            "Transit Depth (ppm)", 
            min_value=1.0, max_value=50000.0, value=84.0,
            help="How much the star's light dims during transit"
        )
        
        impact = st.number_input(
            "Impact Parameter", 
            min_value=0.0, max_value=1.5, value=0.5,
            help="How centrally the planet transits (0 = center, 1 = edge)"
        )
        
        st.markdown("**🌌 Sky Position**")
        ra = st.number_input(
            "Right Ascension (degrees)", 
            min_value=0.0, max_value=360.0, value=180.0,
            help="Celestial longitude coordinate"
        )
        
        dec = st.number_input(
            "Declination (degrees)", 
            min_value=-90.0, max_value=90.0, value=0.0,
            help="Celestial latitude coordinate"
        )
    
    # Create feature dictionary
    features = {
        'period': period,
        'radius': radius,
        'temperature': temperature,
        'insolation': insolation,
        'a_over_rstar': a_over_rstar,
        'duration': duration,
        'depth': depth,
        'impact': impact,
        'ra': ra,
        'dec': dec,
        'magnitude': magnitude
    }
    
    return features

def display_prediction_result(result, features):
    """Display prediction results in a nice format."""
    predicted_class = result['predicted_class']
    confidence = result.get('confidence', 0)
    probabilities = result.get('probabilities', {})
    
    # Main prediction display
    st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
    
    # Class-specific styling
    class_colors = {
        'CONFIRMED': '🪐',
        'CANDIDATE': '🔍', 
        'FALSE_POSITIVE': '❌'
    }
    
    class_descriptions = {
        'CONFIRMED': 'Confirmed Exoplanet',
        'CANDIDATE': 'Planet Candidate',
        'FALSE_POSITIVE': 'False Positive'
    }
    
    emoji = class_colors.get(predicted_class, '❓')
    description = class_descriptions.get(predicted_class, predicted_class)
    
    st.markdown(f"## {emoji} **{description}**")
    st.markdown(f"**Confidence:** {confidence:.1%}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Probability breakdown
    if probabilities:
        st.subheader("📊 Class Probabilities")
        
        # Create probability chart
        prob_df = pd.DataFrame(list(probabilities.items()), columns=['Class', 'Probability'])
        prob_df['Probability'] = prob_df['Probability'] * 100
        prob_df = prob_df.sort_values('Probability', ascending=True)
        
        fig = px.bar(
            prob_df, 
            x='Probability', 
            y='Class',
            orientation='h',
            color='Probability',
            color_continuous_scale='viridis',
            title="Classification Probabilities",
            labels={'Probability': 'Probability (%)'}
        )
        
        fig.update_layout(
            height=300,
            showlegend=False,
            title_x=0.5
        )
        
        fig.update_traces(
            texttemplate='%{x:.1f}%',
            textposition='auto'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def create_sample_selector():
    """Create sample object selector."""
    st.subheader("🎯 Try Sample Objects")
    
    sample_types = {
        'Earth-like Planet': 'earth_like',
        'Hot Jupiter': 'hot_jupiter',
        'Super Earth': 'super_earth',
        'False Positive': 'false_positive'
    }
    
    selected_sample = st.selectbox(
        "Choose a sample object type:",
        list(sample_types.keys())
    )
    
    if st.button("Load Sample", type="secondary"):
        sample_key = sample_types[selected_sample]
        if st.session_state.predictor:
            sample_features = st.session_state.predictor.create_sample_input(sample_key)
            st.success(f"✅ Loaded {selected_sample} sample!")
            return sample_features
        else:
            st.error("Please load the model first!")
    
    return None

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🌌 Exoplanet Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">NASA Space Apps Challenge 2025: "A World Away: Hunting for Exoplanets with AI"</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🚀 Navigation")
        
        # Model loading
        if st.button("🔄 Load AI Model", type="primary", use_container_width=True):
            st.session_state.predictor = load_predictor()
        
        # Model status
        if st.session_state.predictor:
            st.success("✅ Model Loaded")
            model_info = st.session_state.predictor.get_model_info()
            st.info(f"**Model:** {model_info['description']}")
        else:
            st.warning("⚠️ Model Not Loaded")
        
        st.markdown("---")
        
        # About section
        st.markdown("## ℹ️ About")
        st.markdown("""
        This AI-powered tool classifies astronomical objects as:
        - 🪐 **Confirmed Exoplanets**
        - 🔍 **Planet Candidates** 
        - ❌ **False Positives**
        
        Built using NASA's Kepler, K2, and TESS datasets with machine learning.
        """)
    
    # Main content
    if st.session_state.predictor is None:
        st.warning("⚠️ Please load the AI model first using the sidebar.")
        
        # Show instructions
        st.markdown("""
        ## 🚀 Getting Started
        
        1. Click **"Load AI Model"** in the sidebar
        2. Enter exoplanet parameters below or load a sample
        3. Click **"Classify Object"** to get AI prediction
        
        ### 📋 Required Steps Before Using:
        1. Make sure you have NASA data files in the `data/` folder
        2. Run the preprocessing: `python src/preprocess.py`
        3. Train the model: `python src/train.py`
        """)
        return
    
    # Sample selector
    sample_features = create_sample_selector()
    
    st.markdown("---")
    
    # Feature input form
    if sample_features:
        features = sample_features
        st.info("Sample data loaded! You can modify the values below.")
        # Display the sample values in the form
        features = create_feature_input_form()
    else:
        features = create_feature_input_form()
    
    # Prediction button
    if st.button("🚀 Classify Object", type="primary", use_container_width=True):
        with st.spinner("🔄 Analyzing object..."):
            try:
                result = st.session_state.predictor.predict_single(
                    features, return_probabilities=True
                )
                
                # Display results
                display_prediction_result(result, features)
                
                # Add to history
                history_entry = {
                    'features': features,
                    'result': result,
                    'timestamp': pd.Timestamp.now()
                }
                st.session_state.predictions_history.append(history_entry)
                
                # Interpretation
                interpretation = st.session_state.predictor.interpret_prediction(result)
                
                st.subheader("🧠 AI Interpretation")
                st.markdown(interpretation)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        🌌 Built for NASA Space Apps Challenge 2025 | "A World Away: Hunting for Exoplanets with AI" 🪐
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()