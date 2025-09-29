#!/usr/bin/env python3
"""
🌐 NASA Exoplanet Hunter - Fixed Deployment Interface
NASA Space Apps Challenge 2025 - Production Deployment with Fixed CSV Handling

Fixed version with robust error handling and deterministic predictions.
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
import warnings
warnings.filterwarnings('ignore')

# 🔒 DETERMINISTIC SETTINGS
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def set_deterministic_behavior():
    """Ensure deterministic behavior for consistent predictions"""
    np.random.seed(RANDOM_SEED)
    import random
    random.seed(RANDOM_SEED)
    import os
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# Set deterministic behavior
set_deterministic_behavior()

# Configure Streamlit page
st.set_page_config(
    page_title="NASA Exoplanet Hunter",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

class RobustCSVLoader:
    """🔧 Robust CSV loader with comprehensive error handling"""
    
    @staticmethod
    def load_csv_safely(file, encoding_attempts=['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']):
        """Load CSV with multiple encoding attempts and error handling"""
        for encoding in encoding_attempts:
            try:
                file.seek(0)
                
                # Try different parsing strategies
                strategies = [
                    # Standard parsing
                    lambda: pd.read_csv(file, encoding=encoding),
                    # Skip bad lines with warning
                    lambda: pd.read_csv(file, encoding=encoding, on_bad_lines='skip'),
                    # Use python engine with automatic separator detection
                    lambda: pd.read_csv(file, encoding=encoding, sep=None, engine='python', on_bad_lines='skip'),
                    # Read with quotes handling
                    lambda: pd.read_csv(file, encoding=encoding, quotechar='"', skipinitialspace=True, on_bad_lines='skip'),
                    # Try different delimiters
                    lambda: pd.read_csv(file, encoding=encoding, delimiter=';', on_bad_lines='skip'),
                    lambda: pd.read_csv(file, encoding=encoding, delimiter='\t', on_bad_lines='skip'),
                    # Last resort with minimal constraints
                    lambda: pd.read_csv(file, encoding=encoding, header=0, on_bad_lines='skip', engine='python', sep=None)
                ]
                
                for i, strategy in enumerate(strategies):
                    try:
                        file.seek(0)
                        df = strategy()
                        if not df.empty:
                            strategy_name = ['Standard', 'Skip bad lines', 'Auto separator', 'Quote handling', 'Semicolon sep', 'Tab sep', 'Minimal constraints'][i]
                            st.success(f"✅ Successfully loaded CSV using: {strategy_name} with {encoding} encoding")
                            return df, None
                    except Exception as strategy_error:
                        continue
                        
            except Exception as encoding_error:
                continue
        
        # If all strategies fail, try manual line processing
        try:
            file.seek(0)
            content = file.read()
            
            # Try different decodings
            for encoding in encoding_attempts:
                try:
                    decoded_content = content.decode(encoding)
                    lines = decoded_content.split('\n')
                    
                    # Clean lines and filter out problematic ones
                    clean_lines = []
                    header = None
                    
                    for i, line in enumerate(lines):
                        line = line.strip()
                        if not line:
                            continue
                            
                        if i == 0 or header is None:
                            header = line
                            clean_lines.append(line)
                            expected_fields = len(line.split(','))
                        else:
                            fields = line.split(',')
                            if len(fields) == expected_fields:
                                clean_lines.append(line)
                            elif len(fields) > expected_fields:
                                # Truncate extra fields
                                clean_lines.append(','.join(fields[:expected_fields]))
                    
                    if len(clean_lines) > 1:
                        from io import StringIO
                        clean_data = StringIO('\n'.join(clean_lines))
                        df = pd.read_csv(clean_data)
                        st.warning(f"⚠️ Loaded CSV with manual line processing ({encoding} encoding). Some problematic lines were cleaned or skipped.")
                        return df, f"Manual processing with {encoding} encoding"
                        
                except UnicodeDecodeError:
                    continue
                    
        except Exception as final_error:
            return None, f"Failed to load CSV: {final_error}"
        
        return None, "Unable to parse CSV file with any method. Please check file format and encoding."

class SimpleEDA:
    """📊 Simple EDA without complex dependencies"""
    
    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
    def basic_info(self):
        """Display basic dataset information"""
        st.markdown("### 📋 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🔢 Total Rows", f"{len(self.df):,}")
        with col2:
            st.metric("📊 Total Columns", len(self.df.columns))
        with col3:
            st.metric("🔢 Numeric Columns", len(self.numeric_cols))
        with col4:
            st.metric("📝 Text Columns", len(self.categorical_cols))
        
        # Data types
        st.markdown("#### 🏷️ Column Information")
        dtype_df = pd.DataFrame({
            'Column': self.df.columns,
            'Data Type': self.df.dtypes.astype(str),
            'Non-Null Count': self.df.count(),
            'Null Count': self.df.isnull().sum(),
            'Null %': (self.df.isnull().sum() / len(self.df) * 100).round(2)
        })
        st.dataframe(dtype_df, use_container_width=True)
        
    def data_quality(self):
        """Simple data quality analysis"""
        st.markdown("### 🔍 Data Quality Analysis")
        
        # Missing values
        missing_count = self.df.isnull().sum().sum()
        if missing_count > 0:
            st.markdown("#### 🕳️ Missing Values")
            missing_summary = self.df.isnull().sum().sort_values(ascending=False)
            missing_summary = missing_summary[missing_summary > 0]
            
            if not missing_summary.empty:
                fig = px.bar(
                    x=missing_summary.index,
                    y=missing_summary.values,
                    title="Missing Values by Column",
                    labels={'x': 'Columns', 'y': 'Missing Count'}
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found!")
        
        # Duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            st.warning(f"⚠️ Found {duplicates} duplicate rows ({duplicates/len(self.df)*100:.2f}%)")
        else:
            st.success("✅ No duplicate rows found!")
            
    def simple_stats(self):
        """Display simple statistics"""
        if not self.numeric_cols:
            st.info("No numeric columns found for statistical analysis")
            return
            
        st.markdown("### 📈 Statistical Summary")
        desc_stats = self.df[self.numeric_cols].describe()
        st.dataframe(desc_stats, use_container_width=True)
        
        # Simple visualizations
        st.markdown("### 📊 Data Distributions")
        
        # Select a few columns to visualize
        cols_to_plot = self.numeric_cols[:4]  # Limit to first 4 columns
        
        for col in cols_to_plot:
            fig = px.histogram(self.df, x=col, title=f'Distribution of {col}', 
                             marginal="box", hover_data=self.df.columns)
            st.plotly_chart(fig, use_container_width=True)

class ProductionPredictor:
    """🚀 Production-ready exoplanet classifier with deterministic predictions"""
    
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
            le_path = self.models_dir / "label_encoder.pkl"
            if le_path.exists():
                self.label_encoder = joblib.load(le_path)
            
            # Load metadata
            metadata_path = self.models_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                    self.feature_names = self.metadata.get('feature_names', [])
            
            # Load available models
            model_files = {
                'random_forest': 'model_random_forest.pkl',
                'extra_trees': 'model_extra_trees.pkl',
                'ensemble': 'model_ensemble.pkl'
            }
            
            for model_name, filename in model_files.items():
                model_path = self.models_dir / filename
                if model_path.exists():
                    self.models[model_name] = joblib.load(model_path)
            
            if self.models and self.label_encoder:
                self.is_loaded = True
                return True
            
            return False
            
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False
    
    def predict(self, input_data):
        """Make deterministic prediction with input validation"""
        if not self.is_loaded:
            st.error("Models not loaded!")
            return None
        
        # Ensure deterministic behavior for each prediction
        set_deterministic_behavior()
        
        try:
            # Convert to DataFrame with consistent ordering
            if isinstance(input_data, dict):
                # Sort keys for consistent ordering
                sorted_data = {k: input_data[k] for k in sorted(input_data.keys())}
                df = pd.DataFrame([sorted_data])
            else:
                df = pd.DataFrame(input_data)
                # Sort columns for consistency
                df = df.reindex(sorted(df.columns), axis=1)
            
            # Use deterministic model selection (always same order)
            model_priority = ['ensemble', 'extra_trees', 'random_forest']
            best_model_name = None
            
            for model_name in model_priority:
                if model_name in self.models:
                    best_model_name = model_name
                    break
            
            if not best_model_name:
                st.error("No trained models available!")
                return None
            
            model = self.models[best_model_name]
            
            # Prepare features (use available columns)
            available_features = [col for col in self.feature_names if col in df.columns]
            if not available_features:
                # Use all numeric columns if feature names not available
                available_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not available_features:
                st.error("No suitable features found for prediction")
                return None
            
            X = df[available_features].fillna(0)  # Simple imputation
            
            # Scale features if scaler available
            if self.scaler:
                # Ensure we have the right number of features
                if X.shape[1] != len(self.feature_names):
                    # Pad or truncate as needed
                    if X.shape[1] < len(self.feature_names):
                        # Pad with zeros
                        missing_cols = len(self.feature_names) - X.shape[1]
                        X = np.column_stack([X, np.zeros((X.shape[0], missing_cols))])
                    else:
                        # Truncate
                        X = X.iloc[:, :len(self.feature_names)]
                
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
        <p style="color: #b0b0b0; margin: 0;">AI-Powered Exoplanet Classification with Robust CSV Processing</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load predictor
    predictor = load_predictor()
    
    if not predictor or not predictor.is_loaded:
        st.error("❌ Models not loaded! Please ensure model files are available in the 'models' directory.")
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["🔭 Single Prediction", "📊 Batch Processing & EDA", "📈 About Model"])
    
    with tab1:
        st.markdown("### 🔭 Single Exoplanet Classification")
        st.markdown("Enter the parameters of your astronomical object below:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🪐 Planetary Parameters")
            koi_period = st.number_input("Orbital Period (days)", value=365.25, min_value=0.1)
            koi_prad = st.number_input("Planet Radius (Earth radii)", value=1.0, min_value=0.1)
            koi_teq = st.number_input("Equilibrium Temperature (K)", value=288.0, min_value=0.1)
            koi_insol = st.number_input("Insolation Flux (Earth flux)", value=1.0, min_value=0.1)
            
        with col2:
            st.markdown("#### ⭐ Stellar Parameters")
            koi_srad = st.number_input("Stellar Radius (Solar radii)", value=1.0, min_value=0.1)
            koi_dor = st.number_input("Distance/Stellar Radius Ratio", value=215.0, min_value=1.0)
            ra = st.number_input("Right Ascension (deg)", value=290.0, min_value=0.0, max_value=360.0)
            dec = st.number_input("Declination (deg)", value=42.0, min_value=-90.0, max_value=90.0)
        
        # Additional parameters
        with st.expander("🔬 Advanced Parameters"):
            col3, col4 = st.columns(2)
            with col3:
                koi_score = st.slider("KOI Score", 0.0, 1.0, 0.5)
                koi_smass = st.number_input("Stellar Mass (Solar masses)", value=1.0, min_value=0.1)
            with col4:
                koi_sage = st.number_input("Stellar Age (Gyr)", value=4.5, min_value=0.1)
                koi_steff = st.number_input("Stellar Temperature (K)", value=5778.0, min_value=1000.0)
        
        # Predict button
        if st.button("🚀 Classify Exoplanet", type="primary", use_container_width=True):
            input_data = {
                'koi_period': koi_period, 'koi_prad': koi_prad, 'koi_teq': koi_teq, 'koi_insol': koi_insol,
                'koi_srad': koi_srad, 'koi_dor': koi_dor, 'ra': ra, 'dec': dec,
                'koi_score': koi_score, 'koi_smass': koi_smass, 'koi_sage': koi_sage, 'koi_steff': koi_steff
            }
            
            with st.spinner("🤖 Analyzing astronomical data..."):
                result = predictor.predict(input_data)
            
            if result:
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
                    st.metric("Confidence", f"{result['confidence']:.1%}")
                
                with col3:
                    st.info(f"🤖 Model: {result['model_used'].title()}")
                
                # Probability visualization
                prob_df = pd.DataFrame([
                    {"Class": k, "Probability": v}
                    for k, v in result['probabilities'].items()
                ])
                
                fig = px.bar(prob_df, x='Class', y='Probability', 
                           title="Classification Confidence by Class",
                           color='Probability', color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 📊 Batch Classification & Data Analysis")
        st.markdown("Upload a CSV file for analysis and bulk classification:")
        
        uploaded_file = st.file_uploader(
            "📁 Choose CSV file", 
            type=['csv'],
            help="Upload a CSV file with exoplanet parameters"
        )
        
        if uploaded_file:
            with st.spinner("🔍 Processing CSV file..."):
                df, load_message = RobustCSVLoader.load_csv_safely(uploaded_file)
            
            if df is not None:
                st.success(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
                if load_message:
                    st.info(f"ℹ️ {load_message}")
                
                # Create EDA object
                eda = SimpleEDA(df)
                
                # EDA Options
                eda_options = st.multiselect(
                    "Select analysis components:",
                    ["📋 Basic Info", "🔍 Data Quality", "📈 Statistics"],
                    default=["📋 Basic Info"]
                )
                
                if "📋 Basic Info" in eda_options:
                    eda.basic_info()
                
                if "🔍 Data Quality" in eda_options:
                    eda.data_quality()
                
                if "📈 Statistics" in eda_options:
                    eda.simple_stats()
                
                # Sample view
                st.markdown("### 👁️ Data Preview")
                view_option = st.radio(
                    "Select view:",
                    ["First 10 rows", "Last 10 rows", "Random sample"]
                )
                
                if view_option == "First 10 rows":
                    st.dataframe(df.head(10), use_container_width=True)
                elif view_option == "Last 10 rows":
                    st.dataframe(df.tail(10), use_container_width=True)
                else:
                    sample_size = min(10, len(df))
                    st.dataframe(df.sample(sample_size), use_container_width=True)
                
                # Classification
                st.markdown("---")
                st.markdown("## 🚀 Batch Classification")
                
                if predictor and predictor.is_loaded:
                    if st.button("🚀 Classify All Objects", type="primary"):
                        with st.spinner("🤖 Processing batch predictions..."):
                            results = []
                            progress_bar = st.progress(0)
                            
                            for i, (_, row) in enumerate(df.iterrows()):
                                result = predictor.predict(row.to_dict())
                                if result:
                                    results.append({
                                        'index': i,
                                        'prediction': result['prediction'],
                                        'confidence': result['confidence']
                                    })
                                else:
                                    results.append({
                                        'index': i,
                                        'prediction': 'ERROR',
                                        'confidence': 0.0
                                    })
                                progress_bar.progress((i + 1) / len(df))
                            
                            if results:
                                results_df = pd.DataFrame(results)
                                df_with_results = df.copy()
                                df_with_results['prediction'] = [r['prediction'] for r in results]
                                df_with_results['confidence'] = [r['confidence'] for r in results]
                                
                                st.markdown("### 🎯 Classification Results")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    confirmed = len(results_df[results_df['prediction'] == 'CONFIRMED'])
                                    st.metric("🪐 Confirmed", confirmed)
                                with col2:
                                    candidates = len(results_df[results_df['prediction'] == 'CANDIDATE'])
                                    st.metric("🔍 Candidates", candidates)
                                with col3:
                                    false_pos = len(results_df[results_df['prediction'] == 'FALSE_POSITIVE'])
                                    st.metric("❌ False Positives", false_pos)
                                with col4:
                                    avg_confidence = results_df['confidence'].mean()
                                    st.metric("📊 Avg Confidence", f"{avg_confidence:.1%}")
                                
                                # Results visualization
                                pred_counts = results_df['prediction'].value_counts()
                                fig = px.pie(values=pred_counts.values, names=pred_counts.index,
                                           title="Distribution of Classifications")
                                st.plotly_chart(fig, use_container_width=True)
                                
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
                else:
                    st.error("⚠️ Predictor not loaded. Cannot perform classification.")
            else:
                st.error("❌ Failed to load CSV file. Please check the file format and try again.")
        else:
            st.info("👆 Upload a CSV file to begin analysis")
    
    with tab3:
        st.markdown("### 📈 About the Model")
        
        if predictor and predictor.metadata:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Model Details")
                st.info(f"**Training Date**: {predictor.metadata['training_date'][:10]}")
                st.info(f"**Dataset Size**: {predictor.metadata['dataset_size']:,} objects")
                st.info(f"**Features**: {predictor.metadata['n_features']}")
                st.info(f"**Random Seed**: {predictor.metadata.get('random_seed', 'Not specified')}")
                
                st.markdown("#### 🏆 Performance")
                for model, score in predictor.metadata['all_scores'].items():
                    st.metric(f"{model.title()}", f"{score:.1%}")
            
            with col2:
                st.markdown("#### 🔬 Features Used")
                if predictor.feature_names:
                    for feature in predictor.feature_names[:10]:
                        st.text(f"• {feature}")
                    if len(predictor.feature_names) > 10:
                        st.text(f"... and {len(predictor.feature_names) - 10} more")
        
        st.markdown("---")
        st.markdown("""
        #### 🌌 NASA Space Apps Challenge 2025
        
        **Challenge**: "A World Away: Hunting for Exoplanets with AI"
        
        **Features**:
        - 🔒 **Deterministic Predictions**: Same input always produces same output
        - 📊 **Robust CSV Processing**: Handles various file formats and encoding issues
        - 🧠 **Intelligent EDA**: Automated data analysis and quality checks
        - 🚀 **Production Ready**: Pre-trained models for instant classification
        
        **Classifications**:
        - **🪐 CONFIRMED**: Verified exoplanets
        - **🔍 CANDIDATE**: Potential exoplanets requiring further study  
        - **❌ FALSE_POSITIVE**: Objects that mimic planetary signals
        """)

if __name__ == "__main__":
    main()