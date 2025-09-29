#!/usr/bin/env python3
"""
🌐 NASA Exoplanet Hunter - Improved Version
NASA Space Apps Challenge 2025 - Fixed EDA and Proper Deterministic Behavior

Fixes:
1. Proper file-based EDA functionality
2. Deterministic only for SAME inputs, different for DIFFERENT inputs
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import hashlib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="NASA Exoplanet Hunter",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SmartDeterministicBehavior:
    """🔒 Smart deterministic behavior that responds to input changes"""
    
    @staticmethod
    def get_input_hash(input_data):
        """Create hash of input data for deterministic seeding"""
        if isinstance(input_data, dict):
            # Sort keys and create string representation
            sorted_items = sorted(input_data.items())
            input_str = str(sorted_items)
        else:
            input_str = str(input_data)
        
        # Create hash from input
        hash_obj = hashlib.md5(input_str.encode())
        return int(hash_obj.hexdigest()[:8], 16)  # Use first 8 chars as integer
    
    @staticmethod
    def set_deterministic_seed(input_data, base_seed=42):
        """Set seed based on input data for consistent results with same input"""
        input_hash = SmartDeterministicBehavior.get_input_hash(input_data)
        final_seed = base_seed + input_hash
        np.random.seed(final_seed % (2**32))  # Ensure seed is within valid range
        
        import random
        random.seed(final_seed % (2**32))
        
        return final_seed % (2**32)

class RobustCSVLoader:
    """🔧 Enhanced CSV loader with comprehensive error handling"""
    
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
                            st.success(f"✅ CSV loaded using: {strategy_name} with {encoding} encoding")
                            return df, None
                    except Exception as strategy_error:
                        continue
                        
            except Exception as encoding_error:
                continue
        
        return None, "Unable to parse CSV file. Please check file format and encoding."

class ComprehensiveEDA:
    """📊 Comprehensive EDA with robust error handling"""
    
    def __init__(self, df):
        self.df = df.copy()  # Make a copy to avoid modifying original
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
    def basic_info(self):
        """Display comprehensive dataset information"""
        st.markdown("### 📋 Dataset Overview")
        
        # Memory usage
        memory_usage = self.df.memory_usage(deep=True).sum() / 1024**2
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🔢 Total Rows", f"{len(self.df):,}")
        with col2:
            st.metric("📊 Total Columns", len(self.df.columns))
        with col3:
            st.metric("🔢 Numeric Columns", len(self.numeric_cols))
        with col4:
            st.metric("💾 Memory Usage", f"{memory_usage:.2f} MB")
        
        # Column information
        st.markdown("#### 🏷️ Column Information")
        col_info = []
        for col in self.df.columns:
            col_info.append({
                'Column': col,
                'Type': str(self.df[col].dtype),
                'Non-Null': self.df[col].count(),
                'Null': self.df[col].isnull().sum(),
                'Null %': round((self.df[col].isnull().sum() / len(self.df)) * 100, 2),
                'Unique Values': self.df[col].nunique(),
                'Sample': str(self.df[col].iloc[0]) if len(self.df) > 0 else 'N/A'
            })
        
        col_df = pd.DataFrame(col_info)
        st.dataframe(col_df, use_container_width=True)
        
    def data_quality_analysis(self):
        """Comprehensive data quality analysis"""
        st.markdown("### 🔍 Data Quality Analysis")
        
        try:
            # Missing values analysis
            missing_count = self.df.isnull().sum().sum()
            total_cells = len(self.df) * len(self.df.columns)
            missing_percentage = (missing_count / total_cells) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🕳️ Missing Values", missing_count)
            with col2:
                st.metric("📊 Missing Percentage", f"{missing_percentage:.2f}%")
            with col3:
                duplicates = self.df.duplicated().sum()
                st.metric("🔄 Duplicate Rows", duplicates)
            
            # Missing values by column
            if missing_count > 0:
                st.markdown("#### Missing Values by Column")
                missing_data = self.df.isnull().sum().sort_values(ascending=False)
                missing_data = missing_data[missing_data > 0]
                
                if not missing_data.empty:
                    # Convert to safe format for plotting
                    missing_df = pd.DataFrame({
                        'Column': [str(col) for col in missing_data.index],
                        'Missing_Count': missing_data.values.tolist()
                    })
                    
                    fig = px.bar(
                        missing_df,
                        x='Missing_Count',
                        y='Column',
                        orientation='h',
                        title="Missing Values Count by Column",
                        labels={'Missing_Count': 'Missing Count', 'Column': 'Columns'}
                    )
                    fig.update_layout(height=max(400, len(missing_data) * 30))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing values found!")
            
            # Data types distribution (simplified to avoid serialization issues)
            st.markdown("#### Data Types Distribution")
            try:
                # Create a simple summary table instead of a pie chart for data types
                dtype_info = []
                for dtype in self.df.dtypes.unique():
                    dtype_str = str(dtype)
                    count = (self.df.dtypes == dtype).sum()
                    dtype_info.append({'Data Type': dtype_str, 'Column Count': count})
                
                dtype_df = pd.DataFrame(dtype_info)
                st.dataframe(dtype_df, use_container_width=True)
                
                # Only create pie chart if we have safe data
                if len(dtype_info) > 0:
                    safe_names = [info['Data Type'] for info in dtype_info]
                    safe_values = [info['Column Count'] for info in dtype_info]
                    
                    # Create DataFrame for plotly to avoid serialization issues
                    chart_df = pd.DataFrame({
                        'Data_Type': safe_names,
                        'Count': safe_values
                    })
                    
                    fig = px.pie(chart_df, values='Count', names='Data_Type', 
                               title="Distribution of Data Types")
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as dtype_error:
                st.warning(f"Could not create data type visualization: {dtype_error}")
                # Show simple text summary instead
                dtype_summary = self.df.dtypes.value_counts()
                st.text("Data Types Summary:")
                for dtype, count in dtype_summary.items():
                    st.text(f"  {str(dtype)}: {count} columns")
            
        except Exception as e:
            st.error(f"Error in data quality analysis: {e}")
            st.info("Showing basic information instead:")
            
            # Fallback simple analysis
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(self.df))
            with col2:
                st.metric("Columns", len(self.df.columns))
            with col3:
                st.metric("Memory (MB)", f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f}")
        
    def numeric_analysis(self):
        """Comprehensive numeric data analysis"""
        if not self.numeric_cols:
            st.info("📊 No numeric columns found for analysis")
            return
            
        st.markdown("### 🔢 Numeric Data Analysis")
        
        # Statistical summary
        st.markdown("#### 📈 Statistical Summary")
        try:
            desc_stats = self.df[self.numeric_cols].describe()
            st.dataframe(desc_stats, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating statistics: {e}")
            return
        
        # Distribution analysis
        st.markdown("#### 📊 Distribution Analysis")
        
        # Column selector
        selected_cols = st.multiselect(
            "Select columns for distribution analysis:",
            self.numeric_cols,
            default=self.numeric_cols[:3] if len(self.numeric_cols) >= 3 else self.numeric_cols
        )
        
        if selected_cols:
            # Create individual histograms
            for col in selected_cols:
                try:
                    if self.df[col].notna().sum() > 0:  # Check if column has non-null values
                        fig = px.histogram(
                            self.df, 
                            x=col, 
                            title=f'Distribution of {col}',
                            marginal="box",
                            nbins=30
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Basic stats for this column
                        col_data = self.df[col].dropna()
                        if len(col_data) > 0:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric(f"{col} Mean", f"{col_data.mean():.3f}")
                            with col2:
                                st.metric(f"{col} Median", f"{col_data.median():.3f}")
                            with col3:
                                st.metric(f"{col} Std Dev", f"{col_data.std():.3f}")
                            with col4:
                                st.metric(f"{col} Range", f"{col_data.min():.3f} - {col_data.max():.3f}")
                    else:
                        st.warning(f"⚠️ Column '{col}' has no valid data for visualization")
                except Exception as e:
                    st.error(f"Error analyzing column '{col}': {e}")
        
        # Correlation analysis
        if len(selected_cols) > 1:
            st.markdown("#### 🔗 Correlation Analysis")
            try:
                corr_data = self.df[selected_cols].corr()
                
                # Create DataFrame with safe column names for plotly
                corr_df = pd.DataFrame(
                    corr_data.values,
                    index=[str(col) for col in corr_data.index],
                    columns=[str(col) for col in corr_data.columns]
                )
                
                # Correlation heatmap
                fig = px.imshow(
                    corr_df.values,
                    x=corr_df.columns.tolist(),
                    y=corr_df.index.tolist(),
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Matrix",
                    color_continuous_scale='RdBu_r'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Strong correlations
                strong_correlations = []
                for i in range(len(corr_data.columns)):
                    for j in range(i+1, len(corr_data.columns)):
                        corr_val = corr_data.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            strong_correlations.append({
                                'Variable 1': corr_data.columns[i],
                                'Variable 2': corr_data.columns[j],
                                'Correlation': round(corr_val, 3),
                                'Strength': 'Strong Positive' if corr_val > 0.7 else 'Strong Negative'
                            })
                
                if strong_correlations:
                    st.markdown("##### ⚡ Strong Correlations (|r| > 0.7)")
                    strong_df = pd.DataFrame(strong_correlations)
                    st.dataframe(strong_df, use_container_width=True)
                else:
                    st.info("No strong correlations found (|r| > 0.7)")
                    
            except Exception as e:
                st.error(f"Error in correlation analysis: {e}")
    
    def outlier_detection(self):
        """Outlier detection and visualization"""
        if not self.numeric_cols:
            st.info("No numeric columns for outlier detection")
            return
            
        st.markdown("### 🎯 Outlier Detection")
        
        # Column selector
        outlier_col = st.selectbox(
            "Select column for outlier analysis:",
            self.numeric_cols,
            key="outlier_column_selector"
        )
        
        if outlier_col:
            try:
                data = self.df[outlier_col].dropna()
                
                if len(data) == 0:
                    st.warning(f"No valid data in column '{outlier_col}'")
                    return
                
                # IQR method
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_iqr = data[(data < lower_bound) | (data > upper_bound)]
                
                # Z-score method
                mean_val = data.mean()
                std_val = data.std()
                if std_val > 0:
                    z_scores = np.abs((data - mean_val) / std_val)
                    outliers_zscore = data[z_scores > 3]
                else:
                    outliers_zscore = pd.Series([], dtype=float)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🔍 IQR Outliers", len(outliers_iqr))
                    st.metric("📊 Z-Score Outliers", len(outliers_zscore))
                
                with col2:
                    st.metric("📈 IQR Range", f"{lower_bound:.3f} - {upper_bound:.3f}")
                    st.metric("📊 Z-Score Threshold", "±3σ")
                
                with col3:
                    outlier_percentage = (len(outliers_iqr) / len(data)) * 100
                    st.metric("📊 Outlier %", f"{outlier_percentage:.2f}%")
                    st.metric("📈 Data Points", len(data))
                
                # Box plot visualization
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=data,
                    name=outlier_col,
                    boxpoints='outliers',
                    jitter=0.3,
                    pointpos=-1.8
                ))
                fig.update_layout(
                    title=f'Box Plot: {outlier_col}',
                    yaxis_title=outlier_col,
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show outlier values if any
                if len(outliers_iqr) > 0:
                    with st.expander("🔍 View Outlier Values"):
                        outlier_df = pd.DataFrame({
                            'Index': outliers_iqr.index,
                            'Value': outliers_iqr.values,
                            'Method': 'IQR'
                        })
                        st.dataframe(outlier_df, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error in outlier detection: {e}")

class ProductionPredictor:
    """🚀 Smart Production Predictor with proper deterministic behavior"""
    
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
        """Make prediction with smart deterministic behavior"""
        if not self.is_loaded:
            st.error("Models not loaded!")
            return None
        
        try:
            # Set seed based on input data for consistent results with same input
            seed_used = SmartDeterministicBehavior.set_deterministic_seed(input_data)
            
            # Convert to DataFrame with consistent ordering
            if isinstance(input_data, dict):
                # Sort keys for consistent ordering
                sorted_data = {k: input_data[k] for k in sorted(input_data.keys())}
                df = pd.DataFrame([sorted_data])
            else:
                df = pd.DataFrame(input_data)
                # Sort columns for consistency
                df = df.reindex(sorted(df.columns), axis=1)
            
            # Use deterministic model selection
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
            
            # Prepare features
            available_features = [col for col in self.feature_names if col in df.columns]
            if not available_features:
                available_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not available_features:
                st.error("No suitable features found for prediction")
                return None
            
            X = df[available_features].fillna(0)
            
            # Scale features if scaler available
            if self.scaler:
                # Ensure correct number of features
                if X.shape[1] != len(self.feature_names):
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
                'model_used': best_model_name,
                'seed_used': seed_used
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
        <p style="color: #b0b0b0; margin: 0;">Smart Deterministic AI with Comprehensive EDA</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load predictor
    predictor = load_predictor()
    
    if not predictor or not predictor.is_loaded:
        st.error("❌ Models not loaded! Please ensure model files are available.")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔭 Single Prediction", "📊 File Upload & EDA", "📈 About Model"])
    
    with tab1:
        st.markdown("### 🔭 Single Exoplanet Classification")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🪐 Planetary Parameters")
            koi_period = st.number_input("Orbital Period (days)", value=365.25, min_value=0.1, key="period")
            koi_prad = st.number_input("Planet Radius (Earth radii)", value=1.0, min_value=0.1, key="radius")
            koi_teq = st.number_input("Equilibrium Temperature (K)", value=288.0, min_value=0.1, key="temp")
            koi_insol = st.number_input("Insolation Flux", value=1.0, min_value=0.1, key="flux")
            
        with col2:
            st.markdown("#### ⭐ Stellar Parameters")
            koi_srad = st.number_input("Stellar Radius (Solar radii)", value=1.0, min_value=0.1, key="srad")
            koi_dor = st.number_input("Distance/Stellar Radius Ratio", value=215.0, min_value=1.0, key="dor")
            ra = st.number_input("Right Ascension (deg)", value=290.0, min_value=0.0, max_value=360.0, key="ra")
            dec = st.number_input("Declination (deg)", value=42.0, min_value=-90.0, max_value=90.0, key="dec")
        
        # Advanced parameters
        with st.expander("🔬 Advanced Parameters"):
            col3, col4 = st.columns(2)
            with col3:
                koi_score = st.slider("KOI Score", 0.0, 1.0, 0.5, key="score")
                koi_smass = st.number_input("Stellar Mass", value=1.0, min_value=0.1, key="smass")
            with col4:
                koi_sage = st.number_input("Stellar Age (Gyr)", value=4.5, min_value=0.1, key="sage")
                koi_steff = st.number_input("Stellar Temperature (K)", value=5778.0, min_value=1000.0, key="steff")
        
        # Predict button
        if st.button("🚀 Classify Exoplanet", type="primary"):
            input_data = {
                'koi_period': koi_period, 'koi_prad': koi_prad, 'koi_teq': koi_teq, 'koi_insol': koi_insol,
                'koi_srad': koi_srad, 'koi_dor': koi_dor, 'ra': ra, 'dec': dec,
                'koi_score': koi_score, 'koi_smass': koi_smass, 'koi_sage': koi_sage, 'koi_steff': koi_steff
            }
            
            with st.spinner("🤖 Analyzing..."):
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
                
                # Show deterministic seed
                with st.expander("🔒 Deterministic Info"):
                    st.info(f"Seed used: {result.get('seed_used', 'N/A')}")
                    st.info("This ensures same inputs always give same results!")
                
                # Probability chart
                prob_df = pd.DataFrame([
                    {"Class": k, "Probability": v}
                    for k, v in result['probabilities'].items()
                ])
                
                fig = px.bar(prob_df, x='Class', y='Probability', 
                           title="Classification Confidence by Class",
                           color='Probability', color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 📊 File Upload & Comprehensive EDA")
        
        uploaded_file = st.file_uploader(
            "📁 Upload CSV File",
            type=['csv'],
            help="Upload a CSV file for comprehensive analysis"
        )
        
        if uploaded_file:
            with st.spinner("🔍 Loading and analyzing file..."):
                df, error_msg = RobustCSVLoader.load_csv_safely(uploaded_file)
            
            if df is not None:
                st.success(f"✅ File loaded: {len(df)} rows × {len(df.columns)} columns")
                
                # Create EDA instance
                eda = ComprehensiveEDA(df)
                
                # EDA Navigation
                st.markdown("### 🔬 Analysis Options")
                
                analysis_type = st.selectbox(
                    "Choose analysis type:",
                    ["📋 Dataset Overview", "🔍 Data Quality", "🔢 Numeric Analysis", "🎯 Outlier Detection", "🚀 Batch Classification"]
                )
                
                if analysis_type == "📋 Dataset Overview":
                    eda.basic_info()
                    
                elif analysis_type == "🔍 Data Quality":
                    eda.data_quality_analysis()
                    
                elif analysis_type == "🔢 Numeric Analysis":
                    eda.numeric_analysis()
                    
                elif analysis_type == "🎯 Outlier Detection":
                    eda.outlier_detection()
                    
                elif analysis_type == "🚀 Batch Classification":
                    st.markdown("### 🚀 Batch Classification")
                    
                    if predictor and predictor.is_loaded:
                        st.info(f"Ready to classify {len(df)} objects")
                        
                        if st.button("🚀 Start Batch Classification", type="primary"):
                            with st.spinner("Processing predictions..."):
                                results = []
                                progress_bar = st.progress(0)
                                
                                for i, (_, row) in enumerate(df.iterrows()):
                                    try:
                                        result = predictor.predict(row.to_dict())
                                        if result:
                                            results.append({
                                                'Row': i,
                                                'Prediction': result['prediction'],
                                                'Confidence': result['confidence']
                                            })
                                        else:
                                            results.append({
                                                'Row': i,
                                                'Prediction': 'ERROR',
                                                'Confidence': 0.0
                                            })
                                    except Exception as e:
                                        results.append({
                                            'Row': i,
                                            'Prediction': 'ERROR',
                                            'Confidence': 0.0
                                        })
                                    
                                    progress_bar.progress((i + 1) / len(df))
                                
                                # Display results
                                if results:
                                    results_df = pd.DataFrame(results)
                                    
                                    # Add results to original dataframe
                                    df_with_results = df.copy()
                                    df_with_results['AI_Prediction'] = results_df['Prediction']
                                    df_with_results['AI_Confidence'] = results_df['Confidence']
                                    
                                    st.success("✅ Classification complete!")
                                    
                                    # Summary statistics
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        confirmed = len(results_df[results_df['Prediction'] == 'CONFIRMED'])
                                        st.metric("🪐 Confirmed", confirmed)
                                    with col2:
                                        candidates = len(results_df[results_df['Prediction'] == 'CANDIDATE'])
                                        st.metric("🔍 Candidates", candidates)
                                    with col3:
                                        false_pos = len(results_df[results_df['Prediction'] == 'FALSE_POSITIVE'])
                                        st.metric("❌ False Positives", false_pos)
                                    with col4:
                                        valid_results = results_df[results_df['Prediction'] != 'ERROR']
                                        if len(valid_results) > 0:
                                            avg_conf = valid_results['Confidence'].mean()
                                            st.metric("📊 Avg Confidence", f"{avg_conf:.1%}")
                                        else:
                                            st.metric("📊 Avg Confidence", "N/A")
                                    
                                    # Results distribution
                                    pred_counts = results_df['Prediction'].value_counts()
                                    fig = px.pie(values=pred_counts.values, names=pred_counts.index,
                                               title="Classification Results Distribution")
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Show results table
                                    st.dataframe(df_with_results, use_container_width=True)
                                    
                                    # Download option
                                    csv_buffer = io.StringIO()
                                    df_with_results.to_csv(csv_buffer, index=False)
                                    st.download_button(
                                        "💾 Download Results",
                                        data=csv_buffer.getvalue(),
                                        file_name=f"classified_exoplanets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
                    else:
                        st.error("❌ Classifier not available")
                
                # Always show data preview
                st.markdown("---")
                st.markdown("### 👁️ Data Preview")
                preview_rows = st.slider("Number of rows to preview:", 5, 50, 10)
                st.dataframe(df.head(preview_rows), use_container_width=True)
                
            else:
                st.error(f"❌ {error_msg}")
        else:
            st.info("👆 Upload a CSV file to begin analysis")
    
    with tab3:
        st.markdown("### 📈 Model Information")
        
        if predictor and predictor.metadata:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Model Details")
                st.info(f"**Training Date**: {predictor.metadata['training_date'][:10]}")
                st.info(f"**Dataset Size**: {predictor.metadata['dataset_size']:,}")
                st.info(f"**Features**: {predictor.metadata['n_features']}")
                st.info(f"**Deterministic**: ✅ Smart seeding based on input")
                
            with col2:
                st.markdown("#### 🏆 Performance")
                for model, score in predictor.metadata['all_scores'].items():
                    st.metric(f"{model.title()}", f"{score:.1%}")
        
        st.markdown("---")
        st.markdown("""
        #### 🌌 NASA Space Apps Challenge 2025
        
        **Enhanced Features:**
        - 🔒 **Smart Deterministic**: Same inputs → same outputs, different inputs → different outputs
        - 📊 **Comprehensive EDA**: Full data analysis with interactive visualizations
        - 🛠️ **Robust CSV Handling**: Handles various formats and encoding issues
        - 🚀 **Batch Processing**: Classify thousands of objects efficiently
        """)

if __name__ == "__main__":
    main()