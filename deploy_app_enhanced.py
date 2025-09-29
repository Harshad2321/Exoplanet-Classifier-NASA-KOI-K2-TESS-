#!/usr/bin/env python3
"""
🌐 NASA Exoplanet Hunter - Enhanced Deployment Interface with EDA
NASA Space Apps Challenge 2025 - Production Deployment with Data Analysis

This enhanced deployment interface includes:
- Robust CSV parsing with error handling
- Comprehensive Exploratory Data Analysis (EDA)
- Interactive visualizations
- Data quality checks
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
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    sns = None

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None
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
    """
    🔧 Robust CSV loader with error handling and data cleaning
    """
    
    @staticmethod
    def load_csv_safely(file, encoding_attempts=['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']):
        """
        Load CSV with multiple encoding attempts and error handling
        """
        for encoding in encoding_attempts:
            try:
                # Reset file position
                file.seek(0)
                
                # Try different parsing strategies
                strategies = [
                    # Standard parsing
                    lambda: pd.read_csv(file, encoding=encoding),
                    
                    # Skip bad lines with warning
                    lambda: pd.read_csv(file, encoding=encoding, on_bad_lines='skip'),
                    
                    # Use python engine with automatic separator detection
                    lambda: pd.read_csv(file, encoding=encoding, sep=None, engine='python', on_bad_lines='skip'),
                    
                    # Skip problematic rows (common error lines)
                    lambda: pd.read_csv(file, encoding=encoding, skiprows=lambda x: x in [298, 299, 300]),
                    
                    # Read with quotes handling and error recovery
                    lambda: pd.read_csv(file, encoding=encoding, quotechar='"', skipinitialspace=True, on_bad_lines='skip'),
                    
                    # Try with different delimiters
                    lambda: pd.read_csv(file, encoding=encoding, delimiter=';', on_bad_lines='skip'),
                    lambda: pd.read_csv(file, encoding=encoding, delimiter='\t', on_bad_lines='skip'),
                    
                    # Relaxed parsing with warnings
                    lambda: pd.read_csv(file, encoding=encoding, on_bad_lines='warn', engine='python'),
                    
                    # Last resort: try to read with minimal constraints
                    lambda: pd.read_csv(file, encoding=encoding, header=0, on_bad_lines='skip', engine='python', sep=None, skipinitialspace=True)
                ]
                
                for strategy in strategies:
                    try:
                        file.seek(0)
                        df = strategy()
                        if not df.empty:
                            st.success(f"✅ Successfully loaded CSV with encoding: {encoding}")
                            return df, None
                    except Exception as strategy_error:
                        continue
                        
            except Exception as encoding_error:
                continue
        
        # If all strategies fail, try to read line by line
        try:
            file.seek(0)
            lines = []
            for i, line in enumerate(file):
                try:
                    lines.append(line.decode('utf-8').strip())
                except:
                    st.warning(f"Skipping problematic line {i+1}")
                    continue
            
            # Create DataFrame from clean lines
            if lines:
                from io import StringIO
                clean_data = StringIO('\n'.join(lines))
                df = pd.read_csv(clean_data)
                st.warning(f"⚠️ Loaded CSV with line-by-line parsing (some lines may have been skipped)")
                return df, "Line-by-line parsing used"
                
        except Exception as final_error:
            return None, f"Failed to load CSV: {final_error}"
        
        return None, "Unable to parse CSV file with any method"

class ExoplanetEDA:
    """
    📊 Comprehensive Exploratory Data Analysis for Exoplanet Data
    """
    
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
        
        # Memory usage
        memory_usage = self.df.memory_usage(deep=True).sum() / 1024**2
        st.info(f"💾 Memory Usage: {memory_usage:.2f} MB")
        
        # Data types
        st.markdown("#### 🏷️ Column Data Types")
        dtype_df = pd.DataFrame({
            'Column': self.df.columns,
            'Data Type': self.df.dtypes.astype(str),
            'Non-Null Count': self.df.count(),
            'Null Count': self.df.isnull().sum(),
            'Null %': (self.df.isnull().sum() / len(self.df) * 100).round(2)
        })
        st.dataframe(dtype_df, use_container_width=True)
        
    def data_quality(self):
        """Analyze data quality issues"""
        st.markdown("### 🔍 Data Quality Analysis")
        
        # Missing values heatmap
        if self.df.isnull().sum().sum() > 0:
                st.markdown("#### 🕳️ Missing Values Pattern")
                
                missing_matrix = self.df.isnull()
                if MATPLOTLIB_AVAILABLE and len(self.df.columns) <= 20:  # Only show heatmap for manageable number of columns
                    try:
                        fig, ax = plt.subplots(figsize=(12, 8))
                        sns.heatmap(missing_matrix, cbar=True, ax=ax, cmap='viridis')
                        plt.title('Missing Values Heatmap')
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.warning(f"Could not generate heatmap: {e}")
                        st.info("Showing missing values summary instead.")
                else:
                    st.info("Showing missing values summary (heatmap not available).")
                
                # Missing values summary
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
        
        # Duplicate rows
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            st.warning(f"⚠️ Found {duplicates} duplicate rows ({duplicates/len(self.df)*100:.2f}%)")
        else:
            st.success("✅ No duplicate rows found!")
        
    def numeric_analysis(self):
        """Analyze numeric columns"""
        if not self.numeric_cols:
            st.info("No numeric columns found for analysis")
            return
            
        st.markdown("### 🔢 Numeric Data Analysis")
        
        # Statistical summary
        st.markdown("#### 📈 Statistical Summary")
        desc_stats = self.df[self.numeric_cols].describe()
        st.dataframe(desc_stats, use_container_width=True)
        
        # Distribution plots
        st.markdown("#### 📊 Distribution Analysis")
        
        # Select columns for visualization
        selected_cols = st.multiselect(
            "Select columns to visualize:",
            self.numeric_cols,
            default=self.numeric_cols[:4] if len(self.numeric_cols) >= 4 else self.numeric_cols
        )
        
        if selected_cols:
            # Create distribution plots
            if MATPLOTLIB_AVAILABLE:
                try:
                    n_cols = min(2, len(selected_cols))
                    n_rows = (len(selected_cols) + n_cols - 1) // n_cols
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
                    if n_rows == 1:
                        axes = [axes] if n_cols == 1 else axes
                    else:
                        axes = axes.flatten()
                    
                    for i, col in enumerate(selected_cols):
                        if i < len(axes):
                            # Remove outliers for better visualization
                            data = self.df[col].dropna()
                            Q1 = data.quantile(0.25)
                            Q3 = data.quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
                            
                            axes[i].hist(filtered_data, bins=50, alpha=0.7, edgecolor='black')
                            axes[i].set_title(f'Distribution of {col}')
                            axes[i].set_xlabel(col)
                            axes[i].set_ylabel('Frequency')
                    
                    # Hide empty subplots
                    for i in range(len(selected_cols), len(axes)):
                        axes[i].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.warning(f"Could not generate distribution plots: {e}")
                    st.info("Using alternative visualization...")
                    # Fallback to plotly
                    for col in selected_cols[:4]:  # Limit to first 4 columns
                        fig_plotly = px.histogram(self.df, x=col, title=f'Distribution of {col}')
                        st.plotly_chart(fig_plotly, use_container_width=True)
            else:
                st.info("Matplotlib not available, using Plotly for visualization...")
                # Use plotly as fallback
                for col in selected_cols[:4]:  # Limit to first 4 columns
                    fig_plotly = px.histogram(self.df, x=col, title=f'Distribution of {col}')
                    st.plotly_chart(fig_plotly, use_container_width=True)
            
            # Correlation analysis
            if len(selected_cols) > 1:
                st.markdown("#### 🔗 Correlation Matrix")
                corr_matrix = self.df[selected_cols].corr()
                
                if MATPLOTLIB_AVAILABLE:
                    try:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                        plt.title('Correlation Matrix')
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.warning(f"Could not generate correlation heatmap: {e}")
                        # Fallback to plotly
                        fig_plotly = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                                             title="Correlation Matrix", color_continuous_scale='RdBu')
                        st.plotly_chart(fig_plotly, use_container_width=True)
                else:
                    # Use plotly as fallback
                    fig_plotly = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                                         title="Correlation Matrix", color_continuous_scale='RdBu')
                    st.plotly_chart(fig_plotly, use_container_width=True)
                
                # Strong correlations
                strong_corrs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            strong_corrs.append({
                                'Column 1': corr_matrix.columns[i],
                                'Column 2': corr_matrix.columns[j],
                                'Correlation': corr_val
                            })
                
                if strong_corrs:
                    st.markdown("#### ⚡ Strong Correlations (|r| > 0.7)")
                    strong_corr_df = pd.DataFrame(strong_corrs)
                    st.dataframe(strong_corr_df, use_container_width=True)
        
    def outlier_analysis(self):
        """Detect and visualize outliers"""
        if not self.numeric_cols:
            return
            
        st.markdown("### 🎯 Outlier Detection")
        
        # Select column for outlier analysis
        outlier_col = st.selectbox("Select column for outlier analysis:", self.numeric_cols)
        
        if outlier_col:
            data = self.df[outlier_col].dropna()
            
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_iqr = data[(data < lower_bound) | (data > upper_bound)]
            
            # Z-score method (with fallback if scipy not available)
            if SCIPY_AVAILABLE:
                z_scores = np.abs(stats.zscore(data))
                outliers_zscore = data[z_scores > 3]
            else:
                # Manual z-score calculation
                mean_val = data.mean()
                std_val = data.std()
                z_scores = np.abs((data - mean_val) / std_val)
                outliers_zscore = data[z_scores > 3]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("🔍 IQR Outliers", len(outliers_iqr))
                st.metric("📊 Z-Score Outliers", len(outliers_zscore))
            
            with col2:
                st.metric("📈 Normal Range", f"{lower_bound:.2f} - {upper_bound:.2f}")
                st.metric("📊 Mean ± 3σ", f"{data.mean() - 3*data.std():.2f} - {data.mean() + 3*data.std():.2f}")
            
            # Box plot
            fig = go.Figure()
            fig.add_trace(go.Box(y=data, name=outlier_col, boxpoints='outliers'))
            fig.update_layout(title=f'Box Plot: {outlier_col}', yaxis_title=outlier_col)
            st.plotly_chart(fig, use_container_width=True)
            
    def generate_insights(self):
        """Generate automated insights"""
        st.markdown("### 🧠 Automated Insights")
        
        insights = []
        
        # Dataset size insights
        if len(self.df) > 10000:
            insights.append("📊 Large dataset detected - excellent for machine learning!")
        elif len(self.df) < 100:
            insights.append("⚠️ Small dataset - consider data augmentation techniques")
        
        # Missing data insights
        missing_pct = (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        if missing_pct > 20:
            insights.append(f"🕳️ High missing data: {missing_pct:.1f}% - consider imputation strategies")
        elif missing_pct < 5:
            insights.append(f"✅ Low missing data: {missing_pct:.1f}% - data quality is good")
        
        # Numeric columns insights
        if self.numeric_cols:
            # Check for potential target variables
            for col in self.numeric_cols:
                unique_vals = self.df[col].nunique()
                if unique_vals <= 10 and unique_vals > 1:
                    insights.append(f"🎯 '{col}' could be a categorical target (has {unique_vals} unique values)")
        
        # Display insights
        for insight in insights:
            st.info(insight)
        
        if not insights:
            st.success("🔍 Data appears clean and ready for analysis!")

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
                    st.warning(f"Feature mismatch. Expected {len(self.feature_names)}, got {X.shape[1]}")
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
        <p style="color: #b0b0b0; margin: 0;">AI-Powered Exoplanet Classification with Advanced EDA</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load predictor
    predictor = load_predictor()
    
    if not predictor or not predictor.is_loaded:
        st.error("❌ Models not loaded! Please ensure model files are available in the 'models' directory.")
        st.info("""
        Expected files:
        - models/model_random_forest.pkl
        - models/label_encoder.pkl
        - models/scaler.pkl (optional)
        - models/metadata.json (optional)
        """)
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔭 Single Prediction", "📊 Batch Processing & EDA", "📈 About Model", "🔧 Data Tools"])
    
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
        st.markdown("### 📊 Batch Classification & Exploratory Data Analysis")
        st.markdown("Upload a CSV file for comprehensive data analysis and bulk classification:")
        
        # File upload with enhanced options
        uploaded_file = st.file_uploader(
            "📁 Choose CSV file", 
            type=['csv'],
            help="Upload a CSV file with exoplanet parameters for analysis and classification"
        )
        
        if uploaded_file:
            # Load data with robust parsing
            with st.spinner("🔍 Parsing CSV file..."):
                df, load_message = RobustCSVLoader.load_csv_safely(uploaded_file)
            
            if df is not None:
                st.success(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
                if load_message:
                    st.info(f"ℹ️ {load_message}")
                
                # EDA Section
                st.markdown("---")
                st.markdown("## 🔬 Exploratory Data Analysis")
                
                # Create EDA object
                eda = ExoplanetEDA(df)
                
                # EDA Options
                eda_options = st.multiselect(
                    "Select EDA components to display:",
                    ["📋 Basic Info", "🔍 Data Quality", "🔢 Numeric Analysis", "🎯 Outlier Detection", "🧠 Insights"],
                    default=["📋 Basic Info", "🔍 Data Quality"]
                )
                
                # Display selected EDA components
                if "📋 Basic Info" in eda_options:
                    eda.basic_info()
                
                if "🔍 Data Quality" in eda_options:
                    eda.data_quality()
                
                if "🔢 Numeric Analysis" in eda_options:
                    eda.numeric_analysis()
                
                if "🎯 Outlier Detection" in eda_options:
                    eda.outlier_analysis()
                
                if "🧠 Insights" in eda_options:
                    eda.generate_insights()
                
                # Sample view
                st.markdown("---")
                st.markdown("### 👁️ Data Preview")
                
                # Show different views
                view_option = st.radio(
                    "Select view:",
                    ["First 10 rows", "Last 10 rows", "Random sample", "Summary statistics"]
                )
                
                if view_option == "First 10 rows":
                    st.dataframe(df.head(10), use_container_width=True)
                elif view_option == "Last 10 rows":
                    st.dataframe(df.tail(10), use_container_width=True)
                elif view_option == "Random sample":
                    sample_size = min(10, len(df))
                    st.dataframe(df.sample(sample_size), use_container_width=True)
                else:  # Summary statistics
                    st.dataframe(df.describe(include='all'), use_container_width=True)
                
                # Classification section
                st.markdown("---")
                st.markdown("## 🚀 Batch Classification")
                
                if predictor and predictor.is_loaded:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.info("Ready to classify your data! Click the button below to process all rows.")
                    
                    with col2:
                        classify_button = st.button("🚀 Classify All Objects", type="primary")
                    
                    if classify_button:
                        with st.spinner("🤖 Processing batch predictions..."):
                            # Process predictions
                            results = []
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, (_, row) in enumerate(df.iterrows()):
                                status_text.text(f"Processing row {i+1}/{len(df)}")
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
                            
                            status_text.text("✅ Processing complete!")
                            
                            # Create results dataframe
                            if results:
                                results_df = pd.DataFrame(results)
                                df_with_results = df.copy()
                                df_with_results['prediction'] = [r['prediction'] for r in results]
                                df_with_results['confidence'] = [r['confidence'] for r in results]
                                
                                st.markdown("### 🎯 Classification Results")
                                
                                # Summary statistics
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
                                st.markdown("#### 📈 Classification Distribution")
                                pred_counts = results_df['prediction'].value_counts()
                                
                                fig = px.pie(values=pred_counts.values, names=pred_counts.index,
                                           title="Distribution of Classifications")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Confidence distribution
                                fig2 = px.histogram(results_df, x='confidence', nbins=20,
                                                  title="Distribution of Confidence Scores")
                                st.plotly_chart(fig2, use_container_width=True)
                                
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
                else:
                    st.error("⚠️ Predictor not loaded. Cannot perform classification.")
            
            else:
                st.error("❌ Failed to load CSV file. Please check the file format and try again.")
                st.info("""
                **Troubleshooting tips:**
                - Ensure the file is a valid CSV
                - Check for special characters or encoding issues
                - Try saving your file with UTF-8 encoding
                - Remove any merged cells if coming from Excel
                """)
        
        else:
            st.info("👆 Upload a CSV file to begin analysis and classification")
            
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
        
        if predictor and predictor.metadata:
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
    
    with tab4:
        st.markdown("### 🔧 Data Processing Tools")
        
        st.markdown("#### 📊 CSV File Diagnostics")
        st.markdown("Upload a CSV file to diagnose potential parsing issues:")
        
        diagnostic_file = st.file_uploader(
            "📁 Choose CSV file for diagnostics", 
            type=['csv'],
            key="diagnostic_upload"
        )
        
        if diagnostic_file:
            # File info
            file_size = len(diagnostic_file.getvalue()) / 1024
            st.info(f"📏 File size: {file_size:.1f} KB")
            
            # Try to read first few lines
            try:
                diagnostic_file.seek(0)
                lines = []
                for i, line in enumerate(diagnostic_file):
                    if i >= 10:  # Only read first 10 lines
                        break
                    try:
                        lines.append(line.decode('utf-8').strip())
                    except:
                        lines.append(f"[Line {i+1}: Encoding issue]")
                
                st.markdown("#### 👁️ First 10 Lines Preview")
                for i, line in enumerate(lines):
                    st.text(f"{i+1:2d}: {line[:100]}{'...' if len(line) > 100 else ''}")
                
                # Field count analysis
                if lines:
                    header_fields = len(lines[0].split(','))
                    st.info(f"📊 Expected fields (from header): {header_fields}")
                    
                    field_counts = []
                    for i, line in enumerate(lines[1:], 2):
                        field_count = len(line.split(','))
                        field_counts.append(field_count)
                        if field_count != header_fields:
                            st.warning(f"⚠️ Line {i}: Expected {header_fields} fields, found {field_count}")
                    
                    if field_counts and all(fc == header_fields for fc in field_counts):
                        st.success("✅ All lines have consistent field counts")
                
            except Exception as e:
                st.error(f"Error reading file: {e}")
        
        st.markdown("---")
        st.markdown("#### 🛠️ Data Preparation Tips")
        
        tips = [
            "📝 **Encoding**: Save CSV files with UTF-8 encoding",
            "🔗 **Consistency**: Ensure all rows have the same number of columns",
            "❓ **Missing Data**: Use consistent markers (e.g., empty cells or 'NaN')",
            "📊 **Numbers**: Use dots for decimals, not commas",
            "🏷️ **Headers**: Include clear column names in the first row",
            "💾 **Size**: Keep files under 100MB for best performance"
        ]
        
        for tip in tips:
            st.info(tip)

if __name__ == "__main__":
    main()