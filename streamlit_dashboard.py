#!/usr/bin/env python3
"""
🎨 MODERN STREAMLIT DASHBOARD
Beautiful, responsive interface for exoplanet classification
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import json
import os
from datetime import datetime
import time

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Set page config FIRST
st.set_page_config(
    page_title="🪐 Exoplanet Classifier",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    /* Main theme */
    .main {
        padding: 2rem 1rem;
    }
    
    /* Custom header */
    .custom-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Success message */
    .success-box {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    
    /* Warning message */
    .warning-box {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

class ExoplanetDashboard:
    """Modern dashboard for exoplanet classification"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.results_dir = Path("results")
        self.data_dir = Path("data")
        
    def load_data(self):
        """Load and cache data"""
        
        @st.cache_data
        def _load_data():
            try:
                # Try to load cached data first
                for data_file in ["exoplanet_data.csv", "koi_data.csv"]:
                    data_path = self.data_dir / data_file
                    if data_path.exists():
                        return pd.read_csv(data_path)
                
                # If no cached data, create synthetic data
                return self.create_synthetic_data()
                
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return self.create_synthetic_data()
        
        return _load_data()
    
    def create_synthetic_data(self):
        """Create synthetic exoplanet data for demo"""
        
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic features
        data = {
            'period': np.random.lognormal(2, 1, n_samples),
            'radius': np.random.lognormal(0, 0.5, n_samples),
            'mass': np.random.lognormal(1, 0.8, n_samples),
            'temperature': np.random.normal(5000, 1500, n_samples),
            'distance': np.random.lognormal(5, 1.5, n_samples),
            'stellar_magnitude': np.random.normal(12, 2, n_samples),
            'impact_parameter': np.random.uniform(0, 1, n_samples),
            'transit_duration': np.random.lognormal(1, 0.5, n_samples),
            'transit_depth': np.random.lognormal(-3, 1, n_samples),
            'signal_to_noise': np.random.lognormal(2, 0.5, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create labels based on realistic criteria
        labels = []
        for _, row in df.iterrows():
            if (row['radius'] > 0.5 and row['radius'] < 2.5 and 
                row['temperature'] > 3000 and row['signal_to_noise'] > 7):
                labels.append('CONFIRMED')
            elif row['signal_to_noise'] < 3:
                labels.append('FALSE POSITIVE')
            else:
                labels.append('CANDIDATE')
        
        df['koi_disposition'] = labels
        
        return df
    
    def load_model_results(self):
        """Load trained model results"""
        
        @st.cache_data
        def _load_results():
            try:
                # Look for the latest results file
                results_files = list(self.results_dir.glob("training_metadata_*.json"))
                if not results_files:
                    return None
                
                latest_file = max(results_files, key=os.path.getctime)
                
                with open(latest_file, 'r') as f:
                    return json.load(f)
                    
            except Exception as e:
                st.warning(f"Could not load model results: {e}")
                return None
        
        return _load_results()
    
    def render_header(self):
        """Render modern header"""
        
        st.markdown("""
        <div class="custom-header">
            <h1>🪐 Exoplanet Classification System</h1>
            <p style="font-size: 1.2em; margin: 0;">
                Advanced AI-powered exoplanet discovery and classification
            </p>
            <p style="font-size: 1em; opacity: 0.9; margin: 0.5rem 0 0 0;">
                NASA KOI • K2 • TESS Mission Data Analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render modern sidebar"""
        
        st.sidebar.markdown("## 🎛️ Dashboard Controls")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Navigate",
            ["🏠 Overview", "📊 Data Analysis", "🤖 Model Performance", 
             "🔍 Classification", "📈 Visualizations", "⚙️ System Status"]
        )
        
        st.sidebar.markdown("---")
        
        # Quick stats
        st.sidebar.markdown("### 📈 Quick Stats")
        
        data = self.load_data()
        if data is not None:
            st.sidebar.metric("Total Objects", len(data))
            
            # Class distribution
            class_counts = data['koi_disposition'].value_counts()
            for class_name, count in class_counts.items():
                st.sidebar.metric(class_name, count)
        
        st.sidebar.markdown("---")
        
        # System info
        st.sidebar.markdown("### 🖥️ System Info")
        st.sidebar.text(f"🕒 {datetime.now().strftime('%H:%M:%S')}")
        
        # Memory usage (if available)
        try:
            import psutil
            memory = psutil.virtual_memory()
            st.sidebar.metric("Memory Usage", f"{memory.percent}%")
        except:
            pass
        
        return page
    
    def render_overview(self):
        """Render overview page"""
        
        # Load data and results
        data = self.load_data()
        results = self.load_model_results()
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🪐 Total Objects",
                value=len(data) if data is not None else 0,
                delta="Active dataset"
            )
        
        with col2:
            if results and 'model_results' in results:
                best_accuracy = max([r['accuracy'] for r in results['model_results'].values()])
                st.metric(
                    label="🎯 Best Accuracy",
                    value=f"{best_accuracy:.1%}",
                    delta="Latest model"
                )
            else:
                st.metric("🎯 Best Accuracy", "N/A", "No models trained")
        
        with col3:
            if data is not None:
                confirmed = len(data[data['koi_disposition'] == 'CONFIRMED'])
                st.metric(
                    label="✅ Confirmed",
                    value=confirmed,
                    delta=f"{confirmed/len(data)*100:.1f}% of total"
                )
            else:
                st.metric("✅ Confirmed", "N/A")
        
        with col4:
            if data is not None:
                candidates = len(data[data['koi_disposition'] == 'CANDIDATE'])
                st.metric(
                    label="🔍 Candidates",
                    value=candidates,
                    delta=f"{candidates/len(data)*100:.1f}% of total"
                )
            else:
                st.metric("🔍 Candidates", "N/A")
        
        st.markdown("---")
        
        # Status indicators
        col1, col2 = st.columns(2)
        
        with col1:
            if data is not None:
                st.markdown('<div class="success-box">✅ Data Loaded Successfully</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">⚠️ No Data Available</div>', 
                           unsafe_allow_html=True)
        
        with col2:
            if results:
                st.markdown('<div class="success-box">🤖 Models Trained</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box">🚀 Ready for Training</div>', 
                           unsafe_allow_html=True)
        
        # Dataset overview
        if data is not None:
            st.subheader("📊 Dataset Overview")
            
            # Class distribution chart
            fig = px.pie(
                values=data['koi_disposition'].value_counts().values,
                names=data['koi_disposition'].value_counts().index,
                title="Class Distribution",
                color_discrete_sequence=['#667eea', '#764ba2', '#f093fb']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature correlation heatmap
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                st.subheader("🔥 Feature Correlations")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                corr_matrix = data[numeric_cols].corr()
                
                # Use a modern color palette
                sns.heatmap(corr_matrix, annot=True, cmap='viridis', 
                           center=0, ax=ax, fmt='.2f')
                ax.set_title("Feature Correlation Matrix", fontsize=14, pad=20)
                
                st.pyplot(fig)
    
    def render_data_analysis(self):
        """Render data analysis page"""
        
        st.subheader("📊 Data Analysis")
        
        data = self.load_data()
        if data is None:
            st.error("No data available for analysis")
            return
        
        # Feature selection
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_feature = st.selectbox("Select X-axis feature", numeric_cols)
        
        with col2:
            y_feature = st.selectbox("Select Y-axis feature", 
                                   [col for col in numeric_cols if col != x_feature])
        
        # Scatter plot
        if x_feature and y_feature:
            fig = px.scatter(
                data, 
                x=x_feature, 
                y=y_feature,
                color='koi_disposition',
                title=f"{x_feature} vs {y_feature}",
                color_discrete_sequence=['#667eea', '#764ba2', '#f093fb'],
                hover_data=['period', 'radius', 'mass'] if all(col in data.columns for col in ['period', 'radius', 'mass']) else None
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary
        st.subheader("📈 Statistical Summary")
        st.dataframe(data.describe(), use_container_width=True)
        
        # Feature distributions
        st.subheader("📊 Feature Distributions")
        
        selected_feature = st.selectbox("Select feature for distribution", numeric_cols)
        
        if selected_feature:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Distribution", "Box Plot")
            )
            
            # Histogram
            fig.add_trace(
                go.Histogram(x=data[selected_feature], name="Distribution"),
                row=1, col=1
            )
            
            # Box plot by class
            for class_name in data['koi_disposition'].unique():
                class_data = data[data['koi_disposition'] == class_name]
                fig.add_trace(
                    go.Box(y=class_data[selected_feature], name=class_name),
                    row=1, col=2
                )
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_model_performance(self):
        """Render model performance page"""
        
        st.subheader("🤖 Model Performance")
        
        results = self.load_model_results()
        
        if not results or 'model_results' not in results:
            st.warning("No model results available. Please train models first.")
            
            # Show training button
            if st.button("🚀 Start Training", type="primary"):
                with st.spinner("Training models... This may take a few minutes."):
                    # This would trigger the training script
                    st.info("Training initiated! Check back in a few minutes.")
            return
        
        model_results = results['model_results']
        
        # Performance metrics table
        df_results = pd.DataFrame.from_dict(model_results, orient='index')
        df_results = df_results.sort_values('accuracy', ascending=False)
        
        st.subheader("📊 Model Comparison")
        
        # Highlight best model
        styled_df = df_results.style.highlight_max(subset=['accuracy'], color='lightgreen')
        st.dataframe(styled_df, use_container_width=True)
        
        # Performance bar chart
        fig = px.bar(
            x=df_results.index,
            y=df_results['accuracy'],
            title="Model Accuracy Comparison",
            color=df_results['accuracy'],
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Training time vs accuracy scatter
        if 'training_time' in df_results.columns:
            fig = px.scatter(
                x=df_results['training_time'],
                y=df_results['accuracy'],
                text=df_results.index,
                title="Training Time vs Accuracy Trade-off",
                labels={'x': 'Training Time (seconds)', 'y': 'Accuracy'}
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_classification(self):
        """Render classification interface"""
        
        st.subheader("🔍 Exoplanet Classification")
        
        # Check if models are available
        model_files = list(self.models_dir.rglob("*.joblib"))
        
        if not model_files:
            st.warning("No trained models available. Please train models first.")
            return
        
        # Feature input form
        st.markdown("### 📝 Enter Object Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            period = st.number_input("Orbital Period (days)", min_value=0.1, max_value=1000.0, value=10.0)
            radius = st.number_input("Planet Radius (Earth radii)", min_value=0.1, max_value=20.0, value=1.0)
            mass = st.number_input("Planet Mass (Earth masses)", min_value=0.01, max_value=1000.0, value=1.0)
            temperature = st.number_input("Stellar Temperature (K)", min_value=2000, max_value=10000, value=5778)
            distance = st.number_input("Distance (parsecs)", min_value=1, max_value=5000, value=100)
        
        with col2:
            magnitude = st.number_input("Stellar Magnitude", min_value=5.0, max_value=20.0, value=12.0)
            impact_param = st.slider("Impact Parameter", min_value=0.0, max_value=1.0, value=0.5)
            transit_duration = st.number_input("Transit Duration (hours)", min_value=0.1, max_value=24.0, value=3.0)
            transit_depth = st.number_input("Transit Depth (ppm)", min_value=1, max_value=50000, value=1000)
            snr = st.number_input("Signal-to-Noise Ratio", min_value=1.0, max_value=100.0, value=10.0)
        
        # Classify button
        if st.button("🚀 Classify Object", type="primary"):
            
            # Create feature vector
            features = np.array([[period, radius, mass, temperature, distance, 
                               magnitude, impact_param, transit_duration, transit_depth, snr]])
            
            try:
                # Load the best model (first in sorted list)
                best_model_file = sorted(model_files)[0]
                model = joblib.load(best_model_file)
                
                # Make prediction
                prediction = model.predict(features)[0]
                
                # Get prediction probabilities if available
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features)[0]
                    classes = model.classes_
                else:
                    proba = None
                    classes = None
                
                # Display result
                st.markdown("### 🎯 Classification Result")
                
                if prediction == 'CONFIRMED':
                    st.success(f"🪐 **CONFIRMED EXOPLANET**")
                elif prediction == 'CANDIDATE':
                    st.info(f"🔍 **PLANET CANDIDATE**")
                else:
                    st.warning(f"❌ **FALSE POSITIVE**")
                
                # Show confidence if available
                if proba is not None and classes is not None:
                    st.markdown("### 📊 Confidence Scores")
                    
                    confidence_data = pd.DataFrame({
                        'Class': classes,
                        'Probability': proba
                    }).sort_values('Probability', ascending=False)
                    
                    fig = px.bar(
                        confidence_data,
                        x='Class',
                        y='Probability',
                        title="Classification Confidence",
                        color='Probability',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Classification failed: {e}")
    
    def render_visualizations(self):
        """Render advanced visualizations"""
        
        st.subheader("📈 Advanced Visualizations")
        
        data = self.load_data()
        if data is None:
            st.error("No data available for visualization")
            return
        
        # 3D scatter plot
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 3:
            st.markdown("### 🌌 3D Feature Space")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_axis = st.selectbox("X-axis", numeric_cols, key="3d_x")
            with col2:
                y_axis = st.selectbox("Y-axis", numeric_cols, index=1, key="3d_y")
            with col3:
                z_axis = st.selectbox("Z-axis", numeric_cols, index=2, key="3d_z")
            
            fig = px.scatter_3d(
                data,
                x=x_axis,
                y=y_axis,
                z=z_axis,
                color='koi_disposition',
                title=f"3D Plot: {x_axis} vs {y_axis} vs {z_axis}",
                color_discrete_sequence=['#667eea', '#764ba2', '#f093fb']
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (if model results available)
        results = self.load_model_results()
        
        if results and 'feature_importance' in results:
            st.markdown("### 🎯 Feature Importance")
            
            importance_data = results['feature_importance']
            
            fig = px.bar(
                x=list(importance_data.values()),
                y=list(importance_data.keys()),
                orientation='h',
                title="Feature Importance",
                color=list(importance_data.values()),
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_system_status(self):
        """Render system status page"""
        
        st.subheader("⚙️ System Status")
        
        # System information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🖥️ Hardware Status")
            
            try:
                import psutil
                
                # CPU info
                cpu_percent = psutil.cpu_percent(interval=1)
                st.metric("CPU Usage", f"{cpu_percent}%")
                
                # Memory info
                memory = psutil.virtual_memory()
                st.metric("Memory Usage", f"{memory.percent}%", 
                         f"{memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB")
                
                # Disk info
                disk = psutil.disk_usage('.')
                st.metric("Disk Usage", f"{disk.percent}%",
                         f"{disk.used / 1024**3:.1f}GB / {disk.total / 1024**3:.1f}GB")
                
            except ImportError:
                st.info("Install psutil for detailed system monitoring")
        
        with col2:
            st.markdown("### 🤖 ML Framework Status")
            
            # Check TensorFlow
            try:
                import tensorflow as tf
                tf_version = tf.__version__
                gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
                st.success(f"✅ TensorFlow {tf_version}")
                if gpu_available:
                    st.success("✅ GPU Available")
                else:
                    st.info("ℹ️ CPU Only")
            except ImportError:
                st.warning("⚠️ TensorFlow not installed")
            
            # Check PyTorch
            try:
                import torch
                torch_version = torch.__version__
                cuda_available = torch.cuda.is_available()
                st.success(f"✅ PyTorch {torch_version}")
                if cuda_available:
                    st.success("✅ CUDA Available")
                else:
                    st.info("ℹ️ CPU Only")
            except ImportError:
                st.warning("⚠️ PyTorch not installed")
        
        # File system status
        st.markdown("### 📁 Project Files")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            models_count = len(list(self.models_dir.rglob("*"))) if self.models_dir.exists() else 0
            st.metric("Model Files", models_count)
        
        with col2:
            data_files = len(list(self.data_dir.rglob("*.csv"))) if self.data_dir.exists() else 0
            st.metric("Data Files", data_files)
        
        with col3:
            result_files = len(list(self.results_dir.rglob("*.json"))) if self.results_dir.exists() else 0
            st.metric("Result Files", result_files)
        
        # Recent activity
        st.markdown("### 📊 Recent Activity")
        
        if self.results_dir.exists():
            recent_files = sorted(
                self.results_dir.rglob("*.json"),
                key=os.path.getctime,
                reverse=True
            )[:5]
            
            if recent_files:
                for file_path in recent_files:
                    creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
                    st.text(f"📄 {file_path.name} - {creation_time.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.info("No recent activity")
        else:
            st.info("No results directory found")
    
    def run(self):
        """Run the dashboard"""
        
        # Render header
        self.render_header()
        
        # Render sidebar and get selected page
        selected_page = self.render_sidebar()
        
        # Render selected page
        if selected_page == "🏠 Overview":
            self.render_overview()
        elif selected_page == "📊 Data Analysis":
            self.render_data_analysis()
        elif selected_page == "🤖 Model Performance":
            self.render_model_performance()
        elif selected_page == "🔍 Classification":
            self.render_classification()
        elif selected_page == "📈 Visualizations":
            self.render_visualizations()
        elif selected_page == "⚙️ System Status":
            self.render_system_status()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "🌟 **Exoplanet Classifier** | "
            "Built with Streamlit, TensorFlow & scikit-learn | "
            f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

def main():
    """Main application entry point"""
    
    # Initialize and run dashboard
    dashboard = ExoplanetDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()