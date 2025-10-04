#!/usr/bin/env python3
"""
🤖 Smart Model Selection Demo for NASA Exoplanet Classifier
Quick demonstration of automatic model selection capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
from nasa_smart_classifier import SmartNASAExoplanetClassifier
import json

def run_smart_demo():
    """Run the smart classifier demo"""
    st.title("🤖 NASA Smart AI Model Selection Demo")
    
    st.markdown("""
    ## 🌟 Welcome to Smart AI Model Selection!
    
    This demo shows how our Smart AI automatically selects the optimal machine learning model 
    based on your data characteristics. No more guessing which model to use!
    """)
    
    # Create demo scenarios
    st.markdown("### 🎯 Select a Demo Scenario")
    
    scenarios = {
        "Small Clean Dataset": {
            "n_samples": 800, 
            "noise": 0.1, 
            "missing": 0.05,
            "description": "Clean, small astronomical survey with minimal noise"
        },
        "Medium Noisy Dataset": {
            "n_samples": 3000, 
            "noise": 0.4, 
            "missing": 0.15,
            "description": "Ground-based observations with atmospheric interference"
        },
        "Large Imbalanced Dataset": {
            "n_samples": 8000, 
            "noise": 0.2, 
            "missing": 0.1,
            "description": "Space-based survey with few confirmed exoplanets"
        }
    }
    
    selected_scenario = st.selectbox(
        "Choose a scenario:",
        list(scenarios.keys())
    )
    
    scenario_info = scenarios[selected_scenario]
    
    st.info(f"**Scenario:** {scenario_info['description']}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📏 Samples", f"{scenario_info['n_samples']:,}")
    with col2:
        st.metric("📡 Noise Level", f"{scenario_info['noise']:.1f}")
    with col3:
        st.metric("🕳️ Missing Data", f"{scenario_info['missing']:.1%}")
    
    # Run demo button
    if st.button("🚀 Run Smart AI Demo", type="primary"):
        
        with st.spinner("🤖 Smart AI analyzing scenario and selecting optimal model..."):
            # Create synthetic dataset
            df = create_demo_dataset(**scenario_info)
            
            # Initialize smart classifier
            smart_classifier = SmartNASAExoplanetClassifier()
            
            # Run smart training
            results = smart_classifier.smart_train(df, test_size=0.2)
            
            # Display results
            st.success("✅ Smart AI Analysis Complete!")
            
            # Selected model
            st.markdown("### 🎯 Smart AI Decision")
            
            selected_model = smart_classifier.selected_model.replace('_', ' ').title()
            reason = smart_classifier.selection_reason
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"**Selected Model:** {selected_model}")
                st.metric("🎯 Accuracy", f"{results[smart_classifier.selected_model]['accuracy']:.1%}")
            
            with col2:
                st.info("**Why this model?**")
                st.write(reason)
            
            # Data characteristics
            st.markdown("### 📊 Data Analysis")
            
            chars = smart_classifier.data_characteristics
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📏 Samples", f"{chars['n_samples']:,}")
                st.metric("🔬 Features", chars['n_features'])
                
            with col2:
                st.metric("🕳️ Missing Data", f"{chars['missing_ratio']:.1%}")
                st.metric("⚖️ Class Balance", f"{chars['imbalance_ratio']:.3f}")
                
            with col3:
                st.metric("🎯 Outliers", f"{chars['outlier_ratio']:.1%}")
                st.metric("📡 Noise Level", f"{chars['noise_level']:.3f}")
                
            with col4:
                st.metric("🔗 Correlation", f"{chars['feature_correlation']:.3f}")
                st.metric("📊 Numeric Features", chars['numeric_features'])
            
            # Model comparison
            if len(results) > 1:
                st.markdown("### 🏆 All Models Comparison")
                
                comparison_data = []
                for model_name, result in results.items():
                    comparison_data.append({
                        'Model': model_name.replace('_', ' ').title(),
                        'Accuracy': f"{result['accuracy']:.1%}",
                        'CV Score': f"{result['cv_mean']:.1%} ± {result['cv_std']:.1%}",
                        'Selected': '🎯' if result.get('is_selected', False) else ''
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
            
            # Show decision logic
            with st.expander("🧠 View Smart AI Decision Logic"):
                st.markdown("**Smart AI Decision Tree:**")
                
                if chars['n_samples'] < 1000:
                    st.info("Small dataset detected → Tree-based models preferred")
                elif chars['n_samples'] < 5000:
                    st.info("Medium dataset detected → Ensemble methods considered")
                else:
                    st.info("Large dataset detected → All models viable")
                
                if chars['noise_level'] > 0.3:
                    st.warning("High noise detected → Extra Trees recommended for robustness")
                
                if chars['imbalance_ratio'] < 0.3:
                    st.warning("Class imbalance detected → Balanced models preferred")
                
                if chars['outlier_ratio'] > 0.1:
                    st.warning("Outliers detected → Robust algorithms selected")

def create_demo_dataset(n_samples, noise, missing, **kwargs):
    """Create synthetic dataset for demo"""
    np.random.seed(42)
    
    # Create base astronomical features
    data = {
        'koi_period': np.random.lognormal(2, 1, n_samples),
        'koi_prad': np.random.lognormal(0, 0.5, n_samples),
        'koi_teq': 200 + np.random.exponential(200, n_samples),
        'koi_steff': 5000 + np.random.normal(0, 1000, n_samples),
        'koi_smass': np.random.lognormal(0, 0.3, n_samples),
        'koi_srad': np.random.lognormal(0, 0.2, n_samples),
        'koi_dor': np.random.uniform(2, 50, n_samples)
    }
    
    # Add noise based on scenario
    for key in data:
        noise_factor = np.random.normal(1, noise, n_samples)
        data[key] = data[key] * noise_factor
    
    # Create realistic target distribution
    target = []
    for i in range(n_samples):
        if data['koi_period'][i] < 100 and data['koi_prad'][i] < 4:
            if np.random.random() < 0.7:
                target.append('CONFIRMED')
            else:
                target.append('CANDIDATE')
        else:
            if np.random.random() < 0.8:
                target.append('FALSE POSITIVE')
            else:
                target.append('CANDIDATE')
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df['koi_disposition'] = target
    
    # Add missing values based on scenario
    for col in df.columns[:-1]:  # Don't add missing to target
        missing_indices = np.random.choice(n_samples, int(n_samples * missing), replace=False)
        df.loc[missing_indices, col] = np.nan
    
    return df

if __name__ == "__main__":
    run_smart_demo()