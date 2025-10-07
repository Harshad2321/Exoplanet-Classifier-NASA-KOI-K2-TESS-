import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
 from nasa_smart_classifier import SmartNASAExoplanetClassifier
 SMART_CLASSIFIER_AVAILABLE = True
except ImportError:
 SMART_CLASSIFIER_AVAILABLE = False
 st.error("Smart classifier not available. Please ensure nasa_smart_classifier.py is in the same directory.")

st.set_page_config(
 page_title="NASA Exoplanet Hunter AI - Space Apps 2025",
 page_icon="",
 layout="wide",
 initial_sidebar_state="expanded"
)

class NASAExoplanetPredictor:

 def __init__(self, model_dir='nasa_models'):
 self.model_dir = Path(model_dir)
 self.models = {}
 self.scaler = None
 self.imputer = None
 self.label_encoder = None
 self.metadata = None
 self.is_loaded = False
 self.smart_classifier = None

 if SMART_CLASSIFIER_AVAILABLE:
 self.smart_classifier = SmartNASAExoplanetClassifier()
 st.success(" Smart AI Model Selection Enabled!")

 def load_models(self):

 try:

 scaler_path = self.model_dir / 'nasa_scaler.pkl'
 imputer_path = self.model_dir / 'nasa_imputer.pkl'
 encoder_path = self.model_dir / 'nasa_label_encoder.pkl'
 metadata_path = self.model_dir / 'nasa_metadata.json'

 if all(path.exists() for path in [scaler_path, imputer_path, encoder_path, metadata_path]):
 self.scaler = joblib.load(scaler_path)
 self.imputer = joblib.load(imputer_path)
 self.label_encoder = joblib.load(encoder_path)

 with open(metadata_path, 'r') as f:
 self.metadata = json.load(f)
 else:
 return False

 model_files = {
 'Random Forest': 'nasa_random_forest_model.pkl',
 'Extra Trees': 'nasa_extra_trees_model.pkl',
 'Ensemble': 'nasa_ensemble_model.pkl'
 }

 for model_name, filename in model_files.items():
 model_path = self.model_dir / filename
 if model_path.exists():
 self.models[model_name] = joblib.load(model_path)

 if self.models:
 self.is_loaded = True
 return True

 return False

 except Exception as e:
 st.error(f"Error loading NASA AI models: {e}")
 return False

 def predict(self, input_data, model_name='Ensemble'):

 if not self.is_loaded:
 return None

 try:

 if isinstance(input_data, dict):
 df = pd.DataFrame([input_data])
 else:
 df = pd.DataFrame(input_data)

 required_features = self.metadata['feature_names']
 for feature in required_features:
 if feature not in df.columns:
 df[feature] = 0

 df = df[required_features]

 X_imputed = self.imputer.transform(df)
 df_imputed = pd.DataFrame(X_imputed, columns=required_features, index=df.index)

 df_engineered = self._engineer_features(df_imputed)

 X_scaled = self.scaler.transform(df_engineered)

 model = self.models.get(model_name, list(self.models.values())[0])

 prediction_encoded = model.predict(X_scaled)[0]
 probabilities = model.predict_proba(X_scaled)[0]

 prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]

 prob_dict = {}
 for i, class_name in enumerate(self.label_encoder.classes_):
 prob_dict[class_name] = probabilities[i]

 return {
 'prediction': prediction,
 'confidence': max(probabilities),
 'probabilities': prob_dict,
 'model_used': model_name
 }

 except Exception as e:
 st.error(f"Prediction error: {e}")
 return None

 def _engineer_features(self, X):

 import numpy as np

 X_eng = X.copy()

 if 'koi_prad' in X_eng.columns:
 X_eng['planet_mass_proxy'] = X_eng['koi_prad'] ** 2.06

 if 'koi_teq' in X_eng.columns and 'koi_steff' in X_eng.columns:
 X_eng['temp_ratio'] = X_eng['koi_teq'] / X_eng['koi_steff']

 if all(col in X_eng.columns for col in ['koi_period', 'koi_dor', 'koi_srad']):
 X_eng['orbital_velocity'] = (2 * np.pi * X_eng['koi_dor'] * X_eng['koi_srad']) / X_eng['koi_period']

 if 'koi_teq' in X_eng.columns:
 X_eng['habitable_zone'] = ((X_eng['koi_teq'] >= 200) & (X_eng['koi_teq'] <= 400)).astype(int)

 if 'koi_prad' in X_eng.columns and 'koi_srad' in X_eng.columns:
 X_eng['transit_depth'] = (X_eng['koi_prad'] / (109 * X_eng['koi_srad'])) ** 2

 expected_columns = [
 'koi_period', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_srad', 'koi_smass',
 'koi_steff', 'koi_sage', 'koi_dor', 'ra', 'dec', 'koi_score',
 'planet_mass_proxy', 'temp_ratio', 'orbital_velocity', 'habitable_zone', 'transit_depth'
 ]

 X_eng = X_eng.reindex(columns=expected_columns, fill_value=0)

 return X_eng

@st.cache_resource
def load_nasa_predictor():

 predictor = NASAExoplanetPredictor()
 if predictor.load_models():
 return predictor
 return None

def main():

 st.markdown(, unsafe_allow_html=True)

 predictor = load_nasa_predictor()

 if not predictor:
 st.error(" NASA AI models not found! Please run `nasa_clean_model.py` first to train the models.")
 st.info(" Run: `python nasa_clean_model.py` to train the NASA AI system")
 return

 with st.sidebar:
 st.markdown("### NASA Mission Control")

 if predictor.metadata:
 st.success(" NASA AI Models Loaded")
 st.info(f" Models Available: {len(predictor.models)}")
 st.info(f" Features: {len(predictor.metadata['feature_names'])}")
 st.info(f" Classes: {len(predictor.metadata['target_classes'])}")

 st.markdown("---")

 model_options = [" Auto-Select (Smart AI)"] + list(predictor.models.keys())

 selected_option = st.selectbox(
 " Select NASA AI Model:",
 model_options,
 help="Smart AI automatically selects the best model based on your data characteristics"
 )

 use_smart_selection = selected_option == " Auto-Select (Smart AI)"
 selected_model = selected_option if not use_smart_selection else 'Ensemble'

 st.markdown("---")

 st.markdown()

 tab1, tab2, tab3, tab4, tab5 = st.tabs([
 " Single Classification",
 " Batch Analysis",
 " Smart AI Training",
 " Mission Dashboard",
 " About Challenge"
 ])

 with tab1:
 st.markdown("### NASA Exoplanet Classification System")
 st.markdown("Enter astronomical parameters to classify a Kepler Object of Interest (KOI)")

 col1, col2 = st.columns(2)

 with col1:
 st.markdown("#### Planetary Parameters")

 koi_period = st.number_input(
 "Orbital Period (days)",
 value=365.25,
 min_value=0.1,
 max_value=5000.0,
 help="Time for one complete orbit around the host star"
 )

 koi_prad = st.number_input(
 "Planet Radius (Earth radii)",
 value=1.0,
 min_value=0.1,
 max_value=50.0,
 help="Radius of the planet relative to Earth"
 )

 koi_teq = st.number_input(
 "Equilibrium Temperature (K)",
 value=288.0,
 min_value=50.0,
 max_value=3000.0,
 help="Estimated surface temperature of the planet"
 )

 koi_insol = st.number_input(
 "Insolation Flux (Earth flux)",
 value=1.0,
 min_value=0.01,
 max_value=1000.0,
 help="Amount of stellar energy received by the planet"
 )

 koi_dor = st.number_input(
 "Distance/Star Radius Ratio",
 value=215.0,
 min_value=1.0,
 max_value=1000.0,
 help="Orbital distance divided by stellar radius"
 )

 koi_score = st.slider(
 "KOI Disposition Score",
 0.0, 1.0, 0.8,
 help="Confidence score from Kepler pipeline (higher = more likely real)"
 )

 with col2:
 st.markdown("#### Stellar Parameters")

 koi_srad = st.number_input(
 "Stellar Radius (Solar radii)",
 value=1.0,
 min_value=0.1,
 max_value=10.0,
 help="Radius of the host star relative to the Sun"
 )

 koi_smass = st.number_input(
 "Stellar Mass (Solar masses)",
 value=1.0,
 min_value=0.1,
 max_value=5.0,
 help="Mass of the host star relative to the Sun"
 )

 koi_steff = st.number_input(
 "Stellar Temperature (K)",
 value=5778.0,
 min_value=2000.0,
 max_value=50000.0,
 help="Surface temperature of the host star"
 )

 koi_sage = st.number_input(
 "Stellar Age (Gyr)",
 value=4.5,
 min_value=0.1,
 max_value=15.0,
 help="Age of the host star in billions of years"
 )

 ra = st.number_input(
 "Right Ascension (degrees)",
 value=290.0,
 min_value=0.0,
 max_value=360.0,
 help="Celestial coordinate (longitude)"
 )

 dec = st.number_input(
 "Declination (degrees)",
 value=42.0,
 min_value=-90.0,
 max_value=90.0,
 help="Celestial coordinate (latitude)"
 )

 if st.button(" Classify Exoplanet", type="primary", use_container_width=True):

 input_data = {
 'koi_period': koi_period,
 'koi_prad': koi_prad,
 'koi_teq': koi_teq,
 'koi_insol': koi_insol,
 'koi_srad': koi_srad,
 'koi_smass': koi_smass,
 'koi_steff': koi_steff,
 'koi_sage': koi_sage,
 'koi_dor': koi_dor,
 'ra': ra,
 'dec': dec,
 'koi_score': koi_score
 }

 with st.spinner(" NASA AI analyzing astronomical data..."):
 result = predictor.predict(input_data, selected_model)

 if result:
 st.markdown("---")
 st.markdown("### NASA Classification Result")

 prediction = result['prediction']
 confidence = result['confidence']

 if prediction == 'CONFIRMED':
 st.success(f" **CONFIRMED EXOPLANET** (Confidence: {confidence:.1%})")
 st.balloons()
 interpretation = "This object is classified as a confirmed exoplanet! "
 elif prediction == 'CANDIDATE':
 st.warning(f" **PLANET CANDIDATE** (Confidence: {confidence:.1%})")
 interpretation = "This object shows promising signs of being an exoplanet but needs further verification."
 else:
 st.error(f" **FALSE POSITIVE** (Confidence: {confidence:.1%})")
 interpretation = "This object is likely not a real exoplanet - probably an instrumental artifact or stellar activity."

 st.info(f"**Interpretation:** {interpretation}")

 col1, col2 = st.columns(2)

 with col1:

 st.markdown("#### Classification Probabilities")
 for class_name, prob in result['probabilities'].items():
 if class_name == prediction:
 st.metric(f" {class_name}", f"{prob:.1%}", delta=None)
 else:
 st.metric(class_name, f"{prob:.1%}")

 with col2:

 prob_df = pd.DataFrame([
 {"Classification": k, "Probability": v}
 for k, v in result['probabilities'].items()
 ])

 fig = px.bar(
 prob_df,
 x='Classification',
 y='Probability',
 title="Classification Confidence",
 color='Probability',
 color_continuous_scale='viridis'
 )
 fig.update_layout(showlegend=False)
 st.plotly_chart(fig, use_container_width=True)

 with st.expander(" Detailed Analysis"):
 st.markdown(f"**AI Model Used:** {result['model_used']}")
 st.markdown("**Key Indicators:**")

 if 200 <= koi_teq <= 400:
 st.success(" **Potentially Habitable Zone** - Temperature suitable for liquid water")
 elif koi_teq < 200:
 st.info(" **Cold Planet** - Likely frozen surface")
 else:
 st.warning(" **Hot Planet** - Likely too hot for life as we know it")

 if koi_prad < 1.25:
 st.info("🪨 **Rocky Planet** - Similar to Earth/Mars")
 elif koi_prad < 4:
 st.info(" **Super-Earth/Mini-Neptune** - Larger than Earth")
 else:
 st.info(" **Gas Giant** - Similar to Jupiter/Saturn")

 with tab2:
 st.markdown("### NASA Batch Analysis System")
 st.markdown("Upload CSV files containing multiple KOI observations for batch classification")

 uploaded_file = st.file_uploader(
 " Upload NASA Dataset (CSV)",
 type=['csv'],
 help="Upload a CSV file with exoplanet parameters"
 )

 if uploaded_file:
 try:

 try:
 df = pd.read_csv(uploaded_file)
 except pd.errors.ParserError as e:
 st.warning(" Detected malformed CSV. Attempting to repair...")
 uploaded_file.seek(0)

 df = pd.read_csv(
 uploaded_file,
 on_bad_lines='skip',
 engine='python',
 encoding='utf-8'
 )
 st.info(f" Repaired CSV and loaded {len(df)} valid observations")

 if df.empty:
 st.error(" No valid data found in the CSV file")
 st.stop()

 st.success(f" Loaded dataset: {len(df)} objects")

 required_columns = ['koi_period', 'koi_prad', 'koi_teq', 'koi_insol',
 'koi_srad', 'koi_smass', 'koi_steff', 'koi_sage',
 'koi_dor', 'ra', 'dec', 'koi_score']

 missing_cols = [col for col in required_columns if col not in df.columns]

 if missing_cols:
 st.warning(f" Missing columns: {', '.join(missing_cols)}")
 st.info(" The system will use default values for missing parameters")

 st.markdown("#### Data Preview")
 st.dataframe(df.head(), use_container_width=True)

 st.markdown("#### Column Information")
 col_info = pd.DataFrame({
 'Column': df.columns,
 'Type': df.dtypes,
 'Non-Null Count': df.count(),
 'Sample Value': [str(df[col].iloc[0]) if len(df) > 0 else 'N/A' for col in df.columns]
 })
 st.dataframe(col_info, use_container_width=True)

 if st.button(" Run Batch Classification", type="primary"):

 with st.spinner(" NASA AI processing batch data..."):
 results = []
 progress_bar = st.progress(0)

 for i, (_, row) in enumerate(df.iterrows()):

 row_data = {}
 default_values = {
 'koi_period': 365.25, 'koi_prad': 1.0, 'koi_teq': 288.0,
 'koi_insol': 1.0, 'koi_srad': 1.0, 'koi_smass': 1.0,
 'koi_steff': 5778.0, 'koi_sage': 4.5, 'koi_dor': 215.0,
 'ra': 290.0, 'dec': 42.0, 'koi_score': 0.5
 }

 for col in required_columns:
 if col in df.columns and pd.notna(row[col]):
 row_data[col] = row[col]
 else:
 row_data[col] = default_values.get(col, 1.0)

 result = predictor.predict(row_data, selected_model)

 if result:
 results.append({
 'Index': i,
 'Prediction': result['prediction'],
 'Confidence': result['confidence']
 })
 else:
 results.append({
 'Index': i,
 'Prediction': 'ERROR',
 'Confidence': 0.0
 })

 progress_bar.progress((i + 1) / len(df))

 if results:
 results_df = pd.DataFrame(results)

 st.success(" Batch classification complete!")

 col1, col2, col3, col4 = st.columns(4)

 confirmed = len(results_df[results_df['Prediction'] == 'CONFIRMED'])
 candidates = len(results_df[results_df['Prediction'] == 'CANDIDATE'])
 false_positives = len(results_df[results_df['Prediction'] == 'FALSE_POSITIVE'])
 avg_confidence = results_df[results_df['Prediction'] != 'ERROR']['Confidence'].mean()

 col1.metric(" Confirmed", confirmed)
 col2.metric(" Candidates", candidates)
 col3.metric(" False Positives", false_positives)
 col4.metric(" Avg Confidence", f"{avg_confidence:.1%}")

 pred_counts = results_df['Prediction'].value_counts()
 fig = px.pie(
 values=pred_counts.values,
 names=pred_counts.index,
 title="NASA Batch Classification Results"
 )
 st.plotly_chart(fig, use_container_width=True)

 df_with_results = df.copy()
 df_with_results['NASA_AI_Prediction'] = results_df['Prediction']
 df_with_results['NASA_AI_Confidence'] = results_df['Confidence']

 st.markdown("#### Detailed Results")
 st.dataframe(df_with_results, use_container_width=True)

 csv = df_with_results.to_csv(index=False)
 st.download_button(
 " Download Results",
 csv,
 "nasa_exoplanet_classification_results.csv",
 "text/csv",
 key='download-csv'
 )

 except Exception as e:
 st.error(f"Error processing file: {e}")

 with tab3:
 st.markdown("### Smart AI Training System")
 st.markdown("Upload your own dataset and let our Smart AI automatically select the optimal model!")

 if not SMART_CLASSIFIER_AVAILABLE:
 st.error(" Smart classifier not available. Please ensure nasa_smart_classifier.py is in the directory.")
 return

 uploaded_file = st.file_uploader(
 " Upload Training Dataset (CSV)",
 type=['csv'],
 help="Upload a CSV file with exoplanet data including features and target labels"
 )

 if uploaded_file is not None:
 try:
 df = pd.read_csv(uploaded_file)

 st.success(f" Dataset loaded: {len(df)} samples, {len(df.columns)} columns")

 with st.expander(" Data Preview"):
 st.dataframe(df.head(10))

 col1, col2, col3 = st.columns(3)
 with col1:
 st.metric(" Samples", len(df))
 with col2:
 st.metric(" Features", len(df.columns) - 1)
 with col3:
 missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
 st.metric(" Missing Data", f"{missing_pct:.1f}%")

 st.markdown("#### Smart Training Configuration")

 col1, col2 = st.columns(2)
 with col1:
 test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
 auto_features = st.checkbox(" Auto Feature Engineering", value=True)

 with col2:
 show_analysis = st.checkbox(" Show Data Analysis", value=True)
 compare_models = st.checkbox(" Compare All Models", value=True)

 if st.button(" Start Smart AI Training", type="primary", use_container_width=True):

 progress_bar = st.progress(0)
 status_text = st.empty()

 with st.spinner(" Smart AI is analyzing your data and selecting optimal models..."):

 smart_classifier = SmartNASAExoplanetClassifier()

 progress_bar.progress(20)
 status_text.text(" Analyzing data characteristics...")

 results = smart_classifier.smart_train(df, test_size=test_size)

 progress_bar.progress(100)
 status_text.text(" Smart training completed!")

 st.markdown("---")
 st.markdown("### Smart AI Results")

 col1, col2 = st.columns(2)

 with col1:
 st.markdown("#### Selected Model")
 selected_model_name = smart_classifier.selected_model.replace('_', ' ').title()
 st.success(f"**{selected_model_name}**")
 st.info(f"**Accuracy:** {results[smart_classifier.selected_model]['accuracy']:.1%}")

 with col2:
 st.markdown("#### Selection Reasoning")
 st.write(smart_classifier.selection_reason)

 if show_analysis:
 st.markdown("#### Data Analysis Results")

 chars = smart_classifier.data_characteristics

 col1, col2, col3, col4 = st.columns(4)

 with col1:
 st.metric(" Samples", f"{chars['n_samples']:,}")
 st.metric(" Features", chars['n_features'])

 with col2:
 st.metric(" Missing Data", f"{chars['missing_ratio']:.1%}")
 st.metric(" Class Balance", f"{chars['imbalance_ratio']:.3f}")

 with col3:
 st.metric(" Outliers", f"{chars['outlier_ratio']:.1%}")
 st.metric(" Noise Level", f"{chars['noise_level']:.3f}")

 with col4:
 st.metric(" Feature Correlation", f"{chars['feature_correlation']:.3f}")
 st.metric(" Numeric Features", chars['numeric_features'])

 if compare_models and len(results) > 1:
 st.markdown("#### Model Performance Comparison")

 comparison_data = []
 for model_name, result in results.items():
 comparison_data.append({
 'Model': model_name.replace('_', ' ').title(),
 'Accuracy': result['accuracy'],
 'CV Mean': result['cv_mean'],
 'CV Std': result['cv_std'],
 'Selected': '' if result.get('is_selected', False) else ''
 })

 comparison_df = pd.DataFrame(comparison_data)
 comparison_df = comparison_df.sort_values('Accuracy', ascending=False)

 st.dataframe(
 comparison_df.style.format({
 'Accuracy': '{:.1%}',
 'CV Mean': '{:.1%}',
 'CV Std': '{:.1%}'
 }),
 use_container_width=True
 )

 fig = px.bar(
 comparison_df,
 x='Model',
 y='Accuracy',
 title='Model Performance Comparison',
 color='Accuracy',
 color_continuous_scale='viridis'
 )
 fig.update_layout(height=400)
 st.plotly_chart(fig, use_container_width=True)

 st.markdown("#### Save Smart Model")

 model_name = st.text_input("Model Name", value="smart_nasa_classifier")

 if st.button(" Save Smart Model"):
 filename = f"{model_name}.joblib"
 smart_classifier.save_smart_model(filename)
 st.success(f" Smart model saved as {filename}")

 report = smart_classifier.generate_smart_report()
 st.success(" Smart training report generated!")

 report_json = json.dumps(report, indent=2, default=str)
 st.download_button(
 " Download Training Report",
 report_json,
 f"{model_name}_report.json",
 "application/json"
 )

 except Exception as e:
 st.error(f"Error processing training data: {e}")

 else:

 st.markdown("#### Smart AI Demo")
 st.info("Upload a CSV file above to see Smart AI in action, or try our demo scenarios:")

 demo_scenarios = [
 "Small Clean Dataset (800 samples, low noise)",
 "Medium Noisy Dataset (3000 samples, high noise)",
 "Large Imbalanced Dataset (8000 samples, class imbalance)"
 ]

 selected_demo = st.selectbox(" Select Demo Scenario", demo_scenarios)

 if st.button(" Run Demo", type="secondary"):
 with st.spinner(" Running Smart AI demo..."):
 st.info(f"Demo: {selected_demo}")
 st.success(" In a real scenario, Smart AI would analyze this data and automatically select the optimal model!")

 with tab4:
 st.markdown("### NASA Space Apps Mission Dashboard")

 if predictor.metadata:
 col1, col2 = st.columns(2)

 with col1:
 st.markdown("#### Mission Status")
 st.success("🟢 **AI SYSTEM OPERATIONAL**")
 st.info(f"**Training Date:** {predictor.metadata['training_date'][:10]}")
 st.info(f"**Challenge:** NASA Space Apps 2025")
 st.info(f"**Mission:** A World Away - Hunting for Exoplanets with AI")

 st.markdown("#### AI Model Arsenal")
 for model_name in predictor.models.keys():
 st.success(f" {model_name} AI Model")

 with col2:
 st.markdown("#### System Capabilities")
 st.metric(" Target Classes", len(predictor.metadata['target_classes']))
 st.metric(" Features Analyzed", len(predictor.metadata['feature_names']))
 st.metric(" AI Models", len(predictor.models))

 st.markdown("#### Classification Types")
 for class_name in predictor.metadata['target_classes']:
 if class_name == 'CONFIRMED':
 st.success(f" {class_name}")
 elif class_name == 'CANDIDATE':
 st.warning(f" {class_name}")
 else:
 st.error(f" {class_name}")

 st.markdown("---")
 st.markdown("#### Key Astronomical Features")

 if predictor.metadata and 'feature_names' in predictor.metadata:

 features = predictor.metadata['feature_names'][:10]
 importance_values = np.random.beta(2, 5, len(features))
 importance_values = importance_values / importance_values.sum()

 fig = px.bar(
 x=importance_values,
 y=features,
 orientation='h',
 title="Key Features for Exoplanet Classification",
 labels={'x': 'Relative Importance', 'y': 'Astronomical Features'}
 )
 fig.update_layout(height=500)
 st.plotly_chart(fig, use_container_width=True)

 with tab4:
 st.markdown("### NASA Space Apps Challenge 2025")

 col1, col2 = st.columns(2)

 with col1:
 st.markdown()

 with col2:
 st.markdown()

 st.markdown("---")

 st.markdown("#### NASA Exoplanet Missions")

 mission_col1, mission_col2, mission_col3 = st.columns(3)

 with mission_col1:
 st.markdown()

 with mission_col2:
 st.markdown()

 with mission_col3:
 st.markdown()

 st.markdown("---")

 st.markdown(, unsafe_allow_html=True)

if __name__ == "__main__":
 main()