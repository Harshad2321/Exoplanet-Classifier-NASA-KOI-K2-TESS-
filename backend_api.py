

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Optional
import json

logging.basicConfig(
 level=logging.INFO,
 format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
 title="NASA Exoplanet Classifier API",
 description="AI-powered exoplanet classification using NASA data",
 version="2.0.0"
)

app.add_middleware(
 CORSMiddleware,
 allow_origins=["*"],
 allow_credentials=True,
 allow_methods=["*"],
 allow_headers=["*"],
)

MODELS_DIR = Path("nasa_models")
models = {}
scaler = None
imputer = None
label_encoder = None
metadata = None

class ExoplanetInput(BaseModel):

 orbitalPeriod: float
 stellarRadius: float
 planetRadius: float
 stellarMass: float
 equilibriumTemperature: float
 stellarTemperature: float
 stellarAge: float
 insolationFlux: float
 distanceToStarRadius: float
 rightAscension: float
 declination: float
 dispositionScore: float

class ClassificationResponse(BaseModel):

 classification: str
 rationale: str
 confidence: Optional[float] = None
 probabilities: Optional[Dict[str, float]] = None
 model_used: Optional[str] = None

class BatchClassificationResult(BaseModel):

 row_number: int
 classification: str
 confidence: float
 probabilities: Dict[str, float]
 input_data: Dict[str, float]

class BatchClassificationResponse(BaseModel):

 total_rows: int
 successful: int
 failed: int
 results: list[BatchClassificationResult]
 errors: list[Dict[str, str]]

class HealthResponse(BaseModel):

 status: str
 models_loaded: bool
 version: str

def load_models():

 global models, scaler, imputer, label_encoder, metadata

 try:
 logger.info(" Loading NASA AI models...")

 scaler_path = MODELS_DIR / 'nasa_scaler.pkl'
 imputer_path = MODELS_DIR / 'nasa_imputer.pkl'
 encoder_path = MODELS_DIR / 'nasa_label_encoder.pkl'
 metadata_path = MODELS_DIR / 'nasa_metadata.json'

 if all(path.exists() for path in [scaler_path, imputer_path, encoder_path, metadata_path]):
 scaler = joblib.load(scaler_path)
 imputer = joblib.load(imputer_path)
 label_encoder = joblib.load(encoder_path)

 with open(metadata_path, 'r') as f:
 metadata = json.load(f)

 logger.info(" Preprocessing components loaded")
 else:
 logger.warning(" Some preprocessing files not found")
 return False

 model_files = {
 'random_forest': 'nasa_random_forest_model.pkl',
 'extra_trees': 'nasa_extra_trees_model.pkl',
 'ensemble': 'nasa_ensemble_model.pkl'
 }

 for model_name, filename in model_files.items():
 model_path = MODELS_DIR / filename
 if model_path.exists():
 models[model_name] = joblib.load(model_path)
 logger.info(f" Loaded {model_name} model")

 if models:
 logger.info(f" Total models loaded: {len(models)}")
 return True

 return False

 except Exception as e:
 logger.error(f" Error loading models: {e}")
 return False

def preprocess_input(data: ExoplanetInput) -> np.ndarray:

 feature_mapping = {
 'koi_period': data.orbitalPeriod,
 'koi_prad': data.planetRadius,
 'koi_teq': data.equilibriumTemperature,
 'koi_insol': data.insolationFlux,
 'koi_srad': data.stellarRadius,
 'koi_smass': data.stellarMass,
 'koi_steff': data.stellarTemperature,
 'koi_sage': data.stellarAge,
 'koi_dor': data.distanceToStarRadius,
 'ra': data.rightAscension,
 'dec': data.declination,
 'koi_score': data.dispositionScore
 }

 df = pd.DataFrame([feature_mapping])

 required_features = metadata['feature_names']
 for feature in required_features:
 if feature not in df.columns:
 df[feature] = 0

 df = df[required_features]

 X_imputed = imputer.transform(df)
 df_imputed = pd.DataFrame(X_imputed, columns=required_features, index=df.index)

 df_engineered = engineer_features(df_imputed)

 X_scaled = scaler.transform(df_engineered)

 return X_scaled

def engineer_features(X: pd.DataFrame) -> pd.DataFrame:

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

@app.on_event("startup")
async def startup_event():

 logger.info(" Starting NASA Exoplanet Classifier API...")
 success = load_models()
 if success:
 logger.info(" API ready to classify exoplanets!")
 else:
 logger.error(" Failed to load models. API may not function correctly.")

@app.get("/api/health", response_model=HealthResponse)
async def health_check():

 return {
 "status": "healthy" if models else "unhealthy",
 "models_loaded": bool(models),
 "version": "2.0.0"
 }

@app.post("/api/classify", response_model=ClassificationResponse)
async def classify_exoplanet(data: ExoplanetInput):

 try:
 if not models:
 raise HTTPException(status_code=503, detail="Models not loaded")

 logger.info(" Classifying exoplanet...")

 X = preprocess_input(data)

 model = models.get('ensemble', list(models.values())[0])

 prediction_encoded = model.predict(X)[0]
 probabilities = model.predict_proba(X)[0]

 prediction = label_encoder.inverse_transform([prediction_encoded])[0]

 prob_dict = {}
 for i, class_name in enumerate(label_encoder.classes_):
 prob_dict[class_name] = float(probabilities[i])

 logger.info(f" Classification: {prediction} (confidence: {max(probabilities):.1%})")

 confidence_pct = float(max(probabilities)) * 100
 if prediction == "CONFIRMED":
 rationale = f"High confidence ({confidence_pct:.1f}%) confirmed exoplanet based on NASA AI analysis of orbital and stellar parameters."
 elif prediction == "CANDIDATE":
 rationale = f"Moderate confidence ({confidence_pct:.1f}%) exoplanet candidate requiring further observation and verification."
 else:
 rationale = f"High confidence ({confidence_pct:.1f}%) this is likely a false positive, not a true exoplanet detection."

 return {
 "classification": prediction,
 "rationale": rationale,
 "confidence": float(max(probabilities)),
 "probabilities": prob_dict,
 "model_used": "Ensemble AI Model"
 }

 except Exception as e:
 logger.error(f" Classification error: {e}")
 raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/classify-batch", response_model=BatchClassificationResponse)
async def classify_batch(file: UploadFile = File(...)):

 if not models:
 raise HTTPException(status_code=503, detail="Models not loaded")

 if not file.filename.endswith('.csv'):
 raise HTTPException(status_code=400, detail="Only CSV files are supported")

 try:

 contents = await file.read()
 csv_text = contents.decode('utf-8')

 from io import StringIO
 df = pd.read_csv(StringIO(csv_text))

 logger.info(f" Processing batch file: {file.filename} ({len(df)} rows)")

 results = []
 errors = []
 successful = 0
 failed = 0

 for idx, row in df.iterrows():
 try:

 row_data = {
 'orbitalPeriod': float(row.get('orbitalPeriod', row.get('koi_period', 0))),
 'stellarRadius': float(row.get('stellarRadius', row.get('koi_srad', 0))),
 'planetRadius': float(row.get('planetRadius', row.get('koi_prad', 0))),
 'stellarMass': float(row.get('stellarMass', row.get('koi_smass', 0))),
 'equilibriumTemperature': float(row.get('equilibriumTemperature', row.get('koi_teq', 0))),
 'stellarTemperature': float(row.get('stellarTemperature', row.get('koi_steff', 0))),
 'stellarAge': float(row.get('stellarAge', row.get('koi_sage', 0))),
 'insolationFlux': float(row.get('insolationFlux', row.get('koi_insol', 0))),
 'distanceToStarRadius': float(row.get('distanceToStarRadius', row.get('koi_dor', 0))),
 'rightAscension': float(row.get('rightAscension', row.get('ra', 0))),
 'declination': float(row.get('declination', row.get('dec', 0))),
 'dispositionScore': float(row.get('dispositionScore', row.get('koi_score', 0.5)))
 }

 exoplanet = ExoplanetInput(**row_data)

 X = preprocess_input(exoplanet)
 model = models.get('ensemble', list(models.values())[0])

 prediction_encoded = model.predict(X)[0]
 probabilities = model.predict_proba(X)[0]
 prediction = label_encoder.inverse_transform([prediction_encoded])[0]

 prob_dict = {}
 for i, class_name in enumerate(label_encoder.classes_):
 prob_dict[class_name] = float(probabilities[i])

 results.append(BatchClassificationResult(
 row_number=int(idx) + 1,
 classification=prediction,
 confidence=float(max(probabilities)),
 probabilities=prob_dict,
 input_data=row_data
 ))

 successful += 1

 except Exception as row_error:
 failed += 1
 errors.append({
 'row_number': int(idx) + 1,
 'error': str(row_error)
 })
 logger.warning(f" Row {idx + 1} failed: {row_error}")

 logger.info(f" Batch processing complete: {successful} successful, {failed} failed")

 return BatchClassificationResponse(
 total_rows=len(df),
 successful=successful,
 failed=failed,
 results=results,
 errors=errors
 )

 except Exception as e:
 logger.error(f" Batch classification error: {e}")
 raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.get("/api/models")
async def list_models():

 if not models:
 return {"models": [], "status": "No models loaded"}

 return {
 "models": list(models.keys()),
 "status": "ready",
 "metadata": metadata
 }

@app.get("/")
async def serve_frontend():

 frontend_path = Path("frontend/dist/index.html")
 if frontend_path.exists():
 return FileResponse(frontend_path)
 else:
 return {
 "message": "NASA Exoplanet Classifier API",
 "status": "running",
 "docs": "/docs",
 "frontend": "Build the React app first: cd frontend && npm run build"
 }

try:
 app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")
 logger.info(" Frontend static files mounted")
except Exception as e:
 logger.warning(f" Could not mount frontend static files: {e}")
 logger.info(" Build the frontend first: cd frontend && npm install && npm run build")

if __name__ == "__main__":
 import uvicorn
 import os

 port = int(os.getenv("PORT", 7860))
 uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")