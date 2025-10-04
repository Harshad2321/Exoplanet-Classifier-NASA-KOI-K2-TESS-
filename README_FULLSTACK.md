# 🚀 NASA Exoplanet Classifier - Full Stack Application

**NASA Space Apps Challenge 2025 - "A World Away: Hunting for Exoplanets with AI"**

Modern full-stack web application for classifying exoplanets using AI, featuring a React + TypeScript frontend and FastAPI + Smart AI backend.

![NASA](https://img.shields.io/badge/NASA-Space%20Apps%202025-red.svg)
![React](https://img.shields.io/badge/React-19.1-blue.svg)
![TypeScript](https://img.shields.io/badge/TypeScript-5.8-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)
![Python](https://img.shields.io/badge/Python-3.12-yellow.svg)

---

## 🌟 **Architecture Overview**

### **Frontend** (React + TypeScript + Vite)
- 🎨 Modern, responsive UI with animated components
- ⚡ Fast development with Vite
- 🎯 TypeScript for type safety
- 🌌 Beautiful space-themed design with particles
- 📊 Real-time exoplanet classification

### **Backend** (FastAPI + Python)
- 🤖 Smart AI with automatic model selection
- 🚀 RESTful API with FastAPI
- 🧠 Multiple ML models (Random Forest, Extra Trees, Ensemble)
- 📈 Real-time predictions with < 500ms response time
- 📚 Interactive API documentation (Swagger/OpenAPI)

---

## 🚀 **Quick Start**

### **Option 1: Automated Startup (Recommended)**

**Windows PowerShell:**
```powershell
.\start_app.ps1
```

**Python (Cross-platform):**
```bash
python start_app.py
```

### **Option 2: Manual Setup**

#### **1. Install Backend Dependencies**
```bash
pip install -r requirements.txt
```

#### **2. Install Frontend Dependencies**
```bash
cd frontend
npm install
```

#### **3. Build Frontend**
```bash
npm run build
```

#### **4. Start Backend**
```bash
cd ..
python backend_api.py
```

**Access the application at:** http://localhost:8000

---

## 📂 **Project Structure**

```
Exoplanet-Classifier-NASA-KOI-K2-TESS-/
│
├── frontend/                      # React + TypeScript Frontend
│   ├── components/               # React components
│   │   ├── Header.tsx           # Navigation header
│   │   ├── Footer.tsx           # Footer component
│   │   ├── NumericInput.tsx     # Input fields
│   │   ├── ClassificationResult.tsx  # Results display
│   │   ├── Particles.tsx        # Background animations
│   │   └── MagicBento.tsx       # Card grid layout
│   ├── App.tsx                  # Main application
│   ├── index.tsx                # Entry point
│   ├── types.ts                 # TypeScript definitions
│   ├── constants.ts             # Configuration
│   ├── package.json             # Dependencies
│   ├── tsconfig.json            # TypeScript config
│   ├── vite.config.ts           # Vite configuration
│   └── dist/                    # Built production files
│
├── backend_api.py               # FastAPI backend server
├── nasa_smart_classifier.py    # Smart AI model selection
├── nasa_clean_model.py          # Model training script
│
├── nasa_models/                 # Trained AI models
│   ├── nasa_random_forest_model.pkl
│   ├── nasa_extra_trees_model.pkl
│   ├── nasa_ensemble_model.pkl
│   ├── nasa_scaler.pkl
│   ├── nasa_imputer.pkl
│   ├── nasa_label_encoder.pkl
│   └── nasa_metadata.json
│
├── start_app.py                 # Python startup script
├── start_app.ps1                # PowerShell startup script
├── requirements.txt             # Python dependencies
└── README_FULLSTACK.md          # This file
```

---

## 🌐 **API Endpoints**

### **Frontend**
- `GET /` - Serve React application

### **Classification API**
- `POST /api/classify` - Classify exoplanet
  ```json
  {
    "orbitalPeriod": 365.25,
    "stellarRadius": 1.0,
    "planetRadius": 1.0,
    "stellarMass": 1.0,
    "equilibriumTemperature": 288.0,
    "stellarTemperature": 5778.0,
    "stellarAge": 4.5,
    "insolationFlux": 1.0,
    "distanceStarRadius": 215.0,
    "rightAscension": 290.0,
    "declination": 42.0,
    "dispositionScore": 0.8
  }
  ```

### **Health & Info**
- `GET /api/health` - Backend health check
- `GET /api/models` - List available AI models
- `GET /docs` - Interactive API documentation

---

## 🤖 **Smart AI Features**

### **Automatic Model Selection**
The backend intelligently selects the optimal AI model based on data characteristics:

- **Random Forest**: Stable predictions, handles missing data
- **Extra Trees**: Noise resistant, fast training
- **Ensemble**: Maximum accuracy, combines model strengths

### **Data Analysis**
Analyzes 7 key characteristics:
- Dataset size
- Missing data ratio
- Noise level
- Class balance
- Outlier detection
- Feature correlation
- Feature types

---

## 🎨 **Frontend Features**

### **Interactive Interface**
- ✨ Animated particle background
- 🎯 Real-time input validation
- 📊 Beautiful result visualization
- 🌈 Color-coded classifications
- 📱 Fully responsive design

### **Input Fields**
12 astronomical parameters:
1. Orbital Period (days)
2. Stellar Radius (Solar radii)
3. Planet Radius (Earth radii)
4. Stellar Mass (Solar masses)
5. Equilibrium Temperature (K)
6. Stellar Temperature (K)
7. Stellar Age (Gyr)
8. Insolation Flux (Earth flux)
9. Distance/Star Radius Ratio
10. Right Ascension (degrees)
11. Declination (degrees)
12. Disposition Score (0-1)

### **Classification Results**
- **CONFIRMED**: Verified exoplanet ✅
- **CANDIDATE**: Potential exoplanet 🔍
- **FALSE POSITIVE**: Not an exoplanet ❌

---

## 🔧 **Development**

### **Frontend Development Mode**
Run Vite dev server for hot reload:
```bash
cd frontend
npm run dev
```
Frontend: http://localhost:5173  
Backend: http://localhost:8000 (start separately)

### **Backend Development**
```bash
python backend_api.py
```
API: http://localhost:8000  
Docs: http://localhost:8000/docs

### **Build for Production**
```bash
cd frontend
npm run build
```
Serves from backend at http://localhost:8000

---

## 📊 **Model Performance**

| Model | Accuracy | Speed | Best For |
|-------|----------|-------|----------|
| **Ensemble** | **69.2%** | Fast | General use |
| Random Forest | 68.5% | Fast | Stable predictions |
| Extra Trees | 68.2% | Very Fast | Noisy data |

---

## 🛠️ **Technologies**

### **Frontend Stack**
- ⚛️ React 19.1 - UI framework
- 📘 TypeScript 5.8 - Type safety
- ⚡ Vite 6.2 - Build tool
- 🎨 GSAP 3.12 - Animations
- 🌌 OGL - WebGL graphics
- 🤖 Google Gemini API integration

### **Backend Stack**
- 🚀 FastAPI - Modern Python web framework
- 🤖 scikit-learn - Machine learning
- 📊 pandas & numpy - Data processing
- 🔧 Uvicorn - ASGI server
- 📝 Pydantic - Data validation

---

## 🌟 **Key Features**

### **For Users**
- ✅ Instant exoplanet classification
- ✅ No ML expertise required
- ✅ Beautiful, intuitive interface
- ✅ Real-time predictions
- ✅ Detailed confidence scores

### **For Developers**
- ✅ Modern full-stack architecture
- ✅ Type-safe frontend with TypeScript
- ✅ RESTful API design
- ✅ Comprehensive documentation
- ✅ Easy to extend and customize

### **For NASA Space Apps Challenge**
- ✅ Production-ready application
- ✅ Professional UI/UX design
- ✅ Advanced AI capabilities
- ✅ Real-world performance
- ✅ Scalable architecture

---

## 📝 **API Usage Example**

### **Python**
```python
import requests

data = {
    "orbitalPeriod": 365.25,
    "stellarRadius": 1.0,
    "planetRadius": 1.0,
    "stellarMass": 1.0,
    "equilibriumTemperature": 288.0,
    "stellarTemperature": 5778.0,
    "stellarAge": 4.5,
    "insolationFlux": 1.0,
    "distanceStarRadius": 215.0,
    "rightAscension": 290.0,
    "declination": 42.0,
    "dispositionScore": 0.8
}

response = requests.post("http://localhost:8000/api/classify", json=data)
result = response.json()

print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### **JavaScript/TypeScript**
```typescript
const data = {
  orbitalPeriod: 365.25,
  stellarRadius: 1.0,
  // ... other parameters
};

const response = await fetch('/api/classify', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(data)
});

const result = await response.json();
console.log(`Classification: ${result.classification}`);
console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);
```

---

## 🐛 **Troubleshooting**

### **Frontend not loading?**
```bash
cd frontend
npm run build
```

### **Backend errors?**
```bash
pip install -r requirements.txt
```

### **Models not found?**
Train models first:
```bash
python nasa_clean_model.py
```

### **Port 8000 already in use?**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /F /PID <PID>

# Linux/Mac
lsof -i :8000
kill -9 <PID>
```

---

## 📄 **License**

MIT License - NASA Space Apps Challenge 2025

---

## 🌌 **NASA Space Apps Challenge 2025**

**Challenge:** "A World Away: Hunting for Exoplanets with AI"

**Mission:** Develop AI systems to hunt for exoplanets using NASA's data from Kepler, K2, and TESS missions.

**Our Solution:**
- 🤖 Advanced AI with automatic model selection
- 🎨 Modern, professional web interface
- 📊 Real-time classification system
- 🚀 Production-ready full-stack application
- 🌟 Educational and accessible to all

---

## 🙏 **Acknowledgments**

- NASA Exoplanet Archive
- Kepler, K2, and TESS Missions
- NASA Space Apps Challenge organizers
- Open-source community

---

**🌟 Built for NASA Space Apps Challenge 2025 - Ready to discover new worlds! 🪐**