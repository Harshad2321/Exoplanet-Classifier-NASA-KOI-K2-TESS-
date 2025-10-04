# 🚀 Quick Reference Guide - NASA Exoplanet Classifier

## ⚡ **Quick Start Commands**

### **Start Everything (Recommended)**
```powershell
.\start_app.ps1
```
**Then choose:** Option 1 (Production) or Option 2 (Development)

### **Backend Only**
```bash
python backend_api.py
```

### **Frontend Development**
```bash
cd frontend
npm run dev
```

---

## 🌐 **Important URLs**

| What | URL | Use For |
|------|-----|---------|
| **Main App** | http://localhost:8000 | Classify exoplanets |
| **API Docs** | http://localhost:8000/docs | Test API endpoints |
| **Health Check** | http://localhost:8000/api/health | Verify backend status |
| **Dev Server** | http://localhost:5173 | Frontend development (dev mode) |

---

## 🎯 **Common Tasks**

### **Rebuild Frontend**
```bash
cd frontend
npm run build
```

### **Test API Endpoint**
```powershell
curl -X POST http://localhost:8000/api/classify -H "Content-Type: application/json" -d "{\"orbitalPeriod\":365.25,\"stellarRadius\":1.0,\"planetRadius\":1.0,\"stellarMass\":1.0,\"equilibriumTemperature\":288.0,\"stellarTemperature\":5778.0,\"stellarAge\":4.5,\"insolationFlux\":1.0,\"distanceStarRadius\":215.0,\"rightAscension\":290.0,\"declination\":42.0,\"dispositionScore\":0.8}"
```

### **Stop Backend**
Press `Ctrl+C` in the terminal running `python backend_api.py`

### **Kill Port 8000**
```powershell
# Find process
netstat -ano | findstr :8000

# Kill it (replace <PID> with actual number)
taskkill /F /PID <PID>
```

---

## 🔧 **Architecture**

```
User Browser
    ↓
React Frontend (Port 8000)
    ↓ POST /api/classify
FastAPI Backend (Port 8000)
    ↓
NASA AI Models
    ↓
Classification Result
```

---

## 📝 **Input Parameters**

All 12 required fields:
1. `orbitalPeriod` - days
2. `stellarRadius` - solar radii
3. `planetRadius` - earth radii
4. `stellarMass` - solar masses
5. `equilibriumTemperature` - Kelvin
6. `stellarTemperature` - Kelvin
7. `stellarAge` - billion years
8. `insolationFlux` - earth flux
9. `distanceStarRadius` - ratio
10. `rightAscension` - degrees
11. `declination` - degrees
12. `dispositionScore` - 0 to 1

---

## 🎯 **Output Classifications**

- **CONFIRMED** ✅ - Verified exoplanet
- **CANDIDATE** 🔍 - Potential exoplanet
- **FALSE POSITIVE** ❌ - Not an exoplanet

---

## 🐛 **Quick Troubleshooting**

| Problem | Solution |
|---------|----------|
| Frontend blank | `cd frontend && npm run build` |
| API 503 error | Train models: `python nasa_clean_model.py` |
| Port in use | Kill process using port 8000 |
| Models not found | Check `nasa_models/` folder exists |
| Import errors | `pip install -r requirements.txt` |

---

## 📚 **Documentation Files**

- `README_FULLSTACK.md` - Complete guide
- `MIGRATION_SUMMARY.md` - What changed
- `SMART_AI_GUIDE.md` - AI model selection
- `QUICK_REFERENCE.md` - This file

---

## 🚀 **NASA Space Apps Challenge 2025**

**Ready to demonstrate! ✨**

Application: http://localhost:8000  
Status: ✅ Operational  
Models: 3 loaded (Random Forest, Extra Trees, Ensemble)  
Accuracy: 69.2% (Ensemble)