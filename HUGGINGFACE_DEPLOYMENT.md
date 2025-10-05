# 🚀 NASA Exoplanet Classifier - Hugging Face Spaces Deployment

This guide will help you deploy the NASA Exoplanet Classifier to Hugging Face Spaces.

## 📋 Prerequisites

1. A Hugging Face account (free): https://huggingface.co/join
2. Git installed on your computer
3. Your code pushed to GitHub

## 🎯 Step-by-Step Deployment

### Step 1: Create a New Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in the details:
   - **Space name**: `nasa-exoplanet-classifier` (or your choice)
   - **License**: `MIT`
   - **Select the SDK**: Choose **Docker**
   - **Space hardware**: Choose **CPU basic** (free) or **CPU upgrade** (better performance)
   - **Visibility**: Public (recommended for free tier)
4. Click **"Create Space"**

### Step 2: Clone Your Space Repository

```bash
# Clone the new space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/nasa-exoplanet-classifier
cd nasa-exoplanet-classifier
```

### Step 3: Copy Your Project Files

Copy these files from your project to the space repository:

```bash
# Copy all necessary files (run from your project directory)
cp -r frontend/ ../nasa-exoplanet-classifier/
cp backend_api.py ../nasa-exoplanet-classifier/
cp requirements.txt ../nasa-exoplanet-classifier/
cp Dockerfile ../nasa-exoplanet-classifier/
cp -r nasa_models/ ../nasa-exoplanet-classifier/
cp -r data/ ../nasa-exoplanet-classifier/
cp README.md ../nasa-exoplanet-classifier/
```

**Windows PowerShell version:**
```powershell
# Copy all necessary files (run from your project directory)
Copy-Item -Recurse frontend/ ..\nasa-exoplanet-classifier\
Copy-Item backend_api.py ..\nasa-exoplanet-classifier\
Copy-Item requirements.txt ..\nasa-exoplanet-classifier\
Copy-Item Dockerfile ..\nasa-exoplanet-classifier\
Copy-Item -Recurse nasa_models\ ..\nasa-exoplanet-classifier\
Copy-Item -Recurse data\ ..\nasa-exoplanet-classifier\
Copy-Item README.md ..\nasa-exoplanet-classifier\
```

### Step 4: Create README.md for Hugging Face (Optional)

Create a `README.md` in the space with Space-specific metadata:

```markdown
---
title: NASA Exoplanet Classifier
emoji: 🌍
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# 🌍 NASA Exoplanet Classifier

AI-powered exoplanet classification using NASA's KOI, K2, and TESS datasets.

## Features

- 🤖 Smart AI model selection
- 📊 Single planet classification
- 📁 Batch CSV upload
- 🎨 Interactive React UI
- 📈 Real-time predictions

## How to Use

1. Upload exoplanet data (JSON or CSV)
2. View classification results
3. Download batch results

## Models

- Random Forest (68.5% accuracy)
- Extra Trees (68.2% accuracy)
- Ensemble (69.2% accuracy)
```

### Step 5: Push to Hugging Face

```bash
cd ../nasa-exoplanet-classifier

# Add all files
git add .

# Commit
git commit -m "Initial deployment of NASA Exoplanet Classifier"

# Push to Hugging Face
git push
```

### Step 6: Wait for Build

1. Go to your space URL: `https://huggingface.co/spaces/YOUR_USERNAME/nasa-exoplanet-classifier`
2. Wait for the Docker build to complete (5-10 minutes)
3. Once built, your app will be live! 🎉

## 🔧 Troubleshooting

### Build Fails

**Check the logs:**
1. Go to your Space
2. Click "Logs" tab
3. Look for error messages

**Common issues:**

1. **Missing dependencies**
   - Make sure `requirements.txt` is complete
   - Verify `package.json` in frontend folder

2. **Port issues**
   - Hugging Face uses port 7860 (already configured in `backend_api.py`)

3. **Model files missing**
   - Ensure `nasa_models/` folder is included
   - Check `.gitignore` doesn't exclude model files

### App Not Loading

1. Check if build completed successfully
2. Verify Dockerfile syntax
3. Check Space logs for runtime errors

## 📊 Monitoring Your Space

- **Metrics**: View usage statistics in Space settings
- **Logs**: Real-time logs available in Logs tab
- **Analytics**: Track visitors and usage

## 🚀 Upgrading to Better Hardware

Free tier limitations:
- CPU basic (2 vCPU, 16 GB RAM)
- Sleeps after 48 hours of inactivity

Paid upgrades available:
- **CPU upgrade**: $0.60/hour (better performance)
- **GPU T4**: $0.60/hour (ML acceleration)
- **GPU A10G**: $3.15/hour (fastest)

## 🔗 Sharing Your Space

Your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/nasa-exoplanet-classifier`

Share it:
- Embed in websites
- Share on social media
- Add to GitHub README

## 📝 Important Files

- `Dockerfile` - Docker configuration
- `backend_api.py` - FastAPI backend (port 7860)
- `requirements.txt` - Python dependencies
- `frontend/` - React application
- `nasa_models/` - Trained ML models

## 💡 Tips

1. **Keep models updated**: Replace model files to update predictions
2. **Monitor logs**: Check for errors regularly
3. **Version control**: Use git tags for releases
4. **Community**: Share in Hugging Face Discord

## 🆘 Need Help?

- Hugging Face Docs: https://huggingface.co/docs/hub/spaces
- Community: https://discuss.huggingface.co
- Discord: https://hf.co/join/discord

---

**Built by:** Parth Koshti & Harshad Agrawal
**Repository:** https://github.com/Harshad2321/Exoplanet-Classifier-NASA-KOI-K2-TESS-
