# 🚀 Hugging Face Spaces Deployment - Quick Start

## ✅ Files Created for Deployment

1. **`Dockerfile`** - Docker configuration for containerization
2. **`.dockerignore`** - Excludes unnecessary files from Docker build
3. **`HUGGINGFACE_DEPLOYMENT.md`** - Complete deployment guide
4. **`deploy-to-huggingface.ps1`** - Automated deployment script

## 🎯 Two Ways to Deploy

### Option 1: Automated Script (Easiest!) ⚡

Just run this command in PowerShell:

```powershell
.\deploy-to-huggingface.ps1
```

The script will:
- ✅ Check if Git is installed
- ✅ Ask for your Hugging Face username
- ✅ Clone your Space repository
- ✅ Copy all necessary files
- ✅ Create Hugging Face README
- ✅ Push to Hugging Face
- ✅ Give you the live URL

### Option 2: Manual Deployment 📝

Follow the detailed guide in `HUGGINGFACE_DEPLOYMENT.md`

## 📋 Before You Start

### 1. Create Hugging Face Account
- Visit: https://huggingface.co/join
- Sign up (it's free!)

### 2. Create a New Space
- Go to: https://huggingface.co/spaces
- Click "Create new Space"
- **Important Settings:**
  - SDK: Choose **"Docker"** ⚠️
  - Space hardware: CPU basic (free)
  - Visibility: Public

### 3. Install Git (if not installed)
- Download: https://git-scm.com/download/win
- Install with default settings

## 🚀 Quick Deployment Steps

```powershell
# Step 1: Run the deployment script
.\deploy-to-huggingface.ps1

# Step 2: Enter your Hugging Face username when prompted
# Example: Harshad2321

# Step 3: Enter your Space name
# Example: nasa-exoplanet-classifier

# Step 4: Wait for upload to complete
# Takes about 2-5 minutes depending on internet speed

# Step 5: Wait for Docker build on Hugging Face
# Takes about 5-10 minutes
# Visit your Space URL to watch the build progress
```

## 🔍 What Gets Deployed?

✅ **Backend:**
- FastAPI server (Python)
- 3 trained AI models
- Classification API endpoints

✅ **Frontend:**
- React TypeScript app
- Interactive UI
- Batch CSV upload
- Results visualization

✅ **Models:**
- Random Forest (68.5%)
- Extra Trees (68.2%)
- Ensemble (69.2%)

✅ **Data Processing:**
- StandardScaler
- SimpleImputer
- LabelEncoder

## 📊 After Deployment

### Your Space URL will be:
```
https://huggingface.co/spaces/YOUR_USERNAME/nasa-exoplanet-classifier
```

### Features Available:
- 🌐 Public URL (share with anyone)
- 📱 Mobile-responsive UI
- 🔒 HTTPS enabled automatically
- 📊 Usage analytics
- 📝 Build logs
- 💾 Persistent storage

## ⚙️ Configuration Details

### Port
- **Local:** Port 8000
- **Hugging Face:** Port 7860 (automatically configured)

### Resources (Free Tier)
- **CPU:** 2 vCPU
- **RAM:** 16 GB
- **Storage:** 50 GB
- **Sleep:** After 48 hours of inactivity

### Upgrading (Optional)
- **CPU Upgrade:** $0.60/hour (better performance)
- **GPU T4:** $0.60/hour (ML acceleration)
- **GPU A10G:** $3.15/hour (fastest)

## 🐛 Troubleshooting

### "Failed to clone Space"
**Solution:**
1. Make sure you created the Space on Hugging Face first
2. Check your internet connection
3. Verify Git credentials

### "Build fails on Hugging Face"
**Solution:**
1. Check the Logs tab in your Space
2. Ensure all model files are committed
3. Verify `requirements.txt` is complete

### "Port not accessible"
**Solution:**
- Port is automatically configured (no action needed)
- Backend uses environment variable PORT (defaults to 7860)

### Git Credentials Issues
**Solution:**
```powershell
# Configure Git credential helper
git config --global credential.helper wincred

# Or use Git Credential Manager
# Download from: https://github.com/GitCredentialManager/git-credential-manager
```

## 📚 Additional Resources

- **Hugging Face Docs:** https://huggingface.co/docs/hub/spaces
- **Docker Docs:** https://docs.docker.com
- **FastAPI Docs:** https://fastapi.tiangolo.com
- **React Docs:** https://react.dev

## 🆘 Need Help?

1. **Read the full guide:** `HUGGINGFACE_DEPLOYMENT.md`
2. **Hugging Face Discord:** https://hf.co/join/discord
3. **GitHub Issues:** Open an issue on your repo
4. **Community Forum:** https://discuss.huggingface.co

## 🎉 Success Checklist

After deployment, verify:
- [ ] Space builds successfully (check Logs tab)
- [ ] App loads at your Space URL
- [ ] Single planet classification works
- [ ] Batch CSV upload works
- [ ] Models return predictions
- [ ] Footer shows your contact info
- [ ] No errors in logs

## 💡 Pro Tips

1. **Keep models updated:** Replace files in `nasa_models/` folder
2. **Monitor logs:** Check regularly for errors
3. **Version control:** Use git tags for releases
4. **Community:** Share your Space on social media
5. **Embed:** Add Space to your portfolio/website

## 🔄 Update Deployed App

To update your app after changes:

```powershell
# 1. Make changes to your code
# 2. Run deployment script again
.\deploy-to-huggingface.ps1

# Or manually:
cd ..\nasa-exoplanet-classifier-deploy
git add .
git commit -m "Update: [describe your changes]"
git push
```

Hugging Face will automatically rebuild and redeploy! 🚀

---

## 🌟 Next Steps

1. ✅ Run `.\deploy-to-huggingface.ps1`
2. ✅ Wait for build to complete
3. ✅ Share your Space URL
4. ✅ Add to your portfolio
5. ✅ Submit to NASA Space Apps Challenge!

**Good luck with your deployment! 🚀**

---

**Built by:** Parth Koshti & Harshad Agrawal
**GitHub:** https://github.com/Harshad2321/Exoplanet-Classifier-NASA-KOI-K2-TESS-
