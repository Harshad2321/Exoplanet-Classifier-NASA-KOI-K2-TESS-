# Quick Deployment Script for Hugging Face Spaces
# NASA Exoplanet Classifier

Write-Host "NASA Exoplanet Classifier - Hugging Face Deployment" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan

# Check if Git is installed
try {
    git --version | Out-Null
    Write-Host "[OK] Git is installed" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Git is not installed. Please install Git first." -ForegroundColor Red
    Write-Host "Download from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

# Get Hugging Face username
Write-Host ""
$hfUsername = Read-Host "Enter your Hugging Face username"

# Get Space name
Write-Host ""
$spaceName = Read-Host "Enter your Space name (e.g., nasa-exoplanet-classifier)"

# Construct Space URL
$spaceUrl = "https://huggingface.co/spaces/$hfUsername/$spaceName"

Write-Host ""
Write-Host "Deployment Summary:" -ForegroundColor Yellow
Write-Host "   Username: $hfUsername"
Write-Host "   Space Name: $spaceName"
Write-Host "   Space URL: $spaceUrl"

Write-Host ""
$confirm = Read-Host "Continue with deployment? (y/n)"

if ($confirm -ne "y") {
    Write-Host "[CANCELLED] Deployment cancelled" -ForegroundColor Red
    exit 0
}

# Create temporary directory
$tempDir = "..\nasa-exoplanet-classifier-deploy"
Write-Host ""
Write-Host "Creating temporary directory..." -ForegroundColor Cyan

if (Test-Path $tempDir) {
    Write-Host "   Cleaning existing directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $tempDir
}

# Clone the space
Write-Host ""
Write-Host "Cloning Hugging Face Space..." -ForegroundColor Cyan
git clone $spaceUrl $tempDir 2>&1 | Out-Null

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Failed to clone Space. Please make sure:" -ForegroundColor Red
    Write-Host "   1. You created the Space on Hugging Face" -ForegroundColor Yellow
    Write-Host "   2. You're logged in with Git" -ForegroundColor Yellow
    Write-Host "   3. The Space URL is correct: $spaceUrl" -ForegroundColor Yellow
    exit 1
}

# Copy files
Write-Host ""
Write-Host "Copying project files..." -ForegroundColor Cyan

$filesToCopy = @(
    "frontend",
    "backend_api.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "nasa_models",
    "data",
    "README.md"
)

foreach ($file in $filesToCopy) {
    if (Test-Path $file) {
        Write-Host "   [OK] Copying $file" -ForegroundColor Green
        if (Test-Path $file -PathType Container) {
            Copy-Item -Recurse -Force $file $tempDir\
        } else {
            Copy-Item -Force $file $tempDir\
        }
    } else {
        Write-Host "   [WARN] $file not found (skipping)" -ForegroundColor Yellow
    }
}

# Create Hugging Face specific README
Write-Host ""
Write-Host "Creating Hugging Face README..." -ForegroundColor Cyan

# Create README content as array of lines
$readmeLines = @(
    "---"
    "title: NASA Exoplanet Classifier"
    "emoji: 🌍"
    "colorFrom: indigo"
    "colorTo: purple"
    "sdk: docker"
    "pinned: false"
    "license: mit"
    "---"
    ""
    "# NASA Exoplanet Classifier"
    ""
    "AI-powered exoplanet classification using NASA KOI, K2, and TESS datasets."
    ""
    "## Features"
    ""
    "- Smart AI Selection - Automatically chooses best model"
    "- Single Classification - Analyze individual exoplanets"
    "- Batch Processing - Upload CSV for multiple predictions"
    "- Interactive UI - Beautiful React interface"
    "- Real-time Results - Instant classification"
    ""
    "## How to Use"
    ""
    "1. Single Planet - Upload JSON file or paste data"
    "2. Batch Upload - Upload CSV file with multiple planets"
    "3. View Results - See classification and confidence"
    "4. Download - Export batch results as CSV"
    ""
    "## AI Models"
    ""
    "- Random Forest: 68.5% accuracy"
    "- Extra Trees: 68.2% accuracy"
    "- Ensemble: 69.2% accuracy (best)"
    ""
    "## Team"
    ""
    "Built by Parth Koshti and Harshad Agrawal"
    ""
    "- Parth: Parthkoshti2606@gmail.com"
    "- Harshad: Harshad.agrawal2005@gmail.com"
    "- GitHub: https://github.com/Harshad2321/Exoplanet-Classifier-NASA-KOI-K2-TESS-"
    ""
    "## Data Sources"
    ""
    "- NASA Kepler Objects of Interest (KOI)"
    "- NASA K2 Mission"
    "- NASA TESS Mission"
)

$readmeLines | Out-File -FilePath "$tempDir\README.md" -Encoding UTF8

# Navigate to temp directory
Set-Location $tempDir

# Add all files
Write-Host ""
Write-Host "Adding files to Git..." -ForegroundColor Cyan
git add .

# Commit
Write-Host ""
Write-Host "Creating commit..." -ForegroundColor Cyan
git commit -m "Initial deployment - FastAPI backend with 3 AI models, React frontend, Docker containerized"

# Push to Hugging Face
Write-Host ""
Write-Host "Pushing to Hugging Face..." -ForegroundColor Cyan
git push

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Green
    Write-Host "[SUCCESS] DEPLOYMENT SUCCESSFUL!" -ForegroundColor Green
    Write-Host ("=" * 60) -ForegroundColor Green
    Write-Host ""
    Write-Host "Your Space URL: $spaceUrl" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Yellow
    Write-Host "   1. Wait 5-10 minutes for Docker build to complete"
    Write-Host "   2. Visit your Space URL to see the build progress"
    Write-Host "   3. Once built, your app will be live!"
    Write-Host ""
    Write-Host "Monitor your Space:" -ForegroundColor Yellow
    Write-Host "   - View logs: $spaceUrl (Logs tab)"
    Write-Host "   - Check metrics: $spaceUrl (Settings)"
    Write-Host "   - Update anytime: Push to this repo again"
    Write-Host ""
    Write-Host "Happy Classifying!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[ERROR] Push failed. Please check your credentials." -ForegroundColor Red
    Write-Host "Try: git config --global credential.helper wincred" -ForegroundColor Yellow
}

# Go back to original directory
Set-Location ..

Write-Host ""
Write-Host "Cleanup: Remove temporary directory? (y/n)" -ForegroundColor Yellow
$cleanup = Read-Host

if ($cleanup -eq "y") {
    Remove-Item -Recurse -Force $tempDir
    Write-Host "[OK] Cleaned up temporary files" -ForegroundColor Green
} else {
    Write-Host "Temporary files kept at: $tempDir" -ForegroundColor Cyan
}
