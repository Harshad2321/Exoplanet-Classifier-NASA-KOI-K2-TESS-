# 🚀 NASA Exoplanet Classifier Startup Script
# Starts both Frontend and Backend

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host "=" * 69 -ForegroundColor Cyan
Write-Host "🌌 NASA EXOPLANET CLASSIFIER - FULL STACK APPLICATION" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host "=" * 69 -ForegroundColor Cyan
Write-Host "🚀 NASA Space Apps Challenge 2025" -ForegroundColor Green
Write-Host "📡 React + TypeScript Frontend" -ForegroundColor Blue
Write-Host "🤖 FastAPI + Smart AI Backend" -ForegroundColor Magenta
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host "=" * 69 -ForegroundColor Cyan
Write-Host ""

# Check if in correct directory
if (-Not (Test-Path "backend_api.py")) {
    Write-Host "❌ Please run this script from the project root directory" -ForegroundColor Red
    exit 1
}

# Check if frontend is built
if (-Not (Test-Path "frontend\dist")) {
    Write-Host "📦 Building React frontend..." -ForegroundColor Yellow
    Set-Location frontend
    npm run build
    Set-Location ..
    Write-Host "✅ Frontend built successfully!" -ForegroundColor Green
} else {
    Write-Host "✅ Frontend already built" -ForegroundColor Green
}

Write-Host ""
Write-Host "🔧 Starting FastAPI Backend..." -ForegroundColor Cyan
Write-Host "📍 Backend URL: http://localhost:8000" -ForegroundColor Gray
Write-Host "📚 API Docs: http://localhost:8000/docs" -ForegroundColor Gray

# Start backend in new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python backend_api.py"

Write-Host ""
Write-Host "⏳ Waiting for backend to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host "=" * 69 -ForegroundColor Cyan
Write-Host "🎯 CHOOSE FRONTEND MODE:" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host "=" * 69 -ForegroundColor Cyan
Write-Host "1. Production Mode - Serve from FastAPI backend (Port 8000)" -ForegroundColor White
Write-Host "2. Development Mode - Vite dev server (Port 5173)" -ForegroundColor White
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host "=" * 69 -ForegroundColor Cyan

$choice = Read-Host "`nEnter choice (1 or 2) [default: 1]"
if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "1" }

if ($choice -eq "2") {
    Write-Host ""
    Write-Host "🎨 Starting React Frontend (Development Mode)..." -ForegroundColor Cyan
    Write-Host "📍 Frontend URL: http://localhost:5173" -ForegroundColor Gray
    
    # Start frontend in new window
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev"
    
    Start-Sleep -Seconds 3
    
    Write-Host ""
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host "=" * 69 -ForegroundColor Cyan
    Write-Host "🎉 NASA EXOPLANET CLASSIFIER IS RUNNING!" -ForegroundColor Green
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host "=" * 69 -ForegroundColor Cyan
    Write-Host "🎨 Frontend (Dev): http://localhost:5173" -ForegroundColor Blue
    Write-Host "🔧 Backend API: http://localhost:8000" -ForegroundColor Magenta
    Write-Host "📚 API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host "=" * 69 -ForegroundColor Cyan
    
    Write-Host ""
    Write-Host "✨ Both servers running in separate windows" -ForegroundColor Yellow
    Write-Host "🛑 Close the PowerShell windows to stop the servers" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host "=" * 69 -ForegroundColor Cyan
    Write-Host "🎉 NASA EXOPLANET CLASSIFIER IS RUNNING!" -ForegroundColor Green
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host "=" * 69 -ForegroundColor Cyan
    Write-Host "🌐 Application: http://localhost:8000" -ForegroundColor Blue
    Write-Host "📚 API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
    Write-Host "🤖 Smart AI: Automatic model selection enabled" -ForegroundColor Magenta
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host "=" * 69 -ForegroundColor Cyan
    
    Write-Host ""
    Write-Host "✨ Server running in separate window" -ForegroundColor Yellow
    Write-Host "🛑 Close the PowerShell window to stop the server" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🚀 Ready for NASA Space Apps Challenge 2025!" -ForegroundColor Green