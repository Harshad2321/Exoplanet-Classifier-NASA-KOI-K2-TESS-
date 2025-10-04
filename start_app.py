#!/usr/bin/env python3
"""
🚀 Startup Script for NASA Exoplanet Classifier
Starts both React Frontend and FastAPI Backend
"""

import subprocess
import time
import sys
import os
from pathlib import Path

def print_banner():
    """Print startup banner"""
    print("=" * 70)
    print("🌌 NASA EXOPLANET CLASSIFIER - FULL STACK APPLICATION")
    print("=" * 70)
    print("🚀 NASA Space Apps Challenge 2025")
    print("📡 React + TypeScript Frontend")
    print("🤖 FastAPI + Smart AI Backend")
    print("=" * 70)
    print()

def check_frontend_built():
    """Check if React frontend is built"""
    dist_path = Path("frontend/dist")
    if not dist_path.exists():
        print("⚠️  Frontend not built yet!")
        print("📦 Building React frontend...")
        try:
            subprocess.run(["npm", "run", "build"], cwd="frontend", check=True, shell=True)
            print("✅ Frontend built successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to build frontend")
            print("💡 Run manually: cd frontend && npm install && npm run build")
            return False
    else:
        print("✅ Frontend already built")
        return True

def start_backend():
    """Start FastAPI backend"""
    print("\n🔧 Starting FastAPI Backend...")
    print("📍 Backend URL: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    
    try:
        # Start backend process
        backend_process = subprocess.Popen(
            [sys.executable, "backend_api.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Wait for backend to start
        print("⏳ Waiting for backend to initialize...")
        time.sleep(3)
        
        if backend_process.poll() is None:
            print("✅ Backend started successfully!")
            return backend_process
        else:
            print("❌ Backend failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def start_frontend_dev():
    """Start React frontend in development mode"""
    print("\n🎨 Starting React Frontend (Development Mode)...")
    print("📍 Frontend URL: http://localhost:5173")
    
    try:
        # Start frontend dev server
        frontend_process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd="frontend",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            shell=True
        )
        
        # Wait for frontend to start
        print("⏳ Waiting for frontend to initialize...")
        time.sleep(3)
        
        if frontend_process.poll() is None:
            print("✅ Frontend started successfully!")
            return frontend_process
        else:
            print("❌ Frontend failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        return None

def main():
    """Main startup function"""
    print_banner()
    
    # Check if we're in the right directory
    if not Path("backend_api.py").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Check frontend build
    if not check_frontend_built():
        print("❌ Cannot start without frontend build")
        sys.exit(1)
    
    # Start backend
    backend = start_backend()
    if not backend:
        print("❌ Failed to start backend")
        sys.exit(1)
    
    # Option to start frontend in dev mode or serve from backend
    print("\n" + "=" * 70)
    print("🎯 CHOOSE FRONTEND MODE:")
    print("=" * 70)
    print("1. Production Mode - Serve frontend from FastAPI backend (Port 8000)")
    print("2. Development Mode - Run Vite dev server (Port 5173) + Backend (Port 8000)")
    print("=" * 70)
    
    choice = input("\nEnter choice (1 or 2) [default: 1]: ").strip() or "1"
    
    if choice == "2":
        # Start frontend dev server
        frontend = start_frontend_dev()
        if not frontend:
            print("❌ Failed to start frontend")
            backend.terminate()
            sys.exit(1)
        
        print("\n" + "=" * 70)
        print("🎉 NASA EXOPLANET CLASSIFIER IS RUNNING!")
        print("=" * 70)
        print("🎨 Frontend (Dev): http://localhost:5173")
        print("🔧 Backend API: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("=" * 70)
        print("\n🛑 Press Ctrl+C to stop all servers")
        print("=" * 70)
        
        try:
            # Keep running until interrupted
            backend.wait()
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down...")
            frontend.terminate()
            backend.terminate()
            print("✅ All servers stopped")
    else:
        # Production mode - serve from backend
        print("\n" + "=" * 70)
        print("🎉 NASA EXOPLANET CLASSIFIER IS RUNNING!")
        print("=" * 70)
        print("🌐 Application: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("🤖 Smart AI: Automatic model selection enabled")
        print("=" * 70)
        print("\n🛑 Press Ctrl+C to stop the server")
        print("=" * 70)
        
        try:
            # Keep running until interrupted
            backend.wait()
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down...")
            backend.terminate()
            print("✅ Server stopped")

if __name__ == "__main__":
    main()