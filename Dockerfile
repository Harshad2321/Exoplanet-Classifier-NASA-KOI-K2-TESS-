# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install Node.js (needed for building React frontend)
RUN apt-get update && apt-get install -y \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Install frontend dependencies (including dev dependencies for build)
WORKDIR /app/frontend
RUN npm ci

# Build React frontend
RUN npm run build

# Go back to app directory
WORKDIR /app

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Set environment variable for production
ENV PRODUCTION=true

# Run the FastAPI backend
CMD ["python", "backend_api.py"]
