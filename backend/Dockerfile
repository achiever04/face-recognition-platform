# backend/Dockerfile - improved and robust
FROM python:3.11-slim

# Install system packages needed by opencv, dlib, ffmpeg and general builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

# Copy requirements and install
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy application
COPY backend /app/backend

# Create required runtime directories
RUN mkdir -p /app/backend/data/uploads /app/backend/data/fl_weights /app/backend/data/models

VOLUME ["/app/backend/data"]

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Run using uvicorn; you can override CMD in production (e.g., use gunicorn+uvicorn workers)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
