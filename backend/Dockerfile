# backend/Dockerfile - fixed for Python 3.10
FROM python:3.10-slim

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
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

# Copy requirements and install
# The build context should include backend/requirements.txt
COPY backend/requirements.txt /app/backend/requirements.txt

# Upgrade pip and install requirements
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy application code
COPY backend /app/backend

# Create required runtime directories
RUN mkdir -p /app/backend/data/uploads /app/backend/data/fl_weights /app/backend/data/models /app/backend/temp_uploads

VOLUME ["/app/backend/data"]

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Run using uvicorn; you can override CMD in production (e.g., use gunicorn+uvicorn workers)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

