# backend/Dockerfile
FROM python:3.11-slim

# system deps (add more if your models need them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# workdir
WORKDIR /app/backend

# copy only what we need for deps first to leverage layer cache
COPY backend/requirements.pinned.txt /app/backend/requirements.pinned.txt
# if you don't have requirements.pinned, the build may fail — fallback to requirements.txt in instructions
RUN pip install --no-cache-dir -r /app/backend/requirements.pinned.txt

# copy app code
COPY backend /app/backend

# create uploads and data dir
RUN mkdir -p /app/backend/data/uploads
VOLUME ["/app/backend/data"]

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose API port
EXPOSE 8000

# Default command: run uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
