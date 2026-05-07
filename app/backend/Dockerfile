# ---------- STAGE 1 : Build environment ----------
# Use a lightweight Python base image to install and compile dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system dependencies required for psycopg, OpenCV, Pillow, and Ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a separate folder (/install)
COPY app/backend/requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# Copy backend source code
COPY app/backend ./app/backend

# Copy model weights
COPY model/saved ./model/saved

# Copy label map
COPY model/data/splits/label_map.json ./model/data/splits/label_map.json

# ---------- STAGE 2 : Distroless runtime ----------
# Distroless Python image: no shell, no package manager, minimal attack surface
FROM gcr.io/distroless/python3-debian12

WORKDIR /app

# Copy installed Python packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application code and model files
COPY --from=builder /app /app

# Ensure Python can locate installed packages
ENV PYTHONPATH=/usr/local/lib/python3.12/site-packages

EXPOSE 8000

# Run FastAPI using Uvicorn through the distroless Python entrypoint
CMD ["python", "-m", "uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]