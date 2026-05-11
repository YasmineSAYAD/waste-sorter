FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OpenCV, Ultralytics, PostgreSQL
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY app/backend/requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY app/backend ./app/backend

# Copy model and label map
COPY model/saved ./model/saved
COPY model/data/splits/label_map.json ./model/data/splits/label_map.json

# Remove build dependencies to reduce image size
RUN apt-get purge -y gcc libpq-dev && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
