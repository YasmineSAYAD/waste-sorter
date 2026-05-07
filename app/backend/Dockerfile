FROM python:3.12-slim

WORKDIR /app

# Install only the minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY app/backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY app/backend ./app/backend

# Copy model weights
COPY model/saved ./model/saved

# Copy label_map.json
COPY model/data/splits/label_map.json ./model/data/splits/label_map.json

# Remove build tools to reduce vulnerabilities
RUN apt-get purge -y gcc libpq-dev && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

EXPOSE 8000

CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]