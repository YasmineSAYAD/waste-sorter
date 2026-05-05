FROM python:3.12-slim

WORKDIR /app

# System deps for Pillow, psycopg and OpenCV (used by ultralytics)
RUN apt-get update && apt-get install -y \
    libpq-dev gcc libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
COPY app/backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY app/backend ./app/backend

# Copy model weights 
COPY model/saved ./model/saved

# Copy label_map.json
COPY model/data/splits/label_map.json ./model/data/splits/label_map.json 

EXPOSE 8000

CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]