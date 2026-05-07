FROM python:3.12-slim

WORKDIR /app

# Environment variable for backend API
ENV API_URL=http://backend:8000

# Install only minimal system dependencies for Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir streamlit requests pillow

# Copy frontend source
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]