# ---------- STAGE 1 : Build environment ----------
# Use a lightweight Python base image to install dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

# Install only minimal system libraries required by Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies into a separate folder
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --prefix=/install streamlit requests pillow

# Copy the frontend source code
COPY . .

# ---------- STAGE 2 : Distroless runtime ----------
# Distroless Python image: no shell, no package manager, minimal attack surface
FROM gcr.io/distroless/python3-debian12

WORKDIR /app

# Copy installed Python packages from the builder stage
COPY --from=builder /install /usr/local

# Copy the application source code
COPY --from=builder /app /app

# Set Python path so the runtime can find installed packages
ENV PYTHONPATH=/usr/local/lib/python3.12/site-packages

# Backend API URL
ENV API_URL=http://backend:8000

EXPOSE 8501

# Run Streamlit using the distroless Python entrypoint
CMD ["python", "-m", "streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]