FROM python:3.12-slim

WORKDIR /app

# Install streamlit + required dependencies
RUN pip install --no-cache-dir streamlit requests pillow

# Copy the frontend
COPY . .

EXPOSE 8501

# Lancer Streamlit
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]