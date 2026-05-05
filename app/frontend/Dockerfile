FROM python:3.12-slim

WORKDIR /app
ENV API_URL=http://backend:8000
RUN pip install --no-cache-dir streamlit requests pillow

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]