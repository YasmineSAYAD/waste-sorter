import os

def load_config():
    if "API_URL" not in os.environ:
        os.environ["API_URL"] = "http://backend:8000"