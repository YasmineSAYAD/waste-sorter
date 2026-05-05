import requests
import streamlit as st
import os


API_URL = os.getenv("API_URL")


def api_post(endpoint: str, **kwargs):
    headers = get_headers()
    r = requests.post(f"{API_URL}{endpoint}", headers=headers, timeout=30, **kwargs)
    return handle_response(r)


def api_get(endpoint: str):
    headers = get_headers()
    r = requests.get(f"{API_URL}{endpoint}", headers=headers, timeout=10)
    return handle_response(r)


def api_put(endpoint: str, **kwargs):
    headers = get_headers()
    r = requests.put(f"{API_URL}{endpoint}", headers=headers, timeout=30, **kwargs)
    return handle_response(r)


def api_delete(endpoint: str):
    headers = get_headers()
    r = requests.delete(f"{API_URL}{endpoint}", headers=headers, timeout=10)
    return r.status_code in (200, 204), None


def get_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}


def handle_response(r):
    if r.ok:
        return r.json(), None
    try:
        return None, r.json().get("detail", r.text)
    except requests.exceptions.RequestException as e:
        return None, f"RequestException: {str(e)}"
    except ValueError:
        return None, f"Invalid JSON response: {r.text}"
    except Exception as e:
        return None, f"An unexpected error occurred: {str(e)}"
