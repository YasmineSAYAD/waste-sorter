import streamlit as st

from core.session import init_session
from core.config import load_config

from ui.sidebar import render_sidebar
from ui.auth import page_login
from ui.scanner import page_scanner
from ui.history import page_history
from ui.profile import page_profile
from ui.legal import page_cgu, page_politique
from styles.main import load_css


st.set_page_config(
    page_title="waste-sorter",
    page_icon="images/favicon.png",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css" rel="stylesheet">
""", unsafe_allow_html=True)

load_css()
init_session()
load_config()


if not st.session_state.authenticated:
    page_login()
else:
    render_sidebar()

    page = st.session_state.get("active_page", "scanner")

    if page == "scanner":
        page_scanner()
    elif page == "history":
        page_history()
    elif page == "profile":
        page_profile()
    elif page == "cgu":
        page_cgu()
    elif page == "politique":
        page_politique()
