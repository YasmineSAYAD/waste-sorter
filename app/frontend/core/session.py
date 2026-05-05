import streamlit as st

def init_session():
    defaults = {
        "authenticated": False,
        "user": None,
        "token": None,
        "rgpd_accepted": False,
        "current_result": None,
        "active_page": "scanner",
        "register_success": False,
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
