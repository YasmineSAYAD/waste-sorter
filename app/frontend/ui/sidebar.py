import streamlit as st
from ui.auth import do_logout

def render_sidebar():
    user = st.session_state.user or {}
    first = user.get("first_name", "")
    last = user.get("last_name", "")
    initials = f"{first[:1]}{last[:1]}".upper() or "U"
    full_name = f"{first} {last}".strip() or "Utilisateur"

    with st.sidebar:
        st.markdown("""
        <div class="logo-horizontal">
            <span class="logo-icon-horizontal">♻</span>
            <div class="logo-title-horizontal">waste-sorter</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown(f"""
        <div class="sidebar-user">
            <div class="sidebar-avatar">{initials}</div>
            <div class="sidebar-name">{full_name}</div>
        </div>
        """, unsafe_allow_html=True)

        nav_items = [
            ("scanner", "Scanner"),
            ("history", "Historique"),
            ("profile", "Mon compte"),
            ("cgu", "CGU"),
            ("politique", "Politique"),
        ]

        for key, label in nav_items:
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.active_page = key
                st.rerun()

        st.divider()

        if st.button("Déconnexion", key="logout", use_container_width=True):
            do_logout()
            st.rerun()
