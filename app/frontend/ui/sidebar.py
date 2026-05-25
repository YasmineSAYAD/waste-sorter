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
        <div class="hamburger-btn" id="hamburgerBtn" onclick="toggleSidebar()" aria-label="Menu">
            <span></span>
            <span></span>
            <span></span>
        </div>
        <div class="sidebar-overlay" id="sidebarOverlay" onclick="closeSidebar()"></div>

        <script>
        function toggleSidebar() {
            const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
            const overlay = window.parent.document.getElementById('sidebarOverlay');
            const btn = window.parent.document.getElementById('hamburgerBtn');
            if (!sidebar) return;
            sidebar.classList.toggle('sidebar-open');
            overlay.classList.toggle('active');
            // Animate hamburger → X
            const spans = btn.querySelectorAll('span');
            if (sidebar.classList.contains('sidebar-open')) {
                spans[0].style.transform = 'rotate(45deg) translate(5px, 5px)';
                spans[1].style.opacity = '0';
                spans[2].style.transform = 'rotate(-45deg) translate(5px, -5px)';
            } else {
                spans[0].style.transform = '';
                spans[1].style.opacity = '';
                spans[2].style.transform = '';
            }
        }
        function closeSidebar() {
            const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
            const overlay = window.parent.document.getElementById('sidebarOverlay');
            const btn = window.parent.document.getElementById('hamburgerBtn');
            if (!sidebar) return;
            sidebar.classList.remove('sidebar-open');
            overlay.classList.remove('active');
            const spans = btn.querySelectorAll('span');
            spans[0].style.transform = '';
            spans[1].style.opacity = '';
            spans[2].style.transform = '';
        }
        </script>
        """, unsafe_allow_html=True)

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
            <div>
                <div class="sidebar-name">{full_name}</div>
            </div>
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
