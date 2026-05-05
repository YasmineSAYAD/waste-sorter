import os
import time
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://backend:8000")
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

st.markdown("""
<style>

    html,
    body, 
    [class*="css"] { 
        font-family: 'Inter',
        sans-serif;
    }

    #MainMenu,
    footer, 
    header {
        visibility: hidden;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
        min-width: 240px !important;
        max-width: 240px !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
    }
    [data-testid="stSidebarHeader"] {
        margin-bottom: 0 !important;
        height: 2.5rem !important;
    }

    /* Sidebar user card */
    .sidebar-user {
        background: #f8fafc;
        border: none;
        border-radius: 12px;
        margin-bottom: 18%;
        display: flex;
        justify-content: flex-start;
    }

    .sidebar-avatar {
        width: 30px;
        height: 30px;
        background: #4CAF50;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 700; color: white;
        margin-top: auto;
        margin-right: 5%;
        margin-bottom: 0.6rem;
        margin-left: 5%;
    }
    .sidebar-name {
        font-size: 0.95rem; 
        font-weight: 600; 
        color: #1a1a2e;
        margin-bottom: 0.15rem;
    }
    .sidebar-role {
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: capitalize;
    }

    /* Sidebar nav links */
    .sidebar-nav {
        margin-bottom: 1rem;
    }
    
    .sidebar-nav-item {
        display: flex;
        align-items: flex-start !important;
        padding: 0.65rem 0.75rem;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 500;
        color: #374151;
        cursor: pointer;
        text-decoration: none;
        transition: background 0.15s;
    }
   
    .sidebar-nav-item:hover { 
        background: #e2e8f0;
    }

    .sidebar-nav-item.active {
        background: #e2e8f0;
        font-weight: 600;
    }

    .sidebar-nav-icon { 
        font-size: 1rem; 
        width: 20px; 
        text-align: center; 
    }

    [data-testid="stSidebar"] .stButton > button {
        background: none !important;
        border: none !important;
        color: #212121 !important;
        font-weight: 500 !important;
        width: 100% !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.15s !important;
    }

    [data-testid="stSidebar"] .stButton {
        margin-bottom: 0px !important;
    }

    [data-testid="stSidebar"] .stButton > button {
       
        min-height: unset !important;
        line-height: 1.2 !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: #e2e8f0 !important;
    }

    /* Scanner */
    div.st-key-nav_scanner button::before {
        content: "\\f002"; /* fa-search */
    }

    /* Historique */
    div.st-key-nav_history button::before {
        content: "\\f1da"; /* fa-history */
    }

    /* Profil */
    div.st-key-nav_profile button::before {
        content: "\\f007"; /* fa-user */
    }
    /* politique */
    div.st-key-nav_politique button::before {
        content: "\\f3ed"; /* fa-shield-alt */
    }
    /* cgu */
    div.st-key-nav_cgu button::before {
        content: "\\f15c"; /* fa-file-alt */
    }

    /* ── Legal pages ── */
    .legal-section {
        background: #f8fafc; 
        border: 1px solid #e2e8f0; 
        border-radius: 12px;
        padding: 1.5rem 1.75rem; 
        margin-bottom: 1.25rem;
    }

    .legal-section h4 { 
        color: #1a1a2e; 
        margin-bottom: 0.75rem; 
        font-size: 1rem; 
    }

    .legal-section p, .legal-section li { 
        color: #374151; 
        font-size: 0.9rem; 
        line-height: 1.7; 
    }

    .legal-badge {
        display: inline-block; 
        background: #e8f5e9; 
        color: #2e7d32;
        border: 1px solid #a5d6a7; 
        border-radius: 20px;
        padding: 3px 12px; 
        font-size: 0.78rem; 
        font-weight: 600; 
        margin-bottom: 1.5rem;
    }

    /* icon style */
    div.stButton button::before {
        font-family: "Font Awesome 5 Free";
        font-weight: 900;
        color: #212121;
    }

    /* Alignement bouton */
    div.stButton button,
    div.st-key-sidebar_logout button{
        display: flex;
        justify-content: flex-start;
    }

    div.st-key-sidebar_logout button::before {
        content: "\\f2f5"; /* fa-sign-out-alt */
        font-family: "Font Awesome 5 Free";
        font-weight: 900;
        margin-right: 1px !important;
        color: #212121;
    }

    div.st-key-analyser button {
        display: flex !important;
        align-items: center !important;
        padding: 8px 12px !important;
        font-size: 16px !important;
        border: none !important;
        background-color: #4CAF50;
        color: white;
    }

    div.st-key-analyser button:hover {
        background-color: #388E3C !important;
    }

    /* ── Logo ── */
    .logo {
        text-align: center;
        color: #4CAF50;
    }

    .logo-icon {
        font-size: 3.5rem;
        display: block;
        margin-bottom: 0.4rem;
    }

    .logo-sub {
        color: #6b7280;
        font-size: 0.875rem;
        margin-top: 0.25rem;
    }

    .logo-title {
        font-size: 2rem;
        font-weight: 700;
        color: #212121;
        letter-spacing: -0.03em;
    }

    .logo-horizontal {
        color: #4CAF50;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }

    .logo-icon-horizontal {
        font-size: 2rem;
        display: inline-block;
    }
    .logo-title-horizontal {
        font-size: 1.4rem;
        font-weight: 700;
        color: #212121;
        letter-spacing: -0.03em;
        display: inline-block;
    }

    /* ── Auth ── */
    .rgpd-banner {
        background: #f0f9ff; 
        border: 1px solid #bae6fd; 
        border-radius: 10px;
        padding: 1rem 1.25rem; 
        font-size: 0.82rem; 
        color: #0369a1;
        margin: 1rem 0; 
        line-height: 1.6;
    }

    /* ── Result ── */
    .result-header {
        border-radius: 16px; 
        padding: 1rem; 
        color: white;
        margin-bottom: 1.5rem; 
        text-align: center;
    }
    .result-class { 
        font-size: 2rem;
        font-weight: 700; 
        margin: 0.4rem 0;
    }
   
    .badge-rec {
        display: inline-block;
        background: rgba(255,255,255,0.25);
        border: 1.5px solid rgba(255,255,255,0.5);
        padding: 0.3rem 1rem; *
        border-radius: 999px;
        font-size: 0.82rem; 
        font-weight: 600; 
        margin-top: 0.75rem; 
        color: white;
    }

    /* ── Cards ── */
    .info-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
    }

    .info-card-title {
        font-size: 0.72rem; 
        font-weight: 600; 
        color: #6b7280; 
        text-transform: uppercase; 
        letter-spacing: 0.08em; 
        margin-bottom: 0.4rem;
    }

    .info-card-value { 
        font-size: 0.95rem; 
        font-weight: 500; 
        color: #1a1a2e; 
    }

    .advice-card {
        background: #eff6ff; 
        border-left: 4px solid #3b82f6;
        border-radius: 0 10px 10px 0; 
        padding: 1rem 1.2rem;
        margin-bottom: 1rem; 
        font-size: 0.88rem; 
        color: #1e40af; 
        line-height: 1.6;
    }
    .advice-card i {
        color: #FFD700;  
    }

    .warning-card {
        background: #fffbeb; 
        border-left: 4px solid #f59e0b;
        border-radius: 0 10px 10px 0; 
        padding: 1rem 1.2rem;
        font-size: 0.88rem; 
        color: #92400e; 
        line-height: 1.6;
    }

    .upload-hint { 
        text-align: center; 
        color: #9ca3af; 
        font-size: 0.875rem; 
        padding: 2.5rem 1rem; 
    }

    .history-item {
        background: white; 
        border: 1px solid #e2e8f0; 
        border-radius: 10px;
        padding: 0.875rem 1rem; 
        margin-bottom: 0.5rem;
    }

    .st-emotion-cache-1s8qyds h3{
        padding: 0 0 1rem !important;
    }

    /* ── Buttons (main content) ── */
    .main-content .stButton > button,
    .stFormSubmitButton > button {
        background-color: #4CAF50 !important;
        border: 1px solid #4CAF50 !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }

    .stFormSubmitButton > button:hover {
        background-color: #43A047 !important;
        border-color: #43A047 !important;
    }

    /* Override for main content buttons only */
    section.main .stButton > button {
        background-color: #4CAF50;
        border: 1px solid #4CAF50;
        color: white;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    section.main .stButton > button:hover {
        background-color: #43A047;
        border-color: #43A047;
    }

    section.main .stButton > button:active {
        background-color: #388E3C;
        transform: scale(0.98);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #4CAF50 !important;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #4CAF50 !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #43A047; 
    }

    .st-c2 { 
        background-color: #4CAF50;
    }

    .st-emotion-cache-1hkb16d:hover {
        color: #4CAF50;
    }
   
    /* ── Responsive mobile ── */
    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            min-width: 100% !important;
            max-width: 100% !important;
        }

        .result-class { 
            font-size: 1.5rem;
        }
        .block-container { 
            padding-left: 0.5rem; 
            padding-right: 0.5rem; 
        }
    }

</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────
def init_session():
    for k, v in {
        "authenticated": False,
        "user": None,
        "token": None,
        "rgpd_accepted": False,
        "current_result": None,
        "active_page": "scanner",
        "register_success": False,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── Remember me ───────────────────────────────────────────────────
def save_credentials(email: str, password: str):
    st.session_state["remembered_email"] = email
    st.session_state["remembered_password"] = password
    st.session_state["remember_me"] = True


def clear_credentials():
    for k in ["remembered_email", "remembered_password", "remember_me"]:
        st.session_state.pop(k, None)


# ── API helpers ───────────────────────────────────────────────────
def api_post(endpoint: str, **kwargs):
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}
        r = requests.post(f"{API_URL}{endpoint}", headers=headers, timeout=30, **kwargs)
        if r.ok:
            return r.json(), None
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        return None, str(detail)
    except requests.exceptions.ConnectionError:
        return None, "Impossible de joindre le serveur."
    except requests.exceptions.Timeout:
        return None, "Le serveur ne répond pas."


def api_get(endpoint: str):
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}
        r = requests.get(f"{API_URL}{endpoint}", headers=headers, timeout=10)
        return (r.json(), None) if r.ok else (None, r.text)
    except Exception as e:
        return None, str(e)


def api_put(endpoint: str, **kwargs):
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}
        r = requests.put(f"{API_URL}{endpoint}", headers=headers, timeout=30, **kwargs)
        if r.ok:
            return r.json(), None
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        return None, str(detail)
    except Exception as e:
        return None, str(e)


def api_delete(endpoint: str):
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}
        r = requests.delete(f"{API_URL}{endpoint}", headers=headers, timeout=10)
        return r.status_code in (200, 204), None
    except Exception as e:
        return False, str(e)


def do_logout():
    api_post("/api/v1/users/logout")
    remembered = st.session_state.get("remember_me", False)
    em = st.session_state.get("remembered_email", "")
    pw = st.session_state.get("remembered_password", "")
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init_session()
    if remembered:
        st.session_state["remembered_email"] = em
        st.session_state["remembered_password"] = pw
        st.session_state["remember_me"] = True


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────

def render_sidebar():
    user = st.session_state.user or {}
    first = user.get("first_name", "")
    last = user.get("last_name", "")
    initials = f"{first[:1]}{last[:1]}".upper() or "U"
    full_name = f"{first} {last}".strip() or "Utilisateur"
    role = user.get("role", "user")

    with st.sidebar:
        # Logo
        st.markdown("""
        <div class="logo-horizontal">
            <span class="logo-icon-horizontal">♻</span>
            <div class="logo-title-horizontal">waste-sorter</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # User card
        st.markdown(f"""
        <div class="sidebar-user">
            <div class="sidebar-avatar">{initials}</div>
            <div>
                <div class="sidebar-name">{full_name}</div>
                <div class="sidebar-role">{role}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Nav links
        current = st.session_state.get("active_page", "scanner")

        nav_items = [
            ("scanner", "Scanner"),
            ("history", "Historique"),
            ("profile", "Mon compte"),
            ("cgu", "Conditions d'utilisation"),
            ("politique", "Politique de confidentialité")
        ]

        for page_key, label in nav_items:
            if st.button(
                label,
                key=f"nav_{page_key}",
                use_container_width=True,
                type="secondary"
            ):
                st.session_state.active_page = page_key
                st.rerun()

        st.divider()

        # Logout button
        if st.button("Déconnexion", key="sidebar_logout", use_container_width=True):
            do_logout()
            st.rerun()


# ─────────────────────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────────────────────

def page_login():
    _, col, _ = st.columns([1, 1.3, 1])
    with col:
        st.markdown("""
        <div class="logo">
            <div class="logo-icon">♻</div>
            <div class="logo-title">waste-sorter</div>
            <div class="logo-sub">Trions mieux, ensemble pour la planète.</div>
        </div>
        """, unsafe_allow_html=True)

        tab_in, tab_up = st.tabs(["Connexion", "Créer un compte"])

        # ── Login ─────────────────────────────────────────────────
        with tab_in:
            if st.session_state.get("register_success"):
                st.success(" Compte créé ! Connectez-vous ci-dessous.")
                st.session_state.register_success = False

            st.markdown("<br>", unsafe_allow_html=True)

            default_email = st.session_state.get("remembered_email", "")
            default_pwd = st.session_state.get("remembered_password", "")

            with st.form("login"):
                email = st.text_input("Email", value=default_email, placeholder="vous@exemple.fr")
                pwd = st.text_input("Mot de passe", value=default_pwd, type="password", placeholder="••••••••")
                remember = st.checkbox("Se souvenir de moi", value=bool(default_email))
                ok = st.form_submit_button("Se connecter", use_container_width=True, type="primary")

            if ok:
                if not email or not pwd:
                    st.error("Veuillez remplir tous les champs.")
                else:
                    with st.spinner("Connexion..."):
                        data, err = api_post("/api/v1/users/login", json={"email": email, "password": pwd})
                    if err:
                        st.error(f"Identifiants incorrects : {err}")
                        clear_credentials()
                    else:
                        if remember:
                            save_credentials(email, pwd)
                        else:
                            clear_credentials()
                        st.session_state.authenticated = True
                        st.session_state.user = data.get("user", {})
                        st.session_state.token = data.get("access_token")
                        st.success("Bienvenue !")
                        time.sleep(0.4)
                        st.rerun()

        # ── Register ──────────────────────────────────────────────
        with tab_up:
            st.markdown("<br>", unsafe_allow_html=True)

            if not st.session_state.rgpd_accepted:
                st.markdown("""
                <div class="rgpd-banner">
                    <i class="fas fa-lock"></i>
                    <strong> Protection de vos données (RGPD)</strong><br>
                    Vos données (email, prénom, nom) sont collectées uniquement pour l'accès
                    à l'application. Elles ne sont jamais partagées avec des tiers.
                    Vous pouvez demander leur suppression à tout moment.
                    Conformément au RGPD (UE 2016/679).
                </div>
                """, unsafe_allow_html=True)
                if st.checkbox("J'accepte la collecte et le traitement de mes données personnelles."):
                    st.session_state.rgpd_accepted = True
                    st.rerun()
                st.stop()

            with st.form("register"):
                ca, cb = st.columns(2)
                with ca:
                    fn = st.text_input("Prénom", placeholder="Marie")
                with cb:
                    ln = st.text_input("Nom", placeholder="Dupont")
                em = st.text_input("Email", placeholder="vous@exemple.fr")
                p1 = st.text_input("Mot de passe", type="password", placeholder="8 caractères minimum")
                p2 = st.text_input("Confirmer", type="password", placeholder="••••••••")
                ok2 = st.form_submit_button("Créer mon compte", use_container_width=True, type="primary")

            if ok2:
                if not all([fn, ln, em, p1, p2]):
                    st.error("Tous les champs sont obligatoires.")
                elif p1 != p2:
                    st.error("Les mots de passe ne correspondent pas.")
                elif len(p1) < 8:
                    st.error("Mot de passe trop court (8 caractères minimum).")
                else:
                    with st.spinner("Création..."):
                        data, err = api_post(
                            "/api/v1/users/register",
                            json={"first_name": fn, "last_name": ln, "email": em, "password": p1},
                        )
                    if err:
                        st.error(f"Erreur : {err}")
                    else:
                        st.session_state.rgpd_accepted = False
                        st.success("Compte créé avec succès! Connectez-vous dans l'onglet Connexion.")


# ─────────────────────────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────────────────────────

def render_result(result: dict):
    predicted = result.get("predicted_class", "")
    recyclable = result.get("recyclable", False)
    bac = result.get("bac") or ""
    alt = result.get("alt") or ""
    advice = result.get("advice") or ""
    label_fr = result.get("waste_type") or predicted

    gradient = (
        "linear-gradient(135deg, #4CAF50, #166534)"  
        if recyclable
        else "linear-gradient(135deg, #F44336, #D32F2F)"  
    )
    badge_text = "♻ Recyclable" if recyclable else "✕ Non recyclable"

    st.markdown(f"""
    <div class="result-header" style="background:{gradient}">
        <div class="result-class">{label_fr}</div>
        <span class="badge-rec">{badge_text}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<div class="info-card"><div class="info-card-title"><i class="fas fa-trash"></i> Bac recommandé</div>'
        f'<div class="info-card-value">{bac}</div></div>',
        unsafe_allow_html=True,
    )
    
    st.markdown(
        f'<div class="info-card"><div class="info-card-title"><i class="fas fa-exchange-alt"></i> Alternative</div>'
        f'<div class="info-card-value">{alt}</div></div>',
        unsafe_allow_html=True,
    )

    if advice:
        st.markdown(
            f'<div class="advice-card"><i class="fas fa-lightbulb"></i> <strong>Conseil :</strong> {advice}</div>',
            unsafe_allow_html=True,
        )


def page_scanner():
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.subheader("Protégeons l'environnement")
        st.caption(
            "Prenez une photo ou importez une image de votre déchet, "
            "nous vous guidons pour le trier de manière optimale."
        )

        input_mode = st.radio(
            "Mode",
            ["📁 Uploader une image", "📷 Prendre une photo"],
            horizontal=True,
            label_visibility="collapsed",
        )

        image_data = None

        if input_mode == "📁 Uploader une image":
            uploaded = st.file_uploader(
                "Image", type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed",
            )
            if uploaded:
                image_data = (uploaded.name, uploaded.getvalue(), uploaded.type)
                st.image(uploaded, use_container_width=True)
            else:
                st.markdown("""
                <div class="upload-hint">
                    <p style="font-size:2.5rem"><i class="fas fa-folder"></i></p>
                    <p>Glissez une image ici<br>ou cliquez pour sélectionner</p>
                    <p style="font-size:0.75rem;margin-top:0.5rem">JPG · PNG · WEBP · max 10MB</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            camera = st.camera_input("Photo", label_visibility="collapsed")
            if camera:
                image_data = ("camera_capture.jpg", camera.getvalue(), "image/jpeg")
                st.image(camera, use_container_width=True)
            else:
                st.markdown("""
                <div class="upload-hint">
                    <p style="font-size:2.5rem">📷</p>
                    <p>Activez votre caméra<br>et prenez une photo de votre déchet</p>
                </div>
                """, unsafe_allow_html=True)

        if image_data:
            user_id = st.session_state.user.get("id") 
            if st.button("Trier Vos Déchets", use_container_width=True, type="primary",  key="analyser"):
                fname, fdata, ftype = image_data
                params = {"user_id": user_id} 
                with st.spinner("Analyse en cours..."):
                    data, err = api_post(
                        "/api/v1/images/upload",
                        params=params,
                        files={"file": (fname, fdata, ftype)},
                    )
                if err:
                    st.error(f"Erreur : {err}")
                else:
                    st.session_state.current_result = data
                    st.rerun()

    with col_r:
        if st.session_state.current_result:
            render_result(st.session_state.current_result)
        else:
            st.markdown("""
            <div style="text-align:center;padding:4rem 2rem;color:#9ca3af">
                <p style="font-size:3rem">🔍</p>
                <p>Le résultat s'affichera ici<br>après l'analyse.</p>
            </div>
            """, unsafe_allow_html=True)


def page_history():
    st.subheader("Mon historique")
    st.divider()

    user_id = (st.session_state.user or {}).get("id")
    if not user_id:
        st.error("Utilisateur non identifié.")
        return

    with st.spinner("Chargement..."):
        data, err = api_get(f"/api/v1/users/{user_id}/history")

    if err:
        st.error(f"Erreur : {err}")
        return

    if not data:
        st.info("Aucune analyse effectuée pour le moment.")
        return

    st.caption(f"{len(data)} analyse(s) au total")
    for item in data:
        icon = "♻" if item.get("recyclable") is True else "🗑"
        color = "#4CAF50" if item.get("recyclable") is True else "#F44336"
        label = item.get("waste_type") or item.get("predicted_class") or "—"
        confidence = item.get("confidence") or 0
        date = (item.get("uploaded_at") or "")[:10] or "—"
        bac = item.get("bac") or "—"
        alt = item.get("alt") or "—"
        image_id = item.get("image_id")
        image_url = f"http://localhost:8000/api/v1/images/{image_id}/file"

        st.markdown(f"""
        <div class="history-item">
            <img src="{image_url}" style="width: 50px; height: auto; margin-right:2%;" />
            <span style="font-size:1.1rem; color:{color};">{icon}</span>
            <strong style="margin-left:0.5rem">{label}</strong>
            <span style="color:#6b7280;font-size:0.82rem;margin-left:0.5rem">— {date} · {bac} / {alt}</span>
            <span style="float:right;color:{color};font-weight:600;font-size:0.85rem">{confidence*100:.0f}%</span>
        </div>
        """, unsafe_allow_html=True)


def page_profile():
    st.subheader("Mon compte")
    st.divider()

    user = st.session_state.user or {}
    user_id = user.get("id")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### Modifier mes informations")
        with st.form("edit_profile"):
            fn = st.text_input("Prénom", value=user.get("first_name", ""))
            ln = st.text_input("Nom", value=user.get("last_name", ""))
            em = st.text_input("Email", value=user.get("email", ""))
            st.markdown("**Nouveau mot de passe** *(laisser vide pour ne pas modifier)*")
            new_pwd = st.text_input(
                "Mot de passe", type="password", placeholder="8 caractères minimum"
            )
            saved = st.form_submit_button(
                "Enregistrer", use_container_width=True, type="primary"
            )

        if saved:
            payload = {}
            if fn != user.get("first_name"):
                payload["first_name"] = fn
            if ln != user.get("last_name"):
                payload["last_name"] = ln
            if em != user.get("email"):
                payload["email"] = em
            if new_pwd:
                if len(new_pwd) < 8:
                    st.error("Mot de passe trop court.")
                    st.stop()
                payload["password"] = new_pwd

            if not payload:
                st.info("Aucune modification détectée.")
            else:
                data, err = api_put(f"/api/v1/users/{user_id}", json=payload)
                if err:
                    st.error(f"Erreur : {err}")
                else:
                    st.session_state.user.update({
                        "first_name": data.get("first_name", fn),
                        "last_name": data.get("last_name", ln),
                        "email": data.get("email", em),
                    })
                    st.success(" Profil mis à jour.")
                    time.sleep(0.5)
                    st.rerun()

    with col2:
        st.markdown("#### Supprimer mon compte")
        st.warning(
            "! Action irréversible. Toutes vos données seront supprimées définitivement."
        )

        if st.checkbox("Je comprends que cette action est irréversible"):
            if st.button("Supprimer mon compte", type="primary"):
                ok, err = api_delete(f"/api/v1/users/{user_id}")
                if err:
                    st.error(f"Erreur : {err}")
                else:
                    st.success("Compte supprimé.")
                    time.sleep(1)
                    do_logout()
                    st.rerun()

def page_cgu():
    st.subheader("Conditions Générales d'Utilisation")
    st.markdown('<span class="legal-badge">Mise à jour : Avril 2026</span>', unsafe_allow_html=True)
 
    sections = [
        ("1. Objet", """
        Les présentes Conditions Générales d'Utilisation (CGU) régissent l'accès et l'utilisation
        de l'application **waste-sorter**, accessible via l'interface web mise à disposition par l'éditeur.
        En utilisant l'application, vous acceptez sans réserve les présentes CGU.
                """),
                ("2. Description du service", """
        waste-sorter est une application d'aide au tri des déchets utilisant l'intelligence artificielle.
        Elle permet à l'utilisateur de photographier un déchet et d'obtenir des informations sur la
        manière de le trier correctement. Les informations fournies sont données à titre indicatif
        et peuvent varier selon les collectivités territoriales.
                """),
                ("3. Accès au service", """
        L'accès au service est réservé aux personnes physiques majeures (18 ans et plus) ayant créé
        un compte utilisateur. L'utilisateur s'engage à fournir des informations exactes lors de la
        création de son compte et à les maintenir à jour.
                """),
                ("4. Obligations de l'utilisateur", """
        L'utilisateur s'engage à :
        - Utiliser l'application de manière loyale et conforme à sa destination
        - Ne pas tenter de contourner les mesures de sécurité
        - Ne pas uploader de contenu illicite, offensant ou portant atteinte aux droits de tiers
        - Respecter la propriété intellectuelle de l'application et de ses composants
                """),
                ("5. Responsabilité", """
        L'éditeur s'efforce de maintenir l'application accessible et de fournir des informations exactes.
        Toutefois, les recommandations de tri sont données à titre indicatif. L'éditeur ne saurait être
        tenu responsable d'erreurs de classification de l'IA, de l'indisponibilité temporaire du service,
        ou de différences entre les consignes affichées et celles de votre collectivité.
                """),
                ("6. Propriété intellectuelle", """
        L'ensemble des éléments constituant l'application (code source, interface, modèles IA, contenus)
        est protégé par le droit de la propriété intellectuelle français. Toute reproduction ou utilisation
        non autorisée est interdite.
                """),
                ("7. Modification des CGU", """
        L'éditeur se réserve le droit de modifier les présentes CGU à tout moment. Les utilisateurs
        seront informés par email ou via une notification dans l'application. L'utilisation continuée
        du service après modification vaut acceptation des nouvelles CGU.
                """),
                ("8. Droit applicable", """
        Les présentes CGU sont régies par le droit français. Tout litige relatif à leur interprétation
        ou exécution sera soumis aux tribunaux compétents de Paris, France.
        
        **Éditeur :** waste-sorter
        **Contact :** contact@waste-sorter.fr
                """),
            ]
        
    for title, content in sections:
        st.markdown(f"""
        <div class="legal-section">
            <h4>{title}</h4>
            <p>{content.strip()}</p>
        </div>
        """, unsafe_allow_html=True)
        
        
def page_politique():
    st.subheader("Politique de Confidentialité")
    st.markdown('<span class="legal-badge">Conforme RGPD — Règlement UE 2016/679</span>', unsafe_allow_html=True)
        
    sections = [
    ("1. Responsable du traitement", """
    Le responsable du traitement des données personnelles collectées via l'application waste-sorter est :
    **waste-sorter**, joignable à l'adresse : contact@waste-sorter.fr
        
    Conformément au Règlement Général sur la Protection des Données (RGPD) et à la loi Informatique
    et Libertés du 6 janvier 1978 modifiée, nous nous engageons à protéger vos données personnelles.
            """),
            ("2. Données collectées", """
    Nous collectons uniquement les données strictement nécessaires au fonctionnement du service :
        
    - **Identité** : prénom, nom de famille
    - **Contact** : adresse email
    - **Authentification** : mot de passe hashé (non lisible, jamais stocké en clair)
    - **Images** : photos de déchets uploadées pour l'analyse IA
    - **Résultats d'analyse** : classe prédite, score de confiance, date d'analyse
    - **Données techniques** : logs d'accès (adresse IP, horodatage) — conservés 30 jours maximum
            """),
            ("3. Finalités et bases légales", """
    | Finalité | Base légale (RGPD art. 6) |
    |---|---|
    | Création et gestion du compte | Exécution du contrat |
    | Fourniture du service de classification | Exécution du contrat |
    | Amélioration du modèle IA | Intérêt légitime (données anonymisées) |
    | Envoi de notifications | Consentement |
    | Conformité légale | Obligation légale |
            """),
            ("4. Durée de conservation", """
    - **Données de compte** : conservées pendant toute la durée d'activité du compte, puis supprimées sous 30 jours après clôture
    - **Images uploadées** : conservées 12 mois, puis supprimées automatiquement
    - **Historique d'analyse** : conservé 24 mois
    - **Logs techniques** : 30 jours maximum
            """),
            ("5. Vos droits", """
    Conformément au RGPD et à la loi Informatique et Libertés, vous disposez des droits suivants :
        
    - **Droit d'accès** (art. 15) : obtenir une copie de vos données
    - **Droit de rectification** (art. 16) : corriger vos données inexactes
    - **Droit à l'effacement** (art. 17) : supprimer votre compte et vos données
    - **Droit à la portabilité** (art. 20) : recevoir vos données dans un format structuré
    - **Droit d'opposition** (art. 21) : vous opposer à certains traitements
    - **Droit à la limitation** (art. 18) : limiter le traitement de vos données
        
    Pour exercer ces droits : **contact@waste-sorter.fr**
    Réponse garantie sous **30 jours**.
        
    Vous pouvez également introduire une réclamation auprès de la **CNIL** :
    Commission Nationale de l'Informatique et des Libertés — www.cnil.fr
            """),
            ("6. Partage des données", """
    Vos données personnelles ne sont jamais vendues ni partagées à des fins commerciales.
    Elles peuvent être transmises uniquement :
        
    - Aux sous-traitants techniques nécessaires au fonctionnement (hébergement, infrastructure)
    - Sur réquisition judiciaire ou obligation légale
        
    Tous nos sous-traitants sont soumis à des garanties contractuelles conformes au RGPD.
            """),
            ("7. Sécurité", """
    Nous mettons en œuvre des mesures techniques et organisationnelles appropriées :
        
    - Chiffrement des données en transit (HTTPS/TLS)
    - Hashage irréversible des mots de passe (bcrypt)
    - Accès aux données restreint au personnel autorisé
    - Sauvegardes régulières et chiffrées
    - Surveillance des accès et détection des anomalies
            """),
            ("8. Cookies", """
    waste-sorter n'utilise pas de cookies de tracking ou publicitaires.
    Seuls des cookies techniques strictement nécessaires au fonctionnement de la session
    utilisateur sont utilisés. Ces cookies ne nécessitent pas de consentement préalable
    conformément aux lignes directrices de la CNIL.
            """),
            ("9. Contact et DPO", """
    Pour toute question relative à la protection de vos données ou pour exercer vos droits :
        
    **Email :** contact@waste-sorter.fr
    **Adresse :** waste-sorter — Paris, France
    **CNIL :** www.cnil.fr — 3 Place de Fontenoy, 75007 Paris
        
    La présente politique est applicable depuis le **1er janvier 2026**.
    Elle sera mise à jour en cas d'évolution réglementaire ou de changement dans nos pratiques.
            """),
        ]
        
    for title, content in sections:
        st.markdown(f"""
        <div class="legal-section">
            <h4>{title}</h4>
            <p>{content.strip()}</p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────

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
   