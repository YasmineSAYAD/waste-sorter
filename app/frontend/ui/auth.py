import streamlit as st

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