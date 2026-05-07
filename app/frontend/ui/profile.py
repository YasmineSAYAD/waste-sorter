import streamlit as st
import time

from core.api import api_put, api_delete
from ui.auth import do_logout

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
            st.markdown(
                "**Nouveau mot de passe** *(laisser vide pour ne pas modifier)*"
            )
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
            "! Action irréversible. "
            "Toutes vos données seront supprimées définitivement."
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
