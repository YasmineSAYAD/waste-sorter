import streamlit as st

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