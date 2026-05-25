import streamlit as st
from core.api import api_get

ITEMS_PER_PAGE = 6

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

    total = len(data)
    total_pages = max(1, -(-total // ITEMS_PER_PAGE))  # ceiling division

    if "history_page" not in st.session_state:
        st.session_state.history_page = 1

    # Clamp page in case data shrinks
    st.session_state.history_page = min(st.session_state.history_page, total_pages)

    page = st.session_state.history_page
    start = (page - 1) * ITEMS_PER_PAGE
    end = start + ITEMS_PER_PAGE
    page_data = data[start:end]

    st.caption(f"{total} analyse(s) au total · page {page}/{total_pages}")

    for item in page_data:
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
            <img src="{image_url}" style="width: 50px; height: auto; margin-right:2%;"/>
            <span style="font-size:1.1rem; color:{color};">{icon}</span>
            <strong style="margin-left:0.5rem">{label}</strong>
            <span style="color:#6b7280;font-size:0.82rem;margin-left:0.5rem">
                — {date} · {bac} / {alt}
            </span>
            <span style="float:right;color:{color};font-weight:600;font-size:0.85rem">
                {confidence*100:.0f}%
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Pagination controls
    if total_pages > 1:
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        cols = st.columns([1, 1, 2, 1, 1])

        with cols[0]:
            if st.button("«", key="hist_first", disabled=(page == 1), use_container_width=True):
                st.session_state.history_page = 1
                st.rerun()

        with cols[1]:
            if st.button("‹", key="hist_prev", disabled=(page == 1), use_container_width=True):
                st.session_state.history_page = page - 1
                st.rerun()

        with cols[2]:
            st.markdown(
                f"<div style='text-align:center;padding-top:0.4rem;font-size:0.85rem;color:#6b7280;'>"
                f"{page} / {total_pages}</div>",
                unsafe_allow_html=True
            )

        with cols[3]:
            if st.button("›", key="hist_next", disabled=(page == total_pages), use_container_width=True):
                st.session_state.history_page = page + 1
                st.rerun()

        with cols[4]:
            if st.button("»", key="hist_last", disabled=(page == total_pages), use_container_width=True):
                st.session_state.history_page = total_pages
                st.rerun()