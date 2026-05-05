import streamlit as st

from core.api import api_post

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
