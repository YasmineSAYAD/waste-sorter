import streamlit as st

def page_privacy():
    st.subheader("Données personnelles")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Données collectées")
        st.markdown("""
        - **Prénom, Nom** — personnalisation
        - **Email** — authentification
        - **Images uploadées** — analyse IA
        - **Résultats** — historique permanent

        Aucune donnée n'est vendue ou partagée avec des tiers.
        """)
    with c2:
        st.markdown("#### Vos droits (RGPD UE 2016/679)")
        st.markdown("""
        - **Accès** — via l'onglet Profil
        - **Rectification** — modifiez dans Profil
        - **Effacement** — supprimez votre compte dans Profil
        - **Portabilité** — exportez via l'historique
        - **Opposition** — contactez l'administrateur
        """)
    st.divider()
    st.info(
        "Pour toute demande concernant vos données,"
        "contactez l'administrateur de l'application."
    )


def page_cgu():
    st.subheader("Conditions Générales d'Utilisation")
    st.markdown(
        '<span class="legal-badge">Mise à jour : Avril 2026</span>',
        unsafe_allow_html=True
    )

    sections = [
        ("1. Objet", """
Les présentes Conditions Générales d'Utilisation (CGU)
régissent l'accès et l'utilisation de l'application <b>waste-sorter</b>,
accessible via l'interface web mise à disposition par l'éditeur.
En utilisant l'application, vous acceptez sans réserve les présentes CGU.
        """),
        ("2. Description du service", """
waste-sorter est une application d'aide au tri des déchets utilisant l'intelligence
artificielle. Elle permet à l'utilisateur de photographier un déchet et d'obtenir des
informations sur la manière de le trier correctement. Les informations fournies sont
données à titre indicatif et peuvent varier selon les collectivités territoriales.
        """),
        ("3. Accès au service", """
L'accès au service est réservé aux personnes physiques majeures (18 ans et plus) ayant
créé un compte utilisateur. L'utilisateur s'engage à fournir des informations exactes
lors de la création de son compte et à les maintenir à jour.
        """),
        ("4. Obligations de l'utilisateur", """
L'utilisateur s'engage à :

- Utiliser l'application de manière loyale et conforme à sa destination
- Ne pas tenter de contourner les mesures de sécurité
- Ne pas uploader de contenu illicite, offensant ou portant atteinte aux droits de tiers
- Respecter la propriété intellectuelle de l'application et de ses composants
        """),
        ("5. Responsabilité", """
L'éditeur s'efforce de maintenir l'application accessible et de fournir des informations
exactes. Toutefois, les recommandations de tri sont données à titre indicatif. L'éditeur
ne saurait être tenu responsable d'erreurs de classification de l'IA,
de l'indisponibilité temporaire du service, ou de différences entre
les consignes affichées et celles de votre collectivité.
        """),
        ("6. Propriété intellectuelle", """
L'ensemble des éléments constituant l'application (code source, interface, modèles IA,
contenus) est protégé par le droit de la propriété intellectuelle français.
Toute reproduction ou utilisation non autorisée est interdite.
        """),
        ("7. Modification des CGU", """
L'éditeur se réserve le droit de modifier les présentes CGU à tout moment.
Les utilisateurs seront informés par email
ou via une notification dans l'application.
L'utilisation continuée du service après modification
vaut acceptation des nouvelles CGU.
        """),
        ("8. Droit applicable", """
Les présentes CGU sont régies par le droit français. Tout litige relatif
à leur interprétation
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
    st.markdown(
        '<span class="legal-badge">Conforme RGPD — Règlement UE 2016/679</span>',
        unsafe_allow_html=True,
    )

    sections = [
        ("1. Responsable du traitement", """
Le responsable du traitement des données personnelles collectées via l'application
waste-sorter est :<b>waste-sorter</b>, joignable à l'adresse : contact@waste-sorter.fr

Conformément au Règlement Général sur la Protection des Données (RGPD) et à la loi
Informatique et Libertés du 6 janvier 1978 modifiée, nous nous engageons à protéger vos
données personnelles.
        """),
        ("2. Données collectées", """
Nous collectons uniquement les données strictement nécessaires
 au fonctionnement du service :

- **Identité** : prénom, nom de famille
- **Contact** : adresse email
- **Authentification** : mot de passe hashé (non lisible, jamais stocké en clair)
- **Images** : photos de déchets uploadées pour l'analyse IA
- **Résultats d'analyse** : classe prédite, score de confiance, date d'analyse
- **Données techniques** : logs d'accès (adresse IP, horodatage)
 — conservés 30 jours maximum
        """),
        ("4. Durée de conservation", """
Les données sont conservées selon les durées suivantes :

- **Données de compte** : conservées pendant toute la durée d'activité du compte, puis
supprimées sous 30 jours après clôture
- **Images uploadées** : conservées 12 mois, puis supprimées automatiquement
- **Historique d'analyse** : conservé 24 mois
- **Logs techniques** : 30 jours maximum
        """),
        ("5. Vos droits", """
Conformément au RGPD et à la loi Informatique et Libertés,
 vous disposez des droits suivants :

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

- Aux sous-traitants techniques nécessaires au fonctionnement
 (hébergement, infrastructure)
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
Elle sera mise à jour en cas d'évolution réglementaire
ou de changement dans nos pratiques.
        """),
    ]

    for title, content in sections:
        st.markdown(f"""
        <div class="legal-section">
            <h4>{title}</h4>
            <p>{content.strip()}</p>
        </div>
        """, unsafe_allow_html=True)



