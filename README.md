# waste-sorter 

> Application full-stack de classification de déchets par intelligence artificielle — 11 catégories, information de recyclabilité, interface web intuitive.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-cls-00FFFF?logo=ultralytics&logoColor=black)
![MLflow](https://img.shields.io/badge/MLflow-2.x-0194E2?logo=mlflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)
![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)

---

## Table des matières

- [Présentation](#présentation)
- [Architecture](#architecture)
- [Structure du projet](#structure-du-projet)
- [Modèles ML](#modèles-ml)
- [Dataset](#dataset)
- [Mapping recyclabilité](#mapping-recyclabilité)
- [Installation et démarrage](#installation-et-démarrage)
- [Entraînement](#entraînement)
- [API REST](#api-rest)
- [Interface utilisateur](#interface-utilisateur)
- [Base de données](#base-de-données)
- [Monitoring](#monitoring)
- [CI/CD](#cicd)
- [MLflow](#mlflow)
- [Variables d'environnement](#variables-denvironnement)
- [Conformité RGPD](#conformité-rgpd)

---

## Présentation

**waste-sorter** est une application full-stack de tri intelligent des déchets. L'utilisateur photographie un déchet — l'IA l'identifie parmi 11 catégories et indique comment le recycler (bac, alternative, conseils).

Trois architectures CNN ont été entraînées, comparées et trackées avec MLflow :

| Modèle | Architecture | Test Accuracy | Remarques |
|---|---|---|---|
| CNN Scratch | 5 ConvBlocks + GAP | ~70% | Baseline, sans pré-entraînement |
| MobileNetV2 | Transfer learning + fine-tuning | ~85% | Poids ImageNet, 2 phases |
| **YOLOv8n-cls** | YOLOv8 classification | **meilleur** | Vitesse d'inférence optimale |

Le modèle **YOLOv8n-cls** est déployé en production via le **MLflow Model Registry**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        waste-sorter                         │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Streamlit   │───▶│   FastAPI    │───▶│  PostgreSQL  │  │
│  │  Frontend    │    │   Backend    │    │     16       │  │
│  │  :8501       │◀───│   :8000      │    │   :5432      │  │
│  └──────────────┘    └──────┬───────┘    └──────────────┘  │
│                             │                               │
│                    ┌────────▼────────┐                      │
│                    │  YOLOv8n-cls    │                      │
│                    │  MLflow Registry│                      │
│                    └─────────────────┘                      │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   MLflow     │    │  Prometheus  │    │   Grafana    │  │
│  │   :5000      │    │   :9090      │    │   :3001      │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Structure du projet

```
waste-sorter/
│
├── 📁 model/                          # Tout ce qui concerne le ML
│   ├── 📁 notebooks/                  # Notebooks d'exploration et d'entraînement
│   │   ├── eda.ipynb               # Exploration du dataset, splits, stats
│   │   ├── cnn_scratch.ipynb       # CNN from scratch + Optuna + MLflow
│   │   ├── cnn_mobilenet.ipynb     # MobileNetV2 fine-tuning + Optuna + MLflow
│   │   └── cnn_yolo.ipynb          # YOLOv8 classification + Optuna + MLflow
│   │
│   ├── 📁 src/                        # Code Python extrait des notebooks
│   │   └── preprocessing.py           # WasteDataset, transforms, dataloaders
│   │
│   ├── 📁 data/
│   │   ├── 📁 raw/                    # Images originales (git-ignoré, Git LFS)
│   │   ├── 📁 splits/                 # splits.json, label_map.json, stats EDA
│   │   └── 📁 yolo/                   # Structure train/val/test pour YOLO
│   │
│   ├── 📁 saved/                      # Checkpoints des modèles 
│   └── 📁 mlruns/                     # Runs MLflow en local (git-ignoré)
│
├── 📁 app/
│   │
│   ├── 📁 backend/                    # API FastAPI
│   │   ├── 📁 api/
│   │   │   ├── 📁 routes/
│   │   │   │   ├── images.py          # Upload, get, delete, file serving
│   │   │   │   ├── predictions.py     # Get prediction by ID
│   │   │   │   ├── users.py           # Register, login, logout, CRUD, history
│   │   │   │   └── waste.py           # Types et infos de déchets
│   │   │   └── 📁 schemas/
│   │   │       └── schemas.py         # Schémas Pydantic (validation + OpenAPI)
│   │   ├── 📁 core/
│   │   │   ├── config.py              # Settings via variables d'environnement
│   │   │   └── model.py               # Chargement YOLOv8, inférence, LABEL_MAP
│   │   ├── 📁 db/
│   │   │   └── session.py             # Engine SQLAlchemy async, session factory
│   │   ├── 📁 models/
│   │   │   └── tables.py              # ORM : User, Image, Prediction, WasteInfo
│   │   ├── 📁 tests/
│   │   │   └── test_routes.py         # Tests Pytest avec DB mockée
│   │   ├── main.py                    # Entry point FastAPI, routers, Prometheus
│   │   ├── requirements.txt           # Dépendances Python du backend
│   │   └── Dockerfile
│   │
│   └── 📁 frontend/                   # Interface Streamlit
│       ├── 📁 core/
│       │   ├── api.py                 # Helpers HTTP (api_post, api_get, api_put)
│       │   ├── config.py              # Configuration (API_URL, etc.)
│       │   └── session.py             # Gestion de la session utilisateur
│       ├── 📁 ui/
│       │   ├── auth.py                # Pages login et création de compte
│       │   ├── scanner.py             # Page principale — upload/caméra + résultat
│       │   ├── history.py             # Historique des analyses
│       │   ├── profile.py             # Profil utilisateur, modification, suppression
│       │   ├── sidebar.py             # Sidebar de navigation responsive
│       │   └── legal.py               # CGU et Politique de confidentialité
│       ├── 📁 images/                 # Assets (favicon, icônes)
│       ├── 📁 styles/                 # CSS global et thème
│       ├── main.py                     # Entry point Streamlit
│       └── Dockerfile
│
├── 📁 db/
│   └── init.sql                       # Schéma PostgreSQL initial + seed des 11 classes
│
├── 📁 monitoring/
│   ├── 📁 prometheus/
│   │   └── prometheus.yml             # Config scraping (backend :8000/metrics)
│   └── 📁 grafana/
│       └── 📁 dashboards/             # Dashboards JSON prêts à importer
│
├── 📁 .github/
│   └── 📁 workflows/
│       ├── ci.yml                     # Lint, type check, sécurité, tests, notebooks
│       └── cd.yml                     # Build Docker, push GHCR
│
├── docker-compose.yml                 # 6 services orchestrés
├── .env.example                       # Template des variables d'environnement
├── Makefile                           # Commandes raccourcies
├── pyproject.toml                     # Configuration Ruff, Mypy, Pytest
└── README.md
```

---

## Modèles ML

### Processus d'entraînement

```
eda.ipynb
  └── Exploration dataset, calcul mean/std RGB, splits stratifiés 80/10/10

cnn_scratch.ipynb
  └── CNN 5 blocs convolutifs + GAP → Optuna → MLflow tracking

cnn_mobilenet.ipynb
  └── MobileNetV2 pretrained → Phase 1 (frozen) → Phase 2 (unfreeze) → MLflow

04_cnn_yolo.ipynb
  └── YOLOv8n-cls → structure YOLO dataset → Optuna → MLflow → Model Registry
```

### Optimisation des hyperparamètres (Optuna)

Chaque notebook utilise **Optuna** avec le sampler TPE pour optimiser :
- Learning rate, dropout, batch size, weight decay, optimizer

Chaque trial est loggué dans **MLflow** avec ses métriques et paramètres.

### Cycle de vie du modèle (MLflow)

```
Entraînement → MLflow Tracking → Model Registry
                                      │
                              ┌───────┴────────┐
                              ▼                ▼
                           Staging        Production
                       (test_acc > seuil) (validé manuellement)
                                               │
                                    Backend FastAPI charge
                                    le modèle "Production"
                                    au démarrage
```

---

## Dataset

Le dataset est une collection privée de **~11 000 images** réparties en 11 catégories.

> ⚠️ Le dataset n'est **pas inclus** dans ce dépôt (trop volumineux).
> Il est géré via **Git LFS** et stocké de façon privée.

### Récupérer le dataset

```bash
# Via Git LFS après le clone
git lfs install
git lfs pull

# Ou placer les images dans model/data/raw/ puis régénérer les splits
jupyter notebook model/notebooks/eda.ipynb
```

### Répartition des données

```
Train : 9 004 images  (80%)
Val   : 1 126 images  (10%)
Test  : 1 126 images  (10%)
Total : 11 256 images
```

---

## Mapping recyclabilité

Chaque classe est associée à une filière de tri. Ce mapping est la **source de vérité** — stocké dans `model/data/splits/label_map.json` et chargé dans la table `waste_infos` de PostgreSQL.

| Classe      | Type         | Recyclable | Alternative | Bac              | Conseil |
|-------------|------------------------|------------|-------------------------------|------------------|---------|
| battery     | Pile / Batterie        | ✅ Oui     | Supermarché ou magasin        | Point de collecte | Dépose-les dans un point de collecte, ne les jette jamais à la poubelle. |
| cardboard   | Carton                 | ✅ Oui     | —                             | Bac jaune        | À plier pour gagner de la place. |
| electronic  | Électronique           | ✅ Oui     | Magasin (reprise gratuite)    | Déchèterie       | Le magasin reprend ton ancien appareil lors d’un achat, et accepte aussi les petits appareils sans obligation d’achat. |
| glass       | Verre                  | ✅ Oui     | Borne à verre (verte)         | Bac blanc        | Pense à enlever les bouchons et capsules avant de jeter. |
| medical     | Déchets médicaux       | ❌ Non     | Point de collecte médical     | Pharmacie        | Dépose les déchets médicaux dans des points spécialisés. Ne jamais les jeter dans la poubelle classique. |
| metal       | Métal                  | ✅ Oui     | Déchèterie (gros objets)      | Bac jaune        | Boîtes de conserve, canettes acceptées. |
| organic     | Déchets organiques     | ❌ Non     | Compost maison                | Bac marron       | Le tri des biodéchets est obligatoire si une solution existe près de chez toi. |
| paper       | Papier                 | ✅ Oui     | Point tri                     | Bac jaune        | Pas de papier gras ou mouillé. |
| plastic     | Plastique              | ✅ Oui     | Déchèterie (plastiques durs)  | Bac jaune        | Tous les emballages plastiques sont maintenant acceptés. |
| textile     | Textile                | ✅ Oui     | Associations (Emmaüs, Croix-Rouge) | Borne textile | Même usés, à déposer propres et secs dans un sac fermé. |
| trash       | Déchets                | ❌ Non     | Déchèterie                    | Bac gris         | Utilise cette poubelle uniquement si tu ne peux pas trier autrement. |


---

## Installation et démarrage

### Prérequis

- Docker & Docker Compose
- Python 3.12+
- Git LFS
- Make (optionnel)

### Démarrage rapide

```bash
# 1. Cloner le dépôt
git clone https://github.com/YasmineSAYAD/waste-sorter.git
cd waste-sorter

# 2. Récupérer le dataset
git lfs install && git lfs pull

# 3. Configurer l'environnement
cp .env.example .env
# Éditer .env avec vos valeurs

# 4. Lancer tous les services
docker compose up --build
```

### Services disponibles

| Service | URL | Description |
|---|---|---|
| Frontend Streamlit | http://localhost:8501 | Interface utilisateur |
| Backend FastAPI | http://localhost:8000 | API REST |
| Swagger UI | http://localhost:8000/docs | Documentation interactive |
| ReDoc | http://localhost:8000/redoc | Documentation alternative |
| MLflow UI | http://localhost:5000 | Suivi des expériences ML |
| Grafana | http://localhost:3001 | Dashboards de monitoring |
| Prometheus | http://localhost:9090 | Métriques brutes |

### Commandes Makefile

```bash
make up        # Lance tous les services Docker
make down      # Arrête tous les services
make build     # Rebuild sans cache
make test      # Lance les tests Pytest
make lint      # Ruff + Mypy sur le backend
make mlflow    # Lance l'UI MLflow en local
make logs      # Affiche les logs du backend
```

---

## Entraînement

### En local (GPU requis)

```bash
pip install -r app/backend/requirements.txt

# EDA et génération des splits
jupyter notebook model/notebooks/01_eda.ipynb

# Entraînement
jupyter notebook model/notebooks/cnn_scratch.ipynb
jupyter notebook model/notebooks/cnn_mobilenet.ipynb
jupyter notebook model/notebooks/cnn_yolo.ipynb
```

### Sur Google Colab (GPU T4 gratuit — recommandé)

1. Uploader le dataset zippé sur Google Drive dans `waste-sorter/mon_dataset.zip`
2. Ouvrir le notebook sur Colab
3. **Exécution → Modifier le type d'exécution → GPU T4**
4. Lancer toutes les cellules — modèles et runs MLflow sauvegardés automatiquement sur Drive
---

## API REST

### Authentification

Le token d'accès retourné au login doit être passé dans le header `Authorization` :

```
Authorization: Bearer <token>
```

### Endpoints principaux

#### `POST /api/v1/users/register`
Créer un compte utilisateur.

#### `POST /api/v1/users/login`
S'authentifier.

#### `POST /api/v1/images/upload`
Uploader une image et obtenir la classification IA.

```bash
curl -X POST http://localhost:8000/api/v1/images/upload \
  -H "Authorization: Bearer <token>" \
  -F "file=@dechet.jpg" \
  -F "user_id=<uuid>"
```

**Réponse :**

```json
{
  "predicted_class": "plastic",
  "confidence": 0.94,
  "recyclable": true,
  "bac": "Bac jaune",
  "alt": "Déchetterie (plastiques volumineux)",
  "waste_type": "Plastique",
  "advice": "Tous les emballages plastiques sont acceptés.",
  "model_version": "yolov8n-cls-v1",
  "inference_ms": 12.4,
  "image_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "prediction_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"
}
```

#### `GET /api/v1/users/{id}/history`
Historique complet des analyses d'un utilisateur.

#### `GET /health`
```json
{ "status": "ok", "version": "1.0.0" }
```

La documentation Swagger complète est disponible sur **http://localhost:8000/docs**.

---

## Interface utilisateur

L'interface Streamlit propose :

- **Authentification** — inscription avec consentement RGPD, connexion avec "Se souvenir de moi"
- **Scanner** — upload d'image ou prise de photo directe (caméra), résultat coloré avec bac recommandé et conseils
- **Historique** — toutes les analyses persistées en base de données
- **Mon compte** — modification du profil, suppression du compte
- **Pages légales** — CGU et Politique de confidentialité conformes RGPD
- **Sidebar responsive** — navigation desktop et mobile

---

## Base de données

PostgreSQL 16 avec 5 tables :

```sql
users        — Comptes utilisateurs (id, first_name, last_name, email, password, role)
waste_types  — 11 clés de classes (battery, cardboard, ...)
waste_infos  — Infos de recyclage (type_name, recyclable, bac, alt, advice), liées à waste_infos
predictions  — Résultats d'inférence (confidence_score, created_at)
images       — Images uploadées liées aux users, predictions et waste_infos
```

### Migrations

```bash
# Via Docker
docker compose exec backend alembic upgrade head

# En local
cd app/backend && alembic upgrade head
```

---

## Monitoring

### Métriques exposées par le backend (`/metrics`)

| Métrique | Description |
|---|---|
| `http_requests_total` | Nombre de requêtes par endpoint et statut |
| `http_request_duration_seconds` | Latence p50 / p95 / p99 |
| `predictions_total` | Prédictions par classe |
| `model_confidence_avg` | Score de confiance moyen |

### Grafana

Les dashboards JSON sont versionnés dans `monitoring/grafana/dashboards/` et prêts à importer directement dans Grafana.

Accès : **http://localhost:3001** (admin / mot de passe dans .env)

#### Dashboard
Le dashboard inclut les panels suivants :

- **Temps de réponse moyen :** Permet de suivre la performance globale de l’API et de détecter les ralentissements.

- **Latence P95 :** Indique la latence perçue par les utilisateurs dans les pires cas (95e percentile).

- **Taux d’erreur :** Pourcentage de requêtes échouées (codes 5xx). Indispensable pour détecter les anomalies.

- **Total des requêtes par seconde (RPS) :** Mesure la charge réelle sur l’API.

- **Top endpoints / nombre de requêtes :** Identifie les routes les plus sollicitées.

- **Usage CPU du process FastAPI :** Permet de détecter les saturations CPU.

- **Mémoire RAM utilisée :** Suivi de la consommation mémoire du process.

- **GC Collections per second :** Indique la fréquence des collectes du Garbage Collector Python (utile pour détecter des fuites mémoire ou une surcharge d’objets temporaires).

#### Système d’alertes
Un système complet d’alerting a été mis en place afin de garantir la stabilité, la performance et la disponibilité de l’API.
Toutes les alertes sont configurées dans Grafana et envoyées directement par email pour permettre une réaction rapide en cas d’incident :

- **Temps de réponse moyen trop élevé :** Déclenchée lorsque la moyenne des temps de réponse dépasse un seuil critique, indiquant un ralentissement global de l’API.

- **Latence P95 trop élevée :** Surveille les pires temps de réponse (95e percentile). Une hausse du P95 signale une dégradation perceptible par les utilisateurs.

- **Taux d’erreur élevé :** Alerte lorsque le pourcentage de réponses en erreur (5xx) dépasse un seuil défini.
Permet de détecter rapidement les anomalies applicatives.

- **CPU du process trop élevé :** Déclenchée lorsque le process FastAPI consomme une part excessive du CPU, indiquant une saturation potentielle.

- **RAM utilisée trop élevée :** Surveille la mémoire utilisée par le process. Une consommation anormale peut indiquer une fuite mémoire ou un risque d’OOM.

- **GC trop actif (risque de fuite mémoire) :** Alerte lorsque le Garbage Collector Python s’exécute trop fréquemment, ce qui peut révéler une création excessive d’objets ou un comportement anormal.

- **RPS trop bas (API KO ou bloquée) :** Déclenchée lorsque le nombre de requêtes par seconde chute en dessous d’un seuil minimal.
Permet de détecter une API potentiellement indisponible.

- **Endpoint principal anormalement inactif :** Surveille l’endpoint le plus utilisé, si celui-ci ne reçoit plus de trafic, cela peut indiquer une panne ou un blocage.

---

## CI/CD

### CI — déclenché sur chaque push et pull request

```
lint-backend    Ruff (E,F,W) + Mypy sur app/backend/
lint-frontend   Ruff + Mypy sur app/frontend/app.py
security        Bandit (backend + frontend) + Safety + Gitleaks
tests-backend   Pytest + coverage > 70% (DB mockée, sans PostgreSQL)
tests-frontend  Pytest sur app/frontend/tests/
notebooks       nbstripout (vérification outputs) + nbqa ruff
tests-ml        Pytest sur model/tests/ (preprocessing, pipeline)
release         Semantic Release → tag + CHANGELOG (merge main uniquement)
```

### CD — déclenché sur merge dans `main`

```
build     Build images Docker backend + frontend
push      Push vers GitHub Container Registry (GHCR)
trivy     Scan CVE des images Docker (HIGH, CRITICAL)
```

**GitHub Container Registry (GHCR)**

Ce projet publie automatiquement ses images Docker dans GitHub Container Registry.

🔗 **Accéder aux images :**  
https://github.com/YasmineSAYAD/waste-sorter/packages

Les images sont générées via GitHub Actions lors des push sur `main`.

---

## MLflow

Tous les runs d'entraînement sont tracés dans MLflow.

### Visualiser les runs en local

```bash
# Après téléchargement depuis Google Drive
mlflow ui --backend-store-uri model/mlruns
# Ouvrir http://localhost:5000
```

### Cycle de vie des modèles

```
None ──▶ Staging  (automatique si test_accuracy > seuil)
Staging ──▶ Production  (validation manuelle ou gate CI)
```

Le backend FastAPI charge toujours le modèle tagué **Production** au démarrage.

---

## Variables d'environnement

Copier `.env.example` et renseigner les valeurs :

```env
# PostgreSQL
POSTGRES_USER=waste_sorter
POSTGRES_PASSWORD=votre_mot_de_passe
POSTGRES_DB=waste_sorter_db
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# Application
SECRET_KEY=votre_cle_secrete
CORS_ORIGINS=["http://localhost:8501"]
UPLOAD_DIR=uploads
MAX_IMAGE_SIZE_MB=10

# Modèle ML
MODEL_PATH=model/saved/yolo_best.pt
MODEL_VERSION=yolov8n-cls-v1
CONFIDENCE_THRESHOLD=0.5

# Grafana
GF_SECURITY_ADMIN_PASSWORD=admin
```

> ⚠️ Ne jamais commiter le fichier `.env`. Il est listé dans `.gitignore`.

---

## Conformité RGPD

waste-sorter est conforme au **Règlement Général sur la Protection des Données (UE 2016/679)** :

- **Consentement explicite** requis à la création de compte
- **Minimisation des données** — seules les données nécessaires sont collectées
- **Droit d'accès, rectification, effacement, portabilité** disponibles depuis l'interface
- **Suppression en cascade** — la suppression d'un compte efface toutes les données associées
- **Mots de passe** hashés avec bcrypt, jamais stockés en clair
- **Données en transit** chiffrées via HTTPS/TLS
- **Durées de conservation** définies et documentées dans la politique de confidentialité
- **Aucune donnée** vendue ou partagée avec des tiers

Les pages **CGU** et **Politique de confidentialité** sont accessibles directement depuis l'interface.

---

<div align="center">
  <p>Waste sorter pour un monde plus propre.</p>
  <p>
    <a href="http://localhost:8000/docs">API Docs</a> ·
    <a href="http://localhost:8501">Application</a> ·
    <a href="http://localhost:5000">MLflow</a>
    <a href="http://localhost:3001">Monitoring</a>
  </p>
</div>