-- waste-sorter — PostgreSQL initial schema
-- Run automatically by Docker on first start

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── waste_types ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS waste_types (
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    label_key  VARCHAR(100) UNIQUE NOT NULL
);

-- ── waste_infos ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS waste_infos (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type_name     VARCHAR(100) NOT NULL,
    recyclable    BOOLEAN NOT NULL,
    bac           VARCHAR(100),
    alt           VARCHAR(255),
    advice        TEXT,
    waste_type_id UUID NOT NULL REFERENCES waste_types(id)
);

-- ── users ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    first_name VARCHAR(100) NOT NULL,
    last_name  VARCHAR(100) NOT NULL,
    email      VARCHAR(255) UNIQUE NOT NULL,
    password   VARCHAR(255) NOT NULL,
    role       VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ── predictions ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    confidence_score FLOAT NOT NULL,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

-- ── images ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS images (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_path    VARCHAR(500) NOT NULL,
    uploaded_at   TIMESTAMPTZ DEFAULT NOW(),
    user_id       UUID REFERENCES users(id),
    waste_info_id UUID REFERENCES waste_infos(id),
    prediction_id UUID REFERENCES predictions(id)
);

-- ── Seed waste_types ─────────────────────────────────────────────
INSERT INTO waste_types (label_key) VALUES
    ('battery'), ('cardboard'), ('electronic'), ('glass'), ('medical'),
    ('metal'), ('organic'), ('paper'), ('plastic'), ('textile'), ('trash')
ON CONFLICT (label_key) DO NOTHING;

-- ── Seed waste_infos ─────────────────────────────────────────────
INSERT INTO waste_infos (type_name, recyclable, bac, alt, advice, waste_type_id)
SELECT 'Pile/Batterie', true, 'Point de collecte', 'Supermarché ou magasin', 'Dépose-les dans un point de collecte, ne les jette jamais à la poubelle.', id FROM waste_types WHERE label_key = 'battery'
ON CONFLICT DO NOTHING;

INSERT INTO waste_infos (type_name, recyclable, bac, alt, advice, waste_type_id)
SELECT 'Carton', true, 'Bac jaune', NULL, 'À plier pour gagner de la place.', id FROM waste_types WHERE label_key = 'cardboard'
ON CONFLICT DO NOTHING;

INSERT INTO waste_infos (type_name, recyclable, bac, alt, advice, waste_type_id)
SELECT 'Électronique', true, 'Déchèterie', 'Magasin (reprise gratuite)', 'Le magasin reprend ton ancien appareil lors d’un achat, et accepte aussi les petits appareils sans obligation d’achat.', id FROM waste_types WHERE label_key = 'electronic'
ON CONFLICT DO NOTHING;

INSERT INTO waste_infos (type_name, recyclable, bac, alt, advice, waste_type_id)
SELECT 'Verre', true, 'Bac blanc', 'Borne à verre (verte)', 'Pense à enlever les bouchons et capsules avant de jeter.', id FROM waste_types WHERE label_key = 'glass'
ON CONFLICT DO NOTHING;

INSERT INTO waste_infos (type_name, recyclable, bac, alt, advice, waste_type_id)
SELECT 'Déchets médicaux', false, 'Pharmacie', 'Retour à un point de collecte médical','Les déchets médicaux tels que seringues, aiguilles, stylos injecteurs et pansements doivent être déposés dans des points de collecte spécialisés. Ne pas les jeter dans la poubelle classique. Si l''emballage est vide, il peut être jeté dans la poubelle appropriée (comme le plastique ou le papier), mais uniquement après avoir correctement éliminé le contenu et s''assurer qu''il ne présente aucun danger.', id FROM waste_types WHERE label_key = 'medical'
ON CONFLICT DO NOTHING;

INSERT INTO waste_infos (type_name, recyclable, bac, alt, advice, waste_type_id)
SELECT 'Métal', true, 'Bac jaune', 'Déchèterie (pour les gros objets métalliques)', 'Boîtes de conserve, canettes acceptées.', id FROM waste_types WHERE label_key = 'metal'
ON CONFLICT DO NOTHING;

INSERT INTO waste_infos (type_name, recyclable, bac, alt, advice, waste_type_id)
SELECT 'Déchets organiques', false, 'Bac marron', 'Compost maison', 'Le tri des biodéchets est désormais obligatoire (depuis 2024), si une solution est disponible près de chez toi.', id FROM waste_types WHERE label_key = 'organic'
ON CONFLICT DO NOTHING;

INSERT INTO waste_infos (type_name, recyclable, bac, alt, advice, waste_type_id)
SELECT 'Papier', true, 'Bac jaune', 'Point tri', 'Pas de papier gras ou mouillé.', id FROM waste_types WHERE label_key = 'paper'
ON CONFLICT DO NOTHING;

INSERT INTO waste_infos (type_name, recyclable, bac, alt, advice, waste_type_id)
SELECT 'Plastique', true, 'Bac jaune', 'Déchèterie (pour les plastiques durs ou volumineux)', 'Bouteilles, flacons et films plastiques se recyclent dans le bac jaune.', id FROM waste_types WHERE label_key = 'plastic'
ON CONFLICT DO NOTHING;

INSERT INTO waste_infos (type_name, recyclable, bac, alt, advice, waste_type_id)
SELECT 'Textile', true, 'Borne textile', 'Associations (Emmaüs, Croix-Rouge)', 'Même usés, à déposer propres et secs dans un sac fermé.', id FROM waste_types WHERE label_key = 'textile'
ON CONFLICT DO NOTHING;

INSERT INTO waste_infos (type_name, recyclable, bac, alt, advice, waste_type_id)
SELECT 'Déchets', false, 'Bac gris', 'Déchèterie', 'Utilise cette poubelle uniquement si tu ne peux pas trier autrement.', id FROM waste_types WHERE label_key = 'trash'
ON CONFLICT DO NOTHING;
