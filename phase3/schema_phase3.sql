-- CarSpotters Phase 3 schema: the user/gameplay side.
-- Phase 2 owns makes / models / car_catalog; this adds users + their spots.

CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    username      VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    xp            INTEGER NOT NULL DEFAULT 0,
    level         INTEGER NOT NULL DEFAULT 1,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- A user's spotted cars (the "CarDex").
CREATE TABLE IF NOT EXISTS user_collection (
    id            SERIAL PRIMARY KEY,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    make          VARCHAR(100),
    model         VARCHAR(200),
    year          SMALLINT,
    rarity_tier   VARCHAR(20),
    confidence    REAL,
    photo_path    VARCHAR(300) NOT NULL,
    phash         VARCHAR(64),
    lat           DOUBLE PRECISION,
    lng           DOUBLE PRECISION,
    spotted_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (user_id, phash)          -- cheap dup-photo guard (pHash)
);

-- Global firehose of sightings, for future leaderboards / heatmaps.
CREATE TABLE IF NOT EXISTS sighting_log (
    id            SERIAL PRIMARY KEY,
    user_id       INTEGER REFERENCES users(id) ON DELETE SET NULL,
    make          VARCHAR(100),
    model         VARCHAR(200),
    rarity_tier   VARCHAR(20),
    lat           DOUBLE PRECISION,
    lng           DOUBLE PRECISION,
    spotted_at    TIMESTAMPTZ DEFAULT NOW()
);

-- The data flywheel: cars the CV model couldn't confidently recognize.
-- The user supplies the label; we accumulate these for the next training round.
CREATE TABLE IF NOT EXISTS unknown_submissions (
    id                   SERIAL PRIMARY KEY,
    user_id              INTEGER REFERENCES users(id) ON DELETE SET NULL,
    photo_path           VARCHAR(300) NOT NULL,
    phash                VARCHAR(64),
    predicted_label      VARCHAR(300),   -- model's best (low-confidence) guess
    predicted_confidence REAL,
    user_make            VARCHAR(100),   -- the human label = training signal
    user_model           VARCHAR(200),
    user_year            SMALLINT,
    status               VARCHAR(20) NOT NULL DEFAULT 'pending',
                         -- pending | reviewed | added_to_training
    created_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_collection_user ON user_collection (user_id);
CREATE INDEX IF NOT EXISTS idx_unknown_status  ON unknown_submissions (status);
