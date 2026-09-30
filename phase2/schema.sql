-- CarSpotters Phase 2 Schema

CREATE TABLE makes (
    id            SERIAL PRIMARY KEY,
    name          VARCHAR(100) NOT NULL UNIQUE,
    nhtsa_make_id INTEGER UNIQUE,
    country       VARCHAR(100)
);

CREATE TABLE models (
    id             SERIAL PRIMARY KEY,
    make_id        INTEGER NOT NULL REFERENCES makes(id) ON DELETE CASCADE,
    name           VARCHAR(200) NOT NULL,
    nhtsa_model_id INTEGER,
    vehicle_type   VARCHAR(20) NOT NULL DEFAULT 'car',  -- car | motorcycle
    UNIQUE (make_id, name)
);

CREATE TABLE car_catalog (
    id               SERIAL PRIMARY KEY,
    make_id          INTEGER NOT NULL REFERENCES makes(id),
    model_id         INTEGER NOT NULL REFERENCES models(id),
    year             SMALLINT NOT NULL,

    -- Production data
    production_count INTEGER,           -- NULL = unknown
    rarity_tier      VARCHAR(20) NOT NULL DEFAULT 'unknown',
    -- grey | green | blue | purple | orange | mythic | unknown
    rarity_source    VARCHAR(50),       -- 'rule_based' | 'manual' | 'wards'

    -- CV integration
    cv_label         VARCHAR(300),      -- make/model label as used by Phase 1 model

    -- Timestamps
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (make_id, model_id, year)
);

-- Trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION touch_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$;

CREATE TRIGGER car_catalog_updated_at
    BEFORE UPDATE ON car_catalog
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

-- Indexes for common lookups
CREATE INDEX idx_catalog_year        ON car_catalog (year);
CREATE INDEX idx_catalog_rarity      ON car_catalog (rarity_tier);
CREATE INDEX idx_catalog_make_model  ON car_catalog (make_id, model_id);
