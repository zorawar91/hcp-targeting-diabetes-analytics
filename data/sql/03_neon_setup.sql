-- =============================================================================
-- 03_neon_setup.sql
-- HCP Targeting & Brand Performance Analytics — Diabetes
-- Run this on your Neon database BEFORE importing CSVs.
-- Then import the two CSVs using the Neon dashboard or psql \COPY.
-- =============================================================================


-- ── 1. Main scoring table ─────────────────────────────────────────────────────
DROP TABLE IF EXISTS hcp_targeting_scores CASCADE;

CREATE TABLE hcp_targeting_scores (
    npi                         BIGINT        PRIMARY KEY,
    last_name                   TEXT,
    first_name                  TEXT,
    credential                  TEXT,
    specialty                   TEXT,
    state                       TEXT,
    city                        TEXT,
    volume_decile               SMALLINT,
    growth_decile               SMALLINT,
    fills_2022                  NUMERIC,
    fills_2021                  NUMERIC,
    yoy_growth_pct              NUMERIC,
    drug_cost_2022              NUMERIC,
    drugs_2022                  SMALLINT,
    total_payment_usd           NUMERIC,
    payment_count               INT,
    opinion_leader_payments     NUMERIC,
    payment_score_normalised    NUMERIC,
    targeting_score             NUMERIC,
    segment                     TEXT
);

-- ── 2. Drug trends table (replaces the view on local) ────────────────────────
DROP TABLE IF EXISTS drug_trends_raw CASCADE;

CREATE TABLE drug_trends_raw (
    drug_class          TEXT,
    generic_name        TEXT,
    year                SMALLINT,
    prescribers         INT,
    total_fills         NUMERIC,
    total_cost_usd      NUMERIC,
    avg_fills_per_hcp   NUMERIC
);


-- ── 3. Views (match what app.py queries) ──────────────────────────────────────

-- v_drug_trends: aggregated by drug class + year (app queries SUM over this)
CREATE OR REPLACE VIEW v_drug_trends AS
SELECT
    drug_class,
    generic_name,
    year,
    prescribers,
    total_fills,
    total_cost_usd,
    avg_fills_per_hcp
FROM drug_trends_raw;

-- v_state_summary: state-level aggregation used by Territory tab
CREATE OR REPLACE VIEW v_state_summary AS
SELECT
    state,
    COUNT(*)                                        AS total_hcps,
    SUM(fills_2022)                                 AS total_fills_2022,
    AVG(targeting_score)                            AS avg_targeting_score,
    SUM(CASE WHEN segment = 'High Value'  THEN 1 ELSE 0 END) AS high_value_count,
    SUM(CASE WHEN segment = 'Growth'      THEN 1 ELSE 0 END) AS growth_count,
    SUM(CASE WHEN segment = 'Maintenance' THEN 1 ELSE 0 END) AS maintenance_count
FROM hcp_targeting_scores
GROUP BY state;


-- ── 4. Indexes for query performance ─────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_hts_state     ON hcp_targeting_scores(state);
CREATE INDEX IF NOT EXISTS idx_hts_specialty ON hcp_targeting_scores(specialty);
CREATE INDEX IF NOT EXISTS idx_hts_segment   ON hcp_targeting_scores(segment);
CREATE INDEX IF NOT EXISTS idx_hts_score     ON hcp_targeting_scores(targeting_score DESC);


-- =============================================================================
-- After running this script, import CSVs:
--
--   \COPY hcp_targeting_scores FROM 'data/hcp_targeting_scores.csv' CSV HEADER;
--   \COPY drug_trends_raw      FROM 'data/drug_trends.csv'          CSV HEADER;
--
-- Or use the Neon dashboard → Tables → Import CSV.
-- =============================================================================
