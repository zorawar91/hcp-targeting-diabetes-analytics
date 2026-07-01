-- =============================================================================
-- 04_targeting_score.sql
-- HCP Targeting & Commercial Intelligence — Diabetes Portfolio
-- Composite targeting score, segment labels, final call-planning table
-- Run AFTER 03_segmentation.sql
-- =============================================================================


-- =============================================================================
-- STEP 1: Normalise payment score (0 to 1 scale within specialty)
-- Payment engagement is the third pillar of the composite score
-- We normalise within specialty so endocrinologists aren't always top-scored
-- =============================================================================
DROP VIEW IF EXISTS v_payment_score CASCADE;

CREATE VIEW v_payment_score AS
SELECT
    npi,
    specialty,
    total_payment_usd,
    -- Min-max normalisation within specialty: 0 = no payments, 1 = highest payer
    CASE
        WHEN MAX(total_payment_usd) OVER (PARTITION BY specialty) > 0
        THEN ROUND(
            (total_payment_usd /
             MAX(total_payment_usd) OVER (PARTITION BY specialty))::NUMERIC, 4)
        ELSE 0
    END AS payment_score_normalised
FROM v_hcp_deciles;


-- =============================================================================
-- STEP 2: Composite targeting score
-- Formula:
--   Score = (volume_decile / 10 * 0.40)
--         + (growth_decile  / 10 * 0.40)
--         + (payment_score  *      0.20)
-- Range: 0.0 (lowest) to 1.0 (highest)
-- =============================================================================
DROP VIEW IF EXISTS v_hcp_scores CASCADE;

CREATE VIEW v_hcp_scores AS
SELECT
    d.npi,
    d.last_name,
    d.first_name,
    d.credential,
    d.specialty,
    d.state,
    d.city,
    d.volume_decile,
    d.growth_decile,
    d.fills_2022,
    d.fills_2021,
    d.yoy_growth_pct,
    d.drug_cost_2022,
    d.drugs_2022,
    d.total_payment_usd,
    d.payment_count,
    d.opinion_leader_payments,
    ps.payment_score_normalised,
    -- Composite score (0.0 – 1.0)
    ROUND((
        (d.volume_decile::NUMERIC / 10 * 0.40) +
        (d.growth_decile::NUMERIC / 10 * 0.40) +
        (ps.payment_score_normalised            * 0.20)
    )::NUMERIC, 4) AS targeting_score,
    -- Segment label
    CASE
        WHEN d.volume_decile >= 8 AND d.growth_decile >= 8
            THEN 'High Value'
        WHEN d.growth_decile >= 8 AND d.volume_decile < 8
            THEN 'Growth'
        WHEN d.volume_decile >= 8 AND d.growth_decile < 8
            THEN 'Maintenance'
        ELSE 'Deprioritise'
    END AS segment
FROM v_hcp_deciles d
JOIN v_payment_score ps
    ON d.npi = ps.npi
    AND d.specialty = ps.specialty;


-- =============================================================================
-- STEP 3: Materialise as a table for Power BI performance
-- Views recalculate on every query; a table is instant for dashboards
-- =============================================================================
DROP TABLE IF EXISTS hcp_targeting_scores;

CREATE TABLE hcp_targeting_scores AS
SELECT * FROM v_hcp_scores;

-- Add index on NPI for fast lookups
CREATE INDEX idx_hts_npi       ON hcp_targeting_scores(npi);
CREATE INDEX idx_hts_segment   ON hcp_targeting_scores(segment);
CREATE INDEX idx_hts_state     ON hcp_targeting_scores(state);
CREATE INDEX idx_hts_specialty ON hcp_targeting_scores(specialty);
CREATE INDEX idx_hts_score     ON hcp_targeting_scores(targeting_score DESC);

SELECT 'hcp_targeting_scores built' AS status, COUNT(*) AS total_hcps
FROM hcp_targeting_scores;


-- =============================================================================
-- STEP 4: Segment summary — this becomes your headline slide
-- =============================================================================
SELECT
    segment,
    COUNT(*)                                        AS hcps,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct_of_total,
    ROUND(AVG(targeting_score)::NUMERIC, 3)         AS avg_score,
    ROUND(AVG(fills_2022)::NUMERIC, 0)              AS avg_fills_2022,
    ROUND(AVG(yoy_growth_pct)::NUMERIC, 1)          AS avg_growth_pct,
    ROUND(SUM(total_payment_usd)::NUMERIC, 0)       AS total_payments_usd
FROM hcp_targeting_scores
GROUP BY segment
ORDER BY avg_score DESC;


-- =============================================================================
-- STEP 5: Top 20 HCPs overall — your call-planning list preview
-- =============================================================================
SELECT
    RANK() OVER (ORDER BY targeting_score DESC)     AS rank,
    npi,
    last_name || ', ' || first_name                 AS hcp_name,
    credential,
    specialty,
    state,
    ROUND(fills_2022::NUMERIC, 0)                   AS fills_2022,
    yoy_growth_pct                                  AS growth_pct,
    volume_decile,
    growth_decile,
    ROUND(targeting_score::NUMERIC, 3)              AS score,
    segment
FROM hcp_targeting_scores
ORDER BY targeting_score DESC
LIMIT 20;


-- =============================================================================
-- STEP 6: State-level summary for geographic heatmap (Power BI page 2)
-- =============================================================================
DROP VIEW IF EXISTS v_state_summary;

CREATE VIEW v_state_summary AS
SELECT
    state,
    COUNT(*)                                            AS total_hcps,
    COUNT(*) FILTER (WHERE segment = 'High Value')      AS high_value_hcps,
    COUNT(*) FILTER (WHERE segment = 'Growth')          AS growth_hcps,
    ROUND(SUM(fills_2022)::NUMERIC, 0)                  AS total_fills_2022,
    ROUND(AVG(yoy_growth_pct)::NUMERIC, 1)              AS avg_growth_pct,
    ROUND(AVG(targeting_score)::NUMERIC, 3)             AS avg_score
FROM hcp_targeting_scores
GROUP BY state;

SELECT * FROM v_state_summary ORDER BY total_fills_2022 DESC LIMIT 10;


-- =============================================================================
-- STEP 7: Drug class trend view for market performance page (Power BI page 1)
-- =============================================================================
DROP VIEW IF EXISTS v_drug_trends;

CREATE VIEW v_drug_trends AS
SELECT
    drug_class,
    generic_name,
    year,
    COUNT(DISTINCT npi)                             AS prescribers,
    ROUND(SUM(tot_30day_fills)::NUMERIC, 0)         AS total_fills,
    ROUND(SUM(tot_drug_cost)::NUMERIC, 0)           AS total_cost_usd,
    ROUND(AVG(tot_30day_fills)::NUMERIC, 1)         AS avg_fills_per_hcp
FROM fact_prescriptions
GROUP BY drug_class, generic_name, year;

SELECT drug_class, year,
       SUM(total_fills) AS fills,
       SUM(prescribers) AS prescribers
FROM v_drug_trends
GROUP BY drug_class, year
ORDER BY drug_class, year;
