-- =============================================================================
-- 03_segmentation.sql
-- HCP Targeting & Brand Performance Analytics — Diabetes
-- Analysis layer: annual Rx volumes, YoY growth, decile rankings
-- Run AFTER 02_transform.sql in psql or DBeaver (connected to postgres db)
-- =============================================================================


-- =============================================================================
-- VIEW 1: Annual Rx totals per HCP
-- Aggregates prescription records by NPI and year
-- =============================================================================
DROP VIEW IF EXISTS v_hcp_annual_rx CASCADE;

CREATE VIEW v_hcp_annual_rx AS
SELECT
    npi,
    year,
    SUM(tot_30day_fills)    AS total_fills,
    SUM(tot_claims)         AS total_claims,
    SUM(tot_drug_cost)      AS total_drug_cost,
    SUM(tot_beneficiaries)  AS total_beneficiaries,
    COUNT(DISTINCT generic_name) AS distinct_drugs_prescribed
FROM fact_prescriptions
GROUP BY npi, year;

-- Sense check
SELECT year, COUNT(*) AS hcps, ROUND(SUM(total_fills)::NUMERIC,0) AS total_fills
FROM v_hcp_annual_rx
GROUP BY year ORDER BY year;


-- =============================================================================
-- VIEW 2: YoY growth per HCP (2021 → 2022)
-- Uses a self-join to calculate year-on-year change in Rx fills
-- =============================================================================
DROP VIEW IF EXISTS v_hcp_yoy CASCADE;

CREATE VIEW v_hcp_yoy AS
SELECT
    curr.npi,
    curr.total_fills                                        AS fills_2022,
    prev.total_fills                                        AS fills_2021,
    curr.total_drug_cost                                    AS drug_cost_2022,
    curr.distinct_drugs_prescribed                          AS drugs_2022,
    -- YoY absolute change
    (curr.total_fills - prev.total_fills)                   AS fills_change,
    -- YoY % growth (guard against divide-by-zero)
    CASE
        WHEN prev.total_fills > 0
        THEN ROUND(((curr.total_fills - prev.total_fills) / prev.total_fills * 100)::NUMERIC, 1)
        ELSE NULL
    END                                                     AS yoy_growth_pct
FROM v_hcp_annual_rx curr
JOIN v_hcp_annual_rx prev
    ON curr.npi = prev.npi
    AND curr.year = 2022
    AND prev.year = 2021;

-- Sense check: growth distribution
SELECT
    CASE
        WHEN yoy_growth_pct >= 50  THEN 'High growth (>=50%)'
        WHEN yoy_growth_pct >= 10  THEN 'Moderate growth (10-50%)'
        WHEN yoy_growth_pct >= 0   THEN 'Flat (0-10%)'
        WHEN yoy_growth_pct < 0    THEN 'Declining'
        ELSE 'No prior year data'
    END AS growth_band,
    COUNT(*) AS hcps
FROM v_hcp_yoy
GROUP BY growth_band
ORDER BY hcps DESC;


-- =============================================================================
-- VIEW 3: Payment summary per HCP
-- Total industry payments and payment count from Open Payments
-- =============================================================================
DROP VIEW IF EXISTS v_hcp_payments CASCADE;

CREATE VIEW v_hcp_payments AS
SELECT
    npi,
    COUNT(*)                        AS payment_count,
    SUM(payment_amount)             AS total_payment_usd,
    COUNT(DISTINCT company_name)    AS distinct_companies,
    COUNT(DISTINCT payment_nature)  AS distinct_payment_types,
    -- Flag opinion leaders: speaker fees or advisory board payments
    SUM(CASE WHEN LOWER(payment_nature) LIKE '%speaker%'
              OR LOWER(payment_nature) LIKE '%advisory%'
             THEN 1 ELSE 0 END)     AS opinion_leader_payments
FROM fact_payments
GROUP BY npi;

-- Sense check
SELECT COUNT(*) AS hcps_with_payments,
       ROUND(AVG(total_payment_usd)::NUMERIC, 0) AS avg_payment_usd,
       MAX(total_payment_usd) AS max_payment_usd
FROM v_hcp_payments;


-- =============================================================================
-- VIEW 4: Decile rankings
-- NTILE(10) splits HCPs into 10 equal buckets
-- Decile 10 = highest volume/growth, Decile 1 = lowest
-- Ranked within specialty to account for prescribing volume differences
-- between e.g. endocrinologists and GPs
-- =============================================================================
DROP VIEW IF EXISTS v_hcp_deciles CASCADE;

CREATE VIEW v_hcp_deciles AS
SELECT
    y.npi,
    h.specialty,
    h.state,
    h.city,
    h.last_name,
    h.first_name,
    h.credential,
    -- Rx volume decile (within specialty)
    NTILE(10) OVER (
        PARTITION BY h.specialty
        ORDER BY y.fills_2022 ASC
    )                               AS volume_decile,
    -- Rx growth decile (within specialty, among HCPs with both years)
    NTILE(10) OVER (
        PARTITION BY h.specialty
        ORDER BY y.yoy_growth_pct ASC NULLS FIRST
    )                               AS growth_decile,
    -- Raw metrics
    y.fills_2022,
    y.fills_2021,
    y.yoy_growth_pct,
    y.drug_cost_2022,
    y.drugs_2022,
    -- Payment metrics (NULL if no payments on record)
    COALESCE(p.total_payment_usd, 0)        AS total_payment_usd,
    COALESCE(p.payment_count, 0)            AS payment_count,
    COALESCE(p.opinion_leader_payments, 0)  AS opinion_leader_payments
FROM v_hcp_yoy y
JOIN dim_hcp h ON y.npi = h.npi
LEFT JOIN v_hcp_payments p ON y.npi = p.npi;

-- Sense check: decile distribution (should be ~equal)
SELECT volume_decile, COUNT(*) AS hcps
FROM v_hcp_deciles
GROUP BY volume_decile
ORDER BY volume_decile;
