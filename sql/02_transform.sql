-- =============================================================================
-- 02_transform.sql
-- HCP Targeting & Commercial Intelligence — Diabetes Portfolio
-- Builds the star schema from staging tables
-- Run in DBeaver AFTER load_data.py completes
-- =============================================================================


-- =============================================================================
-- STEP 1 — Populate dim_drug from the drug crosswalk
-- =============================================================================

INSERT INTO dim_drug (generic_name, brand_name, drug_class)
SELECT generic_name, brand_name, drug_class
FROM   drug_crosswalk
ON CONFLICT (generic_name, brand_name) DO NOTHING;

SELECT 'dim_drug populated' AS status, COUNT(*) AS rows FROM dim_drug;


-- =============================================================================
-- STEP 2 — Populate dim_hcp from NPPES staging
-- Only load HCPs who appear in the Part D prescriber data (inner join)
-- This keeps the table focused and manageable
-- =============================================================================

INSERT INTO dim_hcp (npi, last_name, first_name, credential, specialty, taxonomy_code, city, state)
SELECT DISTINCT
    h.npi,
    INITCAP(h.last_name)       AS last_name,
    INITCAP(h.first_name)      AS first_name,
    h.credential,
    p.prscrbr_type             AS specialty,
    h.taxonomy_code_1          AS taxonomy_code,
    INITCAP(h.city)            AS city,
    p.prscrbr_state_abrvtn     AS state
FROM stg_hcp h
INNER JOIN (
    SELECT DISTINCT prscrbr_npi, prscrbr_type, prscrbr_state_abrvtn
    FROM   stg_partd
) p ON h.npi = p.prscrbr_npi
ON CONFLICT (npi) DO NOTHING;

SELECT 'dim_hcp populated' AS status, COUNT(*) AS rows FROM dim_hcp;


-- =============================================================================
-- STEP 3 — Populate fact_prescriptions
-- Filter to diabetes drugs using the crosswalk (generic name match)
-- =============================================================================

INSERT INTO fact_prescriptions
    (npi, generic_name, brand_name, drug_class, year,
     tot_claims, tot_30day_fills, tot_day_supply, tot_drug_cost, tot_beneficiaries)
SELECT
    p.prscrbr_npi,
    p.gnrc_name,
    p.brnd_name,
    dc.drug_class,
    p.year::INT,
    NULLIF(p.tot_clms,        '')::NUMERIC,
    NULLIF(p.tot_30day_fills, '')::NUMERIC,
    NULLIF(p.tot_day_suply,   '')::NUMERIC,
    NULLIF(p.tot_drug_cst,    '')::NUMERIC,
    NULLIF(p.tot_benes,       '')::NUMERIC
FROM stg_partd p
INNER JOIN (
    -- Match on the start of the generic name (CMS sometimes appends strength/form)
    SELECT DISTINCT generic_name, drug_class
    FROM   drug_crosswalk
) dc ON LOWER(p.gnrc_name) LIKE LOWER(dc.generic_name) || '%'
WHERE p.prscrbr_npi IS NOT NULL
  AND p.prscrbr_npi != '';

SELECT 'fact_prescriptions populated' AS status, COUNT(*) AS rows FROM fact_prescriptions;

-- Quick sense-check: rows per year
SELECT year, COUNT(*) AS rows, SUM(tot_30day_fills) AS total_fills
FROM   fact_prescriptions
GROUP  BY year
ORDER  BY year;


-- =============================================================================
-- STEP 4 — Populate fact_payments
-- Filter to physician recipients with a valid NPI
-- =============================================================================

INSERT INTO fact_payments
    (npi, company_name, payment_amount, payment_nature, drug_name, drug_category, program_year)
SELECT
    covered_recipient_npi,
    submitting_applicable_manufacturer_or_applicable_gpo_name,
    NULLIF(total_amount_of_payment_usdollars, '')::NUMERIC,
    nature_of_payment_or_transfer_of_value,
    name_of_drug_or_biological_or_device_or_medical_supply_1,
    product_category_or_therapeutic_area_1,
    NULLIF(program_year, '')::INT
FROM stg_payments
WHERE covered_recipient_type = 'Covered Recipient Physician'
  AND covered_recipient_npi IS NOT NULL
  AND covered_recipient_npi != ''
  AND NULLIF(total_amount_of_payment_usdollars, '')::NUMERIC > 0;

SELECT 'fact_payments populated' AS status, COUNT(*) AS rows FROM fact_payments;

-- Quick sense-check: payments per year
SELECT program_year, COUNT(*) AS payments, ROUND(SUM(payment_amount)::NUMERIC, 0) AS total_usd
FROM   fact_payments
GROUP  BY program_year
ORDER  BY program_year;


-- =============================================================================
-- STEP 5 — Quick data quality checks
-- =============================================================================

-- How many unique HCPs prescribed diabetes drugs?
SELECT COUNT(DISTINCT npi) AS diabetes_prescribers FROM fact_prescriptions;

-- Top 5 drugs by total 30-day fills
SELECT generic_name, drug_class,
       ROUND(SUM(tot_30day_fills)::NUMERIC, 0) AS total_fills
FROM   fact_prescriptions
GROUP  BY generic_name, drug_class
ORDER  BY total_fills DESC
LIMIT  10;

-- Top 5 states by prescribers
SELECT h.state, COUNT(DISTINCT f.npi) AS prescribers
FROM   fact_prescriptions f
JOIN   dim_hcp h ON f.npi = h.npi
GROUP  BY h.state
ORDER  BY prescribers DESC
LIMIT  5;
