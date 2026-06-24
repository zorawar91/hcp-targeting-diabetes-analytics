-- =============================================================================
-- 01_schema.sql
-- HCP Targeting & Brand Performance Analytics — Diabetes
-- Creates staging tables and the star schema
-- Run this first in DBeaver against hcp_diabetes_db
-- =============================================================================

-- Drop tables if rebuilding from scratch (comment out if first run)
DROP TABLE IF EXISTS fact_prescriptions CASCADE;
DROP TABLE IF EXISTS fact_payments CASCADE;
DROP TABLE IF EXISTS dim_hcp CASCADE;
DROP TABLE IF EXISTS dim_drug CASCADE;
DROP TABLE IF EXISTS dim_geography CASCADE;
DROP TABLE IF EXISTS stg_partd CASCADE;
DROP TABLE IF EXISTS stg_payments CASCADE;
DROP TABLE IF EXISTS stg_hcp CASCADE;
DROP TABLE IF EXISTS drug_crosswalk CASCADE;


-- =============================================================================
-- STAGING TABLES (raw data, all TEXT to avoid load errors)
-- Populated by load_data.py
-- =============================================================================

CREATE TABLE stg_partd (
    prscrbr_npi             TEXT,
    prscrbr_last_org_name   TEXT,
    prscrbr_first_name      TEXT,
    prscrbr_city            TEXT,
    prscrbr_state_abrvtn    TEXT,
    prscrbr_type            TEXT,
    brnd_name               TEXT,
    gnrc_name               TEXT,
    tot_clms                TEXT,
    tot_30day_fills         TEXT,
    tot_day_suply           TEXT,
    tot_drug_cst            TEXT,
    tot_benes               TEXT,
    year                    TEXT
);

CREATE TABLE stg_payments (
    covered_recipient_type                                          TEXT,
    covered_recipient_npi                                          TEXT,
    covered_recipient_first_name                                   TEXT,
    covered_recipient_last_name                                    TEXT,
    covered_recipient_primary_type_1                               TEXT,
    submitting_applicable_manufacturer_or_applicable_gpo_name      TEXT,
    total_amount_of_payment_usdollars                              TEXT,
    nature_of_payment_or_transfer_of_value                        TEXT,
    name_of_drug_or_biological_or_device_or_medical_supply_1      TEXT,
    product_category_or_therapeutic_area_1                        TEXT,
    program_year                                                   TEXT
);

CREATE TABLE stg_hcp (
    npi             TEXT,
    entity_type     TEXT,
    last_name       TEXT,
    first_name      TEXT,
    credential      TEXT,
    state           TEXT,
    city            TEXT,
    taxonomy_code_1 TEXT
);


-- =============================================================================
-- DRUG CROSSWALK (diabetes drug list — generic → brand → drug class)
-- =============================================================================

CREATE TABLE drug_crosswalk (
    generic_name    TEXT,
    brand_name      TEXT,
    drug_class      TEXT
);

INSERT INTO drug_crosswalk (generic_name, brand_name, drug_class) VALUES
-- GLP-1 Agonists
('semaglutide',             'Ozempic',      'GLP-1 Agonist'),
('semaglutide',             'Wegovy',       'GLP-1 Agonist'),
('semaglutide',             'Rybelsus',     'GLP-1 Agonist'),
('liraglutide',             'Victoza',      'GLP-1 Agonist'),
('liraglutide',             'Saxenda',      'GLP-1 Agonist'),
('dulaglutide',             'Trulicity',    'GLP-1 Agonist'),
('tirzepatide',             'Mounjaro',     'GLP-1 Agonist'),
('tirzepatide',             'Zepbound',     'GLP-1 Agonist'),
('exenatide',               'Byetta',       'GLP-1 Agonist'),
('exenatide',               'Bydureon',     'GLP-1 Agonist'),
-- SGLT2 Inhibitors
('empagliflozin',           'Jardiance',    'SGLT2 Inhibitor'),
('dapagliflozin',           'Farxiga',      'SGLT2 Inhibitor'),
('canagliflozin',           'Invokana',     'SGLT2 Inhibitor'),
('ertugliflozin',           'Steglatro',    'SGLT2 Inhibitor'),
-- DPP-4 Inhibitors
('sitagliptin',             'Januvia',      'DPP-4 Inhibitor'),
('saxagliptin',             'Onglyza',      'DPP-4 Inhibitor'),
('linagliptin',             'Tradjenta',    'DPP-4 Inhibitor'),
('alogliptin',              'Nesina',       'DPP-4 Inhibitor'),
-- Insulins
('insulin glargine',        'Lantus',       'Insulin'),
('insulin glargine',        'Toujeo',       'Insulin'),
('insulin glargine',        'Basaglar',     'Insulin'),
('insulin aspart',          'NovoLog',      'Insulin'),
('insulin aspart',          'Fiasp',        'Insulin'),
('insulin lispro',          'Humalog',      'Insulin'),
('insulin lispro',          'Admelog',      'Insulin'),
('insulin degludec',        'Tresiba',      'Insulin'),
('insulin detemir',         'Levemir',      'Insulin'),
('insulin glargine-yfgn',   'Semglee',      'Insulin'),
-- Metformin
('metformin hcl',           'Glucophage',   'Biguanide'),
('metformin hcl',           'Fortamet',     'Biguanide'),
('metformin hcl er',        'Glumetza',     'Biguanide');


-- =============================================================================
-- STAR SCHEMA — DIMENSION TABLES
-- =============================================================================

CREATE TABLE dim_hcp (
    npi             VARCHAR(10) PRIMARY KEY,
    last_name       TEXT,
    first_name      TEXT,
    credential      TEXT,
    specialty       TEXT,
    taxonomy_code   VARCHAR(20),
    city            TEXT,
    state           CHAR(2)
);

CREATE TABLE dim_drug (
    drug_id         SERIAL PRIMARY KEY,
    generic_name    TEXT,
    brand_name      TEXT,
    drug_class      TEXT,
    UNIQUE(generic_name, brand_name)
);

CREATE TABLE dim_geography (
    state_abbr  CHAR(2) PRIMARY KEY,
    state_name  TEXT
);

INSERT INTO dim_geography VALUES
('AL','Alabama'),('AK','Alaska'),('AZ','Arizona'),('AR','Arkansas'),
('CA','California'),('CO','Colorado'),('CT','Connecticut'),('DE','Delaware'),
('FL','Florida'),('GA','Georgia'),('HI','Hawaii'),('ID','Idaho'),
('IL','Illinois'),('IN','Indiana'),('IA','Iowa'),('KS','Kansas'),
('KY','Kentucky'),('LA','Louisiana'),('ME','Maine'),('MD','Maryland'),
('MA','Massachusetts'),('MI','Michigan'),('MN','Minnesota'),('MS','Mississippi'),
('MO','Missouri'),('MT','Montana'),('NE','Nebraska'),('NV','Nevada'),
('NH','New Hampshire'),('NJ','New Jersey'),('NM','New Mexico'),('NY','New York'),
('NC','North Carolina'),('ND','North Dakota'),('OH','Ohio'),('OK','Oklahoma'),
('OR','Oregon'),('PA','Pennsylvania'),('RI','Rhode Island'),('SC','South Carolina'),
('SD','South Dakota'),('TN','Tennessee'),('TX','Texas'),('UT','Utah'),
('VT','Vermont'),('VA','Virginia'),('WA','Washington'),('WV','West Virginia'),
('WI','Wisconsin'),('WY','Wyoming'),('DC','District of Columbia'),
('PR','Puerto Rico'),('GU','Guam'),('VI','Virgin Islands');


-- =============================================================================
-- STAR SCHEMA — FACT TABLES
-- =============================================================================

CREATE TABLE fact_prescriptions (
    prescription_id     SERIAL PRIMARY KEY,
    npi                 VARCHAR(10),
    generic_name        TEXT,
    brand_name          TEXT,
    drug_class          TEXT,
    year                INT,
    tot_claims          NUMERIC,
    tot_30day_fills     NUMERIC,
    tot_day_supply      NUMERIC,
    tot_drug_cost       NUMERIC,
    tot_beneficiaries   NUMERIC
);

CREATE TABLE fact_payments (
    payment_id          SERIAL PRIMARY KEY,
    npi                 VARCHAR(10),
    company_name        TEXT,
    payment_amount      NUMERIC,
    payment_nature      TEXT,
    drug_name           TEXT,
    drug_category       TEXT,
    program_year        INT
);
