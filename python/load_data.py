"""
load_data.py  (v2 — fast COPY version)
HCP Targeting & Brand Performance Analytics — Diabetes
Loads CMS Part D, Open Payments, and NPPES data into PostgreSQL staging tables.
Uses PostgreSQL COPY for fast bulk loading instead of row-by-row inserts.

Run from terminal:
    pip3 install pandas psycopg2-binary
    python3 python/load_data.py
"""

import pandas as pd
import psycopg2
import io
import time
import sys

# =============================================================================
# CONFIG
# =============================================================================
DB_PARAMS = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "postgres",
    "user":     "postgres",
    "password": "newpassword123"
}
RAW = "/Users/zorawarsinghnandwal/Documents/GitHub/hcp-targeting-diabetes-analytics/data/raw"

conn = psycopg2.connect(**DB_PARAMS)
conn.autocommit = True
cur  = conn.cursor()
cur.execute("SET search_path TO public;")
print("Connected to hcp_diabetes_db\n")


# =============================================================================
# HELPER — fast COPY from a pandas DataFrame
# =============================================================================
def fast_copy(df, table, cur, first=True):
    """Truncate table on first call, then COPY from in-memory CSV buffer."""
    if first:
        cur.execute(f"TRUNCATE TABLE {table}")
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep='')
    buf.seek(0)
    cur.copy_expert(
        f"COPY {table} FROM STDIN WITH (FORMAT CSV, NULL '')",
        buf
    )


# =============================================================================
# 1. CMS MEDICARE PART D — 2021 & 2022 (chunked read + COPY per chunk)
# =============================================================================
print("=" * 60)
print("Loading CMS Medicare Part D...")
print("=" * 60)

partd_cols = [
    'Prscrbr_NPI', 'Prscrbr_Last_Org_Name', 'Prscrbr_First_Name',
    'Prscrbr_City', 'Prscrbr_State_Abrvtn', 'Prscrbr_Type',
    'Brnd_Name', 'Gnrc_Name',
    'Tot_Clms', 'Tot_30day_Fills', 'Tot_Day_Suply', 'Tot_Drug_Cst', 'Tot_Benes'
]

first_write = True
total_partd  = 0
t0 = time.time()

for year in ['2021', '2022']:
    path = f"{RAW}/Medicare_Part_D_Prescribers_by_Provider_and_Drug_{year}.csv"
    print(f"  Reading {year}...", flush=True)
    year_rows = 0
    for chunk in pd.read_csv(path, usecols=partd_cols, dtype=str,
                              low_memory=False, chunksize=200_000):
        chunk['year'] = year
        chunk.columns  = [c.lower() for c in chunk.columns]
        # reorder to match stg_partd column order
        chunk = chunk[['prscrbr_npi','prscrbr_last_org_name','prscrbr_first_name',
                        'prscrbr_city','prscrbr_state_abrvtn','prscrbr_type',
                        'brnd_name','gnrc_name','tot_clms','tot_30day_fills',
                        'tot_day_suply','tot_drug_cst','tot_benes','year']]
        fast_copy(chunk, 'stg_partd', cur, first=first_write)
        first_write = False
        year_rows   += len(chunk)
        total_partd += len(chunk)
        print(f"    {total_partd:,} rows loaded so far...", flush=True)
    print(f"  {year} done: {year_rows:,} rows")

print(f"  stg_partd complete: {total_partd:,} rows in {time.time()-t0:.0f}s\n")


# =============================================================================
# 2. CMS OPEN PAYMENTS — 2021 & 2022
# =============================================================================
print("=" * 60)
print("Loading CMS Open Payments...")
print("=" * 60)

op_cols = [
    'Covered_Recipient_Type',
    'Covered_Recipient_NPI',
    'Covered_Recipient_First_Name',
    'Covered_Recipient_Last_Name',
    'Covered_Recipient_Primary_Type_1',
    'Submitting_Applicable_Manufacturer_or_Applicable_GPO_Name',
    'Total_Amount_of_Payment_USDollars',
    'Nature_of_Payment_or_Transfer_of_Value',
    'Name_of_Drug_or_Biological_or_Device_or_Medical_Supply_1',
    'Product_Category_or_Therapeutic_Area_1',
    'Program_Year'
]

op_col_lower = [
    'covered_recipient_type','covered_recipient_npi',
    'covered_recipient_first_name','covered_recipient_last_name',
    'covered_recipient_primary_type_1',
    'submitting_applicable_manufacturer_or_applicable_gpo_name',
    'total_amount_of_payment_usdollars',
    'nature_of_payment_or_transfer_of_value',
    'name_of_drug_or_biological_or_device_or_medical_supply_1',
    'product_category_or_therapeutic_area_1','program_year'
]

op_files = {
    '2021': f"{RAW}/OP_DTL_GNRL_PGYR2021_P01232026_01102026.csv",
    '2022': f"{RAW}/OP_DTL_GNRL_PGYR2022_P01232026_01102026.csv"
}

first_write  = True
total_op     = 0
t0 = time.time()

for year, path in op_files.items():
    print(f"  Reading {year}...", flush=True)
    year_rows = 0
    for chunk in pd.read_csv(path, usecols=op_cols, dtype=str, low_memory=False,
                              encoding='latin-1', chunksize=200_000):
        chunk.columns = op_col_lower
        fast_copy(chunk, 'stg_payments', cur, first=first_write)
        first_write = False
        year_rows   += len(chunk)
        total_op    += len(chunk)
        print(f"    {total_op:,} rows loaded so far...", flush=True)
    print(f"  {year} done: {year_rows:,} rows")

print(f"  stg_payments complete: {total_op:,} rows in {time.time()-t0:.0f}s\n")


# =============================================================================
# 3. NPPES NPI REGISTRY (chunked — large file)
# =============================================================================
print("=" * 60)
print("Loading NPPES (large file — printing every 500k rows)...")
print("=" * 60)

nppes_path = f"{RAW}/NPPES_Data_Dissemination_June_2026_V2/npidata_pfile_20050523-20260607.csv"

nppes_cols = [
    'NPI', 'Entity Type Code',
    'Provider Last Name (Legal Name)', 'Provider First Name',
    'Provider Credential Text',
    'Provider Business Practice Location Address State Name',
    'Provider Business Practice Location Address City Name',
    'Healthcare Provider Taxonomy Code_1'
]
nppes_col_lower = ['npi','entity_type','last_name','first_name','credential',
                   'state','city','taxonomy_code_1']

first_write  = True
total_nppes  = 0
t0 = time.time()

for i, chunk in enumerate(pd.read_csv(nppes_path, usecols=nppes_cols, dtype=str,
                                       low_memory=False, chunksize=200_000,
                                       encoding='latin-1')):
    chunk = chunk[chunk['Entity Type Code'] == '1'].copy()
    chunk.columns = nppes_col_lower
    fast_copy(chunk, 'stg_hcp', cur, first=first_write)
    first_write  = False
    total_nppes += len(chunk)
    if (i + 1) % 3 == 0:
        print(f"    {total_nppes:,} individual providers loaded...", flush=True)

print(f"  stg_hcp complete: {total_nppes:,} rows in {time.time()-t0:.0f}s\n")


# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 60)
print("All staging tables loaded!")
print("=" * 60)
for tbl in ['stg_partd', 'stg_payments', 'stg_hcp']:
    cur.execute(f"SELECT COUNT(*) FROM {tbl}")
    print(f"  {tbl}: {cur.fetchone()[0]:,} rows")

cur.close()
conn.close()
print("\nNext step: run sql/02_transform.sql in DBeaver")
