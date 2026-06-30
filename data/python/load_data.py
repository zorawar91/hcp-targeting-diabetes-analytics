"""
load_data.py
HCP Targeting & Brand Performance Analytics — Diabetes
Loads CMS Part D, Open Payments, and NPPES data into PostgreSQL staging tables.

Run from terminal:
    pip3 install pandas sqlalchemy psycopg2-binary
    python3 python/load_data.py
"""

import pandas as pd
from sqlalchemy import create_engine
import time

# =============================================================================
# CONFIG — update password if you changed it
# =============================================================================
DB_URL  = "postgresql://postgres:newpassword123@localhost:5432/hcp_diabetes_db"
RAW     = "/Users/zorawarsinghnandwal/Documents/GitHub/hcp-targeting-diabetes-analytics/data/raw"

engine = create_engine(DB_URL)
print("Connected to hcp_diabetes_db\n")


# =============================================================================
# 1. CMS MEDICARE PART D — 2021 & 2022
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

frames = []
for year in ['2021', '2022']:
    path = f"{RAW}/Medicare_Part_D_Prescribers_by_Provider_and_Drug_{year}.csv"
    print(f"  Reading {year} file...")
    t = time.time()
    df = pd.read_csv(path, usecols=partd_cols, dtype=str, low_memory=False)
    df['year'] = year
    frames.append(df)
    print(f"  {year}: {len(df):,} rows loaded in {time.time()-t:.0f}s")

df_partd = pd.concat(frames, ignore_index=True)
df_partd.columns = [c.lower() for c in df_partd.columns]

print(f"\n  Writing {len(df_partd):,} total rows to stg_partd...")
t = time.time()
df_partd.to_sql('stg_partd', engine, if_exists='replace', index=False,
                chunksize=50000, method='multi')
print(f"  Done in {time.time()-t:.0f}s\n")


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

op_files = {
    '2021': f"{RAW}/OP_DTL_GNRL_PGYR2021_P01232026_01102026.csv",
    '2022': f"{RAW}/OP_DTL_GNRL_PGYR2022_P01232026_01102026.csv"
}

frames = []
for year, path in op_files.items():
    print(f"  Reading {year} file...")
    t = time.time()
    df = pd.read_csv(path, usecols=op_cols, dtype=str, low_memory=False,
                     encoding='latin-1')
    frames.append(df)
    print(f"  {year}: {len(df):,} rows loaded in {time.time()-t:.0f}s")

df_payments = pd.concat(frames, ignore_index=True)
df_payments.columns = [c.lower() for c in df_payments.columns]

print(f"\n  Writing {len(df_payments):,} total rows to stg_payments...")
t = time.time()
df_payments.to_sql('stg_payments', engine, if_exists='replace', index=False,
                   chunksize=50000, method='multi')
print(f"  Done in {time.time()-t:.0f}s\n")


# =============================================================================
# 3. NPPES NPI REGISTRY
# Reads in chunks — file is very large (~8GB). Only loads individual providers.
# =============================================================================
print("=" * 60)
print("Loading NPPES NPI Registry (reading in chunks, this takes a few minutes)...")
print("=" * 60)

nppes_path = f"{RAW}/NPPES_Data_Dissemination_June_2026_V2/npidata_pfile_20050523-20260607.csv"

nppes_cols = [
    'NPI',
    'Entity Type Code',
    'Provider Last Name (Legal Name)',
    'Provider First Name',
    'Provider Credential Text',
    'Provider Business Practice Location Address State Name',
    'Provider Business Practice Location Address City Name',
    'Healthcare Provider Taxonomy Code_1'
]

col_rename = {
    'NPI':                                                          'npi',
    'Entity Type Code':                                             'entity_type',
    'Provider Last Name (Legal Name)':                              'last_name',
    'Provider First Name':                                          'first_name',
    'Provider Credential Text':                                     'credential',
    'Provider Business Practice Location Address State Name':       'state',
    'Provider Business Practice Location Address City Name':        'city',
    'Healthcare Provider Taxonomy Code_1':                          'taxonomy_code_1'
}

CHUNK_SIZE = 100_000
total_rows = 0
first_chunk = True
t_start = time.time()

for i, chunk in enumerate(pd.read_csv(nppes_path, usecols=nppes_cols, dtype=str,
                                       low_memory=False, chunksize=CHUNK_SIZE,
                                       encoding='latin-1')):
    # Keep only individual providers (Entity Type Code = 1)
    chunk = chunk[chunk['Entity Type Code'] == '1'].copy()
    chunk.rename(columns=col_rename, inplace=True)

    mode = 'replace' if first_chunk else 'append'
    chunk.to_sql('stg_hcp', engine, if_exists=mode, index=False,
                 chunksize=10000, method='multi')
    first_chunk = False
    total_rows += len(chunk)

    if (i + 1) % 10 == 0:
        print(f"  Processed {(i+1)*CHUNK_SIZE:,} source rows → {total_rows:,} individual providers so far...")

print(f"\n  Done. {total_rows:,} individual HCPs loaded into stg_hcp in {time.time()-t_start:.0f}s\n")


# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 60)
print("All staging tables loaded successfully!")
print("=" * 60)

with engine.connect() as conn:
    for tbl in ['stg_partd', 'stg_payments', 'stg_hcp']:
        result = conn.execute(__import__('sqlalchemy').text(f"SELECT COUNT(*) FROM {tbl}"))
        print(f"  {tbl}: {result.scalar():,} rows")

print("\nNext step: run sql/02_transform.sql in DBeaver")
