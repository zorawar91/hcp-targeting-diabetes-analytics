# HCP Targeting & Brand Performance Analytics — Diabetes

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://hcp-targeting-diabetes-analytics-zoraawar.streamlit.app/)

> **[🚀 Open Live App](https://hcp-targeting-diabetes-analytics-zoraawar.streamlit.app/)**

A production-grade commercial intelligence platform for diabetes portfolio management, built to replicate the core functionality of enterprise HCP targeting tools using open public data.

Built as a portfolio project demonstrating end-to-end data engineering, analytics, and BI skills across SQL, Python, and Streamlit.

---

## What it does

Sales reps and commercial teams in pharma need to know: **who do I call today, and why?**

This platform answers that for a diabetes drug portfolio by:
- Scoring 227,455 US physicians on a composite targeting model (Rx volume, YoY growth, industry engagement)
- Segmenting HCPs into actionable tiers: High Value, Growth, Maintenance, Deprioritise
- Surfacing territory intelligence, KOL identification, and per-HCP call planning
- Generating recommended next actions per HCP based on their decile profile

---

## Tech stack

| Layer | Tools |
|---|---|
| Data warehouse | PostgreSQL 15 (star schema, 83M+ rows) |
| ETL pipeline | Python · psycopg2 · pandas |
| Analytics | SQL window functions · NTILE deciles · YoY growth |
| Web app | Streamlit · Plotly Express · Plotly Graph Objects |
| Data sources | CMS Medicare Part D 2021–2022 · CMS Open Payments 2022 · NPPES NPI Registry |

---

## Scoring methodology

The composite targeting score is a weighted sum of three specialty-normalised decile ranks:

```
Score = (Volume_Decile  × 0.40)
      + (Growth_Decile  × 0.40)
      + (Payment_Decile × 0.20)
```

Each component uses `NTILE(10) OVER (PARTITION BY specialty)` to rank HCPs within their specialty peer group, producing deciles 1–10. The final score is normalised to 0–1.

### Segment definitions

| Segment | Rule |
|---|---|
| 🔴 High Value | Volume ≥ D8 **and** Growth ≥ D8 |
| 🟢 Growth | Growth ≥ D8 **and** Volume < D8 |
| 🟠 Maintenance | Volume ≥ D8 **and** Growth < D8 |
| ⚫ Deprioritise | Below D8 on both dimensions |

### Loyalty tiers

Derived from fills volume quantiles and YoY trajectory: **Loyalist / Intermittent / Tourist / Non-Rx**

---

## Data sources

All data is public, sourced from US government open data portals.

| Dataset | Rows | Description |
|---|---|---|
| CMS Medicare Part D Prescribers 2021 | ~27M | Prescription fills by NPI and drug class |
| CMS Medicare Part D Prescribers 2022 | ~28M | Same, following year for YoY growth |
| CMS Open Payments 2022 | ~12M | Industry payments to physicians (speaker fees, consulting, research) |
| NPPES NPI Registry | ~8M | HCP name, specialty, address, credentials |

Total pipeline: **83M+ rows** loaded into PostgreSQL via Python ETL.

---

## Dashboard features

**Tab 1 — Diabetes Call List**: Ranked HCP table with segment colouring, YoY growth, recommended next action per HCP. CSV export included.

**Tab 2 — Market Intelligence**: Diabetes drug class reference (GLP-1 / SGLT-2 / DPP-4 / Sulfonylureas / Biguanides), Rx trend lines 2021→2022, market share, YoY growth table, specialty growth ranking.

**Tab 3 — Territory Map**: Choropleth map of High Value HCPs / Growth HCPs / Rx volume / avg score by US state.

**Tab 4 — Opinion Leaders**: KOL identification using CMS Open Payments. Speaker bureau and advisory board candidates ranked by payment volume.

**Tab 5 — HCP Profile Drilldown**: Individual HCP view showing scoring breakdown, loyalty tier, simulated call history, brand recommendations, and rep action brief.

---

## How to run

**Prerequisites**: Python 3.10+, PostgreSQL 15, CMS datasets loaded.

```bash
# 1. Clone the repo
git clone https://github.com/zorawarsinghnandwal/hcp-targeting-diabetes-analytics.git
cd hcp-targeting-diabetes-analytics

# 2. Install dependencies
pip install streamlit psycopg2-binary pandas plotly numpy

# 3. Run the app (assumes PostgreSQL running locally)
streamlit run app.py
```

Database defaults: `host=localhost port=5432 dbname=postgres user=postgres`

---

## Prototype scope vs. production CRM

This prototype demonstrates core commercial CRM targeting logic against real public data. The table below maps what is built here against what a full Salesforce Health Cloud / Veeva CRM deployment would deliver.

| Capability | This prototype | Production Salesforce / Veeva |
|---|---|---|
| HCP scoring & segmentation | ✅ Composite score, NTILE deciles, 4-tier segmentation | ✅ Same, plus ML re-scoring on activity data |
| Next Best Action | ✅ Rule-based (segment + specialty + KOL logic) | Einstein NBA — model-scored, channel + message + timing |
| Call cadence engine | ✅ Segment-driven frequency rules | Veeva CRM auto-scheduling + territory routing |
| Activity tracking | Simulated (seeded by NPI) | Real-time call logging, CLM, sample management |
| Account hierarchy | HCPs as flat records | Contacts under Accounts (practice / hospital / IDN) |
| Territory management | Manual sidebar filter | Territory Management 2.0 — automated HCP-to-rep routing |
| Alerts | In-app at-risk + breakthrough cards | Push notifications — Slack, email, Veeva mobile |
| Closed-loop measurement | Not implemented | Pre/post call Rx lift analysis (contacted vs uncontacted cohort) |
| Journey staging | Segments only | Unaware → Aware → Trialing → Committed → Advocate |
| Role-based planning | ✅ Rep / Area / Regional / HoS views | Salesforce role hierarchy + visibility rules |
| KOL identification | ✅ CMS Open Payments | Same + HCP social graph (publications, speaker events) |

> Call activity data in this prototype is simulated — seeded deterministically by NPI for reproducibility. NBA recommendations are rule-based, not ML-driven. In a production environment these would be personalised per HCP interaction history via Einstein.

---

## Project structure

```
hcp-targeting-diabetes-analytics/
├── app.py                  # Streamlit web app (main deliverable)
├── sql/
│   ├── schema.sql          # Star schema + views
│   └── analysis/           # Exploratory SQL scripts
├── pipeline/
│   └── etl.py              # CMS data ingestion + scoring pipeline
└── README.md
```

---

## Live demo

**[https://hcp-targeting-diabetes-analytics-zoraawar.streamlit.app/](https://hcp-targeting-diabetes-analytics-zoraawar.streamlit.app/)**

Hosted on Streamlit Community Cloud · Database on Neon (PostgreSQL) · Free tier

---

## Author

**Zoraawar Nandwal** · [GitHub](https://github.com/zorawarsinghnandwal)

Built with: Python · PostgreSQL · Streamlit · Plotly · CMS Medicare Part D + Open Payments 2021–2022
