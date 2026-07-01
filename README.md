# HCP Targeting & Commercial Intelligence — Diabetes Portfolio

A production-grade commercial analytics platform for a diabetes drug portfolio, built on publicly available CMS Medicare data. Surfaces actionable HCP prioritisation, territory intelligence, and brand performance insights across five sales personas — from field rep to Head of Sales.

**Live app:** [hcp-targeting-diabetes-analytics-zoraawar.streamlit.app](https://hcp-targeting-diabetes-analytics-zoraawar.streamlit.app)

---

## What It Does

The platform transforms raw prescription claims data into decision-ready commercial intelligence:

- **HCP prioritisation** — Score and rank ~227K healthcare providers by volume, growth, and targeting potential
- **Territory coverage analysis** — Flag overdue accounts, quantify revenue at risk, identify white space
- **KOL intelligence** — Detect engagement gaps (advisory payments but declining Rx — a competitive threat signal)
- **Brand trend analysis** — Track prescriber base and fill volume by drug class over time
- **New entrant detection** — Surface rapidly-growing low-base prescribers before competitors do
- **Territory ROI modelling** — Excel model quantifying the financial impact of targeting decisions

---

## Data Sources

| Source | Volume | Key Use |
|--------|--------|---------|
| CMS Medicare Part D 2022 | 83M+ Rx rows → ~227K HCPs | Fill volume, drug cost, specialty, state |
| CMS Medicare Part D 2021 | Aggregated to HCP level | Prior-year fills for YoY growth |
| CMS Open Payments 2022 | ~12M payment records | KOL advisory & speaker fee identification |
| NPPES NPI Registry | ~8M providers | Specialty, geography enrichment |
| CDC T2D Prevalence | State-level estimates | White space market sizing |

All data is publicly available from the US Centers for Medicare & Medicaid Services. No proprietary, client, or employer data is used anywhere in this project.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Database | Neon serverless PostgreSQL with pgBouncer pooler |
| ORM / Query | SQLAlchemy + pandas |
| Application | Python 3.11 + Streamlit 1.35 |
| Deployment | Streamlit Community Cloud |
| Excel Modelling | openpyxl with formula-only model (no hardcoded values) |
| Version Control | Git + GitHub |

---

## Personas & Views

### Sales Rep
- Priority 1/2/3 HCP cards with pre-call brief generator
- Call due status: Overdue / Due Soon / OK per segment cadence
- Mark as Called simulation (session state — CRM write-back pattern)
- Next Best Action: 6-type rule-based recommendation engine

### Area Manager — 4 Tabs
| Tab | Content |
|-----|---------|
| My Team | Rep scorecards (P1 count, calls logged, overdue alerts) |
| Territory Gaps | Uncovered HCPs, revenue at risk KPIs |
| Strategic Accounts | KOL accounts + HV declining HCPs with pre-call briefs |
| Territory Health | Segment donut, at-risk table, conversion targets |

### Regional Manager — 4 Tabs
| Tab | Content |
|-----|---------|
| My AMs | Area Manager scorecards by state (Southwest region) |
| Regional Trends | YoY fill chart by state + segment migration signal cards |
| KOL Intelligence | Engagement gap (declining KOLs) vs healthy KOL split |
| White Space | T2D prevalence vs HCP coverage scatter — bubble-sized by HCP count |

### Head of Sales — 4 Tabs
| Tab | Content |
|-----|---------|
| National Scorecard | 4 national KPIs + region cards (RAG) |
| Franchise Accounts | Top 50 HCPs nationally with trend + at-risk alert |
| Brand Intelligence | Drug class prescriber base and fill trends over time |
| Strategic Priorities | Revenue at risk from HV decline + new entrant pipeline |

---

## Key Analytics Logic

```python
# HCP Segmentation
targeting_score = 0.5*(vol_decile/10) + 0.3*(growth_decile/10) + 0.2*(kol_flag)

# Segment assignment
if vol_decile >= 8 and growth_decile >= 8:    segment = "High Value"
elif growth_decile >= 8:                       segment = "Growth"
elif vol_decile >= 5:                          segment = "Maintenance"
else:                                          segment = "Deprioritise"

# Priority labels
if score >= 0.72:  Priority 1 (blue)
elif score >= 0.50: Priority 2 (amber)
else:              Priority 3 (grey)

# Call cadence (days between visits)
CALL_CADENCE = {"High Value": 28, "Growth": 42, "Maintenance": 84, "Deprioritise": 180}

# KOL engagement gap — competitive threat signal
engagement_gap = (opinion_leader_payments > 0) AND (yoy_growth_pct < -3)

# New entrant detection
new_entrant = (growth_decile >= 8) AND (fills_2021 <= 50) AND (fills_2022 >= 20)
```

---

## Repository Structure

```
hcp-targeting-diabetes-analytics/
├── app.py                              # Main Streamlit application (~3,400 lines)
├── requirements.txt                    # Python dependencies
├── .streamlit/
│   └── secrets.toml.example           # DB credentials template (real secrets not committed)
├── sql/
│   ├── 01_create_schema.sql            # Neon PostgreSQL schema
│   ├── 02_load_hcp_data.sql            # HCP aggregation from raw claims
│   ├── 03_views.sql                    # v_drug_trends and analytics views
│   └── week6_pipeline.sql              # ETL pipeline
├── HCP_Territory_ROI_Calculator.xlsx   # Week 10: standalone territory ROI model
├── AI_ML_Extension_Blueprint.docx      # Week 11: ML upgrade architecture document
├── HCP_Targeting_Case_Study.pdf        # Week 12: full project case study
└── platform_scope_and_crm_notes.txt   # Project notes and CRM integration design
```

---

## Deliverables

### Week 10 — Territory ROI Calculator (Excel)
Standalone Excel model quantifying the commercial impact of HCP targeting decisions.
- 5 sheets: Cover, Inputs, Segment ROI, Territory Compare, Scenario Analysis, Priority Matrix
- 102 formulas, zero errors — verified via LibreOffice headless recalculation
- Industry colour convention: Blue = inputs, Black = formulas, Green = cross-sheet links
- Key metric: call cost ($180) vs revenue per fill ($45) × segment-specific Rx lift rate

### Week 11 — AI/ML Extension Blueprint (Word Doc)
Substantive architecture document bridging rule-based analytics to production ML.
- Module 1: Gradient Boosting Regressor for momentum scoring — with feature schema and sklearn code
- Module 2: XGBoost churn classifier — label construction, class imbalance handling, intervention triggers
- Module 3: Multi-label NBA classifier — action taxonomy, model code, confidence ranking
- Dynamic cadence optimisation via Weibull survival analysis
- FastAPI scoring microservice + Salesforce Health Cloud field mapping
- SHAP explainability layer for rep-facing recommendations

### Week 12 — Case Study PDF
Full project story: problem, data, architecture, persona views, key insights, production path.

---

## Local Development

```bash
git clone https://github.com/zoraawar/hcp-targeting-diabetes-analytics.git
cd hcp-targeting-diabetes-analytics
pip install -r requirements.txt

# Add your Neon DB credentials to .streamlit/secrets.toml
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your DATABASE_URL

streamlit run app.py
```

---

## Project Context

This is a 12-week structured upskilling exercise in commercial analytics engineering for the pharmaceutical industry. The project covers:

**Weeks 1–5** — Problem framing, data sourcing, project charter  
**Weeks 6–7** — SQL pipeline: ETL, aggregation, analytics views  
**Weeks 8–9** — Streamlit application: multi-persona UI, scoring engine, visualisations  
**Week 10** — Financial modelling: territory ROI calculator in Excel  
**Week 11** — ML strategy: architecture blueprint for upgrading rule-based to ML  
**Week 12** — Capstone: case study, GitHub finalisation  

---

*All data used is publicly available from the US Centers for Medicare & Medicaid Services (data.cms.gov). This project does not use, reference, or infer any proprietary, confidential, or employer data.*
