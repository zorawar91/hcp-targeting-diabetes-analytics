"""
HCP Targeting & Brand Performance Analytics — Diabetes
Commercial Intelligence Platform
Stack: Python · PostgreSQL · Streamlit · Plotly
Run:  python3 -m streamlit run app.py
"""

import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HCP Analytics | Diabetes",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── STATE NAMES ────────────────────────────────────────────────────────────────
STATE_NAMES = {
    "AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California",
    "CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia",
    "HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa",
    "KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland",
    "MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi",
    "MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire",
    "NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina",
    "ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania",
    "RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota","TN":"Tennessee",
    "TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington",
    "WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming","DC":"District of Columbia",
    "PR":"Puerto Rico","GU":"Guam","VI":"Virgin Islands",
}

def state_full(abbrev):
    return STATE_NAMES.get(str(abbrev).upper(), str(abbrev))

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
SEG_COLORS = {
    "High Value":   "#FF3B30",
    "Growth":       "#34C759",
    "Maintenance":  "#FF9500",
    "Deprioritise": "#8E8E93",
}
SEG_BG = {
    "High Value":   "#FFF0EF",
    "Growth":       "#EDFBF1",
    "Maintenance":  "#FFF8ED",
    "Deprioritise": "#F5F5F7",
}

APPLE_BLUE = "#0071E3"

CHART_LAYOUT = dict(
    paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
    font=dict(color="#1D1D1F",
              family='-apple-system,BlinkMacSystemFont,"SF Pro Display","Helvetica Neue",sans-serif'),
)

# ── HELPER FUNCTIONS ───────────────────────────────────────────────────────────

def recommended_action(row):
    gd  = int(row.get("growth_decile",  5) or 5)
    vd  = int(row.get("volume_decile",  5) or 5)
    kol = float(row.get("opinion_leader_payments", 0) or 0) > 0
    sp  = str(row.get("specialty", "") or "").lower()
    if kol:
        return "Advisory Board / Speaker invitation"
    elif gd >= 8 and vd >= 8:
        return "Detail GLP-1 + defend Rx share"
    elif gd >= 8:
        return "Detail + sample — GLP-1 Agonists"
    elif vd >= 8:
        return "Defend Rx share — SGLT-2 Inhibitors"
    elif "endo" in sp:
        return "DPP-4 add-on therapy education"
    elif gd >= 6:
        return "GLP-1 introduction call"
    else:
        return "Foundation call — Biguanides"


def loyalty_tier(row, full_df):
    p75   = full_df["fills_2022"].quantile(0.75)
    p25   = full_df["fills_2022"].quantile(0.25)
    fills = float(row.get("fills_2022") or 0)
    yoy   = float(row.get("yoy_growth_pct") or 0)
    if fills >= p75 and yoy > 0:
        return "Loyalist",     "#166534", "#EDFBF1"
    elif fills >= p25 and yoy >= -10:
        return "Intermittent", "#92400e", "#FFF8ED"
    elif fills > 0:
        return "Tourist",      "#991b1b", "#FFF0EF"
    else:
        return "Non-Rx",       "#6E6E73", "#F5F5F7"


def sim_calls(npi):
    try:
        seed = int(str(int(npi or 0))[-6:])
    except Exception:
        seed = 42
    rng      = np.random.RandomState(seed)
    types    = ["F2F Call","P2P Call","Virtual Meeting","Sample Drop","Email / Digital"]
    brands   = ["GLP-1 Agonists","SGLT-2 Inhibitors","DPP-4 Inhibitors","Biguanides","Sulfonylureas"]
    outcomes = ["Product detailed","Samples requested","Follow-up booked","Left materials","Event invited"]
    n, base  = int(rng.randint(3, 7)), 0
    rows = []
    for _ in range(n):
        base += int(rng.randint(18, 85))
        dt    = pd.Timestamp("2024-06-01") - pd.Timedelta(days=base)
        ctype = str(rng.choice(types))
        rows.append({
            "Date":    dt.strftime("%b %Y"),
            "Type":    ctype,
            "Mins":    str(int(rng.randint(5,18))) if "Email" not in ctype else "—",
            "Brand":   str(rng.choice(brands)),
            "Outcome": str(rng.choice(outcomes)),
        })
    return pd.DataFrame(rows)


def brand_recs(row):
    sp  = str(row.get("specialty","") or "").lower()
    vd  = int(row.get("volume_decile",  5) or 5)
    gd  = int(row.get("growth_decile",  5) or 5)
    kol = float(row.get("opinion_leader_payments", 0) or 0) > 0
    rows = []
    if gd >= 8:
        rows.append(["1","GLP-1 Agonists","Detail + Sample",f"Growth D{gd}/10 — accelerating momentum"])
    if vd >= 8:
        rows.append([str(len(rows)+1),"SGLT-2 Inhibitors","Defend Rx Share",f"Volume D{vd}/10 — protect prescriptions"])
    if kol:
        rows.append([str(len(rows)+1),"Speaker / Advisory","Advisory Board","KOL — industry engagement opportunity"])
    if "endo" in sp:
        rows.append([str(len(rows)+1),"DPP-4 Inhibitors","Sample + Educate","Endocrinologist — add-on therapy"])
    if "cardio" in sp:
        rows.append([str(len(rows)+1),"SGLT-2 Inhibitors","CV Outcome Data","Cardiologist — cardiovascular benefit"])
    if not rows:
        rows.append(["1","Biguanides","Educate + Build","Foundation diabetes therapy"])
    return pd.DataFrame(rows, columns=["#","Brand / Programme","Action","Rationale"])

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  #MainMenu, footer, header { visibility: hidden; }

  html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                 "SF Pro Text", "Helvetica Neue", Arial, sans-serif !important;
  }

  /* Background */
  .stApp { background: #F5F5F7; }

  /* ── Sidebar ── */
  .stSidebar { background: #1C1C1E !important; border-right: 1px solid #2C2C2E; }
  .stSidebar * { color: #F5F5F7 !important; }

  /* Selectbox / multiselect — force all internal text white */
  .stSidebar [data-testid="stSelectbox"] div,
  .stSidebar [data-testid="stSelectbox"] span,
  .stSidebar [data-testid="stSelectbox"] p,
  .stSidebar [data-baseweb="select"] div,
  .stSidebar [data-baseweb="select"] span,
  .stSidebar [data-baseweb="select"] input,
  .stSidebar [data-testid="stMultiSelect"] div,
  .stSidebar [data-testid="stMultiSelect"] span {
    color: #F5F5F7 !important;
    background-color: transparent !important;
  }
  /* Selectbox container background */
  .stSidebar [data-baseweb="select"] > div:first-child {
    background-color: #2C2C2E !important;
    border-color: #3A3A3C !important;
    border-radius: 10px !important;
  }
  /* Dropdown menu */
  .stSidebar [data-baseweb="popover"] * { color: #1D1D1F !important; }

  .stSidebar .stSelectbox label,
  .stSidebar .stSlider label,
  .stSidebar .stMultiSelect label {
    color: #8E8E93 !important; font-size: 0.67rem !important;
    font-weight: 600 !important; text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
  }
  .stSidebar hr { border-color: #2C2C2E !important; margin: 1rem 0 !important; }
  .stSidebar [data-testid="stToggle"] {
    margin-top: 1.1rem !important;
    padding: 0.6rem 0 0.2rem !important;
    border-top: 1px solid #2C2C2E !important;
  }
  .stSidebar [data-testid="stToggle"] p,
  .stSidebar [data-testid="stToggle"] label {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    color: #F5F5F7 !important;
  }
  .stSidebar [data-testid="stSlider"] { padding-bottom: 0.6rem !important; }

  /* ── KPI cards ── */
  div[data-testid="metric-container"] {
    background: #FFFFFF; border: none;
    border-radius: 16px; padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
  }
  div[data-testid="metric-container"] label {
    color: #8E8E93 !important; font-size: 0.7rem !important;
    font-weight: 600 !important; text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
  }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-size: 1.8rem !important; font-weight: 700 !important;
    color: #1D1D1F !important; letter-spacing: -0.02em !important;
  }

  /* ── Section labels ── */
  .sec {
    font-size: 0.63rem; font-weight: 700; color: #8E8E93;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 0.9rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid #E5E5EA;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: #E5E5EA; border-radius: 12px;
    padding: 3px; gap: 2px; border: none;
    box-shadow: none; margin-bottom: 1.2rem;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent; color: #6E6E73;
    border-radius: 10px; font-weight: 500; font-size: 0.84rem;
    padding: 0.48rem 1rem;
  }
  .stTabs [aria-selected="true"] {
    background: #FFFFFF !important; color: #1D1D1F !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.12) !important;
  }

  /* ── Insight strip ── */
  .insight {
    background: #EBF5FB; border: none;
    border-left: 3px solid #0071E3; border-radius: 12px;
    padding: 1rem 1.2rem; font-size: 0.82rem;
    color: #0051A2; line-height: 1.65; margin-top: 1rem;
  }
  .insight strong { color: #003D7A; }

  /* ── Buttons ── */
  .stDownloadButton > button {
    background: #0071E3 !important; color: white !important;
    border: none !important; border-radius: 980px !important;
    font-weight: 600 !important; font-size: 0.84rem !important;
    width: 100%; padding: 0.6rem 1.4rem !important;
    letter-spacing: -0.01em !important;
  }

  /* ── Tables ── */
  .stDataFrame {
    border-radius: 14px !important; border: none !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    overflow: hidden !important;
  }

  /* ── Profile badges ── */
  .badge {
    display: inline-block; padding: 3px 11px; border-radius: 980px;
    font-size: 0.67rem; font-weight: 600; letter-spacing: 0.03em;
    margin-right: 5px; margin-top: 5px;
  }
</style>
""", unsafe_allow_html=True)

# ── DATABASE ───────────────────────────────────────────────────────────────────
# Production: set DATABASE_URL in Streamlit Cloud secrets dashboard.
# Local: falls back to localhost PostgreSQL.
@st.cache_resource(show_spinner=False)
def get_conn():
    try:
        dsn = st.secrets["DATABASE_URL"]
    except Exception:
        dsn = "postgresql://postgres:newpassword123@localhost:5432/postgres"
    return psycopg2.connect(dsn)

@st.cache_data(ttl=600, show_spinner=False)
def load_hcp():
    return pd.read_sql("SELECT * FROM hcp_targeting_scores", get_conn())

@st.cache_data(ttl=600, show_spinner=False)
def load_drug_trends():
    return pd.read_sql("""
        SELECT drug_class, year,
               SUM(prescribers)    AS prescribers,
               SUM(total_fills)    AS total_fills,
               SUM(total_cost_usd) AS total_cost_usd
        FROM v_drug_trends
        GROUP BY drug_class, year ORDER BY drug_class, year
    """, get_conn())

with st.spinner("Loading data…"):
    df      = load_hcp()
    drug_df = load_drug_trends()

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.4rem 0 1rem'>
      <div style='width:46px;height:46px;background:#0071E3;border-radius:13px;
                  display:flex;align-items:center;justify-content:center;
                  font-size:1.3rem;margin:0 auto'>🎯</div>
      <div style='font-size:0.94rem;font-weight:700;color:#F5F5F7;margin-top:9px;
                  letter-spacing:-0.01em'>HCP Intelligence</div>
      <div style='font-size:0.68rem;color:#8E8E93;margin-top:2px'>Diabetes Portfolio</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    # State dropdown with full names
    state_abbrevs = sorted(df["state"].dropna().unique().tolist())
    state_options = ["🌎 All States"] + [f"{state_full(s)} ({s})" for s in state_abbrevs]
    sel_st_label  = st.selectbox("📍 State", state_options)
    st_val = None if sel_st_label == "🌎 All States" else sel_st_label.split("(")[-1].rstrip(")")

    specs  = ["All Specialties"] + sorted(df["specialty"].dropna().unique().tolist())
    sel_sp = st.selectbox("🏥 Specialty", specs)
    sp_val = None if sel_sp == "All Specialties" else sel_sp

    seg_sel = st.multiselect("🏷️ Segment",
        ["High Value","Growth","Maintenance","Deprioritise"],
        default=["High Value","Growth"])

    min_sc   = st.slider("⚡ Min Score", 0.0, 1.0, 0.5, 0.01)
    kol_only = st.toggle("⭐ KOL / Speaker Only", value=False)

    st.markdown("---")

    with st.expander("📊 Score Methodology"):
        st.markdown("""
        <div style='font-size:0.72rem;color:#F5F5F7;line-height:1.7'>
          <div style='font-weight:700;color:#FFFFFF;margin-bottom:0.5rem'>
            Composite Targeting Score
          </div>
          <div style='background:#2C2C2E;border-radius:10px;padding:10px 12px;margin-bottom:0.7rem;
                      font-family:monospace;font-size:0.68rem;color:#D1E8FF'>
            Score = (Vol_D × 0.40)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ (Growth_D × 0.40)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ (Pay_D × 0.20)
          </div>
          <div style='color:#8E8E93;font-size:0.67rem;margin-bottom:0.6rem'>
            Each component is ranked via <strong style='color:#D1E8FF'>NTILE(10)</strong>
            within specialty, producing deciles 1–10.
            Score is normalised 0–1.
          </div>
          <div style='border-top:1px solid #3A3A3C;padding-top:0.5rem;margin-top:0.2rem'>
            <div style='color:#AEAEB2;font-size:0.66rem;font-weight:700;
                        text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem'>
              Components
            </div>
            <div style='margin-bottom:0.3rem'>
              <span style='color:#0071E3;font-weight:700'>Vol (40%)</span>
              <span style='color:#8E8E93'> — 2022 Rx fills</span>
            </div>
            <div style='margin-bottom:0.3rem'>
              <span style='color:#34C759;font-weight:700'>Growth (40%)</span>
              <span style='color:#8E8E93'> — YoY fill growth 2021→2022</span>
            </div>
            <div>
              <span style='color:#FF9500;font-weight:700'>Payment (20%)</span>
              <span style='color:#8E8E93'> — CMS Open Payments received</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.66rem;color:#636366;line-height:2;padding:0.1rem 0'>
      <div style='color:#8E8E93;font-size:0.59rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:0.3rem'>Data Sources</div>
      CMS Medicare Part D 2021–22<br>
      CMS Open Payments 2021–22<br>
      NPPES NPI Registry 2026
      <div style='color:#8E8E93;font-size:0.59rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.1em;margin:0.75rem 0 0.3rem'>Pipeline</div>
      83M+ rows · PostgreSQL<br>
      227,455 HCPs scored
    </div>""", unsafe_allow_html=True)

# ── FILTER ─────────────────────────────────────────────────────────────────────
filt = df.copy()
if st_val:   filt = filt[filt["state"]     == st_val]
if sp_val:   filt = filt[filt["specialty"] == sp_val]
if seg_sel:  filt = filt[filt["segment"].isin(seg_sel)]
filt = filt[filt["targeting_score"] >= min_sc]
if kol_only: filt = filt[filt["opinion_leader_payments"] > 0]
filt = filt.sort_values("targeting_score", ascending=False).reset_index(drop=True)

# ── HERO ───────────────────────────────────────────────────────────────────────
terr_state = state_full(st_val) if st_val else "National"
terr_spec  = sp_val or "All Specialties"
st.markdown(f"""
<div style='background:#FFFFFF;border-radius:18px;padding:1.8rem 2.2rem;
            margin-bottom:1.2rem;box-shadow:0 2px 12px rgba(0,0,0,0.06)'>
  <div style='display:flex;align-items:center;gap:0.85rem;margin-bottom:0.5rem'>
    <div style='width:40px;height:40px;background:#0071E3;border-radius:11px;
                display:flex;align-items:center;justify-content:center;
                font-size:1.2rem;flex-shrink:0'>🎯</div>
    <div>
      <div style='font-size:1.5rem;font-weight:700;color:#1D1D1F;
                  letter-spacing:-0.03em;line-height:1.1'>
        HCP Targeting &amp; Brand Performance Analytics
      </div>
      <div style='font-size:0.8rem;color:#8E8E93;margin-top:3px;font-weight:400'>
        {terr_state} &nbsp;·&nbsp; {terr_spec} &nbsp;·&nbsp; {datetime.now().strftime('%d %B %Y')}
      </div>
    </div>
  </div>
  <div style='margin-top:0.8rem;display:flex;gap:6px;flex-wrap:wrap'>
    {"".join(f'<span style="background:#F5F5F7;color:#1D1D1F;padding:3px 12px;border-radius:980px;font-size:0.65rem;font-weight:600;letter-spacing:0.04em;border:1px solid #E5E5EA">{t}</span>' for t in ["💊 Diabetes Portfolio","PostgreSQL","227K HCPs","83M+ Rows","CMS 2021–2022","Python · Streamlit · Plotly"])}
  </div>
</div>""", unsafe_allow_html=True)

# ── KPIs ───────────────────────────────────────────────────────────────────────
sc     = filt["segment"].value_counts()
hv_n   = sc.get("High Value",  0)
gr_n   = sc.get("Growth",      0)
avg_s  = filt["targeting_score"].mean()
kols_n = (filt["opinion_leader_payments"] > 0).sum()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Diabetes Prescribers",   f"{len(filt):,}")
k2.metric("🔴 High Value",          f"{hv_n:,}")
k3.metric("🟢 Growth Opportunity",  f"{gr_n:,}")
k4.metric("Avg Targeting Score",    f"{avg_s:.3f}" if len(filt) > 0 and not np.isnan(avg_s) else "—")
k5.metric("⭐ KOLs / Speakers",     f"{kols_n:,}")

# ── SEGMENT LEGEND + DATA FRESHNESS ───────────────────────────────────────────
st.html("""
<div style="display:flex;align-items:center;justify-content:space-between;
            flex-wrap:wrap;gap:8px;margin:0.6rem 0 0.2rem">
  <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
    <span style="font-size:0.62rem;font-weight:700;color:#8E8E93;
                 text-transform:uppercase;letter-spacing:0.08em;margin-right:4px">
      Segment Key
    </span>
    <span style="background:#FFF0EF;color:#CC2200;padding:3px 11px;border-radius:980px;
                 font-size:0.65rem;font-weight:700;border:1px solid #FFCDD9">
      🔴 High Value — Vol ≥ D8 &amp; Growth ≥ D8
    </span>
    <span style="background:#EDFBF1;color:#1A7A35;padding:3px 11px;border-radius:980px;
                 font-size:0.65rem;font-weight:700;border:1px solid #C3F2D0">
      🟢 Growth — Growth ≥ D8, Vol &lt; D8
    </span>
    <span style="background:#FFF8ED;color:#CC7700;padding:3px 11px;border-radius:980px;
                 font-size:0.65rem;font-weight:700;border:1px solid #FFE4B2">
      🟠 Maintenance — Vol ≥ D8, Growth &lt; D8
    </span>
    <span style="background:#F5F5F7;color:#6E6E73;padding:3px 11px;border-radius:980px;
                 font-size:0.65rem;font-weight:600;border:1px solid #E5E5EA">
      ⚫ Deprioritise — below D8 on both
    </span>
  </div>
  <div style="font-size:0.62rem;color:#AEAEB2;white-space:nowrap">
    📅 Data: CMS Medicare Part D 2021–2022 · Open Payments 2022 · NPPES NPI Registry
  </div>
</div>
""")

# ── TODAY'S PRIORITIES ─────────────────────────────────────────────────────────
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
st.markdown('<div class="sec">Today\'s Top 5 — Highest Priority Diabetes Prescribers to Contact</div>',
            unsafe_allow_html=True)

if len(filt) == 0:
    st.info("No HCPs match your current filters.")
else:
    top5 = filt.head(5)
    cols = st.columns(5)
    for i, (col, (_, row)) in enumerate(zip(cols, top5.iterrows())):
        seg    = row.get("segment", "Deprioritise")
        sc_    = SEG_COLORS.get(seg, "#8E8E93")
        sc_bg  = SEG_BG.get(seg, "#F5F5F7")
        name   = f"Dr {row.get('last_name','')}, {str(row.get('first_name',''))[:1]}."
        spec   = str(row.get("specialty",""))[:28]
        loc    = f"{row.get('city','')}, {state_full(row.get('state',''))}"
        score  = float(row.get("targeting_score", 0))
        action = recommended_action(row)
        kol_mk = " ⭐" if float(row.get("opinion_leader_payments", 0) or 0) > 0 else ""
        yoy_v  = row.get("yoy_growth_pct", None)
        yoy_s  = (f"+{yoy_v:.0f}% YoY" if yoy_v >= 0 else f"{yoy_v:.0f}% YoY") if pd.notna(yoy_v) else ""

        with col:
            st.html(f"""
            <div style="background:#FFFFFF;border-radius:16px;padding:16px 18px;
                        height:100%;border-top:3px solid {sc_};
                        box-shadow:0 2px 10px rgba(0,0,0,0.06)">
              <div style="font-size:0.6rem;font-weight:700;color:#8E8E93;
                          text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px">
                Priority #{i+1}
              </div>
              <div style="font-size:0.9rem;font-weight:700;color:#1D1D1F;line-height:1.2">
                {name}{kol_mk}
              </div>
              <div style="font-size:0.72rem;color:#6E6E73;margin-top:3px">{spec}</div>
              <div style="font-size:0.68rem;color:#8E8E93;margin-top:1px">{loc}</div>
              <div style="display:flex;gap:5px;margin:10px 0 8px;flex-wrap:wrap">
                <span style="background:{sc_bg};color:{sc_};padding:2px 9px;border-radius:980px;
                             font-size:0.63rem;font-weight:700">{seg}</span>
                <span style="background:#F5F5F7;color:#1D1D1F;padding:2px 9px;border-radius:980px;
                             font-size:0.63rem;font-weight:700">{score:.3f}</span>
                {f'<span style="background:#F5F5F7;color:#6E6E73;padding:2px 9px;border-radius:980px;font-size:0.63rem">{yoy_s}</span>' if yoy_s else ''}
              </div>
              <div style="font-size:0.72rem;color:#0071E3;font-weight:600;
                          margin-top:4px;line-height:1.4">
                &#8594; {action}
              </div>
            </div>
            """)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋  Diabetes Call List",
    "📈  Market Intelligence",
    "🗺️  Territory Map",
    "⭐  Opinion Leaders",
    "🩺  HCP Profile",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — CALL LIST
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    col_main, col_side = st.columns([3, 2])

    with col_main:
        st.markdown('<div class="sec">Diabetes HCP Call List — ranked by targeting score</div>',
                    unsafe_allow_html=True)

        disp = filt.head(200).copy()
        disp["action"] = disp.apply(recommended_action, axis=1)
        disp["state_full"] = disp["state"].apply(state_full)

        show = disp[[
            "last_name","first_name","specialty","city","state_full",
            "fills_2022","yoy_growth_pct","targeting_score","action","segment"
        ]].copy()
        show.columns = [
            "Last","First","Specialty","City","State",
            "Fills 2022","YoY %","Score","Next Action","Segment"
        ]
        show.index = range(1, len(show)+1)

        show["Fills 2022"] = show["Fills 2022"].apply(
            lambda x: f"{x:,.0f}" if pd.notna(x) else "—")
        show["YoY %"] = show["YoY %"].apply(
            lambda x: f"+{x:.1f}%" if pd.notna(x) and x >= 0
            else (f"{x:.1f}%" if pd.notna(x) else "—"))
        show["Score"] = show["Score"].apply(lambda x: f"{x:.3f}")

        def style_row(row):
            styles = [""] * len(row)
            seg = row.get("Segment","")
            seg_map = {
                "High Value":   "background-color:#FFF0EF;color:#CC2200;font-weight:700",
                "Growth":       "background-color:#EDFBF1;color:#1A7A35;font-weight:700",
                "Maintenance":  "background-color:#FFF8ED;color:#CC7700;font-weight:600",
                "Deprioritise": "background-color:#F5F5F7;color:#6E6E73",
            }
            if seg in seg_map:
                idx = list(row.index).index("Segment")
                styles[idx] = seg_map[seg]
            # YoY colour
            yoy = row.get("YoY %","")
            if "+" in str(yoy):
                idx = list(row.index).index("YoY %")
                styles[idx] = "color:#1A7A35;font-weight:600"
            elif str(yoy).startswith("-"):
                idx = list(row.index).index("YoY %")
                styles[idx] = "color:#CC2200;font-weight:600"
            # Action colour
            idx = list(row.index).index("Next Action")
            styles[idx] = "color:#0071E3;font-weight:500"
            return styles

        styled = show.style.apply(style_row, axis=1)
        st.dataframe(styled, use_container_width=True, height=520)

        csv_out = filt[["npi","last_name","first_name","credential","specialty",
                         "state","city","fills_2022","yoy_growth_pct",
                         "total_payment_usd","targeting_score","segment"]].copy()
        csv_out["state"] = csv_out["state"].apply(state_full)
        csv_out["action"] = filt.apply(recommended_action, axis=1)
        csv_out.columns = ["NPI","Last Name","First Name","Credential","Specialty",
                           "State","City","Fills 2022","YoY Growth %",
                           "Total Payments $","Targeting Score","Segment","Next Action"]
        st.download_button(
            f"⬇️  Export Call List — {terr_state} ({len(filt):,} HCPs)",
            csv_out.to_csv(index=False),
            f"call_list_{terr_state.lower().replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv", use_container_width=True
        )

    with col_side:
        st.markdown('<div class="sec">Segment Breakdown</div>', unsafe_allow_html=True)
        sd = filt["segment"].value_counts().reset_index()
        sd.columns = ["Segment","Count"]
        fig_d = px.pie(sd, names="Segment", values="Count",
                       color="Segment", color_discrete_map=SEG_COLORS, hole=0.6)
        fig_d.update_traces(textposition="outside", textinfo="label+percent", textfont_size=11)
        fig_d.update_layout(**CHART_LAYOUT, height=260, showlegend=False,
                            margin=dict(t=5,b=5,l=5,r=5))
        st.plotly_chart(fig_d, use_container_width=True)

        st.markdown('<div class="sec">Score Distribution</div>', unsafe_allow_html=True)
        fig_h = px.histogram(filt, x="targeting_score", color="segment",
                             color_discrete_map=SEG_COLORS, nbins=25, barmode="stack",
                             labels={"targeting_score":"Score","segment":"Segment"})
        fig_h.update_layout(**CHART_LAYOUT, height=200, showlegend=False,
                            margin=dict(t=5,b=5,l=5,r=5),
                            xaxis=dict(gridcolor="#F5F5F7"),
                            yaxis=dict(gridcolor="#F5F5F7"))
        st.plotly_chart(fig_h, use_container_width=True)

        st.markdown('<div class="sec">Top Specialties — Diabetes Prescribers</div>', unsafe_allow_html=True)
        ts = filt.groupby("specialty").size().reset_index(name="Count").nlargest(8,"Count")
        ts["specialty"] = ts["specialty"].apply(lambda x: x[:28] + "…" if len(str(x)) > 28 else x)
        fig_ts = px.bar(ts, x="Count", y="specialty", orientation="h",
                        color="Count", color_continuous_scale=["#D1E8FF","#0071E3"],
                        text="Count",
                        labels={"specialty":"","Count":"HCPs"})
        fig_ts.update_traces(texttemplate="%{text:,}", textposition="outside",
                             textfont=dict(size=10, color="#6E6E73"))
        fig_ts.update_layout(**CHART_LAYOUT, height=290, coloraxis_showscale=False,
                             margin=dict(t=5,b=5,l=10,r=50),
                             xaxis=dict(gridcolor="#F5F5F7", showticklabels=False),
                             yaxis=dict(gridcolor="#F5F5F7", autorange="reversed",
                                        tickfont=dict(size=11)))
        st.plotly_chart(fig_ts, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — MARKET INTELLIGENCE
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    # Diabetes drug class reference
    st.html("""
    <div style="background:#FFFFFF;border-radius:16px;padding:1.2rem 1.6rem;
                margin-bottom:1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.05)">
      <div style="font-size:0.7rem;font-weight:700;color:#8E8E93;text-transform:uppercase;
                  letter-spacing:0.1em;margin-bottom:0.9rem">
        Diabetes Drug Class Reference — Oral & Injectable Antidiabetics (CMS Medicare Part D)
      </div>
      <div style="display:flex;gap:12px;flex-wrap:wrap">
        <div style="flex:1;min-width:150px;background:#EBF5FF;border-radius:12px;padding:10px 14px;
                    border-left:3px solid #0071E3">
          <div style="font-size:0.75rem;font-weight:700;color:#0071E3">GLP-1 Agonists</div>
          <div style="font-size:0.68rem;color:#3A3A3C;margin-top:3px">Ozempic · Mounjaro · Wegovy</div>
          <div style="font-size:0.65rem;color:#6E6E73;margin-top:2px">Fastest growing · Diabetes + obesity dual use</div>
        </div>
        <div style="flex:1;min-width:150px;background:#EDFBF1;border-radius:12px;padding:10px 14px;
                    border-left:3px solid #34C759">
          <div style="font-size:0.75rem;font-weight:700;color:#1A7A35">SGLT-2 Inhibitors</div>
          <div style="font-size:0.68rem;color:#3A3A3C;margin-top:3px">Jardiance · Farxiga · Invokana</div>
          <div style="font-size:0.65rem;color:#6E6E73;margin-top:2px">Cardio-renal protection · Second-line</div>
        </div>
        <div style="flex:1;min-width:150px;background:#FFF8ED;border-radius:12px;padding:10px 14px;
                    border-left:3px solid #FF9500">
          <div style="font-size:0.75rem;font-weight:700;color:#CC7700">DPP-4 Inhibitors</div>
          <div style="font-size:0.68rem;color:#3A3A3C;margin-top:3px">Januvia · Tradjenta · Onglyza</div>
          <div style="font-size:0.65rem;color:#6E6E73;margin-top:2px">Oral add-on · Low hypoglycaemia risk</div>
        </div>
        <div style="flex:1;min-width:150px;background:#FFF0EF;border-radius:12px;padding:10px 14px;
                    border-left:3px solid #FF3B30">
          <div style="font-size:0.75rem;font-weight:700;color:#CC2200">Sulfonylureas</div>
          <div style="font-size:0.68rem;color:#3A3A3C;margin-top:3px">Glipizide · Glimepiride · Glyburide</div>
          <div style="font-size:0.65rem;color:#6E6E73;margin-top:2px">Older class · Declining as GLP-1 grows</div>
        </div>
        <div style="flex:1;min-width:150px;background:#F5F0FF;border-radius:12px;padding:10px 14px;
                    border-left:3px solid #BF5AF2">
          <div style="font-size:0.75rem;font-weight:700;color:#7B2FBE">Biguanides</div>
          <div style="font-size:0.68rem;color:#3A3A3C;margin-top:3px">Metformin (generic)</div>
          <div style="font-size:0.65rem;color:#6E6E73;margin-top:2px">First-line standard of care · Highest volume</div>
        </div>
      </div>
    </div>
    """)

    # Key insight callout first
    if len(drug_df) > 0:
        try:
            yoy_tbl = drug_df.pivot(index="drug_class", columns="year",
                                    values="total_fills").reset_index()
            yoy_tbl.columns = ["Drug Class","Fills 2021","Fills 2022"]
            yoy_tbl["Growth %"] = ((yoy_tbl["Fills 2022"] - yoy_tbl["Fills 2021"])
                                   / yoy_tbl["Fills 2021"] * 100).round(1)
            top_class  = yoy_tbl.nlargest(1,"Growth %").iloc[0]
            top_pct    = top_class["Growth %"]
            top_name   = top_class["Drug Class"]
        except Exception:
            top_name, top_pct = "GLP-1 Agonists", 23.0

        st.html(f"""
        <div style="background:#FFFFFF;border-radius:16px;padding:1.4rem 1.8rem;
                    margin-bottom:1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);
                    display:flex;align-items:center;gap:1.5rem">
          <div style="font-size:2.4rem;font-weight:900;color:#0071E3;
                      letter-spacing:-0.03em;flex-shrink:0">+{top_pct:.0f}%</div>
          <div>
            <div style="font-size:1rem;font-weight:700;color:#1D1D1F">
              {top_name} is the fastest-growing diabetes drug class year-over-year
            </div>
            <div style="font-size:0.82rem;color:#6E6E73;margin-top:4px">
              Driven by dual diabetes + obesity indications. Prioritise these diabetes prescribers
              in your Growth segment before competitors establish relationships.
            </div>
          </div>
        </div>
        """)

    col_l, col_r = st.columns([3,2])

    with col_l:
        st.markdown('<div class="sec">Diabetes Rx Trends by Drug Class (2021 → 2022)</div>',
                    unsafe_allow_html=True)
        fig_tr = px.line(drug_df, x="year", y="total_fills", color="drug_class",
                         markers=True, line_shape="spline",
                         color_discrete_sequence=["#0071E3","#34C759","#FF9500","#FF3B30","#BF5AF2"],
                         labels={"total_fills":"Total Rx Fills","drug_class":"Drug Class","year":"Year"})
        fig_tr.update_traces(line_width=3, marker_size=10)
        fig_tr.update_layout(**CHART_LAYOUT, height=340,
                             margin=dict(t=5,b=5,l=5,r=5),
                             xaxis=dict(tickvals=[2021,2022], gridcolor="#F5F5F7"),
                             yaxis=dict(gridcolor="#F5F5F7"),
                             legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_tr, use_container_width=True)

    with col_r:
        st.markdown('<div class="sec">2022 Diabetes Rx Market Share</div>', unsafe_allow_html=True)
        s22 = (drug_df[drug_df["year"]==2022]
               .groupby("drug_class")["total_fills"].sum().reset_index())
        s22.columns = ["Drug Class","Fills"]
        fig_pie = px.pie(s22, names="Drug Class", values="Fills",
                         color_discrete_sequence=["#0071E3","#34C759","#FF9500","#FF3B30","#BF5AF2"],
                         hole=0.55)
        fig_pie.update_traces(textposition="outside", textinfo="percent+label", textfont_size=10)
        fig_pie.update_layout(**CHART_LAYOUT, height=260, showlegend=False,
                              margin=dict(t=5,b=5,l=5,r=5))
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown('<div class="sec">YoY Growth by Drug Class</div>', unsafe_allow_html=True)
        if len(drug_df) > 0:
            try:
                y2 = drug_df.pivot(index="drug_class", columns="year",
                                   values="total_fills").reset_index()
                y2.columns = ["Drug Class","Fills 2021","Fills 2022"]
                y2["Growth"] = ((y2["Fills 2022"] - y2["Fills 2021"])
                                / y2["Fills 2021"] * 100).round(1)
                y2["Fills 2021"] = y2["Fills 2021"].apply(lambda x: f"{x/1e6:.1f}M")
                y2["Fills 2022"] = y2["Fills 2022"].apply(lambda x: f"{x/1e6:.1f}M")
                y2["Growth %"] = y2["Growth"].apply(
                    lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%")
                st.dataframe(y2[["Drug Class","Fills 2021","Fills 2022","Growth %"]]
                             .sort_values("Growth", ascending=False),
                             use_container_width=True, hide_index=True)
            except Exception:
                pass

    st.markdown("---")
    st.markdown('<div class="sec">Avg YoY Rx Growth by Specialty — Top 20 Diabetes Specialties (min 50 HCPs)</div>',
                unsafe_allow_html=True)
    sg = (df.groupby("specialty")
          .agg(avg_growth=("yoy_growth_pct","mean"), n=("npi","count"))
          .reset_index().dropna().query("n >= 50").nlargest(20,"avg_growth"))
    fig_sg = px.bar(sg, x="avg_growth", y="specialty", orientation="h",
                    text="avg_growth", color="avg_growth",
                    color_continuous_scale=["#D1E8FF","#0071E3"],
                    labels={"avg_growth":"Avg YoY Growth %","specialty":""})
    fig_sg.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_sg.update_layout(**CHART_LAYOUT, height=420, coloraxis_showscale=False,
                         margin=dict(t=5,b=5,l=5,r=80),
                         xaxis=dict(gridcolor="#F5F5F7"),
                         yaxis=dict(autorange="reversed", gridcolor="#F5F5F7"))
    st.plotly_chart(fig_sg, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — TERRITORY
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    # Territory insight callout
    sa_pre = df.groupby("state").agg(
        high_value  =("segment", lambda x: (x=="High Value").sum()),
        growth      =("segment", lambda x: (x=="Growth").sum()),
        total_fills =("fills_2022","sum"),
        total_hcps  =("npi","count"),
    ).reset_index()
    top_hv_state   = sa_pre.nlargest(1,"high_value").iloc[0]
    top_fill_state = sa_pre.nlargest(1,"total_fills").iloc[0]
    st.html(f"""
    <div style="background:#FFFFFF;border-radius:16px;padding:1.2rem 1.6rem;
                margin-bottom:1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);
                display:flex;gap:2.5rem;flex-wrap:wrap;align-items:center">
      <div style="display:flex;align-items:center;gap:1.2rem;flex:1;min-width:220px">
        <div style="font-size:2.2rem;font-weight:900;color:#FF3B30;
                    letter-spacing:-0.03em;flex-shrink:0">
          {int(top_hv_state['high_value']):,}
        </div>
        <div>
          <div style="font-size:0.95rem;font-weight:700;color:#1D1D1F">
            High Value diabetes HCPs in {state_full(top_hv_state['state'])}
          </div>
          <div style="font-size:0.78rem;color:#6E6E73;margin-top:3px">
            Largest concentration of priority-1 diabetes prescribers nationally
          </div>
        </div>
      </div>
      <div style="width:1px;background:#F0F0F0;align-self:stretch"></div>
      <div style="display:flex;align-items:center;gap:1.2rem;flex:1;min-width:220px">
        <div style="font-size:2.2rem;font-weight:900;color:#0071E3;
                    letter-spacing:-0.03em;flex-shrink:0">
          {top_fill_state['total_fills']/1e6:.1f}M
        </div>
        <div>
          <div style="font-size:0.95rem;font-weight:700;color:#1D1D1F">
            Diabetes Rx fills in {state_full(top_fill_state['state'])} (2022)
          </div>
          <div style="font-size:0.78rem;color:#6E6E73;margin-top:3px">
            Highest total prescription volume of any US state
          </div>
        </div>
      </div>
    </div>
    """)

    ctrl, _ = st.columns([2,3])
    with ctrl:
        map_metric = st.selectbox("View by:", [
            "High Value HCPs",
            "Growth Opportunity HCPs",
            "Total Rx Fills 2022",
            "Avg Targeting Score",
        ])

    sa = df.groupby("state").agg(
        total_hcps  =("npi","count"),
        high_value  =("segment", lambda x: (x=="High Value").sum()),
        growth      =("segment", lambda x: (x=="Growth").sum()),
        total_fills =("fills_2022","sum"),
        avg_score   =("targeting_score","mean"),
    ).reset_index()

    mcol = {
        "High Value HCPs":         "high_value",
        "Growth Opportunity HCPs": "growth",
        "Total Rx Fills 2022":     "total_fills",
        "Avg Targeting Score":     "avg_score",
    }[map_metric]

    # Add full state name for hover
    sa["State Name"] = sa["state"].apply(state_full)

    fig_map = px.choropleth(
        sa, locations="state", locationmode="USA-states",
        color=mcol, scope="usa",
        color_continuous_scale=["#D1E8FF","#0071E3"],
        hover_name="State Name",
        hover_data={
            "high_value": True, "growth": True,
            "total_hcps": True, "avg_score": ":.3f", "state": False,
        },
        labels={"high_value":"High Value","growth":"Growth",
                "total_hcps":"Total HCPs","avg_score":"Avg Score",
                "total_fills":"Total Fills", mcol: map_metric}
    )
    fig_map.update_layout(
        height=480,
        geo=dict(bgcolor="#F5F5F7", lakecolor="#F5F5F7",
                 landcolor="#F0F0F0", showlakes=False),
        paper_bgcolor="#F5F5F7",
        margin=dict(t=5,b=5,l=5,r=5),
        coloraxis_colorbar=dict(title=dict(text=map_metric)),
        font=dict(color="#1D1D1F")
    )
    st.plotly_chart(fig_map, use_container_width=True)

    ca, cb, cc = st.columns(3)
    with ca:
        st.markdown('<div class="sec">Top States — High Value Diabetes HCPs</div>', unsafe_allow_html=True)
        t = sa.nlargest(10,"high_value")[["State Name","high_value","growth","total_hcps"]]
        t.columns = ["State","High Value","Growth","Total HCPs"]
        st.dataframe(t, use_container_width=True, hide_index=True)
    with cb:
        st.markdown('<div class="sec">Top States — Rx Volume</div>', unsafe_allow_html=True)
        t2 = sa.nlargest(10,"total_fills")[["State Name","total_fills","high_value","avg_score"]].copy()
        t2["total_fills"] = t2["total_fills"].apply(lambda x: f"{x/1e6:.1f}M")
        t2["avg_score"]   = t2["avg_score"].apply(lambda x: f"{x:.3f}")
        t2.columns = ["State","Total Fills","High Value","Avg Score"]
        st.dataframe(t2, use_container_width=True, hide_index=True)
    with cc:
        st.markdown('<div class="sec">Top States — Growth Opportunity</div>', unsafe_allow_html=True)
        t3 = sa.nlargest(10,"growth")[["State Name","growth","high_value","avg_score"]].copy()
        t3["avg_score"] = t3["avg_score"].apply(lambda x: f"{x:.3f}")
        t3.columns = ["State","Growth HCPs","High Value","Avg Score"]
        st.dataframe(t3, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — OPINION LEADERS
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    kol_df = df[df["opinion_leader_payments"] > 0].copy()
    if st_val: kol_df = kol_df[kol_df["state"]     == st_val]
    if sp_val: kol_df = kol_df[kol_df["specialty"] == sp_val]
    kol_df = kol_df.sort_values("total_payment_usd", ascending=False)

    # Insight callout
    if len(kol_df) > 0:
        top_kol = kol_df.iloc[0]
        st.html(f"""
        <div style="background:#FFFFFF;border-radius:16px;padding:1.2rem 1.6rem;
                    margin-bottom:1.2rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);
                    display:flex;align-items:center;gap:1.2rem">
          <div style="font-size:1.9rem;font-weight:900;color:#FF9500;
                      letter-spacing:-0.02em;flex-shrink:0">
            ${kol_df['total_payment_usd'].max():,.0f}
          </div>
          <div>
            <div style="font-size:0.95rem;font-weight:700;color:#1D1D1F">
              Highest industry payment — {top_kol.get('first_name','')} {top_kol.get('last_name','')},
              {top_kol.get('specialty','')}
            </div>
            <div style="font-size:0.8rem;color:#6E6E73;margin-top:3px">
              {state_full(top_kol.get('state',''))} &nbsp;·&nbsp;
              {len(kol_df):,} diabetes KOLs identified &nbsp;·&nbsp;
              {(kol_df['segment']=='High Value').sum():,} are High Value diabetes prescribers
            </div>
          </div>
        </div>
        """)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total KOLs",      f"{len(kol_df):,}")
    m2.metric("Avg Payment",     f"${kol_df['total_payment_usd'].mean():,.0f}" if len(kol_df) else "—")
    m3.metric("Max KOL Payment", f"${kol_df['total_payment_usd'].max():,.0f}"  if len(kol_df) else "—")
    m4.metric("High Value KOLs", f"{(kol_df['segment']=='High Value').sum():,}")

    st.markdown("---")
    col_kl, col_kc = st.columns([3,2])

    with col_kl:
        st.markdown('<div class="sec">Diabetes KOLs — Speaker Bureau & Advisory Board Candidates</div>',
                    unsafe_allow_html=True)
        kd = kol_df.head(100)[[
            "last_name","first_name","credential","specialty","state","city",
            "opinion_leader_payments","total_payment_usd","fills_2022","targeting_score","segment"
        ]].copy()
        kd["state"] = kd["state"].apply(state_full)
        kd.columns = ["Last","First","Cred","Specialty","State","City",
                      "Speaker Events","Total $","Fills 2022","Score","Segment"]
        kd["Total $"]    = kd["Total $"].apply(lambda x: f"${x:,.0f}")
        kd["Fills 2022"] = kd["Fills 2022"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "—")
        kd["Score"]      = kd["Score"].apply(lambda x: f"{x:.3f}")
        sk = kd.style.applymap(lambda v: {
            "High Value":  "background-color:#FFF0EF;color:#CC2200;font-weight:700",
            "Growth":      "background-color:#EDFBF1;color:#1A7A35;font-weight:700",
            "Maintenance": "background-color:#FFF8ED;color:#CC7700",
        }.get(v,""), subset=["Segment"])
        st.dataframe(sk, use_container_width=True, height=440)

    with col_kc:
        st.markdown('<div class="sec">Industry Payments by Specialty</div>',
                    unsafe_allow_html=True)
        ksp = (kol_df.groupby("specialty")
               .agg(total_pay=("total_payment_usd","sum"))
               .reset_index().nlargest(12,"total_pay"))
        fig_kol = px.bar(ksp, x="total_pay", y="specialty",
                         orientation="h", text="total_pay",
                         color="total_pay",
                         color_continuous_scale=["#D1E8FF","#0071E3"],
                         labels={"total_pay":"Total Payments ($)","specialty":""})
        fig_kol.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig_kol.update_layout(**CHART_LAYOUT, height=400, coloraxis_showscale=False,
                              margin=dict(t=5,b=5,l=5,r=120),
                              xaxis=dict(gridcolor="#F5F5F7"),
                              yaxis=dict(autorange="reversed", gridcolor="#F5F5F7"))
        st.plotly_chart(fig_kol, use_container_width=True)

        st.markdown("""
        <div class="insight">
        💡 <strong>Medical Affairs priority:</strong> KOLs with both high Rx volume and
        speaker/advisory engagement are priority candidates for Phase IV investigator
        programmes and brand ambassador initiatives.
        </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — HCP PROFILE
# ──────────────────────────────────────────────────────────────────────────────
with tab5:
    if len(filt) == 0:
        st.warning("No HCPs match the current filters.")
    else:
        st.markdown(
            '<div class="sec">HCP Profile Drilldown — filter in the sidebar, then select any HCP</div>',
            unsafe_allow_html=True)

        top_n = filt.head(200).reset_index(drop=True)

        # Two-column picker: name search col + quick-stats col
        pick_col, stat_col = st.columns([3, 2])
        with pick_col:
            labels = [
                f"{'🔴' if r['segment']=='High Value' else '🟢' if r['segment']=='Growth' else '🟠' if r['segment']=='Maintenance' else '⚫'}  "
                f"Dr {r['last_name']}, {str(r['first_name'])[:1]}.  {r['credential'] or ''}  —  "
                f"{str(r['specialty'])[:30]}  ·  Score {r['targeting_score']:.3f}"
                for _, r in top_n.iterrows()
            ]
            sel_idx = st.selectbox(
                "Select HCP (type to search by name or specialty):",
                range(len(labels)),
                format_func=lambda i: labels[i],
                key="profile_sel"
            )
        hcp = top_n.iloc[sel_idx]
        with stat_col:
            seg_  = hcp.get("segment","")
            sc_c  = SEG_COLORS.get(seg_,"#8E8E93")
            sc_bg = SEG_BG.get(seg_,"#F5F5F7")
            fills_ = hcp.get("fills_2022", None)
            yoy_   = hcp.get("yoy_growth_pct", None)
            st.html(f"""
            <div style="background:#F5F5F7;border-radius:12px;padding:10px 14px;
                        margin-top:4px;display:flex;gap:16px;flex-wrap:wrap">
              <div><div style="font-size:0.6rem;color:#8E8E93;font-weight:700;text-transform:uppercase">Segment</div>
                <span style="background:{sc_bg};color:{sc_c};padding:2px 9px;border-radius:980px;
                             font-size:0.72rem;font-weight:700">{seg_}</span></div>
              <div><div style="font-size:0.6rem;color:#8E8E93;font-weight:700;text-transform:uppercase">Score</div>
                <div style="font-size:0.9rem;font-weight:700;color:#1D1D1F">{hcp.get('targeting_score',0):.3f}</div></div>
              <div><div style="font-size:0.6rem;color:#8E8E93;font-weight:700;text-transform:uppercase">Fills 2022</div>
                <div style="font-size:0.9rem;font-weight:700;color:#1D1D1F">{f"{fills_:,.0f}" if pd.notna(fills_) else "—"}</div></div>
              <div><div style="font-size:0.6rem;color:#8E8E93;font-weight:700;text-transform:uppercase">YoY</div>
                <div style="font-size:0.9rem;font-weight:700;color:{'#1A7A35' if pd.notna(yoy_) and yoy_>=0 else '#CC2200'}">{f"+{yoy_:.1f}%" if pd.notna(yoy_) and yoy_>=0 else (f"{yoy_:.1f}%" if pd.notna(yoy_) else "—")}</div></div>
            </div>
            """)

        # Derived
        tier, tier_color, tier_bg = loyalty_tier(hcp, df)
        seg_col_  = SEG_COLORS.get(str(hcp.get("segment","")), "#8E8E93")
        seg_bg_   = SEG_BG.get(str(hcp.get("segment","")), "#F5F5F7")
        is_kol    = float(hcp.get("opinion_leader_payments",0) or 0) > 0
        first_n   = str(hcp.get("first_name","") or "")
        last_n    = str(hcp.get("last_name","")  or "")
        cred      = str(hcp.get("credential","") or "")
        spec      = str(hcp.get("specialty","")  or "N/A")
        city_s    = str(hcp.get("city","")  or "")
        state_s   = state_full(hcp.get("state",""))
        npi_s     = str(int(hcp.get("npi",0) or 0))
        score     = float(hcp.get("targeting_score",0) or 0)
        vd        = int(hcp.get("volume_decile",5)  or 5)
        gd        = int(hcp.get("growth_decile",5)  or 5)
        fills     = hcp.get("fills_2022",None)
        yoy_v     = hcp.get("yoy_growth_pct",None)
        total_pay = float(hcp.get("total_payment_usd",0) or 0)
        kol_pay   = float(hcp.get("opinion_leader_payments",0) or 0)
        initials  = f"{first_n[:1]}{last_n[:1]}".upper() or "HC"
        score_pct = score * 100
        cred_str  = f", {cred}" if cred else ""
        action    = recommended_action(hcp)
        seg_label = str(hcp.get("segment",""))

        seg_badge  = f'<span class="badge" style="background:{seg_bg_};color:{seg_col_}">{seg_label.upper()}</span>'
        tier_badge = f'<span class="badge" style="background:{tier_bg};color:{tier_color};border:1px solid {tier_color}30">{tier.upper()}</span>'
        kol_badge  = '<span class="badge" style="background:#FFF8ED;color:#CC7700">&#11088; KOL / SPEAKER</span>' if is_kol else ""

        # Profile card
        st.html(f"""
        <div style="background:#FFFFFF;border-radius:18px;padding:1.8rem 2rem;
                    margin-bottom:1.2rem;position:relative;overflow:hidden;
                    box-shadow:0 4px 16px rgba(0,0,0,0.07)">
          <div style="position:absolute;left:0;top:0;bottom:0;width:4px;background:{seg_col_}"></div>
          <div style="display:flex;align-items:flex-start;gap:1.4rem;padding-left:0.5rem">
            <div style="width:68px;height:68px;border-radius:50%;
                        background:linear-gradient(135deg,#0071E3,#40AAFF);
                        display:flex;align-items:center;justify-content:center;
                        font-size:1.5rem;font-weight:800;color:white;flex-shrink:0">
              {initials}
            </div>
            <div style="flex:1;min-width:0">
              <div style="font-size:1.3rem;font-weight:700;color:#1D1D1F;
                          letter-spacing:-0.02em">{first_n} {last_n}{cred_str}</div>
              <div style="color:#6E6E73;font-size:0.86rem;margin-top:3px">
                {spec} &nbsp;&middot;&nbsp; {city_s}, {state_s}
              </div>
              <div style="color:#8E8E93;font-size:0.7rem;margin-top:2px;
                          font-family:monospace">NPI: {npi_s}</div>
              <div style="margin-top:0.6rem">{seg_badge}{tier_badge}{kol_badge}</div>
              <div style="margin-top:0.7rem;font-size:0.78rem;color:#0071E3;font-weight:600">
                &#8594; {action}
              </div>
            </div>
            <div style="text-align:center;background:#F5F5F7;border-radius:14px;
                        padding:1rem 1.4rem;min-width:150px;flex-shrink:0">
              <div style="font-size:0.58rem;font-weight:700;color:#8E8E93;
                          text-transform:uppercase;letter-spacing:0.1em">Targeting Score</div>
              <div style="font-size:2.6rem;font-weight:800;color:{seg_col_};
                          line-height:1.1;margin:4px 0">{score:.3f}</div>
              <div style="background:#E5E5EA;border-radius:4px;height:6px;
                          margin:6px 0;overflow:hidden">
                <div style="background:{seg_col_};width:{score_pct:.0f}%;
                            height:100%;border-radius:4px"></div>
              </div>
              <div style="font-size:0.62rem;color:#6E6E73">
                Vol D{vd}/10 &nbsp;&middot;&nbsp; Grwth D{gd}/10
              </div>
              <div style="font-size:0.6rem;color:#8E8E93;margin-top:2px">
                Rank #{sel_idx+1} of {len(filt):,}
              </div>
            </div>
          </div>
        </div>
        """)

        # KPI row
        pm1, pm2, pm3, pm4, pm5 = st.columns(5)
        fills_str = f"{int(fills):,}" if pd.notna(fills) else "—"
        yoy_str   = (f"+{yoy_v:.1f}%" if yoy_v >= 0 else f"{yoy_v:.1f}%") if pd.notna(yoy_v) else "—"
        pm1.metric("Rx Fills 2022",  fills_str)
        pm2.metric("YoY Rx Growth",  yoy_str)
        pm3.metric("Total Payments", f"${total_pay:,.0f}" if total_pay > 0 else "No record")
        pm4.metric("Speaker Events", f"{int(kol_pay)} events" if kol_pay > 0 else "None")
        pm5.metric("Loyalty Tier",   tier)

        st.markdown("---")
        col_recs, col_gauge = st.columns([3,2])

        with col_recs:
            st.markdown('<div class="sec">Brand Priority Recommendations</div>',
                        unsafe_allow_html=True)
            recs_df = brand_recs(hcp)
            st.dataframe(recs_df.style.applymap(
                lambda v: "color:#0071E3;font-weight:600" if "Detail" in str(v)
                else ("color:#CC2200;font-weight:600" if "Defend" in str(v)
                else ("color:#CC7700;font-weight:600" if "Advisory" in str(v) else "")),
                subset=["Action"]
            ), use_container_width=True, hide_index=True, height=200)

        with col_gauge:
            st.markdown('<div class="sec">Score Components</div>', unsafe_allow_html=True)
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                x=["Volume\nDecile","Growth\nDecile","Payment\nScore","Composite"],
                y=[vd/10, gd/10, min(total_pay/50000,1.0), score],
                marker_color=["#0071E3","#34C759","#FF9500", seg_col_],
                text=[f"{v:.2f}" for v in [vd/10, gd/10, min(total_pay/50000,1.0), score]],
                textposition="outside", textfont=dict(size=12, color="#1D1D1F"),
            ))
            fig_comp.update_layout(
                **CHART_LAYOUT, height=220,
                margin=dict(t=30,b=5,l=5,r=5),
                yaxis=dict(range=[0,1.2], gridcolor="#F5F5F7", tickformat=".1f"),
                xaxis=dict(gridcolor="rgba(0,0,0,0)"), showlegend=False,
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("---")
        col_calls, col_brief = st.columns([3,2])

        with col_calls:
            st.markdown(
                '<div class="sec">Simulated Call History '
                '<span style="font-size:0.6rem;color:#8E8E93;font-weight:400;'
                'text-transform:none">— illustrative, based on prescribing profile</span></div>',
                unsafe_allow_html=True)
            calls_df = sim_calls(hcp.get("npi",42))
            st.dataframe(
                calls_df.style.applymap(
                    lambda v: "color:#0071E3;font-weight:600" if "F2F" in str(v) or "P2P" in str(v)
                    else ("color:#34C759;font-weight:600" if "Virtual" in str(v) else ""),
                    subset=["Type"]
                ),
                use_container_width=True, hide_index=True
            )

        with col_brief:
            growth_word = "accelerating" if gd >= 8 else ("stable" if gd >= 5 else "declining")
            action_detail = (
                f"As a KOL with <strong>${total_pay:,.0f}</strong> in industry engagements, "
                "prioritise advisory board and speaker programme outreach."
                if is_kol else
                f"<strong>{action}</strong> on next territory visit."
            )
            priority_label = {
                "High Value":   "🔴 MUST CALL — High Value HCP",
                "Growth":       "🟢 SCHEDULE — Growth opportunity",
                "Maintenance":  "🟡 MAINTAIN — Scheduled cadence",
                "Deprioritise": "⚪ LOW PRIORITY",
            }.get(seg_label, "")

            st.markdown(f"""
            <div class="insight" style="margin-top:0.2rem">
            <strong>Rep Action Brief</strong><br><br>
            <strong>{first_n} {last_n}</strong> is a <strong>{tier}</strong> prescriber
            with {growth_word} Rx trajectory (Vol D{vd}/10, Growth D{gd}/10).<br><br>
            {action_detail}<br><br>
            <strong>{priority_label}</strong>
            </div>
            """, unsafe_allow_html=True)

            profile_csv = pd.DataFrame([{
                "NPI": npi_s, "Name": f"{first_n} {last_n}", "Credential": cred,
                "Specialty": spec, "City": city_s, "State": state_s,
                "Fills 2022": fills, "YoY Growth %": yoy_v,
                "Volume Decile": vd, "Growth Decile": gd,
                "Total Payments": total_pay, "Speaker Events": kol_pay,
                "Targeting Score": score, "Segment": seg_label,
                "Loyalty Tier": tier, "Next Action": action,
            }])
            st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)
            st.download_button(
                "⬇️  Export HCP Profile",
                profile_csv.to_csv(index=False),
                f"hcp_profile_{last_n.lower()}_{npi_s[-4:]}.csv",
                "text/csv", use_container_width=True
            )

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#8E8E93;font-size:0.72rem;padding:0.5rem 0'>
  Built by <strong style='color:#1D1D1F'>Zoraawar Nandwal</strong> &nbsp;·&nbsp;
  Python · PostgreSQL · Streamlit · Plotly &nbsp;·&nbsp;
  CMS Medicare Part D + Open Payments 2021–2022 &nbsp;·&nbsp;
  <a href='https://github.com/zorawar91/hcp-targeting-diabetes-analytics'
     style='color:#0071E3;text-decoration:none;font-weight:500'>GitHub →</a>
</div>
""", unsafe_allow_html=True)
