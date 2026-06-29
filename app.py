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
from datetime import datetime, timedelta
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

IQVIA_BLUE = "#003DA5"

CHART_LAYOUT = dict(
    paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
    font=dict(color="#1A2140",
              family='-apple-system,BlinkMacSystemFont,"SF Pro Display","Helvetica Neue",sans-serif'),
)

# ── CALL CADENCE ───────────────────────────────────────────────────────────────
CALL_CADENCE_DAYS = {
    "High Value":   28,   # Every 4 weeks  — defend Rx share
    "Growth":       42,   # Every 6 weeks  — convert & grow
    "Maintenance":  84,   # Every 12 weeks — keep warm
    "Deprioritise": 180,  # Every 6 months — minimal touch
}
KOL_CADENCE_DAYS = 21    # Every 3 weeks  — speaker / advisory board

def cadence_days(row):
    is_kol = float(row.get("opinion_leader_payments", 0) or 0) > 0
    return KOL_CADENCE_DAYS if is_kol else CALL_CADENCE_DAYS.get(
        str(row.get("segment", "Deprioritise")), 180)

def cadence_label(row):
    d = cadence_days(row)
    return {21:"Every 3 weeks",28:"Every 4 weeks",
            42:"Every 6 weeks",84:"Every 12 weeks",180:"Every 6 months"}.get(d, f"Every {d} days")

def sim_last_call(npi):
    """Deterministic seeded last-call date per NPI (3–90 days ago)."""
    try:
        seed = int(str(int(npi or 0))[-6:]) % (2**31)
    except Exception:
        seed = 42
    rng = np.random.RandomState(seed)
    days_ago = int(rng.randint(3, 91))
    return datetime.now() - timedelta(days=days_ago)

def call_due_status(row):
    """Returns (status_label, detail, color, bg_color)."""
    last     = sim_last_call(row.get("npi", 0))
    cadence  = cadence_days(row)
    days_since = (datetime.now() - last).days
    days_left  = cadence - days_since
    next_date  = (last + timedelta(days=cadence)).strftime("%-d %b")
    if days_left < 0:
        return "Overdue",  f"{abs(days_left)}d overdue",  "#FF3B30", "#FFF0EF"
    elif days_left <= 7:
        return "Due Soon", f"Due {next_date}",            "#FF9500", "#FFF8ED"
    else:
        return "On Track", f"Due {next_date}",            "#34C759", "#EDFBF1"

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

  /* ══ SIDEBAR: locked open, never collapses ══════════════════════════════ */
  /* Force the flex parent to never shrink the sidebar */
  [data-testid="stAppViewContainer"] {
    display: flex !important;
    flex-wrap: nowrap !important;
    overflow: visible !important;
  }
  /* Lock the sidebar itself */
  section[data-testid="stSidebar"] {
    flex: 0 0 22rem !important;
    min-width: 22rem !important;
    max-width: 22rem !important;
    width: 22rem !important;
    transform: translateX(0) !important;
    transition: none !important;
    display: flex !important;
    flex-direction: column !important;
    visibility: visible !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    position: relative !important;
  }
  section[data-testid="stSidebar"] > div {
    min-width: 22rem !important;
    width: 22rem !important;
  }
  /* Hide the collapse button — sidebar stays open */
  button[data-testid="stSidebarCollapseButton"],
  [data-testid="collapsedControl"],
  [data-testid="stSidebarCollapsedControl"],
  button[title="Collapse sidebar"], button[title="Close sidebar"],
  button[aria-label="Collapse sidebar"], button[aria-label="Close sidebar"] {
    display: none !important;
  }

  /* ── Main content: tight padding ── */
  [data-testid="stMainBlockContainer"] {
    padding-top: 1.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
  }

  html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                 "SF Pro Text", "Helvetica Neue", Arial, sans-serif !important;
  }

  /* Background */
  .stApp { background: #EEF3FC; }

  /* ── Sidebar ── */
  .stSidebar { background: #001F5B !important; border-right: none; }
  .stSidebar * { color: #E8F0FF !important; }

  /* Selectbox / multiselect */
  .stSidebar [data-testid="stSelectbox"] div,
  .stSidebar [data-testid="stSelectbox"] span,
  .stSidebar [data-testid="stSelectbox"] p,
  .stSidebar [data-baseweb="select"] div,
  .stSidebar [data-baseweb="select"] span,
  .stSidebar [data-baseweb="select"] input,
  .stSidebar [data-testid="stMultiSelect"] div,
  .stSidebar [data-testid="stMultiSelect"] span {
    color: #E8F0FF !important;
    background-color: transparent !important;
  }
  .stSidebar [data-baseweb="select"] > div:first-child {
    background-color: #0A3278 !important;
    border-color: #1A4EA0 !important;
    border-radius: 8px !important;
  }
  .stSidebar [data-baseweb="popover"] * { color: #1D1D1F !important; }

  .stSidebar .stSelectbox label,
  .stSidebar .stSlider label,
  .stSidebar .stMultiSelect label {
    color: #7B9AC0 !important; font-size: 0.67rem !important;
    font-weight: 600 !important; text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
  }
  .stSidebar hr { border-color: #0A3278 !important; margin: 1rem 0 !important; }
  .stSidebar [data-testid="stToggle"] {
    margin-top: 1.1rem !important;
    padding: 0.6rem 0 0.2rem !important;
    border-top: 1px solid #0A3278 !important;
  }
  .stSidebar [data-testid="stToggle"] p,
  .stSidebar [data-testid="stToggle"] label {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    color: #E8F0FF !important;
  }
  .stSidebar [data-testid="stSlider"] { padding-bottom: 0.6rem !important; }

  /* ── KPI cards ── */
  div[data-testid="metric-container"] {
    background: #FFFFFF; border: none;
    border-radius: 12px; padding: 1.2rem 1.4rem;
    box-shadow: 0 1px 6px rgba(0,31,91,0.08);
  }
  div[data-testid="metric-container"] label {
    color: #7B8EA0 !important; font-size: 0.7rem !important;
    font-weight: 600 !important; text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
  }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-size: 1.8rem !important; font-weight: 700 !important;
    color: #001F5B !important; letter-spacing: -0.02em !important;
  }

  /* ── Section labels ── */
  .sec {
    font-size: 0.63rem; font-weight: 700; color: #003DA5;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 0.9rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid #C8DCFF;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: #DDE6F8; border-radius: 10px;
    padding: 3px; gap: 2px; border: none;
    box-shadow: none; margin-bottom: 1.2rem;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent; color: #4B6A96;
    border-radius: 8px; font-weight: 500; font-size: 0.84rem;
    padding: 0.48rem 1rem;
  }
  .stTabs [aria-selected="true"] {
    background: #FFFFFF !important; color: #001F5B !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 4px rgba(0,31,91,0.12) !important;
  }

  /* ── Insight strip ── */
  .insight {
    background: #E3EEFB; border: none;
    border-left: 3px solid #003DA5; border-radius: 10px;
    padding: 1rem 1.2rem; font-size: 0.82rem;
    color: #003DA5; line-height: 1.65; margin-top: 1rem;
  }
  .insight strong { color: #001F5B; }

  /* ── Buttons ── */
  .stDownloadButton > button {
    background: #003DA5 !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 0.84rem !important;
    width: 100%; padding: 0.6rem 1.4rem !important;
  }

  /* ── Tables ── */
  .stDataFrame {
    border-radius: 10px !important; border: none !important;
    box-shadow: 0 1px 6px rgba(0,31,91,0.07) !important;
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

# ── PERSONA ROLE STATE ─────────────────────────────────────────────────────────
ROLES = {
    "Sales Rep":         {"icon":"👤","scope":"Territory · City-level",    "desc":"Your assigned territory — city-level HCP targeting"},
    "Area Manager":      {"icon":"👥","scope":"Area · Multi-state",         "desc":"Team performance across your area states"},
    "Regional Manager":  {"icon":"🏢","scope":"Region · Multi-region",      "desc":"Cross-region portfolio and pipeline oversight"},
    "Head of Sales":     {"icon":"🎯","scope":"National · All regions",      "desc":"National strategy and quarterly call volume targets"},
}
if "persona_role" not in st.session_state:
    st.session_state.persona_role = None

# ── DATABASE ───────────────────────────────────────────────────────────────────
# Production: set DATABASE_URL in Streamlit Cloud secrets dashboard.
# Local: falls back to localhost PostgreSQL.
def get_conn():
    """Always create a fresh connection — Neon is serverless and drops idle connections."""
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

# ── ROLE PICKER (shown on first visit before dashboard) ───────────────────────
if st.session_state.persona_role is None:
    st.markdown("""
    <div style='text-align:center;padding:3rem 0 2rem'>
      <div style='font-size:0.75rem;font-weight:700;color:#003DA5;text-transform:uppercase;
                  letter-spacing:0.15em;margin-bottom:0.6rem'>HCP Targeting & Brand Performance Analytics</div>
      <div style='font-size:2rem;font-weight:700;color:#001F5B;letter-spacing:-0.03em;margin-bottom:0.4rem'>
        Select your role to continue
      </div>
      <div style='font-size:0.9rem;color:#4B6A96'>
        Your view, filters and planning horizons will be tailored to your role
      </div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    for col, (role, meta) in zip([c1,c2,c3,c4], ROLES.items()):
        with col:
            st.html(f"""
            <div style='background:#FFFFFF;border-radius:16px;padding:2rem 1.4rem 1rem;
                        text-align:center;border:1.5px solid #C8DCFF;
                        box-shadow:0 2px 12px rgba(0,31,91,0.07);height:200px;
                        display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px'>
              <div style='font-size:2.4rem'>{meta["icon"]}</div>
              <div style='font-size:1rem;font-weight:700;color:#001F5B'>{role}</div>
              <div style='font-size:0.67rem;font-weight:700;color:#003DA5;text-transform:uppercase;
                          letter-spacing:0.08em'>{meta["scope"]}</div>
              <div style='font-size:0.74rem;color:#4B6A96;line-height:1.45;margin-top:4px'>{meta["desc"]}</div>
            </div>""")
            if st.button(f"Continue as {role}", key=f"pick_{role}", use_container_width=True):
                st.session_state.persona_role = role
                st.rerun()
    st.stop()

role = st.session_state.persona_role

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    rmeta = ROLES[role]
    st.markdown(f"""
    <div style='padding:1.2rem 0.4rem 0.8rem'>
      <div style='font-size:0.58rem;font-weight:700;color:#7B9AC0;text-transform:uppercase;
                  letter-spacing:0.12em;margin-bottom:0.5rem'>Viewing as</div>
      <div style='display:flex;align-items:center;gap:10px'>
        <div style='font-size:1.6rem'>{rmeta["icon"]}</div>
        <div>
          <div style='font-size:0.95rem;font-weight:700;color:#E8F0FF;line-height:1.1'>{role}</div>
          <div style='font-size:0.62rem;color:#7B9AC0;margin-top:2px'>{rmeta["scope"]}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)
    if st.button("↩ Change Role", use_container_width=True, key="change_role"):
        st.session_state.persona_role = None
        st.rerun()
    st.markdown("---")

    # ── Geographic filters — scoped by role ──────────────────────────────────
    state_abbrevs = sorted(df["state"].dropna().unique().tolist())

    if role in ("Sales Rep", "Area Manager"):
        state_options = ["🌎 All States"] + [f"{state_full(s)} ({s})" for s in state_abbrevs]
        sel_st_label  = st.selectbox("📍 State", state_options)
        st_val = None if sel_st_label == "🌎 All States" else sel_st_label.split("(")[-1].rstrip(")")
        sel_regions, region_states = [], []
    elif role == "Regional Manager":
        sel_regions = st.multiselect("🗺️ Region", list(US_REGIONS.keys()), default=["Northeast","Southeast"])
        region_states = [s for r in sel_regions for s in US_REGIONS.get(r,[])]
        st_val = None
    else:  # Head of Sales — national, no geo filter
        sel_regions = list(US_REGIONS.keys())
        region_states = []
        st_val = None

    # ── Territory / City filter (Sales Rep only) ──────────────────────────────
    city_val = None
    if role == "Sales Rep" and st_val:
        cities = sorted(df[df["state"] == st_val]["city"].dropna().unique().tolist())
        city_options = ["🏙️ All Cities (Full State)"] + cities
        sel_city = st.selectbox("🗺️ Territory (City)", city_options)
        city_val = None if sel_city == "🏙️ All Cities (Full State)" else sel_city
        if city_val:
            # Show territory code for CRM realism
            import hashlib
            tc = int(hashlib.md5(f"{st_val}{city_val}".encode()).hexdigest(),16) % 9000 + 1000
            st.markdown(f"<div style='font-size:0.62rem;color:#7B9AC0;margin-top:-8px;padding-left:2px'>Territory code: <strong style='color:#E8F0FF'>{st_val}-{tc}</strong></div>", unsafe_allow_html=True)
    elif role not in ("Sales Rep",):
        pass  # state handled above

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
          <div style='background:#0A2860;border-radius:8px;padding:10px 12px;margin-bottom:0.7rem;
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
          <div style='border-top:1px solid #0A3278;padding-top:0.5rem;margin-top:0.2rem'>
            <div style='color:#AEAEB2;font-size:0.66rem;font-weight:700;
                        text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem'>
              Components
            </div>
            <div style='margin-bottom:0.3rem'>
              <span style='color:#003DA5;font-weight:700'>Vol (40%)</span>
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
# Geographic scope by role
if role == "Regional Manager" and region_states:
    filt = filt[filt["state"].isin(region_states)]
elif st_val:
    filt = filt[filt["state"] == st_val]
if city_val:
    filt = filt[filt["city"] == city_val]
if sp_val:   filt = filt[filt["specialty"] == sp_val]
if seg_sel:  filt = filt[filt["segment"].isin(seg_sel)]
filt = filt[filt["targeting_score"] >= min_sc]
if kol_only: filt = filt[filt["opinion_leader_payments"] > 0]
filt = filt.sort_values("targeting_score", ascending=False).reset_index(drop=True)

# ── HERO ───────────────────────────────────────────────────────────────────────
terr_state = state_full(st_val) if st_val else ("·".join(sel_regions) if sel_regions else "National")
terr_city  = f" · {city_val}" if city_val else ""
terr_spec  = sp_val or "All Specialties"
st.markdown(f"""
<div style='background:#FFFFFF;border-radius:16px;padding:1.4rem 1.8rem;
            margin-bottom:1rem;box-shadow:0 1px 8px rgba(0,31,91,0.08)'>
  <div style='display:flex;align-items:center;gap:0.85rem'>
    <div style='width:40px;height:40px;background:#003DA5;border-radius:10px;
                display:flex;align-items:center;justify-content:center;
                font-size:1.2rem;flex-shrink:0'>🎯</div>
    <div style='flex:1'>
      <div style='font-size:1.3rem;font-weight:700;color:#001F5B;
                  letter-spacing:-0.02em;line-height:1.1'>
        HCP Targeting &amp; Brand Performance Analytics
      </div>
      <div style='font-size:0.78rem;color:#4B6A96;margin-top:3px'>
        {terr_state}{terr_city} &nbsp;·&nbsp; {terr_spec} &nbsp;·&nbsp; {datetime.now().strftime('%d %B %Y')}
      </div>
    </div>
    <div style='text-align:right;flex-shrink:0'>
      <div style='font-size:0.6rem;font-weight:700;color:#7B9AC0;text-transform:uppercase;
                  letter-spacing:0.1em'>Role</div>
      <div style='font-size:0.88rem;font-weight:700;color:#003DA5;margin-top:2px'>
        {rmeta["icon"]} {role}
      </div>
    </div>
  </div>
  <div style='margin-top:0.8rem;display:flex;gap:5px;flex-wrap:wrap'>
    {"".join(f'<span style="background:#E3EEFB;color:#001F5B;padding:2px 10px;border-radius:5px;font-size:0.62rem;font-weight:600;letter-spacing:0.04em;border:1px solid #C8DCFF">{t}</span>' for t in ["💊 Diabetes Portfolio","PostgreSQL","227K HCPs","83M+ Rows","CMS 2021–2022","Python · Streamlit · Plotly"])}
  </div>
</div>""", unsafe_allow_html=True)

# ── PLATFORM NOTES ────────────────────────────────────────────────────────────
with st.expander("ℹ️ Platform Scope & Production CRM Notes", expanded=False):
    st.html("""
    <div style="padding:0.2rem 0.4rem">
      <div style="font-size:0.78rem;font-weight:700;color:#1D1D1F;margin-bottom:0.8rem">
        What this prototype demonstrates vs. a production Salesforce / Veeva deployment
      </div>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">

        <div style="background:#EDFBF1;border-radius:12px;padding:12px 14px;border-left:3px solid #34C759">
          <div style="font-size:0.68rem;font-weight:700;color:#1A7A35;margin-bottom:6px;
                      text-transform:uppercase;letter-spacing:0.08em">✅ Implemented here</div>
          <div style="font-size:0.72rem;color:#3A3A3C;line-height:1.8">
            Composite HCP targeting score (Vol + Growth + Payment)<br>
            NTILE(10) specialty-normalised decile ranking<br>
            4-tier segmentation (High Value / Growth / Maintenance / Deprioritise)<br>
            Rule-based Next Best Action per HCP<br>
            Loyalty tier model (Loyalist / Intermittent / Tourist / Non-Rx)<br>
            Call cadence engine by segment (4–24 week cycles)<br>
            Role-based planning views (Rep / Area / Regional / Head of Sales)<br>
            KOL identification via CMS Open Payments<br>
            Territory intelligence — state-level choropleth<br>
            At-Risk and Breakthrough alert detection
          </div>
        </div>

        <div style="background:#FFF8ED;border-radius:12px;padding:12px 14px;border-left:3px solid #FF9500">
          <div style="font-size:0.68rem;font-weight:700;color:#CC7700;margin-bottom:6px;
                      text-transform:uppercase;letter-spacing:0.08em">🔧 In a production CRM deployment</div>
          <div style="font-size:0.72rem;color:#3A3A3C;line-height:1.8">
            <strong>Account hierarchy</strong> — HCPs modelled as Contacts under Accounts (practice / hospital)<br>
            <strong>Territory Management 2.0</strong> — automated HCP-to-rep routing, no manual state filter<br>
            <strong>Activity sync</strong> — every call, sample drop and email logged against HCP record in real time<br>
            <strong>Einstein NBA</strong> — ML-driven Next Best Action on channel, message and timing<br>
            <strong>Journey stages</strong> — Unaware → Aware → Trialing → Committed → Advocate progression<br>
            <strong>Push alerts</strong> — overdue and at-risk notifications via Slack, email, mobile<br>
            <strong>Closed-loop measurement</strong> — Rx impact of calls (contacted vs uncontacted HCP cohort lift)<br>
            <strong>Veeva CLM</strong> — approved email, e-detailing and sample management integration
          </div>
        </div>

      </div>

      <div style="margin-top:10px;padding:10px 14px;background:#EEF3FC;border-radius:10px;
                  font-size:0.7rem;color:#4B6A96;line-height:1.6">
        <strong style="color:#1D1D1F">Call activity data in this prototype is simulated</strong>
        — seeded deterministically by NPI for reproducibility. Cadence logic, loyalty tiers and
        segmentation rules mirror commercial pharma CRM methodology. NBA recommendations are
        rule-based (not ML); in Salesforce Health Cloud + Einstein, these would be model-scored
        and personalised per HCP interaction history.
      </div>
    </div>
    """)

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

# ── ALERT CARDS ───────────────────────────────────────────────────────────────
at_risk = filt[
    (filt["segment"] == "High Value") &
    (filt["yoy_growth_pct"].notna()) &
    (filt["yoy_growth_pct"] < 0)
].head(3)

breakthrough = filt[
    (filt["segment"] == "Growth") &
    (filt["volume_decile"] >= 7)
].head(3)

if len(at_risk) > 0 or len(breakthrough) > 0:
    ac1, ac2 = st.columns(2)
    with ac1:
        if len(at_risk) > 0:
            names = ", ".join(f"Dr {r['last_name']}" for _, r in at_risk.iterrows())
            st.html(f"""
            <div style="background:#FFF0EF;border-radius:14px;padding:1rem 1.3rem;
                        border-left:4px solid #FF3B30;margin-bottom:0.5rem">
              <div style="font-size:0.65rem;font-weight:700;color:#FF3B30;
                          text-transform:uppercase;letter-spacing:0.1em">
                ⚠️ At Risk — High Value HCPs with Declining Rx
              </div>
              <div style="font-size:0.88rem;font-weight:700;color:#1D1D1F;margin-top:5px">
                {len(at_risk)} High Value prescribers showing negative YoY growth
              </div>
              <div style="font-size:0.75rem;color:#6E6E73;margin-top:3px">
                {names} — defend Rx share before competitors move in
              </div>
            </div>""")
    with ac2:
        if len(breakthrough) > 0:
            names2 = ", ".join(f"Dr {r['last_name']}" for _, r in breakthrough.iterrows())
            st.html(f"""
            <div style="background:#EDFBF1;border-radius:14px;padding:1rem 1.3rem;
                        border-left:4px solid #34C759;margin-bottom:0.5rem">
              <div style="font-size:0.65rem;font-weight:700;color:#1A7A35;
                          text-transform:uppercase;letter-spacing:0.1em">
                🚀 About to Break Through — Growth HCPs near Vol D8
              </div>
              <div style="font-size:0.88rem;font-weight:700;color:#1D1D1F;margin-top:5px">
                {len(breakthrough)} Growth HCPs at Vol D7 — one push from High Value
              </div>
              <div style="font-size:0.75rem;color:#6E6E73;margin-top:3px">
                {names2} — prioritise now to convert segment
              </div>
            </div>""")

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
              <div style="font-size:0.72rem;color:#003DA5;font-weight:600;
                          margin-top:4px;line-height:1.4">
                &#8594; {action}
              </div>
            </div>
            """)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋  Diabetes Call List",
    "📈  Market Intelligence",
    "🗺️  Territory Map",
    "⭐  Opinion Leaders",
    "🩺  HCP Profile",
    "🗓️  Rep Planner",
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
        disp["action"]     = disp.apply(recommended_action, axis=1)
        disp["state_full"] = disp["state"].apply(state_full)
        disp["_status"]    = disp.apply(lambda r: call_due_status(r)[0], axis=1)
        disp["_due"]       = disp.apply(lambda r: call_due_status(r)[1], axis=1)
        disp["call_status"]= disp.apply(lambda r: f"{call_due_status(r)[0]} · {call_due_status(r)[1]}", axis=1)

        show = disp[[
            "last_name","first_name","specialty","city","state_full",
            "fills_2022","yoy_growth_pct","targeting_score",
            "call_status","action","segment"
        ]].copy()
        show.columns = [
            "Last","First","Specialty","City","State",
            "Fills 2022","YoY %","Score",
            "Call Status","Next Action","Segment"
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
            yoy = row.get("YoY %","")
            if "+" in str(yoy):
                idx = list(row.index).index("YoY %")
                styles[idx] = "color:#1A7A35;font-weight:600"
            elif str(yoy).startswith("-"):
                idx = list(row.index).index("YoY %")
                styles[idx] = "color:#CC2200;font-weight:600"
            idx = list(row.index).index("Next Action")
            styles[idx] = "color:#003DA5;font-weight:500"
            # Call status colour
            cs = str(row.get("Call Status",""))
            if "Overdue" in cs:
                idx = list(row.index).index("Call Status")
                styles[idx] = "color:#CC2200;font-weight:700"
            elif "Due Soon" in cs:
                idx = list(row.index).index("Call Status")
                styles[idx] = "color:#CC7700;font-weight:600"
            elif "On Track" in cs:
                idx = list(row.index).index("Call Status")
                styles[idx] = "color:#1A7A35;font-weight:500"
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
                        color="Count", color_continuous_scale=["#D1E8FF","#003DA5"],
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
                    border-left:3px solid #003DA5">
          <div style="font-size:0.75rem;font-weight:700;color:#003DA5">GLP-1 Agonists</div>
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
          <div style="font-size:2.4rem;font-weight:900;color:#003DA5;
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
                         color_discrete_sequence=["#003DA5","#34C759","#FF9500","#FF3B30","#BF5AF2"],
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
                         color_discrete_sequence=["#003DA5","#34C759","#FF9500","#FF3B30","#BF5AF2"],
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
                    color_continuous_scale=["#D1E8FF","#003DA5"],
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
        <div style="font-size:2.2rem;font-weight:900;color:#003DA5;
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
        color_continuous_scale=["#D1E8FF","#003DA5"],
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
        sk = kd.style.map(lambda v: {
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
                         color_continuous_scale=["#D1E8FF","#003DA5"],
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
                        background:linear-gradient(135deg,#003DA5,#40AAFF);
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
              <div style="margin-top:0.7rem;font-size:0.78rem;color:#003DA5;font-weight:600">
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
            st.dataframe(recs_df.style.map(
                lambda v: "color:#003DA5;font-weight:600" if "Detail" in str(v)
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
                marker_color=["#003DA5","#34C759","#FF9500", seg_col_],
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
                calls_df.style.map(
                    lambda v: "color:#003DA5;font-weight:600" if "F2F" in str(v) or "P2P" in str(v)
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

            # Visit schedule card
            p_status, p_detail, p_st_c, p_st_bg = call_due_status(hcp)
            last_c   = sim_last_call(hcp.get("npi",0))
            next_c   = last_c + timedelta(days=cadence_days(hcp))
            cad_lbl  = cadence_label(hcp)
            st.html(f"""
            <div style="background:#FFFFFF;border-radius:14px;padding:1rem 1.4rem;
                        margin:0.6rem 0;box-shadow:0 2px 8px rgba(0,0,0,0.05)">
              <div style="font-size:0.65rem;font-weight:700;color:#8E8E93;
                          text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.6rem">
                Visit Schedule
              </div>
              <div style="display:flex;gap:20px;flex-wrap:wrap">
                <div>
                  <div style="font-size:0.6rem;color:#AEAEB2;font-weight:700;text-transform:uppercase">Last Contact</div>
                  <div style="font-size:0.9rem;font-weight:700;color:#1D1D1F">{last_c.strftime('%-d %b %Y')}</div>
                </div>
                <div>
                  <div style="font-size:0.6rem;color:#AEAEB2;font-weight:700;text-transform:uppercase">Next Due</div>
                  <div style="font-size:0.9rem;font-weight:700;color:{p_st_c}">{next_c.strftime('%-d %b %Y')}</div>
                </div>
                <div>
                  <div style="font-size:0.6rem;color:#AEAEB2;font-weight:700;text-transform:uppercase">Cadence</div>
                  <div style="font-size:0.9rem;font-weight:700;color:#1D1D1F">{cad_lbl}</div>
                </div>
                <div>
                  <div style="font-size:0.6rem;color:#AEAEB2;font-weight:700;text-transform:uppercase">Status</div>
                  <span style="background:{p_st_bg};color:{p_st_c};padding:3px 10px;
                               border-radius:980px;font-size:0.72rem;font-weight:700">{p_status} · {p_detail}</span>
                </div>
              </div>
            </div>""")

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

# ──────────────────────────────────────────────────────────────────────────────
# TAB 6 — REP PLANNER (role-aware)
# ──────────────────────────────────────────────────────────────────────────────

# US regions for Area / Regional views
US_REGIONS = {
    "Northeast": ["CT","ME","MA","NH","RI","VT","NY","NJ","PA","DC","MD","DE"],
    "Southeast": ["AL","AR","FL","GA","KY","LA","MS","NC","SC","TN","VA","WV"],
    "Midwest":   ["IL","IN","IA","KS","MI","MN","MO","NE","ND","OH","SD","WI"],
    "Southwest": ["AZ","NM","OK","TX"],
    "West":      ["AK","CA","CO","HI","ID","MT","NV","OR","UT","WA","WY"],
}

with tab6:

    # ── Role selector ──────────────────────────────────────────────────────────
    st.html("""
    <div style="background:#FFFFFF;border-radius:14px;padding:1rem 1.4rem;
                margin-bottom:1rem;box-shadow:0 2px 8px rgba(0,0,0,0.05)">
      <div style="font-size:0.65rem;font-weight:700;color:#8E8E93;
                  text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem">
        Select Your Role — planning horizon and geography adapt automatically
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <span style="background:#EBF5FF;color:#003DA5;padding:3px 12px;border-radius:980px;
                     font-size:0.7rem;font-weight:700;border:1px solid #B3D7FF">
          👤 Sales Rep → Daily · Weekly · Monthly · 1 state
        </span>
        <span style="background:#FFF8ED;color:#CC7700;padding:3px 12px;border-radius:980px;
                     font-size:0.7rem;font-weight:700;border:1px solid #FFE4B2">
          👥 Area Manager → Weekly · Monthly · Quarterly · 1 region
        </span>
        <span style="background:#F5F0FF;color:#7B2FBE;padding:3px 12px;border-radius:980px;
                     font-size:0.7rem;font-weight:700;border:1px solid #DDD0FF">
          🏢 Regional Manager → Monthly · Quarterly · Multi-region
        </span>
        <span style="background:#EDFBF1;color:#1A7A35;padding:3px 12px;border-radius:980px;
                     font-size:0.7rem;font-weight:700;border:1px solid #C3F2D0">
          🎯 Head of Sales → Quarterly · National
        </span>
      </div>
    </div>""")

    role = st.selectbox(
        "I am a:",
        ["👤 Sales Rep", "👥 Area Manager", "🏢 Regional Manager", "🎯 Head of Sales"],
        label_visibility="collapsed"
    )
    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SALES REP — Daily / Weekly / Monthly (single state required)
    # ══════════════════════════════════════════════════════════════════════════
    if role == "👤 Sales Rep":
        if not st_val:
            st.warning("📍 Select your territory state from the sidebar to generate your personal call plan.")
        else:
            rep_views = st.radio("Plan View:", ["📅 Daily Plan","🗓️ Weekly Calendar","📊 Monthly Coverage"],
                                 horizontal=True, label_visibility="collapsed")
            st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
            rep_filt = filt.copy()

            # ── Daily ─────────────────────────────────────────────────────────
            if rep_views == "📅 Daily Plan":
                st.markdown('<div class="sec">Today\'s Call Plan — ' + state_full(st_val) + ' Territory · ' + datetime.now().strftime("%A, %-d %B %Y") + '</div>',
                            unsafe_allow_html=True)
                hv  = rep_filt[rep_filt["segment"]=="High Value"].head(2)
                gr  = rep_filt[rep_filt["segment"]=="Growth"].head(2)
                eod = rep_filt[rep_filt["opinion_leader_payments"]>0].head(1)
                if len(eod)==0: eod = rep_filt[rep_filt["segment"]=="Maintenance"].head(1)

                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Calls Today", str(len(hv)+len(gr)+len(eod)))
                m2.metric("High Value", str(len(hv)))
                m3.metric("Growth", str(len(gr)))
                m4.metric("KOL / EOD", str(len(eod)))

                for label, desc, sdf, col in [
                    ("🌅 Morning 9:00 AM","High Value — defend Rx share",hv,"#FF3B30"),
                    ("☀️ Afternoon 1:00 PM","Growth HCPs — GLP-1 detailing",gr,"#34C759"),
                    ("🌇 End of Day 4:30 PM","KOL / Speaker engagement",eod,"#FF9500"),
                ]:
                    if len(sdf)==0: continue
                    st.html(f'''<div style="margin:10px 0 4px;font-size:0.72rem;font-weight:700;
                        color:{col};text-transform:uppercase;letter-spacing:0.08em">
                        {label} &nbsp;·&nbsp; <span style="color:#6E6E73;font-weight:400">{desc}</span>
                        </div>''')
                    ccols = st.columns(max(len(sdf),1))
                    for cw,(_, row) in zip(ccols, sdf.iterrows()):
                        seg=row.get("segment",""); sc_c=SEG_COLORS.get(seg,"#8E8E93")
                        sc_bg=SEG_BG.get(seg,"#F5F5F7")
                        status,detail,st_c,st_bg=call_due_status(row)
                        with cw:
                            st.html(f'''<div style="background:#FFFFFF;border-radius:14px;
                                padding:14px 16px;border-left:3px solid {col};
                                box-shadow:0 2px 8px rgba(0,0,0,0.06);margin-bottom:8px">
                              <div style="font-size:0.88rem;font-weight:700;color:#1D1D1F">
                                Dr {row.get("last_name","")}, {str(row.get("first_name",""))[:1]}.
                                {row.get("credential","") or ""}
                              </div>
                              <div style="font-size:0.72rem;color:#6E6E73">{str(row.get("specialty",""))[:30]}</div>
                              <div style="font-size:0.68rem;color:#8E8E93">📍 {row.get("city","")}</div>
                              <div style="display:flex;gap:5px;margin:8px 0 6px;flex-wrap:wrap">
                                <span style="background:{sc_bg};color:{sc_c};padding:2px 8px;
                                  border-radius:980px;font-size:0.63rem;font-weight:700">{seg}</span>
                                <span style="background:{st_bg};color:{st_c};padding:2px 8px;
                                  border-radius:980px;font-size:0.63rem;font-weight:700">{status} · {detail}</span>
                              </div>
                              <div style="font-size:0.72rem;color:#003DA5;font-weight:600">
                                → {recommended_action(row)}
                              </div>
                              <div style="font-size:0.65rem;color:#AEAEB2;margin-top:3px">
                                {cadence_label(row)}
                              </div>
                            </div>''')

            # ── Weekly ────────────────────────────────────────────────────────
            elif rep_views == "🗓️ Weekly Calendar":
                st.markdown('<div class="sec">Weekly Call Calendar — ' + state_full(st_val) + ' Territory</div>',
                            unsafe_allow_html=True)
                today   = datetime.now()
                mon     = today - timedelta(days=today.weekday())
                wd      = [(mon+timedelta(days=i)) for i in range(5)]
                pool    = rep_filt.copy()
                pool["_o1"] = pool.apply(lambda r: call_due_status(r)[0], axis=1).map(
                    {"Overdue":0,"Due Soon":1,"On Track":2}).fillna(3)
                pool["_o2"] = pool["segment"].map(
                    {"High Value":0,"Growth":1,"Maintenance":2,"Deprioritise":3}).fillna(4)
                pool = pool.sort_values(["_o1","_o2"]).head(25).reset_index(drop=True)

                dcols = st.columns(5)
                for d_i,(dc,dt) in enumerate(zip(dcols,wd)):
                    day_hcps = pool.iloc[d_i*5:(d_i+1)*5]
                    is_today = dt.date()==today.date()
                    with dc:
                        st.html(f'''<div style="background:{"#EBF5FF" if is_today else "#F5F5F7"};
                            border-radius:12px;padding:8px 10px;margin-bottom:8px;
                            border:{"2px solid #003DA5" if is_today else "1px solid #E5E5EA"}">
                          <div style="font-size:0.65rem;font-weight:700;
                              color:{"#003DA5" if is_today else "#8E8E93"};
                              text-transform:uppercase;letter-spacing:0.08em">
                            {dt.strftime("%A")}{"  ← Today" if is_today else ""}
                          </div>
                          <div style="font-size:0.75rem;color:#1D1D1F;font-weight:600">
                            {dt.strftime("%-d %b")} · {len(day_hcps)} calls
                          </div>
                        </div>''')
                        for _,row in day_hcps.iterrows():
                            seg=row.get("segment",""); sc_c=SEG_COLORS.get(seg,"#8E8E93")
                            sc_bg=SEG_BG.get(seg,"#F5F5F7")
                            status,detail,st_c,_=call_due_status(row)
                            st.html(f'''<div style="background:#FFFFFF;border-radius:10px;
                                padding:10px 12px;margin-bottom:6px;border-left:3px solid {sc_c};
                                box-shadow:0 1px 4px rgba(0,0,0,0.05)">
                              <div style="font-size:0.78rem;font-weight:700;color:#1D1D1F">
                                Dr {row.get("last_name","")}, {str(row.get("first_name",""))[:1]}.
                              </div>
                              <div style="font-size:0.65rem;color:#6E6E73">{str(row.get("specialty",""))[:22]}</div>
                              <div style="font-size:0.63rem;color:#8E8E93">📍 {row.get("city","")}</div>
                              <div style="display:flex;gap:4px;margin-top:5px;flex-wrap:wrap">
                                <span style="background:{sc_bg};color:{sc_c};padding:1px 7px;
                                  border-radius:980px;font-size:0.6rem;font-weight:700">{seg}</span>
                                <span style="color:{st_c};font-size:0.6rem;font-weight:600">{status}</span>
                              </div>
                            </div>''')

            # ── Monthly ───────────────────────────────────────────────────────
            else:
                st.markdown('<div class="sec">Monthly Coverage — ' + state_full(st_val) + ' Territory</div>',
                            unsafe_allow_html=True)
                total = len(rep_filt)
                contacted = sum(1 for _,r in rep_filt.head(300).iterrows()
                                if (datetime.now()-sim_last_call(r.get("npi",0))).days<=30)
                overdue   = sum(1 for _,r in rep_filt.head(300).iterrows()
                                if call_due_status(r)[0]=="Overdue")
                pct = contacted/total*100 if total>0 else 0
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Territory HCPs",    f"{total:,}")
                c2.metric("Contacted (30d)",    f"{contacted:,}")
                c3.metric("Coverage %",         f"{pct:.0f}%")
                c4.metric("Overdue Calls",       f"{overdue:,}")

                seg_cover=[]
                for sn,cad in CALL_CADENCE_DAYS.items():
                    sd=rep_filt[rep_filt["segment"]==sn]
                    if not len(sd): continue
                    ct=sum(1 for _,r in sd.head(200).iterrows()
                           if (datetime.now()-sim_last_call(r.get("npi",0))).days<=cad)
                    seg_cover.append({"Segment":sn,"HCPs":len(sd),"Contacted":ct,
                                      "Coverage %":round(ct/min(len(sd),200)*100),"Target":85})
                if seg_cover:
                    sc_df=pd.DataFrame(seg_cover)
                    fig_c=px.bar(sc_df,x="Coverage %",y="Segment",orientation="h",
                                 color="Segment",color_discrete_map=SEG_COLORS,text="Coverage %",
                                 labels={"Coverage %":"Coverage %","Segment":""})
                    fig_c.update_traces(texttemplate="%{text:.0f}%",textposition="outside")
                    fig_c.add_vline(x=85,line_dash="dot",line_color="#8E8E93",
                                    annotation_text="85% target",annotation_position="top right")
                    fig_c.update_layout(**CHART_LAYOUT,height=220,showlegend=False,
                                        margin=dict(t=5,b=5,l=5,r=60),
                                        xaxis=dict(range=[0,110],gridcolor="#F5F5F7"),
                                        yaxis=dict(autorange="reversed",gridcolor="#F5F5F7"))
                    st.plotly_chart(fig_c,use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # AREA MANAGER — Weekly · Monthly · Quarterly (1 region)
    # ══════════════════════════════════════════════════════════════════════════
    elif role == "👥 Area Manager":
        region_sel = st.selectbox("Select your region:",
                                  list(US_REGIONS.keys()), key="am_region")
        region_states = US_REGIONS[region_sel]
        am_filt = df[df["state"].isin(region_states)].copy()
        if st_val: am_filt = am_filt[am_filt["state"]==st_val]
        if sp_val: am_filt = am_filt[am_filt["specialty"]==sp_val]
        if seg_sel: am_filt = am_filt[am_filt["segment"].isin(seg_sel)]
        am_filt = am_filt[am_filt["targeting_score"]>=min_sc].sort_values(
            "targeting_score",ascending=False)

        am_views = st.radio("View:", ["🗓️ Weekly Team Overview","📊 Monthly Coverage","🎯 Quarterly Strategy"],
                            horizontal=True, label_visibility="collapsed")

        a1,a2,a3,a4 = st.columns(4)
        a1.metric(f"{region_sel} Region HCPs", f"{len(am_filt):,}")
        a2.metric("High Value",  f"{(am_filt['segment']=='High Value').sum():,}")
        a3.metric("Growth",      f"{(am_filt['segment']=='Growth').sum():,}")
        a4.metric("States Covered", str(len(region_states)))

        if am_views == "🗓️ Weekly Team Overview":
            st.markdown(f'<div class="sec">Weekly State-by-State Snapshot — {region_sel} Region</div>',
                        unsafe_allow_html=True)
            state_summary=[]
            for st_code in region_states:
                sd=am_filt[am_filt["state"]==st_code]
                if not len(sd): continue
                ov=sum(1 for _,r in sd.head(100).iterrows() if call_due_status(r)[0]=="Overdue")
                state_summary.append({
                    "State": state_full(st_code), "Total HCPs": len(sd),
                    "High Value": (sd["segment"]=="High Value").sum(),
                    "Growth":     (sd["segment"]=="Growth").sum(),
                    "Overdue Calls": ov,
                    "Avg Score":  round(float(sd["targeting_score"].mean()),3),
                })
            if state_summary:
                ss_df=pd.DataFrame(state_summary).sort_values("High Value",ascending=False)
                st.dataframe(ss_df,use_container_width=True,hide_index=True)

            # Top 10 HCPs across region to coach reps on
            st.markdown(f'<div class="sec">Top 10 Regional Priority HCPs — coaching focus</div>',
                        unsafe_allow_html=True)
            top10=am_filt.head(10).copy()
            top10["State"]=top10["state"].apply(state_full)
            top10["Status"]=top10.apply(lambda r: f"{call_due_status(r)[0]} · {call_due_status(r)[1]}",axis=1)
            top10["Action"]=top10.apply(recommended_action,axis=1)
            disp10=top10[["last_name","first_name","specialty","city","State",
                          "targeting_score","segment","Status","Action"]].copy()
            disp10.columns=["Last","First","Specialty","City","State","Score","Segment","Status","Action"]
            disp10.index=range(1,len(disp10)+1)
            st.dataframe(disp10,use_container_width=True,hide_index=False)

        elif am_views == "📊 Monthly Coverage":
            st.markdown(f'<div class="sec">Monthly Coverage by State — {region_sel} Region</div>',
                        unsafe_allow_html=True)
            cover_rows=[]
            for st_code in region_states:
                sd=am_filt[am_filt["state"]==st_code]
                if not len(sd): continue
                ct=sum(1 for _,r in sd.head(200).iterrows()
                       if (datetime.now()-sim_last_call(r.get("npi",0))).days<=30)
                ov=sum(1 for _,r in sd.head(200).iterrows() if call_due_status(r)[0]=="Overdue")
                cover_rows.append({
                    "State":state_full(st_code),"HCPs":len(sd),
                    "Contacted (30d)":ct,"Coverage %":round(ct/min(len(sd),200)*100),
                    "Overdue":ov,"Target":85
                })
            if cover_rows:
                cv_df=pd.DataFrame(cover_rows).sort_values("Coverage %")
                fig_am=px.bar(cv_df,x="Coverage %",y="State",orientation="h",
                              color="Coverage %",color_continuous_scale=["#FF3B30","#FF9500","#34C759"],
                              text="Coverage %",labels={"State":"","Coverage %":"Coverage %"})
                fig_am.update_traces(texttemplate="%{text:.0f}%",textposition="outside")
                fig_am.add_vline(x=85,line_dash="dot",line_color="#8E8E93",
                                 annotation_text="85% target",annotation_position="top right")
                fig_am.update_layout(**CHART_LAYOUT,height=max(250,len(cover_rows)*35),
                                     coloraxis_showscale=False,showlegend=False,
                                     margin=dict(t=5,b=5,l=5,r=60),
                                     xaxis=dict(range=[0,110],gridcolor="#F5F5F7"),
                                     yaxis=dict(autorange="reversed",gridcolor="#F5F5F7"))
                st.plotly_chart(fig_am,use_container_width=True)
                st.dataframe(cv_df,use_container_width=True,hide_index=True)

        else:  # Quarterly
            q=(datetime.now().month-1)//3+1
            st.markdown(f'<div class="sec">Q{q} Quarterly Strategy — {region_sel} Region</div>',
                        unsafe_allow_html=True)
            seg_c=am_filt["segment"].value_counts()
            qs1,qs2,qs3,qs4=st.columns(4)
            for cw,sn,em in zip([qs1,qs2,qs3,qs4],
                ["High Value","Growth","Maintenance","Deprioritise"],["🔴","🟢","🟠","⚫"]):
                n=seg_c.get(sn,0); tot=len(am_filt)
                cw.metric(f"{em} {sn}",f"{n:,}",f"{n/tot*100:.0f}% of region" if tot else "")

            ql2,qr2=st.columns(2)
            with ql2:
                st.markdown('<div class="sec">Conversion Targets — Growth → High Value</div>',
                            unsafe_allow_html=True)
                conv=am_filt[(am_filt["segment"]=="Growth")&(am_filt["volume_decile"]>=7)].head(10).copy()
                conv["State"]=conv["state"].apply(state_full)
                conv2=conv[["last_name","first_name","specialty","city","State",
                             "volume_decile","growth_decile","targeting_score"]].copy()
                conv2.columns=["Last","First","Specialty","City","State","Vol D","Growth D","Score"]
                conv2.index=range(1,len(conv2)+1)
                if len(conv2): st.dataframe(conv2,use_container_width=True,hide_index=False)
                else: st.info("No near-threshold Growth HCPs in this region.")
            with qr2:
                st.markdown('<div class="sec">At-Risk HCPs — High Value, Declining Rx</div>',
                            unsafe_allow_html=True)
                risk=am_filt[(am_filt["segment"]=="High Value")&
                             (am_filt["yoy_growth_pct"].notna())&
                             (am_filt["yoy_growth_pct"]<0)].head(10).copy()
                risk["State"]=risk["state"].apply(state_full)
                risk2=risk[["last_name","first_name","specialty","State",
                             "fills_2022","yoy_growth_pct"]].copy()
                risk2["yoy_growth_pct"]=risk2["yoy_growth_pct"].apply(lambda x:f"{x:.1f}%")
                risk2.columns=["Last","First","Specialty","State","Fills 2022","YoY %"]
                risk2.index=range(1,len(risk2)+1)
                if len(risk2): st.dataframe(risk2,use_container_width=True,hide_index=False)
                else: st.success("No at-risk HCPs in this region.")

    # ══════════════════════════════════════════════════════════════════════════
    # REGIONAL MANAGER — Monthly · Quarterly (multi-region)
    # ══════════════════════════════════════════════════════════════════════════
    elif role == "🏢 Regional Manager":
        rm_regions = st.multiselect("Select your regions:",
                                    list(US_REGIONS.keys()),
                                    default=list(US_REGIONS.keys())[:2],
                                    key="rm_regions")
        rm_states  = [s for r in rm_regions for s in US_REGIONS.get(r,[])]
        rm_filt    = df[df["state"].isin(rm_states)].copy() if rm_states else df.copy()
        if sp_val: rm_filt = rm_filt[rm_filt["specialty"]==sp_val]
        if seg_sel: rm_filt= rm_filt[rm_filt["segment"].isin(seg_sel)]

        rm_views = st.radio("View:", ["📊 Monthly Region View","🎯 Quarterly Strategy"],
                            horizontal=True, label_visibility="collapsed")

        r1,r2,r3,r4=st.columns(4)
        r1.metric("Total HCPs",  f"{len(rm_filt):,}")
        r2.metric("High Value",  f"{(rm_filt['segment']=='High Value').sum():,}")
        r3.metric("Regions",     str(len(rm_regions)))
        r4.metric("States",      str(len(rm_states)))

        if rm_views == "📊 Monthly Region View":
            st.markdown('<div class="sec">Monthly Performance by Region</div>', unsafe_allow_html=True)
            region_rows=[]
            for reg in rm_regions:
                rstates=US_REGIONS[reg]
                rdf=rm_filt[rm_filt["state"].isin(rstates)]
                if not len(rdf): continue
                ct=sum(1 for _,r in rdf.head(300).iterrows()
                       if (datetime.now()-sim_last_call(r.get("npi",0))).days<=30)
                ov=sum(1 for _,r in rdf.head(300).iterrows() if call_due_status(r)[0]=="Overdue")
                region_rows.append({
                    "Region":reg,"States":len(rstates),"HCPs":len(rdf),
                    "High Value":(rdf["segment"]=="High Value").sum(),
                    "Growth":(rdf["segment"]=="Growth").sum(),
                    "Contacted (30d)":ct,
                    "Coverage %":round(ct/min(len(rdf),300)*100),
                    "Overdue":ov,
                    "Avg Score":round(float(rdf["targeting_score"].mean()),3),
                })
            if region_rows:
                rdf2=pd.DataFrame(region_rows)
                fig_rm=px.bar(rdf2,x="Region",y="Coverage %",color="Coverage %",
                              color_continuous_scale=["#FF3B30","#FF9500","#34C759"],
                              text="Coverage %",labels={"Region":"","Coverage %":"Coverage %"})
                fig_rm.update_traces(texttemplate="%{text:.0f}%",textposition="outside")
                fig_rm.add_hline(y=85,line_dash="dot",line_color="#8E8E93",
                                 annotation_text="85% target",annotation_position="top right")
                fig_rm.update_layout(**CHART_LAYOUT,height=280,coloraxis_showscale=False,
                                     margin=dict(t=5,b=5,l=5,r=20),
                                     yaxis=dict(range=[0,110],gridcolor="#F5F5F7"),
                                     xaxis=dict(gridcolor="#F5F5F7"))
                st.plotly_chart(fig_rm,use_container_width=True)
                st.dataframe(rdf2,use_container_width=True,hide_index=True)

        else:  # Quarterly
            q=(datetime.now().month-1)//3+1
            st.markdown(f'<div class="sec">Q{q} Quarterly Strategy — Multi-Region</div>',
                        unsafe_allow_html=True)
            seg_c=rm_filt["segment"].value_counts(); tot=len(rm_filt)
            sq1,sq2,sq3,sq4=st.columns(4)
            for cw,sn,em in zip([sq1,sq2,sq3,sq4],
                ["High Value","Growth","Maintenance","Deprioritise"],["🔴","🟢","🟠","⚫"]):
                n=seg_c.get(sn,0)
                cw.metric(f"{em} {sn}",f"{n:,}",f"{n/tot*100:.0f}% of portfolio" if tot else "")

            # Region comparison chart
            reg_comp=[{"Region":r,"High Value":(rm_filt[rm_filt["state"].isin(US_REGIONS[r])]["segment"]=="High Value").sum(),
                       "Growth":(rm_filt[rm_filt["state"].isin(US_REGIONS[r])]["segment"]=="Growth").sum(),
                       "Avg Score":round(float(rm_filt[rm_filt["state"].isin(US_REGIONS[r])]["targeting_score"].mean() if len(rm_filt[rm_filt["state"].isin(US_REGIONS[r])])>0 else 0),3)}
                      for r in rm_regions]
            rc_df=pd.DataFrame(reg_comp)
            fig_rc=px.bar(rc_df,x="Region",y=["High Value","Growth"],barmode="group",
                          color_discrete_map={"High Value":"#FF3B30","Growth":"#34C759"},
                          labels={"value":"HCPs","variable":"Segment"})
            fig_rc.update_layout(**CHART_LAYOUT,height=280,
                                 margin=dict(t=5,b=5,l=5,r=5),
                                 xaxis=dict(gridcolor="#F5F5F7"),
                                 yaxis=dict(gridcolor="#F5F5F7"))
            st.plotly_chart(fig_rc,use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # HEAD OF SALES — Quarterly National View
    # ══════════════════════════════════════════════════════════════════════════
    else:
        q=(datetime.now().month-1)//3+1
        st.markdown(f'<div class="sec">Q{q} {datetime.now().year} National Portfolio Overview</div>',
                    unsafe_allow_html=True)

        h1,h2,h3,h4,h5=st.columns(5)
        h1.metric("National HCPs",  f"{len(df):,}")
        h2.metric("High Value",     f"{(df['segment']=='High Value').sum():,}")
        h3.metric("Growth",         f"{(df['segment']=='Growth').sum():,}")
        h4.metric("KOLs / Speakers",f"{(df['opinion_leader_payments']>0).sum():,}")
        h5.metric("Avg Score",      f"{df['targeting_score'].mean():.3f}")

        # National segment distribution
        ncl,ncr=st.columns([2,3])
        with ncl:
            seg_n=df["segment"].value_counts().reset_index()
            seg_n.columns=["Segment","Count"]
            fig_ns=px.pie(seg_n,names="Segment",values="Count",
                          color="Segment",color_discrete_map=SEG_COLORS,hole=0.6)
            fig_ns.update_traces(textposition="outside",textinfo="label+percent",textfont_size=11)
            fig_ns.update_layout(**CHART_LAYOUT,height=280,showlegend=False,
                                 margin=dict(t=5,b=5,l=5,r=5))
            st.plotly_chart(fig_ns,use_container_width=True)

        with ncr:
            st.markdown('<div class="sec">National Performance by Region</div>',
                        unsafe_allow_html=True)
            nat_rows=[]
            for reg,rstates in US_REGIONS.items():
                rdf=df[df["state"].isin(rstates)]
                if not len(rdf): continue
                nat_rows.append({
                    "Region":reg,"States":len(rstates),"HCPs":len(rdf),
                    "High Value":(rdf["segment"]=="High Value").sum(),
                    "Growth":(rdf["segment"]=="Growth").sum(),
                    "KOLs":(rdf["opinion_leader_payments"]>0).sum(),
                    "Avg Score":round(float(rdf["targeting_score"].mean()),3),
                    "Q Calls Target":round(sum(
                        len(rdf[rdf["segment"]==sn]) * (91/cd)
                        for sn,cd in CALL_CADENCE_DAYS.items()
                    )),
                })
            if nat_rows:
                nr_df=pd.DataFrame(nat_rows)
                st.dataframe(nr_df,use_container_width=True,hide_index=True)

        # Q call targets by region
        st.markdown(f'<div class="sec">Q{q} Call Volume Targets by Region & Segment</div>',
                    unsafe_allow_html=True)
        tgt_rows=[]
        for reg,rstates in US_REGIONS.items():
            rdf=df[df["state"].isin(rstates)]
            for sn,cd in CALL_CADENCE_DAYS.items():
                n=len(rdf[rdf["segment"]==sn])
                if not n: continue
                calls_q=round(n*(91/cd))
                tgt_rows.append({"Region":reg,"Segment":sn,"HCPs":n,
                                 "Calls/Quarter":calls_q,"Calls/Month":round(calls_q/3),
                                 "Calls/Week":round(calls_q/13)})
        if tgt_rows:
            tdf=pd.DataFrame(tgt_rows)
            fig_tgt=px.bar(tdf,x="Region",y="Calls/Quarter",color="Segment",
                           color_discrete_map=SEG_COLORS,barmode="stack",
                           labels={"Calls/Quarter":"Quarterly Calls","Region":""})
            fig_tgt.update_layout(**CHART_LAYOUT,height=320,
                                  margin=dict(t=5,b=5,l=5,r=5),
                                  xaxis=dict(gridcolor="#F5F5F7"),
                                  yaxis=dict(gridcolor="#F5F5F7"),
                                  legend=dict(orientation="h",y=-0.2))
            st.plotly_chart(fig_tgt,use_container_width=True)
            st.dataframe(tdf,use_container_width=True,hide_index=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#8E8E93;font-size:0.72rem;padding:0.5rem 0'>
  Built by <strong style='color:#1D1D1F'>Zoraawar Nandwal</strong> &nbsp;·&nbsp;
  Python · PostgreSQL · Streamlit · Plotly &nbsp;·&nbsp;
  CMS Medicare Part D + Open Payments 2021–2022 &nbsp;·&nbsp;
  <a href='https://github.com/zorawarsinghnandwal/hcp-targeting-diabetes-analytics'
     style='color:#003DA5;text-decoration:none;font-weight:500'>GitHub →</a>
</div>
""", unsafe_allow_html=True)
