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
    page_title="HCP Analytics | Diabetes Portfolio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── LIGHT THEME CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }

    .stApp { background: #f1f5f9; }
    .stSidebar { background: #1e293b !important; }
    .stSidebar * { color: #e2e8f0 !important; }
    .stSidebar .stSelectbox label,
    .stSidebar .stSlider label,
    .stSidebar .stMultiSelect label {
        color: #94a3b8 !important; font-size: 0.7rem !important;
        font-weight: 700 !important; text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }

    .hero {
        background: linear-gradient(135deg, #1e40af 0%, #2563eb 60%, #3b82f6 100%);
        border-radius: 14px; padding: 1.8rem 2.5rem; margin-bottom: 1.2rem;
    }
    .hero-title { font-size: 1.65rem; font-weight: 800; color: white; margin: 0; letter-spacing: -0.02em; }
    .hero-sub   { font-size: 0.82rem; color: rgba(255,255,255,0.7); margin-top: 0.35rem; }
    .hero-tag   {
        display: inline-block; background: rgba(255,255,255,0.2); color: white;
        padding: 2px 10px; border-radius: 20px; font-size: 0.68rem;
        font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase;
        margin: 8px 4px 0 0;
    }

    div[data-testid="metric-container"] {
        background: white; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    div[data-testid="metric-container"] label { color: #64748b !important; font-size: 0.75rem !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 1.9rem !important; font-weight: 800 !important; color: #0f172a !important;
    }

    .sec {
        font-size: 0.68rem; font-weight: 700; color: #64748b;
        text-transform: uppercase; letter-spacing: 0.1em;
        margin-bottom: 0.7rem; padding-bottom: 0.4rem;
        border-bottom: 2px solid #e2e8f0;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: white; border-radius: 10px;
        padding: 4px; gap: 4px; border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; color: #64748b;
        border-radius: 8px; font-weight: 600; font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] { background: #2563eb !important; color: white !important; }

    .card {
        background: white; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin-bottom: 0.8rem;
    }

    .insight {
        background: #eff6ff; border: 1px solid #bfdbfe;
        border-left: 4px solid #2563eb; border-radius: 8px;
        padding: 0.9rem 1.1rem; font-size: 0.82rem;
        color: #1e40af; line-height: 1.6; margin-top: 0.8rem;
    }
    .insight strong { color: #1e3a8a; }

    .stDownloadButton > button {
        background: #2563eb !important; color: white !important;
        border: none !important; border-radius: 8px !important;
        font-weight: 600 !important; width: 100%; padding: 0.55rem !important;
    }

    .stDataFrame { border-radius: 10px !important; border: 1px solid #e2e8f0 !important; }

    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.06em;
        margin-right: 4px; margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── DATABASE ───────────────────────────────────────────────────────────────────
DB = dict(host="localhost", port=5432, dbname="postgres",
          user="postgres", password="newpassword123")

SEG_COLORS = {
    "High Value":   "#dc2626",
    "Growth":       "#16a34a",
    "Maintenance":  "#d97706",
    "Deprioritise": "#94a3b8",
}

CHART_LAYOUT = dict(
    paper_bgcolor="white", plot_bgcolor="white",
    font=dict(color="#0f172a", family="Inter, sans-serif"),
)

# ── HELPER FUNCTIONS ───────────────────────────────────────────────────────────

def loyalty_tier(hcp_row, full_df):
    """Derive brand loyalty tier from Rx fills and YoY growth."""
    p75   = full_df["fills_2022"].quantile(0.75)
    p25   = full_df["fills_2022"].quantile(0.25)
    fills = float(hcp_row.get("fills_2022") or 0)
    yoy   = float(hcp_row.get("yoy_growth_pct") or 0)
    if fills >= p75 and yoy > 0:
        return "Loyalist",     "#166534", "#dcfce7"
    elif fills >= p25 and yoy >= -10:
        return "Intermittent", "#92400e", "#fef3c7"
    elif fills > 0:
        return "Tourist",      "#991b1b", "#fee2e2"
    else:
        return "Non-Rx",       "#475569", "#f1f5f9"


def sim_calls(npi):
    """Reproducible simulated call history seeded by NPI (illustrative)."""
    try:
        seed = int(str(int(npi or 0))[-6:])
    except Exception:
        seed = 42
    rng      = np.random.RandomState(seed)
    types    = ["F2F Call", "P2P Call", "Virtual Meeting", "Sample Drop", "Email / Digital"]
    brands   = ["GLP-1 Agonists", "SGLT-2 Inhibitors", "DPP-4 Inhibitors", "Biguanides", "Sulfonylureas"]
    outcomes = ["Product detailed", "Samples requested", "Follow-up booked", "Left materials", "Event invited"]
    n, base  = int(rng.randint(3, 7)), 0
    rows = []
    for _ in range(n):
        base += int(rng.randint(18, 85))
        dt    = pd.Timestamp("2024-06-01") - pd.Timedelta(days=base)
        ctype = str(rng.choice(types))
        rows.append({
            "Date":    dt.strftime("%b %Y"),
            "Type":    ctype,
            "Mins":    str(int(rng.randint(5, 18))) if "Email" not in ctype else "—",
            "Brand":   str(rng.choice(brands)),
            "Outcome": str(rng.choice(outcomes)),
        })
    return pd.DataFrame(rows)


def brand_recs(hcp_row):
    """Rule-based brand + action recommendations for this HCP."""
    spec = str(hcp_row.get("specialty", "") or "").lower()
    vd   = int(hcp_row.get("volume_decile",  5) or 5)
    gd   = int(hcp_row.get("growth_decile",  5) or 5)
    kol  = float(hcp_row.get("opinion_leader_payments", 0) or 0) > 0
    rows = []
    if gd >= 8:
        rows.append(["1", "GLP-1 Agonists",      "Detail + Sample",   f"Growth D{gd}/10 — accelerating prescribing momentum"])
    if vd >= 8:
        rows.append([str(len(rows)+1), "SGLT-2 Inhibitors",  "Defend Rx Share",   f"Volume D{vd}/10 — protect existing prescription base"])
    if kol:
        rows.append([str(len(rows)+1), "Speaker / Advisory", "Advisory Board",    "Established KOL — industry engagement opportunity"])
    if "endo" in spec:
        rows.append([str(len(rows)+1), "DPP-4 Inhibitors",   "Sample + Educate",  "Endocrinologist — add-on therapy opportunity"])
    if "cardio" in spec:
        rows.append([str(len(rows)+1), "SGLT-2 Inhibitors",  "CV Outcome Data",   "Cardiologist — lead with cardiovascular benefit"])
    if not rows:
        rows.append(["1", "Biguanides",           "Educate + Build",   "Foundation diabetes therapy — establish relationship"])
    return pd.DataFrame(rows, columns=["#", "Brand / Programme", "Recommended Action", "Clinical Rationale"])


# ── DATA LOADING ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_conn():
    return psycopg2.connect(**DB)

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
    <div style='text-align:center;padding:1.2rem 0 0.8rem'>
      <div style='font-size:1.8rem'>🎯</div>
      <div style='font-size:1rem;font-weight:800;color:white;margin-top:4px'>HCP Intelligence</div>
      <div style='font-size:0.72rem;color:#64748b'>Diabetes Portfolio</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    states = ["🌎 All States"] + sorted(df["state"].dropna().unique().tolist())
    sel_st = st.selectbox("📍 State", states)
    st_val = None if sel_st == "🌎 All States" else sel_st

    specs  = ["All Specialties"] + sorted(df["specialty"].dropna().unique().tolist())
    sel_sp = st.selectbox("🏥 Specialty", specs)
    sp_val = None if sel_sp == "All Specialties" else sel_sp

    seg_sel = st.multiselect("🏷️ Segments",
        ["High Value", "Growth", "Maintenance", "Deprioritise"],
        default=["High Value", "Growth"])

    min_sc   = st.slider("⚡ Min Score", 0.0, 1.0, 0.5, 0.01)
    kol_only = st.toggle("⭐ KOL / Speaker Only", value=False)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.68rem;color:#475569;line-height:1.9'>
      <b style='color:#94a3b8'>Data Sources</b><br>
      CMS Medicare Part D 2021–22<br>
      CMS Open Payments 2021–22<br>
      NPPES NPI Registry 2026<br><br>
      <b style='color:#94a3b8'>Pipeline</b><br>
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
filt.index += 1

# ── HERO ───────────────────────────────────────────────────────────────────────
terr = f"{st_val or 'National'} · {sp_val or 'All Specialties'}"
st.markdown(f"""
<div class="hero">
  <div class="hero-title">🎯 HCP Targeting & Brand Performance Analytics</div>
  <div class="hero-sub">Diabetes Portfolio Intelligence Platform &nbsp;·&nbsp; {terr} &nbsp;·&nbsp; {datetime.now().strftime('%d %B %Y')}</div>
  <div>
    <span class="hero-tag">PostgreSQL</span>
    <span class="hero-tag">227K HCPs</span>
    <span class="hero-tag">83M+ Rows</span>
    <span class="hero-tag">CMS 2021–2022</span>
    <span class="hero-tag">Live Filters</span>
  </div>
</div>""", unsafe_allow_html=True)

# ── KPIs ───────────────────────────────────────────────────────────────────────
sc     = filt["segment"].value_counts()
hv_n   = sc.get("High Value", 0)
gr_n   = sc.get("Growth",     0)
avg_s  = filt["targeting_score"].mean()
kols_n = (filt["opinion_leader_payments"] > 0).sum()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("HCPs in View",          f"{len(filt):,}")
k2.metric("🔴 High Value",         f"{hv_n:,}")
k3.metric("🟢 Growth Opportunity", f"{gr_n:,}")
k4.metric("Avg Targeting Score",    f"{avg_s:.3f}" if not np.isnan(avg_s) else "—")
k5.metric("⭐ KOLs / Speakers",    f"{kols_n:,}")

st.markdown("---")

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋  Rep Command Centre",
    "📈  Market Intelligence",
    "🗺️  Territory Map",
    "⭐  Opinion Leaders",
    "🩺  HCP Profile",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — REP COMMAND CENTRE
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    col_main, col_side = st.columns([3, 2])

    with col_main:
        st.markdown('<div class="sec">Priority Call List — Top 200 HCPs by Targeting Score</div>', unsafe_allow_html=True)

        disp = filt.head(200)[[
            "last_name","first_name","credential","specialty","state","city",
            "fills_2022","yoy_growth_pct","volume_decile","growth_decile",
            "total_payment_usd","opinion_leader_payments","targeting_score","segment"
        ]].copy()
        disp.columns = [
            "Last","First","Cred","Specialty","St","City",
            "Fills 2022","YoY %","Vol D","Grw D",
            "Payments $","KOL","Score","Segment"
        ]
        disp["Fills 2022"]  = disp["Fills 2022"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
        disp["YoY %"]       = disp["YoY %"].apply(lambda x: f"+{x:.1f}%" if pd.notna(x) and x>=0 else (f"{x:.1f}%" if pd.notna(x) else ""))
        disp["Payments $"]  = disp["Payments $"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) and x>0 else "—")
        disp["KOL"]         = disp["KOL"].apply(lambda x: "⭐" if x>0 else "")
        disp["Score"]       = disp["Score"].apply(lambda x: f"{x:.3f}")

        def style_seg(v):
            return {
                "High Value":   "background-color:#fee2e2;color:#991b1b;font-weight:700",
                "Growth":       "background-color:#dcfce7;color:#166534;font-weight:700",
                "Maintenance":  "background-color:#fef3c7;color:#92400e;font-weight:600",
                "Deprioritise": "background-color:#f1f5f9;color:#64748b",
            }.get(v, "")

        styled = (disp.style
                  .applymap(style_seg, subset=["Segment"])
                  .applymap(lambda v: "color:#16a34a;font-weight:600" if "+" in str(v)
                            else ("color:#dc2626;font-weight:600" if str(v).startswith("-") else ""),
                            subset=["YoY %"]))
        st.dataframe(styled, use_container_width=True, height=500)

        csv_out = filt[["npi","last_name","first_name","credential","specialty",
                         "state","city","fills_2022","yoy_growth_pct",
                         "total_payment_usd","targeting_score","segment"]].copy()
        csv_out.columns = ["NPI","Last Name","First Name","Credential","Specialty",
                           "State","City","Fills 2022","YoY Growth %",
                           "Total Payments $","Targeting Score","Segment"]
        st.download_button(
            f"⬇️  Export Call List — {st_val or 'National'} ({len(filt):,} HCPs)",
            csv_out.to_csv(index=False),
            f"call_list_{(st_val or 'national').lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv", use_container_width=True
        )

    with col_side:
        st.markdown('<div class="sec">Segment Breakdown</div>', unsafe_allow_html=True)
        sd = filt["segment"].value_counts().reset_index()
        sd.columns = ["Segment", "Count"]
        fig_d = px.pie(sd, names="Segment", values="Count",
                       color="Segment", color_discrete_map=SEG_COLORS, hole=0.55)
        fig_d.update_traces(textposition="outside", textinfo="label+percent", textfont_size=11)
        fig_d.update_layout(**CHART_LAYOUT, height=260, showlegend=False, margin=dict(t=5,b=5,l=5,r=5))
        st.plotly_chart(fig_d, use_container_width=True)

        st.markdown('<div class="sec">Score Distribution</div>', unsafe_allow_html=True)
        fig_h = px.histogram(filt, x="targeting_score", color="segment",
                             color_discrete_map=SEG_COLORS, nbins=25, barmode="stack",
                             labels={"targeting_score": "Score", "segment": "Segment"})
        fig_h.update_layout(**CHART_LAYOUT, height=200, showlegend=False,
                            margin=dict(t=5,b=5,l=5,r=5),
                            xaxis=dict(gridcolor="#f1f5f9"),
                            yaxis=dict(gridcolor="#f1f5f9"))
        st.plotly_chart(fig_h, use_container_width=True)

        st.markdown('<div class="sec">Top Specialties</div>', unsafe_allow_html=True)
        ts = filt.groupby("specialty").size().reset_index(name="Count").nlargest(8, "Count")
        fig_ts = px.bar(ts, x="Count", y="specialty", orientation="h",
                        color="Count", color_continuous_scale=["#bfdbfe","#1d4ed8"],
                        labels={"specialty": "", "Count": "HCPs"})
        fig_ts.update_layout(**CHART_LAYOUT, height=260, coloraxis_showscale=False,
                             margin=dict(t=5,b=5,l=5,r=5),
                             xaxis=dict(gridcolor="#f1f5f9"),
                             yaxis=dict(gridcolor="#f1f5f9"))
        st.plotly_chart(fig_ts, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — MARKET INTELLIGENCE
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="sec">Drug Class Rx Trends (2021 → 2022)</div>', unsafe_allow_html=True)
        fig_tr = px.line(drug_df, x="year", y="total_fills", color="drug_class",
                         markers=True, line_shape="spline",
                         color_discrete_sequence=["#2563eb","#16a34a","#d97706","#dc2626","#7c3aed"],
                         labels={"total_fills": "Total Rx Fills", "drug_class": "Drug Class", "year": "Year"})
        fig_tr.update_traces(line_width=3, marker_size=12)
        fig_tr.update_layout(**CHART_LAYOUT, height=360,
                             margin=dict(t=5,b=5,l=5,r=5),
                             xaxis=dict(tickvals=[2021, 2022], gridcolor="#f1f5f9"),
                             yaxis=dict(gridcolor="#f1f5f9"),
                             legend=dict(orientation="h", y=-0.18))
        st.plotly_chart(fig_tr, use_container_width=True)

        st.markdown("""
        <div class="insight">
        💡 <strong>Key Commercial Insight:</strong> GLP-1 Agonists (Ozempic, Mounjaro, Wegovy) are the
        fastest-growing class — driven by dual diabetes + obesity indication. Biguanides (Metformin)
        lead volume but growth is flat. <strong>Strategic implication:</strong> prioritise GLP-1
        prescribers in the Growth segment before competitors lock in relationships.
        </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="sec">2022 Market Share</div>', unsafe_allow_html=True)
        s22 = drug_df[drug_df["year"] == 2022].groupby("drug_class")["total_fills"].sum().reset_index()
        s22.columns = ["Drug Class", "Fills"]
        fig_pie = px.pie(s22, names="Drug Class", values="Fills",
                         color_discrete_sequence=["#2563eb","#16a34a","#d97706","#dc2626","#7c3aed"],
                         hole=0.5)
        fig_pie.update_traces(textposition="outside", textinfo="percent+label", textfont_size=10)
        fig_pie.update_layout(**CHART_LAYOUT, height=280, showlegend=False, margin=dict(t=5,b=5,l=5,r=5))
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown('<div class="sec">YoY Growth by Drug Class</div>', unsafe_allow_html=True)
        yoy = drug_df.pivot(index="drug_class", columns="year", values="total_fills").reset_index()
        yoy.columns = ["Drug Class", "Fills 2021", "Fills 2022"]
        yoy["Growth %"] = ((yoy["Fills 2022"] - yoy["Fills 2021"]) / yoy["Fills 2021"] * 100).round(1)
        yoy["Fills 2021"] = yoy["Fills 2021"].apply(lambda x: f"{x/1e6:.1f}M")
        yoy["Fills 2022"] = yoy["Fills 2022"].apply(lambda x: f"{x/1e6:.1f}M")
        yoy["Growth %"]   = yoy["Growth %"].apply(lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%")
        st.dataframe(yoy.sort_values("Growth %", ascending=False),
                     use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<div class="sec">Avg YoY Rx Growth by Specialty — Top 20 (min 50 HCPs)</div>', unsafe_allow_html=True)
    sg = (df.groupby("specialty")
          .agg(avg_growth=("yoy_growth_pct","mean"), n=("npi","count"))
          .reset_index().dropna().query("n >= 50").nlargest(20, "avg_growth"))
    fig_sg = px.bar(sg, x="avg_growth", y="specialty", orientation="h",
                    text="avg_growth", color="avg_growth",
                    color_continuous_scale=["#bfdbfe","#1d4ed8"],
                    labels={"avg_growth": "Avg YoY Growth %", "specialty": ""})
    fig_sg.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_sg.update_layout(**CHART_LAYOUT, height=440, coloraxis_showscale=False,
                         margin=dict(t=5, b=5, l=5, r=80),
                         xaxis=dict(gridcolor="#f1f5f9"),
                         yaxis=dict(autorange="reversed", gridcolor="#f1f5f9"))
    st.plotly_chart(fig_sg, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — TERRITORY MAP
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    ctrl, _ = st.columns([2, 3])
    with ctrl:
        map_metric = st.selectbox("Colour map by:", [
            "High Value HCPs",
            "Growth Opportunity HCPs",
            "Total Rx Fills 2022",
            "Avg Targeting Score",
        ])

    sa = df.groupby("state").agg(
        total_hcps  =("npi","count"),
        high_value  =("segment", lambda x: (x == "High Value").sum()),
        growth      =("segment", lambda x: (x == "Growth").sum()),
        total_fills =("fills_2022","sum"),
        avg_score   =("targeting_score","mean"),
    ).reset_index()

    mcol = {
        "High Value HCPs":         "high_value",
        "Growth Opportunity HCPs": "growth",
        "Total Rx Fills 2022":     "total_fills",
        "Avg Targeting Score":     "avg_score",
    }[map_metric]

    fig_map = px.choropleth(
        sa, locations="state", locationmode="USA-states",
        color=mcol, scope="usa", color_continuous_scale="Blues",
        hover_name="state",
        hover_data={"high_value":True,"growth":True,"total_hcps":True,"avg_score":":.3f","state":False},
        labels={"high_value":"High Value","growth":"Growth",
                "total_hcps":"Total HCPs","avg_score":"Avg Score",
                "total_fills":"Total Fills", mcol: map_metric}
    )
    fig_map.update_layout(
        height=500,
        geo=dict(bgcolor="white", lakecolor="white", landcolor="#f8fafc", showlakes=False),
        paper_bgcolor="white",
        margin=dict(t=5, b=5, l=5, r=5),
        coloraxis_colorbar=dict(title=dict(text=map_metric)),
        font=dict(color="#0f172a")
    )
    st.plotly_chart(fig_map, use_container_width=True)

    ca, cb, cc = st.columns(3)
    with ca:
        st.markdown('<div class="sec">Top States — High Value HCPs</div>', unsafe_allow_html=True)
        t = sa.nlargest(10, "high_value")[["state","high_value","growth","total_hcps"]]
        t.columns = ["State","High Value","Growth","Total HCPs"]
        st.dataframe(t, use_container_width=True, hide_index=True)
    with cb:
        st.markdown('<div class="sec">Top States — Rx Volume</div>', unsafe_allow_html=True)
        t2 = sa.nlargest(10, "total_fills")[["state","total_fills","high_value","avg_score"]].copy()
        t2["total_fills"] = t2["total_fills"].apply(lambda x: f"{x/1e6:.1f}M")
        t2["avg_score"]   = t2["avg_score"].apply(lambda x: f"{x:.3f}")
        t2.columns = ["State","Total Fills","High Value","Avg Score"]
        st.dataframe(t2, use_container_width=True, hide_index=True)
    with cc:
        st.markdown('<div class="sec">Growth Opportunity by State</div>', unsafe_allow_html=True)
        t3 = sa.nlargest(10, "growth")[["state","growth","high_value","avg_score"]].copy()
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

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total KOLs",      f"{len(kol_df):,}")
    m2.metric("Avg Payment",     f"${kol_df['total_payment_usd'].mean():,.0f}" if len(kol_df) else "—")
    m3.metric("Max KOL Payment", f"${kol_df['total_payment_usd'].max():,.0f}"  if len(kol_df) else "—")
    m4.metric("High Value KOLs", f"{(kol_df['segment']=='High Value').sum():,}")

    st.markdown("---")
    col_kl, col_kc = st.columns([3, 2])

    with col_kl:
        st.markdown('<div class="sec">Speaker Bureau & Advisory Board Candidates</div>', unsafe_allow_html=True)
        kd = kol_df.head(100)[[
            "last_name","first_name","credential","specialty","state","city",
            "opinion_leader_payments","total_payment_usd","fills_2022","targeting_score","segment"
        ]].copy()
        kd.columns = ["Last","First","Cred","Specialty","St","City",
                      "Speaker Events","Total $","Fills 2022","Score","Segment"]
        kd["Total $"]    = kd["Total $"].apply(lambda x: f"${x:,.0f}")
        kd["Fills 2022"] = kd["Fills 2022"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
        kd["Score"]      = kd["Score"].apply(lambda x: f"{x:.3f}")
        sk = kd.style.applymap(lambda v: {
            "High Value":  "background-color:#fee2e2;color:#991b1b;font-weight:700",
            "Growth":      "background-color:#dcfce7;color:#166534;font-weight:700",
            "Maintenance": "background-color:#fef3c7;color:#92400e",
        }.get(v, ""), subset=["Segment"])
        st.dataframe(sk, use_container_width=True, height=460)

    with col_kc:
        st.markdown('<div class="sec">Industry Payments by Specialty</div>', unsafe_allow_html=True)
        ksp = (kol_df.groupby("specialty")
               .agg(total_pay=("total_payment_usd","sum"))
               .reset_index().nlargest(12, "total_pay"))
        fig_kol = px.bar(ksp, x="total_pay", y="specialty",
                         orientation="h", text="total_pay",
                         color="total_pay",
                         color_continuous_scale=["#bfdbfe","#1d4ed8"],
                         labels={"total_pay": "Total Payments ($)", "specialty": ""})
        fig_kol.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig_kol.update_layout(**CHART_LAYOUT, height=420,
                              coloraxis_showscale=False, margin=dict(t=5,b=5,l=5,r=120),
                              xaxis=dict(gridcolor="#f1f5f9"),
                              yaxis=dict(autorange="reversed", gridcolor="#f1f5f9"))
        st.plotly_chart(fig_kol, use_container_width=True)

        st.markdown("""
        <div class="insight">
        💡 <strong>Medical Affairs:</strong> KOLs with both high Rx volume and speaker/advisory
        engagement are priority candidates for Phase IV investigator programmes and
        brand ambassador initiatives.
        </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — HCP PROFILE DRILLDOWN
# ──────────────────────────────────────────────────────────────────────────────
with tab5:

    if len(filt) == 0:
        st.warning("No HCPs match the current sidebar filters. Adjust filters to see profiles.")
    else:
        st.markdown(
            '<div class="sec">🩺 HCP Profile Drilldown — Filter in the sidebar, then select any HCP</div>',
            unsafe_allow_html=True
        )

        top_n = filt.head(200).reset_index(drop=True)
        hcp_labels = [
            f"#{i+1}  {r['last_name']}, {r['first_name']} "
            f"{r['credential'] or ''} — {r['specialty']} · "
            f"{r['city']}, {r['state']} · Score {r['targeting_score']:.3f} · {r['segment']}"
            for i, (_, r) in enumerate(top_n.iterrows())
        ]

        sel_idx = st.selectbox(
            "Search or scroll to select HCP:",
            range(len(hcp_labels)),
            format_func=lambda i: hcp_labels[i],
            key="profile_sel"
        )

        hcp = top_n.iloc[sel_idx]

        # ── Derived values ─────────────────────────────────────────────────
        tier, tier_color, tier_bg = loyalty_tier(hcp, df)
        seg_col   = SEG_COLORS.get(str(hcp.get("segment", "")), "#94a3b8")
        is_kol    = float(hcp.get("opinion_leader_payments", 0) or 0) > 0
        first_n   = str(hcp.get("first_name", "") or "")
        last_n    = str(hcp.get("last_name",  "") or "")
        cred      = str(hcp.get("credential", "") or "")
        spec      = str(hcp.get("specialty",  "") or "N/A")
        city_s    = str(hcp.get("city",  "") or "")
        state_s   = str(hcp.get("state", "") or "")
        npi_s     = str(int(hcp.get("npi", 0) or 0))
        score     = float(hcp.get("targeting_score", 0) or 0)
        vd        = int(hcp.get("volume_decile", 5) or 5)
        gd        = int(hcp.get("growth_decile", 5) or 5)
        fills     = hcp.get("fills_2022", None)
        yoy_val   = hcp.get("yoy_growth_pct", None)
        total_pay = float(hcp.get("total_payment_usd", 0) or 0)
        kol_pay   = float(hcp.get("opinion_leader_payments", 0) or 0)
        initials  = f"{first_n[:1]}{last_n[:1]}".upper() or "HC"
        score_pct = score * 100
        cred_str  = f", {cred}" if cred else ""

        seg_badge  = f'<span class="badge" style="background:{seg_col};color:white">{hcp["segment"].upper()}</span>'
        tier_badge = f'<span class="badge" style="background:{tier_bg};color:{tier_color};border:1px solid {tier_color}40">{tier.upper()}</span>'
        kol_badge  = '<span class="badge" style="background:#fef9c3;color:#854d0e">⭐ KOL / SPEAKER</span>' if is_kol else ""

        # ── Profile header card ─────────────────────────────────────────────
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#f8fafc 0%,#eff6ff 100%);
                    border:1px solid #bfdbfe;border-radius:16px;
                    padding:1.8rem 2rem;margin-bottom:1.2rem;
                    position:relative;overflow:hidden;
                    box-shadow:0 4px 16px rgba(37,99,235,0.08)">
          <div style="position:absolute;left:0;top:0;bottom:0;width:5px;background:{seg_col}"></div>
          <div style="display:flex;align-items:flex-start;gap:1.5rem;padding-left:0.6rem">

            <div style="width:72px;height:72px;border-radius:50%;
                        background:linear-gradient(135deg,#1e40af,#3b82f6);
                        display:flex;align-items:center;justify-content:center;
                        font-size:1.6rem;font-weight:900;color:white;flex-shrink:0;
                        box-shadow:0 4px 12px rgba(37,99,235,0.25)">
              {initials}
            </div>

            <div style="flex:1;min-width:0">
              <div style="font-size:1.35rem;font-weight:900;color:#0f172a;letter-spacing:-0.02em">
                {first_n} {last_n}{cred_str}
              </div>
              <div style="color:#475569;font-size:0.88rem;margin-top:3px">
                📋 {spec} &nbsp;·&nbsp; 📍 {city_s}, {state_s}
              </div>
              <div style="color:#94a3b8;font-size:0.72rem;margin-top:2px;font-family:monospace">
                NPI: {npi_s}
              </div>
              <div style="margin-top:0.65rem">
                {seg_badge}{tier_badge}{kol_badge}
              </div>
            </div>

            <div style="text-align:center;background:white;border:2px solid {seg_col}30;
                        border-radius:14px;padding:1.1rem 1.6rem;min-width:160px;flex-shrink:0;
                        box-shadow:0 2px 8px rgba(0,0,0,0.06)">
              <div style="font-size:0.6rem;font-weight:700;color:#94a3b8;
                          text-transform:uppercase;letter-spacing:0.1em">Targeting Score</div>
              <div style="font-size:2.8rem;font-weight:900;color:{seg_col};line-height:1.1;margin:4px 0">
                {score:.3f}
              </div>
              <div style="background:#e2e8f0;border-radius:4px;height:7px;margin:6px 0;overflow:hidden">
                <div style="background:{seg_col};width:{score_pct:.0f}%;height:100%;border-radius:4px"></div>
              </div>
              <div style="font-size:0.65rem;color:#64748b">Vol D{vd}/10 &nbsp;·&nbsp; Grwth D{gd}/10</div>
              <div style="font-size:0.6rem;color:#94a3b8;margin-top:2px">Rank #{sel_idx+1} of {len(filt):,}</div>
            </div>

          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── KPI row ─────────────────────────────────────────────────────────
        pm1, pm2, pm3, pm4, pm5 = st.columns(5)
        fills_str = f"{int(fills):,}" if pd.notna(fills) else "—"
        yoy_str   = (f"+{yoy_val:.1f}%" if yoy_val >= 0 else f"{yoy_val:.1f}%") if pd.notna(yoy_val) else "—"
        pay_str   = f"${total_pay:,.0f}" if total_pay > 0 else "No record"
        kol_str   = f"{int(kol_pay)} events" if kol_pay > 0 else "None"

        pm1.metric("Rx Fills 2022",  fills_str)
        pm2.metric("YoY Rx Growth",  yoy_str)
        pm3.metric("Total Payments", pay_str)
        pm4.metric("Speaker Events", kol_str)
        pm5.metric("Loyalty Tier",   tier)

        st.markdown("---")

        # ── Brand recommendations + Score bar chart ──────────────────────────
        col_recs, col_gauge = st.columns([3, 2])

        with col_recs:
            st.markdown('<div class="sec">Brand Priority Recommendations</div>', unsafe_allow_html=True)
            recs_df = brand_recs(hcp)
            recs_styled = recs_df.style.applymap(
                lambda v: "background-color:#dcfce7;color:#166534;font-weight:700" if "Sample" in str(v)
                else ("background-color:#fee2e2;color:#991b1b;font-weight:700" if "Defend" in str(v)
                else ("background-color:#fef9c3;color:#854d0e;font-weight:700" if "Advisory" in str(v) else "")),
                subset=["Recommended Action"]
            )
            st.dataframe(recs_styled, use_container_width=True, hide_index=True, height=210)

        with col_gauge:
            st.markdown('<div class="sec">Score Component Breakdown</div>', unsafe_allow_html=True)
            vol_score    = vd / 10
            growth_score = gd / 10
            pay_score    = min(total_pay / 50000, 1.0)

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                x=["Volume\nDecile", "Growth\nDecile", "Payment\nScore", "Composite"],
                y=[vol_score, growth_score, pay_score, score],
                marker_color=["#3b82f6", "#16a34a", "#d97706", seg_col],
                text=[f"{v:.2f}" for v in [vol_score, growth_score, pay_score, score]],
                textposition="outside",
                textfont=dict(size=12, color="#0f172a"),
            ))
            fig_comp.update_layout(
                **CHART_LAYOUT, height=230,
                margin=dict(t=30, b=5, l=5, r=5),
                yaxis=dict(range=[0, 1.2], gridcolor="#f1f5f9", tickformat=".1f"),
                xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                showlegend=False,
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        # ── Call history + Action brief ──────────────────────────────────────
        st.markdown("---")
        col_calls, col_brief = st.columns([3, 2])

        with col_calls:
            st.markdown(
                '<div class="sec">Simulated Call History '
                '<span style="font-size:0.6rem;color:#94a3b8;font-style:italic;'
                'text-transform:none;font-weight:400">— Illustrative, based on prescribing profile</span></div>',
                unsafe_allow_html=True
            )
            calls_df = sim_calls(hcp.get("npi", 42))
            calls_styled = calls_df.style.applymap(
                lambda v: "color:#2563eb;font-weight:600" if "F2F" in str(v) or "P2P" in str(v)
                else ("color:#16a34a;font-weight:600" if "Virtual" in str(v)
                else ("color:#94a3b8" if "Email" in str(v) else "")),
                subset=["Type"]
            )
            st.dataframe(calls_styled, use_container_width=True, hide_index=True)

        with col_brief:
            growth_word = "accelerating" if gd >= 8 else ("stable" if gd >= 5 else "declining")
            if is_kol:
                action_txt = (f"As a KOL with <strong>${total_pay:,.0f}</strong> in industry "
                              "engagements, prioritise advisory board and speaker programme outreach.")
            elif gd >= 7:
                action_txt = "Target with <strong>GLP-1 Agonist detailing + sampling</strong> on next territory visit."
            else:
                action_txt = "Target with <strong>SGLT-2 maintenance calls</strong> to defend existing Rx share."

            seg_priority = {
                "High Value":   "🔴 MUST CALL — High Value HCP",
                "Growth":       "🟢 SCHEDULE — High growth opportunity",
                "Maintenance":  "🟡 MAINTAIN — Scheduled call cadence",
                "Deprioritise": "⚪ LOW PRIORITY — Deprioritise territory",
            }.get(str(hcp.get("segment","")), "")

            st.markdown(f"""
            <div class="insight" style="margin-top:0.4rem">
            💡 <strong>Rep Action Brief</strong><br><br>
            <strong>{first_n} {last_n}</strong> is a <strong>{tier}</strong> prescriber
            with {growth_word} diabetes Rx trajectory.<br><br>
            {action_txt}<br><br>
            <strong>Call Priority:</strong> {seg_priority}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)
            profile_csv = pd.DataFrame([{
                "NPI": npi_s, "Name": f"{first_n} {last_n}", "Credential": cred,
                "Specialty": spec, "City": city_s, "State": state_s,
                "Fills 2022": fills, "YoY Growth %": yoy_val,
                "Volume Decile": vd, "Growth Decile": gd,
                "Total Payments": total_pay, "Speaker Events": kol_pay,
                "Targeting Score": score, "Segment": hcp.get("segment",""),
                "Loyalty Tier": tier,
            }])
            st.download_button(
                "⬇️  Export This HCP Profile",
                profile_csv.to_csv(index=False),
                f"hcp_profile_{last_n.lower()}_{npi_s[-4:]}.csv",
                "text/csv", use_container_width=True
            )

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#94a3b8;font-size:0.75rem;padding:0.4rem 0'>
  Built by <strong style='color:#64748b'>Zoraawar Nandwal</strong> &nbsp;·&nbsp;
  Python · PostgreSQL · Streamlit · Plotly &nbsp;·&nbsp;
  CMS Medicare Part D + Open Payments 2021–2022 &nbsp;·&nbsp;
  <a href='https://github.com/zorawar91/hcp-targeting-diabetes-analytics'
     style='color:#2563eb;text-decoration:none'>GitHub →</a>
</div>
""", unsafe_allow_html=True)
