"""SmartQ brand styling (Brand Guideline 09/2022) and chart chrome."""
import plotly.graph_objects as go
import streamlit as st

BRAND_YELLOW = "#FCC529"
BRAND_DARK = "#333333"
INK = "#333333"
INK_SECONDARY = "#5f5d58"
INK_MUTED = "#8a8884"
GRID_LINE = "#e8e6e0"
AXIS_LINE = "#c9c7c0"

# Fixed counter → color mapping for chart marks. Brand hues snapped to the
# nearest steps that pass categorical-palette accessibility checks (raw
# FCC529 / 3F99A8 are too light/gray for data marks). Never re-ordered.
COUNTER_COLORS = {
    "North Non Veg": "#155493",   # brand blue
    "North Veg": "#0E93AE",       # brand cyan, chroma-corrected
    "South Non Veg": "#E0A80A",   # brand yellow, deepened for marks
    "South Veg": "#ED6940",       # brand orange
}
# marks light enough to need dark ink for inside labels
LIGHT_MARKS = ("#E0A80A", "#0E93AE")

RISK_BADGES = {
    "LOW": ("✓ LOW", "#1a7d1a", "#e9f3e7"),
    "MEDIUM": ("⚠ MEDIUM", "#8a5a00", "#fdf3d9"),
    "HIGH": ("▲ HIGH", "#c03333", "#fbe9e7"),
}

PLOTLY_CONFIG = {"displayModeBar": False}

_BRAND_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

  html, body, [class*="st-"], [data-testid="stAppViewContainer"] * {
    font-family: 'Poppins', system-ui, -apple-system, sans-serif;
  }
  /* keep Streamlit's material icon glyphs on their icon font */
  span[data-testid="stIconMaterial"], [data-testid="stExpanderToggleIcon"],
  [data-testid="stFileUploaderDropzone"] span[translate="no"] {
    font-family: 'Material Symbols Rounded' !important;
  }
  .block-container {padding-top: 3.4rem; max-width: 1200px;}

  /* multiselect chips — dark ink pills (white-on-yellow fails contrast) */
  [data-baseweb="tag"] {background:#333333 !important; border-radius:999px !important;}
  [data-baseweb="tag"] span, [data-baseweb="tag"] svg {color:#ffffff !important; fill:#ffffff !important;}
  h1, h2, h3 {font-family:'Poppins',sans-serif; font-weight:700; color:#333333;
              letter-spacing:-0.01em;}

  /* brand running head, like the guideline pages */
  .sq-topline {display:flex; align-items:center; gap:.9rem; margin-bottom:.4rem;}
  .sq-topline .sq-label {font-size:.72rem; font-weight:700; letter-spacing:.14em;
                         color:#333333; white-space:nowrap;}
  .sq-topline .sq-rule {flex:1; height:2px; background:#333333; opacity:.85;}

  @keyframes rise {from {opacity:0; transform:translateY(6px);} to {opacity:1; transform:none;}}
  .fc-card {
    background:#ffffff; border:1px solid rgba(51,51,51,0.08); border-radius:18px;
    padding: 1rem 1.15rem; height:100%;
    box-shadow: 0 1px 3px rgba(51,51,51,0.06);
    transition: box-shadow .18s ease, transform .18s ease;
    animation: rise .28s ease both;
  }
  .fc-card:hover {box-shadow: 0 8px 22px rgba(51,51,51,0.10); transform: translateY(-2px);}
  .fc-kicker {font-size:.72rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase;
              color:#5f5d58; display:flex; align-items:center; gap:.45rem;}
  .fc-dot {width:.62rem; height:.62rem; border-radius:50%; display:inline-block; flex:none;}
  .fc-value {font-size:2.05rem; font-weight:800; color:#333333; line-height:1.15; margin:.15rem 0 .1rem;}
  .fc-sub {font-size:.82rem; color:#8a8884;}
  .fc-row {display:flex; justify-content:space-between; font-size:.86rem; color:#5f5d58;
           padding:.28rem 0; border-top:1px dashed #e8e6e0;}
  .fc-row b {color:#333333; font-variant-numeric: tabular-nums;}
  .fc-badge {display:inline-block; font-size:.72rem; font-weight:700; letter-spacing:.04em;
             padding:.18rem .55rem; border-radius:999px;}
  .fc-why {font-size:.8rem; color:#5f5d58; margin-top:.55rem; line-height:1.45;
           border-left:3px solid #FCC529; padding-left:.6rem;}

  /* hero — brand yellow block, dark ink (guideline cover style) */
  .fc-hero {background:#FCC529; border:none; border-radius:20px;
            padding:1.2rem 1.5rem; animation: rise .28s ease both;
            box-shadow: 0 2px 10px rgba(252,197,41,0.35);}
  .fc-hero .fc-kicker {color:#333333; opacity:.75;}
  .fc-hero .fc-value {font-size:2.7rem; color:#333333;}
  .fc-hero .fc-sub {color:#333333; opacity:.7; font-weight:500;}

  div[data-testid="stMetric"] {
    background:#ffffff; border:1px solid rgba(51,51,51,0.08); border-radius:16px;
    padding:.8rem 1rem; box-shadow: 0 1px 3px rgba(51,51,51,0.06);
  }
  div[data-testid="stMetricLabel"] {color:#5f5d58;}

  .stTabs [data-baseweb="tab-list"] {gap:.35rem; border-bottom:2px solid #e8e6e0;}
  .stTabs [data-baseweb="tab"] {border-radius:12px 12px 0 0; padding:.55rem 1.1rem; font-weight:600;}
  .stTabs [aria-selected="true"] {background: rgba(252,197,41,0.18);}

  /* buttons — yellow pill with dark ink, per logo lockup */
  div.stButton > button[kind="primary"] {
    background:#FCC529; color:#333333; font-weight:700; border:none;
    border-radius:999px; padding:.6rem 1.2rem;
    transition: filter .15s ease, transform .15s ease;
  }
  div.stButton > button[kind="primary"]:hover {filter:brightness(.95); transform:translateY(-1px);
    color:#333333; background:#FCC529;}
  div.stButton > button, div.stDownloadButton > button {border-radius:999px; font-weight:600;}

  /* sidebar — brand yellow panel */
  section[data-testid="stSidebar"] {background:#FCC529; border-right:none;}
  section[data-testid="stSidebar"] * {color:#333333;}
  section[data-testid="stSidebar"] hr {border-color: rgba(51,51,51,0.25);}
  .sq-logo {font-size:1.5rem; font-weight:800; letter-spacing:.06em; color:#333333;}
  .sq-logo .q {color:#ffffff;}
  .sq-compass {font-size:.6rem; font-weight:700; letter-spacing:.18em; color:#333333;
               opacity:.8; margin-top:-.2rem;}
  .sq-appname {font-size:1.05rem; font-weight:700; color:#333333; margin-top:1.1rem;}
  .sq-note {background: rgba(255,255,255,0.55); border-radius:14px; padding:.7rem .85rem;
            font-size:.8rem; color:#333333; line-height:1.5;}
</style>
"""


def inject_brand_styles() -> None:
    st.markdown(_BRAND_CSS, unsafe_allow_html=True)


def render_running_head() -> None:
    st.markdown(
        '<div class="sq-topline"><span class="sq-label">SMARTQ · DEMAND FORECAST</span>'
        '<span class="sq-rule"></span></div>',
        unsafe_allow_html=True)


def render_sidebar_wordmark() -> None:
    st.markdown(
        '<div class="sq-logo">SMART<span class="q">Q</span></div>'
        '<div class="sq-compass">A COMPASS GROUP COMPANY</div>'
        '<div class="sq-appname">Lunch Counter Demand Forecast</div>',
        unsafe_allow_html=True)


def hero_card(kicker: str, value: str, subtitle: str) -> str:
    return (f'<div class="fc-hero"><div class="fc-kicker">{kicker}</div>'
            f'<div class="fc-value">{value}</div>'
            f'<div class="fc-sub">{subtitle}</div></div>')


def chart_layout(fig: go.Figure, height: int = 380, **layout_kwargs) -> go.Figure:
    """Apply the shared brand chart chrome to a Plotly figure."""
    fig.update_layout(
        height=height,
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(family="Poppins, system-ui, -apple-system, sans-serif",
                  color=INK_SECONDARY, size=13),
        margin=dict(l=10, r=30, t=64, b=10),
        hoverlabel=dict(bgcolor=INK, font_color="#ffffff",
                        font_size=12, bordercolor=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    x=1, xanchor="right", font=dict(size=12, color=INK_SECONDARY)),
        title_x=0, title_xanchor="left",
        **layout_kwargs,
    )
    fig.update_xaxes(gridcolor=GRID_LINE, linecolor=AXIS_LINE, zeroline=False,
                     tickfont=dict(color=INK_MUTED))
    fig.update_yaxes(gridcolor=GRID_LINE, linecolor=AXIS_LINE, zeroline=False,
                     tickfont=dict(color=INK_MUTED))
    return fig
