"""Siemens Lunch Counter Demand — Streamlit frontend.

A clean, guided UI over the frozen LightGBM model bundle in ./siemens_model_bundle:
  * Forecast — build or upload a menu plan, get per-counter predictions with
    calibrated P10-P90 ranges, suggested orders, risk levels and explanations.
  * History Explorer — trends, weekday seasonality and counter shares.
  * Model Performance — locked test metrics, baselines and diagnostic figures.

Run:  streamlit run app.py
"""
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
BUNDLE = ROOT / "siemens_model_bundle"
sys.path.insert(0, str(BUNDLE))

from features import build_all, CAT_FEATURES, NUM_FEATURES, PAN_FLAGS  # noqa: E402

# ---------------------------------------------------------------------------
# MVP_MODE = True  →  lean MVP: plan in, numbers out (prediction, range,
# suggested order, risk) + evaluation metrics on trained data. Explanations,
# result cards, charts, history explorer and the about page are disabled
# ("commented out") for now. Flip to False to restore the full tool.
# ---------------------------------------------------------------------------
MVP_MODE = True

DEFAULT_HISTORY = ROOT / "Lunch_Master_Data_FINAL(cleaned).xlsx"
COUNTERS = ["North Non Veg", "North Veg", "South Non Veg", "South Veg"]
WEEKDAY_LEVELS = ["Friday", "Monday", "Thursday", "Tuesday", "Wednesday"]
DAY_TYPES = ["Regular", "Previous Day of Holiday", "Next Day of Holiday"]

# SmartQ brand (Brand Guideline 09/2022): yellow FCC529, blue 155493,
# cyan 3F99A8, orange ED6940, dark gray 333333, Poppins typography.
BRAND_YELLOW, BRAND_DARK = "#FCC529", "#333333"

# Fixed entity → color mapping for chart marks. Brand hues, snapped to the
# nearest steps that pass the categorical-palette checks (raw FCC529/3F99A8
# are too light/gray for data marks); never re-ordered.
COUNTER_COLORS = {
    "North Non Veg": "#155493",   # brand blue
    "North Veg": "#0E93AE",       # brand cyan, chroma-corrected
    "South Non Veg": "#E0A80A",   # brand yellow, deepened for marks
    "South Veg": "#ED6940",       # brand orange
}
RISK_STYLE = {
    "LOW": ("✓ LOW", "#1a7d1a", "#e9f3e7"),
    "MEDIUM": ("⚠ MEDIUM", "#8a5a00", "#fdf3d9"),
    "HIGH": ("▲ HIGH", "#c03333", "#fbe9e7"),
}
INK, INK2, MUTED, GRID = "#333333", "#5f5d58", "#8a8884", "#e8e6e0"

st.set_page_config(
    page_title="Lunch Counter Forecast",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------- styling (SmartQ brand) ----
st.markdown("""
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

  /* brand top strip, like the guideline running head */
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
""", unsafe_allow_html=True)


# ------------------------------------------------------------ data & model ---
@st.cache_data(show_spinner="Loading history workbook…")
def load_history(file_bytes: bytes | None) -> pd.DataFrame:
    src = file_bytes if file_bytes is not None else DEFAULT_HISTORY
    hist = pd.read_excel(src, sheet_name="Lunch Master")
    hist["Date"] = pd.to_datetime(hist["Date"])
    return hist


@st.cache_resource(show_spinner="Loading frozen model bundle…")
def load_models():
    import pickle
    import lightgbm as lgb

    art = BUNDLE / "artifacts"
    with open(art / "final_config.pkl", "rb") as f:
        cfg = pickle.load(f)
    boosters = {q: lgb.Booster(model_file=str(art / f"model_{q}.txt"))
                for q in ("point", "q10", "q75", "q90")}
    return boosters, cfg


@st.cache_data(show_spinner=False)
def counter_day_history(hist: pd.DataFrame) -> pd.DataFrame:
    """Counter-day aggregate of the raw item-level history, for the explorer."""
    g = hist.groupby(["Date", "Counter Name"], as_index=False).agg(
        consumed=("Counter Consumed", "first"),
        ordered=("Counter Ordered", "first"),
        total=("Total Lunch Consumed", "first"),
        weekday=("Weekday", "first"),
        items=("Item Name", "nunique"),
    )
    return g.sort_values(["Date", "Counter Name"])


@st.cache_data(show_spinner=False)
def item_catalog(hist: pd.DataFrame):
    """Per-counter item lists (frequency-ordered), item→category map, last menus."""
    cat_map = (hist.groupby("Item Name")["Category"]
               .agg(lambda s: s.mode().iat[0]).to_dict())
    per_counter = {
        c: (hist[hist["Counter Name"] == c]["Item Name"]
            .value_counts().index.tolist())
        for c in COUNTERS
    }
    last_menu = {}
    for c in COUNTERS:
        sub = hist[hist["Counter Name"] == c]
        if len(sub):
            last_date = sub["Date"].max()
            last_menu[c] = sorted(sub.loc[sub["Date"] == last_date, "Item Name"].unique())
        else:
            last_menu[c] = []
    return per_counter, cat_map, last_menu


def explain(row) -> str:
    """Plain-language drivers, mirrored from the bundle's predict.py."""
    bits = []
    if row["has_nv_biryani"]:
        bits.append("non-veg biryani on the menu (historically the strongest pull item)")
    elif row["star_score"] >= 4:
        bits.append("a very-high-pull star item on the menu")
    if row["oth_has_nv_biryani"] and not row["has_nv_biryani"]:
        bits.append("a competing counter serves non-veg biryani (drains this counter)")
    if row["star_minus_oth"] > 1:
        bits.append("this counter has the strongest menu among active counters today")
    if row["dt_prev_holiday"] or row["dt_next_holiday"]:
        bits.append("holiday-adjacent day (attendance typically drops)")
    if row["weekday"] == "Friday":
        bits.append("Friday (lowest-attendance weekday)")
    if row["weekday"] in ("Tuesday", "Wednesday"):
        bits.append(f"{row['weekday']} (peak-attendance weekday)")
    bits.append(f"recent same-weekday average for this counter is {row['wd_roll4']:.0f}")
    return "; ".join(bits).capitalize() + "."


def run_forecast(hist: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    """Score a plan exactly as the bundle's predict.py CLI does."""
    plan = plan.copy()
    plan["Date"] = pd.to_datetime(plan["Date"])
    for col, default in [("Day Type", "Regular"), ("Panchangam", "Regular")]:
        if col not in plan:
            plan[col] = default
        plan[col] = plan[col].fillna(default)
    plan["Month"] = plan["Date"].dt.month_name()
    plan["Weekday"] = plan["Date"].dt.day_name()
    for col in ["Receiving Qty", "Bainmarie Wastage"]:
        plan[col] = np.nan
    for col in ["Headcount", "Total Lunch Consumed", "Counter Ordered", "Counter Consumed"]:
        plan[col] = 0

    full = pd.concat([hist, plan[hist.columns]], ignore_index=True)
    cd = build_all(full)
    target = cd[cd["Date"].isin(plan["Date"].unique())].copy()

    models, cfg = load_models()
    cat_levels = {"Counter Name": COUNTERS, "weekday": WEEKDAY_LEVELS}
    X = target[NUM_FEATURES + CAT_FEATURES].copy()
    for c in CAT_FEATURES:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])

    target["pred"] = np.clip(models["point"].predict(X), 0, None).round(0)
    target["lo"] = np.clip(np.clip(models["q10"].predict(X), 0, None) - cfg["cqr_Q80"], 0, None).round(0)
    target["hi"] = (np.clip(models["q90"].predict(X), 0, None) + cfg["cqr_Q80"]).round(0)
    target["order"] = (np.clip(models["q75"].predict(X), 0, None) + cfg["order_corr"]).round(-1)
    width = (target["hi"] - target["lo"]) / target["pred"].clip(lower=1)
    target["risk"] = np.select([width > 0.45, width > 0.30], ["HIGH", "MEDIUM"], "LOW")
    if not MVP_MODE:  # plain-language explanations — disabled in the MVP
        target["why"] = target.apply(explain, axis=1)
    return target.sort_values(["Date", "pred"], ascending=[True, False])


def base_layout(fig: go.Figure, height=380, **kw):
    fig.update_layout(
        height=height,
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(family='Poppins, system-ui, -apple-system, sans-serif',
                  color=INK2, size=13),
        margin=dict(l=10, r=30, t=64, b=10),
        hoverlabel=dict(bgcolor="#333333", font_color="#ffffff",
                        font_size=12, bordercolor="#333333"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    x=1, xanchor="right", font=dict(size=12, color=INK2)),
        title_x=0, title_xanchor="left",
        **kw,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor="#c9c7c0", zeroline=False,
                     tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor="#c9c7c0", zeroline=False,
                     tickfont=dict(color=MUTED))
    return fig


PLOTLY_CFG = {"displayModeBar": False}


# --------------------------------------------------------------- sidebar -----
with st.sidebar:
    st.markdown(
        '<div class="sq-logo">SMART<span class="q">Q</span></div>'
        '<div class="sq-compass">A COMPASS GROUP COMPANY</div>'
        '<div class="sq-appname">Lunch Counter Demand Forecast</div>',
        unsafe_allow_html=True)
    st.caption("LightGBM · Poisson objective · frozen bundle, "
               "conformalized P10–P90 intervals.")
    st.divider()

    st.markdown("**History data**")
    up_hist = st.file_uploader(
        "Replace bundled history (.xlsx)", type=["xlsx"],
        help="Sheet 'Lunch Master', same columns as the master workbook. "
             "Keep it current through the last served day — every lag feature "
             "is rebuilt from it.",
    )
    try:
        hist = load_history(up_hist.getvalue() if up_hist else None)
    except Exception as e:  # bad upload → fall back loudly
        st.error(f"Could not read that workbook: {e}")
        hist = load_history(None)

    last_day = hist["Date"].max()
    st.markdown(
        f'<div class="sq-note">History loaded ✓<br>'
        f'<b>{hist["Date"].min():%d %b %Y} → {last_day:%d %b %Y}</b><br>'
        f'{hist.shape[0]:,} item rows · {hist["Date"].nunique()} working days</div>',
        unsafe_allow_html=True)
    if not MVP_MODE:  # sidebar model-stats blurb — disabled in the MVP
        st.divider()
        st.markdown(
            "**Locked June test**  \n"
            "Counter WAPE **6.06%** · Day WAPE **3.54%**  \n"
            "vs moving-average practice **26.4%**"
        )
        st.caption("Predictions use only information available at vendor-ordering "
                   "time (the evening before service).")

cd_hist = counter_day_history(hist)
items_by_counter, item_to_cat, last_menus = item_catalog(hist)

# ---------------------------------------------------------------- header -----
st.markdown(
    '<div class="sq-topline"><span class="sq-label">SMARTQ · DEMAND FORECAST</span>'
    '<span class="sq-rule"></span></div>',
    unsafe_allow_html=True)
st.title("Lunch counter demand forecast")
if MVP_MODE:
    st.caption("MVP — menu plan in, numbers out: per-counter demand, calibrated "
               "range, suggested order and risk, one day ahead of service.")
    tab_fc, tab_perf = st.tabs(["🔮  Forecast", "📈  Evaluation metrics"])
    tab_hist = tab_about = None
else:
    st.caption("Plan tomorrow's menu → get per-counter demand, a calibrated range, "
               "a suggested order quantity and the risk level — one day ahead of service.")
    tab_fc, tab_hist, tab_perf, tab_about = st.tabs(
        ["🔮  Forecast", "📊  History explorer", "📈  Model performance", "ℹ️  About the model"]
    )

# ══════════════════════════════════════════════════════════════ FORECAST ═════
with tab_fc:
    st.session_state.setdefault("plan_days", {})   # {date_iso: plan rows DataFrame}
    st.session_state.setdefault("forecast", None)

    left, right = st.columns([1.05, 1], gap="large")

    # ---- input side -----------------------------------------------------
    with left:
        st.subheader("1 · Menu plan")
        mode = st.radio("How do you want to provide the plan?",
                        ["🧾 Build it here", "📤 Upload plan.csv"],
                        horizontal=True, label_visibility="collapsed")

        if mode == "📤 Upload plan.csv":
            with open(BUNDLE / "plan_template.csv", "rb") as f:
                st.download_button("Download plan template", f,
                                   file_name="plan_template.csv", mime="text/csv")
            up_plan = st.file_uploader("plan.csv — one row per planned item", type=["csv"])
            if up_plan is not None:
                try:
                    pdf = pd.read_csv(up_plan)
                    missing = {"Date", "Counter Name", "Item Name", "Category"} - set(pdf.columns)
                    if missing:
                        st.error(f"Missing columns: {', '.join(sorted(missing))}")
                    else:
                        pdf["Date"] = pd.to_datetime(pdf["Date"])
                        bad_counters = set(pdf["Counter Name"]) - set(COUNTERS)
                        if bad_counters:
                            st.error(f"Unknown counters: {', '.join(sorted(bad_counters))}")
                        else:
                            st.session_state.plan_days = {
                                d.date().isoformat(): g.reset_index(drop=True)
                                for d, g in pdf.groupby("Date")
                            }
                            st.success(f"Plan loaded — {pdf['Date'].nunique()} day(s), "
                                       f"{len(pdf)} item rows.")
                except Exception as e:
                    st.error(f"Could not parse that CSV: {e}")

        else:
            default_day = (last_day + timedelta(days=1)).date()
            while default_day.weekday() >= 5:
                default_day += timedelta(days=1)
            c1, c2 = st.columns(2)
            plan_date = c1.date_input("Service date", value=default_day,
                                      min_value=(last_day + timedelta(days=1)).date())
            day_type = c2.selectbox("Day type", DAY_TYPES)
            pan_sel = st.multiselect("Panchangam observances (if any)", PAN_FLAGS,
                                     help="Leave empty for a Regular day.")
            pan_val = "; ".join(pan_sel) if pan_sel else "Regular"

            if plan_date.weekday() >= 5:
                st.warning("That's a weekend — the model was trained on working days "
                           "(Mon–Fri) only and can't score it.")

            active = st.multiselect("Active counters", COUNTERS,
                                    default=[c for c in COUNTERS if c != "North Non Veg"],
                                    help="Closed counters are simply left out.")
            if "North Non Veg" in active:
                st.info("North Non Veg ran only 18 historical days — treat its "
                        "forecast as low-confidence.", icon="ℹ️")

            rows = []
            for c in active:
                with st.expander(f"{c} — menu", expanded=True):
                    sel = st.multiselect(
                        f"Items for {c}", options=items_by_counter[c],
                        default=[i for i in last_menus[c] if i in items_by_counter[c]],
                        key=f"menu_{c}",
                        help="Pre-filled with this counter's most recent menu — edit freely.",
                    )
                    for it in sel:
                        rows.append({"Date": pd.Timestamp(plan_date), "Counter Name": c,
                                     "Item Name": it, "Category": item_to_cat.get(it, "Veg Gravy"),
                                     "Day Type": day_type, "Panchangam": pan_val})

            with st.expander("➕ Add items not in the list"):
                extra = st.data_editor(
                    pd.DataFrame(columns=["Counter Name", "Item Name", "Category"]),
                    num_rows="dynamic", use_container_width=True, key="extra_items",
                    column_config={
                        "Counter Name": st.column_config.SelectboxColumn(options=COUNTERS, required=True),
                        "Item Name": st.column_config.TextColumn(required=True),
                        "Category": st.column_config.SelectboxColumn(
                            options=sorted(hist["Category"].unique()), required=True),
                    },
                )
                for _, r in extra.dropna(subset=["Counter Name", "Item Name", "Category"]).iterrows():
                    if r["Counter Name"] in active:
                        rows.append({"Date": pd.Timestamp(plan_date), "Counter Name": r["Counter Name"],
                                     "Item Name": r["Item Name"], "Category": r["Category"],
                                     "Day Type": day_type, "Panchangam": pan_val})

            add_col, clear_col = st.columns(2)
            if add_col.button("➕ Add this day to the plan", use_container_width=True,
                              disabled=not rows or plan_date.weekday() >= 5):
                st.session_state.plan_days[plan_date.isoformat()] = pd.DataFrame(rows)
                st.toast(f"Added {plan_date:%a %d %b} — {len(rows)} items.", icon="✅")
            if clear_col.button("🗑️ Clear plan", use_container_width=True,
                                disabled=not st.session_state.plan_days):
                st.session_state.plan_days = {}
                st.session_state.forecast = None
                st.rerun()

        # plan summary
        if st.session_state.plan_days:
            days = sorted(st.session_state.plan_days)
            summary = pd.DataFrame([
                {"Date": d,
                 "Counters": g["Counter Name"].nunique(),
                 "Items": len(g),
                 "Menu highlights": ", ".join(
                     g.loc[g["Item Name"].str.contains("Bir|bir|Mutton|Fish|Chicken|Paneer",
                                                       regex=True), "Item Name"].unique()[:3]) or "—"}
                for d in days for g in [st.session_state.plan_days[d]]
            ])
            st.markdown(f"**Plan queue — {len(days)} day(s)**")
            st.dataframe(summary, use_container_width=True, hide_index=True)

        run = st.button("🔮  Run forecast", type="primary", use_container_width=True,
                        disabled=not st.session_state.plan_days)
        if run:
            plan_all = pd.concat(st.session_state.plan_days.values(), ignore_index=True)
            overlap = set(pd.to_datetime(plan_all["Date"]).dt.date) & set(hist["Date"].dt.date)
            if overlap:
                st.error(f"Plan dates already exist in history: "
                         f"{', '.join(str(d) for d in sorted(overlap))}. "
                         "Forecast only dates after the last served day.")
            else:
                with st.spinner("Rebuilding lag features and scoring…"):
                    st.session_state.forecast = run_forecast(hist, plan_all)

    # ---- results side ----------------------------------------------------
    with right:
        st.subheader("2 · Forecast")
        fc = st.session_state.forecast
        if fc is None:
            st.markdown(
                '<div class="fc-card" style="text-align:center; padding:2.6rem 1.2rem;">'
                '<div style="font-size:2.2rem;">🍛</div>'
                '<div class="fc-value" style="font-size:1.15rem;">No forecast yet</div>'
                '<div class="fc-sub">Build or upload a menu plan on the left, '
                'then hit <b>Run forecast</b>.</div></div>',
                unsafe_allow_html=True)
        elif MVP_MODE:
            # ---- MVP output: just the numbers ------------------------------
            for dt in sorted(fc["Date"].unique()):
                g = fc[fc["Date"] == dt]
                tot = int(g["pred"].sum())
                st.markdown(
                    f'<div class="fc-hero"><div class="fc-kicker">'
                    f'{pd.Timestamp(dt):%A, %d %B %Y} · {len(g)} active counters</div>'
                    f'<div class="fc-value">{tot:,} plates</div>'
                    f'<div class="fc-sub">Predicted total lunch consumed</div></div>',
                    unsafe_allow_html=True)
                st.write("")
                out = g[["Counter Name", "pred", "lo", "hi", "order", "risk"]].copy()
                out.columns = ["Counter", "Predicted", "P10", "P90",
                               "Suggested order", "Risk"]
                for c in ["Predicted", "P10", "P90", "Suggested order"]:
                    out[c] = out[c].astype(int)
                st.dataframe(out, use_container_width=True, hide_index=True)

            full_out = fc[["Date", "Counter Name", "pred", "lo", "hi", "order", "risk"]].copy()
            full_out.columns = ["Date", "Counter", "Predicted", "P10", "P90",
                                "Suggested order", "Risk"]
            full_out["Date"] = pd.to_datetime(full_out["Date"]).dt.date
            st.download_button("⬇️ Download predictions.csv",
                               full_out.to_csv(index=False).encode(),
                               file_name="predictions_out.csv", mime="text/csv",
                               use_container_width=True)
        else:
            day_keys = sorted(fc["Date"].unique())
            for dt in day_keys:
                g = fc[fc["Date"] == dt]
                tot = int(g["pred"].sum())
                wd = g["weekday"].iloc[0]
                wd_hist = cd_hist.groupby(["Date"]).agg(total=("total", "first"),
                                                        weekday=("weekday", "first"))
                ref = wd_hist.loc[wd_hist["weekday"] == wd, "total"].tail(4).mean()
                delta = f"{tot - ref:+,.0f} vs recent {wd}s" if pd.notna(ref) else ""

                st.markdown(
                    f'<div class="fc-hero"><div class="fc-kicker">'
                    f'{pd.Timestamp(dt):%A, %d %B %Y} · {len(g)} active counters</div>'
                    f'<div class="fc-value">{tot:,} plates</div>'
                    f'<div class="fc-sub">Predicted total lunch consumed · {delta}</div></div>',
                    unsafe_allow_html=True)
                st.write("")

                cols = st.columns(2)
                for i, (_, r) in enumerate(g.iterrows()):
                    label, fg, bg = RISK_STYLE[r["risk"]]
                    share = 100 * r["pred"] / max(tot, 1)
                    with cols[i % 2]:
                        st.markdown(f"""
<div class="fc-card">
  <div class="fc-kicker"><span class="fc-dot" style="background:{COUNTER_COLORS[r['Counter Name']]}"></span>
    {r['Counter Name']}
    <span class="fc-badge" style="margin-left:auto; color:{fg}; background:{bg};">{label} RISK</span>
  </div>
  <div class="fc-value">{int(r['pred']):,}</div>
  <div class="fc-sub">predicted plates · {share:.0f}% of the day</div>
  <div style="height:.55rem;"></div>
  <div class="fc-row"><span>Likely range (P10–P90)</span><b>{int(r['lo']):,} – {int(r['hi']):,}</b></div>
  <div class="fc-row"><span>Suggested order</span><b>{int(r['order']):,} plates</b></div>
  <div class="fc-why">💡 {r['why']}</div>
</div>""", unsafe_allow_html=True)
                        st.write("")

                # range chart — prediction with P10–P90 whiskers + order tick
                gg = g.sort_values("pred")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=gg["Counter Name"], x=gg["pred"], orientation="h",
                    marker=dict(color=[COUNTER_COLORS[c] for c in gg["Counter Name"]]),
                    width=0.5, name="Predicted",
                    text=[f"{v:,.0f}" for v in gg["pred"]],
                    textposition="inside", insidetextanchor="middle",
                    textfont=dict(
                        size=12,
                        color=["#333333" if COUNTER_COLORS[c] in ("#E0A80A", "#0E93AE")
                               else "#ffffff" for c in gg["Counter Name"]]),
                    error_x=dict(type="data", array=gg["hi"] - gg["pred"],
                                 arrayminus=gg["pred"] - gg["lo"],
                                 color=INK2, thickness=1.4, width=5),
                    hovertemplate="<b>%{y}</b><br>Predicted %{x:,.0f} plates"
                                  "<br>Range %{customdata[0]:,.0f}–%{customdata[1]:,.0f}<extra></extra>",
                    customdata=np.stack([gg["lo"], gg["hi"]], axis=1),
                ))
                fig.add_trace(go.Scatter(
                    y=gg["Counter Name"], x=gg["order"], mode="markers",
                    marker=dict(symbol="line-ns", size=16,
                                line=dict(width=2.5, color=INK)),
                    name="Suggested order",
                    hovertemplate="<b>%{y}</b><br>Suggested order %{x:,.0f} plates<extra></extra>",
                ))
                base_layout(fig, height=90 + 62 * len(gg),
                            title=dict(text=f"Prediction with calibrated P10–P90 range — {pd.Timestamp(dt):%d %b}",
                                       font=dict(size=14, color=INK)),
                            bargap=0.35)
                fig.update_xaxes(title_text="plates", rangemode="tozero")
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)

            with st.expander("📋 Results table & download"):
                out = fc[["Date", "Counter Name", "pred", "lo", "hi", "order", "risk"]].copy()
                out.columns = ["Date", "Counter", "Predicted", "P10", "P90",
                               "Suggested order", "Risk"]
                out["Date"] = pd.to_datetime(out["Date"]).dt.date
                st.dataframe(out, use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download predictions.csv",
                                   out.to_csv(index=False).encode(),
                                   file_name="predictions_out.csv", mime="text/csv",
                                   use_container_width=True)

# ═════════════════════════════════════════════════════ HISTORY EXPLORER ══════
# History explorer — disabled in MVP mode
if not MVP_MODE:
    with tab_hist:
        st.subheader("What the counters have been doing")

        dmin, dmax = cd_hist["Date"].min().date(), cd_hist["Date"].max().date()
        c1, c2 = st.columns([1, 1.4])
        rng = c1.date_input("Date range", value=(max(dmin, dmax - timedelta(days=120)), dmax),
                            min_value=dmin, max_value=dmax)
        sel_counters = c2.multiselect("Counters", COUNTERS,
                                      default=[c for c in COUNTERS if c != "North Non Veg"])
        if len(rng) != 2:
            st.stop()
        view = cd_hist[(cd_hist["Date"].dt.date >= rng[0]) & (cd_hist["Date"].dt.date <= rng[1])
                       & (cd_hist["Counter Name"].isin(sel_counters))]

        day_tot = view.groupby("Date").agg(total=("total", "first"), weekday=("weekday", "first"))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg daily total", f"{day_tot['total'].mean():,.0f}" if len(day_tot) else "—",
                  help="Mean Total Lunch Consumed per working day in the selected window.")
        m2.metric("Busiest weekday",
                  day_tot.groupby("weekday")["total"].mean().idxmax() if len(day_tot) else "—")
        m3.metric("Peak day", f"{day_tot['total'].max():,.0f}" if len(day_tot) else "—")
        m4.metric("Working days", f"{day_tot.shape[0]}")
        st.write("")

        if view.empty:
            st.info("No rows in the selected window — widen the date range or counters.")
        else:
            fig = go.Figure()
            for c in [c for c in COUNTERS if c in sel_counters]:
                sub = view[view["Counter Name"] == c]
                fig.add_trace(go.Scatter(
                    x=sub["Date"], y=sub["consumed"], mode="lines", name=c,
                    line=dict(color=COUNTER_COLORS[c], width=2),
                    hovertemplate=f"<b>{c}</b><br>%{{x|%a %d %b %Y}}<br>%{{y:,.0f}} plates<extra></extra>",
                ))
            base_layout(fig, height=400,
                        title=dict(text="Daily consumption per counter",
                                   font=dict(size=15, color=INK)),
                        hovermode="x unified")
            fig.update_yaxes(title_text="plates consumed", rangemode="tozero")
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)

            cA, cB = st.columns(2)
            with cA:
                wk = (view.groupby(["weekday", "Counter Name"])["consumed"].mean()
                      .reset_index())
                order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                fig2 = go.Figure()
                for c in [c for c in COUNTERS if c in sel_counters]:
                    sub = wk[wk["Counter Name"] == c].set_index("weekday").reindex(order)
                    fig2.add_trace(go.Bar(
                        x=order, y=sub["consumed"], name=c,
                        marker_color=COUNTER_COLORS[c], width=0.18,
                        hovertemplate=f"<b>{c}</b><br>%{{x}} mean %{{y:,.0f}} plates<extra></extra>",
                    ))
                base_layout(fig2, height=360,
                            title=dict(text="Weekday profile (mean plates) — the dominant pattern",
                                       font=dict(size=14, color=INK)),
                            bargap=0.25, bargroupgap=0.12)
                fig2.update_yaxes(rangemode="tozero")
                st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CFG)

            with cB:
                shares = view.copy()
                shares["share"] = shares["consumed"] / shares["total"]
                sh = shares.groupby("Counter Name")["share"].agg(["mean", "std"]).reindex(
                    [c for c in COUNTERS if c in sel_counters])
                fig3 = go.Figure(go.Bar(
                    x=sh.index, y=sh["mean"] * 100, width=0.4,
                    marker_color=[COUNTER_COLORS[c] for c in sh.index],
                    error_y=dict(type="data", array=sh["std"] * 100, color=INK2, thickness=1.4),
                    text=[f"{v*100:.0f}%" for v in sh["mean"]], textposition="outside",
                    textfont=dict(color=INK, size=12),
                    hovertemplate="<b>%{x}</b><br>mean share %{y:.1f}%<extra></extra>",
                ))
                base_layout(fig3, height=360, showlegend=False,
                            title=dict(text="Share of the day per counter (±1σ) — SNV is the volatile one",
                                       font=dict(size=14, color=INK)))
                fig3.update_yaxes(title_text="% of daily total", rangemode="tozero")
                st.plotly_chart(fig3, use_container_width=True, config=PLOTLY_CFG)

            with st.expander("🔎 Raw counter-day table"):
                st.dataframe(view.assign(Date=view["Date"].dt.date)
                             .rename(columns={"consumed": "Consumed", "ordered": "Ordered",
                                              "total": "Day total", "weekday": "Weekday",
                                              "items": "# items"}),
                             use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════ MODEL PERFORMANCE ═══════
with tab_perf:
    st.subheader("How good is it? (scored once, on locked data)")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("June test — counter WAPE", "6.06%", "target ±6–8% met",
              help="Weighted absolute % error at Date+Counter grain, June 2026 internal test (30 counter-days).")
    k2.metric("June test — day WAPE", "3.54%",
              help="Sum of counter predictions vs actual daily total.")
    k3.metric("Current practice (proxy)", "26.4%", "-20.3 pts vs model",
              delta_color="inverse",
              help="7-day moving average — the incumbent ordering method — on the same June window.")
    k4.metric("P10–P90 coverage", "87%", "target 80%",
              help="Share of June counter-days whose actual fell inside the calibrated range.")
    st.write("")

    st.markdown("**Every split, every metric** — the June column was scored exactly once:")
    perf = pd.DataFrame({
        "Split": ["Train (Aug–Mar)", "Validation (Apr–May)", "June test (locked)"],
        "MAE": [33.3, 52.1, 37.3], "RMSE": [42.0, 65.2, 46.3],
        "MAPE %": [7.4, 10.2, 6.1], "WAPE %": [6.28, 9.10, 6.06],
        "Bias": [-0.1, 6.0, -4.8], "Over %": [47, 53, 50], "Under %": [53, 48, 50],
    })
    st.dataframe(perf, use_container_width=True, hide_index=True)
    st.caption("June is only 30 counter-days — treat 6.06% as an encouraging point "
               "estimate. Validation's 9.10% over 120 rows is the conservative "
               "expectation for live months.")

    st.markdown("**Baselines on the same June window**")
    base = pd.DataFrame({
        "Method": ["Final LightGBM model", "Same-weekday rolling-4 baseline",
                   "7-day moving average (business practice)"],
        "Counter WAPE %": [6.06, 7.70, 26.4],
    })
    if MVP_MODE:
        st.dataframe(base, use_container_width=True, hide_index=True)
        st.stop()  # MVP: numbers only — chart + diagnostic figures disabled below
    figb = go.Figure(go.Bar(
        y=base["Method"][::-1], x=base["Counter WAPE %"][::-1], orientation="h",
        width=0.45,
        marker_color=["#c9c7c0", "#c9c7c0", "#155493"],
        text=[f"{v}%" for v in base["Counter WAPE %"][::-1]], textposition="outside",
        textfont=dict(color=INK, size=13),
        hovertemplate="<b>%{y}</b><br>WAPE %{x}%<extra></extra>",
    ))
    base_layout(figb, height=240, showlegend=False,
                title=dict(text="Counter-level WAPE, June 2026 (lower is better)",
                           font=dict(size=14, color=INK)))
    figb.update_xaxes(title_text="WAPE %", range=[0, 31])
    st.plotly_chart(figb, use_container_width=True, config=PLOTLY_CFG)

    cimg1, cimg2 = st.columns(2)
    with cimg1:
        st.image(str(BUNDLE / "figs" / "final_pred_vs_actual.png"),
                 caption="June test — predicted vs actual, per counter-day",
                 use_container_width=True)
    with cimg2:
        st.image(str(BUNDLE / "figs" / "learning_curve.png"),
                 caption="Learning curve — still in the 'more data helps' regime; retrain monthly",
                 use_container_width=True)

    with st.expander("🖼️ EDA figures"):
        st.image(str(BUNDLE / "figs" / "eda_main.png"), use_container_width=True)
        st.image(str(BUNDLE / "figs" / "eda_structure.png"), use_container_width=True)

# ═══════════════════════════════════════════════════════════════ ABOUT ═══════
# About page — disabled in MVP mode
if not MVP_MODE:
    with tab_about:
        st.subheader("What this model is (and isn't)")
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown("""
##### How a forecast is made
1. **You provide tomorrow's plan** — the planned menu per active counter.
   Everything else the model needs (calendar, holidays, Panchangam, history)
   it derives itself.
2. **63 features are rebuilt leakage-safe** — every historical feature is
   shifted so it only uses information available *the evening before service*
   (when the vendor order is placed). Headcount and same-day totals are
   deliberately excluded.
3. **Four LightGBM models score the day** — a Poisson point model plus
   quantile models at P10 / P75 / P90, conformally calibrated so the
   P10–P90 band actually covers ~80% of outcomes.
4. **The suggested order is the calibrated P75** — it covers demand ~75% of
   the time, matching the business's current service posture with less
   over-provision.

##### What drives the number
The dominant signals, in validated importance order: the counter's
**same-weekday rolling-4 average**, yesterday's consumption, weekday share
history, holiday adjacency, **menu strength** (biryani/mutton/fish/paneer
pull scores), and what the *competing* counters are serving — non-veg
biryani on one counter visibly drains the veg counters.
""")
    with b:
        st.markdown("""
##### Honest limitations
- June's 6.06% comes from **30 counter-days** — expect live months nearer
  the 8–10% validation band.
- Intervals are calibrated on Apr–May data; a **regime change** (new counter,
  policy shift) breaks coverage until recalibration.
- Star-item scoring is keyword-based — genuinely **novel dishes** get
  conservative scores.
- **North Non Veg** rests on 18 historical days — low confidence if
  reactivated.
- The model assumes ordering happens the **evening before** service; a
  two-day-ahead process needs a retrained variant.

##### Operating checklist (live months)
1. Keep the history workbook current through the last served day.
2. Build tomorrow's plan here (or upload `plan.csv`) and run the forecast.
3. Order the **suggested order** quantity; note HIGH-risk counters.
4. At month end, score against actuals and **retrain monthly** — the
   learning curve says every extra month of data helps.
""")
    st.divider()
    st.caption("Bundle: `siemens_model_bundle/` — features.py (leakage-audited "
               "builder) · predict.py (CLI equivalent of this app) · evaluate.py "
               "(metric harness) · frozen boosters + conformal config in artifacts/. "
               "Full methodology in HANDOFF.md.")
