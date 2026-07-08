"""Lunch counter demand forecast — Streamlit entry point.

Menu plan in → per-counter demand, calibrated P10–P90 range, suggested order
quantity and risk level, one day ahead of service. Backed by the frozen
LightGBM bundle in ./siemens_model_bundle (see HANDOFF.md) with a lag-2
fallback when yesterday's actuals are missing, and silent shadow scoring of
the challenger models (model_research/FINDINGS.md).

Run:  streamlit run app.py
"""
import streamlit as st

from ui import branding
from ui.data import counter_day_history, load_history
from ui.plan_input import render_plan_input
from ui.results import render_empty_state, render_forecast_numbers

# MVP_MODE = True shows the lean product: plan in, numbers out. Flip to False
# to restore the full tool (result cards with drivers, charts, history
# explorer, model performance and about pages).
MVP_MODE = True

st.set_page_config(
    page_title="Lunch Counter Forecast",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)
branding.inject_brand_styles()


def render_sidebar():
    """Brand block + history workbook loader; returns the loaded history."""
    with st.sidebar:
        branding.render_sidebar_wordmark()
        st.caption("LightGBM · Poisson objective · frozen bundle, "
                   "conformalized P10–P90 intervals.")
        st.divider()

        st.markdown("**History data**")
        uploaded = st.file_uploader(
            "Replace bundled history (.xlsx)", type=["xlsx"],
            help="Sheet 'Lunch Master', same columns as the master workbook. "
                 "Keep it current through the last served day — every lag "
                 "feature is rebuilt from it.")
        try:
            history = load_history(uploaded.getvalue() if uploaded else None)
        except Exception as error:
            st.error(f"Could not read that workbook: {error}")
            history = load_history(None)

        st.markdown(
            f'<div class="sq-note">History loaded ✓<br>'
            f'<b>{history["Date"].min():%d %b %Y} → {history["Date"].max():%d %b %Y}</b><br>'
            f'{history.shape[0]:,} item rows · {history["Date"].nunique()} working days</div>',
            unsafe_allow_html=True)

        if not MVP_MODE:
            st.divider()
            st.markdown("**Locked June test**  \n"
                        "Counter WAPE **6.06%** · Day WAPE **3.54%**  \n"
                        "vs moving-average practice **26.4%**")
            st.caption("Predictions use only information available at "
                       "vendor-ordering time (the evening before service).")
    return history


def render_forecast_page(history) -> None:
    plan_column, results_column = st.columns([1.05, 1], gap="large")
    with plan_column:
        render_plan_input(history, include_drivers=not MVP_MODE)
    with results_column:
        st.subheader("2 · Forecast")
        forecast = st.session_state.get("forecast_result")
        if forecast is None:
            render_empty_state()
        elif MVP_MODE:
            render_forecast_numbers(forecast)
        else:
            from ui import full_tool
            full_tool.render_rich_results(forecast, counter_day_history(history))


def main() -> None:
    history = render_sidebar()

    branding.render_running_head()
    st.title("Lunch counter demand forecast")

    if MVP_MODE:
        st.caption("MVP — menu plan in, numbers out: per-counter demand, calibrated "
                   "range, suggested order and risk, one day ahead of service.")
        render_forecast_page(history)
        return

    from ui import full_tool
    st.caption("Plan tomorrow's menu → get per-counter demand, a calibrated range, "
               "a suggested order quantity and the risk level — one day ahead of service.")
    forecast_tab, history_tab, performance_tab, about_tab = st.tabs(
        ["🔮  Forecast", "📊  History explorer", "📈  Model performance", "ℹ️  About the model"])
    with forecast_tab:
        render_forecast_page(history)
    with history_tab:
        full_tool.render_history_explorer(counter_day_history(history))
    with performance_tab:
        full_tool.render_model_performance()
    with about_tab:
        full_tool.render_about()


main()
