"""Forecast results panel (MVP: numbers only)."""
import pandas as pd
import streamlit as st

from ui.branding import hero_card

DISPLAY_COLUMNS = {"Counter Name": "Counter", "predicted": "Predicted",
                   "range_low": "P10", "range_high": "P90",
                   "suggested_order": "Suggested order", "risk": "Risk"}
INTEGER_COLUMNS = ["Predicted", "P10", "P90", "Suggested order"]


def _display_table(forecast: pd.DataFrame, with_date: bool = False) -> pd.DataFrame:
    columns = (["Date"] if with_date else []) + list(DISPLAY_COLUMNS)
    table = forecast[columns].rename(columns=DISPLAY_COLUMNS)
    for column in INTEGER_COLUMNS:
        table[column] = table[column].astype(int)
    if with_date:
        table["Date"] = pd.to_datetime(table["Date"]).dt.date
    return table


def render_empty_state() -> None:
    st.markdown(
        '<div class="fc-card" style="text-align:center; padding:2.6rem 1.2rem;">'
        '<div style="font-size:2.2rem;">🍛</div>'
        '<div class="fc-value" style="font-size:1.15rem;">No forecast yet</div>'
        '<div class="fc-sub">Build or upload a menu plan on the left, '
        'then hit <b>Run forecast</b>.</div></div>',
        unsafe_allow_html=True)


def render_forecast_warnings(forecast: pd.DataFrame) -> None:
    if (forecast["regime"] == "stale").any():
        st.warning("Yesterday's actuals are not in the history yet — using the "
                   "**lag-2 fallback model** (slightly wider uncertainty). "
                   "Update the history workbook for the sharpest forecast.",
                   icon="⏳")
    if forecast["Date"].nunique() > 1:
        st.warning("Multi-day plan: predictions for day 2 onwards can't see the "
                   "earlier days' actuals and are less reliable. For best "
                   "accuracy forecast one day at a time with updated history.",
                   icon="📅")


def render_forecast_numbers(forecast: pd.DataFrame) -> None:
    """MVP output: a daily hero total, a plain numbers table, CSV download."""
    render_forecast_warnings(forecast)
    for date in sorted(forecast["Date"].unique()):
        day_rows = forecast[forecast["Date"] == date]
        total_plates = int(day_rows["predicted"].sum())
        st.markdown(hero_card(
            f"{pd.Timestamp(date):%A, %d %B %Y} · {len(day_rows)} active counters",
            f"{total_plates:,} plates",
            "Predicted total lunch consumed"), unsafe_allow_html=True)
        st.write("")
        st.dataframe(_display_table(day_rows), use_container_width=True, hide_index=True)

    st.download_button("⬇️ Download predictions.csv",
                       _display_table(forecast, with_date=True).to_csv(index=False).encode(),
                       file_name="predictions_out.csv", mime="text/csv",
                       use_container_width=True)
