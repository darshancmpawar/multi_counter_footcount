"""Full-tool pages, shown only when MVP_MODE is off: rich forecast results,
history explorer, model performance and the about page."""
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.branding import (COUNTER_COLORS, INK, INK_SECONDARY, LIGHT_MARKS,
                         PLOTLY_CONFIG, RISK_BADGES, chart_layout, hero_card)
from ui.data import BUNDLE_DIR, COUNTERS, LOW_CONFIDENCE_COUNTER
from ui.results import _display_table, render_forecast_warnings

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


# ------------------------------------------------------------ forecast tab ---
def render_rich_results(forecast: pd.DataFrame, counter_day_hist: pd.DataFrame) -> None:
    """Per-counter cards with drivers, plus a range chart per day."""
    render_forecast_warnings(forecast)
    for date in sorted(forecast["Date"].unique()):
        day_rows = forecast[forecast["Date"] == date]
        _render_day_hero(date, day_rows, counter_day_hist)
        _render_counter_cards(day_rows)
        _render_range_chart(date, day_rows)

    with st.expander("📋 Results table & download"):
        table = _display_table(forecast, with_date=True)
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download predictions.csv",
                           table.to_csv(index=False).encode(),
                           file_name="predictions_out.csv", mime="text/csv",
                           use_container_width=True)


def _render_day_hero(date, day_rows: pd.DataFrame,
                     counter_day_hist: pd.DataFrame) -> None:
    total_plates = int(day_rows["predicted"].sum())
    weekday = day_rows["weekday"].iloc[0]
    daily_totals = counter_day_hist.groupby("Date").agg(
        total=("total", "first"), weekday=("weekday", "first"))
    recent_same_weekday = daily_totals.loc[daily_totals["weekday"] == weekday,
                                           "total"].tail(4).mean()
    delta = (f"{total_plates - recent_same_weekday:+,.0f} vs recent {weekday}s"
             if pd.notna(recent_same_weekday) else "")
    st.markdown(hero_card(
        f"{pd.Timestamp(date):%A, %d %B %Y} · {len(day_rows)} active counters",
        f"{total_plates:,} plates",
        f"Predicted total lunch consumed · {delta}"), unsafe_allow_html=True)
    st.write("")


def _render_counter_cards(day_rows: pd.DataFrame) -> None:
    total_plates = max(int(day_rows["predicted"].sum()), 1)
    columns = st.columns(2)
    for position, (_, row) in enumerate(day_rows.iterrows()):
        badge_text, badge_fg, badge_bg = RISK_BADGES[row["risk"]]
        day_share = 100 * row["predicted"] / total_plates
        with columns[position % 2]:
            st.markdown(f"""
<div class="fc-card">
  <div class="fc-kicker"><span class="fc-dot" style="background:{COUNTER_COLORS[row['Counter Name']]}"></span>
    {row['Counter Name']}
    <span class="fc-badge" style="margin-left:auto; color:{badge_fg}; background:{badge_bg};">{badge_text} RISK</span>
  </div>
  <div class="fc-value">{int(row['predicted']):,}</div>
  <div class="fc-sub">predicted plates · {day_share:.0f}% of the day</div>
  <div style="height:.55rem;"></div>
  <div class="fc-row"><span>Likely range (P10–P90)</span><b>{int(row['range_low']):,} – {int(row['range_high']):,}</b></div>
  <div class="fc-row"><span>Suggested order</span><b>{int(row['suggested_order']):,} plates</b></div>
  <div class="fc-why">💡 {row['drivers']}</div>
</div>""", unsafe_allow_html=True)
            st.write("")


def _render_range_chart(date, day_rows: pd.DataFrame) -> None:
    ordered = day_rows.sort_values("predicted")
    colors = [COUNTER_COLORS[c] for c in ordered["Counter Name"]]
    figure = go.Figure()
    figure.add_trace(go.Bar(
        y=ordered["Counter Name"], x=ordered["predicted"], orientation="h",
        marker=dict(color=colors), width=0.5, name="Predicted",
        text=[f"{v:,.0f}" for v in ordered["predicted"]],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(size=12,
                      color=[INK if c in LIGHT_MARKS else "#ffffff" for c in colors]),
        error_x=dict(type="data",
                     array=ordered["range_high"] - ordered["predicted"],
                     arrayminus=ordered["predicted"] - ordered["range_low"],
                     color=INK_SECONDARY, thickness=1.4, width=5),
        hovertemplate="<b>%{y}</b><br>Predicted %{x:,.0f} plates"
                      "<br>Range %{customdata[0]:,.0f}–%{customdata[1]:,.0f}<extra></extra>",
        customdata=np.stack([ordered["range_low"], ordered["range_high"]], axis=1),
    ))
    figure.add_trace(go.Scatter(
        y=ordered["Counter Name"], x=ordered["suggested_order"], mode="markers",
        marker=dict(symbol="line-ns", size=16, line=dict(width=2.5, color=INK)),
        name="Suggested order",
        hovertemplate="<b>%{y}</b><br>Suggested order %{x:,.0f} plates<extra></extra>",
    ))
    chart_layout(figure, height=90 + 62 * len(ordered),
                 title=dict(text=f"Prediction with calibrated P10–P90 range — {pd.Timestamp(date):%d %b}",
                            font=dict(size=14, color=INK)),
                 bargap=0.35)
    figure.update_xaxes(title_text="plates", rangemode="tozero")
    st.plotly_chart(figure, use_container_width=True, config=PLOTLY_CONFIG)


# ---------------------------------------------------------- history explorer -
def render_history_explorer(counter_day_hist: pd.DataFrame) -> None:
    st.subheader("What the counters have been doing")

    date_min = counter_day_hist["Date"].min().date()
    date_max = counter_day_hist["Date"].max().date()
    range_col, counters_col = st.columns([1, 1.4])
    date_range = range_col.date_input(
        "Date range", value=(max(date_min, date_max - timedelta(days=120)), date_max),
        min_value=date_min, max_value=date_max)
    selected_counters = counters_col.multiselect(
        "Counters", COUNTERS,
        default=[c for c in COUNTERS if c != LOW_CONFIDENCE_COUNTER])
    if len(date_range) != 2:
        st.stop()
    view = counter_day_hist[
        (counter_day_hist["Date"].dt.date >= date_range[0])
        & (counter_day_hist["Date"].dt.date <= date_range[1])
        & (counter_day_hist["Counter Name"].isin(selected_counters))]

    daily_totals = view.groupby("Date").agg(total=("total", "first"),
                                            weekday=("weekday", "first"))
    tile1, tile2, tile3, tile4 = st.columns(4)
    tile1.metric("Avg daily total",
                 f"{daily_totals['total'].mean():,.0f}" if len(daily_totals) else "—",
                 help="Mean Total Lunch Consumed per working day in the selected window.")
    tile2.metric("Busiest weekday",
                 daily_totals.groupby("weekday")["total"].mean().idxmax()
                 if len(daily_totals) else "—")
    tile3.metric("Peak day", f"{daily_totals['total'].max():,.0f}" if len(daily_totals) else "—")
    tile4.metric("Working days", f"{daily_totals.shape[0]}")
    st.write("")

    if view.empty:
        st.info("No rows in the selected window — widen the date range or counters.")
        return

    _render_daily_trend(view, selected_counters)
    weekday_col, share_col = st.columns(2)
    with weekday_col:
        _render_weekday_profile(view, selected_counters)
    with share_col:
        _render_counter_shares(view, selected_counters)

    with st.expander("🔎 Raw counter-day table"):
        st.dataframe(view.assign(Date=view["Date"].dt.date)
                     .rename(columns={"consumed": "Consumed", "ordered": "Ordered",
                                      "total": "Day total", "weekday": "Weekday",
                                      "items": "# items"}),
                     use_container_width=True, hide_index=True)


def _render_daily_trend(view: pd.DataFrame, selected_counters: list) -> None:
    figure = go.Figure()
    for counter in [c for c in COUNTERS if c in selected_counters]:
        counter_rows = view[view["Counter Name"] == counter]
        figure.add_trace(go.Scatter(
            x=counter_rows["Date"], y=counter_rows["consumed"], mode="lines",
            name=counter, line=dict(color=COUNTER_COLORS[counter], width=2),
            hovertemplate=f"<b>{counter}</b><br>%{{x|%a %d %b %Y}}"
                          f"<br>%{{y:,.0f}} plates<extra></extra>"))
    chart_layout(figure, height=400,
                 title=dict(text="Daily consumption per counter",
                            font=dict(size=15, color=INK)),
                 hovermode="x unified")
    figure.update_yaxes(title_text="plates consumed", rangemode="tozero")
    st.plotly_chart(figure, use_container_width=True, config=PLOTLY_CONFIG)


def _render_weekday_profile(view: pd.DataFrame, selected_counters: list) -> None:
    weekday_means = (view.groupby(["weekday", "Counter Name"])["consumed"]
                     .mean().reset_index())
    figure = go.Figure()
    for counter in [c for c in COUNTERS if c in selected_counters]:
        means = (weekday_means[weekday_means["Counter Name"] == counter]
                 .set_index("weekday").reindex(WEEKDAY_ORDER))
        figure.add_trace(go.Bar(
            x=WEEKDAY_ORDER, y=means["consumed"], name=counter,
            marker_color=COUNTER_COLORS[counter], width=0.18,
            hovertemplate=f"<b>{counter}</b><br>%{{x}} mean %{{y:,.0f}} plates<extra></extra>"))
    chart_layout(figure, height=360,
                 title=dict(text="Weekday profile (mean plates) — the dominant pattern",
                            font=dict(size=14, color=INK)),
                 bargap=0.25, bargroupgap=0.12)
    figure.update_yaxes(rangemode="tozero")
    st.plotly_chart(figure, use_container_width=True, config=PLOTLY_CONFIG)


def _render_counter_shares(view: pd.DataFrame, selected_counters: list) -> None:
    shares = view.assign(share=view["consumed"] / view["total"])
    share_stats = (shares.groupby("Counter Name")["share"].agg(["mean", "std"])
                   .reindex([c for c in COUNTERS if c in selected_counters]))
    figure = go.Figure(go.Bar(
        x=share_stats.index, y=share_stats["mean"] * 100, width=0.4,
        marker_color=[COUNTER_COLORS[c] for c in share_stats.index],
        error_y=dict(type="data", array=share_stats["std"] * 100,
                     color=INK_SECONDARY, thickness=1.4),
        text=[f"{v*100:.0f}%" for v in share_stats["mean"]], textposition="outside",
        textfont=dict(color=INK, size=12),
        hovertemplate="<b>%{x}</b><br>mean share %{y:.1f}%<extra></extra>"))
    chart_layout(figure, height=360, showlegend=False,
                 title=dict(text="Share of the day per counter (±1σ) — SNV is the volatile one",
                            font=dict(size=14, color=INK)))
    figure.update_yaxes(title_text="% of daily total", rangemode="tozero")
    st.plotly_chart(figure, use_container_width=True, config=PLOTLY_CONFIG)


# --------------------------------------------------------- model performance -
def render_model_performance() -> None:
    st.subheader("How good is it? (scored once, on locked data)")

    tile1, tile2, tile3, tile4 = st.columns(4)
    tile1.metric("June test — counter WAPE", "6.06%", "target ±6–8% met",
                 help="Weighted absolute % error at Date+Counter grain, "
                      "June 2026 internal test (30 counter-days).")
    tile2.metric("June test — day WAPE", "3.54%",
                 help="Sum of counter predictions vs actual daily total.")
    tile3.metric("Current practice (proxy)", "26.4%", "-20.3 pts vs model",
                 delta_color="inverse",
                 help="7-day moving average — the incumbent ordering method — "
                      "on the same June window.")
    tile4.metric("P10–P90 coverage", "87%", "target 80%",
                 help="Share of June counter-days whose actual fell inside "
                      "the calibrated range.")
    st.write("")

    st.markdown("**Every split, every metric** — the June column was scored exactly once:")
    split_metrics = pd.DataFrame({
        "Split": ["Train (Aug–Mar)", "Validation (Apr–May)", "June test (locked)"],
        "MAE": [33.3, 52.1, 37.3], "RMSE": [42.0, 65.2, 46.3],
        "MAPE %": [7.4, 10.2, 6.1], "WAPE %": [6.28, 9.10, 6.06],
        "Bias": [-0.1, 6.0, -4.8], "Over %": [47, 53, 50], "Under %": [53, 48, 50],
    })
    st.dataframe(split_metrics, use_container_width=True, hide_index=True)
    st.caption("June is only 30 counter-days — treat 6.06% as an encouraging point "
               "estimate. Validation's 9.10% over 120 rows is the conservative "
               "expectation for live months.")

    st.markdown("**Baselines on the same June window**")
    baselines = pd.DataFrame({
        "Method": ["Final LightGBM model", "Same-weekday rolling-4 baseline",
                   "7-day moving average (business practice)"],
        "Counter WAPE %": [6.06, 7.70, 26.4],
    })
    figure = go.Figure(go.Bar(
        y=baselines["Method"][::-1], x=baselines["Counter WAPE %"][::-1],
        orientation="h", width=0.45,
        marker_color=["#c9c7c0", "#c9c7c0", "#155493"],
        text=[f"{v}%" for v in baselines["Counter WAPE %"][::-1]],
        textposition="outside", textfont=dict(color=INK, size=13),
        hovertemplate="<b>%{y}</b><br>WAPE %{x}%<extra></extra>"))
    chart_layout(figure, height=240, showlegend=False,
                 title=dict(text="Counter-level WAPE, June 2026 (lower is better)",
                            font=dict(size=14, color=INK)))
    figure.update_xaxes(title_text="WAPE %", range=[0, 31])
    st.plotly_chart(figure, use_container_width=True, config=PLOTLY_CONFIG)

    left_img, right_img = st.columns(2)
    with left_img:
        st.image(str(BUNDLE_DIR / "figs" / "final_pred_vs_actual.png"),
                 caption="June test — predicted vs actual, per counter-day",
                 use_container_width=True)
    with right_img:
        st.image(str(BUNDLE_DIR / "figs" / "learning_curve.png"),
                 caption="Learning curve — still in the 'more data helps' regime; retrain monthly",
                 use_container_width=True)

    with st.expander("🖼️ EDA figures"):
        st.image(str(BUNDLE_DIR / "figs" / "eda_main.png"), use_container_width=True)
        st.image(str(BUNDLE_DIR / "figs" / "eda_structure.png"), use_container_width=True)


# ---------------------------------------------------------------- about page -
def render_about() -> None:
    st.subheader("What this model is (and isn't)")
    how_col, limits_col = st.columns(2, gap="large")
    with how_col:
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
    with limits_col:
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
- Ordering must happen the **evening before** service for the main model;
  when yesterday's actuals are missing the app switches to the lag-2
  fallback automatically.

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
