"""Menu-plan input: file upload or interactive builder, plus the plan queue.

Session state:
  plan_queue      {date_iso: DataFrame of planned item rows}
  forecast_result output of run_forecast, or None
"""
from datetime import timedelta

import pandas as pd
import streamlit as st

from ui.data import (BUNDLE_DIR, COUNTERS, DAY_TYPES, LOW_CONFIDENCE_COUNTER,
                     item_catalog)
from ui.forecasting import run_forecast

REQUIRED_PLAN_COLUMNS = {"Date", "Counter Name", "Item Name", "Category"}
BUILD_MODE, UPLOAD_MODE = "🧾 Build it here", "📤 Upload plan file"
STAR_ITEM_PATTERN = "Bir|bir|Mutton|Fish|Chicken|Paneer"


def render_plan_input(history: pd.DataFrame, include_drivers: bool) -> None:
    st.session_state.setdefault("plan_queue", {})
    st.session_state.setdefault("forecast_result", None)

    st.subheader("1 · Menu plan")
    input_mode = st.radio("How do you want to provide the plan?",
                          [BUILD_MODE, UPLOAD_MODE],
                          horizontal=True, label_visibility="collapsed")
    if input_mode == UPLOAD_MODE:
        _render_upload_section()
    else:
        _render_builder_section(history)
    _render_plan_queue()
    _render_run_button(history, include_drivers)


def _render_upload_section() -> None:
    template = pd.read_csv(BUNDLE_DIR / "plan_template.csv")
    template["Day Type"] = "Regular"
    template["Panchangam"] = "Regular"
    st.download_button("Download plan template", template.to_csv(index=False).encode(),
                       file_name="plan_template.csv", mime="text/csv")
    st.caption("One row per planned item. Required: **Date, Counter Name, "
               "Item Name, Category**. Also include **Day Type** "
               "(Regular / Previous Day of Holiday / Next Day of Holiday) and "
               "**Panchangam** (Regular or observances like Ekadashi, Poornima…) — "
               "both are prediction features; missing values default to Regular.")
    uploaded = st.file_uploader("Plan file (.csv or .xlsx) — one row per planned item",
                                type=["csv", "xlsx", "xls"])
    if uploaded is None:
        return
    try:
        if uploaded.name.lower().endswith((".xlsx", ".xls")):
            plan = pd.read_excel(uploaded)
        else:
            plan = pd.read_csv(uploaded)
    except Exception as error:
        st.error(f"Could not parse that file: {error}")
        return

    plan.columns = [str(c).strip() for c in plan.columns]
    missing_columns = REQUIRED_PLAN_COLUMNS - set(plan.columns)
    if missing_columns:
        st.error(f"Missing columns: {', '.join(sorted(missing_columns))}")
        return
    plan["Date"] = pd.to_datetime(plan["Date"])
    unknown_counters = set(plan["Counter Name"]) - set(COUNTERS)
    if unknown_counters:
        st.error(f"Unknown counters: {', '.join(sorted(unknown_counters))}")
        return

    st.session_state.plan_queue = {
        date.date().isoformat(): day_rows.reset_index(drop=True)
        for date, day_rows in plan.groupby("Date")
    }
    st.success(f"Plan loaded — {plan['Date'].nunique()} day(s), {len(plan)} item rows.")
    _report_calendar_columns(plan)


def _report_calendar_columns(plan: pd.DataFrame) -> None:
    for column in ("Day Type", "Panchangam"):
        if column in plan.columns:
            values = sorted(plan[column].dropna().astype(str).unique())
            st.caption(f"✓ {column} column detected: {', '.join(values)}")
        else:
            st.warning(f"No **{column}** column — assuming Regular for all days. "
                       "Add it if any plan day is holiday-adjacent or an "
                       "observance day; it changes the prediction.")
    if "Day Type" in plan.columns:
        unrecognised = set(plan["Day Type"].dropna().astype(str).unique()) - set(DAY_TYPES)
        if unrecognised:
            st.warning(f"Unrecognised Day Type values (treated as Regular by the "
                       f"model): {', '.join(sorted(unrecognised))}. "
                       f"Expected: {', '.join(DAY_TYPES)}.")


def _render_builder_section(history: pd.DataFrame) -> None:
    from features import PAN_FLAGS

    items_by_counter, category_by_item, latest_menus = item_catalog(history)
    last_served = history["Date"].max()

    next_working_day = (last_served + timedelta(days=1)).date()
    while next_working_day.weekday() >= 5:
        next_working_day += timedelta(days=1)

    date_col, day_type_col = st.columns(2)
    service_date = date_col.date_input("Service date", value=next_working_day,
                                       min_value=(last_served + timedelta(days=1)).date())
    day_type = day_type_col.selectbox("Day type", DAY_TYPES)
    observances = st.multiselect("Panchangam observances (if any)", PAN_FLAGS,
                                 help="Leave empty for a Regular day.")
    panchangam = "; ".join(observances) if observances else "Regular"

    is_weekend = service_date.weekday() >= 5
    if is_weekend:
        st.warning("That's a weekend — the model was trained on working days "
                   "(Mon–Fri) only and can't score it.")

    active_counters = st.multiselect(
        "Active counters", COUNTERS,
        default=[c for c in COUNTERS if c != LOW_CONFIDENCE_COUNTER],
        help="Closed counters are simply left out.")
    if LOW_CONFIDENCE_COUNTER in active_counters:
        st.info(f"{LOW_CONFIDENCE_COUNTER} ran only 18 historical days — treat "
                "its forecast as low-confidence.", icon="ℹ️")

    planned_items = []
    for counter in active_counters:
        with st.expander(f"{counter} — menu", expanded=True):
            selected = st.multiselect(
                f"Items for {counter}", options=items_by_counter[counter],
                default=[i for i in latest_menus[counter]
                         if i in items_by_counter[counter]],
                key=f"menu_{counter}",
                help="Pre-filled with this counter's most recent menu — edit freely.")
            planned_items += [
                {"Date": pd.Timestamp(service_date), "Counter Name": counter,
                 "Item Name": item,
                 "Category": category_by_item.get(item, "Veg Gravy"),
                 "Day Type": day_type, "Panchangam": panchangam}
                for item in selected
            ]

    with st.expander("➕ Add items not in the list"):
        custom_items = st.data_editor(
            pd.DataFrame(columns=["Counter Name", "Item Name", "Category"]),
            num_rows="dynamic", use_container_width=True, key="custom_items",
            column_config={
                "Counter Name": st.column_config.SelectboxColumn(options=COUNTERS, required=True),
                "Item Name": st.column_config.TextColumn(required=True),
                "Category": st.column_config.SelectboxColumn(
                    options=sorted(history["Category"].unique()), required=True),
            })
        for _, item_row in custom_items.dropna(
                subset=["Counter Name", "Item Name", "Category"]).iterrows():
            if item_row["Counter Name"] in active_counters:
                planned_items.append(
                    {"Date": pd.Timestamp(service_date),
                     "Counter Name": item_row["Counter Name"],
                     "Item Name": item_row["Item Name"],
                     "Category": item_row["Category"],
                     "Day Type": day_type, "Panchangam": panchangam})

    add_col, clear_col = st.columns(2)
    if add_col.button("➕ Add this day to the plan", use_container_width=True,
                      disabled=not planned_items or is_weekend):
        st.session_state.plan_queue[service_date.isoformat()] = pd.DataFrame(planned_items)
        st.toast(f"Added {service_date:%a %d %b} — {len(planned_items)} items.", icon="✅")
    if clear_col.button("🗑️ Clear plan", use_container_width=True,
                        disabled=not st.session_state.plan_queue):
        st.session_state.plan_queue = {}
        st.session_state.forecast_result = None
        st.rerun()


def _render_plan_queue() -> None:
    plan_queue = st.session_state.plan_queue
    if not plan_queue:
        return
    summary = pd.DataFrame([
        {"Date": date_iso,
         "Counters": day_rows["Counter Name"].nunique(),
         "Items": len(day_rows),
         "Menu highlights": ", ".join(
             day_rows.loc[day_rows["Item Name"].str.contains(STAR_ITEM_PATTERN, regex=True),
                          "Item Name"].unique()[:3]) or "—"}
        for date_iso, day_rows in sorted(plan_queue.items())
    ])
    st.markdown(f"**Plan queue — {len(plan_queue)} day(s)**")
    st.dataframe(summary, use_container_width=True, hide_index=True)


def _render_run_button(history: pd.DataFrame, include_drivers: bool) -> None:
    if not st.button("🔮  Run forecast", type="primary", use_container_width=True,
                     disabled=not st.session_state.plan_queue):
        return
    plan = pd.concat(st.session_state.plan_queue.values(), ignore_index=True)
    dates_in_history = (set(pd.to_datetime(plan["Date"]).dt.date)
                        & set(history["Date"].dt.date))
    if dates_in_history:
        st.error(f"Plan dates already exist in history: "
                 f"{', '.join(str(d) for d in sorted(dates_in_history))}. "
                 "Forecast only dates after the last served day.")
        return
    with st.spinner("Rebuilding lag features and scoring…"):
        st.session_state.forecast_result = run_forecast(history, plan, include_drivers)
