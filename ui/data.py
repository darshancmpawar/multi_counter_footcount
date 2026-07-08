"""Paths, domain constants and cached data/model loaders."""
import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = REPO_ROOT / "siemens_model_bundle"
DEFAULT_HISTORY_XLSX = REPO_ROOT / "Lunch_Master_Data_FINAL(cleaned).xlsx"
SHADOW_LOG_CSV = REPO_ROOT / "shadow_log.csv"
HISTORY_SHEET = "Lunch Master"

if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

COUNTERS = ["North Non Veg", "North Veg", "South Non Veg", "South Veg"]
DAY_TYPES = ["Regular", "Previous Day of Holiday", "Next Day of Holiday"]
LOW_CONFIDENCE_COUNTER = "North Non Veg"   # only 18 historical service days


@st.cache_data(show_spinner="Loading history workbook…")
def load_history(uploaded_bytes: bytes | None) -> pd.DataFrame:
    source = io.BytesIO(uploaded_bytes) if uploaded_bytes is not None else DEFAULT_HISTORY_XLSX
    history = pd.read_excel(source, sheet_name=HISTORY_SHEET)
    history["Date"] = pd.to_datetime(history["Date"])
    return history


@st.cache_data(show_spinner=False)
def load_holiday_dates(uploaded_bytes: bytes | None) -> set:
    """Holiday dates from the workbook's 'Holiday List' sheet — drives the
    automatic Day Type derivation. Empty set if the sheet is missing."""
    import auto_calendar
    source = io.BytesIO(uploaded_bytes) if uploaded_bytes is not None else DEFAULT_HISTORY_XLSX
    return auto_calendar.load_holiday_dates(source)


@st.cache_resource(show_spinner="Loading frozen model bundle…")
def load_incumbent_models():
    """The frozen, June-anchored production boosters + conformal config."""
    import pickle
    import lightgbm as lgb

    artifacts = BUNDLE_DIR / "artifacts"
    with open(artifacts / "final_config.pkl", "rb") as f:
        conformal_config = pickle.load(f)
    boosters = {name: lgb.Booster(model_file=str(artifacts / f"model_{name}.txt"))
                for name in ("point", "q10", "q75", "q90")}
    return boosters, conformal_config


@st.cache_resource(show_spinner=False)
def load_shadow_models():
    """Challenger / blend / lag-2 fallback set (None if never built)."""
    import shadow
    return shadow.load_shadow()


@st.cache_data(show_spinner=False)
def counter_day_history(history: pd.DataFrame) -> pd.DataFrame:
    """Item-level history aggregated to one row per Date + Counter."""
    aggregated = history.groupby(["Date", "Counter Name"], as_index=False).agg(
        consumed=("Counter Consumed", "first"),
        ordered=("Counter Ordered", "first"),
        total=("Total Lunch Consumed", "first"),
        weekday=("Weekday", "first"),
        items=("Item Name", "nunique"),
    )
    return aggregated.sort_values(["Date", "Counter Name"])


@st.cache_data(show_spinner=False)
def item_catalog(history: pd.DataFrame):
    """Menu-builder inputs: frequency-ordered items per counter, an
    item → most-common-category map, and each counter's latest served menu."""
    category_by_item = (history.groupby("Item Name")["Category"]
                        .agg(lambda s: s.mode().iat[0]).to_dict())
    items_by_counter, latest_menu_by_counter = {}, {}
    for counter in COUNTERS:
        counter_rows = history[history["Counter Name"] == counter]
        items_by_counter[counter] = counter_rows["Item Name"].value_counts().index.tolist()
        if len(counter_rows):
            latest_date = counter_rows["Date"].max()
            latest_menu_by_counter[counter] = sorted(
                counter_rows.loc[counter_rows["Date"] == latest_date, "Item Name"].unique())
        else:
            latest_menu_by_counter[counter] = []
    return items_by_counter, category_by_item, latest_menu_by_counter
