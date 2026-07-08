"""Thin Streamlit wrapper over the bundle's scoring pipeline (shadow.score_plan).

Official numbers: frozen incumbent when yesterday's actuals are in the
history, dedicated lag-2 fallback when they aren't. Every run silently logs
the shadow roster (challenger4, tuned ExtraTrees, LGB+KNN hybrid) to
shadow_log.csv for the month-end referee.
"""
import pandas as pd

from ui.data import SHADOW_LOG_CSV

FORECAST_COLUMNS = ["predicted", "range_low", "range_high", "suggested_order",
                    "risk", "regime"]


def run_forecast(history: pd.DataFrame, plan: pd.DataFrame, holiday_dates: set,
                 order_policy: str = "standard",
                 include_drivers: bool = False) -> pd.DataFrame:
    """Score a plan and return one row per Date + Counter with
    FORECAST_COLUMNS (+ 'drivers' when include_drivers)."""
    import shadow

    target, shadow_preds = shadow.score_plan(history, plan, holiday_dates,
                                             order_policy=order_policy)
    if include_drivers:
        target["drivers"] = target.apply(explain_drivers, axis=1)
    if shadow_preds:
        try:
            shadow.log_shadow_run(target, shadow_preds, SHADOW_LOG_CSV)
        except Exception:
            pass  # shadow logging must never break the official forecast
    return target.sort_values(["Date", "predicted"], ascending=[True, False])


def explain_drivers(row: pd.Series) -> str:
    """Plain-language prediction drivers, mirroring the bundle's predict.py."""
    drivers = []
    if row["has_nv_biryani"]:
        drivers.append("non-veg biryani on the menu (historically the strongest pull item)")
    elif row["star_score"] >= 4:
        drivers.append("a very-high-pull star item on the menu")
    if row["oth_has_nv_biryani"] and not row["has_nv_biryani"]:
        drivers.append("a competing counter serves non-veg biryani (drains this counter)")
    if row["star_minus_oth"] > 1:
        drivers.append("this counter has the strongest menu among active counters today")
    if row["dt_prev_holiday"] or row["dt_next_holiday"]:
        drivers.append("holiday-adjacent day (attendance typically drops)")
    if row["weekday"] == "Friday":
        drivers.append("Friday (lowest-attendance weekday)")
    if row["weekday"] in ("Tuesday", "Wednesday"):
        drivers.append(f"{row['weekday']} (peak-attendance weekday)")
    drivers.append(f"recent same-weekday average for this counter is {row['wd_roll4']:.0f}")
    return "; ".join(drivers).capitalize() + "."
