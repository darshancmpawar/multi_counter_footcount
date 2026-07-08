"""Plan scoring: official forecast, lag-regime routing and shadow logging.

Official numbers come from the frozen incumbent boosters when yesterday's
actuals are in the history ("fresh" regime). When the history is stale the
dedicated lag-2 fallback set takes over — a lag-1 model silently fed stale
lags loses ~2.1pt WAPE, the fallback only ~0.5pt (model_research/FINDINGS.md).
Challenger and blend predictions are logged silently for month-end review.
"""
import numpy as np
import pandas as pd

from ui.data import SHADOW_LOG_CSV, load_incumbent_models, load_shadow_models

# columns of the dataframe returned by run_forecast
FORECAST_COLUMNS = ["predicted", "range_low", "range_high", "suggested_order",
                    "risk", "regime"]

_PLACEHOLDER_NAN = ["Receiving Qty", "Bainmarie Wastage"]
_PLACEHOLDER_ZERO = ["Headcount", "Total Lunch Consumed", "Counter Ordered",
                     "Counter Consumed"]


def normalize_plan(plan: pd.DataFrame) -> pd.DataFrame:
    """Shape a plan (one row per planned item) like a history row so the
    bundle's feature builder can consume it. Target-side columns get
    placeholders that never enter the feature set."""
    plan = plan.copy()
    plan["Date"] = pd.to_datetime(plan["Date"])
    for column, default in [("Day Type", "Regular"), ("Panchangam", "Regular")]:
        if column not in plan:
            plan[column] = default
        plan[column] = plan[column].fillna(default)
    plan["Month"] = plan["Date"].dt.month_name()
    plan["Weekday"] = plan["Date"].dt.day_name()
    for column in _PLACEHOLDER_NAN:
        plan[column] = np.nan
    for column in _PLACEHOLDER_ZERO:
        plan[column] = 0
    return plan


def detect_lag_regime(history: pd.DataFrame, plan: pd.DataFrame) -> str:
    """'fresh' when the last served day is the working day right before the
    first plan day (yesterday's actuals available), else 'stale'."""
    business_day_gap = int(np.busday_count(history["Date"].max().date(),
                                           plan["Date"].min().date()))
    return "fresh" if business_day_gap <= 1 else "stale"


def _predict_official(target: pd.DataFrame, features, regime: str,
                      shadow_models) -> pd.DataFrame:
    if regime == "fresh":
        boosters, conformal = load_incumbent_models()
        point = boosters["point"].predict(features)
        q10 = boosters["q10"].predict(features)
        q75 = boosters["q75"].predict(features)
        q90 = boosters["q90"].predict(features)
        cqr_width, order_correction = conformal["cqr_Q80"], conformal["order_corr"]
    else:
        import shadow
        meta = shadow_models["meta"]
        point = shadow.seed_avg(shadow_models["fb_point"], features)
        q10 = shadow_models["fb_q10"].predict(features)
        q75 = shadow_models["fb_q75"].predict(features)
        q90 = shadow_models["fb_q90"].predict(features)
        cqr_width, order_correction = meta["fb_cqr_Q80"], meta["fb_order_corr"]

    target["predicted"] = np.clip(point, 0, None).round(0)
    target["range_low"] = np.clip(np.clip(q10, 0, None) - cqr_width, 0, None).round(0)
    target["range_high"] = (np.clip(q90, 0, None) + cqr_width).round(0)
    target["suggested_order"] = (np.clip(q75, 0, None) + order_correction).round(-1)
    return target


def _log_shadow_predictions(target: pd.DataFrame, extra_features,
                            regime: str, shadow_models) -> None:
    """Score challenger + blend and append to the shadow log. Must never
    break the official forecast, hence the broad except."""
    import shadow
    try:
        challenger_set = "challenger" if regime == "fresh" else "chal_fb"
        challenger = shadow.seed_avg(shadow_models[challenger_set], extra_features)
        blend = np.full(len(target), np.nan)
        if regime == "fresh" and shadow_models["blend_et"] is not None:
            blend_member = shadow_models["blend_et"]
            sk_features, _, _ = shadow.design_sk(target, blend_member["medians"],
                                                 blend_member["columns"])
            tree_prediction = np.clip(blend_member["model"].predict(sk_features), 0, None)
            blend = shadow.BLEND_W * challenger + (1 - shadow.BLEND_W) * tree_prediction
        shadow.log_shadow(
            [{"Date": str(pd.Timestamp(date).date()), "Counter Name": counter,
              "regime": regime, "pred_official": float(official),
              "pred_challenger": float(chal),
              "pred_blend": (None if np.isnan(bl) else float(bl)),
              "logged_at": pd.Timestamp.now().isoformat(timespec="seconds")}
             for date, counter, official, chal, bl
             in zip(target["Date"], target["Counter Name"],
                    target["predicted"], challenger, blend)],
            SHADOW_LOG_CSV)
    except Exception:
        pass


def run_forecast(history: pd.DataFrame, plan: pd.DataFrame,
                 include_drivers: bool = False) -> pd.DataFrame:
    """Score a plan and return one row per Date + Counter with
    FORECAST_COLUMNS (+ 'drivers' when include_drivers)."""
    import shadow

    plan = normalize_plan(plan)
    regime = detect_lag_regime(history, plan)
    lag_depth = 1 if regime == "fresh" else 2

    combined = pd.concat([history, plan[history.columns]], ignore_index=True)
    counter_days = shadow.build_cd_k(combined, k=lag_depth)
    target = counter_days[counter_days["Date"].isin(plan["Date"].unique())].copy()
    base_features = shadow.design_lgb(target)
    extended_features = shadow.design_lgb(target, shadow.EXTRA_FEATURES)

    shadow_models = load_shadow_models()
    target = _predict_official(target, base_features, regime, shadow_models)

    relative_width = ((target["range_high"] - target["range_low"])
                      / target["predicted"].clip(lower=1))
    target["risk"] = np.select([relative_width > 0.45, relative_width > 0.30],
                               ["HIGH", "MEDIUM"], "LOW")
    target["regime"] = regime
    if include_drivers:
        target["drivers"] = target.apply(explain_drivers, axis=1)

    if shadow_models is not None:
        _log_shadow_predictions(target, extended_features, regime, shadow_models)
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
