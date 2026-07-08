"""Shadow-mode + lag-fallback support for the lunch-counter model.

Three model sets live alongside the frozen incumbent:
  * challenger   — LGB with 6 extra features (lag2, wd_lag2, ewm3/7, calendar
                   gaps), 3-seed averaged. Shadow only: logged, never shown.
  * blend        — 0.7*challenger + 0.3*ExtraTrees. Shadow only.
  * fallback     — models retrained on 2-day-old lags, used for the OFFICIAL
                   numbers when yesterday's actuals are not in the history yet
                   (a lag-1 model silently fed stale lags loses ~2.1pt WAPE;
                   the dedicated fallback loses only ~0.5pt).

Every history feature here is shift-protected exactly like features.py;
build_cd_k(k=2) shifts the daily chain one extra day (weekday-chain features
come from ~a week back and are unaffected by a missing yesterday).
"""
from pathlib import Path

import numpy as np
import pandas as pd

import features as F
from features import NUM_FEATURES, CAT_FEATURES

BUNDLE_DIR = Path(__file__).resolve().parent
SHADOW_DIR = BUNDLE_DIR / "artifacts_shadow"

EXTRA_FEATURES = ["lag2", "wd_lag2", "ewm3", "ewm7", "gap_prev", "gap_next"]
SUBCAT_FEATURES = ["sc_pop_mean", "sc_pop_max", "sc_rel_pop", "sc_dsl_mean",
                   "sc_repeat2", "sc_novel14", "n_subcats"]
INTERACTION_FEATURES = ["ix_wdshare_gap", "ix_wdlag2_gap"]
CHALLENGER4_FEATURES = EXTRA_FEATURES + SUBCAT_FEATURES + INTERACTION_FEATURES
CHALLENGER_PARAMS = {"nl": 10, "mcs": 8, "ff": 0.35, "rl": 0.5, "l1": 0.5,
                     "lr": 0.015, "bf": 0.8, "bfreq": 1}
INCUMBENT_PARAMS = {"nl": 7, "mcs": 10, "ff": 0.6, "rl": 1, "lr": 0.03}
CAT_LEVELS = {"Counter Name": ["North Non Veg", "North Veg", "South Non Veg", "South Veg"],
              "weekday": ["Friday", "Monday", "Thursday", "Tuesday", "Wednesday"]}
BLEND_W = 0.7  # challenger weight in the LGB/ExtraTrees blend


def build_cd_k(df, k=1, holiday_dates=None):
    """features.build_all + the challenger features, with the daily lag chain
    shifted by k. IMPORTANT: all shift-chain columns are computed on one
    frame BEFORE any merge (merges reset the index and would misalign
    groupby results).

    gap_next uses the CALENDAR (weekends + holiday_dates) when holiday dates
    are provided — genuinely known the evening before, and immune to the
    unplanned-closure leak that a data-derived gap has. Falls back to the
    data-derived gap when holiday_dates is None (legacy behaviour)."""
    cd = F.build_counterday(df)
    cd = F.add_calendar(cd)
    cd = F.add_active_and_competitor(cd)
    cd = cd.sort_values(["Counter Name", "Date"])
    gc = cd.groupby("Counter Name", group_keys=False)
    gw = cd.groupby(["Counter Name", "weekday"], group_keys=False)
    cd["share"] = cd["cc"] / cd["tlc"]
    cd["lag1"] = gc["cc"].shift(k)
    cd["roll3"] = gc["cc"].apply(lambda s: s.shift(k).rolling(3, min_periods=1).mean())
    cd["roll7"] = gc["cc"].apply(lambda s: s.shift(k).rolling(7, min_periods=2).mean())
    cd["roll14"] = gc["cc"].apply(lambda s: s.shift(k).rolling(14, min_periods=3).mean())
    cd["roll7_std"] = gc["cc"].apply(lambda s: s.shift(k).rolling(7, min_periods=3).std())
    cd["wd_lag1"] = gw["cc"].shift(1)
    cd["wd_roll4"] = gw["cc"].apply(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    cd["share_roll7"] = gc["share"].apply(lambda s: s.shift(k).rolling(7, min_periods=2).mean())
    cd["wd_share_roll4"] = gw["share"].apply(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    cd["lag2"] = gc["cc"].shift(k + 1)
    cd["wd_lag2"] = gw["cc"].shift(2)
    cd["ewm3"] = gc["cc"].apply(lambda s: s.shift(k).ewm(span=3, min_periods=1).mean())
    cd["ewm7"] = gc["cc"].apply(lambda s: s.shift(k).ewm(span=7, min_periods=2).mean())
    dtot = cd.groupby("Date")["tlc"].first().sort_index()
    dd = pd.DataFrame({"tlc_lag1": dtot.shift(k),
                       "tlc_roll5": dtot.shift(k).rolling(5, min_periods=2).mean(),
                       "tlc_roll10": dtot.shift(k).rolling(10, min_periods=3).mean()}).reset_index()
    wmap = cd.groupby("Date")["weekday"].first()
    dw = (dtot.groupby(wmap, group_keys=False)
          .apply(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
          .rename("tlc_wd_roll4").rename_axis("Date").reset_index())
    dts = cd[["Date"]].drop_duplicates().sort_values("Date")
    dts["gap_prev"] = dts["Date"].diff().dt.days
    if holiday_dates is not None:
        dts["gap_next"] = dts["Date"].apply(
            lambda d: _calendar_gap_next(d, holiday_dates))
    else:
        dts["gap_next"] = dts["Date"].diff(-1).dt.days.abs()
    cd = cd.merge(dd, on="Date", how="left").merge(dw, on="Date", how="left") \
           .merge(dts, on="Date", how="left")
    cd = cd.sort_values(["Date", "Counter Name"]).reset_index(drop=True)
    cd = _add_subcat_features(df, cd, k)
    cd["ix_wdshare_gap"] = cd["wd_share_roll4"] / (cd["gap_next"] + 1)
    cd["ix_wdlag2_gap"] = cd["wd_lag2"] / (cd["gap_next"] + 1)
    return cd


def _calendar_gap_next(date, holiday_dates):
    """Days until the next working day, walked on the calendar."""
    from datetime import timedelta
    day = pd.Timestamp(date).date() + timedelta(days=1)
    gap = 1
    while day.weekday() >= 5 or day in holiday_dates:
        day += timedelta(days=1)
        gap += 1
    return gap


_DSL_CAP_DAYS = 60      # days-since-last cap for never/rarely-seen subcats
_STAPLE_SHARE = 0.5     # subcats on >50% of counter-days are staples: no signal


def _add_subcat_features(item_df, cd, k):
    """Menu-composition features from the curated Sub Category taxonomy.

    Leakage rules: popularity stats use target values (cc) and therefore lag
    by k working days (matching the regime's information set); recency stats
    (days-since-last, repeat, novel) use only menu APPEARANCES, which are
    known in advance, and lag by one working day. Staple subcats that appear
    on most counter-days carry no signal and are excluded from aggregates.
    """
    items = item_df.copy()
    if "Sub Category" not in items or items["Sub Category"].isna().all():
        items["Sub Category"] = items["Category"]
    items["Sub Category"] = items["Sub Category"].fillna(items["Category"])

    subcats_by_cd = (items.groupby([items["Date"].dt.normalize(), "Counter Name"])
                     ["Sub Category"].agg(lambda s: sorted(set(s))))
    dates = sorted(cd["Date"].unique())
    date_index = {d: i for i, d in enumerate(dates)}
    cc_by_cd = cd.set_index(["Date", "Counter Name"])["cc"].to_dict()

    pop_sum, pop_count = {}, {}                  # per-subcat, lagged by k
    overall_sum, overall_count = 0.0, 0
    last_seen, appear_count = {}, {}             # per-subcat, lagged by 1
    pending_pop = []                             # per-date cc contributions awaiting the k-lag
    rows = {}

    for i, date in enumerate(dates):
        # dates now k working days in the past enter the popularity stats
        if len(pending_pop) >= k:
            for subs, cc in pending_pop.pop(0):
                if not np.isnan(cc):
                    overall_sum += cc
                    overall_count += 1
                    for s in subs:
                        pop_sum[s] = pop_sum.get(s, 0.0) + cc
                        pop_count[s] = pop_count.get(s, 0) + 1

        day_rows = [(counter, subs) for (d, counter), subs in subcats_by_cd.items()
                    if d == date]
        for counter, subs in day_rows:
            informative = [s for s in subs
                           if appear_count.get(s, 0) <= _STAPLE_SHARE * max(i, 1)]
            pops = [pop_sum[s] / pop_count[s] for s in informative
                    if pop_count.get(s, 0) > 0]
            overall_mean = overall_sum / overall_count if overall_count else np.nan
            dsl = [min(i - last_seen[s], _DSL_CAP_DAYS) if s in last_seen
                   else _DSL_CAP_DAYS for s in informative]
            rows[(date, counter)] = {
                "sc_pop_mean": np.mean(pops) if pops else np.nan,
                "sc_pop_max": np.max(pops) if pops else np.nan,
                "sc_rel_pop": (np.mean(pops) / overall_mean
                               if pops and overall_mean else np.nan),
                "sc_dsl_mean": np.mean(dsl) if dsl else np.nan,
                "sc_repeat2": np.mean([d <= 2 for d in dsl]) if dsl else np.nan,
                "sc_novel14": np.mean([d > 14 for d in dsl]) if dsl else np.nan,
                "n_subcats": len(subs),
            }
        pending_pop.append([(subs, cc_by_cd.get((date, counter), np.nan))
                            for counter, subs in day_rows])
        # menu appearances become known history after the day itself
        for counter, subs in day_rows:
            for s in subs:
                last_seen[s] = i
                appear_count[s] = appear_count.get(s, 0) + 1

    feature_frame = pd.DataFrame.from_dict(rows, orient="index")
    feature_frame.index = pd.MultiIndex.from_tuples(feature_frame.index,
                                                    names=["Date", "Counter Name"])
    return cd.merge(feature_frame.reset_index(), on=["Date", "Counter Name"],
                    how="left")


def design_lgb(cd, extra=()):
    X = cd[list(NUM_FEATURES) + list(extra) + CAT_FEATURES].copy()
    for c in CAT_FEATURES:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])
    return X


def design_sk(cd, med=None, cols=None):
    """One-hot design for the sklearn entrants (ExtraTrees, KNN)."""
    X = cd[list(NUM_FEATURES) + CHALLENGER4_FEATURES].copy()
    X = pd.concat([X, pd.get_dummies(cd["Counter Name"], prefix="cn").astype(int),
                   pd.get_dummies(cd["weekday"], prefix="wd").astype(int)], axis=1)
    if cols is not None:
        X = X.reindex(columns=cols, fill_value=0)
    if med is None:
        med = X.median(numeric_only=True)
    return X.fillna(med), med, X.columns


def load_shadow():
    """Load all shadow/fallback artifacts (None if not built)."""
    import json
    import lightgbm as lgb
    if not (SHADOW_DIR / "meta.json").exists():
        return None
    meta = json.load(open(SHADOW_DIR / "meta.json"))
    out = {"meta": meta}
    for name in ["challenger4", "chal4_fb", "fb_point"]:
        out[name] = [lgb.Booster(model_file=str(SHADOW_DIR / f"{name}_s{s}.txt"))
                     for s in meta["seeds"]]
    for q in (10, 75, 90):
        out[f"fb_q{q}"] = lgb.Booster(model_file=str(SHADOW_DIR / f"fb_q{q}.txt"))
    import joblib
    for name in ("et_tuned", "knn"):
        try:
            out[name] = joblib.load(SHADOW_DIR / f"{name}.joblib")
        except Exception:
            out[name] = None
    return out


def load_incumbent():
    """The frozen production boosters + conformal config."""
    import pickle
    import lightgbm as lgb
    artifacts = BUNDLE_DIR / "artifacts"
    with open(artifacts / "final_config.pkl", "rb") as f:
        config = pickle.load(f)
    boosters = {name: lgb.Booster(model_file=str(artifacts / f"model_{name}.txt"))
                for name in ("point", "q10", "q75", "q90")}
    return boosters, config


def seed_avg(boosters, X):
    return np.mean([np.clip(b.predict(X), 0, None) for b in boosters], axis=0)


# ------------------------------------------------------------ plan scoring ---
PLACEHOLDER_NAN = ["Receiving Qty", "Bainmarie Wastage"]
PLACEHOLDER_ZERO = ["Headcount", "Total Lunch Consumed", "Counter Ordered",
                    "Counter Consumed"]

# suggested-order policies: quantile head + conformal treatment + buffer.
# Service rates measured on June 2026 (model_research/FINDINGS.md round 2).
ORDER_POLICIES = {
    "standard": {"label": "Standard (P75, ~70% cover)", "q": "q75",
                 "use_cqr": False, "use_corr": True, "buffer": 1.0},
    "high": {"label": "High service (P90+CQR, ~87% cover)", "q": "q90",
             "use_cqr": True, "use_corr": False, "buffer": 1.0},
    "no_shortage": {"label": "No shortage (P90+CQR+5%, 0 short days in June)",
                    "q": "q90", "use_cqr": True, "use_corr": False, "buffer": 1.05},
}


def normalize_plan(plan, holiday_dates):
    """Shape a plan (one row per planned item) like a history row. Day Type /
    Panchangam auto-derive from the date (user-provided values win); target-
    side columns get placeholders that never enter the feature set."""
    import auto_calendar
    plan = plan.copy()
    plan["Date"] = pd.to_datetime(plan["Date"])
    plan = auto_calendar.fill_calendar_columns(plan, holiday_dates)
    plan["Month"] = plan["Date"].dt.month_name()
    plan["Weekday"] = plan["Date"].dt.day_name()
    for column in PLACEHOLDER_NAN:
        plan[column] = np.nan
    for column in PLACEHOLDER_ZERO:
        plan[column] = 0
    return plan


def detect_lag_regime(history, plan):
    """'fresh' when the last served day is the working day right before the
    first plan day (yesterday's actuals available), else 'stale'."""
    gap = int(np.busday_count(history["Date"].max().date(),
                              pd.to_datetime(plan["Date"]).min().date()))
    return "fresh" if gap <= 1 else "stale"


def _subcat_for_plan(history, plan):
    """Plan rows need a Sub Category for the feature builder: map from the
    item catalog, fall back to the item's Category for novel items."""
    if "Sub Category" not in history.columns:
        return plan
    mapping = (history.dropna(subset=["Sub Category"])
               .groupby("Item Name")["Sub Category"]
               .agg(lambda s: s.mode().iat[0]).to_dict())
    plan = plan.copy()
    if "Sub Category" not in plan.columns:
        plan["Sub Category"] = plan["Item Name"].map(mapping).fillna(plan["Category"])
    if "Review Flag" in history.columns and "Review Flag" not in plan.columns:
        plan["Review Flag"] = np.nan
    return plan


def score_plan(history, plan, holiday_dates, order_policy="standard"):
    """Score a normalized plan with the official model (incumbent when fresh,
    lag-2 fallback when stale) and every shadow entrant. Returns the target
    frame with official columns plus a dict of shadow prediction arrays."""
    plan = normalize_plan(plan, holiday_dates)
    plan = _subcat_for_plan(history, plan)
    regime = detect_lag_regime(history, plan)
    lag_depth = 1 if regime == "fresh" else 2

    combined = pd.concat([history, plan[history.columns]], ignore_index=True)
    counter_days = build_cd_k(combined, k=lag_depth, holiday_dates=holiday_dates)
    target = counter_days[counter_days["Date"].isin(plan["Date"].unique())].copy()
    base_X = design_lgb(target)
    c4_X = design_lgb(target, CHALLENGER4_FEATURES)

    shadow_models = load_shadow()
    policy = ORDER_POLICIES[order_policy]
    if regime == "fresh":
        boosters, config = load_incumbent()
        point = boosters["point"].predict(base_X)
        quantiles = {q: boosters[q].predict(base_X) for q in ("q10", "q75", "q90")}
        cqr, corr = config["cqr_Q80"], config["order_corr"]
    else:
        meta = shadow_models["meta"]
        point = seed_avg(shadow_models["fb_point"], base_X)
        quantiles = {q: shadow_models[f"fb_{q}"].predict(base_X)
                     for q in ("q10", "q75", "q90")}
        cqr, corr = meta["fb_cqr_Q80"], meta["fb_order_corr"]

    target["predicted"] = np.clip(point, 0, None).round(0)
    target["range_low"] = np.clip(np.clip(quantiles["q10"], 0, None) - cqr, 0, None).round(0)
    target["range_high"] = (np.clip(quantiles["q90"], 0, None) + cqr).round(0)
    order = np.clip(quantiles[policy["q"]], 0, None)
    if policy["use_cqr"]:
        order = order + cqr
    if policy["use_corr"]:
        order = order + corr
    target["suggested_order"] = (order * policy["buffer"]).round(-1)
    relative_width = ((target["range_high"] - target["range_low"])
                      / target["predicted"].clip(lower=1))
    target["risk"] = np.select([relative_width > 0.45, relative_width > 0.30],
                               ["HIGH", "MEDIUM"], "LOW")
    target["regime"] = regime

    shadow_preds = {}
    if shadow_models is not None:
        entrants = shadow_models["challenger4" if regime == "fresh" else "chal4_fb"]
        shadow_preds["challenger4"] = seed_avg(entrants, c4_X)
        shadow_preds["et"] = np.full(len(target), np.nan)
        shadow_preds["hybrid"] = np.full(len(target), np.nan)
        if regime == "fresh":
            if shadow_models["et_tuned"] is not None:
                et = shadow_models["et_tuned"]
                X_sk, _, _ = design_sk(target, et["medians"], et["columns"])
                shadow_preds["et"] = np.clip(et["model"].predict(X_sk), 0, None)
            if shadow_models["knn"] is not None:
                knn = shadow_models["knn"]
                X_sk, _, _ = design_sk(target, knn["medians"], knn["columns"])
                knn_pred = np.clip(knn["model"].predict(knn["scaler"].transform(X_sk)), 0, None)
                w = shadow_models["meta"].get("hybrid_lgb_weight", 0.65)
                shadow_preds["hybrid"] = (w * shadow_preds["challenger4"]
                                          + (1 - w) * knn_pred)
    return target, shadow_preds


def log_shadow_run(target, shadow_preds, path):
    """Append this run's official + shadow predictions to the shadow log."""
    rows = [{"Date": str(pd.Timestamp(d).date()), "Counter Name": c,
             "regime": r, "pred_official": float(p),
             **{f"pred_{name}": (None if np.isnan(v[i]) else float(v[i]))
                for name, v in shadow_preds.items()},
             "logged_at": pd.Timestamp.now().isoformat(timespec="seconds")}
            for i, (d, c, r, p) in enumerate(zip(target["Date"], target["Counter Name"],
                                                 target["regime"], target["predicted"]))]
    return log_shadow(rows, path)


def log_shadow(rows, path):
    """Append shadow predictions; dedupe on (Date, Counter), keep latest."""
    new = pd.DataFrame(rows)
    if Path(path).exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True)
    new = new.drop_duplicates(subset=["Date", "Counter Name"], keep="last")
    new.to_csv(path, index=False)
    return len(new)
