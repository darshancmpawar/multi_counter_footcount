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
# calendar-known festival/holiday-severity flags (auto_calendar.festival_flags)
# — pre-registered "festival" shadow entrant = incumbent features + these
FESTIVAL_FEATURES = ["fest_any", "fest_op_today", "adj_hol_important",
                     "adj_hol_compulsory", "adj_hol_shutdown"]

# Trailing level-shift corrector for the OFFICIAL numbers (July 2026 round):
# if >=80% of the last 10 served days erred in the same direction at day
# level, scale the forecast by the trailing actual/predicted ratio (max
# +/-15%). Gate keeps it dormant in stable regimes (CV-neutral); in the
# June-July 2026 demand shift it recovered ~3.3pt counter WAPE.
DEBIAS = {"window": 10, "min_days": 5, "gate": 0.80, "clip": (0.85, 1.15)}

# Dates whose Headcount is IMPUTED as TLC / weekday-median(TLC/HC)
# (workbook Considerations D1 + G4). Their plates-per-head is circular —
# it equals the historical ratio by construction — so they must be masked
# out of any propensity estimation, drift monitoring or per-head features.
IMPUTED_HEADCOUNT_DATES = frozenset(pd.Timestamp(d).date() for d in (
    "2026-04-13", "2026-04-20", "2026-04-24",           # D1
    "2026-07-01", "2026-07-02", "2026-07-03", "2026-07-06",  # G4
))
CHALLENGER_PARAMS = {"nl": 10, "mcs": 8, "ff": 0.35, "rl": 0.5, "l1": 0.5,
                     "lr": 0.015, "bf": 0.8, "bfreq": 1}
INCUMBENT_PARAMS = {"nl": 7, "mcs": 10, "ff": 0.6, "rl": 1, "lr": 0.03}
CAT_LEVELS = {"Counter Name": ["North Non Veg", "North Veg", "South Non Veg", "South Veg"],
              "weekday": ["Friday", "Monday", "Thursday", "Tuesday", "Wednesday"]}
BLEND_W = 0.7  # challenger weight in the LGB/ExtraTrees blend


def build_cd_k(df, k=1, holiday_dates=None, subcat_map=None):
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
    cd = _add_subcat_features(df, cd, k, subcat_map)
    cd["ix_wdshare_gap"] = cd["wd_share_roll4"] / (cd["gap_next"] + 1)
    cd["ix_wdlag2_gap"] = cd["wd_lag2"] / (cd["gap_next"] + 1)
    cd = _add_festival_features(cd, holiday_dates)
    return cd


def _add_festival_features(cd, holiday_dates):
    """Calendar-derived festival flags (auto_calendar.festival_flags) — a
    function of the date and the Holiday List alone, so identical at train
    and score time (plans carry no Festival column). All zero when no
    holiday calendar is available."""
    import auto_calendar
    if holiday_dates is None:
        for name in FESTIVAL_FEATURES:
            cd[name] = 0
        return cd
    flags = {date: auto_calendar.festival_flags(date, holiday_dates)
             for date in cd["Date"].unique()}
    for name in FESTIVAL_FEATURES:
        cd[name] = cd["Date"].map(lambda d: flags[d][name])
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


_HEADWORD_MIN_ITEMS = 3   # a (Category, head-word) group needs >=3 distinct items


def freeze_headword_map(item_df):
    """Item -> subcat label, derived ONCE on training data and stored in
    meta.json. subcat = Category + '|' + last token of Item Name when that
    (Category, head-word) group has >= 3 distinct items, else Category.

    Freezing is mandatory: recomputing the >=3 rule on accumulating data
    flips ~12 items' labels retroactively and breaks truncation invariance
    (model_research/FINDINGS.md round 4). Novel items at score time fall back
    to their Category via the fillna in _add_subcat_features."""
    head = item_df["Item Name"].str.strip().str.split().str[-1].str.lower()
    group_sizes = (item_df.assign(head=head)
                   .groupby(["Category", "head"])["Item Name"].nunique())
    valid = set(group_sizes[group_sizes >= _HEADWORD_MIN_ITEMS].index)
    labels = np.where([(c, h) in valid for c, h in zip(item_df["Category"], head)],
                      item_df["Category"] + "|" + head, item_df["Category"])
    return dict(zip(item_df["Item Name"], labels))


def _add_subcat_features(item_df, cd, k, subcat_map=None):
    """Menu-composition features on the frozen head-word subcategory (the
    variant that helps, leakage-safe with a frozen map — round 4).

    Popularity = expanding mean of the WHOLE counter-day cc over the subcat's
    own appearance sequence at that counter, shifted by k appearances.
    Recency = calendar days since the subcat last appeared at that counter.
    Aggregated over unique subcats per counter-day. NaNs left for LightGBM.
    """
    items = item_df.copy()
    items["Date"] = pd.to_datetime(items["Date"])
    if subcat_map is not None:
        items["subcat"] = items["Item Name"].map(subcat_map).fillna(items["Category"])
    elif "Sub Category" in items and not items["Sub Category"].isna().all():
        items["subcat"] = items["Sub Category"].fillna(items["Category"]).astype(str)
    else:
        items["subcat"] = items["Category"].astype(str)

    counter_day_cc = (items.groupby(["Date", "Counter Name"])["Counter Consumed"]
                      .first().reset_index().rename(columns={"Counter Consumed": "cc_item"}))
    presence = (items.groupby(["Date", "Counter Name", "subcat"]).size()
                .reset_index(name="_n").merge(counter_day_cc, on=["Date", "Counter Name"]))

    # popularity: expanding mean over the subcat's appearance sequence, shift k
    presence = presence.sort_values(["Counter Name", "subcat", "Date"])
    seq = presence.groupby(["Counter Name", "subcat"], group_keys=False)
    presence["pop"] = seq["cc_item"].apply(lambda s: s.shift(k).expanding().mean())
    presence["dsl"] = seq["Date"].diff().dt.days

    # rel-pop denominator: counter's own expanding mean cc, shift 1 day
    cday = counter_day_cc.sort_values(["Counter Name", "Date"])
    cday["cmean"] = (cday.groupby("Counter Name", group_keys=False)["cc_item"]
                     .apply(lambda s: s.shift(1).expanding().mean()))
    presence = presence.merge(cday[["Date", "Counter Name", "cmean"]],
                              on=["Date", "Counter Name"])

    def _agg(group):
        return pd.Series({
            "sc_pop_mean": group["pop"].mean(),
            "sc_pop_max": group["pop"].max(),
            "sc_rel_pop": (group["pop"].mean() / group["cmean"].iloc[0]
                           if pd.notna(group["cmean"].iloc[0]) else np.nan),
            "sc_dsl_mean": group["dsl"].clip(upper=21).mean(),
            "sc_repeat2": (group["dsl"] <= 2).mean(),
            "sc_novel14": ((group["dsl"] > 14) | group["dsl"].isna()).mean(),
            "n_subcats": group["subcat"].nunique(),
        })
    features = (presence.groupby(["Date", "Counter Name"]).apply(_agg).reset_index())
    return cd.merge(features, on=["Date", "Counter Name"], how="left")

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


_MODEL_CACHE = {}   # process-lifetime cache; restart (or clear) after a promote


def clear_model_cache():
    _MODEL_CACHE.clear()


def load_shadow():
    """Load all shadow/fallback artifacts (None if not built). Cached for
    the process lifetime — score_plan runs on every forecast and must not
    re-read a dozen boosters from disk each time."""
    import json
    import lightgbm as lgb
    if "shadow" in _MODEL_CACHE:
        return _MODEL_CACHE["shadow"]
    if not (SHADOW_DIR / "meta.json").exists():
        return None   # not cached: the bundle may be built later in-process
    meta = json.load(open(SHADOW_DIR / "meta.json"))
    out = {"meta": meta}
    for name in ["challenger4", "chal4_fb", "fb_point"]:
        out[name] = [lgb.Booster(model_file=str(SHADOW_DIR / f"{name}_s{s}.txt"))
                     for s in meta["seeds"]]
    for name in ["festival", "fest_fb"]:   # pre-registered July 2026 entrant
        paths = [SHADOW_DIR / f"{name}_s{s}.txt" for s in meta["seeds"]]
        out[name] = ([lgb.Booster(model_file=str(p)) for p in paths]
                     if all(p.exists() for p in paths) else None)
    for q in (10, 75, 90):
        out[f"fb_q{q}"] = lgb.Booster(model_file=str(SHADOW_DIR / f"fb_q{q}.txt"))
    import joblib
    import warnings
    for name in ("et_tuned", "knn"):
        path = SHADOW_DIR / f"{name}.joblib"
        if not path.exists():
            out[name] = None
            continue
        try:
            out[name] = joblib.load(path)
        except Exception as error:
            # a present-but-unloadable entrant must be LOUD: this exact
            # failure (missing pyarrow) silently dropped et/knn from the
            # month-end referee for weeks
            warnings.warn(f"shadow entrant '{name}' exists but failed to "
                          f"load ({error!r}) — it will log NaN and be "
                          f"missing from the month-end adjudication")
            out[name] = None
    _MODEL_CACHE["shadow"] = out
    return out


def load_incumbent():
    """The deployed production boosters + conformal config (cached)."""
    import pickle
    import lightgbm as lgb
    if "incumbent" in _MODEL_CACHE:
        return _MODEL_CACHE["incumbent"]
    artifacts = BUNDLE_DIR / "artifacts"
    with open(artifacts / "final_config.pkl", "rb") as f:
        config = pickle.load(f)
    boosters = {name: lgb.Booster(model_file=str(artifacts / f"model_{name}.txt"))
                for name in ("point", "q10", "q75", "q90")}
    _MODEL_CACHE["incumbent"] = (boosters, config)
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


def trailing_debias_factor(counter_days, last_history_date, predict_fn):
    """Level-shift corrector for the official numbers (config in DEBIAS).

    Reconstructs what the current official model predicts for the last N
    served days (features are truncation-invariant, so this equals what it
    would have predicted those evenings) and compares with actuals. Applies
    only when the day-level errors are consistently one-sided — the gate
    keeps the corrector dormant through ordinary noise and activates it in
    genuine demand-level shifts (e.g. the 15 Jun 2026 per-head drop).
    Returns 1.0 whenever the evidence is insufficient."""
    served = counter_days[counter_days["Date"] <= last_history_date]
    dates = sorted(served["Date"].unique())[-DEBIAS["window"]:]
    if len(dates) < DEBIAS["min_days"]:
        return 1.0
    rows = served[served["Date"].isin(dates)]
    daily = (rows.assign(pred=predict_fn(rows))
             .groupby("Date").agg(actual=("cc", "sum"), pred=("pred", "sum")))
    over_share = float((daily["pred"] > daily["actual"]).mean())
    if max(over_share, 1.0 - over_share) < DEBIAS["gate"]:
        return 1.0
    factor = daily["actual"].sum() / max(daily["pred"].sum(), 1e-9)
    return float(np.clip(factor, *DEBIAS["clip"]))


def detect_lag_regime(history, plan, holiday_dates=None):
    """'fresh' when the last served day is the WORKING day right before the
    first plan day (yesterday's actuals available), else 'stale'.

    Holiday-aware: without the holiday calendar, the first working day after
    a mid-week holiday counts a busday gap of 2 and is misrouted to the
    lag-2 fallback even though the last working day's actuals are current."""
    holidays = sorted(pd.Timestamp(d).date() for d in (holiday_dates or ()))
    gap = int(np.busday_count(history["Date"].max().date(),
                              pd.to_datetime(plan["Date"]).min().date(),
                              holidays=holidays))
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
    # reference-only columns the workbook has but a plan never will
    # (e.g. Festival) — NaN-fill so plan[history.columns] can't KeyError
    for column in history.columns.difference(plan.columns):
        plan[column] = np.nan
    regime = detect_lag_regime(history, plan, holiday_dates)
    lag_depth = 1 if regime == "fresh" else 2

    shadow_models = load_shadow()
    if regime == "stale" and shadow_models is None:
        raise RuntimeError(
            "Yesterday's actuals are missing (stale regime) and the lag-2 "
            "fallback set is not built — run retrain.py to build "
            "artifacts_shadow/, or update the history workbook through the "
            "last served day.")
    subcat_map = (shadow_models["meta"].get("subcat_map")
                  if shadow_models is not None else None)
    combined = pd.concat([history, plan[history.columns]], ignore_index=True)
    counter_days = build_cd_k(combined, k=lag_depth, holiday_dates=holiday_dates,
                              subcat_map=subcat_map)
    target = counter_days[counter_days["Date"].isin(plan["Date"].unique())].copy()
    base_X = design_lgb(target)
    c4_X = design_lgb(target, CHALLENGER4_FEATURES)

    policy = ORDER_POLICIES[order_policy]
    if regime == "fresh":
        boosters, config = load_incumbent()
        point = boosters["point"].predict(base_X)
        quantiles = {q: boosters[q].predict(base_X) for q in ("q10", "q75", "q90")}
        cqr, corr = config["cqr_Q80"], config["order_corr"]
        predict_official = lambda rows: np.clip(  # noqa: E731
            boosters["point"].predict(design_lgb(rows)), 0, None)
    else:
        meta = shadow_models["meta"]
        point = seed_avg(shadow_models["fb_point"], base_X)
        quantiles = {q: shadow_models[f"fb_{q}"].predict(base_X)
                     for q in ("q10", "q75", "q90")}
        cqr, corr = meta["fb_cqr_Q80"], meta["fb_order_corr"]
        predict_official = lambda rows: seed_avg(  # noqa: E731
            shadow_models["fb_point"], design_lgb(rows))

    # level-shift corrector: scales the model outputs, leaves the calibrated
    # conformal margins (cqr/corr, added below) untouched
    factor = trailing_debias_factor(counter_days, history["Date"].max(),
                                    predict_official)
    point_raw = np.clip(point, 0, None)
    point = point_raw * factor
    quantiles = {q: np.clip(v, 0, None) * factor for q, v in quantiles.items()}

    target["pred_raw"] = point_raw.round(0)
    target["debias_factor"] = factor
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
        # raw official goes to the log too: the referee compares models on
        # their own outputs, the debias layer is scored as official-vs-raw
        shadow_preds["official_raw"] = point_raw
        entrants = shadow_models["challenger4" if regime == "fresh" else "chal4_fb"]
        shadow_preds["challenger4"] = seed_avg(entrants, c4_X)
        festival_set = shadow_models.get("festival" if regime == "fresh"
                                         else "fest_fb")
        if festival_set is not None:
            fest_X = design_lgb(target, FESTIVAL_FEATURES)
            shadow_preds["festival"] = seed_avg(festival_set, fest_X)
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
