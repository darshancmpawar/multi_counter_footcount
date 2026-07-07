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
CHALLENGER_PARAMS = {"nl": 10, "mcs": 8, "ff": 0.35, "rl": 0.5, "l1": 0.5,
                     "lr": 0.015, "bf": 0.8, "bfreq": 1}
INCUMBENT_PARAMS = {"nl": 7, "mcs": 10, "ff": 0.6, "rl": 1, "lr": 0.03}
CAT_LEVELS = {"Counter Name": ["North Non Veg", "North Veg", "South Non Veg", "South Veg"],
              "weekday": ["Friday", "Monday", "Thursday", "Tuesday", "Wednesday"]}
BLEND_W = 0.7  # challenger weight in the LGB/ExtraTrees blend


def build_cd_k(df, k=1):
    """features.build_all + the 6 extra features, with the daily lag chain
    shifted by k. IMPORTANT: all shift-chain columns are computed on one
    frame BEFORE any merge (merges reset the index and would misalign
    groupby results)."""
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
    dts["gap_next"] = dts["Date"].diff(-1).dt.days.abs()
    cd = cd.merge(dd, on="Date", how="left").merge(dw, on="Date", how="left") \
           .merge(dts, on="Date", how="left")
    return cd.sort_values(["Date", "Counter Name"]).reset_index(drop=True)


def design_lgb(cd, extra=()):
    X = cd[list(NUM_FEATURES) + list(extra) + CAT_FEATURES].copy()
    for c in CAT_FEATURES:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])
    return X


def design_sk(cd, med=None, cols=None):
    """One-hot design for the sklearn blend member."""
    X = cd[list(NUM_FEATURES) + EXTRA_FEATURES].copy()
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
    for name in ["challenger", "chal_fb", "fb_point"]:
        out[name] = [lgb.Booster(model_file=str(SHADOW_DIR / f"{name}_s{s}.txt"))
                     for s in meta["seeds"]]
    for q in (10, 75, 90):
        out[f"fb_q{q}"] = lgb.Booster(model_file=str(SHADOW_DIR / f"fb_q{q}.txt"))
    try:
        import joblib
        out["blend_et"] = joblib.load(SHADOW_DIR / "blend_et.joblib")
    except Exception:
        out["blend_et"] = None
    return out


def seed_avg(boosters, X):
    return np.mean([np.clip(b.predict(X), 0, None) for b in boosters], axis=0)


def log_shadow(rows, path):
    """Append shadow predictions; dedupe on (Date, Counter), keep latest."""
    new = pd.DataFrame(rows)
    if Path(path).exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True)
    new = new.drop_duplicates(subset=["Date", "Counter Name"], keep="last")
    new.to_csv(path, index=False)
    return len(new)
