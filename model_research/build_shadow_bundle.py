"""Train and freeze the July shadow roster + lag-2 fallbacks into
siemens_model_bundle/artifacts_shadow/. Pre-registration: run BEFORE live
actuals accumulate; no mid-month changes.

Roster (CV WAPE on the 4-fold expanding-window harness, round 3):
  challenger4_s{seed}.txt   LGB, challenger params + lag/EWMA/gap/interaction
                            features (10.80)
  et_tuned.joblib           ExtraTrees msl=5 mf=0.6 on the same design (10.70)
  knn.joblib                KNN k=10, L1, distance-weighted, scaled (10.92);
                            scored as hybrid 0.65*LGB + 0.35*KNN (~10.4)
  chal4_fb_s{seed}.txt      lag-2 fallback of challenger4 (stale regime)
  fb_point/fb_q10/75/90     lag-2 fallback OFFICIAL set + conformal meta

Protocol: iteration counts from early stopping on Apr-May validation
(train <= Mar), retrain on the full history at that count; fallback conformal
corrections fitted on Apr-May only. gap_next is calendar-based (Holiday List).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

BUNDLE = Path(__file__).resolve().parent.parent / "siemens_model_bundle"
sys.path.insert(0, str(BUNDLE))
import auto_calendar  # noqa: E402
from shadow import (CHALLENGER4_FEATURES, CHALLENGER_PARAMS, INCUMBENT_PARAMS,  # noqa: E402
                    build_cd_k, design_lgb, design_sk, freeze_headword_map)

HIST = Path(__file__).resolve().parent.parent / "Lunch_Master_Data_FINAL(cleaned).xlsx"
OUT = BUNDLE / "artifacts_shadow"
OUT.mkdir(exist_ok=True)
TRAIN_END, VAL_END = pd.Timestamp("2026-03-31"), pd.Timestamp("2026-05-31")
SEEDS = (1, 42, 99)
ET_CONFIG = dict(n_estimators=500, min_samples_leaf=5, max_features=0.6,
                 n_jobs=-1, random_state=42)
KNN_CONFIG = dict(n_neighbors=10, weights="distance", p=1)
HYBRID_LGB_WEIGHT = 0.65


def lgb_params(p, objective="poisson", seed=42, alpha=None):
    d = dict(objective=objective, learning_rate=p.get("lr", 0.03),
             num_leaves=p.get("nl", 7), min_child_samples=p.get("mcs", 10),
             feature_fraction=p.get("ff", 0.6), lambda_l2=p.get("rl", 1),
             lambda_l1=p.get("l1", 0.0), bagging_fraction=p.get("bf", 1.0),
             bagging_freq=p.get("bfreq", 0), verbose=-1, seed=seed,
             feature_fraction_seed=seed, bagging_seed=seed)
    if alpha is not None:
        d["alpha"] = alpha
    return d


def train_frozen(cd, params, feats, objective="poisson", seed=42, alpha=None):
    tr = cd[cd["Date"] <= TRAIN_END]
    va = cd[(cd["Date"] > TRAIN_END) & (cd["Date"] <= VAL_END)]
    probe = lgb.train(lgb_params(params, objective, seed, alpha),
                      lgb.Dataset(design_lgb(tr, feats), tr["cc"]), 3000,
                      valid_sets=[lgb.Dataset(design_lgb(va, feats), va["cc"])],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
    final = lgb.train(lgb_params(params, objective, seed, alpha),
                      lgb.Dataset(design_lgb(cd, feats), cd["cc"]),
                      probe.best_iteration)
    return final, probe.best_iteration


def main():
    hist = pd.read_excel(HIST, sheet_name="Lunch Master")
    hist["Date"] = pd.to_datetime(hist["Date"])
    holidays = auto_calendar.load_holiday_dates(HIST)
    # freeze the head-word subcat map on ALL current history (this is the
    # train set at build time); stored in meta and reused verbatim at score time
    subcat_map = freeze_headword_map(hist)
    cd1 = build_cd_k(hist, k=1, holiday_dates=holidays, subcat_map=subcat_map)
    cd2 = build_cd_k(hist, k=2, holiday_dates=holidays, subcat_map=subcat_map)
    meta = {"built_on": str(hist["Date"].max().date()),
            "challenger4_features": CHALLENGER4_FEATURES,
            "challenger_params": CHALLENGER_PARAMS,
            "incumbent_params": INCUMBENT_PARAMS,
            "et_config": {k: v for k, v in ET_CONFIG.items() if k != "n_jobs"},
            "knn_config": KNN_CONFIG,
            "hybrid_lgb_weight": HYBRID_LGB_WEIGHT,
            "subcat_map": subcat_map,
            "seeds": list(SEEDS), "n_iters": {}}

    for seed in SEEDS:
        booster, n = train_frozen(cd1, CHALLENGER_PARAMS, CHALLENGER4_FEATURES, seed=seed)
        booster.save_model(str(OUT / f"challenger4_s{seed}.txt"))
        meta["n_iters"][f"challenger4_s{seed}"] = n
        booster, n = train_frozen(cd2, CHALLENGER_PARAMS, CHALLENGER4_FEATURES, seed=seed)
        booster.save_model(str(OUT / f"chal4_fb_s{seed}.txt"))
        meta["n_iters"][f"chal4_fb_s{seed}"] = n

    X_full, medians, columns = design_sk(cd1)
    et = ExtraTreesRegressor(**ET_CONFIG).fit(X_full, cd1["cc"])
    joblib.dump({"model": et, "medians": medians, "columns": list(columns)},
                OUT / "et_tuned.joblib")
    scaler = StandardScaler().fit(X_full)
    knn = KNeighborsRegressor(**KNN_CONFIG).fit(scaler.transform(X_full), cd1["cc"])
    joblib.dump({"model": knn, "scaler": scaler, "medians": medians,
                 "columns": list(columns)}, OUT / "knn.joblib")

    for seed in SEEDS:
        booster, n = train_frozen(cd2, INCUMBENT_PARAMS, (), seed=seed)
        booster.save_model(str(OUT / f"fb_point_s{seed}.txt"))
        meta["n_iters"][f"fb_point_s{seed}"] = n
    fb_val_preds = {}
    tr, va = cd2[cd2["Date"] <= TRAIN_END], cd2[(cd2["Date"] > TRAIN_END) & (cd2["Date"] <= VAL_END)]
    for q in (0.10, 0.75, 0.90):
        booster, n = train_frozen(cd2, INCUMBENT_PARAMS, (), "quantile", 42, q)
        booster.save_model(str(OUT / f"fb_q{int(q*100)}.txt"))
        meta["n_iters"][f"fb_q{int(q*100)}"] = n
        probe = lgb.train(lgb_params(INCUMBENT_PARAMS, "quantile", 42, q),
                          lgb.Dataset(design_lgb(tr, ()), tr["cc"]), 3000,
                          valid_sets=[lgb.Dataset(design_lgb(va, ()), va["cc"])],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
        fb_val_preds[q] = np.clip(probe.predict(design_lgb(va, ())), 0, None)
    y = va["cc"].values
    conformity = np.maximum(fb_val_preds[0.10] - y, y - fb_val_preds[0.90])
    meta["fb_cqr_Q80"] = float(np.quantile(conformity, 0.80))
    meta["fb_order_corr"] = float(max(0.0, np.quantile(y - fb_val_preds[0.75], 0.75)))

    json.dump(meta, open(OUT / "meta.json", "w"), indent=1)
    print(f"froze {len(list(OUT.iterdir()))} artifacts; "
          f"fallback conformal ±{meta['fb_cqr_Q80']:.1f} / +{meta['fb_order_corr']:.1f}")


if __name__ == "__main__":
    main()
