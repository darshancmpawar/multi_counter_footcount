"""Train and freeze the shadow/fallback model set into
siemens_model_bundle/artifacts_shadow/.

Contents:
  challenger_s{seed}.txt      3-seed LGB challenger (lag-1 features + extras)
  blend_et.joblib             ExtraTrees member of the 70/30 blend (shadow only)
  fb_point_s{seed}.txt        lag-2 FALLBACK point models (incumbent config)
  fb_q10/q75/q90.txt          lag-2 fallback quantile models + conformal meta
  chal_fb_s{seed}.txt         lag-2 fallback of the challenger (shadow in stale mode)
  meta.json                   feature lists, configs, iterations, conformal corr.

Protocol: n_iters from early stopping on Apr-May validation (train <= Mar),
then retrain on the full history at that iteration count. Conformal
corrections for the fallback quantiles are fitted on Apr-May only.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor

BUNDLE = Path(__file__).resolve().parent.parent / "siemens_model_bundle"
sys.path.insert(0, str(BUNDLE))
from shadow import (EXTRA_FEATURES, CHALLENGER_PARAMS, INCUMBENT_PARAMS,  # noqa: E402
                    build_cd_k, design_lgb, design_sk)
from features import NUM_FEATURES  # noqa: E402

HIST = Path(__file__).resolve().parent.parent / "Lunch_Master_Data_FINAL(cleaned).xlsx"
OUT = BUNDLE / "artifacts_shadow"
OUT.mkdir(exist_ok=True)
TRAIN_END, VAL_END = pd.Timestamp("2026-03-31"), pd.Timestamp("2026-05-31")
SEEDS = (1, 42, 99)


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
    d = lgb.train(lgb_params(params, objective, seed, alpha),
                  lgb.Dataset(design_lgb(tr, feats), tr["cc"]), 3000,
                  valid_sets=[lgb.Dataset(design_lgb(va, feats), va["cc"])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    n_it = d.best_iteration
    full = cd  # retrain through the last served day
    bst = lgb.train(lgb_params(params, objective, seed, alpha),
                    lgb.Dataset(design_lgb(full, feats), full["cc"]), n_it)
    return bst, n_it


def main():
    hist = pd.read_excel(HIST, sheet_name="Lunch Master")
    hist["Date"] = pd.to_datetime(hist["Date"])
    cd1, cd2 = build_cd_k(hist, k=1), build_cd_k(hist, k=2)
    meta = {"built_on": str(hist["Date"].max().date()),
            "extra_features": EXTRA_FEATURES,
            "challenger_params": CHALLENGER_PARAMS,
            "incumbent_params": INCUMBENT_PARAMS,
            "seeds": list(SEEDS), "n_iters": {}}

    # challenger (fresh, k=1) + its stale fallback (k=2)
    for seed in SEEDS:
        b, n = train_frozen(cd1, CHALLENGER_PARAMS, EXTRA_FEATURES, seed=seed)
        b.save_model(str(OUT / f"challenger_s{seed}.txt"))
        meta["n_iters"][f"challenger_s{seed}"] = n
        b, n = train_frozen(cd2, CHALLENGER_PARAMS, EXTRA_FEATURES, seed=seed)
        b.save_model(str(OUT / f"chal_fb_s{seed}.txt"))
        meta["n_iters"][f"chal_fb_s{seed}"] = n

    # ExtraTrees blend member (shadow, fresh regime only)
    Xf, med, cols = design_sk(cd1)
    et = ExtraTreesRegressor(500, min_samples_leaf=5, max_features=0.4,
                             n_jobs=-1, random_state=42).fit(Xf, cd1["cc"])
    joblib.dump({"model": et, "medians": med, "columns": list(cols)}, OUT / "blend_et.joblib")

    # lag-2 fallback for the OFFICIAL numbers (incumbent architecture)
    for seed in SEEDS:
        b, n = train_frozen(cd2, INCUMBENT_PARAMS, (), seed=seed)
        b.save_model(str(OUT / f"fb_point_s{seed}.txt"))
        meta["n_iters"][f"fb_point_s{seed}"] = n
    for q in (0.10, 0.75, 0.90):
        b, n = train_frozen(cd2, INCUMBENT_PARAMS, (), objective="quantile",
                            seed=42, alpha=q)
        b.save_model(str(OUT / f"fb_q{int(q*100)}.txt"))
        meta["n_iters"][f"fb_q{int(q*100)}"] = n

    # conformal corrections for the fallback quantiles, fitted on Apr-May
    va = cd2[(cd2["Date"] > TRAIN_END) & (cd2["Date"] <= VAL_END)]
    tr = cd2[cd2["Date"] <= TRAIN_END]

    def val_model(objective, alpha=None):
        b = lgb.train(lgb_params(INCUMBENT_PARAMS, objective, 42, alpha),
                      lgb.Dataset(design_lgb(tr, ()), tr["cc"]), 3000,
                      valid_sets=[lgb.Dataset(design_lgb(va, ()), va["cc"])],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        return np.clip(b.predict(design_lgb(va, ())), 0, None)

    q10v = val_model("quantile", 0.10)
    q90v = val_model("quantile", 0.90)
    q75v = val_model("quantile", 0.75)
    y = va["cc"].values
    conf = np.maximum(q10v - y, y - q90v)
    meta["fb_cqr_Q80"] = float(np.quantile(conf, 0.80))
    short = y - q75v
    meta["fb_order_corr"] = float(max(0.0, np.quantile(short, 0.75)))
    json.dump(meta, open(OUT / "meta.json", "w"), indent=1)
    print("saved", len(list(OUT.iterdir())), "artifacts to", OUT)
    print("fallback cqr_Q80:", round(meta["fb_cqr_Q80"], 1),
          "order_corr:", round(meta["fb_order_corr"], 1))


if __name__ == "__main__":
    main()
