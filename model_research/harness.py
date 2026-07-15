"""Shared harness for model-improvement experiments.
Anchors to the bundle's exact feature builder and metric so results are
comparable with the frozen incumbent (val 9.10 / June 6.06 WAPE)."""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "siemens_model_bundle"))

import numpy as np
import pandas as pd
import lightgbm as lgb

from evaluate import wape  # noqa: F401  (canonical metric, re-exported)
from features import build_all, NUM_FEATURES, CAT_FEATURES
from shadow import CAT_LEVELS  # single source for categorical levels

HIST = str(REPO_ROOT / "Lunch_Master_Data_FINAL(cleaned).xlsx")

TRAIN_END = pd.Timestamp("2026-03-31")
VAL_END = pd.Timestamp("2026-05-31")   # June 1-12 = locked test, touch ONCE at the end


_cd_cache = {}


def load_cd():
    if "cd" not in _cd_cache:
        hist = pd.read_excel(HIST, sheet_name="Lunch Master")
        hist["Date"] = pd.to_datetime(hist["Date"])
        _cd_cache["cd"] = build_all(hist)
    return _cd_cache["cd"]


def design(cd, extra_num=()):
    feats = list(NUM_FEATURES) + list(extra_num)
    X = cd[feats + CAT_FEATURES].copy()
    for c in CAT_FEATURES:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])
    return X


def fit_lgb(Xtr, ytr, Xva, yva, params, objective="poisson", seed=42,
            num_boost_round=3000, early=150, weights=None):
    base = dict(objective=objective, learning_rate=params.get("lr", 0.03),
                num_leaves=params.get("nl", 7), min_child_samples=params.get("mcs", 10),
                feature_fraction=params.get("ff", 0.6), lambda_l2=params.get("rl", 1),
                lambda_l1=params.get("l1", 0.0),
                bagging_fraction=params.get("bf", 1.0), bagging_freq=params.get("bfreq", 0),
                min_split_gain=params.get("msg", 0.0),
                verbose=-1, seed=seed, feature_fraction_seed=seed, bagging_seed=seed)
    if objective == "tweedie":
        base["tweedie_variance_power"] = params.get("tvp", 1.2)
    if params.get("boosting"):
        base["boosting"] = params["boosting"]
    dtr = lgb.Dataset(Xtr, ytr, weight=weights)
    dva = lgb.Dataset(Xva, yva, reference=dtr)
    bst = lgb.train(base, dtr, num_boost_round=num_boost_round, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(early, verbose=False)])
    return bst


# expanding-window CV folds for model SELECTION (never touches June)
CV_FOLDS = [
    ("2025-12-31", "2026-01-01", "2026-02-28"),   # train ≤ Dec, val Jan-Feb
    ("2026-01-31", "2026-02-01", "2026-03-31"),   # train ≤ Jan, val Feb-Mar
    ("2026-02-28", "2026-03-01", "2026-04-30"),   # train ≤ Feb, val Mar-Apr
    ("2026-03-31", "2026-04-01", "2026-05-31"),   # train ≤ Mar, val Apr-May (orig val)
]


def cv_wape(cd, params, objective="poisson", seeds=(42,), extra_num=(),
            weight_fn=None, folds=CV_FOLDS):
    """Mean WAPE across expanding-window folds; multi-seed averaged predictions."""
    scores = []
    for tr_end, va_start, va_end in folds:
        tr = cd[cd["Date"] <= tr_end]
        va = cd[(cd["Date"] >= va_start) & (cd["Date"] <= va_end)]
        Xtr, Xva = design(tr, extra_num), design(va, extra_num)
        w = weight_fn(tr) if weight_fn else None
        preds = np.zeros(len(va))
        for s in seeds:
            bst = fit_lgb(Xtr, tr["cc"], Xva, va["cc"], params, objective, seed=s, weights=w)
            preds += np.clip(bst.predict(Xva), 0, None)
        preds /= len(seeds)
        scores.append(wape(va["cc"], preds))
    return float(np.mean(scores)), [round(s, 2) for s in scores]
