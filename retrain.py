"""Monthly retraining pipeline for the lunch-counter demand models.

Retrains the complete model set from an updated history workbook using the
same protocol the frozen bundle was built with:

  1. Validation window = the trailing `--val-months` months of data;
     everything before it is the tuning-train split.
  2. Each model early-stops on the validation window to pick its iteration
     count, then retrains on ALL data at that count (no data is wasted).
  3. Conformal corrections (interval widening, order-quantile shift) are
     fitted on the validation predictions of the train-only models —
     genuinely out-of-sample.
  4. A validation scoreboard (new vs currently deployed vs seasonal baseline)
     is written to model_research/retrain_reports/.

Artifacts land in a versioned directory and are NEVER deployed automatically:
review the report first, then re-run with --promote (the current deployment
is backed up before being replaced).

Usage:
  python3 retrain.py                          # train + report, no deploy
  python3 retrain.py --promote                # ...and deploy after review
  python3 retrain.py --history new_data.xlsx --val-months 2
"""
import argparse
import json
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

REPO_ROOT = Path(__file__).resolve().parent
BUNDLE_DIR = REPO_ROOT / "siemens_model_bundle"
sys.path.insert(0, str(BUNDLE_DIR))

from shadow import (BLEND_W, CHALLENGER_PARAMS, EXTRA_FEATURES,  # noqa: E402
                    INCUMBENT_PARAMS, build_cd_k, design_lgb, design_sk)

REPORT_DIR = REPO_ROOT / "model_research" / "retrain_reports"
QUANTILES = (0.10, 0.75, 0.90)
SEEDS = (1, 42, 99)
INTERVAL_COVERAGE = 0.80    # nominal coverage of the P10-P90 band
ORDER_SERVICE_LEVEL = 0.75  # service level of the suggested-order quantile


def wape(actual, predicted) -> float:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return float(100 * np.abs(predicted - actual).sum() / actual.sum())


def lgb_params(config: dict, objective: str = "poisson", seed: int = 42,
               quantile_alpha: float | None = None) -> dict:
    params = dict(objective=objective, learning_rate=config.get("lr", 0.03),
                  num_leaves=config.get("nl", 7),
                  min_child_samples=config.get("mcs", 10),
                  feature_fraction=config.get("ff", 0.6),
                  lambda_l2=config.get("rl", 1), lambda_l1=config.get("l1", 0.0),
                  bagging_fraction=config.get("bf", 1.0),
                  bagging_freq=config.get("bfreq", 0),
                  verbose=-1, seed=seed, feature_fraction_seed=seed,
                  bagging_seed=seed)
    if quantile_alpha is not None:
        params["alpha"] = quantile_alpha
    return params


class ModelTrainer:
    """Early-stop on the validation window, retrain on everything."""

    def __init__(self, counter_days: pd.DataFrame, val_start: pd.Timestamp):
        self.full = counter_days
        self.train = counter_days[counter_days["Date"] < val_start]
        self.val = counter_days[counter_days["Date"] >= val_start]

    def fit(self, config: dict, features=(), objective: str = "poisson",
            seed: int = 42, quantile_alpha: float | None = None):
        """Returns (final booster trained on all data, validation predictions
        of the train-only booster, chosen iteration count)."""
        params = lgb_params(config, objective, seed, quantile_alpha)
        train_set = lgb.Dataset(design_lgb(self.train, features), self.train["cc"])
        val_set = lgb.Dataset(design_lgb(self.val, features), self.val["cc"])
        probe = lgb.train(params, train_set, 3000, valid_sets=[val_set],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
        val_predictions = np.clip(probe.predict(design_lgb(self.val, features)), 0, None)
        final = lgb.train(params,
                          lgb.Dataset(design_lgb(self.full, features), self.full["cc"]),
                          probe.best_iteration)
        return final, val_predictions, probe.best_iteration

    def fit_seed_averaged(self, config: dict, features=()):
        boosters, val_predictions = [], []
        iteration_counts = {}
        for seed in SEEDS:
            booster, val_pred, n_iters = self.fit(config, features, seed=seed)
            boosters.append(booster)
            val_predictions.append(val_pred)
            iteration_counts[seed] = n_iters
        return boosters, np.mean(val_predictions, axis=0), iteration_counts


def conformal_corrections(actual, q10_pred, q75_pred, q90_pred) -> tuple[float, float]:
    """CQR widening for the P10-P90 band and the one-sided shift that lifts
    the order quantile to the target service level."""
    conformity = np.maximum(q10_pred - actual, actual - q90_pred)
    cqr_width = float(np.quantile(conformity, INTERVAL_COVERAGE))
    shortfall = actual - q75_pred
    order_shift = float(max(0.0, np.quantile(shortfall, ORDER_SERVICE_LEVEL)))
    return cqr_width, order_shift


def train_official_set(trainer: ModelTrainer, output_dir: Path) -> dict:
    """Incumbent-architecture point + quantile models + conformal config,
    saved with the exact artifact names predict.py and the app load."""
    point, point_val, point_iters = trainer.fit(INCUMBENT_PARAMS)
    point.save_model(str(output_dir / "model_point.txt"))

    quantile_val, quantile_iters = {}, {}
    for alpha in QUANTILES:
        booster, val_pred, n_iters = trainer.fit(INCUMBENT_PARAMS,
                                                 objective="quantile",
                                                 quantile_alpha=alpha)
        booster.save_model(str(output_dir / f"model_q{int(alpha * 100)}.txt"))
        quantile_val[alpha] = val_pred
        quantile_iters[alpha] = n_iters

    cqr_width, order_shift = conformal_corrections(
        trainer.val["cc"].values, quantile_val[0.10], quantile_val[0.75],
        quantile_val[0.90])
    config = {"params": INCUMBENT_PARAMS, "n_iters": point_iters,
              "q_iters": quantile_iters, "cqr_Q80": cqr_width,
              "order_corr": order_shift}
    with open(output_dir / "final_config.pkl", "wb") as f:
        pickle.dump(config, f)
    return {"val_wape": wape(trainer.val["cc"], point_val),
            "cqr_Q80": cqr_width, "order_corr": order_shift}


def train_shadow_set(fresh: ModelTrainer, stale: ModelTrainer,
                     output_dir: Path) -> dict:
    """Challenger (3 seeds), ExtraTrees blend member, and the lag-2 fallback
    set, saved with the artifact names shadow.load_shadow expects."""
    from sklearn.ensemble import ExtraTreesRegressor

    meta = {"built_on": str(fresh.full["Date"].max().date()),
            "extra_features": EXTRA_FEATURES,
            "challenger_params": CHALLENGER_PARAMS,
            "incumbent_params": INCUMBENT_PARAMS,
            "seeds": list(SEEDS), "n_iters": {}}
    scores = {}

    challengers, challenger_val, iters = fresh.fit_seed_averaged(
        CHALLENGER_PARAMS, EXTRA_FEATURES)
    for seed, booster in zip(SEEDS, challengers):
        booster.save_model(str(output_dir / f"challenger_s{seed}.txt"))
        meta["n_iters"][f"challenger_s{seed}"] = iters[seed]
    scores["challenger_val_wape"] = wape(fresh.val["cc"], challenger_val)

    features_full, medians, columns = design_sk(fresh.full)
    features_train, _, _ = design_sk(fresh.train, medians, columns)
    features_val, _, _ = design_sk(fresh.val, medians, columns)
    tree_config = dict(n_estimators=500, min_samples_leaf=5, max_features=0.4,
                       n_jobs=-1, random_state=42)
    blend_probe = ExtraTreesRegressor(**tree_config).fit(features_train,
                                                         fresh.train["cc"])
    blend_val = (BLEND_W * challenger_val
                 + (1 - BLEND_W) * np.clip(blend_probe.predict(features_val), 0, None))
    scores["blend_val_wape"] = wape(fresh.val["cc"], blend_val)
    blend_final = ExtraTreesRegressor(**tree_config).fit(features_full,
                                                         fresh.full["cc"])
    joblib.dump({"model": blend_final, "medians": medians,
                 "columns": list(columns)}, output_dir / "blend_et.joblib")

    fallback_challengers, _, fb_iters = stale.fit_seed_averaged(
        CHALLENGER_PARAMS, EXTRA_FEATURES)
    for seed, booster in zip(SEEDS, fallback_challengers):
        booster.save_model(str(output_dir / f"chal_fb_s{seed}.txt"))
        meta["n_iters"][f"chal_fb_s{seed}"] = fb_iters[seed]

    fallback_points, fallback_val, fp_iters = stale.fit_seed_averaged(INCUMBENT_PARAMS)
    for seed, booster in zip(SEEDS, fallback_points):
        booster.save_model(str(output_dir / f"fb_point_s{seed}.txt"))
        meta["n_iters"][f"fb_point_s{seed}"] = fp_iters[seed]
    scores["fallback_val_wape"] = wape(stale.val["cc"], fallback_val)

    fallback_quantile_val = {}
    for alpha in QUANTILES:
        booster, val_pred, n_iters = stale.fit(INCUMBENT_PARAMS,
                                               objective="quantile",
                                               quantile_alpha=alpha)
        booster.save_model(str(output_dir / f"fb_q{int(alpha * 100)}.txt"))
        fallback_quantile_val[alpha] = val_pred
        meta["n_iters"][f"fb_q{int(alpha * 100)}"] = n_iters
    meta["fb_cqr_Q80"], meta["fb_order_corr"] = conformal_corrections(
        stale.val["cc"].values, fallback_quantile_val[0.10],
        fallback_quantile_val[0.75], fallback_quantile_val[0.90])

    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    return scores


def deployed_val_wape(counter_days: pd.DataFrame, val_start: pd.Timestamp) -> float | None:
    """Score the currently deployed point model on the new validation window
    (its training data overlaps this window, so treat as optimistic context)."""
    deployed_path = BUNDLE_DIR / "artifacts" / "model_point.txt"
    if not deployed_path.exists():
        return None
    validation = counter_days[counter_days["Date"] >= val_start]
    booster = lgb.Booster(model_file=str(deployed_path))
    predictions = np.clip(booster.predict(design_lgb(validation)), 0, None)
    return wape(validation["cc"], predictions)


def write_report(path: Path, context: dict) -> None:
    lines = [
        f"# Retrain report — {context['run_stamp']}",
        "",
        f"History: {context['history_file']} · rows through **{context['as_of']}**",
        f"Splits: train < {context['val_start']} ({context['train_rows']} rows) · "
        f"validation ≥ {context['val_start']} ({context['val_rows']} rows)",
        "",
        "| Model (validation WAPE %) | New |",
        "|---|---|",
        f"| Official point (incumbent architecture) | {context['official']['val_wape']:.2f} |",
        f"| Challenger (3-seed) | {context['shadow']['challenger_val_wape']:.2f} |",
        f"| Blend (0.7 challenger + 0.3 ExtraTrees) | {context['shadow']['blend_val_wape']:.2f} |",
        f"| Lag-2 fallback point | {context['shadow']['fallback_val_wape']:.2f} |",
        f"| Seasonal baseline (wd_roll4) | {context['baseline_val_wape']:.2f} |",
        "",
        f"Currently deployed model on the same window: "
        f"{context['deployed_val_wape']:.2f} (optimistic — its training data "
        f"overlaps this window)" if context["deployed_val_wape"] is not None else
        "No deployed model found to compare against.",
        "",
        f"Conformal: interval widening ±{context['official']['cqr_Q80']:.1f} pax, "
        f"order shift +{context['official']['order_corr']:.1f} pax "
        f"(fallback: ±{context['shadow_meta_cqr']:.1f} / +{context['shadow_meta_order']:.1f}).",
        "",
        f"Artifacts: `{context['output_dir']}`",
        "Deploy with:  `python3 retrain.py --promote`  (backs up the current "
        "deployment first).",
    ]
    path.write_text("\n".join(lines))


def promote(versioned_dir: Path) -> None:
    """Deploy a versioned retrain: back up, then replace artifacts/ and
    artifacts_shadow/ with the new set."""
    backup_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BUNDLE_DIR / f"artifacts_backup_{backup_stamp}"
    backup_dir.mkdir()
    for name in ("artifacts", "artifacts_shadow"):
        source = BUNDLE_DIR / name
        if source.exists():
            shutil.copytree(source, backup_dir / name)

    official_names = ["model_point.txt", "model_q10.txt", "model_q75.txt",
                      "model_q90.txt", "final_config.pkl"]
    for name in official_names:
        shutil.copy2(versioned_dir / name, BUNDLE_DIR / "artifacts" / name)
    shadow_dir = BUNDLE_DIR / "artifacts_shadow"
    shadow_dir.mkdir(exist_ok=True)
    for artifact in versioned_dir.iterdir():
        if artifact.name not in official_names:
            shutil.copy2(artifact, shadow_dir / artifact.name)
    print(f"Promoted {versioned_dir.name}; previous deployment backed up to "
          f"{backup_dir.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--history",
                        default=str(REPO_ROOT / "Lunch_Master_Data_FINAL(cleaned).xlsx"))
    parser.add_argument("--val-months", type=int, default=2,
                        help="trailing months used for early stopping, "
                             "conformal calibration and the scoreboard")
    parser.add_argument("--promote", action="store_true",
                        help="deploy the newly trained set after training")
    args = parser.parse_args()

    history = pd.read_excel(args.history, sheet_name="Lunch Master")
    history["Date"] = pd.to_datetime(history["Date"])
    as_of = history["Date"].max()
    val_start = as_of - pd.DateOffset(months=args.val_months) + pd.Timedelta(days=1)

    print(f"History through {as_of.date()}; validation window starts {val_start.date()}")
    counter_days_fresh = build_cd_k(history, k=1)
    counter_days_stale = build_cd_k(history, k=2)
    fresh = ModelTrainer(counter_days_fresh, val_start)
    stale = ModelTrainer(counter_days_stale, val_start)
    if len(fresh.val) < 30 or len(fresh.train) < 200:
        sys.exit(f"Refusing to retrain: only {len(fresh.train)} train / "
                 f"{len(fresh.val)} validation rows.")

    run_stamp = as_of.strftime("%Y%m%d")
    output_dir = BUNDLE_DIR / f"artifacts_retrain_{run_stamp}"
    output_dir.mkdir(exist_ok=True)

    print("Training official set (point + quantiles + conformal)…")
    official_scores = train_official_set(fresh, output_dir)
    print("Training shadow set (challenger, blend, lag-2 fallbacks)…")
    shadow_scores = train_shadow_set(fresh, stale, output_dir)
    with open(output_dir / "meta.json") as f:
        shadow_meta = json.load(f)

    baseline = fresh.val["wd_roll4"].fillna(fresh.val["roll7"])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"retrain_{run_stamp}.md"
    write_report(report_path, {
        "run_stamp": run_stamp, "history_file": Path(args.history).name,
        "as_of": as_of.date(), "val_start": val_start.date(),
        "train_rows": len(fresh.train), "val_rows": len(fresh.val),
        "official": official_scores, "shadow": shadow_scores,
        "baseline_val_wape": wape(fresh.val["cc"], baseline),
        "deployed_val_wape": deployed_val_wape(counter_days_fresh, val_start),
        "shadow_meta_cqr": shadow_meta["fb_cqr_Q80"],
        "shadow_meta_order": shadow_meta["fb_order_corr"],
        "output_dir": output_dir.relative_to(REPO_ROOT),
    })
    print(f"\n{report_path.read_text()}")

    if args.promote:
        promote(output_dir)
    else:
        print("\nNot deployed. Review the report, then run with --promote.")


if __name__ == "__main__":
    main()
