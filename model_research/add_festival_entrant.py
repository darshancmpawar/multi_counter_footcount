"""Pre-register the FESTIVAL shadow entrant (July 2026 round).

Adds two seed-averaged LightGBM sets to siemens_model_bundle/artifacts_shadow/
WITHOUT touching the frozen official model or the existing pre-registered
roster (challenger4 / ExtraTrees / KNN / lag-2 fallbacks):

  festival_s{seed}.txt   incumbent architecture + 5 calendar festival flags
                         (auto_calendar.festival_flags), fresh (k=1) regime
  fest_fb_s{seed}.txt    same, lag-2 fallback (k=2) regime

Same protocol as retrain.train_shadow_set: early-stop each seed on the
trailing-2-month validation window, refit on the FULL current history at that
iteration count. Trained on history through the workbook's last served day
(9 Jul 2026); scored on genuinely future days (10 Jul onward, and the August
festival season when the flags actually activate). meta.json gains
festival_features + the new iteration counts; everything else is preserved.

Why this entrant: the Festival column is 21/23 redundant with the existing
holiday-adjacency flag, but the *Holiday Type* it implies (Shutdown vs
Important vs Compulsory) splits one flag into a severity gradient the model
otherwise can't see. Offline evidence (model_research/FINDINGS.md, this
round): CV-neutral over stable Jan-May folds, +1.26pt on Jul 1-9 with a 95%
bootstrap CI excluding zero. Adjudication follows the standard rule —
adopt for a future retrain only if the shadow month's CI excludes zero.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
BUNDLE = REPO / "siemens_model_bundle"
SHADOW = BUNDLE / "artifacts_shadow"
sys.path.insert(0, str(BUNDLE))
sys.path.insert(0, str(REPO))

import auto_calendar  # noqa: E402
from evaluate import wape  # noqa: E402
from shadow import FESTIVAL_FEATURES, build_cd_k, freeze_headword_map  # noqa: E402
from retrain import ModelTrainer, INCUMBENT_PARAMS, SEEDS  # noqa: E402


def main():
    workbook = REPO / "Lunch_Master_Data_FINAL(cleaned).xlsx"
    history = pd.read_excel(workbook, sheet_name="Lunch Master")
    history["Date"] = pd.to_datetime(history["Date"])
    holidays = auto_calendar.load_holiday_dates(workbook)
    as_of = history["Date"].max()
    val_start = as_of - pd.DateOffset(months=2) + pd.Timedelta(days=1)
    subcat_map = freeze_headword_map(history)
    print(f"history through {as_of.date()}; early-stop val window from {val_start.date()}")

    cd_fresh = build_cd_k(history, k=1, holiday_dates=holidays, subcat_map=subcat_map)
    cd_stale = build_cd_k(history, k=2, holiday_dates=holidays, subcat_map=subcat_map)
    fresh = ModelTrainer(cd_fresh, val_start)
    stale = ModelTrainer(cd_stale, val_start)

    meta = json.load(open(SHADOW / "meta.json"))
    meta["festival_features"] = FESTIVAL_FEATURES
    meta["festival_built_on"] = str(as_of.date())
    meta.setdefault("n_iters", {})

    print("training festival entrant (fresh k=1, 3-seed)…")
    fresh_models, fresh_val, fresh_iters = fresh.fit_seed_averaged(
        INCUMBENT_PARAMS, FESTIVAL_FEATURES)
    for seed, booster in zip(SEEDS, fresh_models):
        booster.save_model(str(SHADOW / f"festival_s{seed}.txt"))
        meta["n_iters"][f"festival_s{seed}"] = fresh_iters[seed]

    print("training festival fallback (stale k=2, 3-seed)…")
    stale_models, _, stale_iters = stale.fit_seed_averaged(
        INCUMBENT_PARAMS, FESTIVAL_FEATURES)
    for seed, booster in zip(SEEDS, stale_models):
        booster.save_model(str(SHADOW / f"fest_fb_s{seed}.txt"))
        meta["n_iters"][f"fest_fb_s{seed}"] = stale_iters[seed]

    # sanity: a matched incumbent-feature baseline on the SAME split, so the
    # report isolates the feature effect from the extra-data effect
    base_models, base_val, _ = fresh.fit_seed_averaged(INCUMBENT_PARAMS, ())
    festival_wape = wape(fresh.val["cc"], fresh_val)
    base_wape = wape(fresh.val["cc"], base_val)
    meta["festival_val_wape"] = round(festival_wape, 3)
    meta["festival_base_val_wape"] = round(base_wape, 3)

    with open(SHADOW / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)

    print(f"\nsame-split validation ({val_start.date()}..{as_of.date()}, "
          f"n={len(fresh.val)}):")
    print(f"  incumbent features : {base_wape:.2f}% WAPE")
    print(f"  + festival features: {festival_wape:.2f}% WAPE  "
          f"(delta {festival_wape - base_wape:+.2f})")
    print(f"\nwrote festival_s* / fest_fb_s* into {SHADOW.relative_to(REPO)}")
    print("frozen official model and existing roster untouched.")


if __name__ == "__main__":
    main()
