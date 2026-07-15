"""Leakage gate: features for any date must be identical whether or not the
future exists in the data. Runs standalone (python3 tests/test_leakage.py)
and is invoked by retrain.py before every training run.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "siemens_model_bundle"))

import auto_calendar  # noqa: E402
from features import NUM_FEATURES  # noqa: E402
from shadow import (CHALLENGER4_FEATURES, FESTIVAL_FEATURES,  # noqa: E402
                    build_cd_k, freeze_headword_map)

WORKBOOK = REPO_ROOT / "Lunch_Master_Data_FINAL(cleaned).xlsx"
ALL_FEATURES = list(NUM_FEATURES) + CHALLENGER4_FEATURES + FESTIVAL_FEATURES


def _columns_equal(a: pd.Series, b: pd.Series) -> bool:
    return bool((a.isna() == b.isna()).all()
                and np.allclose(a.astype(float).fillna(-9e9),
                                b.astype(float).fillna(-9e9)))


def run_leakage_test(history: pd.DataFrame | None = None, verbose: bool = True) -> bool:
    """True iff every feature is truncation-invariant at interior cut points
    for both lag regimes."""
    if history is None:
        history = pd.read_excel(WORKBOOK, sheet_name="Lunch Master")
        history["Date"] = pd.to_datetime(history["Date"])
    holiday_dates = auto_calendar.load_holiday_dates(WORKBOOK)
    subcat_map = freeze_headword_map(history)

    dates = sorted(history["Date"].unique())
    cuts = [dates[len(dates) // 3], dates[2 * len(dates) // 3]]
    passed = True
    for k in (1, 2):
        full = (build_cd_k(history, k=k, holiday_dates=holiday_dates,
                           subcat_map=subcat_map)
                .set_index(["Date", "Counter Name"]).sort_index())
        for cut in cuts:
            truncated = (build_cd_k(history[history["Date"] < cut], k=k,
                                    holiday_dates=holiday_dates,
                                    subcat_map=subcat_map)
                         .set_index(["Date", "Counter Name"]).sort_index())
            visible = full[full.index.get_level_values("Date") < cut]
            leaking = [c for c in ALL_FEATURES
                       if not _columns_equal(visible[c], truncated[c])]
            status = "clean" if not leaking else f"LEAK: {leaking}"
            if verbose:
                print(f"  k={k} cut={pd.Timestamp(cut).date()}: {status}")
            passed &= not leaking
    return passed


def test_leakage():
    """Pytest entry point — the gate itself (slow: full feature rebuilds)."""
    assert run_leakage_test(verbose=False)


if __name__ == "__main__":
    print(f"leakage gate over {len(ALL_FEATURES)} features, 2 cuts × 2 lag regimes")
    ok = run_leakage_test()
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
