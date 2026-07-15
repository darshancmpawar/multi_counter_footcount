"""Fast unit tests for the scoring path's decision logic (no workbook I/O).
Covers the defects found in the August 2026 audit: holiday-aware lag-regime
routing, the debias gate's on/off/clip behavior, and festival-flag derivation.
Run: pytest tests/ -q
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "siemens_model_bundle"))

from auto_calendar import HolidayCalendar, derive_day_type, festival_flags  # noqa: E402
from shadow import DEBIAS, detect_lag_regime, trailing_debias_factor  # noqa: E402


def _hist(day):
    return pd.DataFrame({"Date": [pd.Timestamp(day)]})


def _plan(day):
    return pd.DataFrame({"Date": [pd.Timestamp(day)]})


# --------------------------------------------------------- lag regime -------
def test_next_working_day_is_fresh():
    assert detect_lag_regime(_hist("2025-08-25"), _plan("2025-08-26"), set()) == "fresh"


def test_over_weekend_is_fresh():
    # Fri -> Mon
    assert detect_lag_regime(_hist("2025-08-22"), _plan("2025-08-25"), set()) == "fresh"


def test_day_after_midweek_holiday_is_fresh():
    # Tue served, Wed holiday, plan Thu — lag-1 actuals are current
    calendar = HolidayCalendar(closed={date(2025, 8, 27)})
    assert detect_lag_regime(_hist("2025-08-26"), _plan("2025-08-28"), calendar) == "fresh"


def test_two_working_day_gap_is_stale():
    assert detect_lag_regime(_hist("2025-08-26"), _plan("2025-08-28"), set()) == "stale"


# --------------------------------------------------------- debias gate ------
def _counter_days(n_days=12):
    dates = pd.bdate_range("2026-01-05", periods=n_days)
    return pd.DataFrame([{"Date": d, "Counter Name": c, "cc": 100.0}
                         for d in dates for c in ("A", "B", "C")])


def test_debias_fires_on_consistent_overprediction():
    cd = _counter_days()
    factor = trailing_debias_factor(cd, cd["Date"].max(),
                                    lambda rows: rows["cc"].values * 1.10)
    assert abs(factor - 1 / 1.10) < 0.01


def test_debias_dormant_on_balanced_errors():
    cd = _counter_days()
    order = {d: i for i, d in enumerate(sorted(cd["Date"].unique()))}

    def alternating(rows):
        parity = rows["Date"].map(order).to_numpy() % 2
        return rows["cc"].values * np.where(parity == 0, 1.10, 0.90)

    assert trailing_debias_factor(cd, cd["Date"].max(), alternating) == 1.0


def test_debias_clipped_at_bound():
    cd = _counter_days()
    factor = trailing_debias_factor(cd, cd["Date"].max(),
                                    lambda rows: rows["cc"].values * 2.0)
    assert factor == DEBIAS["clip"][0]


def test_debias_needs_min_history():
    cd = _counter_days(n_days=3)
    factor = trailing_debias_factor(cd, cd["Date"].max(),
                                    lambda rows: rows["cc"].values * 1.5)
    assert factor == 1.0


# ------------------------------------------------------- festival flags -----
def test_adjacent_important_holiday_flagged():
    calendar = HolidayCalendar(closed={date(2026, 3, 19)},
                               holiday_types={date(2026, 3, 19): "Important"})
    flags = festival_flags("2026-03-18", calendar)   # Wed before Thu holiday
    assert flags["adj_hol_important"] == 1 and flags["fest_any"] == 1


def test_operated_festival_day_flagged():
    calendar = HolidayCalendar(operated={date(2025, 9, 5)})
    flags = festival_flags("2025-09-05", calendar)   # Onam, facility open
    assert flags["fest_op_today"] == 1 and flags["adj_hol_important"] == 0


def test_plain_day_has_no_flags():
    flags = festival_flags("2026-02-10", HolidayCalendar())
    assert flags["fest_any"] == 0


# ----------------------------------------------------------- day type -------
def test_day_before_midweek_closure_is_holiday_adjacent():
    calendar = HolidayCalendar(closed={date(2026, 3, 19)})
    assert derive_day_type("2026-03-18", calendar) == "Previous Day of Holiday"


def test_regular_day_stays_regular():
    assert derive_day_type("2026-02-10", HolidayCalendar()) == "Regular"
