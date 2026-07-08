"""Validate auto_calendar against every recorded Day Type and Panchangam
label in the history workbook. Run after touching auto_calendar.py or when a
new month of labelled history arrives.

Expected: Day Type 203/203 · observances 83/84 (one Pradosham the panchang
records a day earlier than the astronomy; the one Navratri 'extra' is Rama
Navami day, astronomically Navratri day 9).
"""
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "siemens_model_bundle"))
import auto_calendar  # noqa: E402

WORKBOOK = REPO_ROOT / "Lunch_Master_Data_FINAL(cleaned).xlsx"
PANCHANGAM_FLAGS = ["Shukla Ekadashi", "Krishna Ekadashi", "Poornima",
                    "Amavasya", "Pradosham", "Sankashti", "Shivaratri",
                    "Shravan Month", "Navratri"]


def main() -> None:
    history = pd.read_excel(WORKBOOK, sheet_name="Lunch Master")
    history["Date"] = pd.to_datetime(history["Date"])
    recorded = history.groupby("Date").agg(
        day_type=("Day Type", "first"),
        panchangam=("Panchangam", "first")).sort_index()
    holiday_dates = auto_calendar.load_holiday_dates(WORKBOOK)
    print(f"{len(recorded)} working days · {len(holiday_dates)} listed holidays")

    derived_day_type = pd.Series(
        {d: auto_calendar.derive_day_type(d, holiday_dates) for d in recorded.index})
    matches = int((derived_day_type == recorded["day_type"]).sum())
    print(f"\nDay Type: {matches}/{len(recorded)} exact")
    for date in recorded.index[derived_day_type != recorded["day_type"]]:
        print(f"  mismatch {date.date()}: recorded {recorded.loc[date, 'day_type']}"
              f" / derived {derived_day_type[date]}")

    derived_panchangam = pd.Series(
        {d: auto_calendar.derive_panchangam(d) for d in recorded.index})
    print("\nPanchangam flags (recorded | hits / misses / extras):")
    for flag in PANCHANGAM_FLAGS:
        actual = recorded["panchangam"].str.contains(flag, case=False)
        derived = derived_panchangam.str.contains(flag, case=False)
        print(f"  {flag:18s} {int(actual.sum()):2d} | {int((actual & derived).sum()):2d}"
              f" / {int((actual & ~derived).sum()):2d}"
              f" / {int((~actual & derived).sum()):2d}")


if __name__ == "__main__":
    main()
