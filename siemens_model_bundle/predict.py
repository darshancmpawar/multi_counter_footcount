"""Siemens lunch counter demand — CLI scorer over the PRODUCTION pipeline.

Thin wrapper around shadow.score_plan, so the CLI produces byte-identical
official numbers to the Streamlit app: same lag-regime routing (lag-2
fallback when yesterday's actuals are missing), same auto-derived Day Type /
Panchangam, same level-shift corrector, same order policies. (The previous
version of this script was an independent re-implementation missing all of
those — it quoted ~9% higher orders than the app on the same plan.)

Usage:
    python predict.py --plan plan.csv
    python predict.py --plan plan.xlsx --history updated.xlsx --policy high

plan file — one row per planned ITEM, columns:
    Date (yyyy-mm-dd), Counter Name, Item Name, Category,
    optional: Day Type, Panchangam (auto-derived from the date when absent)
Only ACTIVE counters appear in the plan. Closed counters are simply absent.

Outputs per active counter: predicted consumed, calibrated P10–P90 range,
suggested order, risk level. Writes predictions_out.csv.
"""
import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
BUNDLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BUNDLE_DIR.parent
sys.path.insert(0, str(BUNDLE_DIR))

import auto_calendar  # noqa: E402
import shadow  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plan", required=True,
                        help="plan file (.csv/.xlsx): Date, Counter Name, Item Name, Category")
    parser.add_argument("--history",
                        default=str(REPO_ROOT / "Lunch_Master_Data_FINAL(cleaned).xlsx"))
    parser.add_argument("--policy", default="standard",
                        choices=list(shadow.ORDER_POLICIES))
    parser.add_argument("--out", default="predictions_out.csv")
    args = parser.parse_args()

    history = pd.read_excel(args.history, sheet_name="Lunch Master")
    history["Date"] = pd.to_datetime(history["Date"])
    holiday_dates = auto_calendar.load_holiday_dates(args.history)
    if args.plan.lower().endswith((".xlsx", ".xls")):
        plan = pd.read_excel(args.plan)
    else:
        plan = pd.read_csv(args.plan)
    plan["Date"] = pd.to_datetime(plan["Date"])

    target, _ = shadow.score_plan(history, plan, holiday_dates,
                                  order_policy=args.policy)

    factor = target["debias_factor"].iloc[0]
    print("=" * 90)
    print(f"regime: {target['regime'].iloc[0]} · order policy: {args.policy}"
          + (f" · level-shift correction x{factor:.3f}" if abs(factor - 1) > 1e-9 else ""))
    for date, day in target.groupby("Date"):
        total = day["predicted"].sum()
        print(f"\nPREDICTION — {pd.Timestamp(date).date()} "
              f"({day['weekday'].iloc[0]}) | active counters: {len(day)}")
        print(f"Predicted Total Lunch Consumed: {int(total)}")
        for _, row in day.sort_values("predicted", ascending=False).iterrows():
            print(f"\n  {row['Counter Name']}: predicted {int(row['predicted'])} "
                  f"(share {100 * row['predicted'] / total:.0f}%) "
                  f"| range {int(row['range_low'])}-{int(row['range_high'])} "
                  f"| suggested order {int(row['suggested_order'])} "
                  f"| risk {row['risk']}")
    print("\n" + "=" * 90)
    out = target[["Date", "Counter Name", "predicted", "range_low",
                  "range_high", "suggested_order", "risk", "regime",
                  "debias_factor"]]
    out.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
