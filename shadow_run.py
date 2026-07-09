"""Evening shadow run: score tomorrow's plan with the official model and the
full pre-registered shadow roster, append everything to shadow_log.csv.

The Streamlit app does the same logging on every forecast; this CLI is the
scriptable/cron equivalent so the July referee never misses a day.

Usage:
  python3 shadow_run.py --plan plan.csv            # or .xlsx
  python3 shadow_run.py --plan plan.csv --history updated.xlsx --policy high
"""
import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "siemens_model_bundle"))

import auto_calendar  # noqa: E402
import shadow  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plan", required=True,
                        help="plan file (.csv/.xlsx): Date, Counter Name, Item Name, Category")
    parser.add_argument("--history",
                        default=str(REPO_ROOT / "Lunch_Master_Data_FINAL(cleaned).xlsx"))
    parser.add_argument("--policy", default="standard",
                        choices=list(shadow.ORDER_POLICIES))
    parser.add_argument("--log", default=str(REPO_ROOT / "shadow_log.csv"))
    args = parser.parse_args()

    history = pd.read_excel(args.history, sheet_name="Lunch Master")
    history["Date"] = pd.to_datetime(history["Date"])
    holiday_dates = auto_calendar.load_holiday_dates(args.history)
    if args.plan.lower().endswith((".xlsx", ".xls")):
        plan = pd.read_excel(args.plan)
    else:
        plan = pd.read_csv(args.plan)
    plan["Date"] = pd.to_datetime(plan["Date"])

    target, shadow_preds = shadow.score_plan(history, plan, holiday_dates,
                                             order_policy=args.policy)
    logged = shadow.log_shadow_run(target, shadow_preds, args.log)

    print(f"regime: {target['regime'].iloc[0]} · order policy: {args.policy} · "
          f"log now has {logged} counter-days\n")
    view = target[["Date", "Counter Name", "predicted", "range_low",
                   "range_high", "suggested_order", "risk"]].copy()
    view["Date"] = pd.to_datetime(view["Date"]).dt.date
    for name, preds in shadow_preds.items():
        view[f"shadow_{name}"] = preds.round(0)
    print(view.to_string(index=False))


if __name__ == "__main__":
    main()
