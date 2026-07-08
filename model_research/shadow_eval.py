"""Month-end shadow referee: join shadow_log.csv with actuals from the
updated history workbook and score every logged model — counter and day
WAPE, paired bootstrap vs the official model, and the kitchen's own ordering
as the business benchmark.

Usage:  python3 shadow_eval.py [--history <xlsx>] [--log <csv>]

Adoption rule: a challenger becomes next month's model only if its 95% CI
vs the official model excludes zero on the full month (~90 counter-days).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "siemens_model_bundle"))


def wape(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return 100 * np.abs(predicted - actual).sum() / actual.sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history",
                        default=str(REPO_ROOT / "Lunch_Master_Data_FINAL(cleaned).xlsx"))
    parser.add_argument("--log", default=str(REPO_ROOT / "shadow_log.csv"))
    args = parser.parse_args()

    log = pd.read_csv(args.log, parse_dates=["Date"])
    history = pd.read_excel(args.history, sheet_name="Lunch Master")
    history["Date"] = pd.to_datetime(history["Date"])
    actuals = (history.groupby(["Date", "Counter Name"])["Counter Consumed"]
               .first().rename("actual").reset_index())
    scored = log.merge(actuals, on=["Date", "Counter Name"], how="inner")
    if scored.empty:
        print("No logged predictions have actuals yet — update the history "
              "workbook through the forecasted dates and re-run.")
        return

    model_columns = [c for c in scored.columns if c.startswith("pred_")]
    print(f"{len(scored)} counter-days with actuals "
          f"({scored['Date'].min():%d %b} → {scored['Date'].max():%d %b}), "
          f"regimes: {dict(scored['regime'].value_counts())}\n")
    print(f"{'model':16s} {'counter WAPE%':>13s} {'day WAPE%':>10s} {'MAE':>6s} {'bias':>6s}")
    for column in model_columns:
        rows = scored.dropna(subset=[column])
        if rows.empty:
            continue
        daily = rows.groupby("Date").agg(a=("actual", "sum"), p=(column, "sum"))
        print(f"{column[5:]:16s} {wape(rows['actual'], rows[column]):13.2f} "
              f"{wape(daily['a'], daily['p']):10.2f} "
              f"{np.abs(rows[column] - rows['actual']).mean():6.1f} "
              f"{(rows[column] - rows['actual']).mean():+6.1f}")

    from evaluate import kitchen_benchmark
    month = f"{scored['Date'].dt.year.iloc[0]}-{scored['Date'].dt.month.iloc[0]:02d}"
    print(f"\nbusiness benchmark — {kitchen_benchmark(history, month).round(2).to_dict()}")

    # paired bootstrap of every challenger vs the official model
    rng = np.random.default_rng(0)
    y = None
    for column in model_columns:
        if column == "pred_official":
            continue
        rows = scored.dropna(subset=[column, "pred_official"])
        if rows.empty:
            continue
        y = rows["actual"].values
        challenger, official = rows[column].values, rows["pred_official"].values
        diffs = np.array([wape(y[i], challenger[i]) - wape(y[i], official[i])
                          for i in (rng.integers(0, len(y), len(y))
                                    for _ in range(10000))])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        verdict = ("ADOPT" if hi < 0 else "keep official" if lo > 0 else "ambiguous")
        print(f"{column[5:]:16s} vs official: {wape(y, challenger) - wape(y, official):+5.2f} "
              f"CI [{lo:+5.2f}, {hi:+5.2f}] → {verdict}")


if __name__ == "__main__":
    main()
