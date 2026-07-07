"""Month-end shadow comparison: joins shadow_log.csv with actuals from the
(updated) history workbook and scores official vs challenger vs blend.

Usage:  python3 shadow_eval.py [--history <xlsx>] [--log <csv>]

Adopt the challenger only if it wins here on a full month (~90 counter-days)
— June's 30 rows could not adjudicate (95% CI on the WAPE diff spanned zero).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def wape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return 100 * np.abs(p - y).sum() / y.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default=str(ROOT / "Lunch_Master_Data_FINAL(cleaned).xlsx"))
    ap.add_argument("--log", default=str(ROOT / "shadow_log.csv"))
    args = ap.parse_args()

    log = pd.read_csv(args.log, parse_dates=["Date"])
    hist = pd.read_excel(args.history, sheet_name="Lunch Master")
    hist["Date"] = pd.to_datetime(hist["Date"])
    act = (hist.groupby(["Date", "Counter Name"])["Counter Consumed"]
           .first().rename("actual").reset_index())
    df = log.merge(act, on=["Date", "Counter Name"], how="inner")
    if df.empty:
        print("No logged predictions have actuals yet — update the history "
              "workbook through the forecasted dates and re-run.")
        return
    print(f"{len(df)} counter-days with actuals "
          f"({df['Date'].min():%d %b} → {df['Date'].max():%d %b}), "
          f"regimes: {dict(df['regime'].value_counts())}\n")

    models = [("official", "pred_official"), ("challenger", "pred_challenger"),
              ("blend", "pred_blend")]
    print(f"{'model':12s} {'WAPE%':>7s} {'MAE pax':>8s} {'bias':>6s}  (fresh-regime rows only in parentheses)")
    fresh = df[df["regime"] == "fresh"]
    for name, col in models:
        sub = df.dropna(subset=[col])
        if sub.empty:
            continue
        fr = fresh.dropna(subset=[col])
        extra = f"  ({wape(fr['actual'], fr[col]):.2f}%)" if len(fr) else ""
        print(f"{name:12s} {wape(sub['actual'], sub[col]):7.2f} "
              f"{np.abs(sub[col]-sub['actual']).mean():8.1f} "
              f"{(sub[col]-sub['actual']).mean():+6.1f}{extra}")

    # paired bootstrap: challenger vs official
    sub = df.dropna(subset=["pred_challenger"])
    y = sub["actual"].values
    po, pc = sub["pred_official"].values, sub["pred_challenger"].values
    rng = np.random.default_rng(0)
    diffs = np.array([wape(y[i], pc[i]) - wape(y[i], po[i])
                      for i in (rng.integers(0, len(y), len(y)) for _ in range(10000))])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\nchallenger − official WAPE: {wape(y, pc) - wape(y, po):+.2f} "
          f"(95% CI [{lo:+.2f}, {hi:+.2f}], P(better) {np.mean(diffs < 0):.0%})")
    verdict = ("ADOPT challenger for next retrain" if hi < 0 else
               "KEEP official" if lo > 0 else
               "still ambiguous — extend the shadow period")
    print("verdict:", verdict)


if __name__ == "__main__":
    main()
