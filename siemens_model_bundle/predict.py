"""Siemens lunch counter demand — LIVE SCORING for unseen dates (July 2026+).

Usage:
    python predict.py --plan plan.csv [--history Lunch_Master_Data_FINAL_cleaned_.xlsx]

plan.csv — one row per planned ITEM, columns:
    Date (yyyy-mm-dd), Counter Name, Item Name, Category,
    optional: Day Type (default Regular), Panchangam (default Regular)
Only ACTIVE counters appear in the plan. Closed counters are simply absent → predicted 0.

Outputs per active counter: predicted consumed, calibrated P10–P90 range,
suggested order (calibrated P75), risk level, plain-language explanation.
Also prints predicted Total Lunch Consumed = sum of counter predictions.

The script rebuilds ALL historical lag features from the full history + plan,
so it works for any future date as long as history is current.
"""
import argparse
import pickle
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, '.')
from features import build_all, NUM_FEATURES, CAT_FEATURES

ART = 'artifacts'


def load_models():
    cfg = pickle.load(open(f'{ART}/final_config.pkl', 'rb'))
    return {
        'point': lgb.Booster(model_file=f'{ART}/model_point.txt'),
        'q10': lgb.Booster(model_file=f'{ART}/model_q10.txt'),
        'q75': lgb.Booster(model_file=f'{ART}/model_q75.txt'),
        'q90': lgb.Booster(model_file=f'{ART}/model_q90.txt'),
        'cfg': cfg,
    }


def prep(d, cat_levels):
    X = d[NUM_FEATURES + CAT_FEATURES].copy()
    for c in CAT_FEATURES:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def explain(row, hist_wd_mean):
    bits = []
    if row['has_nv_biryani']:
        bits.append('non-veg biryani on the menu (historically the strongest pull item)')
    elif row['star_score'] >= 4:
        bits.append('a very-high-pull star item on the menu')
    if row['oth_has_nv_biryani'] and not row['has_nv_biryani']:
        bits.append('a competing counter serves non-veg biryani (drains this counter)')
    if row['star_minus_oth'] > 1:
        bits.append('this counter has the strongest menu among active counters today')
    if row['dt_prev_holiday'] or row['dt_next_holiday']:
        bits.append('holiday-adjacent day (attendance typically drops)')
    if row['weekday'] == 'Friday':
        bits.append('Friday (lowest-attendance weekday)')
    if row['weekday'] in ('Tuesday', 'Wednesday'):
        bits.append(f"{row['weekday']} (peak-attendance weekday)")
    base = f"recent same-weekday average for this counter is {row['wd_roll4']:.0f}"
    bits.append(base)
    return '; '.join(bits).capitalize() + '.'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('--history', default='Lunch_Master_Data_FINAL_cleaned_.xlsx')
    args = ap.parse_args()

    hist = pd.read_excel(args.history, sheet_name='Lunch Master')
    plan = pd.read_csv(args.plan, parse_dates=['Date'])
    for col, default in [('Day Type', 'Regular'), ('Panchangam', 'Regular')]:
        if col not in plan:
            plan[col] = default
    plan['Month'] = plan['Date'].dt.month_name()
    plan['Weekday'] = plan['Date'].dt.day_name()
    # placeholder values for target-side columns (never used as features)
    for col in ['Receiving Qty', 'Bainmarie Wastage']:
        plan[col] = np.nan
    for col in ['Headcount', 'Total Lunch Consumed', 'Counter Ordered', 'Counter Consumed']:
        plan[col] = 0

    full = pd.concat([hist, plan[hist.columns]], ignore_index=True)
    cd = build_all(full)
    target = cd[cd['Date'].isin(plan['Date'].unique())].copy()

    models = load_models()
    cfg = models['cfg']
    cat_levels = {'Counter Name': ['North Non Veg', 'North Veg', 'South Non Veg', 'South Veg'],
                  'weekday': ['Friday', 'Monday', 'Thursday', 'Tuesday', 'Wednesday']}
    X = prep(target, cat_levels)

    target['pred'] = np.clip(models['point'].predict(X), 0, None).round(0)
    target['lo'] = np.clip(np.clip(models['q10'].predict(X), 0, None) - cfg['cqr_Q80'], 0, None).round(0)
    target['hi'] = (np.clip(models['q90'].predict(X), 0, None) + cfg['cqr_Q80']).round(0)
    target['order'] = (np.clip(models['q75'].predict(X), 0, None) + cfg['order_corr']).round(-1)
    width = (target['hi'] - target['lo']) / target['pred'].clip(lower=1)
    target['risk'] = np.select([width > 0.45, width > 0.30], ['HIGH', 'MEDIUM'], 'LOW')

    hist_wd = None
    print('=' * 90)
    for dt, g in target.groupby('Date'):
        print(f"\nPREDICTION — {dt.date()} ({g['weekday'].iloc[0]}) | active counters: {len(g)}")
        print(f"Predicted Total Lunch Consumed: {int(g['pred'].sum())}")
        tot = g['pred'].sum()
        for _, r in g.sort_values('pred', ascending=False).iterrows():
            print(f"\n  {r['Counter Name']}: predicted {int(r['pred'])} "
                  f"(share {100*r['pred']/tot:.0f}%) | range {int(r['lo'])}-{int(r['hi'])} "
                  f"| suggested order {int(r['order'])} | risk {r['risk']}")
            print(f"    Why: {explain(r, hist_wd)}")
    print('\n' + '=' * 90)
    out = target[['Date', 'Counter Name', 'pred', 'lo', 'hi', 'order', 'risk']]
    out.to_csv('predictions_out.csv', index=False)
    print('Saved: predictions_out.csv')


if __name__ == '__main__':
    main()
