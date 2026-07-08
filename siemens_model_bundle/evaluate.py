"""Shared evaluation harness — every model is scored on identical metrics."""
import numpy as np
import pandas as pd


def metrics(y_true, y_pred, name=''):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    ae = np.abs(err)
    return pd.Series({
        'MAE': ae.mean(),
        'RMSE': np.sqrt((err ** 2).mean()),
        'MAPE%': 100 * (ae / y_true).mean(),
        'WAPE%': 100 * ae.sum() / y_true.sum(),
        'Bias': err.mean(),
        'Over%': 100 * (err > 0).mean(),
        'Under%': 100 * (err < 0).mean(),
    }, name=name)


def wape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return 100 * np.abs(y_pred - y_true).sum() / y_true.sum()


def kitchen_benchmark(history, month=None):
    """Business KPI line: how well does the kitchen's own ordering track
    demand? WAPE(Counter Ordered vs Counter Consumed) at counter-day grain.
    Pass month='2026-06' to score one month, None for lifetime."""
    cd = (history.groupby(['Date', 'Counter Name'])
          .agg(ordered=('Counter Ordered', 'first'),
               consumed=('Counter Consumed', 'first')).reset_index())
    if month is not None:
        cd = cd[cd['Date'].astype(str).str.startswith(month)]
    err = cd['ordered'] - cd['consumed']
    return pd.Series({
        'WAPE%': 100 * err.abs().sum() / cd['consumed'].sum(),
        'Bias (plates/day)': err.mean(),
        'Short days %': 100 * (err < 0).mean(),
        'n': len(cd),
    }, name=f'kitchen ({month or "lifetime"})')
