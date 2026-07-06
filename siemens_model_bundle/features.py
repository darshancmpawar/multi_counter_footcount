"""Siemens lunch counter demand — leakage-safe feature builder.
Grain: one row = Date + Counter Name. Every historical feature is shift(1)-protected:
it uses only information available the evening BEFORE service (vendor ordering time).
Known-in-advance inputs: calendar, planned menu, active-counter plan.
"""
import pandas as pd
import numpy as np

STAR_RULES = [  # (keyword regex, flag name, star score)
    (r'biriyani|biryani', 'has_biryani', None),      # score decided by veg/nonveg below
    (r'mutton|lamb|gosht', 'has_mutton', 4.5),
    (r'fish|prawn|meen|seafood|apollo', 'has_fish', 4.0),
    (r'chicken|murg', 'has_chicken', 3.5),
    (r'egg|anda', 'has_egg', 2.0),
    (r'paneer', 'has_paneer', 3.5),
    (r'mushroom', 'has_mushroom', 3.0),
    (r'chole|chana|chholay', 'has_chole', 2.5),
    (r'gobi manchurian|manchurian|65|noodles|fried rice', 'has_indochinese', 2.5),
]
PAN_FLAGS = ['Ekadashi', 'Poornima', 'Amavasya', 'Pradosham', 'Shravan',
             'Navratri', 'Sankashti', 'Shivaratri']


def build_counterday(df):
    """Aggregate item-level rows to counter-day grain with menu features."""
    df = df.copy()
    df['item_l'] = df['Item Name'].str.lower()
    g = df.groupby(['Date', 'Counter Name'])
    cd = g.agg(cc=('Counter Consumed', 'first'), co=('Counter Ordered', 'first'),
               tlc=('Total Lunch Consumed', 'first'), hc=('Headcount', 'first'),
               weekday=('Weekday', 'first'), daytype=('Day Type', 'first'),
               pan=('Panchangam', 'first'),
               n_items=('Item Name', 'nunique'),
               n_categories=('Category', 'nunique'),
               items=('Item Name', lambda s: ' | '.join(sorted(s)))).reset_index()

    # category flags
    cat = df.groupby(['Date', 'Counter Name'])['Category'].agg(set).reset_index(name='cats')
    cd = cd.merge(cat, on=['Date', 'Counter Name'])
    cd['has_dessert'] = cd['cats'].apply(lambda s: any('Dessert' in c for c in s)).astype(int)
    cd['has_rice'] = cd['cats'].apply(lambda s: any('Rice' in c for c in s)).astype(int)
    cd['has_bread'] = cd['cats'].apply(lambda s: any('Bread' in c for c in s)).astype(int)
    cd['has_gravy'] = cd['cats'].apply(lambda s: any('Gravy' in c for c in s)).astype(int)
    cd['has_dry'] = cd['cats'].apply(lambda s: any('Dry' in c for c in s)).astype(int)

    # star item keyword flags from concatenated item text
    txt = g['item_l'].agg(' ; '.join).reset_index(name='txt')
    cd = cd.merge(txt, on=['Date', 'Counter Name'])
    for pat, flag, _ in STAR_RULES:
        cd[flag] = cd['txt'].str.contains(pat, regex=True).astype(int)
    cd['is_nonveg_counter'] = cd['Counter Name'].str.contains('Non Veg').astype(int)
    cd['is_south'] = cd['Counter Name'].str.startswith('South').astype(int)
    cd['has_nv_biryani'] = ((cd['has_biryani'] == 1) & (cd['is_nonveg_counter'] == 1)).astype(int)

    # rule-based star score (max pull) and menu strength (sum of pulls)
    def score_row(r):
        scores = []
        if r['has_nv_biryani']:
            scores.append(5.0)
        elif r['has_biryani']:
            scores.append(2.5)
        for pat, flag, sc in STAR_RULES:
            if sc is not None and r[flag]:
                scores.append(sc)
        return (max(scores) if scores else 0.0), sum(scores)
    ss = cd.apply(score_row, axis=1, result_type='expand')
    cd['star_score'] = ss[0]
    cd['menu_strength'] = ss[1]
    return cd.drop(columns=['cats', 'txt'])


def add_calendar(cd):
    cd = cd.copy()
    cd['month'] = cd['Date'].dt.month
    cd['dom'] = cd['Date'].dt.day
    cd['weeknum'] = cd['Date'].dt.isocalendar().week.astype(int)
    cd['is_monday'] = (cd['weekday'] == 'Monday').astype(int)
    cd['is_friday'] = (cd['weekday'] == 'Friday').astype(int)
    cd['is_month_start'] = (cd['dom'] <= 3).astype(int)
    cd['is_month_end'] = (cd['dom'] >= 27).astype(int)
    cd['dt_prev_holiday'] = (cd['daytype'] == 'Previous Day of Holiday').astype(int)
    cd['dt_next_holiday'] = (cd['daytype'] == 'Next Day of Holiday').astype(int)
    for p in PAN_FLAGS:
        cd[f'pan_{p.lower()}'] = cd['pan'].str.contains(p, case=False).astype(int)
    cd['pan_any'] = (cd['pan'] != 'Regular').astype(int)
    return cd


def add_active_and_competitor(cd):
    cd = cd.copy().sort_values(['Date', 'Counter Name'])
    day = cd.groupby('Date').agg(
        n_active=('Counter Name', 'nunique'),
        n_active_veg=('is_nonveg_counter', lambda s: int((1 - s).sum())),
        n_active_nonveg=('is_nonveg_counter', 'sum'),
        combo=('Counter Name', lambda s: '+'.join(sorted(x.replace(' ', '')[:2] + ('N' if 'Non' in x else '')
                                                          for x in s)))).reset_index()
    cd = cd.merge(day, on='Date')
    # competitor aggregates: other ACTIVE counters that day
    comp_cols = ['has_biryani', 'has_nv_biryani', 'has_paneer', 'has_mutton',
                 'has_chicken', 'has_fish', 'star_score']
    def comp(g):
        out = {}
        for i, r in g.iterrows():
            others = g.drop(i)
            if len(others) == 0:
                out[i] = {f'oth_{c}': 0 for c in comp_cols} | {'oth_max_star': 0.0}
            else:
                d = {f'oth_{c}': others[c].max() for c in comp_cols}
                d['oth_max_star'] = others['star_score'].max()
                out[i] = d
        return pd.DataFrame.from_dict(out, orient='index')
    comps = cd.groupby('Date', group_keys=False).apply(comp)
    cd = cd.join(comps)
    cd['star_minus_oth'] = cd['star_score'] - cd['oth_max_star']
    cd['star_rank'] = cd.groupby('Date')['star_score'].rank(ascending=False, method='min')
    return cd


def add_history(cd):
    """All shifted — usable at prediction time (previous working day's actuals known)."""
    cd = cd.copy().sort_values(['Counter Name', 'Date'])
    cd['share'] = cd['cc'] / cd['tlc']
    gc = cd.groupby('Counter Name', group_keys=False)
    cd['lag1'] = gc['cc'].shift(1)
    cd['roll3'] = gc['cc'].apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    cd['roll7'] = gc['cc'].apply(lambda s: s.shift(1).rolling(7, min_periods=2).mean())
    cd['roll14'] = gc['cc'].apply(lambda s: s.shift(1).rolling(14, min_periods=3).mean())
    cd['roll7_std'] = gc['cc'].apply(lambda s: s.shift(1).rolling(7, min_periods=3).std())
    # same-weekday history per counter
    gw = cd.groupby(['Counter Name', 'weekday'], group_keys=False)
    cd['wd_lag1'] = gw['cc'].shift(1)
    cd['wd_roll4'] = gw['cc'].apply(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    # share history
    cd['share_roll7'] = gc['share'].apply(lambda s: s.shift(1).rolling(7, min_periods=2).mean())
    cd['wd_share_roll4'] = gw['share'].apply(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    # daily total history (merge at date level)
    dtot = cd.groupby('Date')['tlc'].first().sort_index()
    dtot_df = pd.DataFrame({'tlc_lag1': dtot.shift(1),
                            'tlc_roll5': dtot.shift(1).rolling(5, min_periods=2).mean(),
                            'tlc_roll10': dtot.shift(1).rolling(10, min_periods=3).mean()}).reset_index()
    wd_map = cd.groupby('Date')['weekday'].first()
    dtot_wd = dtot.groupby(wd_map, group_keys=False).apply(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    dtot_wd = dtot_wd.rename('tlc_wd_roll4').rename_axis('Date').reset_index()
    cd = cd.merge(dtot_df, on='Date', how='left').merge(dtot_wd, on='Date', how='left')
    return cd.sort_values(['Date', 'Counter Name']).reset_index(drop=True)


NUM_FEATURES = ['n_items', 'n_categories', 'has_dessert', 'has_rice', 'has_bread',
                'has_gravy', 'has_dry', 'has_biryani', 'has_nv_biryani', 'has_mutton',
                'has_fish', 'has_chicken', 'has_egg', 'has_paneer', 'has_mushroom',
                'has_chole', 'has_indochinese', 'is_nonveg_counter', 'is_south',
                'star_score', 'menu_strength', 'month', 'dom', 'is_monday', 'is_friday',
                'is_month_start', 'is_month_end', 'dt_prev_holiday', 'dt_next_holiday',
                'pan_ekadashi', 'pan_poornima', 'pan_amavasya', 'pan_pradosham',
                'pan_shravan', 'pan_navratri', 'pan_sankashti', 'pan_shivaratri', 'pan_any',
                'n_active', 'n_active_veg', 'n_active_nonveg',
                'oth_has_biryani', 'oth_has_nv_biryani', 'oth_has_paneer', 'oth_has_mutton',
                'oth_has_chicken', 'oth_has_fish', 'oth_max_star', 'star_minus_oth', 'star_rank',
                'lag1', 'roll3', 'roll7', 'roll14', 'roll7_std', 'wd_lag1', 'wd_roll4',
                'share_roll7', 'wd_share_roll4', 'tlc_lag1', 'tlc_roll5', 'tlc_roll10', 'tlc_wd_roll4']
CAT_FEATURES = ['Counter Name', 'weekday']


def build_all(df):
    cd = build_counterday(df)
    cd = add_calendar(cd)
    cd = add_active_and_competitor(cd)
    cd = add_history(cd)
    return cd


def design_matrix(cd, feature_medians=None):
    """One-hot categoricals; median-impute NaNs (medians fit on TRAIN only)."""
    X = cd[NUM_FEATURES].copy()
    for c in CAT_FEATURES:
        X = pd.concat([X, pd.get_dummies(cd[c], prefix=c[:2]).astype(int)], axis=1)
    if feature_medians is None:
        feature_medians = X.median(numeric_only=True)
    X = X.fillna(feature_medians)
    return X, feature_medians
