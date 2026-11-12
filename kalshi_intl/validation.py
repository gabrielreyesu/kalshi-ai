"""Offline, paired evaluation of OOS predictions. No market or future-data downloads."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .calibration import probabilities, scores, temporal_calibrate

MARKET = ('market_H', 'market_D', 'market_A')


def evaluate_oos(df: pd.DataFrame, min_train: int = 300, bootstrap: int = 1000, seed: int = 17):
    """Compare raw vs temporal calibration only on rows actually calibrated.

    Bootstrap resamples entire calendar dates (clusters), not individual markets.
    Market columns are optional; market comparisons use the same rows.
    """
    if bootstrap < 0:
        raise ValueError('bootstrap must be nonnegative')
    required = {'date', 'temporada', 'res', 'p_H', 'p_D', 'p_A'}
    if not required.issubset(df.columns):
        raise ValueError(f'missing columns: {sorted(required - set(df.columns))}')
    if df['date'].isna().any() or pd.to_datetime(df['date'], errors='coerce').isna().any():
        raise ValueError('missing or invalid dates')
    if df.duplicated(subset=['date', 'home', 'away']).any() if {'home','away'}.issubset(df.columns) else False:
        raise ValueError('duplicate fixture rows')
    d = temporal_calibrate(df, min_train=min_train)
    d = d[d['calibrated']].copy()
    if d.empty:
        return d, pd.DataFrame(), pd.DataFrame()
    labels = d['res'].map({'H':0,'D':1,'A':2})
    if labels.isna().any():
        raise ValueError('invalid outcome')
    y = labels.to_numpy(dtype=int)
    methods = {'original': ('p_H','p_D','p_A'), 'calibrated': ('cal_H','cal_D','cal_A')}
    if set(MARKET).issubset(d.columns):
        valid = d[list(MARKET)].notna().all(axis=1)
        if valid.any():
            methods['market'] = MARKET
    rows = []
    for name, cols in methods.items():
        sub = d if name != 'market' else d[d[list(MARKET)].notna().all(axis=1)]
        rows.append({'method': name, **scores(probabilities(sub, cols), sub['res'])})
    metrics = pd.DataFrame(rows)
    p = probabilities(d, methods['original']); q = probabilities(d, methods['calibrated'])
    onehot = np.eye(3)[y]
    differences = {
        'logloss': -np.log(np.maximum(q[np.arange(len(y)), y], 1e-12)) + np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)),
        'brier': ((q-onehot)**2).sum(axis=1) - ((p-onehot)**2).sum(axis=1),
        'accuracy_error': (q.argmax(axis=1) != y).astype(float) - (p.argmax(axis=1) != y).astype(float),
    }
    dates = pd.to_datetime(d['date']).dt.normalize().to_numpy()
    groups = [np.flatnonzero(dates == dt) for dt in np.unique(dates)]
    rng = np.random.default_rng(seed)
    intervals = []
    for metric, values in differences.items():
        mean = float(values.mean())
        if bootstrap and len(groups) >= 2:
            draws = []
            for _ in range(bootstrap):
                sampled = rng.integers(0, len(groups), size=len(groups))
                ix = np.concatenate([groups[i] for i in sampled])
                draws.append(float(values[ix].mean()))
            low, high = np.percentile(draws, [2.5, 97.5])
        else:
            low = high = np.nan
        intervals.append({'metric': metric, 'delta_calibrated_minus_original': mean,
                          'ci95_low': float(low), 'ci95_high': float(high),
                          'n_matches': len(d), 'n_dates': len(groups)})
    return d, metrics, pd.DataFrame(intervals)
