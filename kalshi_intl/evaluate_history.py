"""Reproducible international 1X2 walk-forward evaluation from historical ESPN cache.

Run: python -m kalshi_intl.evaluate_history --cache data/intl --first-year 2016
Requires historical ESPN cache or network connectivity. No synthetic matches.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from .data import load_group
from .model import IntlEngine, CONFIGS, GROUP_FEATURES, walk_forward
from .validation import evaluate_oos


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', default='data/intl')
    p.add_argument('--first-year', type=int, default=2016)
    p.add_argument('--output', default='reports/intl_real_validation')
    p.add_argument('--min-calibration', type=int, default=300)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--offline', action='store_true', help='Refuse network use; require existing cache')
    a = p.parse_args()
    dest = Path(a.output); dest.mkdir(parents=True, exist_ok=True)
    if a.offline:
        from .data import GROUPS, months
        missing = [str(Path(a.cache)/code/(ym+'.json')) for code in GROUPS['selecciones']['target'].keys() | set(GROUPS['selecciones']['feed']) for ym in months(a.first_year) if not (Path(a.cache)/code/(ym+'.json')).exists()]
        if missing:
            raise SystemExit(f'Offline mode: missing {len(missing)} cache files. First missing: {missing[:3]}')
    games = load_group('selecciones', first_year=a.first_year, cache_dir=a.cache)
    features = IntlEngine(CONFIGS['selecciones']).process(games)
    raw = walk_forward(features, first_year=max(a.first_year+4, 2020), features=GROUP_FEATURES['selecciones'])
    if raw.empty:
        raise SystemExit('Insufficient historical games for walk-forward validation')
    raw.to_csv(dest/'oos_raw.csv', index=False)
    paired, metrics, ci = evaluate_oos(raw, min_train=a.min_calibration, bootstrap=a.bootstrap)
    paired.to_csv(dest/'oos_paired.csv', index=False)
    metrics.to_csv(dest/'metrics.csv', index=False)
    ci.to_csv(dest/'confidence_intervals.csv', index=False)
    print('Raw OOS matches:', len(raw), '| calibrated OOS matches:', len(paired))
    print(metrics.to_string(index=False) if not metrics.empty else 'Insufficient earlier OOS seasons to fit calibration')
    print(ci.to_string(index=False) if not ci.empty else '')
    print('Reports:', dest)

if __name__ == '__main__':
    main()
