"""CLI for diagnostics on international OOS predictions."""
import argparse
from pathlib import Path
import pandas as pd
from .calibration import report, reliability, PROB_COLS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='reports/intl_calibration')
    parser.add_argument('--min-train', type=int, default=300)
    args = parser.parse_args()
    if args.min_train < 1:
        parser.error('--min-train must be positive')
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.input, parse_dates=['date'])
    calibrated, metrics = report(raw, args.min_train)
    calibrated.to_csv(dest / 'predictions_calibrated.csv', index=False)
    metrics.to_csv(dest / 'metrics_by_season.csv', index=False)
    reliability(calibrated, PROB_COLS).to_csv(dest / 'reliability_original.csv', index=False)
    reliability(calibrated, ('cal_H','cal_D','cal_A')).to_csv(dest / 'reliability_calibrated.csv', index=False)
    print(metrics.to_string(index=False))
    print('Reports:', dest)


if __name__ == '__main__':
    main()
