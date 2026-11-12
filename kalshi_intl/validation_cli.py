"""CLI for comparing original and calibrated international 1X2 probabilities."""
import argparse
from pathlib import Path
import pandas as pd
from .validation import evaluate_oos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', default='reports/intl_validation')
    ap.add_argument('--min-train', type=int, default=300)
    ap.add_argument('--bootstrap', type=int, default=1000)
    args = ap.parse_args()
    if args.min_train < 1 or args.bootstrap < 0:
        ap.error('min-train must be positive and bootstrap nonnegative')
    d, metrics, intervals = evaluate_oos(pd.read_csv(args.input), args.min_train, args.bootstrap)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out/'paired_predictions.csv', index=False)
    metrics.to_csv(out/'paired_metrics.csv', index=False)
    intervals.to_csv(out/'paired_intervals.csv', index=False)
    print(metrics.to_string(index=False) if not metrics.empty else 'Insufficient eligible predictions')
    print(intervals.to_string(index=False) if not intervals.empty else '')


if __name__ == '__main__':
    main()
