"""Versioned *candidate* calibrator training from genuine historical OOS rows.

This command only writes a JSON candidate artifact. It never changes the
production engine or publishes betting predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .calibration import probabilities


FEATURES = ('log(p_H/p_D)', 'log(p_A/p_D)')
OUTCOMES = ('H', 'D', 'A')


def _cutoff(value):
    date = pd.Timestamp(value)
    if pd.isna(date):
        raise ValueError('Invalid as_of timestamp')
    if date.tzinfo is None:
        date = date.tz_localize('UTC')
    else:
        date = date.tz_convert('UTC')
    return date


def train_candidate(df: pd.DataFrame, as_of: str, min_samples: int = 600):
    """Train only on final historical OOS results strictly BEFORE UTC cutoff."""
    required = {'date', 'res', 'p_H', 'p_D', 'p_A'}
    missing = required - set(df)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if min_samples < 10:
        raise ValueError('min_samples must be at least 10')
    dates = pd.to_datetime(df.date, utc=True, errors='coerce')
    if dates.isna().any():
        raise ValueError('Invalid dates')
    if 'event_id' in df and df.event_id.duplicated().any():
        raise ValueError('Duplicate historical event_id')
    cutoff = _cutoff(as_of)
    if cutoff > pd.Timestamp.now(tz='UTC'):
        raise ValueError('Cannot train with a cutoff in the future')
    eligible = dates < cutoff
    history = df.loc[eligible].copy()
    if len(history) < min_samples:
        raise ValueError(f'Only {len(history)} historical matches before cutoff; need {min_samples}')
    y = history.res.map({'H': 0, 'D': 1, 'A': 2})
    if y.isna().any() or y.nunique() != 3:
        raise ValueError('Invalid outcomes or fewer than three classes')
    p = probabilities(history)
    x = np.log(p[:, [0, 2]] / p[:, [1]])
    fitted = LogisticRegression(C=1.0, max_iter=1000, random_state=0).fit(x, y.astype(int))
    if list(fitted.classes_) != [0, 1, 2]:
        raise ValueError('Unexpected outcome classes')
    return {
        'schema_version': 1,
        'model_type': 'research_multinomial_logit',
        'classes': list(OUTCOMES),
        'features': list(FEATURES),
        'coef': fitted.coef_.tolist(),
        'intercept': fitted.intercept_.tolist(),
        'trained_before_utc': cutoff.isoformat(),
        'max_training_match_utc': dates[eligible].max().isoformat(),
        'n_historical_matches': int(eligible.sum()),
        'validation_status': 'CHALLENGER_ONLY_NOT_PRODUCTION_APPROVED',
    }


def predict_candidate(artifact: dict, original_probabilities: np.ndarray) -> np.ndarray:
    """Use serialized JSON coefficients without loading arbitrary pickle data."""
    if artifact.get('schema_version') != 1 or artifact.get('classes') != list(OUTCOMES):
        raise ValueError('Incompatible candidate schema')
    p = np.asarray(original_probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError('Probabilities must have three strictly positive finite columns')
    p = p / p.sum(axis=1, keepdims=True)
    x = np.log(p[:, [0, 2]] / p[:, [1]])
    coef, intercept = np.asarray(artifact['coef'], dtype=float), np.asarray(artifact['intercept'], dtype=float)
    if coef.shape != (3, 2) or intercept.shape != (3,):
        raise ValueError('Malformed model coefficients')
    z = x @ coef.T + intercept
    z -= z.max(axis=1, keepdims=True)
    ex = np.exp(z)
    return ex / ex.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='reports/intl_real_validation/oos_raw.csv')
    ap.add_argument('--output', default='reports/intl_training/calibrator_candidate.json')
    ap.add_argument('--as-of', default=pd.Timestamp.now(tz='UTC').date().isoformat(),
                    help='Exclusive UTC cutoff; defaults to start of today')
    ap.add_argument('--min-samples', type=int, default=600)
    args = ap.parse_args()
    source = pd.read_csv(args.input)
    candidate = train_candidate(source, as_of=args.as_of, min_samples=args.min_samples)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(candidate, indent=2, sort_keys=True) + '\n', encoding='utf8')
    print(f'Candidate trained on {candidate["n_historical_matches"]} completed OOS matches before {candidate["trained_before_utc"]}')
    print(f'Artifact: {target}. NO production change.')


if __name__ == '__main__':
    main()
