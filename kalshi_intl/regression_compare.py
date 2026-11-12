"""Paired OOS regression audit of corrected vs old international 1X2 engines."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .calibration import probabilities, scores


def compare_models(baseline: pd.DataFrame, candidate: pd.DataFrame, bootstrap: int = 1000, seed: int = 42):
    required = {'event_id','res','date','p_H','p_D','p_A'}
    for name, data in [('baseline',baseline),('candidate',candidate)]:
        if not required.issubset(data):
            raise ValueError(f'{name}: missing columns {sorted(required-set(data))}')
        if data.event_id.isna().any() or data.event_id.duplicated().any():
            raise ValueError(f'{name}: invalid or duplicated event_id')
    a = baseline.copy().set_index('event_id').sort_index()
    b = candidate.copy().set_index('event_id').sort_index()
    if len(a) != len(b) or not a.index.equals(b.index):
        raise ValueError('The two engines evaluated different fixtures')
    if not (a[['res','date']] == b[['res','date']]).all().all():
        raise ValueError('Different dates/outcomes between engines')
    if bootstrap < 0:
        raise ValueError('bootstrap must be nonnegative')
    pa, pb = probabilities(a), probabilities(b)
    y = a.res.map({'H':0,'D':1,'A':2}).to_numpy(dtype=int)
    metrics = pd.DataFrame([
        {'model':'baseline',**scores(pa,a.res)},
        {'model':'candidate',**scores(pb,b.res)},
    ])
    idx=np.arange(len(y))
    dates=pd.to_datetime(a.date,utc=True,errors='coerce')
    if dates.isna().any():
        raise ValueError('Invalid dates')
    groups=list(pd.DataFrame({'day':dates.dt.date.to_numpy(), 'i':idx}).groupby('day')['i'].apply(np.asarray))
    rng=np.random.default_rng(seed)
    target=np.eye(3)[y]
    differences = {
        'logloss': -np.log(np.maximum(pb[idx,y],1e-12)) + np.log(np.maximum(pa[idx,y],1e-12)),
        'brier': ((pb-target)**2).sum(axis=1) - ((pa-target)**2).sum(axis=1),
        'accuracy_error': (pb.argmax(axis=1)!=y).astype(float)-(pa.argmax(axis=1)!=y).astype(float),
    }
    rows=[]
    for label,diff in differences.items():
        if bootstrap and len(groups)>1:
            draws=[]
            for _ in range(bootstrap):
                s=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
                draws.append(float(diff[s].mean()))
            lo,hi=np.percentile(draws,[2.5,97.5])
        else:
            lo=hi=np.nan
        rows.append({'metric':label,'delta_candidate_minus_baseline':float(diff.mean()),
                     'ci95_low':float(lo),'ci95_high':float(hi), 'matches':len(a), 'dates':len(groups)})
    return metrics,pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--baseline',required=True)
    ap.add_argument('--candidate',required=True)
    ap.add_argument('--output',default='reports/intl_regression_comparison')
    ap.add_argument('--bootstrap',type=int,default=1000)
    a=ap.parse_args()
    metrics,intervals=compare_models(pd.read_csv(a.baseline),pd.read_csv(a.candidate),a.bootstrap)
    p=Path(a.output)
    p.mkdir(parents=True,exist_ok=True)
    metrics.to_csv(p/'metrics.csv',index=False)
    intervals.to_csv(p/'paired_confidence_intervals.csv',index=False)
    print(metrics.to_string(index=False))
    print(intervals.to_string(index=False))
    print('Research only: not a production change.')


if __name__=='__main__':
    main()
