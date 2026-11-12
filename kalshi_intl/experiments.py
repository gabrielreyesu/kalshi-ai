"""Walk-forward research tournament: no production changes or auto-promotion."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from .auto_train import train_candidate, predict_candidate
from .calibration import probabilities, scores

METHODS = ('original', 'logit_full', 'blend_25', 'blend_50', 'temperature')

def _temperature_fit(p, y):
    def loss(t):
        z = np.log(np.maximum(p, 1e-12)) / t
        z -= z.max(axis=1, keepdims=True)
        q = np.exp(z); q /= q.sum(axis=1, keepdims=True)
        return float(-np.log(np.maximum(q[np.arange(len(y)), y], 1e-12)).mean())
    return float(minimize_scalar(loss, bounds=(.5, 2.5), method='bounded').x)

def _apply_temperature(p, t):
    z = np.log(np.maximum(p, 1e-12))/t
    z -= z.max(axis=1, keepdims=True)
    q = np.exp(z)
    return q/q.sum(axis=1,keepdims=True)

def evaluate_methods(df, min_train=300, bootstrap=1000, seed=42):
    if min_train < 50 or bootstrap < 0:
        raise ValueError('Invalid parameters')
    required={'temporada','date','res','p_H','p_D','p_A'}
    if not required.issubset(df):
        raise ValueError(f'Missing columns: {sorted(required-set(df))}')
    if 'event_id' in df and df.event_id.duplicated().any():
        raise ValueError('Duplicate event_id')
    d=df.copy().reset_index(drop=True)
    d['_season']=pd.to_numeric(d.temporada.astype(str).str.extract(r'^(\d{4})')[0], errors='raise').astype(int)
    dates=pd.to_datetime(d.date,utc=True,errors='coerce')
    if dates.isna().any():
        raise ValueError('Invalid dates')
    y=d.res.map({'H':0,'D':1,'A':2})
    if y.isna().any():
        raise ValueError('Invalid 1X2 results')
    p=probabilities(d)
    outputs=[]; fits=[]
    for season in sorted(d['_season'].unique()):
        cutoff=pd.Timestamp(f'{season}-07-01',tz='UTC')
        train=(dates<cutoff)&(d['_season']<season)
        test=(d['_season']==season)&(dates>=cutoff)
        if int(train.sum())<min_train or not test.any():
            continue
        candidate=train_candidate(d.loc[train],as_of=cutoff.isoformat(),min_samples=min_train)
        raw=p[test]
        calibrated=predict_candidate(candidate,raw)
        t=_temperature_fit(p[train],y[train].to_numpy(dtype=int))
        arr={'original':raw, 'logit_full':calibrated,
             'blend_25':.75*raw+.25*calibrated,
             'blend_50':.5*raw+.5*calibrated,
             'temperature':_apply_temperature(raw,t)}
        selected=d.loc[test,['temporada','date','res']].copy()
        if 'event_id' in d: selected['event_id']=d.loc[test,'event_id'].to_numpy()
        fits.append({'season':int(season),'n_training':int(train.sum()),
                     'n_evaluated':int(test.sum()),'temperature':t})
        for method,q in arr.items():
            part=selected.copy()
            part['method']=method
            for j,c in enumerate('HDA'): part['p_'+c]=q[:,j]
            outputs.append(part)
    if not outputs:
        raise ValueError('Insufficient OOS history')
    predictions=pd.concat(outputs,ignore_index=True)
    by_season=pd.DataFrame([
        {'temporada':s,'method':m,**scores(probabilities(g),g.res)}
        for (s,m),g in predictions.groupby(['temporada','method'])
    ])
    overall=pd.DataFrame([
        {'method':m,**scores(probabilities(g),g.res)}
        for m,g in predictions.groupby('method')
    ])
    original=predictions[predictions.method=='original'].reset_index(drop=True)
    labels=original.res.map({'H':0,'D':1,'A':2}).to_numpy(dtype=int)
    date_groups=original.assign(_day=pd.to_datetime(original.date,utc=True).dt.date).groupby('_day',sort=True).indices
    groups=list(date_groups.values());rng=np.random.default_rng(seed)
    b=probabilities(original)
    baseline=-np.log(np.maximum(b[np.arange(len(labels)),labels],1e-12))
    intervals=[]
    for method in METHODS[1:]:
        new=predictions[predictions.method==method].reset_index(drop=True)
        if not original[['temporada','date','res']].equals(new[['temporada','date','res']]):
            raise ValueError('Unpaired evaluation rows')
        q=probabilities(new)
        delta=-np.log(np.maximum(q[np.arange(len(labels)),labels],1e-12))-baseline
        if bootstrap and len(groups)>1:
            samples=[]
            for _ in range(bootstrap):
                idx=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
                samples.append(delta[idx].mean())
            low,high=np.percentile(samples,[2.5,97.5])
        else: low,high=np.nan,np.nan
        intervals.append({'method':method,'n':len(labels),
                          'logloss_delta_vs_original':float(delta.mean()),
                          'ci95_low':float(low),'ci95_high':float(high)})
    return predictions,by_season,overall,pd.DataFrame(intervals),pd.DataFrame(fits)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='reports/intl_real_validation/oos_raw.csv')
    ap.add_argument('--output',default='reports/intl_experiments')
    ap.add_argument('--min-train',type=int,default=300)
    ap.add_argument('--bootstrap',type=int,default=1000)
    a=ap.parse_args()
    pred,by,overall,ci,fits=evaluate_methods(pd.read_csv(a.input),a.min_train,a.bootstrap)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    for name,data in {'predictions.csv':pred,'metrics_by_season.csv':by,
        'overall_metrics.csv':overall,'confidence_intervals.csv':ci,
        'fit_manifest.csv':fits}.items():
        data.to_csv(out/name,index=False)
    print(overall.to_string(index=False))
    print(ci.to_string(index=False))
    print('Research only. Production unchanged. Reports:',out)

if __name__=='__main__':main()
