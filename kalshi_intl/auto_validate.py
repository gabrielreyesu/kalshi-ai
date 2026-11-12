"""Conservative automated champion/challenger validation on paired historical OOS predictions.

No bet placement, no model overwrites. Promotion is a recommendation only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .calibration import probabilities, scores
from .segment_policy import evaluate_policy


def run(df, *, min_history=150, min_gain=0.005, bootstrap=2000, seed=42):
    if bootstrap < 100: raise ValueError('bootstrap must be >= 100')
    if df.empty: raise ValueError('empty dataset')
    required={'temporada','date','res','friendly','p_H','p_D','p_A','cal_H','cal_D','cal_A'}
    if not required.issubset(df.columns): raise ValueError(f'missing {sorted(required-set(df.columns))}')
    if df.duplicated().any(): raise ValueError('duplicate rows detected')
    dates=pd.to_datetime(df.date,utc=True,errors='coerce')
    if dates.isna().any(): raise ValueError('invalid dates')
    season=pd.to_numeric(df.temporada.astype(str).str.extract(r'^(\d{4})')[0],errors='raise')
    if season.nunique()<3: raise ValueError('at least three seasons required')
    latest=int(season.max())
    predicted, decisions, all_metrics=evaluate_policy(df,min_history=min_history,min_gain=min_gain)
    hold=predicted.loc[season.eq(latest)].copy()
    if len(hold)<50: raise ValueError('latest season has fewer than 50 evaluated matches')
    raw=probabilities(hold,('p_H','p_D','p_A'))
    cal=probabilities(hold,('cal_H','cal_D','cal_A'))
    policy=probabilities(hold,('policy_H','policy_D','policy_A'))
    y=hold.res.map({'H':0,'D':1,'A':2}).to_numpy(dtype=int)
    row=np.arange(len(y))
    losses={name:-np.log(p[row,y]) for name,p in [('original',raw),('calibrated',cal),('adaptive',policy)]}
    briers={name:((p-np.eye(3)[y])**2).sum(axis=1) for name,p in [('original',raw),('calibrated',cal),('adaptive',policy)]}
    metrics=pd.DataFrame([{'method':name,**scores(p,hold.res)} for name,p in [('original',raw),('calibrated',cal),('adaptive',policy)]])
    groups=hold.assign(_day=pd.to_datetime(hold.date,utc=True).dt.date).groupby('_day',sort=True).indices
    indices=list(groups.values()); rng=np.random.default_rng(seed)
    comparisons=[]
    for candidate in ('calibrated','adaptive'):
        diff=losses[candidate]-losses['original']
        db=briers[candidate]-briers['original']
        boots=np.empty(bootstrap)
        for b in range(bootstrap):
            selected=rng.integers(0,len(indices),len(indices))
            ids=np.concatenate([indices[j] for j in selected])
            boots[b]=diff[ids].mean()
        low,high=np.quantile(boots,[0.025,0.975])
        comparisons.append({'candidate':candidate,'holdout_season':latest,'n':len(hold),
                            'delta_logloss':float(diff.mean()),'ci95_low':float(low),'ci95_high':float(high),
                            'delta_brier':float(db.mean()),
                            'delta_accuracy':float(metrics.set_index('method').loc[candidate,'accuracy']-metrics.set_index('method').loc['original','accuracy']),
                            'promote_recommended':bool(high<0 and db.mean()<0 and metrics.set_index('method').loc[candidate,'accuracy']>=metrics.set_index('method').loc['original','accuracy'])})
    return metrics,pd.DataFrame(comparisons),decisions,hold


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='reports/intl_real_validation/oos_paired.csv')
    ap.add_argument('--output',default='reports/intl_auto_validation')
    ap.add_argument('--bootstrap',type=int,default=2000)
    args=ap.parse_args()
    metrics,comparison,decisions,hold=run(pd.read_csv(args.input),bootstrap=args.bootstrap)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    metrics.to_csv(out/'holdout_metrics.csv',index=False)
    comparison.to_csv(out/'holdout_comparisons.csv',index=False)
    decisions.to_csv(out/'historical_decisions.csv',index=False)
    (out/'status.json').write_text(json.dumps({'holdout_season':int(comparison.holdout_season.iloc[0]),'holdout_matches':len(hold),
       'any_promotion_recommended':bool(comparison.promote_recommended.any()),'production_changed':False,
       'note':'No automated model promotion or betting; strict validation required.'},indent=2))
    print(metrics.to_string(index=False));print(comparison.to_string(index=False))
    print('Production unchanged. Reports:',out)
if __name__=='__main__':main()
