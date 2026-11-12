"""Prospective season-by-season selection of raw vs calibrated 1X2 probabilities.

Only prior OOS seasons determine each segment's choice. Exploratory research,
not a production betting signal. Input is paired walk-forward predictions.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .calibration import probabilities, scores


def evaluate_policy(df: pd.DataFrame, min_history: int = 150, min_gain: float = 0.005):
    needed = {'temporada','date','res','friendly','p_H','p_D','p_A','cal_H','cal_D','cal_A'}
    if not needed.issubset(df):
        raise ValueError(f'Missing columns: {sorted(needed-set(df))}')
    if min_history < 1 or min_gain < 0:
        raise ValueError('Invalid policy parameters')
    d = df.copy().reset_index(drop=True)
    d['season_start'] = pd.to_numeric(d.temporada.astype(str).str.extract(r'^(\d{4})')[0],errors='raise').astype(int)
    dates = pd.to_datetime(d.date,utc=True,errors='coerce')
    if dates.isna().any():
        raise ValueError('Invalid dates')
    d['segment'] = np.where(d.friendly.astype(int).eq(1),'friendly','competitive')
    p = probabilities(d,('p_H','p_D','p_A'))
    q = probabilities(d,('cal_H','cal_D','cal_A'))
    y = d.res.map({'H':0,'D':1,'A':2})
    if y.isna().any(): raise ValueError('Invalid results')
    y = y.to_numpy(dtype=int)
    idx = np.arange(len(d))
    raw_loss = -np.log(p[idx,y]); cal_loss = -np.log(q[idx,y])
    use_cal = np.zeros(len(d),dtype=bool)
    decisions=[]
    for season in sorted(d.season_start.unique()):
        for segment in ('friendly','competitive'):
            historical = (d.season_start.to_numpy()<season)&(dates < pd.Timestamp(f'{season}-07-01',tz='UTC'))&(d.segment.to_numpy()==segment)
            target = (d.season_start.to_numpy()==season)&(d.segment.to_numpy()==segment)
            n = int(historical.sum())
            improvement = float((raw_loss[historical]-cal_loss[historical]).mean()) if n else np.nan
            chosen = bool(n>=min_history and improvement>min_gain)
            use_cal[target] = chosen
            decisions.append({'season':int(season),'segment':segment,'prior_matches':n,'prior_logloss_gain':improvement,'selected':'calibrated' if chosen else 'original','evaluated_matches':int(target.sum())})
    combined = np.where(use_cal[:,None],q,p)
    d['policy'] = np.where(use_cal,'calibrated','original')
    for j,c in enumerate(('H','D','A')): d['policy_'+c] = combined[:,j]
    results = pd.DataFrame([{'method':label,**scores(arr,d.res)} for label,arr in [('original',p),('calibrated',q),('adaptive_policy',combined)]])
    return d,pd.DataFrame(decisions),results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',default='reports/intl_policy_research')
    parser.add_argument('--min-history',type=int,default=150)
    parser.add_argument('--min-gain',type=float,default=0.005)
    a=parser.parse_args()
    df=pd.read_csv(a.input)
    out,decisions,metrics=evaluate_policy(df,a.min_history,a.min_gain)
    path=Path(a.output);path.mkdir(parents=True,exist_ok=True)
    out.to_csv(path/'policy_predictions.csv',index=False)
    decisions.to_csv(path/'policy_decisions.csv',index=False)
    metrics.to_csv(path/'policy_metrics.csv',index=False)
    print(metrics.to_string(index=False));print('Policy research only; no production changes.')
if __name__=='__main__':main()
