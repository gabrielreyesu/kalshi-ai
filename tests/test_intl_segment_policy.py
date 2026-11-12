import numpy as np
import pandas as pd
from kalshi_intl.segment_policy import evaluate_policy

def fixtures():
    rows=[]
    for season in [2021,2022,2023]:
        for j in range(160):
            home = j%2==0
            rows.append(dict(temporada=f'{season}/{str((season+1)%100).zfill(2)}',date=f'{season}-08-{1+j%20:02d}',res='H' if home else 'A',friendly=j%3==0,p_H=.55,p_D=.1,p_A=.35,cal_H=.75,cal_D=.1,cal_A=.15))
    return pd.DataFrame(rows)

def test_policy_does_not_see_future():
    d=fixtures()
    a,_,_=evaluate_policy(d,min_history=30,min_gain=0)
    changed=d.copy();changed.loc[changed.temporada.str.startswith('2023'),'res']='D'
    b,_,_=evaluate_policy(changed,min_history=30,min_gain=0)
    assert (a.loc[a.season_start==2023,'policy'].to_numpy()==b.loc[b.season_start==2023,'policy'].to_numpy()).all()

def test_policy_valid_probabilities():
    d,decisions,metrics=evaluate_policy(fixtures())
    assert np.allclose(d[['policy_H','policy_D','policy_A']].sum(axis=1),1)
    assert len(decisions)==6 and len(metrics)==3
