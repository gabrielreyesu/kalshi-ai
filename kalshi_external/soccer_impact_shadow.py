"""Prospective, chronologically held-out soccer player-absence challenger.

Never modifies production. An absence exposure proxy must prove out-of-sample
logloss and Brier improvement with date-cluster uncertainty, and even then
research status is advisory pending independent human review.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import utc
from .soccer_impact_collect import read_rows

ALPHAS=(0.0,0.05,0.10,0.20,0.35,0.50)
CLASSES={"H":0,"D":1,"A":2}


def adjust_probabilities(probabilities, home_advantage, strength):
    base=np.asarray(probabilities,dtype=float)
    if base.shape!=(3,) or not np.isfinite(base).all() or (base<=0).any():
        raise ValueError("Invalid champion 1X2 probabilities")
    if not np.isfinite(home_advantage) or not np.isfinite(strength):
        raise ValueError("Invalid player-absence inputs")
    # Conservative, bounded log-odds change; draw is the unshifted anchor.
    impact=np.clip(float(home_advantage)/100,-1,1)*float(strength)
    q=base*np.exp(np.array([impact,0,-impact]))
    return q/q.sum()


def _paired_frame(history,outcomes,as_of):
    now=utc(as_of)
    if outcomes is None or outcomes.empty:
        return pd.DataFrame()
    required={"sport","event_id","result"}
    if not required.issubset(outcomes):
        raise ValueError("Incomplete resolved results")
    results=outcomes.copy()
    results["event_id"]=results["event_id"].astype(str)
    if results.duplicated(["sport","event_id"]).any():
        raise ValueError("Duplicate results are unsafe")
    labels={}
    for r in results.to_dict("records"):
        labels[(str(r["sport"]),str(r["event_id"]))]=r["result"]
    records={}
    for row in history:
        if (row.get("source_id")!="soccer_starter_replacement_research" or
            row.get("status")!="RESEARCH_ONLY_NOT_MODEL_INPUT"):
            continue
        baseline=row.get("model_baseline_asof")
        difference=row.get("home_minus_away_absence_exposure")
        if not isinstance(baseline,dict) or difference is None:
            continue
        group=baseline.get("model_group")
        if group not in ("club","intl"):
            continue
        key=(group,str(row["event_id"]))
        result=labels.get(key)
        if result not in CLASSES:
            continue
        seen,kick=utc(row["captured_utc"]),utc(row["kickoff_utc"])
        model_seen=utc(baseline["model_created_utc"])
        if seen>=kick-pd.Timedelta(minutes=15) or model_seen>seen or now<kick+pd.Timedelta(hours=2):
            continue
        p=np.array([baseline.get("p_H"),baseline.get("p_D"),baseline.get("p_A")],dtype=float)
        if not np.isfinite(p).all() or (p<=0).any() or abs(p.sum()-1)>0.02:
            continue
        value=float(difference)
        if not np.isfinite(value):
            continue
        prev=records.get(key)
        if prev is not None and utc(prev["captured_utc"])>=seen:
            continue
        records[key]={
            "sport":group,"event_id":str(row["event_id"]),
            "captured_utc":seen.isoformat(),"kickoff_utc":kick.isoformat(),
            "p_H":p[0],"p_D":p[1],"p_A":p[2],
            "absent_role_diff":value,"target":int(CLASSES[result]),
        }
    return pd.DataFrame(records.values())


def evaluate(history,outcomes,as_of=None,min_train=120,min_test=50,
             min_test_dates=20,bootstrap=500,seed=27):
    if min_train<1 or min_test<1 or min_test_dates<1 or bootstrap<0:
        raise ValueError("Invalid evaluation minimum")
    now=utc(as_of or pd.Timestamp.now(tz="UTC"))
    frame=_paired_frame(history,outcomes,now)
    report={"status":"INSUFFICIENT_PROSPECTIVE_OUTCOMES",
            "n_settled_matched":len(frame),
            "n_train":0,"n_test":0,
            "selected_alpha":None,
            "holdout_logloss_gain":None,
            "holdout_brier_gain":None,
            "bootstrap_95pct_logloss_gain":[None,None],
            "candidate_research_only":False,
            "model_promoted":False,
            "warning":"No source can change production from this evaluation. No post-kickoff data used."}
    if frame.empty or len(frame)<min_train+min_test:
        return report
    frame=frame.sort_values(["kickoff_utc","event_id"]).reset_index(drop=True)
    # Chronological cut across entire kickoff DATE, never same-date leakage.
    unique_dates=sorted(pd.to_datetime(frame.kickoff_utc,utc=True).dt.strftime("%Y-%m-%d").unique())
    choices=[]
    for d in unique_dates:
        mask=frame.kickoff_utc.str.slice(0,10)<d
        n=int(mask.sum())
        if n>=min_train and len(frame)-n>=min_test:
            choices.append((n,d))
    if not choices:
        return report
    boundary=max(choices,key=lambda x:x[0])[1]
    train=frame[frame.kickoff_utc.str.slice(0,10)<boundary]
    test=frame[frame.kickoff_utc.str.slice(0,10)>=boundary]
    test_dates=test.kickoff_utc.str.slice(0,10)
    report.update({"n_train":len(train),"n_test":len(test),
                   "holdout_unique_dates":test_dates.nunique()})
    if test_dates.nunique()<min_test_dates:
        return report

    def _loss(batch,alpha):
        p=np.asarray(batch[["p_H","p_D","p_A"]],dtype=float)
        z=np.asarray(batch["absent_role_diff"],dtype=float)
        actual=np.asarray(batch["target"],dtype=int)
        pred=np.stack([adjust_probabilities(row,value,alpha) for row,value in zip(p,z)])
        idx=np.arange(len(actual))
        onehot=np.eye(3)[actual]
        return -np.log(np.clip(pred[idx,actual],1e-12,1)),((pred-onehot)**2).sum(axis=1)
    selected=min(ALPHAS,key=lambda alpha:float(_loss(train,alpha)[0].mean()))
    baseline_ll,baseline_brier=_loss(test,0)
    tuned_ll,tuned_brier=_loss(test,selected)
    gain=baseline_ll-tuned_ll
    brier_gain=baseline_brier-tuned_brier
    low=high=None
    unique=test_dates.unique()
    if bootstrap and len(unique)>1:
        group_indices=[np.flatnonzero(test_dates.to_numpy()==d) for d in unique]
        rng=np.random.default_rng(seed)
        samples=np.empty(bootstrap)
        for i in range(bootstrap):
            ix=np.concatenate([group_indices[j] for j in
                               rng.integers(len(group_indices),size=len(group_indices))])
            samples[i]=gain[ix].mean()
        low,high=map(float,np.percentile(samples,[2.5,97.5]))
    good=bool(selected>0 and low is not None and low>0
              and float(brier_gain.mean())>0)
    report.update({
        "status":"RESEARCH_ONLY_PAIRED_HOLDOUT",
        "selected_alpha":float(selected),
        "holdout_logloss_gain":float(gain.mean()),
        "holdout_brier_gain":float(brier_gain.mean()),
        "baseline_holdout_logloss":float(baseline_ll.mean()),
        "candidate_holdout_logloss":float(tuned_ll.mean()),
        "bootstrap_95pct_logloss_gain":[low,high],
        "candidate_research_only":good,
        "model_promoted":False,
    })
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--history",default="data/external_signals/soccer_starter_impact.jsonl")
    parser.add_argument("--outcomes",default="reports/external_learning/resolved_outcomes.csv")
    parser.add_argument("--output",default="reports/external_learning/soccer_impact/shadow_results.json")
    a=parser.parse_args()
    h=read_rows(a.history)
    o=pd.read_csv(a.outcomes,dtype={"event_id":str}) if Path(a.outcomes).exists() else pd.DataFrame(columns=["sport","event_id","result"])
    result=evaluate(h,o)
    dest=Path(a.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(json.dumps(result))


if __name__=="__main__":
    main()
