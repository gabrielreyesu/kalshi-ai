"""Chronological evaluation of research blends against the original model.

Only permissioned source kinds are fit; no production promotion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cli import load_completed_intl

# Public read-only observation sources are intentionally excluded from fitting.
TRAINABLE_KINDS={"market","analyst"}
WEIGHTS=(0.0,0.10,0.20,0.30,0.40)


def _loss(p,y):
    return float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean())


def fit_blend_candidates(history,outcomes,min_train=60,min_test=25,seed=23,bootstrap=500):
    """Train on earliest resolved dates; evaluate later matches, per source."""
    if bootstrap < 0 or min_train < 30 or min_test < 10:
        raise ValueError("Invalid thresholds")
    result=[]
    if not history:
        return result
    d=pd.DataFrame(history).copy()
    needed={"source_id","source_kind","sport","event_id","kickoff_utc",
            "captured_utc","published_utc","p_H","p_D","p_A",
            "model_p_H","model_p_D","model_p_A"}
    if not needed.issubset(d):
        raise ValueError("Invalid source history")
    d=d[d.source_kind.isin(TRAINABLE_KINDS)&d.sport.eq("intl")].copy()
    if d.empty:return result
    for c in ("kickoff_utc","captured_utc","published_utc"):
        d[c]=pd.to_datetime(d[c],utc=True,errors="coerce")
    if d[["kickoff_utc","captured_utc","published_utc"]].isna().any().any():
        raise ValueError("Invalid pre-match timestamps")
    d=d[(d.published_utc<=d.captured_utc)&
        (d.captured_utc < d.kickoff_utc-pd.Timedelta(minutes=15))]
    d=d.sort_values("captured_utc").drop_duplicates(["source_id","event_id"],keep="last")
    o=outcomes.copy()
    if not {"event_id","sport","result"}.issubset(o):
        raise ValueError("Outcomes missing required columns")
    o=o[o.sport=="intl"].copy()
    if o.event_id.duplicated().any():
        raise ValueError("Duplicate completed events")
    d.event_id=d.event_id.astype(str)
    o.event_id=o.event_id.astype(str)
    d=d.merge(o[["event_id","result"]],on="event_id",how="inner",validate="many_to_one")
    for source, group in d.groupby("source_id"):
        g=group.sort_values(["kickoff_utc","event_id"]).reset_index(drop=True)
        labels=g.result.map({"H":0,"D":1,"A":2})
        if labels.isna().any():
            continue
        # All games on a day go into the SAME split to prevent contamination.
        days=g.kickoff_utc.dt.normalize()
        unique_days=days.drop_duplicates().sort_values().to_list()
        if len(unique_days)<2:continue
        boundary=unique_days[max(1,int(len(unique_days)*0.70))]
        train=(days<boundary).to_numpy()
        test=~train
        if sum(train)<min_train or sum(test)<min_test:
            continue
        source_p=g[["p_H","p_D","p_A"]].to_numpy(float)
        model_p=g[["model_p_H","model_p_D","model_p_A"]].to_numpy(float)
        if not np.isfinite(source_p).all() or not np.isfinite(model_p).all():
            continue
        if (source_p<0).any() or (model_p<0).any():
            continue
        source_p/=source_p.sum(axis=1,keepdims=True)
        model_p/=model_p.sum(axis=1,keepdims=True)
        y=labels.to_numpy(dtype=int)
        training_scores={}
        for w in WEIGHTS:
            q=(1-w)*model_p[train]+w*source_p[train]
            training_scores[w]=_loss(q,y[train])
        weight=min(WEIGHTS,key=lambda w:(training_scores[w],w))
        blended=(1-weight)*model_p[test]+weight*source_p[test]
        base=model_p[test]
        yy=y[test]
        diff=-np.log(np.maximum(blended[np.arange(len(yy)),yy],1e-12))+np.log(
            np.maximum(base[np.arange(len(yy)),yy],1e-12))
        groups=pd.DataFrame({"day":days[test].to_numpy(),"ix":np.arange(len(yy))}).groupby("day")["ix"].apply(lambda s:s.to_numpy())
        groups=list(groups)
        if bootstrap and len(groups)>1:
            rng=np.random.default_rng(seed)
            draws=[]
            for _ in range(bootstrap):
                ids=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))])
                draws.append(float(diff[ids].mean()))
            lo,hi=np.percentile(draws,[2.5,97.5])
        else:
            lo=hi=np.nan
        result.append({
            "source_id":source,"candidate_weight":weight,
            "train_before_utc":pd.Timestamp(boundary).isoformat(),
            "n_train":int(sum(train)),"n_test":int(sum(test)),
            "test_baseline_logloss":_loss(base,yy),
            "test_candidate_logloss":_loss(blended,yy),
            "test_candidate_minus_baseline":float(diff.mean()),
            "ci95_low":float(lo),"ci95_high":float(hi),
            "status":"RESEARCH_ONLY_NO_AUTOMATIC_PRODUCTION_PROMOTION"
        })
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--history",default="data/external_signals/history.jsonl")
    parser.add_argument("--results",default="reports/intl_real_validation/oos_raw.csv")
    parser.add_argument("--output",default="reports/external_learning/candidate_weights.json")
    args=parser.parse_args()
    cache=Path(args.history)
    rows=[]
    if cache.exists():
        with cache.open(encoding="utf8") as f:
            rows=[json.loads(line) for line in f if line.strip()]
    outcomes=load_completed_intl(args.results)
    candidates=fit_blend_candidates(rows,outcomes)
    p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({"candidates":candidates,"production_changed":False,
                             "status":"RESEARCH_ONLY"},indent=2)+"\n",encoding="utf8")
    print(f"{len(candidates)} candidate blend(s) evaluated. No production changes.")


if __name__=="__main__":
    main()
