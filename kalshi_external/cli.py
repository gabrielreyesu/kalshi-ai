"""Scheduled external forecast intake and retrospective skill monitoring.

Research-only: data cache is not published to the public repository. Reports
contain only aggregate metrics, not a resale feed of raw third-party odds.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import requests

from .core import historical_scores, load_registry, utc
from .providers import fetch_odds, fetch_football, fetch_licensed_analysts
from .skorbot import fetch_skorbot


def append_history(path, records, max_bytes=25_000_000):
    file=Path(path)
    history=[]
    if file.exists():
        if file.stat().st_size>max_bytes:
            raise ValueError("Forecast history reached storage limit; private archive required")
        with file.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj=json.loads(line)
                    if not isinstance(obj,dict):
                        raise ValueError("Invalid historical forecast JSON")
                    history.append(obj)
    # One archived snapshot per source/event/UTC calendar day; avoids double counting.
    signatures={(r.get("source_id"),r.get("sport"),str(r.get("event_id")),
                 str(r.get("captured_utc",""))[:10]) for r in history}
    added=0
    for r in records:
        key=(r["source_id"],r["sport"],r["event_id"],r["captured_utc"][:10])
        if key not in signatures:
            history.append(r)
            signatures.add(key)
            added+=1
    if added:
        file.parent.mkdir(parents=True,exist_ok=True)
        tmp=file.with_suffix(file.suffix+".tmp")
        with tmp.open("w",encoding="utf8") as out:
            for item in history:
                out.write(json.dumps(item,ensure_ascii=False,separators=(",",":"))+"\n")
        tmp.replace(file)
    return history,added


def load_completed_intl(path):
    file=Path(path)
    if not file.exists():
        return pd.DataFrame(columns=["event_id","sport","result"])
    d=pd.read_csv(file,dtype={"event_id":str})
    required={"event_id","res","status"}
    if not required.issubset(d):
        raise ValueError("Invalid OOS result file")
    # Regulation-only 1X2. Exclude ET/penalties; market settlement could differ.
    d=d[d.status.isin(["STATUS_FULL_TIME","STATUS_FINAL"]) & d.res.isin(["H","D","A"])]
    if "target" in d:
        d=d[d.target.astype(str).str.lower().isin(["true","1"])]
    d=d[["event_id","res"]].rename(columns={"res":"result"}).copy()
    d["sport"]="intl"
    return d.drop_duplicates(["sport","event_id"])


def run(registry_dir="predictions",history_path="data/external_signals/history.jsonl",
        results_path="reports/intl_real_validation/oos_raw.csv",
        reports_dir="reports/external_learning", now=None, session=None):
    now=utc(now or pd.Timestamp.now(tz="UTC"))
    all_parts=[];warn=[]
    for sport,filename in [("intl","intl_registry.csv"),("nfl","nfl_registry.csv"),("nba","nba_registry.csv")]:
        path=Path(registry_dir)/filename
        if not path.exists():
            warn.append(f"No local {sport} prediction registry")
            continue
        try: all_parts.append(load_registry(path,sport))
        except (ValueError,KeyError) as e:
            warn.append(f"Invalid {sport} registry: {type(e).__name__}")
    if not all_parts:
        raise ValueError("No current model prediction registries found")
    reg=pd.concat(all_parts,ignore_index=True)
    s=session or requests.Session()
    fetched=[]
    for function,kwargs in [
        (fetch_odds,{"key":os.environ.get("THE_ODDS_API_KEY","")}),
        (fetch_football,{"key":os.environ.get("API_FOOTBALL_KEY","")}),
    ]:
        r,errors=function(s,reg,now,**kwargs)
        fetched.extend(r);warn.extend(errors)
    r,errors=fetch_skorbot(s,reg,now)
    fetched.extend(r);warn.extend(errors)
    r,errors=fetch_licensed_analysts(s,reg,now,os.environ.get("LICENSED_ANALYST_FEED_URL"),
                                    os.environ.get("LICENSED_ANALYST_FEED_HOST"))
    fetched.extend(r);warn.extend(errors)
    history,added=append_history(history_path,fetched)
    outcomes=load_completed_intl(results_path)
    scored=historical_scores(history,outcomes,min_cases=30)
    dest=Path(reports_dir);dest.mkdir(parents=True,exist_ok=True)
    scored.to_csv(dest/"source_performance.csv",index=False)
    sport_counts={}
    for row in fetched:
        key=row["source_id"]
        sport_counts[key]=sport_counts.get(key,0)+1
    summary={
        "run_utc":now.isoformat(),
        "sources_checked":4,
        "new_records":added,
        "observations_in_private_cache":len(history),
        "new_by_source":sport_counts,
        "sources_with_30_plus_resolved_cases":int(scored.eligible_for_review.sum()) if len(scored) else 0,
        "status":"RESEARCH_ONLY_NOT_USED_TO_MODIFY_PRODUCTION",
        "notes":[
            "Historical source data are kept in Actions cache only; cache is not durable storage.",
            "Only finished 90-minute international OOS results are scored currently.",
            "A source's predictive signal is not promoted without prospective paired validation.",
        ],
        "warnings":warn,
    }
    (dest/"status.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--registry-dir",default="predictions")
    parser.add_argument("--history",default="data/external_signals/history.jsonl")
    parser.add_argument("--results",default="reports/intl_real_validation/oos_raw.csv")
    parser.add_argument("--output",default="reports/external_learning")
    args=parser.parse_args()
    run(args.registry_dir,args.history,args.results,args.output)


if __name__=="__main__":
    main()
