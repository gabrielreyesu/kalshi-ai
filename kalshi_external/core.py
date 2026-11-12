"""Time-safe normalization and matching of third-party sports forecasts.

All forecast signals are research-only. Nothing here changes production predictions.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable
import numpy as np
import pandas as pd

ALIASES = {
    "turkiye": "turkey", "republic of ireland": "ireland",
    "czech republic": "czechia", "korea republic": "south korea",
    "united states": "usa", "united states of america": "usa",
    "ivory coast": "cote divoire", "cote d ivoire": "cote divoire",
}

def utc(value):
    dt = pd.Timestamp(value)
    if pd.isna(dt):
        raise ValueError("Invalid timestamp")
    return dt.tz_localize("UTC") if dt.tzinfo is None else dt.tz_convert("UTC")

def team_name(value):
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
    return ALIASES.get(text, text)

def load_registry(path, sport):
    d = pd.read_csv(path, dtype={"event_id": str})
    required = {"event_id", "kickoff_utc", "home", "away", "created_utc"}
    if not required.issubset(d):
        raise ValueError(f"Missing registry columns: {sorted(required-set(d))}")
    if sport == "intl":
        d = d[d.get("grupo", pd.Series("",index=d.index)).eq("selecciones")]
    d = d.drop_duplicates("event_id", keep="first").copy()
    d["sport"] = sport
    return d

def future_match(registry, sport, home, away, kickoff, captured, tolerance_minutes=60):
    """Require a unique two-team match at a tightly matching kickoff time."""
    kick, seen = utc(kickoff), utc(captured)
    if seen >= kick - pd.Timedelta(minutes=15):
        return None
    d = registry[registry.sport == sport]
    if d.empty:
        return None
    ko = pd.to_datetime(d.kickoff_utc, utc=True, errors="coerce")
    delta = (ko - kick).abs()
    found = d.loc[(delta <= pd.Timedelta(minutes=tolerance_minutes)) &
                  d.home.map(team_name).eq(team_name(home)) &
                  d.away.map(team_name).eq(team_name(away))]
    if len(found) != 1:
        return None
    record = found.iloc[0]
    if utc(record.created_utc) > seen:
        return None
    return record

def make_record(registry_match, source_id, captured, probs, published=None, source_kind="analyst"):
    if not source_id or len(source_id) > 90:
        raise ValueError("Invalid source identifier")
    kick = utc(registry_match.kickoff_utc)
    capture = utc(captured)
    publication = utc(published) if published is not None else capture
    if publication > capture or capture >= kick - pd.Timedelta(minutes=15):
        raise ValueError("Source forecast was not verifiably available pre-match")
    if utc(registry_match.created_utc) > capture:
        raise ValueError("Original model prediction was created after snapshot")
    p = np.asarray(probs, dtype=float)
    n = 3 if registry_match.sport == "intl" else 2
    if p.shape != (n,) or not np.isfinite(p).all() or np.any(p < 0) or p.sum() <= 0:
        raise ValueError("Invalid probability vector")
    p = p / p.sum()
    result = {"event_id": str(registry_match.event_id), "sport": str(registry_match.sport),
              "source_id": str(source_id), "source_kind": source_kind,
              "kickoff_utc": kick.isoformat(), "captured_utc": capture.isoformat(),
              "published_utc": publication.isoformat(), "home": str(registry_match.home),
              "away": str(registry_match.away), "p_H": float(p[0]),
              "p_D": float(p[1]) if n == 3 else None,
              "p_A": float(p[-1])}
    if n == 3:
        own = [registry_match.get("p_H"),registry_match.get("p_D"),registry_match.get("p_A")]
    else:
        own = [registry_match.get("p_home"),1-float(registry_match.get("p_home"))]
    own = np.asarray(own, dtype=float)
    if own.shape != (n,) or not np.isfinite(own).all() or np.any(own < 0) or own.sum() <= 0:
        raise ValueError("Missing valid pre-match baseline")
    own /= own.sum()
    result.update({"model_p_H":float(own[0]),"model_p_D":float(own[1]) if n==3 else None,
                   "model_p_A":float(own[-1])})
    return result

def historical_scores(rows: Iterable[dict], outcomes: pd.DataFrame, min_cases=30):
    """Evaluate only distinct match/source pairs with known final outcomes."""
    forecasts=pd.DataFrame(rows)
    empty=pd.DataFrame(columns=["source_id","sport","n","source_logloss","model_logloss",
                                "gain_vs_model","source_brier","model_brier","eligible_for_review"])
    if forecasts.empty:
        return empty
    required={"source_id","sport","event_id","p_H","p_D","p_A","model_p_H","model_p_D","model_p_A",
              "captured_utc","published_utc","kickoff_utc"}
    if not required.issubset(forecasts):
        raise ValueError("Source history missing required fields")
    f=forecasts.copy()
    for col in ("captured_utc","published_utc","kickoff_utc"):
        f[col]=pd.to_datetime(f[col],utc=True,errors="coerce")
    if f[["captured_utc","published_utc","kickoff_utc"]].isna().any().any():
        raise ValueError("History contains invalid timestamps")
    f=f[(f.published_utc<=f.captured_utc)&(f.captured_utc<f.kickoff_utc-pd.Timedelta(minutes=15))]
    f=f.sort_values("captured_utc").drop_duplicates(["source_id","sport","event_id"],keep="last")
    required_out={"event_id","sport","result"}
    if not required_out.issubset(outcomes):
        raise ValueError("Outcomes are missing identifier/sport/result")
    o=outcomes[list(required_out)].copy()
    if o.duplicated(["event_id","sport"]).any():
        raise ValueError("Duplicate outcomes")
    o["event_id"]=o.event_id.astype(str)
    f["event_id"]=f.event_id.astype(str)
    joined=f.merge(o,on=["event_id","sport"],how="inner",validate="many_to_one")
    metrics=[]
    for (source,sport),d in joined.groupby(["source_id","sport"]):
        labels={"H":0,"D":1,"A":2} if sport=="intl" else {"H":0,"A":1}
        idx=d.result.map(labels)
        if idx.isna().any(): continue
        cols=["p_H","p_D","p_A"] if sport=="intl" else ["p_H","p_A"]
        base=["model_p_H","model_p_D","model_p_A"] if sport=="intl" else ["model_p_H","model_p_A"]
        p=d[cols].to_numpy(float); q=d[base].to_numpy(float)
        if not np.isfinite(p).all() or not np.isfinite(q).all():continue
        if (p<0).any() or (q<0).any() or (p.sum(axis=1)<=0).any() or (q.sum(axis=1)<=0).any():
            continue
        y=idx.to_numpy(dtype=int)
        p=p/p.sum(axis=1,keepdims=True);q=q/q.sum(axis=1,keepdims=True)
        oh=np.eye(len(cols))[y]
        ll=lambda a:float(-np.log(np.maximum(a[np.arange(len(y)),y],1e-12)).mean())
        br=lambda a:float(((a-oh)**2).sum(axis=1).mean())
        source_ll, model_ll=ll(p),ll(q)
        metrics.append({"source_id":source,"sport":sport,"n":len(y),
                        "source_logloss":source_ll,"model_logloss":model_ll,
                        "gain_vs_model":model_ll-source_ll,
                        "source_brier":br(p),"model_brier":br(q),
                        "eligible_for_review":bool(len(y)>=min_cases)})
    return pd.DataFrame(metrics) if metrics else empty
