"""Guardrail for archived pregame XI and player-impact research observations."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import pandas as pd

from .core import utc
from .soccer_impact_collect import read_rows


def audit(confirmed_lineups, impact_rows, now=None):
    current=utc(now or pd.Timestamp.now(tz="UTC"))
    seen=set()
    for category, rows, expected in (
        ("lineups",confirmed_lineups,"api_football_pregame_confirmed_XI"),
        ("impact",impact_rows,"soccer_starter_replacement_research"),
    ):
        for r in rows:
            if r.get("source_id")!=expected or r.get("sport")!="soccer":
                raise ValueError("Unexpected soccer research source")
            captured,kick=utc(r["captured_utc"]),utc(r["kickoff_utc"])
            if captured>current+pd.Timedelta(minutes=5):
                raise ValueError("Soccer research snapshot was captured in the future")
            if captured>=kick-pd.Timedelta(minutes=5 if category=="lineups" else 15):
                raise ValueError("Soccer player research was captured after safe pregame cutoff")
            key=(expected,str(r.get("competition_code")),str(r.get("event_id")),captured.isoformat())
            if key in seen:
                raise ValueError("Duplicate soccer player research snapshot")
            seen.add(key)
            team_ids={str(r.get("home_id") or ""),str(r.get("away_id") or "")}
            if len(team_ids)!=2 or "" in team_ids:
                raise ValueError("Invalid soccer fixture team identity")
            if category=="lineups":
                if r.get("lineup_confirmed") is not True or len(r.get("teams") or [])!=2:
                    raise ValueError("Incomplete soccer starting XI")
                teams_seen=set()
                for team in r["teams"]:
                    tid=str(team.get("espn_team_id") or "")
                    if tid not in team_ids or tid in teams_seen:
                        raise ValueError("Invalid XI team")
                    teams_seen.add(tid)
                    eleven=team.get("startXI") or []
                    if len(eleven)!=11:
                        raise ValueError("Official XI is not complete")
                    ids=[str(p.get("provider_player_id") or "") for p in eleven]
                    if not all(ids) or len(set(ids))!=11:
                        raise ValueError("XI has duplicate or missing players")
            else:
                if r.get("status")!="RESEARCH_ONLY_NOT_MODEL_INPUT":
                    raise ValueError("Research preview was incorrectly promoted")
                if len(r.get("teams") or [])!=2 or {
                    str(t.get("espn_team_id")) for t in r["teams"]}!=team_ids:
                    raise ValueError("Impact preview has invalid fixture teams")
                diff=r.get("home_minus_away_absence_exposure")
                if diff is not None and (
                    not isinstance(diff,(int,float)) or not math.isfinite(diff)
                    or abs(diff)>1000):
                    raise ValueError("Invalid absence exposure difference")
                for team in r["teams"]:
                    for item in team.get("missing_starter_replacement_research") or []:
                        role=item.get("player_role_index")
                        if role is not None and (not isinstance(role,(int,float)) or not 0<=role<=100):
                            raise ValueError("Invalid soccer role index")
                        for cover in item.get("replacement_candidates") or []:
                            c=cover.get("role_index_0_100")
                            if not isinstance(c,(int,float)) or not 0<=c<=100:
                                raise ValueError("Invalid replacement role index")
                b=r.get("model_baseline_asof")
                if b is not None:
                    p=[b.get(k) for k in ("p_H","p_D","p_A")]
                    if (any(not isinstance(x,(int,float)) or not math.isfinite(x)
                            or x<=0 for x in p) or abs(sum(p)-1)>0.02):
                        raise ValueError("Malformed pregame baseline")
                    if utc(b["model_created_utc"])>captured:
                        raise ValueError("A model forecast cannot be from the future")
                if r.get("eligible_for_matched_1x2_evaluation") is True and (b is None or diff is None):
                    raise ValueError("Incomplete paired evaluation evidence")
    return {
        "status":"VALID_SOCCER_PLAYER_IMPACT_RESEARCH",
        "confirmed_pregame_XI_snapshots":len(confirmed_lineups),
        "pregame_player_importance_snapshots":len(impact_rows),
        "model_changed":False,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--archive",default="data/external_signals")
    parser.add_argument("--output",default="reports/external_learning/soccer_impact/audit.json")
    a=parser.parse_args()
    path=Path(a.archive)
    result=audit(read_rows(path/"soccer_confirmed_lineups.jsonl"),
                 read_rows(path/"soccer_starter_impact.jsonl"))
    dest=Path(a.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(json.dumps(result))


if __name__=="__main__":
    main()
