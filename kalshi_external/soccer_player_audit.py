"""Quality gate for private soccer player research archives.

A failed audit blocks saving to the durable archive; coverage may be zero.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import math
import pandas as pd
from .core import utc
from .soccer_player_intel import FINAL, SOCCER_MEASURES


SOURCES = {
    "games":("espn_soccer_final_player_stats","historical_player_stats"),
    "lineups":("espn_soccer_pregame_roster","soccer_pregame_roster"),
    "injuries":("api_football_soccer_injuries","soccer_provider_injuries"),
    "previews":("soccer_player_opponent_preview","player_context_research"),
}


def read_jsonl(path):
    p=Path(path)
    if not p.exists():
        return []
    result=[]
    for n,line in enumerate(p.read_text(encoding="utf8").splitlines(),1):
        if not line.strip():
            continue
        try:
            x=json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid soccer JSONL in {p.name} line {n}") from e
        if not isinstance(x,dict):
            raise ValueError("Non-object soccer player record")
        result.append(x)
    return result


def audit(games,lineups,injuries,previews,now=None):
    check=utc(now or pd.Timestamp.now(tz="UTC"))
    counts={}
    keyset=set()
    player_ids=set()
    for category,rows in (("games",games),("lineups",lineups),
                          ("injuries",injuries),("previews",previews)):
        if not isinstance(rows,list):
            raise ValueError("Archive is not a list")
        src,kind=SOURCES[category]
        counts[category]=len(rows)
        for r in rows:
            if (r.get("source_id")!=src or r.get("source_kind")!=kind or
                r.get("sport")!="soccer" or not r.get("event_id")):
                raise ValueError("Unexpected soccer player source")
            captured=utc(r["captured_utc"])
            if captured>check+pd.Timedelta(minutes=5):
                raise ValueError("Future-dated soccer capture")
            key=(src,str(r.get("competition_code") or ""),str(r["event_id"]),captured.isoformat())
            if key in keyset:
                raise ValueError("Duplicate soccer player snapshot")
            keyset.add(key)
            if category=="injuries":
                # observed_day_utc is the *target fixture day*, not the
                # capture day. Reports for tomorrow are valid today.
                target_day=pd.Timestamp(str(r.get("observed_day_utc")),tz="UTC")
                if not (captured.normalize()<=target_day<=
                        captured.normalize()+pd.Timedelta(days=2)):
                    raise ValueError("Injury fixture date is not an upcoming day")
                seen=set()
                for p in r.get("players") or []:
                    kick=utc(p["fixture_kickoff_utc"])
                    if kick<=captured+pd.Timedelta(minutes=5):
                        raise ValueError("Injury report includes already started match")
                    ids=(str(p.get("fixture_id")),str(p.get("provider_team_id")),
                         str(p.get("provider_player_id") or p.get("player")))
                    if not all(ids) or ids in seen:
                        raise ValueError("Invalid or duplicate provider injury")
                    seen.add(ids)
                    if p.get("availability") not in ("reported_absent","uncertain","unknown_provider_status"):
                        raise ValueError("Unexpected injury status")
                continue
            kick=utc(r["kickoff_utc"])
            if category=="games" and captured<=kick:
                raise ValueError("Completed player stats captured before kickoff")
            if category in ("lineups","previews") and captured>=kick-pd.Timedelta(minutes=5):
                raise ValueError("Pregame soccer record captured too late")
            if category in ("games","lineups"):
                if category=="games" and r.get("final_status") not in FINAL:
                    raise ValueError("Game player stats without final status")
                if category=="lineups" and r.get("lineup_confirmed") is not False:
                    raise ValueError("Unverified lineup incorrectly marked confirmed")
                teams={str(r.get("home_id")),str(r.get("away_id"))}
                if len(teams)!=2 or any(not x or x=="None" for x in teams):
                    raise ValueError("Invalid ESPN team IDs")
                seen=set()
                has_metric=False
                for t in r.get("teams") or []:
                    tid=str(t.get("team_id"))
                    if tid not in teams:
                        raise ValueError("Unverified soccer team")
                    if str(t.get("opponent_id")) not in teams-{tid}:
                        raise ValueError("Invalid opponent mapping")
                    for p in t.get("members") or []:
                        pid=str(p.get("player_id") or "")
                        if not pid or (tid,pid) in seen:
                            raise ValueError("Missing or duplicated ESPN player ID")
                        seen.add((tid,pid))
                        player_ids.add((tid,pid))
                        for metric,value in (p.get("metrics") or {}).items():
                            if metric not in SOCCER_MEASURES or not isinstance(value,(int,float)):
                                raise ValueError("Unknown soccer player metric")
                            if not math.isfinite(value) or not 0<=value<=150:
                                raise ValueError("Invalid soccer player stat value")
                            has_metric=True
                if category=="games" and not has_metric:
                    raise ValueError("Historical player game contains no statistics")
            else:
                if not r.get("home_id") or not r.get("away_id"):
                    raise ValueError("Soccer preview missing canonical team ID")
                if r.get("lineup_confirmed") is True:
                    raise ValueError("Unverified preview marked lineup confirmed")
    return {
        "archive_status":"VALID_SOCCER_RESEARCH_NO_MODEL_CHANGE",
        "record_counts":counts,"unique_espn_player_team_pairs":len(player_ids),
        "no_betting_or_probability_changes":True,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--archive",default="data/external_signals")
    parser.add_argument("--output",default="reports/external_learning/soccer_players/audit.json")
    a=parser.parse_args()
    p=Path(a.archive)
    result=audit(
        read_jsonl(p/"soccer_player_games.jsonl"),
        read_jsonl(p/"soccer_lineups.jsonl"),
        read_jsonl(p/"soccer_injuries.jsonl"),
        read_jsonl(p/"soccer_previews.jsonl"),
    )
    dest=Path(a.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(json.dumps(result))


if __name__=="__main__":
    main()
