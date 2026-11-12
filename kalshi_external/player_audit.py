"""Fail-closed data-quality checks for private NBA/NFL player archives."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from .player_intel import timestamp, MEASURES


def load_rows(path):
    p = Path(path)
    if not p.exists():
        return []
    rows=[]
    for i, line in enumerate(p.read_text(encoding="utf8").splitlines(),1):
        if not line.strip():
            continue
        try:
            item=json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Bad JSON at {p.name}:{i}") from error
        if not isinstance(item,dict):
            raise ValueError("Expected player snapshot object")
        rows.append(item)
    return rows


def audit(injuries, boxscores):
    keys=set()
    count={"nba":0,"nfl":0}
    unique_players=set()
    for category, records, expected in (
        ("injuries",injuries,"espn_official_injury_feed"),
        ("boxscores",boxscores,"espn_completed_player_boxscore"),
    ):
        if not isinstance(records,list):
            raise ValueError("Player archives must be JSONL lists")
        for row in records:
            if row.get("source_id")!=expected or row.get("sport") not in MEASURES:
                raise ValueError("Unexpected player record source or sport")
            identifier=(row["source_id"],row["sport"],str(row["event_id"]),row["captured_utc"])
            if identifier in keys:
                raise ValueError("Duplicate player archive snapshot")
            keys.add(identifier)
            capture=timestamp(row["captured_utc"])
            if capture>pd.Timestamp.now(tz="UTC")+pd.Timedelta(minutes=5):
                raise ValueError("Future player snapshot")
            if category=="injuries":
                seen=set()
                for a in row.get("reported_athletes") or []:
                    team_id=str(a.get("team_id") or "")
                    player_id=str(a.get("player_id") or "")
                    name=str(a.get("player") or "")
                    if not team_id or not (player_id or name):
                        raise ValueError("Injury item lacks team or athlete")
                    ident=(team_id,player_id or name.casefold())
                    if ident in seen:
                        raise ValueError("Duplicate injury-listed player")
                    seen.add(ident)
                    if a.get("reported_utc") and timestamp(a["reported_utc"])>capture:
                        raise ValueError("Injury report dated after capture")
                    unique_players.add((row["sport"],team_id,player_id or name))
            else:
                kickoff=timestamp(row["kickoff_utc"])
                if kickoff>=capture:
                    raise ValueError("Postgame boxscore captured before kickoff")
                if not row.get("home_id") or not row.get("away_id"):
                    raise ValueError("Missing boxscore teams")
                seen=set()
                for a in row.get("players") or []:
                    player_id=str(a.get("player_id") or "")
                    team_id=str(a.get("team_id") or "")
                    opponent_id=str(a.get("opponent_id") or "")
                    if (not player_id or
                        {team_id,opponent_id}!={str(row["home_id"]),str(row["away_id"])}):
                        raise ValueError("Bad team/athlete identity in boxscore")
                    if (team_id,player_id) in seen:
                        raise ValueError("Duplicate player in game boxscore")
                    seen.add((team_id,player_id))
                    for metric,value in (a.get("metrics") or {}).items():
                        if metric not in MEASURES[row["sport"]]:
                            raise ValueError("Unexpected player stat type")
                        if (not isinstance(value,(float,int)) or not np.isfinite(value) or
                            value < (-200 if metric.endswith("_yards") else 0) or value > 1500):
                            raise ValueError("Invalid player stat")
                    unique_players.add((row["sport"],team_id,player_id))
            count[row["sport"]]+=1
    return {
        "status":"VALID_PLAYER_RESEARCH_NO_PRODUCTION_CHANGE",
        "injury_batches":len(injuries),
        "completed_game_boxscores":len(boxscores),
        "athlete_team_pairs_seen":len(unique_players),
        "by_sport_record_count":count,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--injuries",default="data/external_signals/player_injuries.jsonl")
    parser.add_argument("--boxscores",default="data/external_signals/player_boxscores.jsonl")
    parser.add_argument("--output",default="reports/external_learning/player_context/audit.json")
    a=parser.parse_args()
    result=audit(load_rows(a.injuries),load_rows(a.boxscores))
    target=Path(a.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(json.dumps(result))


if __name__=="__main__":
    main()
