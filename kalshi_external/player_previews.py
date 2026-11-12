"""Research-only per-game player context from private pregame and completed game archives."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from .cli import append_history
from .player_intel import player_matchup_context, timestamp


def _read(path):
    p=Path(path)
    if not p.exists():
        return []
    rows=[json.loads(line) for line in p.read_text(encoding="utf8").splitlines() if line.strip()]
    if any(not isinstance(x,dict) for x in rows):
        raise ValueError("Invalid private player archive")
    return rows


def build(registry, injury_history, box_history, observed, per_team=12):
    if not 1 <= per_team <= 30:
        raise ValueError("Invalid player limit")
    now=timestamp(observed)
    output=[]
    for sport, df in registry.items():
        if sport not in ("nba","nfl") or df is None or df.empty:
            continue
        needed={"event_id","home_id","away_id","home","away","kickoff_utc"}
        if not needed.issubset(df):
            raise ValueError("Registry missing canonical identifiers")
        for game in df.to_dict("records"):
            kickoff=timestamp(game["kickoff_utc"])
            if not now+pd.Timedelta(minutes=15)<kickoff<=now+pd.Timedelta(hours=48):
                continue
            context=player_matchup_context({**game,"sport":sport},box_history,injury_history,now)
            teams=[]
            for team_id,name in ((str(game["home_id"]),game["home"]),
                                 (str(game["away_id"]),game["away"])):
                players=[p for p in context["players"] if p["team_id"]==team_id]
                def rank(p):
                    metric="points" if sport=="nba" else "passing_yards"
                    score=p["metrics"].get(metric,{}).get("recent_mean") or 0
                    if sport=="nfl" and not score:
                        score=p["metrics"].get("receiving_yards",{}).get("recent_mean") or 0
                    return (p["availability"] in ("reported_out","uncertain"),score,p["prior_games"])
                players.sort(key=rank,reverse=True)
                teams.append({
                    "team_id":team_id,"team":str(name),
                    "player_context":players[:per_team],
                    "players_with_prior_games":sum(p["prior_games"]>0 for p in players),
                    "known_reported_out":sum(p["availability"]=="reported_out" for p in players),
                    "listed_as_uncertain":sum(p["availability"]=="uncertain" for p in players),
                })
            output.append({
                "source_id":"espn_player_pregame_context",
                "source_kind":"player_context_research",
                "sport":sport,"event_id":str(game["event_id"]),
                "captured_utc":now.isoformat(),"kickoff_utc":kickoff.isoformat(),
                "injury_snapshot_utc":context["injury_snapshot_utc"],
                "injury_data_present":context["injury_data_present"],
                "teams":teams,
                "note":"Research only: no causal effects or missing-player availability assumed.",
            })
    return output


def run(registry_dir="predictions",
        injuries_path="data/external_signals/player_injuries.jsonl",
        boxscores_path="data/external_signals/player_boxscores.jsonl",
        previews_path="data/external_signals/player_previews.jsonl",
        output="reports/external_learning/player_context/preview_coverage.json",
        observed_at=None):
    now=timestamp(observed_at or pd.Timestamp.now(tz="UTC"))
    registry={}
    for sport in ("nba","nfl"):
        p=Path(registry_dir)/f"{sport}_registry.csv"
        if p.exists():
            registry[sport]=pd.read_csv(p,dtype={"event_id":str,"home_id":str,"away_id":str})
    contexts=build(registry,_read(injuries_path),_read(boxscores_path),now)
    _,new=append_history(previews_path,contexts,max_bytes=80_000_000)
    report={
        "captured_utc":now.isoformat(),
        "new_pregame_contexts":new,
        "events":len(contexts),
        "events_with_recent_injury_report":sum(r["injury_data_present"] for r in contexts),
        "events_with_player_history":sum(
            any(t["players_with_prior_games"]>0 for t in r["teams"]) for r in contexts),
        "status":"RESEARCH_ONLY_NO_PRODUCTION_EFFECT",
        "notes":["Detailed individual player contexts are stored only in the private archive.",
                 "Nonlisted athletes have UNKNOWN game availability.",
                 "Opponent splits are heavily shrunk descriptive signals, not win-rate adjustments."],
    }
    p=Path(output)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report))
    return report


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",default="reports/external_learning/player_context/preview_coverage.json")
    a=ap.parse_args()
    run(output=a.output)
