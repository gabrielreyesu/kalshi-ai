"""Scheduled ESPN player research snapshots for NBA and NFL.

Capture a timestamped public injury report and a small budget of *completed*
game boxscores. All player-level records live only in a private GitHub archive.
Does not modify predictions, make bets, or infer that an unlisted player plays.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from .cli import append_history
from .player_intel import parse_injury_feed, parse_completed_boxscore, player_matchup_context, timestamp

API_ROOT = "https://site.api.espn.com/apis/site/v2/sports"
PATHS = {"nba": "basketball/nba", "nfl": "football/nfl"}
UA = {"User-Agent": "KalshiAI-player-research/1.0 (+https://github.com/gabrielreyesu/kalshi-ai)"}


def _get(session, url, *, params=None):
    p = urlparse(url)
    if (p.scheme != "https" or p.hostname != "site.api.espn.com" or
        not any(p.path == f"/apis/site/v2/sports/{sub}/{part}"
                for sub in PATHS.values() for part in ("injuries","scoreboard","summary"))):
        raise ValueError("Unexpected ESPN data endpoint")
    r = session.get(url, params=params, timeout=18, headers=UA, allow_redirects=False)
    if r.status_code != 200 or len(r.content) > 3_000_000:
        raise RuntimeError(f"ESPN API response unavailable: HTTP {r.status_code}")
    response = r.json()
    if not isinstance(response, dict):
        raise ValueError("ESPN response not an object")
    return response


def _load(path):
    p = Path(path)
    if not p.exists():
        return []
    records = []
    for line in p.read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError("Invalid JSONL record")
        records.append(record)
    return records


def _fetch_injuries(session, sport, observed_at):
    response = _get(session, f"{API_ROOT}/{PATHS[sport]}/injuries")
    return parse_injury_feed(response, sport, observed_at)


def _recent_completed(session, sport, observed_at, history, days=3,
                      max_new_games=6, pause_seconds=0.1):
    """Bound request volume; do not try to invent retroactive as-of injuries."""
    if days < 1 or days > 14 or max_new_games < 1 or max_new_games > 20:
        raise ValueError("Invalid collection budget")
    seen = {str(g["event_id"]) for g in history
            if g.get("sport")==sport and g.get("source_id")=="espn_completed_player_boxscore"}
    now = timestamp(observed_at)
    ready = []
    for offset in range(days):
        day = (now-pd.Timedelta(days=offset)).strftime("%Y%m%d")
        response = _get(session, f"{API_ROOT}/{PATHS[sport]}/scoreboard",
                        params={"dates": day})
        for event in response.get("events") or []:
            event_id = str(event.get("id") or "")
            state = str(((event.get("status") or {}).get("type") or {}).get("name") or "")
            start = event.get("date")
            if not (event_id.isdigit() and state == "STATUS_FINAL" and start):
                continue
            if timestamp(start) >= now or event_id in seen:
                continue
            ready.append((timestamp(start),event_id))
    records = []
    failures = []
    # Oldest first: reduces long-term missing-game bias when request budgets
    # prevent fetching every finished game in a busy NBA schedule.
    for _,event_id in sorted(set(ready))[:max_new_games]:
        try:
            if pause_seconds:
                time.sleep(pause_seconds)
            detail = _get(session, f"{API_ROOT}/{PATHS[sport]}/summary",
                          params={"event": event_id})
            record = parse_completed_boxscore(detail,sport,event_id,now)
            if record is not None:
                records.append(record)
            else:
                failures.append(f"{sport} event {event_id}: no verified final player boxscore")
        except (requests.RequestException, ValueError, RuntimeError, TypeError, KeyError) as err:
            failures.append(f"{sport} event {event_id}: {type(err).__name__}")
    return records,failures,len(set(ready))


def run(injuries_path="data/external_signals/player_injuries.jsonl",
        boxscores_path="data/external_signals/player_boxscores.jsonl",
        report_dir="reports/external_learning/player_context",
        registry_dir="predictions", observed_at=None, session=None, days=3,
        max_new_games=6, pause_seconds=0.1):
    now = timestamp(observed_at or pd.Timestamp.now(tz="UTC"))
    s = session or requests.Session()
    injuries_new, games_new, issues, candidates = [],[],[],{}
    prior_injuries = _load(injuries_path)
    prior_games = _load(boxscores_path)
    for sport in PATHS:
        try:
            injuries_new.append(_fetch_injuries(s,sport,now))
        except (requests.RequestException,ValueError,RuntimeError,KeyError,TypeError) as err:
            issues.append(f"{sport} injury report: {type(err).__name__}")
        try:
            batch,warn,n = _recent_completed(s,sport,now,prior_games,days=days,
                                              max_new_games=max_new_games,
                                              pause_seconds=pause_seconds)
            games_new.extend(batch)
            issues.extend(warn)
            candidates[sport] = n
        except (requests.RequestException,ValueError,RuntimeError,KeyError,TypeError) as err:
            issues.append(f"{sport} completed games: {type(err).__name__}")
            candidates[sport] = 0
    all_injuries, n_injuries = append_history(injuries_path,injuries_new,max_bytes=100_000_000)
    all_games, n_games = append_history(boxscores_path,games_new,max_bytes=100_000_000)
    # Only aggregate health stats are committed or put in public artifact paths.
    summaries = []
    for sport in PATHS:
        last = [r for r in all_injuries if r.get("sport")==sport]
        latest = max(last,key=lambda r: timestamp(r["captured_utc"])) if last else None
        statuses = {}
        if latest:
            for p in latest.get("reported_athletes") or []:
                name = str(p.get("status") or "unknown").lower()
                statuses[name] = statuses.get(name,0)+1
        summaries.append({
            "sport":sport,
            "latest_injury_snapshot_utc":latest["captured_utc"] if latest else None,
            "players_in_latest_injury_report":len(latest["reported_athletes"]) if latest else 0,
            "reported_status_counts":statuses,
            "historical_games_with_player_stats":len({
                g["event_id"] for g in all_games if g.get("sport")==sport
            }),
            "eligible_new_completed_game_candidates":candidates.get(sport,0),
        })
    # Matchup context stays in the PRIVATE archive and is never sent to a
    # probability model automatically. A compact availability report includes
    # names/statuses only in the private archive, not public GH artifacts.
    report = {
        "observed_utc":now.isoformat(), "source":"ESPN site API public injury/boxscores",
        "new_injury_snapshot_batches":n_injuries,
        "new_completed_game_boxscores":n_games,
        "sports":summaries,
        "warnings":issues[:40],
        "status":"PLAYER_RESEARCH_ONLY_NO_PREDICTION_CHANGES",
        "notes":["Unknown/unlisted never means healthy or active.",
                 "Opponent splits are retrospective associations, not causal wins.",
                 "The ESPN public API may be incomplete or change schema."],
    }
    dest = Path(report_dir)
    dest.mkdir(parents=True,exist_ok=True)
    (dest/"coverage.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print(json.dumps(report,ensure_ascii=False))
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--injuries",default="data/external_signals/player_injuries.jsonl")
    ap.add_argument("--boxscores",default="data/external_signals/player_boxscores.jsonl")
    ap.add_argument("--output",default="reports/external_learning/player_context")
    ap.add_argument("--days",type=int,default=3)
    ap.add_argument("--max-new-games",type=int,default=6)
    args=ap.parse_args()
    run(args.injuries,args.boxscores,args.output,days=args.days,
        max_new_games=args.max_new_games)


if __name__ == "__main__":
    main()
