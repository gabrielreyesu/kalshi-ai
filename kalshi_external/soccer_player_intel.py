"""ESPN soccer match player statistics + API-Football availability, research only.

Important: source-specific athlete IDs are not cross-provider IDs. Everything
used for a future prediction must have been observed BEFORE its kickoff.
"""
from __future__ import annotations

from collections import defaultdict
import math
import unicodedata
import re

import numpy as np
import pandas as pd

from .core import utc

FINAL = {"STATUS_FINAL", "STATUS_FULL_TIME", "STATUS_FINAL_AET", "STATUS_FINAL_PEN"}
SOCCER_MEASURES = ("minutes", "goals", "assists", "shots", "shots_on_target", "saves")
STAT_ALIASES = {
    "MIN": "minutes", "MINS": "minutes", "MINUTES": "minutes",
    "G": "goals", "GL": "goals", "GOALS": "goals",
    "A": "assists", "AST": "assists", "ASSISTS": "assists",
    "SH": "shots", "SHOTS": "shots",
    "SOG": "shots_on_target", "SOT": "shots_on_target", "SHOTS ON GOAL": "shots_on_target",
    "SV": "saves", "SAVES": "saves",
}


def numeric(value):
    if value is None or isinstance(value,bool):
        return None
    s=str(value).strip()
    if s in ("", "-", "--", "DNP", "N/A"):
        return None
    if ":" in s:
        try:
            a,b=s.split(":")
            n=float(a)+float(b)/60
            return n if 0<=n<=150 else None
        except (ValueError,TypeError):
            return None
    try:
        n=float(s)
        return n if math.isfinite(n) and 0<=n<=150 else None
    except (ValueError,TypeError):
        return None


def canonical_name(name):
    s=unicodedata.normalize("NFKD",str(name or "")).encode("ascii","ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]+"," ",s).strip()


def _competition(payload):
    competitions=(payload.get("header") or {}).get("competitions") or []
    if len(competitions)!=1:
        return None
    c=competitions[0]
    sides={x.get("homeAway"):x for x in c.get("competitors") or []}
    if set(sides)!={"home","away"}:
        return None
    home=str((sides["home"].get("team") or {}).get("id") or "")
    away=str((sides["away"].get("team") or {}).get("id") or "")
    date=c.get("date")
    if not (home and away and date):
        return None
    state=str(((c.get("status") or {}).get("type") or {}).get("name") or
              ((payload.get("header") or {}).get("status") or {}).get("name") or "")
    return {"home_id":home,"away_id":away,"kickoff_utc":utc(date).isoformat(),"status":state}


def _player_rows(block):
    """Handle roster and scoreboard-style ESPN soccer shapes if provided."""
    rows=[]
    entries=block.get("roster") or block.get("athletes") or []
    for entry in entries:
        athlete=entry.get("athlete") or entry.get("player") or {}
        pid=str(athlete.get("id") or "")
        if not pid:
            continue
        measures={}
        for stat in entry.get("stats") or []:
            if isinstance(stat,dict):
                name=str(stat.get("abbreviation") or stat.get("name") or "").upper()
                metric=STAT_ALIASES.get(name)
                number=numeric(stat.get("value") if "value" in stat else stat.get("displayValue"))
                if metric and number is not None:
                    measures[metric]=number
        for group in block.get("statistics") or []:
            labels=[str(x).upper() for x in (group.get("labels") or [])]
            for row in group.get("athletes") or []:
                a=row.get("athlete") or {}
                if str(a.get("id") or "")!=pid:
                    continue
                for key,value in zip(labels,row.get("stats") or []):
                    metric=STAT_ALIASES.get(key)
                    number=numeric(value)
                    if metric and number is not None:
                        measures[metric]=number
        if isinstance(entry.get("statistics"),dict):
            for key,value in entry["statistics"].items():
                metric=STAT_ALIASES.get(str(key).upper())
                number=numeric(value)
                if metric and number is not None:
                    measures[metric]=number
        rows.append({
            "player_id":pid,"player":str(athlete.get("displayName") or athlete.get("fullName") or ""),
            "position":str((athlete.get("position") or {}).get("abbreviation") or ""),
            "starter_reported":entry.get("starter") is True,
            "metrics":measures,
        })
    if not rows:
        # Some ESPN soccer summaries provide only grouped player boxscores.
        for group in block.get("statistics") or []:
            labels=[str(x).upper() for x in group.get("labels") or []]
            for item in group.get("athletes") or []:
                athlete=item.get("athlete") or {}
                pid=str(athlete.get("id") or "")
                if not pid:
                    continue
                metrics={}
                for key,value in zip(labels,item.get("stats") or []):
                    metric=STAT_ALIASES.get(key)
                    n=numeric(value)
                    if metric and n is not None:
                        metrics[metric]=n
                rows.append({
                    "player_id":pid,"player":str(athlete.get("displayName") or ""),
                    "position":str((athlete.get("position") or {}).get("abbreviation") or ""),
                    "starter_reported":item.get("starter") is True,"metrics":metrics,
                })
    by_id={}
    for entry in rows:
        if entry["player_id"] not in by_id:
            by_id[entry["player_id"]]=entry
        else:
            by_id[entry["player_id"]]["metrics"].update(entry["metrics"])
    return list(by_id.values())


def parse_espn_summary(payload, code, event_id, captured, expected_kickoff=None):
    """Parse ESPN player appearance data; match exact competition, teams and UTC."""
    if not isinstance(payload,dict):
        raise ValueError("Invalid ESPN soccer summary")
    info=_competition(payload)
    if not info:
        return None
    seen=utc(captured)
    kick=utc(info["kickoff_utc"])
    if expected_kickoff and abs(kick-utc(expected_kickoff))>pd.Timedelta(hours=2):
        # Rescheduled game must never be matched to old kickoff metadata.
        return None
    if not str(event_id).isdigit() or not code:
        return None
    teams=[]
    source=payload.get("rosters") or (payload.get("boxscore") or {}).get("players") or []
    if not isinstance(source,list):
        return None
    valid_teams={info["home_id"],info["away_id"]}
    for block in source:
        team_id=str((block.get("team") or {}).get("id") or "")
        if team_id not in valid_teams:
            continue
        members=_player_rows(block)
        if not members:
            continue
        teams.append({"team_id":team_id,"opponent_id":info["away_id"] if team_id==info["home_id"] else info["home_id"],
                      "members":members})
    if not teams:
        return None
    base={
        "sport":"soccer","event_id":str(event_id),"competition_code":str(code),
        "captured_utc":seen.isoformat(),"kickoff_utc":kick.isoformat(),
        "home_id":info["home_id"],"away_id":info["away_id"],
        "teams":teams,
    }
    if info["status"] in FINAL and seen>kick:
        if not any(p["metrics"] for group in teams for p in group["members"]):
            # A roster without observed individual statistics is not a player boxscore.
            return None
        return {
            **base,"source_id":"espn_soccer_final_player_stats",
            "source_kind":"historical_player_stats",
            "final_status":info["status"],
            "warning":"Captured after match only; cannot backfill as if known pregame.",
        }
    if seen<kick-pd.Timedelta(minutes=5) and info["status"] not in FINAL:
        return {
            **base,"source_id":"espn_soccer_pregame_roster",
            "source_kind":"soccer_pregame_roster",
            "lineup_confirmed":False,
            "warning":"Starter flags unverified as confirmed; do NOT infer official starting XI.",
        }
    return None


def parse_api_football_injuries(payload, captured, day):
    """Provider-level IDs, NOT ESPN IDs. Skip fabricated/future or malformed."""
    if not isinstance(payload,dict) or not isinstance(payload.get("response"),list):
        raise ValueError("API-Football injury response absent or invalid")
    if payload.get("errors"):
        raise ValueError("API-Football provider returned an error")
    seen=utc(captured)
    injuries=[]
    unique=set()
    for item in payload["response"]:
        fixture=item.get("fixture") or {}
        team=item.get("team") or {}
        player=item.get("player") or {}
        date=fixture.get("date")
        fixture_id=str(fixture.get("id") or "")
        team_id=str(team.get("id") or "")
        player_id=str(player.get("id") or "")
        if not (date and fixture_id.isdigit() and team_id and (player_id or player.get("name"))):
            continue
        kickoff=utc(date)
        if kickoff <= seen+pd.Timedelta(minutes=5) or kickoff.strftime("%Y-%m-%d")!=day:
            continue
        typ=str(player.get("type") or "").casefold()
        if typ=="missing fixture":
            state="reported_absent"
        elif typ=="questionable":
            state="uncertain"
        else:
            state="unknown_provider_status"
        ident=(fixture_id,team_id,player_id or canonical_name(player.get("name")))
        if ident in unique:
            continue
        unique.add(ident)
        injuries.append({
            "fixture_id":fixture_id,"fixture_kickoff_utc":kickoff.isoformat(),
            "league":str((item.get("league") or {}).get("name") or ""),
            "provider_team_id":team_id,"team":str(team.get("name") or ""),
            "provider_player_id":player_id,"player":str(player.get("name") or ""),
            "type":str(player.get("type") or ""),"reason":str(player.get("reason") or "")[:150],
            "availability":state,
        })
    return {
        "source_id":"api_football_soccer_injuries",
        "source_kind":"soccer_provider_injuries",
        "sport":"soccer","event_id":f"soccer_injuries_{day}",
        "captured_utc":seen.isoformat(),"observed_day_utc":day,
        "players":injuries,
        "warning":"API-Football IDs and ESPN IDs are NOT interchangeable; no lineup confirmation.",
    }


def player_opponent_context(fixture, player_games, injury_reports, observed_at, min_h2h=2, shrink=6):
    """As-of descriptive player stats and provider injury reports for future event."""
    now=utc(observed_at)
    kickoff=utc(fixture["kickoff_utc"])
    if now>=kickoff:
        raise ValueError("No post-kickoff research context")
    home,away=str(fixture["home_id"]),str(fixture["away_id"])
    if not home or not away or home==away:
        raise ValueError("Invalid canonical team IDs")
    games=defaultdict(list)
    for g in player_games:
        if g.get("source_id")!="espn_soccer_final_player_stats":
            continue
        if utc(g["captured_utc"])>now or utc(g["kickoff_utc"])>=kickoff:
            continue
        for t in g.get("teams") or []:
            tid=str(t.get("team_id"))
            if tid not in (home,away):
                continue
            for p in t.get("members") or []:
                if not p.get("player_id"):
                    continue
                games[(tid,str(p["player_id"]))].append({
                    "when":utc(g["kickoff_utc"]),"opponent":str(t["opponent_id"]),
                    "name":p.get("player"),"position":p.get("position"),
                    "starter":p.get("starter_reported",False),
                    "metrics":p.get("metrics") or {},
                })
    players=[]
    for (tid,pid), entries in sorted(games.items()):
        entries=sorted(entries,key=lambda x:x["when"])[-15:]
        opponent=away if tid==home else home
        metrics={}
        for metric in SOCCER_MEASURES:
            base=[x["metrics"][metric] for x in entries if metric in x["metrics"]]
            against=[x["metrics"][metric] for x in entries if x["opponent"]==opponent and metric in x["metrics"]]
            if not base:
                continue
            average=float(np.mean(base))
            h2h=float(np.mean(against)) if len(against)>=min_h2h else None
            shrunk=(average+len(against)/(len(against)+shrink)*(h2h-average)
                    if h2h is not None else None)
            metrics[metric]={
                "recent_mean":round(average,3),
                "prior_opponent_games":len(against),
                "opponent_mean":round(h2h,3) if h2h is not None else None,
                "shrunk_opponent_mean":round(shrunk,3) if shrunk is not None else None,
            }
        players.append({
            "team_id":tid,"opponent_id":opponent,"espn_player_id":pid,
            "player":str(entries[-1].get("name") or ""),
            "position":str(entries[-1].get("position") or ""),
            "games":len(entries),"past_starts":sum(bool(x["starter"]) for x in entries),
            "metrics":metrics,
        })
    # Vendor injuries may be displayed only if matching to EXACTLY ONE ESPN
    # fixture in the same known registry. Never join vendor athlete IDs.
    return {
        "source_id":"soccer_player_opponent_preview","source_kind":"player_context_research",
        "sport":"soccer","event_id":str(fixture["event_id"]),
        "competition_code":str(fixture.get("competition_code") or ""),
        "captured_utc":now.isoformat(),"kickoff_utc":kickoff.isoformat(),
        "home_id":home,"away_id":away,
        "players":players,
        "provider_injury_context":injury_reports,
        "warning":"No causal player impact estimated. Opponent rates shrunk; injury and ESPN athlete IDs are not merged.",
    }
