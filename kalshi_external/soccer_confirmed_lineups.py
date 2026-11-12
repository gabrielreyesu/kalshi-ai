"""Strictly pre-kickoff API-Football XI and bench snapshots (no postgame reconstruction)."""
from __future__ import annotations
import pandas as pd
from .core import utc
from .soccer_player_intel import canonical_name


def parse_confirmed_lineups(payload, fixture, vendor_fixture, captured):
    now=utc(captured)
    kick=utc(fixture["kickoff_utc"])
    if (now >= kick-pd.Timedelta(minutes=5) or
        now < kick-pd.Timedelta(hours=2)):
        return None
    if (abs(kick-utc(vendor_fixture["kickoff_utc"]))>pd.Timedelta(minutes=60) or
        canonical_name(fixture["home"])!=canonical_name(vendor_fixture["home"]) or
        canonical_name(fixture["away"])!=canonical_name(vendor_fixture["away"])):
        return None
    if not isinstance(payload,dict) or not isinstance(payload.get("response"),list):
        return None
    provider_to_espn={
        str(vendor_fixture["home_provider_id"]):str(fixture["home_id"]),
        str(vendor_fixture["away_provider_id"]):str(fixture["away_id"]),
    }
    teams=[]
    seen=set()
    for block in payload["response"]:
        if not isinstance(block,dict):
            continue
        source_team=str((block.get("team") or {}).get("id") or "")
        if source_team not in provider_to_espn or source_team in seen:
            return None
        seen.add(source_team)
        def members(items):
            out=[]
            for row in items or []:
                player=(row or {}).get("player") or {}
                pid=str(player.get("id") or "")
                name=str(player.get("name") or "").strip()
                if not pid or not name:
                    continue
                out.append({
                    "provider_player_id":pid,"player":name,
                    "position":str(player.get("pos") or ""),
                })
            return out
        xi=members(block.get("startXI"))
        bench=members(block.get("substitutes"))
        if len(xi)!=11 or len({p["provider_player_id"] for p in xi})!=11:
            return None
        if {p["provider_player_id"] for p in xi} & {p["provider_player_id"] for p in bench}:
            return None
        teams.append({
            "provider_team_id":source_team,
            "espn_team_id":provider_to_espn[source_team],
            "formation":str(block.get("formation") or ""),
            "startXI":xi,"substitutes":bench,
        })
    if len(teams)!=2 or set(x["espn_team_id"] for x in teams)!={
        str(fixture["home_id"]),str(fixture["away_id"])}:
        return None
    return {
        "source_id":"api_football_pregame_confirmed_XI",
        "source_kind":"soccer_pregame_confirmed_XI",
        "sport":"soccer","event_id":str(fixture["event_id"]),
        "competition_code":str(fixture["competition_code"]),
        "provider_fixture_id":str(vendor_fixture["fixture_id"]),
        "captured_utc":now.isoformat(),"kickoff_utc":kick.isoformat(),
        "home_id":str(fixture["home_id"]),"away_id":str(fixture["away_id"]),
        "lineup_confirmed":True,"teams":teams,
        "note":"Two complete pregame XI lists from API-Football. Do not use as if seen before captured_utc.",
    }


def match_lineup_to_espn_roster(lineup, espn_roster, observed):
    """Return a conservative map of uniquely named XI athletes to ESPN IDs."""
    if (not lineup or lineup.get("lineup_confirmed") is not True
            or utc(lineup["captured_utc"])>utc(observed)):
        return None
    starter_ids={}
    for team in lineup.get("teams") or []:
        tid=str(team["espn_team_id"])
        by_name={}
        for pid,p in espn_roster.get(tid,{}).items():
            name=canonical_name(p["player"])
            if name:
                by_name.setdefault(name,[]).append(pid)
        ids=[]
        for member in team.get("startXI") or []:
            options=by_name.get(canonical_name(member["player"]),[])
            if len(options)==1:
                ids.append(options[0])
        starter_ids[tid]=ids
    return {
        "lineup_confirmed":True,"event_id":str(lineup["event_id"]),
        "captured_utc":lineup["captured_utc"],
        "starter_ids_by_espn_team":starter_ids,
        "source":"API-Football; ESPN athlete ID matched by unique exact name within team",
    }
