"""Scheduled soccer-player research: ESPN clubs/national teams + optional API-Football.

No scraping social networks. ESPN event IDs are distinct from API-Football
fixture IDs; join ONLY with exact sides + kickoff and unique match.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests

from .cli import append_history
from .core import utc
from .soccer_player_intel import (
    FINAL, canonical_name, parse_api_football_injuries,
    parse_espn_summary, player_opponent_context,
)
from .soccer_confirmed_lineups import parse_confirmed_lineups

ROOT="https://site.api.espn.com/apis/site/v2/sports/soccer"
FOOTBALL_ROOT="https://v3.football.api-sports.io"
CODES=(
    "uefa.champions","uefa.europa","uefa.europa.conf",
    "eng.1","esp.1","ita.1","ger.1","fra.1",
    "uefa.nations","fifa.worldq.uefa",
)
HEADERS={"User-Agent":"KalshiAI-soccer-research/1.0 (+https://github.com/gabrielreyesu/kalshi-ai)"}


def _get(session, url, params=None, api_key=None):
    p=urlparse(url)
    if p.scheme!="https" or p.username or p.password:
        raise ValueError("Invalid soccer provider URL")
    if p.hostname=="site.api.espn.com":
        allowed={f"/apis/site/v2/sports/soccer/{code}/{name}"
                 for code in CODES for name in ("scoreboard","summary")}
        if p.path not in allowed:
            raise ValueError("Unapproved ESPN soccer path")
        limit=5_000_000
        headers=HEADERS
    elif p.hostname=="v3.football.api-sports.io" and api_key:
        if p.path not in ("/injuries","/fixtures","/fixtures/lineups"):
            raise ValueError("Unapproved API-Football path")
        limit=5_000_000
        headers={**HEADERS,"x-apisports-key":api_key}
    else:
        raise ValueError("Soccer provider unavailable or not authorized")
    request_url=url
    redirects=2
    for attempt in range(4):
        r=session.get(request_url, params=params, headers=headers,timeout=15,allow_redirects=False)
        if r.status_code in (301,302,303,307,308):
            location=(getattr(r,"headers",{}) or {}).get("Location")
            if not location or not redirects:
                raise RuntimeError("Soccer provider redirect rejected")
            destination=urljoin(request_url,location)
            target=urlparse(destination)
            if (target.scheme!="https" or target.hostname!=p.hostname or
                target.username or target.password or target.path!=p.path):
                raise ValueError("Soccer provider redirected to unauthorized endpoint")
            request_url=destination
            if target.query:
                params=None
            redirects-=1
            continue
        if r.status_code in (429,500,502,503,504) and attempt<3:
            time.sleep(0.3*(attempt+1))
            continue
        if r.status_code!=200:
            raise RuntimeError(f"Soccer provider HTTP {r.status_code}")
        if len(r.content)>limit:
            raise ValueError("Soccer provider response too large")
        result=r.json()
        if not isinstance(result,dict):
            raise ValueError("Soccer provider response is not a JSON object")
        if p.hostname=="v3.football.api-sports.io" and result.get("errors"):
            raise ValueError("API-Football reported coverage, plan, or quota error")
        return result
    raise RuntimeError("Soccer provider exhausted retries")

def parse_scoreboard(payload,code):
    if not isinstance(payload,dict):
        raise ValueError("Invalid ESPN soccer scoreboard")
    output=[]
    for e in payload.get("events") or []:
        eid=str(e.get("id") or "")
        c=(e.get("competitions") or [{}])[0]
        sides={x.get("homeAway"):x for x in c.get("competitors") or []}
        if not eid.isdigit() or set(sides)!={"home","away"}:
            continue
        teams={}
        for side in ("home","away"):
            t=sides[side].get("team") or {}
            tid=str(t.get("id") or "")
            name=str(t.get("displayName") or t.get("name") or "")
            if not tid or not name:
                break
            teams[side]={"id":tid,"name":name}
        if len(teams)!=2:
            continue
        try:
            date=utc(e.get("date") or c.get("date"))
        except (TypeError,ValueError):
            continue
        status=str(((e.get("status") or {}).get("type") or {}).get("name") or
                   ((c.get("status") or {}).get("type") or {}).get("name") or "")
        output.append({
            "sport":"soccer","event_id":eid,"competition_code":code,
            "kickoff_utc":date.isoformat(),"home_id":teams["home"]["id"],
            "away_id":teams["away"]["id"],"home":teams["home"]["name"],
            "away":teams["away"]["name"],"status":status,
        })
    return output


def parse_api_fixtures(payload):
    if not isinstance(payload,dict) or not isinstance(payload.get("response"),list):
        raise ValueError("API-Football fixture list invalid")
    result=[]
    for x in payload["response"]:
        fixture=x.get("fixture") or {}
        teams=x.get("teams") or {}
        home=teams.get("home") or {}
        away=teams.get("away") or {}
        if not (fixture.get("id") and fixture.get("date") and
                home.get("name") and away.get("name")):
            continue
        result.append({
            "fixture_id":str(fixture["id"]), "kickoff_utc":utc(fixture["date"]).isoformat(),
            "home":str(home["name"]),"away":str(away["name"]),
            "home_provider_id":str(home.get("id") or ""),
            "away_provider_id":str(away.get("id") or ""),
        })
    return result


def join_vendor_injuries(espn_fixtures, api_fixtures, reports):
    """Only attach to a unique matched event with both team names + time."""
    result={}
    reverse={}
    for v in api_fixtures:
        vk=utc(v["kickoff_utc"])
        hits=[
            e for e in espn_fixtures
            if abs(vk-utc(e["kickoff_utc"]))<=pd.Timedelta(minutes=60)
            and canonical_name(v["home"])==canonical_name(e["home"])
            and canonical_name(v["away"])==canonical_name(e["away"])
        ]
        if len(hits)!=1:
            continue
        eid=str(hits[0]["event_id"])
        reverse.setdefault(eid,[]).append(v)
    for event_id,matched in reverse.items():
        if len(matched)!=1:
            continue
        vendor=matched[0]
        event=next((e for e in espn_fixtures if str(e["event_id"])==event_id),None)
        if event is None:
            continue
        injuries=[]
        for report in reports:
            for player in report.get("players") or []:
                if str(player["fixture_id"])!=vendor["fixture_id"]:
                    continue
                allowed={
                    vendor["home_provider_id"]:vendor["home"],
                    vendor["away_provider_id"]:vendor["away"],
                }
                if player["provider_team_id"] not in allowed:
                    continue
                if canonical_name(player["team"])!=canonical_name(allowed[player["provider_team_id"]]):
                    continue
                # Verified fixture side identifies TEAM only. Athlete ID
                # remains provider-specific and must never be equated to ESPN.
                espn_team_id=(str(event["home_id"]) if
                              player["provider_team_id"]==vendor["home_provider_id"]
                              else str(event["away_id"]))
                injuries.append({
                    "espn_team_id":espn_team_id,
                    "player":player["player"],
                    "provider_player_id":player["provider_player_id"],
                    "provider_team_id":player["provider_team_id"],
                    "availability":player["availability"],
                    "reason":player["reason"],
                    "type":player["type"],
                    "source":"API-Football",
                    "identity_match":"unverified_cross_provider_player",
                })
        result[event_id]=injuries
    return result


def _load(path):
    p=Path(path)
    if not p.exists():
        return []
    result=[]
    for line in p.read_text(encoding="utf8").splitlines():
        if line.strip():
            x=json.loads(line)
            if not isinstance(x,dict):
                raise ValueError("Corrupt soccer player JSONL")
            result.append(x)
    return result


def _select_summary_matches(fixtures, history, now, max_finished=7, max_pregame=5):
    seen={(g.get("competition_code"),str(g.get("event_id"))) for g in history}
    finished=[g for g in fixtures if g["status"] in FINAL
              and (g["competition_code"],g["event_id"]) not in seen
              and utc(g["kickoff_utc"])<now]
    # Pregame squad information can become available shortly before kickoff.
    pregame=[g for g in fixtures if g["status"] not in FINAL
             and now+pd.Timedelta(minutes=6)<utc(g["kickoff_utc"])
                    <=now+pd.Timedelta(hours=3)]
    finished.sort(key=lambda g:utc(g["kickoff_utc"]))
    pregame.sort(key=lambda g:utc(g["kickoff_utc"]))
    return finished[:max_finished]+pregame[:max_pregame]


def run(archive="data/external_signals", output="reports/external_learning/soccer_players",
        observed_at=None, session=None, codes=CODES, api_key=None,
        max_summaries=12):
    if not 1<=max_summaries<=30:
        raise ValueError("Invalid soccer summary request budget")
    now=utc(observed_at or pd.Timestamp.now(tz="UTC"))
    s=session or requests.Session()
    cache=Path(archive)
    issues=[]
    all_fixtures={}
    # Limit to pinned country/league paths. Scoreboard is official ESPN site
    # JSON rather than uncontrolled scraping or expensive site crawling.
    for code in codes:
        if code not in CODES:
            raise ValueError("Unsupported soccer competition")
        for day_offset in (-1,0,1):
            date=(now+pd.Timedelta(days=day_offset)).strftime("%Y%m%d")
            try:
                feed=_get(s,f"{ROOT}/{code}/scoreboard",params={"dates":date,"limit":100})
                for g in parse_scoreboard(feed,code):
                    all_fixtures[(g["competition_code"],g["event_id"])]=g
            except (ValueError,RuntimeError,requests.RequestException) as exc:
                issues.append(f"ESPN {code} {date}: {type(exc).__name__}")
    fixtures=list(all_fixtures.values())
    prior_stats=_load(cache/"soccer_player_games.jsonl")
    prior_lineups=_load(cache/"soccer_lineups.jsonl")
    prior_injuries=_load(cache/"soccer_injuries.jsonl")
    matches=_select_summary_matches(fixtures,prior_stats,now)
    summaries=0
    new_stats=[]
    new_lineups=[]
    for game in matches[:max_summaries]:
        try:
            detail=_get(s,f"{ROOT}/{game['competition_code']}/summary",
                        params={"event":game["event_id"]})
            record=parse_espn_summary(detail,game["competition_code"],game["event_id"],
                                      now,expected_kickoff=game["kickoff_utc"],
                                      expected_status=game["status"])
            summaries+=1
            if record and record["home_id"]==game["home_id"] and record["away_id"]==game["away_id"]:
                if record["source_kind"]=="historical_player_stats":
                    new_stats.append(record)
                else:
                    new_lineups.append(record)
        except (ValueError,RuntimeError,requests.RequestException) as exc:
            issues.append(f"ESPN summary {game['event_id']}: {type(exc).__name__}")
    key=api_key if api_key is not None else os.getenv("API_FOOTBALL_KEY","")
    new_reports=[]
    api_fixtures=[]
    if key:
        days={ (now+pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in (0,1)}
        already={r.get("observed_day_utc") for r in prior_injuries
                 if utc(r["captured_utc"]).strftime("%Y-%m-%d")==now.strftime("%Y-%m-%d")}
        for day in sorted(days-already):
            try:
                j=_get(s,FOOTBALL_ROOT+"/injuries",{"date":day},api_key=key)
                injuries=parse_api_football_injuries(j,now,day)
                try:
                    f=_get(s,FOOTBALL_ROOT+"/fixtures",{"date":day},api_key=key)
                    # API-Football's date endpoint also returns matches that
                    # have already started or ended. Never archive those as
                    # pre-match fixture mappings or let them enter future
                    # availability context.
                    injuries["provider_fixtures"]=[
                        v for v in parse_api_fixtures(f)
                        if utc(v["kickoff_utc"])>now+pd.Timedelta(minutes=5)
                        and utc(v["kickoff_utc"]).strftime("%Y-%m-%d")==day
                    ]
                except (ValueError,RuntimeError,requests.RequestException) as error:
                    issues.append(f"API-Football fixture index {day}: {type(error).__name__}")
                    injuries["provider_fixtures"]=[]
                new_reports.append(injuries)
            except (ValueError,RuntimeError,requests.RequestException) as exc:
                issues.append(f"API-Football {day}: {type(exc).__name__}")
        # If a report was already captured today, keep it but matching the
        # source-specific fixture requires a fresh fixture list; skipping a
        # query is safer than pretending to have a verified cross-ID join.
    else:
        issues.append("API_FOOTBALL_KEY not configured; no provider injury statuses")
    all_stats,n_stats=append_history(cache/"soccer_player_games.jsonl",new_stats,max_bytes=100_000_000)
    all_lineups,n_lineups=append_history(cache/"soccer_lineups.jsonl",new_lineups,max_bytes=100_000_000)
    all_injuries,n_injuries=append_history(cache/"soccer_injuries.jsonl",new_reports,max_bytes=100_000_000)
    # Retain a reusable same-day fixture mapping rather than losing all
    # injury context in the second scheduled run of the day.
    latest_by_day={}
    for report in sorted(all_injuries,key=lambda x:utc(x["captured_utc"])):
        day=report.get("observed_day_utc")
        if day in {(now+pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in (0,1)} and (
            pd.Timedelta(0)<=now-utc(report["captured_utc"])<=pd.Timedelta(hours=36)
        ):
            latest_by_day[day]=report
    current_reports=list(latest_by_day.values())
    fixture_map={}
    for report in current_reports:
        for provider_fixture in report.get("provider_fixtures") or []:
            fixture_map[provider_fixture["fixture_id"]]=provider_fixture
    matched=join_vendor_injuries(fixtures,list(fixture_map.values()),current_reports)

    # Official XI observations typically appear shortly before kickoff.
    # Budget <=4 optional requests, no extra schedule and no postgame lineup
    # retrieval disguised as a pregame observation.
    new_confirmed=[]
    if key:
        future=[g for g in fixtures if g["event_id"] in matched and
                now+pd.Timedelta(minutes=15)<utc(g["kickoff_utc"])<=
                    now+pd.Timedelta(hours=2)]
        for game in sorted(future,key=lambda g:utc(g["kickoff_utc"]))[:4]:
            matches=[
                v for v in fixture_map.values()
                if abs(utc(v["kickoff_utc"])-utc(game["kickoff_utc"]))<=pd.Timedelta(minutes=60)
                and canonical_name(v["home"])==canonical_name(game["home"])
                and canonical_name(v["away"])==canonical_name(game["away"])
            ]
            if len(matches)!=1:
                continue
            try:
                response=_get(s,FOOTBALL_ROOT+"/fixtures/lineups",
                              {"fixture":matches[0]["fixture_id"]},api_key=key)
                lineup=parse_confirmed_lineups(response,game,matches[0],now)
                if lineup:
                    new_confirmed.append(lineup)
            except (ValueError,RuntimeError,requests.RequestException) as error:
                issues.append(f"XI {game['event_id']}: {type(error).__name__}")
    _,n_confirmed=append_history(cache/"soccer_confirmed_lineups.jsonl",
                                  new_confirmed,max_bytes=50_000_000)
    previews=[]
    for game in sorted(fixtures,key=lambda g:utc(g["kickoff_utc"])):
        kickoff=utc(game["kickoff_utc"])
        if now+pd.Timedelta(minutes=15)>=kickoff or kickoff>now+pd.Timedelta(hours=48):
            continue
        try:
            context=player_opponent_context(game,all_stats,matched.get(game["event_id"],[]),now)
            context["api_injury_coverage_confirmed"]=bool(
                key and game["event_id"] in matched)
            # Summary coverage only; unverified lineups not treated as starters
            roster=[r for r in all_lineups if r["event_id"]==game["event_id"]
                    and r.get("competition_code")==game["competition_code"]
                    and utc(r["captured_utc"])<=now]
            context["roster_observation_present"]=bool(roster)
            context["lineup_confirmed"]=False
            if roster:
                latest=max(roster,key=lambda x:utc(x["captured_utc"]))
                context["unverified_pregame_roster"]=[
                    {"team_id":str(t["team_id"]),"members":[
                        {"player_id":p["player_id"],"player":p["player"],
                         "starter_reported":bool(p["starter_reported"])}
                        for p in t["members"][:30]]
                    }
                    for t in latest["teams"]
                ]
            previews.append(context)
        except (ValueError,KeyError,TypeError):
            continue
        if len(previews)>=18:
            break
    _,n_previews=append_history(cache/"soccer_previews.jsonl",previews,max_bytes=100_000_000)
    report={
        "observed_utc":now.isoformat(),
        "sports":["soccer_clubs","soccer_national_teams"],
        "fixtures_matched_by_espn":len(fixtures),
        "ESPN_match_summaries_queried":summaries,
        "new_completed_match_player_stats":n_stats,
        "new_pregame_roster_snapshots_unconfirmed":n_lineups,
        "new_verified_pregame_starting_XI_snapshots":n_confirmed,
        "new_api_football_injury_day_snapshots":n_injuries,
        "new_future_match_player_contexts":n_previews,
        "future_contexts_with_history":sum(bool(p["players"]) for p in previews),
        "api_football_configured":bool(key),
        "warnings":issues[:30],
        "production_changed":False,
        "status":"SOCCER_PLAYER_RESEARCH_ONLY",
    }
    path=Path(output)
    path.mkdir(parents=True,exist_ok=True)
    (path/"coverage.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report,ensure_ascii=False))
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--archive",default="data/external_signals")
    parser.add_argument("--output",default="reports/external_learning/soccer_players")
    args=parser.parse_args()
    run(archive=args.archive,output=args.output)


if __name__=="__main__":
    main()
