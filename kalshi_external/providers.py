"""Official API adapters; no scraping of prediction websites or individual profiles.

Optional sources:
 - The Odds API: bookmaker h2h consensus (not independent human forecasts).
 - API-Football: provider's independent statistical 1X2 forecast.
 - Explicitly licensed JSON analyst feed over HTTPS.
All adapters keep provenance and reject in-play forecasts.
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

from .core import future_match, make_record, utc

ODDS_BASE="https://api.the-odds-api.com/v4"
FOOTBALL_BASE="https://v3.football.api-sports.io"
DEFAULT_SPORTS=("americanfootball_nfl","basketball_nba","soccer_uefa_champs_league")
SPORT_MAP={"americanfootball_nfl":"nfl","basketball_nba":"nba",
           "soccer_uefa_champs_league":"intl"}


def _get(session, url, *, params=None, headers=None):
    # URL and headers are always constructed in this module, never untrusted
    # redirects to third parties or caller-provided HTTP destinations.
    res=session.get(url,params=params,headers=headers,timeout=18,allow_redirects=False)
    if res.status_code!=200:
        raise RuntimeError(f"Approved data API returned HTTP {res.status_code}")
    if len(res.content)>3_000_000:
        raise ValueError("Provider response too large")
    return res.json()


def odds_consensus(events, registry, observed_at, sports=None, min_books=2):
    """Convert *pre-match* bookmaker h2h odds into de-vigged probabilities.

    Do not count multiple bookmakers as independent expert model forecasts.
    """
    rows=[]
    for event in events:
        sport=(sports or SPORT_MAP).get(event.get("sport_key"))
        if not sport:continue
        found=future_match(registry,sport,event.get("home_team",""),event.get("away_team",""),
                           event.get("commence_time"),observed_at)
        if found is None:continue
        books=[]
        for book in event.get("bookmakers",[]):
            markets=[m for m in book.get("markets",[]) if m.get("key")=="h2h"]
            if not markets:continue
            market=markets[0]
            if market.get("last_update"):
                lu=utc(market["last_update"])
                if lu>utc(observed_at) or (utc(observed_at)-lu)>pd.Timedelta(hours=48):
                    continue
            opts=market.get("outcomes",[])
            price={o.get("name"):o.get("price") for o in opts}
            names=[event["home_team"]]+(["Draw"] if sport=="intl" else [])+[event["away_team"]]
            try:
                x=np.array([float(price[n]) for n in names],dtype=float)
            except (ValueError,TypeError,KeyError):
                continue
            if not np.isfinite(x).all() or (x<=1).any():continue
            implied=1/x
            books.append(implied/implied.sum())
        if len(books)<min_books:continue
        p=np.mean(books,axis=0)
        try:rec=make_record(found,"oddsapi_bookmaker_consensus",observed_at,p,source_kind="market")
        except ValueError:continue
        rec["n_bookmakers"]=len(books)
        rows.append(rec)
    return rows


def fetch_odds(session, registry, captured, key, sports=DEFAULT_SPORTS):
    results=[]; errors=[]
    if not key:return results,["THE_ODDS_API_KEY is not configured; odds source skipped"]
    for sport in sports:
        if sport not in SPORT_MAP:
            errors.append(f"Unsupported odds sport: {sport}")
            continue
        try:
            data=_get(session,f"{ODDS_BASE}/sports/{sport}/odds",
                      params={"apiKey":key,"regions":"us,uk,eu","markets":"h2h","oddsFormat":"decimal"})
            if not isinstance(data,list):raise ValueError("Malformed odds response")
            results.extend(odds_consensus(data,registry,captured))
        except (requests.RequestException, ValueError,RuntimeError) as exc:
            # Never include request URLs: the API key appears in its query.
            errors.append(f"Odds provider {sport}: {type(exc).__name__} (no credentials logged)")
    return results,errors


def _pct(value):
    if isinstance(value,str):
        value=value.strip().rstrip("%")
    return float(value)


def parse_football_forecast(fixture, forecast, registry, captured):
    if fixture.get("fixture",{}).get("status",{}).get("short")!="NS":
        return None
    f=fixture["fixture"]
    found=future_match(registry,"intl",
                       fixture["teams"]["home"]["name"],fixture["teams"]["away"]["name"],
                       f["date"],captured)
    if found is None:return None
    predictions=forecast.get("response",[])
    if not predictions:return None
    percent=predictions[0].get("predictions",{}).get("percent",{})
    try:
        p=[_pct(percent[x]) for x in ("home","draw","away")]
        return make_record(found,"api_football_stat_model",captured,p,source_kind="model")
    except (ValueError,TypeError,KeyError):
        return None


def fetch_football(session,registry,captured,key,max_forecasts=12):
    out=[];errors=[]
    if not key:return out,["API_FOOTBALL_KEY is not configured; statistical source skipped"]
    headers={"x-apisports-key":key}
    start=utc(captured).date()
    end=(utc(captured)+pd.Timedelta(days=3)).date()
    try:
        payload=_get(session,f"{FOOTBALL_BASE}/fixtures",
                     params={"from":start.isoformat(),"to":end.isoformat(),"timezone":"UTC"},headers=headers)
        fixtures=payload.get("response",[])
    except (requests.RequestException,ValueError,RuntimeError,KeyError) as exc:
        return [],[f"API-Football fixture lookup failed: {type(exc).__name__}"]
    matched=[]
    for fixture in fixtures:
        try:
            if fixture.get("fixture",{}).get("status",{}).get("short")!="NS":continue
            if future_match(registry,"intl",fixture["teams"]["home"]["name"],
                            fixture["teams"]["away"]["name"],fixture["fixture"]["date"],captured) is not None:
                matched.append(fixture)
        except (KeyError,ValueError):
            continue
    for fixture in matched[:max_forecasts]:
        try:
            response=_get(session,f"{FOOTBALL_BASE}/predictions",
                          params={"fixture":fixture["fixture"]["id"]},headers=headers)
            record=parse_football_forecast(fixture,response,registry,captured)
            if record:out.append(record)
        except (requests.RequestException,ValueError,RuntimeError,KeyError) as exc:
            errors.append(f"API-Football forecast unavailable: {type(exc).__name__}")
    return out,errors


def parse_licensed_analysts(items,registry,captured):
    """Author/agency must publish a timestamped 1X2 or 2-way forecast.

    Ignore malformed/late items; never infer probabilities from prose.
    """
    result=[]
    if not isinstance(items,list):raise ValueError("Analyst feed must be a list")
    if len(items)>1000:raise ValueError("Analyst feed too large")
    for item in items:
        try:
            if not isinstance(item,dict) or not item.get("licensed_for_modeling",False):continue
            sport=item["sport"]
            entry=registry[(registry.sport==sport)&(registry.event_id.astype(str)==str(item["event_id"]))]
            if len(entry)!=1:continue
            found=entry.iloc[0]
            if utc(item["published_utc"])>=utc(found.kickoff_utc)-pd.Timedelta(minutes=15):
                continue
            if utc(item["published_utc"])>utc(captured):continue
            p=[item["p_H"]]+([item["p_D"]] if sport=="intl" else [])+[item["p_A"]]
            rec=make_record(found,item["source_id"],captured,p,item["published_utc"],source_kind="analyst")
            result.append(rec)
        except (ValueError,KeyError,TypeError):
            continue
    return result


def fetch_licensed_analysts(session,registry,captured,url,allowed_host):
    if not url or not allowed_host:return [],["No licensed analyst feed configured; skipped"]
    parsed=urlparse(url)
    if parsed.scheme!="https" or parsed.hostname!=allowed_host or parsed.username or parsed.password:
        return [],["Analyst feed URL is not an approved HTTPS host"]
    try:
        response=session.get(url,timeout=15,allow_redirects=False)
        if response.status_code!=200 or len(response.content)>250_000:
            raise ValueError("Analyst feed unavailable or too large")
        payload=response.json()
        return parse_licensed_analysts(payload,registry,captured),[]
    except (requests.RequestException,ValueError,TypeError) as exc:
        return [],[f"Licensed analyst feed unavailable: {type(exc).__name__}"]
