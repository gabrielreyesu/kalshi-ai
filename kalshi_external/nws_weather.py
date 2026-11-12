"""NOAA/NWS hourly forecasts at verified outdoor NFL stadiums (research only).

Independent public-domain weather source; NO changes to probability models.
We require a corroborated ESPN venue on the exact event before accessing the
stadium point, so neutral/international/dome games are never guessed.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from .cli import append_history
from .core import utc


# Coordinates checked against public geographic references and venue names.
# Only fixed, confidently outdoor NFL fields; NOT an exhaustive stadium list.
# Keys match ESPN's displayName used by nfl_registry.csv.
STADIUMS = {
    "Green Bay Packers": (44.50097, -88.06168, ("lambeau",)),
    "Kansas City Chiefs": (39.048786, -94.484566, ("arrowhead",)),
    "New York Giants": (40.81352, -74.07435, ("metlife",)),
    "New York Jets": (40.81352, -74.07435, ("metlife",)),
    "Philadelphia Eagles": (39.90084, -75.16747, ("lincoln financial",)),
    "Pittsburgh Steelers": (40.44673, -80.01582, ("acrisure", "heinz field")),
    "Seattle Seahawks": (47.5952, -122.3316, ("lumen", "centurylink")),
}
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
NWS = "https://api.weather.gov"
HEADERS = {"User-Agent": "KalshiAI-weather-research (https://github.com/gabrielreyesu/kalshi-ai)",
           "Accept": "application/geo+json, application/json"}


def _get(session, url, *, params=None):
    p = urlparse(url)
    if p.scheme != "https" or p.hostname not in {"api.weather.gov", "site.api.espn.com"}:
        raise ValueError("Untrusted forecast endpoint")
    if p.hostname == "api.weather.gov" and not (p.path.startswith("/points/") or p.path.startswith("/gridpoints/")):
        raise ValueError("Unexpected NOAA forecast path")
    if p.hostname == "site.api.espn.com" and p.path not in {urlparse(ESPN_SUMMARY).path, urlparse(ESPN_SCOREBOARD).path}:
        raise ValueError("Unexpected ESPN forecast venue path")
    response = session.get(url, params=params, headers=HEADERS, timeout=15, allow_redirects=False)
    if response.status_code != 200 or len(response.content) > 2_000_000:
        raise ValueError(f"Forecast source returned HTTP {response.status_code} or oversized data")
    return response.json()


def _venue_name(competition):
    venue = (competition or {}).get("venue") or {}
    return (venue.get("fullName") or venue.get("name") or "").strip()


def _venue_for_event(session, event_id, kickoff):
    """Find the actual venue, first in summary then in that day's scoreboard.

    Do not infer stadium location from the listed home team's city alone.
    """
    if not str(event_id).isdigit():
        raise ValueError("Invalid ESPN event id")
    data = {}
    try:
        data = _get(session, ESPN_SUMMARY, params={"event": str(event_id)})
    except (ValueError, requests.RequestException):
        # ESPN sometimes leaves venue metadata out of the summary payload.
        pass
    contests = ((data.get("header") or {}).get("competitions") or [])
    if len(contests) == 1:
        venue = _venue_name(contests[0])
        if venue:
            return venue
    venue = _venue_name((data.get("gameInfo") or {}))
    if venue:
        return venue
    date = utc(kickoff).strftime("%Y%m%d")
    board = _get(session, ESPN_SCOREBOARD, params={"dates": date, "limit": 500})
    found = [e for e in (board.get("events") or []) if str(e.get("id")) == str(event_id)]
    if len(found) != 1:
        return None
    games = found[0].get("competitions") or []
    if len(games) != 1:
        return None
    return _venue_name(games[0]) or None

def _max_mph(text):
    found = re.findall(r"\d+(?:\.\d+)?", str(text or ""))
    if "mph" not in str(text or "").lower() or not found:
        return None
    return max(float(v) for v in found)


def choose_hour(periods, kickoff):
    kick = utc(kickoff)
    for period in periods:
        try:
            start, end = utc(period["startTime"]), utc(period["endTime"])
        except (KeyError, ValueError, TypeError):
            continue
        if start <= kick < end:
            return period
    return None


def make_weather(event, venue, period, captured, issued):
    kick, seen = utc(event["kickoff_utc"]), utc(captured)
    if seen >= kick - pd.Timedelta(minutes=15):
        raise ValueError("Weather snapshot captured too late")
    if issued and utc(issued) > seen:
        raise ValueError("Weather forecast issued in the future")
    temperature = period.get("temperature")
    if temperature is not None:
        temperature = float(temperature)
        if period.get("temperatureUnit") == "C":
            temperature = 32 + 9 * temperature / 5
    rain = (period.get("probabilityOfPrecipitation") or {}).get("value")
    if rain is not None:
        rain = float(rain)
        if not 0 <= rain <= 100:
            raise ValueError("Invalid precipitation forecast")
    return {
        "source_id": "noaa_nws_hourly",
        "source_kind": "environment",
        "sport": "nfl",
        "event_id": str(event["event_id"]),
        "home": str(event["home"]), "away": str(event["away"]),
        "venue": str(venue),
        "kickoff_utc": kick.isoformat(), "captured_utc": seen.isoformat(),
        "published_utc": utc(issued).isoformat() if issued else seen.isoformat(),
        "hour_start_utc": utc(period["startTime"]).isoformat(),
        "temperature_f": temperature,
        "wind_speed_max_mph": _max_mph(period.get("windSpeed")),
        "wind_direction": str(period.get("windDirection", ""))[:20],
        "precip_probability_pct": rain,
        "short_forecast": str(period.get("shortForecast", ""))[:180],
        "attribution": "NOAA National Weather Service (weather.gov)",
        "training_permission": "public_domain_research_not_production_validated",
    }


def forecast_event(session, event, observed_at, diagnostics=None):
    home = event["home"]
    if home not in STADIUMS:
        return None
    kick, now = utc(event["kickoff_utc"]), utc(observed_at)
    if not now < kick - pd.Timedelta(minutes=15) or kick > now + pd.Timedelta(days=7):
        return None
    lat, lon, names = STADIUMS[home]
    venue = _venue_for_event(session, event["event_id"], kick)
    if not venue:
        if diagnostics is not None:
            diagnostics.append(f"Event {event['event_id']}: ESPN venue metadata unavailable")
        return None
    if not any(n in venue.casefold() for n in names):
        if diagnostics is not None:
            diagnostics.append(f"Event {event['event_id']}: verified venue differs from expected home stadium")
        return None
    location = _get(session, f"{NWS}/points/{lat},{lon}")
    hourly_url = (location.get("properties") or {}).get("forecastHourly")
    if not isinstance(hourly_url, str):
        if diagnostics is not None: diagnostics.append(f"Event {event['event_id']}: NWS hourly URL absent")
        return None
    hourly = _get(session, hourly_url)
    prop = hourly.get("properties") or {}
    period = choose_hour(prop.get("periods") or [], kick)
    if period is None:
        if diagnostics is not None: diagnostics.append(f"Event {event['event_id']}: NWS forecast hour unavailable")
        return None
    return make_weather(event, venue, period, now, prop.get("updated"))


def run(registry_path="predictions/nfl_registry.csv",
        history_path="data/external_signals/weather.jsonl",
        output="reports/external_learning/weather_status.json",
        observed_at=None, session=None, max_events=8):
    if max_events < 1 or max_events > 20:
        raise ValueError("max_events must be 1..20")
    now = utc(observed_at or pd.Timestamp.now(tz="UTC"))
    path = Path(registry_path)
    issues = []
    records = []
    available = 0
    if not path.exists():
        issues.append("NFL registry is missing")
    else:
        reg = pd.read_csv(path, dtype={"event_id": str})
        required = {"event_id", "home", "away", "kickoff_utc"}
        if not required.issubset(reg):
            raise ValueError("NFL registry missing columns")
        reg = reg.drop_duplicates("event_id", keep="first")
        reg["kickoff_utc"] = pd.to_datetime(reg.kickoff_utc, utc=True, errors="coerce")
        future = reg[(reg.kickoff_utc > now + pd.Timedelta(minutes=15)) &
                     (reg.kickoff_utc <= now + pd.Timedelta(days=7)) &
                     reg.home.isin(STADIUMS)].sort_values("kickoff_utc")
        available = len(future)
        sess = session or requests.Session()
        for event in future.head(max_events).to_dict("records"):
            try:
                record = forecast_event(sess, event, now, diagnostics=issues)
                if record:
                    records.append(record)
            except (requests.RequestException, ValueError, TypeError, KeyError) as err:
                issues.append(f"Event {event['event_id']}: {type(err).__name__}")
    _, added = append_history(history_path, records)
    status = {
        "captured_utc": now.isoformat(),
        "source": "NOAA/NWS official hourly (research only)",
        "eligible_games_near_outdoor_stadiums": available,
        "new_forecast_snapshots": added,
        "matched_weather_events": len(records),
        "warnings": issues[:20],
        "production_changed": False,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(status, indent=2, ensure_ascii=False)+"\n", encoding="utf8")
    print(json.dumps(status, ensure_ascii=False))
    return status


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--registry",default="predictions/nfl_registry.csv")
    ap.add_argument("--history",default="data/external_signals/weather.jsonl")
    ap.add_argument("--output",default="reports/external_learning/weather_status.json")
    args=ap.parse_args()
    run(args.registry,args.history,args.output)


if __name__ == "__main__":
    main()
