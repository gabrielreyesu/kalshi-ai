"""Viajes y hora del partido (NBA/NFL), solo con información conocida antes del inicio (investigación).

Protocolo pre-registrado en TRAVEL_RESEARCH.md. Sede = estadio del local (los partidos marcados neutrales quedan en 0).
Variables (positivo = favorece al LOCAL):
  travel   (km recorridos por el visitante desde su partido anterior − km del local) / 1000
  tz       (husos cruzados por el visitante − husos cruzados por el local) desde su partido anterior, en valor absoluto
  clock    huso de la sede − huso de la ciudad del visitante (positivo = el visitante viene del oeste: juega "más temprano")
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

NY, CHI, DEN, LA, PHX, TOR = "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles", "America/Phoenix", "America/Toronto"
NBA = {"1": (33.757, -84.396, NY), "2": (42.366, -71.062, NY), "3": (29.949, -90.082, CHI), "4": (41.881, -87.674, CHI),
       "5": (41.497, -81.688, NY), "6": (32.790, -96.810, CHI), "7": (39.749, -105.008, DEN), "8": (42.341, -83.055, NY),
       "9": (37.768, -122.388, LA), "10": (29.751, -95.362, CHI), "11": (39.764, -86.156, NY), "12": (33.945, -118.343, LA),
       "13": (34.043, -118.267, LA), "14": (25.781, -80.188, NY), "15": (43.045, -87.917, CHI), "16": (44.979, -93.276, CHI),
       "17": (40.683, -73.976, NY), "18": (40.751, -73.993, NY), "19": (28.539, -81.384, NY), "20": (39.901, -75.172, NY),
       "21": (33.446, -112.071, PHX), "22": (45.532, -122.667, LA), "23": (38.580, -121.500, LA), "24": (29.427, -98.438, CHI),
       "25": (35.463, -97.515, CHI), "26": (40.768, -111.901, DEN), "27": (38.898, -77.021, NY), "28": (43.643, -79.379, TOR),
       "29": (35.138, -90.051, CHI), "30": (35.225, -80.839, NY)}
NFL = {"1": (33.755, -84.401, NY), "2": (42.774, -78.787, NY), "3": (41.862, -87.617, CHI), "4": (39.095, -84.516, NY),
       "5": (41.506, -81.700, NY), "6": (32.748, -97.093, CHI), "7": (39.744, -105.020, DEN), "8": (42.340, -83.046, NY),
       "9": (44.501, -88.062, CHI), "10": (36.166, -86.771, CHI), "11": (39.760, -86.164, NY), "12": (39.049, -94.484, CHI),
       "13": (36.091, -115.184, LA), "14": (33.953, -118.339, LA), "15": (25.958, -80.239, NY), "16": (44.974, -93.258, CHI),
       "17": (42.091, -71.264, NY), "18": (29.951, -90.081, CHI), "19": (40.813, -74.074, NY), "20": (40.813, -74.074, NY),
       "21": (39.901, -75.168, NY), "22": (33.528, -112.263, PHX), "23": (40.447, -80.016, NY), "24": (33.953, -118.339, LA),
       "25": (37.403, -121.970, LA), "26": (47.595, -122.332, LA), "27": (27.976, -82.503, NY), "28": (38.908, -76.864, NY),
       "29": (35.226, -80.853, NY), "30": (30.324, -81.637, NY), "33": (39.278, -76.623, NY), "34": (29.685, -95.411, CHI)}
NFL_MOVES = {("14", 2015): (38.633, -90.188, CHI), ("24", 2015): (32.783, -117.120, LA), ("24", 2016): (32.783, -117.120, LA),
             ("13", 2015): (37.752, -122.201, LA), ("13", 2016): (37.752, -122.201, LA), ("13", 2017): (37.752, -122.201, LA),
             ("13", 2018): (37.752, -122.201, LA), ("13", 2019): (37.752, -122.201, LA)}
TRAVEL_FEATURES = ["travel", "tz", "clock"]


def km(a, b) -> float:
    (la1, lo1), (la2, lo2) = a[:2], b[:2]
    p1, p2, dl = math.radians(la1), math.radians(la2), math.radians(lo2 - lo1)
    c = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(min(1.0, c)))


def utc_offset(tz: str, when: pd.Timestamp) -> float:
    t = when.tz_convert(tz) if when.tzinfo else when.tz_localize("UTC").tz_convert(tz)
    return t.utcoffset().total_seconds() / 3600


def site(league: str, team: str, season: int):
    if league == "nfl" and (team, season) in NFL_MOVES:
        return NFL_MOVES[(team, season)]
    return (NBA if league == "nba" else NFL).get(team)


def compute(games: pd.DataFrame, league: str) -> pd.DataFrame:
    """games: event_id, kickoff_utc, season_year, home_id, away_id, (neutral). Orden cronológico; sin datos futuros."""
    g = games.copy()
    g["kick"] = pd.to_datetime(g["kickoff_utc"], utc=True, format="ISO8601")
    last = {}                                   # equipo -> (sede anterior)
    rows = []
    for x in g.sort_values("kick").itertuples():
        h, a = str(x.home_id), str(x.away_id)
        ven = site(league, h, int(x.season_year))
        home_city = site(league, h, int(x.season_year)); away_city = site(league, a, int(x.season_year))
        if ven is None or away_city is None or bool(getattr(x, "neutral", False)):
            rows.append({"event_id": str(x.event_id), "travel": 0.0, "tz": 0.0, "clock": 0.0})
        else:
            off_v = utc_offset(ven[2], x.kick)
            def trip(t, city):
                prev = last.get(t, city)
                return km(prev, ven), abs(off_v - utc_offset(prev[2], x.kick))
            ka, ta = trip(a, away_city); kh, th = trip(h, home_city)
            rows.append({"event_id": str(x.event_id), "travel": (ka - kh) / 1000, "tz": ta - th,
                         "clock": off_v - utc_offset(away_city[2], x.kick)})
        if ven is not None and not bool(getattr(x, "neutral", False)):
            last[h] = ven; last[a] = ven
    return pd.DataFrame(rows)
