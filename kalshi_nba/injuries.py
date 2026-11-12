"""Bajas ACTUALES de NBA y NFL (ESPN, gratis). Se guarda una foto por día para ir construyendo un historial propio,
que es lo que hace falta para poder probar más adelante si las lesiones mejoran el modelo.
Uso:  python -m kalshi_nba.injuries nba   |   python -m kalshi_nba.injuries nfl
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .data import LEAGUES, _json

OUT = {"out", "doubtful", "injured reserve", "suspension", "out for season"}


def fetch(league: str) -> pd.DataFrame:
    j = _json(f"https://site.api.espn.com/apis/site/v2/sports/{LEAGUES[league]['path']}/injuries") or {}
    rows = []
    for t in j.get("injuries", []):
        for x in t.get("injuries", []):
            st = str(x.get("status") or "")
            if st.lower() == "active":
                continue
            a = x.get("athlete") or {}
            rows.append({"team_id": str(t.get("id")), "team": t.get("displayName"), "player": a.get("displayName"),
                         "position": (a.get("position") or {}).get("abbreviation"), "status": st,
                         "type": (x.get("details") or {}).get("type"), "reported": x.get("date")})
    return pd.DataFrame(rows, columns=["team_id", "team", "player", "position", "status", "type", "reported"])


def record(league: str, folder: str = "predictions") -> int:
    path = Path(folder) / f"{league}_injuries.csv"
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    if path.exists() and today in set(pd.read_csv(path, usecols=["day"])["day"]):
        return 0                                           # una foto por día
    d = fetch(league).assign(day=today)
    path.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(path, mode="a", header=not path.exists(), index=False)
    return len(d)


def out_counts(league: str, folder: str = "predictions") -> dict:
    """{team_id: nº de jugadores fuera o dudosos} según la foto más reciente."""
    path = Path(folder) / f"{league}_injuries.csv"
    if not path.exists():
        return {}
    d = pd.read_csv(path, dtype={"team_id": str})
    d = d[d["day"] == d["day"].max()]
    return d[d["status"].str.lower().isin(OUT)].groupby("team_id").size().to_dict()


if __name__ == "__main__":
    lg = sys.argv[1] if len(sys.argv) > 1 else "nba"
    print(f"[{lg}] bajas registradas hoy: {record(lg)}")
