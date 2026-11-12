"""Bajas de fútbol desde API-Football (v3.football.api-sports.io). Requiere la variable API_FOOTBALL_KEY.
Guarda una foto por día en predictions/futbol_lesiones.csv (3 consultas diarias: hoy y los 2 días siguientes).
Uso:  python -m kalshi_intl.lesiones
NOTA: escrito según la documentación pública; el plan gratuito puede limitar temporadas o fechas. Si la API devuelve
un aviso de plan, se imprime tal cual y el resto del flujo diario continúa.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

from kalshi_soccer.kalshi import _same

URL = "https://v3.football.api-sports.io/injuries"
FILE = Path("predictions/futbol_lesiones.csv")
COLS = ["day", "match_date", "fixture_id", "league", "team", "player", "tipo", "motivo"]


def parse(payload: dict) -> list:
    rows = []
    for x in payload.get("response") or []:
        rows.append({"match_date": str((x.get("fixture") or {}).get("date", ""))[:10], "fixture_id": (x.get("fixture") or {}).get("id"),
                     "league": (x.get("league") or {}).get("name"), "team": (x.get("team") or {}).get("name"),
                     "player": (x.get("player") or {}).get("name"), "tipo": (x.get("player") or {}).get("type"),
                     "motivo": (x.get("player") or {}).get("reason")})
    return rows


def record(days: int = 3) -> int:
    key = (os.environ.get("API_FOOTBALL_KEY") or "").strip()
    if not key:
        print("Sin API_FOOTBALL_KEY: se omiten las bajas de fútbol."); return 0
    today = pd.Timestamp.now(tz="UTC").normalize()
    if FILE.exists() and today.strftime("%Y-%m-%d") in set(pd.read_csv(FILE, usecols=["day"])["day"]):
        print("Bajas de fútbol: ya hay foto de hoy."); return 0
    rows = []
    for k in range(days):
        d = (today + pd.Timedelta(days=k)).strftime("%Y-%m-%d")
        r = requests.get(URL, params={"date": d}, headers={"x-apisports-key": key}, timeout=40)
        j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if r.status_code != 200 or j.get("errors"):
            print(f"API-Football {d}: HTTP {r.status_code} | aviso: {j.get('errors') or r.text[:200]}")
            continue
        got = parse(j)
        print(f"API-Football {d}: {len(got)} bajas")
        rows += got
    if not rows:
        return 0
    d = pd.DataFrame(rows).assign(day=today.strftime("%Y-%m-%d"))[COLS]
    FILE.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(FILE, mode="a", header=not FILE.exists(), index=False)
    return len(d)


def load_latest(path: Path = FILE) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=COLS)
    d = pd.read_csv(path)
    d = d[d["day"] == d["day"].max()]
    return d[d["tipo"].astype(str).str.lower().str.contains("missing")]          # solo bajas confirmadas


def count_out(latest: pd.DataFrame, team: str, kickoff) -> int | None:
    """Bajas confirmadas del equipo para el partido de esa fecha (±1 día). None si no hay dato o el nombre es ambiguo."""
    if latest.empty:
        return None
    day = pd.Timestamp(kickoff).tz_convert("UTC").normalize().tz_localize(None) if pd.Timestamp(kickoff).tzinfo else pd.Timestamp(kickoff).normalize()
    near = latest[(pd.to_datetime(latest["match_date"]) - day).abs() <= pd.Timedelta(days=1)]
    names = [n for n in near["team"].dropna().unique() if _same(str(n), team)]
    if len(names) != 1:
        return None
    return int((near["team"] == names[0]).sum())


if __name__ == "__main__":
    print("Bajas de fútbol registradas:", record())
