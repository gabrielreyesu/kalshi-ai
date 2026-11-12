"""Descarga de NBA desde las API públicas de ESPN: resultados por día y líneas históricas por partido.

Solo lectura y con caché en disco (data/nba/). Los días ya cerrados nunca se vuelven a pedir.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SB = "https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard"
ODDS = "https://sports.core.api.espn.com/v2/sports/{sport}/leagues/{lg}/events/{e}/competitions/{e}/odds"
KEEP_TYPES = {2, 3, 5}            # 2 regular, 3 playoffs, 5 play-in

# start/end = (mes, día); year_offset: el año de 'season_year' respecto al año en que EMPIEZA la temporada
LEAGUES = {
    "nba":   dict(path="basketball/nba", params={}, start=(10, 1), end=(6, 30), year_offset=1, n_teams=30,
                  total_range=(150.0, 300.0), max_spread=30.0),
    "ncaab": dict(path="basketball/mens-college-basketball", params={"groups": "50", "limit": "500"}, start=(11, 1),
                  end=(4, 10), year_offset=1, n_teams=None, total_range=(90.0, 210.0), max_spread=50.0),
    "nfl":   dict(path="football/nfl", params={}, start=(9, 1), end=(2, 20), year_offset=0, n_teams=32,
                  total_range=(25.0, 75.0), max_spread=30.0),
}
PREFERRED = ["ESPN BET", "Caesars Sportsbook", "DraftKings", "consensus", "teamrankings"]

_S = requests.Session()
_S.headers["User-Agent"] = "kalshi-nba/1.0"


def _json(url, params=None, retries=6):
    for i in range(retries):
        try:
            r = _S.get(url, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(1.5 * (i + 1)); continue
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(1.5 * (i + 1)); continue
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            time.sleep(1.5 * (i + 1)); continue
        try:
            return r.json()
        except ValueError:
            time.sleep(1.0)
    raise RuntimeError(f"ESPN no respondió: {url} {params}")


def _am(x):
    try:
        return float(str(x).replace("+", ""))
    except (TypeError, ValueError):
        return np.nan


def parse_event(e: dict, day: str) -> dict | None:
    c = (e.get("competitions") or [None])[0]
    if not c:
        return None
    side = {t["homeAway"]: t for t in c.get("competitors", [])}
    if "home" not in side or "away" not in side:
        return None
    sc = lambda t: float(t["score"]) if str(t.get("score", "")).replace(".", "").isdigit() else np.nan
    return {"event_id": str(e["id"]), "day": day, "kickoff_utc": e["date"],
            "season_year": e.get("season", {}).get("year"), "season_type": e.get("season", {}).get("type"),
            "status": e["status"]["type"]["name"],
            "home_id": str(side["home"]["team"]["id"]), "home": side["home"]["team"]["displayName"],
            "away_id": str(side["away"]["team"]["id"]), "away": side["away"]["team"]["displayName"],
            "hs": sc(side["home"]), "as_": sc(side["away"]), "neutral": bool(c.get("neutralSite", False))}


def fetch_day(day: date, cache: Path, force=False, league="nba") -> list:
    p = cache / "sb" / f"{day:%Y%m%d}.json"
    closed = day < date.today() - timedelta(days=2)
    if p.exists() and closed and not force:
        return json.loads(p.read_text())
    L = LEAGUES[league]
    j = _json(SB.format(path=L["path"]), {"dates": f"{day:%Y%m%d}", **L["params"]})
    rows = [r for r in (parse_event(e, day.isoformat()) for e in (j or {}).get("events", [])) if r]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows))
    return rows


def compact_odds(j: dict | None) -> list:
    out = []
    for it in (j or {}).get("items", []):
        name = it.get("provider", {}).get("name", "")
        if "live" in name.lower():
            continue
        h, a = it.get("homeTeamOdds", {}) or {}, it.get("awayTeamOdds", {}) or {}
        ml = lambda t: _am(((t.get("close") or {}).get("moneyLine") or {}).get("american", t.get("moneyLine")))
        out.append({"provider": name, "spread": it.get("spread"), "total": it.get("overUnder"),
                    "ml_home": ml(h), "ml_away": ml(a), "over_odds": it.get("overOdds"), "under_odds": it.get("underOdds"),
                    "home_spread_odds": h.get("spreadOdds"), "away_spread_odds": a.get("spreadOdds")})
    return out


def fetch_odds(event_id: str, cache: Path, league="nba") -> list:
    p = cache / "odds" / f"{event_id}.json"
    if p.exists():
        return json.loads(p.read_text())
    sport, lg = LEAGUES[league]["path"].split("/")
    items = compact_odds(_json(ODDS.format(sport=sport, lg=lg, e=event_id)))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(items))
    return items


def fetch_live_line(event_id: str, league="nba") -> dict:
    """Línea de consenso ACTUAL (mediana entre casas) de un partido próximo. Sin caché: las líneas se mueven."""
    L = LEAGUES[league]
    sport, lg = L["path"].split("/")
    return pick_line(compact_odds(_json(ODDS.format(sport=sport, lg=lg, e=event_id))), L["total_range"], L["max_spread"])


def keep_franchises(games: pd.DataFrame, n_teams: int = 30) -> pd.DataFrame:
    """Quita All-Star, exhibiciones y selecciones: solo las franquicias reales (las de más partidos)."""
    cnt = pd.concat([games["home_id"], games["away_id"]]).value_counts()
    valid = set(cnt.index[:n_teams])
    if len(cnt) < n_teams or cnt.iloc[n_teams - 1] < 0.5 * cnt.iloc[0]:
        raise RuntimeError("No se pudieron identificar las 30 franquicias NBA")
    return games[games["home_id"].isin(valid) & games["away_id"].isin(valid)].copy()


def fix_same_day_conflicts(games: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """Si un equipo figura dos veces el mismo día (error de fecha en ESPN), el partido más temprano pasa al día previo."""
    g = games.copy()
    moved = 0
    for _ in range(5):
        long = pd.concat([g[["event_id", "date", "kickoff_utc", "home_id"]].rename(columns={"home_id": "t"}),
                          g[["event_id", "date", "kickoff_utc", "away_id"]].rename(columns={"away_id": "t"})])
        dup = long[long.duplicated(["date", "t"], keep=False)].sort_values(["date", "t", "kickoff_utc"])
        if dup.empty:
            break
        early = dup.groupby(["date", "t"]).head(1)["event_id"].unique()
        g.loc[g["event_id"].isin(early), "date"] -= pd.Timedelta(days=1)
        g.loc[g["event_id"].isin(early), "day"] = g.loc[g["event_id"].isin(early), "date"].dt.strftime("%Y-%m-%d")
        moved += len(early)
    if verbose and moved:
        print(f"AVISO: {moved} partido(s) con fecha duplicada movidos al día anterior")
    return g


def season_days(season_year: int, today: date | None = None, league="nba"):
    today = today or date.today()
    L = LEAGUES[league]
    y0 = season_year - L["year_offset"]
    d = date(y0, *L["start"])
    end = min(date(y0 + (1 if L["end"] < L["start"] else 0), *L["end"]), today + timedelta(days=7))
    while d <= end:
        yield d
        d += timedelta(days=1)


def current_season(league="nba", today: date | None = None) -> int:
    t = today or date.today()
    start_year = t.year if t.month >= 8 else t.year - 1
    return start_year + LEAGUES[league]["year_offset"]


def _dec(ml):
    ml = float(ml)
    return 1 + ml / 100 if ml > 0 else 1 + 100 / abs(ml)


def pick_line(items: list, total_range=(150.0, 300.0), max_spread=30.0) -> dict:
    """Línea de consenso = MEDIANA entre casas (robusta a una casa con datos corruptos o en vivo).
    Se descartan totales fuera de rango y moneylines inválidos. Sin al menos 1 casa válida, queda vacío."""
    sp = [float(i["spread"]) for i in items if i.get("spread") is not None and abs(float(i["spread"])) <= max_spread]
    tt = [float(i["total"]) for i in items if i.get("total") is not None and total_range[0] <= float(i["total"]) <= total_range[1]]
    ph = []
    for i in items:
        h, a = i.get("ml_home"), i.get("ml_away")
        if h is None or a is None or h != h or a != a or abs(h) < 100 or abs(a) < 100:
            continue
        ih, ia = 1 / _dec(h), 1 / _dec(a)
        if 1.0 <= ih + ia <= 1.15:                 # margen razonable; descarta pares incoherentes
            ph.append(ih / (ih + ia))
    if not sp or not tt:
        return {}
    return {"provider": f"mediana({len(sp)})", "spread": float(np.median(sp)), "total": float(np.median(tt)),
            "p_home": float(np.median(ph)) if ph else None, "n_books": len(sp)}


def load_league(league="nba", first_season: int = 2016, last_season: int | None = None, cache_dir: str | None = None,
                workers=8, with_odds=True, verbose=True) -> pd.DataFrame:
    L = LEAGUES[league]
    cache = Path(cache_dir or f"data/{league}")
    last_season = last_season or current_season(league)
    days = [d for y in range(first_season, last_season + 1) for d in season_days(y, league=league)]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda d: fetch_day(d, cache, league=league), days))
    games = pd.DataFrame([r for day in res for r in day])
    if games.empty:
        raise RuntimeError("ESPN no devolvió partidos")
    games = games[games["season_type"].isin(KEEP_TYPES)].drop_duplicates("event_id").copy()
    dead = {"STATUS_CANCELED", "STATUS_POSTPONED", "STATUS_FORFEIT", "STATUS_UNCONTESTED", "STATUS_SUSPENDED", "STATUS_ABANDONED"}
    placeholder = games["home_id"].str.startswith("-") | games["away_id"].str.startswith("-")       # equipos "TBD"
    games = games[~games["status"].isin(dead) & ~placeholder].copy()
    if L["n_teams"]:
        games = keep_franchises(games, L["n_teams"])
    final = games["status"] == "STATUS_FINAL"
    if verbose:
        print(f"[{league}] Partidos ESPN: {len(games)} ({final.sum()} finalizados) | {games.day.min()} -> {games.day.max()}")
    if with_odds:
        ids = games.loc[final, "event_id"].tolist()
        with ThreadPoolExecutor(workers) as ex:
            odds = dict(zip(ids, ex.map(lambda e: fetch_odds(e, cache, league=league), ids)))
        lines = {e: pick_line(v, L["total_range"], L["max_spread"]) for e, v in odds.items()}
        for col in ("provider", "spread", "total", "p_home", "n_books"):
            games["line_" + col] = games["event_id"].map(lambda e, c=col: lines.get(e, {}).get(c))
        if verbose:
            print(f"[{league}] Finalizados con línea (spread+total): {100 * games.loc[final, 'line_spread'].notna().mean():.1f}%")
    games["date"] = pd.to_datetime(games["day"])
    games = fix_same_day_conflicts(games, verbose)
    return games.sort_values(["date", "kickoff_utc", "home"]).reset_index(drop=True)


def load_nba(first_season: int = 2016, last_season: int | None = None, cache_dir="data/nba", workers=8,
             with_odds=True, verbose=True) -> pd.DataFrame:
    return load_league("nba", first_season, last_season, cache_dir, workers, with_odds, verbose)
