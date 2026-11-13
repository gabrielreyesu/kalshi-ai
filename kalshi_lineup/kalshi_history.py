"""Precios históricos de Kalshi ANTES de cada partido (mercados de ganador liquidados), con caché en disco.

Fuentes (API pública, sin clave):
  * /historical/markets?series_ticker=...            mercados antiguos ya archivados
  * /markets?series_ticker=...&status=settled        mercados liquidados recientes
  * /historical/markets/{t}/candlesticks  y  /series/{s}/markets/{t}/candlesticks   velas horarias (bid/ask)
Para cada partido se toma la última vela que CIERRA al menos `lead` minutos antes del inicio (sin mirar el futuro).
Kalshi limita la frecuencia: se espera entre peticiones y se reintenta con espera creciente ante 429.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"
_S = requests.Session()
_S.headers["User-Agent"] = "MochaPicks-research/1.0 (+https://github.com/gabrielreyesu/mocha-picks)"
PAUSE = 0.35


def get(path, params=None, retries=6):
    for i in range(retries):
        try:
            r = _S.get(BASE + path, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(2 * (i + 1)); continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 * (i + 1)); continue
        time.sleep(PAUSE)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Kalshi no respondió: {path}")


def list_markets(series: str, cache: Path) -> list:
    """Todos los mercados liquidados de la serie (archivados + recientes), con caché de 12 h."""
    p = cache / f"{series}_markets.json"
    if p.exists() and time.time() - p.stat().st_mtime < 12 * 3600:
        return json.loads(p.read_text())
    out = []
    for path, extra, src in (("/historical/markets", {}, "historical"), ("/markets", {"status": "settled"}, "live")):
        cur = None
        while True:
            j = get(path, {"series_ticker": series, "limit": 1000, **extra, **({"cursor": cur} if cur else {})}) or {}
            ms = j.get("markets", [])
            out += [{**m, "_src": src} for m in ms]
            cur = j.get("cursor")
            if not cur or not ms:
                break
    seen, uniq = set(), []
    for m in out:
        if m["ticker"] not in seen:
            seen.add(m["ticker"]); uniq.append(m)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(uniq))
    return uniq


def candles(series: str, m: dict, start: pd.Timestamp, end: pd.Timestamp, cache: Path) -> list:
    p = cache / "candles" / f"{m['ticker']}.json"
    if p.exists():
        return json.loads(p.read_text())
    path = (f"/historical/markets/{m['ticker']}/candlesticks" if m.get("_src") == "historical"
            else f"/series/{series}/markets/{m['ticker']}/candlesticks")
    j = get(path, {"start_ts": int(start.timestamp()), "end_ts": int(end.timestamp()), "period_interval": 60})
    c = (j or {}).get("candlesticks", [])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(c))
    return c


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def price_before(c: list, kickoff: pd.Timestamp, lead_min: int) -> dict:
    """Última vela que cierra <= inicio - lead. mid = (bid+ask)/2 si ambos existen y el spread es razonable."""
    cut = (kickoff - pd.Timedelta(minutes=lead_min)).timestamp()
    ok = [x for x in c if x.get("end_period_ts", 1e18) <= cut]
    if not ok:
        return {"mid": np.nan, "bid": np.nan, "ask": np.nan, "spread": np.nan, "age_h": np.nan}
    x = ok[-1]
    cl = lambda k: (x.get(k) or {}).get("close", (x.get(k) or {}).get("close_dollars"))   # archivados: close; recientes: close_dollars
    bid, ask, last = _num(cl("yes_bid")), _num(cl("yes_ask")), _num(cl("price"))
    mid = (bid + ask) / 2 if np.isfinite(bid) and np.isfinite(ask) and 0 < ask - bid <= 0.10 else last
    return {"mid": mid, "bid": bid, "ask": ask, "spread": ask - bid, "age_h": (kickoff.timestamp() - x["end_period_ts"]) / 3600}


def pregame_prices(series: str, league: str, games: pd.DataFrame, cache_dir="data/kalshi", leads=(60, 360, 1440)) -> pd.DataFrame:
    """Una fila por partido ESPN con precio Kalshi del LOCAL a varios plazos antes del inicio."""
    from kalshi_nba.kalshi import attach_games, parse_markets
    cache = Path(cache_dir)
    ms = list_markets(series, cache)
    by_ticker = {m["ticker"]: m for m in ms}
    mk = attach_games(parse_markets("winner", ms), games, league)
    mk = mk[mk["side"] == "home"].drop_duplicates("event_id")
    kick = games.set_index(games["event_id"].astype(str))["kickoff_utc"]
    rows = []
    for r in mk.itertuples():
        k = pd.Timestamp(kick[str(r.event_id)]).tz_convert("UTC") if pd.Timestamp(kick[str(r.event_id)]).tzinfo else pd.Timestamp(kick[str(r.event_id)], tz="UTC")
        m = by_ticker[r.ticker]
        try:
            c = candles(series, m, k - pd.Timedelta(days=3), k, cache)
        except RuntimeError:
            continue
        row = {"event_id": str(r.event_id), "ticker": r.ticker, "result": m.get("result")}
        for L in leads:
            pr = price_before(c, k, L)
            row.update({f"k{L}_{n}": v for n, v in pr.items()})
        rows.append(row)
    return pd.DataFrame(rows)
