"""Precios de Polymarket (API pública, solo lectura) para el mercado de GANADOR de NBA y NFL.

Se une cada evento con un partido de ESPN por fecha de inicio (±6 h) y por el apodo del equipo
("Mavericks" dentro de "Dallas Mavericks"). Si la coincidencia no es única y clara, queda sin precio.
"""
from __future__ import annotations

import json
import re
import time

import pandas as pd
import requests

G = "https://gamma-api.polymarket.com"
SLUG = {"nba": re.compile(r"^nba-[a-z0-9]+-[a-z0-9]+-\d{4}-\d{2}-\d{2}$"), "nfl": re.compile(r"^nfl-[a-z0-9]+-[a-z0-9]+-\d{4}-\d{2}-\d{2}$")}


def _get(path, params, retries=4):
    for i in range(retries):
        try:
            r = requests.get(G + path, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"Polymarket no respondió: {path}")


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fetch_games(league: str) -> list:
    """[{slug, start, outcomes:[{name, price}], bid, ask}] — bid/ask corresponden al PRIMER resultado."""
    sid = next(str(x["series"]) for x in _get("/sports", {}) if x.get("sport") == league)
    out, off = [], 0
    while True:
        page = _get("/events", {"series_id": sid, "closed": "false", "limit": 100, "offset": off})
        for e in page:
            if not SLUG[league].match(e.get("slug", "")):
                continue
            for m in e.get("markets", []):
                if m.get("sportsMarketType") != "moneyline":
                    continue
                names, prices = json.loads(m.get("outcomes") or "[]"), json.loads(m.get("outcomePrices") or "[]")
                if len(names) == 2 and len(prices) == 2:
                    out.append({"slug": e["slug"], "start": pd.Timestamp(m.get("gameStartTime") or e.get("endDate"), tz="UTC") if not str(m.get("gameStartTime") or e.get("endDate")).endswith(("Z", "+00")) else pd.Timestamp(m.get("gameStartTime") or e.get("endDate")),
                                "outcomes": [{"name": n, "price": _num(p)} for n, p in zip(names, prices)],
                                "bid": _num(m.get("bestBid")), "ask": _num(m.get("bestAsk"))})
        if len(page) < 100:
            return out
        off += 100


def attach(games: pd.DataFrame, pm: list) -> pd.DataFrame:
    """games: event_id, kickoff_utc, home, away -> event_id, pm_home (prob. implícita del local), pm_bid_home, pm_ask_home, pm_slug."""
    rows = []
    for g in games.itertuples():
        kick = pd.Timestamp(g.kickoff_utc)
        kick = kick.tz_localize("UTC") if kick.tzinfo is None else kick
        hits = []
        for e in pm:
            start = e["start"].tz_localize("UTC") if e["start"].tzinfo is None else e["start"]
            if abs((start - kick).total_seconds()) > 6 * 3600:
                continue
            side = {}
            for i, o in enumerate(e["outcomes"]):
                inh, ina = o["name"].lower() in g.home.lower(), o["name"].lower() in g.away.lower()
                if inh != ina and len(o["name"]) >= 3:
                    side["home" if inh else "away"] = i
            if set(side) == {"home", "away"}:
                hits.append((e, side["home"]))
        if len(hits) != 1:
            continue
        e, ih = hits[0]
        p = e["outcomes"][ih]["price"]
        bid, ask = (e["bid"], e["ask"]) if ih == 0 else ((1 - e["ask"]) if e["ask"] is not None else None, (1 - e["bid"]) if e["bid"] is not None else None)
        mid = (bid + ask) / 2 if bid is not None and ask is not None and 0 < bid <= ask < 1 else p
        rows.append({"event_id": g.event_id, "pm_home": mid, "pm_bid_home": bid, "pm_ask_home": ask, "pm_slug": e["slug"]})
    return pd.DataFrame(rows, columns=["event_id", "pm_home", "pm_bid_home", "pm_ask_home", "pm_slug"])
