"""Mercados NBA de Kalshi (ganador, spread, total) unidos a partidos de ESPN por abreviaturas + fecha.
Solo lectura (API pública). Un evento que no coincide de forma única queda fuera: nunca se adivina."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from kalshi_soccer.kalshi import _f, _get

SERIES = {"winner": "KXNBAGAME", "spread": "KXNBASPREAD", "total": "KXNBATOTAL"}
LEAGUE_SERIES = {"nba": SERIES, "nfl": {"winner": "KXNFLGAME", "spread": "KXNFLSPREAD", "total": "KXNFLTOTAL"}}
NFL_ABBR = {"1": "ATL", "2": "BUF", "3": "CHI", "4": "CIN", "5": "CLE", "6": "DAL", "7": "DEN", "8": "DET", "9": "GB",
            "10": "TEN", "11": "IND", "12": "KC", "13": "LV", "14": "LAR", "15": "MIA", "16": "MIN", "17": "NE",
            "18": "NO", "19": "NYG", "20": "NYJ", "21": "PHI", "22": "ARI", "23": "PIT", "24": "LAC", "25": "SF",
            "26": "SEA", "27": "TB", "28": "WAS", "29": "CAR", "30": "JAC", "33": "BAL", "34": "HOU"}
ABBR = {"1": "ATL", "2": "BOS", "3": "NOP", "4": "CHI", "5": "CLE", "6": "DAL", "7": "DEN", "8": "DET", "9": "GSW",
        "10": "HOU", "11": "IND", "12": "LAC", "13": "LAL", "14": "MIA", "15": "MIL", "16": "MIN", "17": "BKN",
        "18": "NYK", "19": "ORL", "20": "PHI", "21": "PHX", "22": "POR", "23": "SAC", "24": "SAS", "25": "OKC",
        "26": "UTA", "27": "WAS", "28": "TOR", "29": "MEM", "30": "CHA"}


def fetch_open(kind: str, league: str = "nba") -> list:
    out, cur = [], None
    while True:
        p = {"series_ticker": LEAGUE_SERIES[league][kind], "status": "open", "limit": 200}
        if cur:
            p["cursor"] = cur
        j = _get("/markets", p)
        out += j.get("markets", [])
        cur = j.get("cursor")
        if not cur or not j.get("markets"):
            return out


def parse_markets(kind: str, markets: list) -> pd.DataFrame:
    rows = []
    for m in markets:
        ev = m["event_ticker"]
        pair = re.sub(r"^\d{2}[A-Z]{3}\d{2}", "", ev.split("-", 1)[1])          # 26NOV15CHALAC -> CHALAC
        suffix = m["ticker"].rsplit("-", 1)[1]
        team = re.match(r"[A-Z]+", suffix).group(0) if kind != "total" and re.match(r"[A-Z]+", suffix) else None
        rows.append({"kind": kind, "ticker": m["ticker"], "event": ev, "pair": pair, "team": team,
                     "strike": _f(m.get("floor_strike")), "title": m.get("title"),
                     "when": pd.Timestamp(m.get("occurrence_datetime") or m["close_time"]),
                     "bid": _f(m.get("yes_bid_dollars")), "ask": _f(m.get("yes_ask_dollars")),
                     "last": _f(m.get("last_price_dollars")), "vol": _f(m.get("volume_fp"))})
    return pd.DataFrame(rows)


def attach_games(mk: pd.DataFrame, games: pd.DataFrame, league: str = "nba") -> pd.DataFrame:
    """Agrega event_id y lado (home/away) a cada mercado. `games`: event_id, kickoff_utc, home_id, away_id."""
    if mk.empty or games.empty:
        return mk.assign(event_id=None, side=None)
    ab = {"nba": ABBR, "nfl": NFL_ABBR}[league]
    g = games.assign(h=games["home_id"].map(ab), a=games["away_id"].map(ab),
                     kick=pd.to_datetime(games["kickoff_utc"], utc=True))
    eid, side = [], []
    for r in mk.itertuples():
        c = g[((g["a"] + g["h"] == r.pair) | (g["h"] + g["a"] == r.pair))
              & ((g["kick"] - r.when).abs() <= pd.Timedelta(hours=30))]
        if len(c) != 1:
            eid.append(None); side.append(None); continue
        x = c.iloc[0]
        eid.append(x["event_id"])
        side.append(None if r.team is None else ("home" if r.team == x["h"] else ("away" if r.team == x["a"] else "?")))
    out = mk.assign(event_id=eid, side=side)
    return out[out["event_id"].notna() & (out["side"] != "?")]


def model_probability(mk: pd.DataFrame, preds: pd.DataFrame, models) -> pd.DataFrame:
    """p_model de cada mercado. preds: event_id, p_home, mu_margin, mu_total."""
    d = mk.merge(preds[["event_id", "p_home", "mu_margin", "mu_total"]], on="event_id", how="inner")
    for c in ("strike", "bid", "ask", "last"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    p = np.full(len(d), np.nan)
    w, s, t = (d["kind"] == k for k in ("winner", "spread", "total"))
    home = (d["side"] == "home").to_numpy()
    p[w.to_numpy()] = np.where(home, d["p_home"], 1 - d["p_home"])[w.to_numpy()]
    over = models.p_home_wins_by_more_than(d["mu_margin"], d["strike"])                    # local gana por más de x
    away = 1 - models.p_home_wins_by_more_than(d["mu_margin"], -d["strike"])               # P(margen < -x)
    p[s.to_numpy()] = np.where(home, over, away)[s.to_numpy()]
    p[t.to_numpy()] = np.asarray(models.p_total_over(d["mu_total"], d["strike"]))[t.to_numpy()]
    d["p_model"] = p
    d["mid"] = np.where(d["bid"].notna() & d["ask"].notna() & (d["ask"] > 0), (d["bid"] + d["ask"]) / 2, d["last"])
    d["ventaja_bruta"] = d["p_model"] - d["ask"]                                           # comprar Yes al ask, SIN comisiones
    return d


def settle(snap: pd.DataFrame, finals: pd.DataFrame) -> pd.DataFrame:
    """Resultado real (y) de cada mercado. finals: event_id, hs, as_."""
    d = snap.merge(finals[["event_id", "hs", "as_"]], on="event_id", how="inner")
    d["strike"] = pd.to_numeric(d["strike"], errors="coerce")
    mar, tot = d["hs"] - d["as_"], d["hs"] + d["as_"]
    home = d["side"] == "home"
    y = np.where(d["kind"] == "winner", np.where(home, mar > 0, mar < 0),
                 np.where(d["kind"] == "spread", np.where(home, mar > d["strike"], -mar > d["strike"]), tot > d["strike"]))
    return d.assign(y=y.astype(int))
