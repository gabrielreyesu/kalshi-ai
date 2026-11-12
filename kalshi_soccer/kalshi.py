"""Precios de Kalshi (API pública de datos de mercado, sin autenticación) para ganador/empate de fútbol.

Solo lectura. Une cada evento de Kalshi con un partido del calendario por liga + hora + nombres de equipo;
si el emparejamiento no es único y claro, el partido queda SIN precio (nunca se adivina).
"""
from __future__ import annotations

import time
from datetime import timezone

import pandas as pd
import requests

from .names import _norm

BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = {"Premier League": "KXEPLGAME", "La Liga": "KXLALIGAGAME", "Serie A": "KXSERIEAGAME",
          "Bundesliga": "KXBUNDESLIGAGAME", "Ligue 1": "KXLIGUE1GAME"}


def _get(path: str, params: dict, retries: int = 6, session=None) -> dict:
    s = session or requests
    for i in range(retries):
        r = s.get(BASE + path, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(1.5 * (i + 1)); continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Kalshi: límite de peticiones persistente en {path}")


def _f(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def fetch_open_events(league: str, session=None) -> list[dict]:
    """Eventos abiertos de ganador/empate: {event, kickoff_utc, outcomes:[{kind,team,bid,ask,vol}]}."""
    markets, cursor = [], None
    while True:
        p = {"series_ticker": SERIES[league], "status": "open", "limit": 200}
        if cursor:
            p["cursor"] = cursor
        j = _get("/markets", p, session=session)
        markets += j.get("markets", [])
        cursor = j.get("cursor")
        if not cursor or not j.get("markets"):
            break
    ev: dict = {}
    for m in markets:
        e = ev.setdefault(m["event_ticker"], {"event": m["event_ticker"], "outcomes": [],
                                              "kickoff_utc": pd.Timestamp(m.get("occurrence_datetime") or m["close_time"])})
        sub = (m.get("yes_sub_title") or "").strip()
        is_tie = sub.lower() == "tie" or (m.get("title", "").lower().startswith("tie"))
        e["outcomes"].append({"kind": "D" if is_tie else "team", "team": None if is_tie else sub,
                              "bid": _f(m.get("yes_bid_dollars")), "ask": _f(m.get("yes_ask_dollars")),
                              "last": _f(m.get("last_price_dollars")), "vol": _f(m.get("volume_fp")),
                              "ticker": m["ticker"]})
    return [e for e in ev.values() if sum(o["kind"] == "D" for o in e["outcomes"]) == 1
            and sum(o["kind"] == "team" for o in e["outcomes"]) == 2]


# Nombre de Kalshi -> nombre en el histórico, solo para los casos que la coincidencia automática no resuelve
# o resuelve de forma ambigua. Cualquier otro nombre debe coincidir de forma única o el partido queda sin precio.
KALSHI_ALIASES = {
    ("La Liga", "Barcelona"): "Barcelona", ("La Liga", "Deportivo De La Coruna"): "La Coruna",
    ("La Liga", "Celta Vigo"): "Celta", ("La Liga", "Bilbao"): "Ath Bilbao",
    ("Serie A", "Milan"): "Milan", ("Bundesliga", "M\u00b4gladbach"): "M'gladbach",
    ("Bundesliga", "Bayern Munich"): "Bayern Munich", ("Ligue 1", "PSG"): "Paris SG",
    ("Ligue 1", "Stade Brest 29"): "Brest",
}


def team_matches(league: str, kname: str, api_name: str, hist_name: str) -> bool:
    alias = KALSHI_ALIASES.get((league, kname))
    if alias is not None:
        return alias == hist_name
    return _same(kname, api_name) or _same(kname, hist_name)


def _same(kname: str, other: str) -> bool:
    a, b = _norm(kname), _norm(other)
    return len(a) >= 4 and len(b) >= 4 and (a == b or a in b or b in a)


def attach_kalshi(preds: pd.DataFrame, fixtures: pd.DataFrame, events_by_league: dict) -> pd.DataFrame:
    """Agrega k_bid_/k_ask_/k_last_ para H/D/A y k_event. `fixtures` trae home_api/away_api por match_id."""
    fx = fixtures.set_index("match_id")[["home_api", "away_api"]]
    cols = {f"k_{k}_{c}": None for k in ("bid", "ask", "last") for c in "HDA"}
    out = preds.copy()
    for c in list(cols) + ["k_event", "k_vol"]:
        out[c] = None
    for i, r in out.iterrows():
        if str(r["match_id"]) not in fx.index:
            continue
        ha, aa = fx.loc[str(r["match_id"]), ["home_api", "away_api"]]
        kick = pd.Timestamp(r["kickoff_utc"])
        hits = []
        for e in events_by_league.get(r["competition"], []):
            if abs((e["kickoff_utc"] - kick).total_seconds()) > 36 * 3600:
                continue
            teams = [o for o in e["outcomes"] if o["kind"] == "team"]
            side = {}
            for o in teams:
                mh = team_matches(r["competition"], o["team"], ha, r["home"])
                ma = team_matches(r["competition"], o["team"], aa, r["away"])
                if mh != ma:                 # debe coincidir con exactamente un lado
                    side["H" if mh else "A"] = o
            if set(side) == {"H", "A"}:
                side["D"] = next(o for o in e["outcomes"] if o["kind"] == "D")
                hits.append((e, side))
        if len(hits) != 1:
            continue
        e, side = hits[0]
        for c in "HDA":
            o = side[c]
            out.at[i, f"k_bid_{c}"], out.at[i, f"k_ask_{c}"], out.at[i, f"k_last_{c}"] = o["bid"], o["ask"], o["last"]
        out.at[i, "k_event"] = e["event"]
        out.at[i, "k_vol"] = sum((o["vol"] or 0) for o in e["outcomes"])
    return out


def kalshi_mid(df: pd.DataFrame) -> pd.DataFrame:
    """Probabilidad implícita de Kalshi: punto medio bid/ask de cada resultado, normalizado a suma 1.
    Si falta bid o ask se usa el último precio; si tampoco hay, la fila queda vacía."""
    mid = {}
    for c in "HDA":
        b, a, l = (pd.to_numeric(df[f"k_{k}_{c}"], errors="coerce") for k in ("bid", "ask", "last"))
        m = (b + a) / 2
        mid[c] = m.where(b.notna() & a.notna() & (a > 0), l)
    M = pd.DataFrame(mid)
    return M.div(M.sum(axis=1), axis=0).add_prefix("k_mid_")


def edges(df: pd.DataFrame, min_edge: float = 0.0) -> pd.DataFrame:
    """Ventaja bruta del modelo sobre comprar 'Yes' al ask: p_modelo - ask (SIN comisiones)."""
    rows = []
    for _, r in df.iterrows():
        for c, name in (("H", r["home"]), ("D", "Empate"), ("A", r["away"])):
            ask, p = _f(r.get(f"k_ask_{c}")), _f(r.get(f"p_{c}"))
            if ask is None or p is None:
                continue
            rows.append({"liga": r["competition"], "kickoff_utc": r["kickoff_utc"], "partido": f"{r['home']} - {r['away']}",
                         "resultado": name, "p_modelo": round(p, 4), "ask": ask, "ventaja_bruta": round(p - ask, 4),
                         "volumen_evento": r.get("k_vol")})
    t = pd.DataFrame(rows)
    return t[t["ventaja_bruta"] >= min_edge].sort_values("ventaja_bruta", ascending=False) if len(t) else t
