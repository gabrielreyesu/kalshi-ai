"""Fútbol: precios históricos de Kalshi (local / empate / visitante) antes del inicio, emparejados con ESPN.

Reutiliza el emparejamiento de producción (kalshi_intl.cli.match_kalshi: nombres de ambos equipos y hora ±36 h, único
o se descarta) y las velas de kalshi_history. Salida: data/kalshi/soccer_pregame.csv con k60_H/D/A (mid normalizado).
  python -m kalshi_lineup.kalshi_soccer_history [--limit N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_intl.data import load_group

from .kalshi_history import candles, list_markets, prefetch_batch, price_before

SERIES = {"clubes": ["KXUCLGAME", "KXUELGAME", "KXUECLGAME", "KXEPLGAME", "KXLALIGAGAME", "KXSERIEAGAME",
                     "KXBUNDESLIGAGAME", "KXLIGUE1GAME"],
          "selecciones": ["KXUEFANLGAME", "KXFIFAGAME", "KXWCGAME", "KXINTLFRIENDLYGAME"]}
ORDER = [("clubes", "KXUCLGAME"), ("clubes", "KXUELGAME"), ("clubes", "KXUECLGAME"), ("clubes", "KXEPLGAME"),
         ("selecciones", "KXUEFANLGAME"), ("selecciones", "KXFIFAGAME"), ("selecciones", "KXWCGAME"),
         ("selecciones", "KXINTLFRIENDLYGAME"), ("clubes", "KXLALIGAGAME"), ("clubes", "KXSERIEAGAME"),
         ("clubes", "KXBUNDESLIGAGAME"), ("clubes", "KXLIGUE1GAME")]


def settled_events(series: str, cache: Path) -> list:
    ev = {}
    for m in list_markets(series, cache):
        e = ev.setdefault(m["event_ticker"], {"event": m["event_ticker"], "series": series, "teams": [], "tie": None,
                                              "when": pd.Timestamp(m.get("occurrence_datetime") or m["close_time"])})
        o = {"name": (m.get("yes_sub_title") or "").strip(), "m": m, "bid": None, "ask": None, "last": None}
        if o["name"].lower() in ("tie", "draw"):
            e["tie"] = o
        else:
            e["teams"].append(o)
    return [e for e in ev.values() if len(e["teams"]) == 2 and e["tie"]]


def match(group_games: pd.DataFrame, events: list) -> list:
    """[(event_id ESPN, kickoff, {H: market, D: market, A: market}, series)] con emparejamiento único."""
    from kalshi_soccer.kalshi import _same
    g = group_games[group_games["final"]].copy()
    out = []
    for e in events:
        c = g[(g["kick"] - e["when"]).abs() <= pd.Timedelta(hours=36)]
        hits = []
        for x in c.itertuples():
            side = {}
            for o in e["teams"]:
                mh, ma = _same(o["name"], x.home), _same(o["name"], x.away)
                if mh != ma:
                    side["H" if mh else "A"] = o["m"]
            if set(side) == {"H", "A"}:
                hits.append((str(x.event_id), x.kick, {**side, "D": e["tie"]["m"]}))
        if len(hits) == 1:
            out.append((*hits[0], e["series"]))
    return out


def build(cache_dir="data/kalshi", limit=None) -> pd.DataFrame:
    cache = Path(cache_dir)
    rows = []
    groups = {}
    for grp, s in ORDER:
        if grp not in groups:
            groups[grp] = load_group(grp, verbose=False)
        g = groups[grp]
        if True:
            ms = match(g, settled_events(s, cache))
            print(f"{s}: {len(ms)} partidos emparejados", flush=True)
            got = prefetch_batch([(mk[c]["ticker"], kick) for _, kick, mk, _ in ms[:limit] for c in "HDA" if mk[c].get("_src") == "live"], cache)
            print(f"   velas descargadas por lotes: {got}", flush=True)
            for eid, kick, mk, ser in ms[:limit]:
                row = {"event_id": eid, "group": grp, "series": ser}
                ok = True
                for c in "HDA":
                    try:
                        cs = candles(ser, mk[c], kick - pd.Timedelta(days=3), kick, cache)
                    except RuntimeError:
                        ok = False; break
                    for L in (60, 1440):
                        row[f"k{L}_{c}"] = price_before(cs, kick, L)["mid"]
                if ok:
                    rows.append(row)
            pd.DataFrame(rows).to_csv(cache / "soccer_pregame_partial.csv", index=False)      # avance utilizable
    d = pd.DataFrame(rows)
    for L in (60, 1440):
        cols = [f"k{L}_{c}" for c in "HDA"]
        s = d[cols].sum(1)
        bad = d[cols].isna().any(axis=1) | (s <= 0)
        d.loc[~bad, cols] = d.loc[~bad, cols].div(s[~bad], axis=0)          # quitar el margen: normalizar a 1
        d.loc[bad, cols] = np.nan
    d.to_csv(cache / "soccer_pregame.csv", index=False)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    d = build(limit=a.limit)
    print("filas:", len(d), "| con precio T-60:", int(d["k60_H"].notna().sum()))


if __name__ == "__main__":
    main()
