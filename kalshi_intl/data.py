"""Resultados de fútbol internacional desde ESPN (consultas por MES, con caché en disco).

Dos grupos, cada uno con un rating común para que la fuerza "viaje" entre competiciones:
  clubes       -> objetivo: Champions, Europa League, Conference. Alimentan el rating: ligas domésticas.
  selecciones  -> objetivo: Mundial, eliminatorias UEFA/CONMEBOL, Nations League, Euro, Copa América, amistosos.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SB = "https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/scoreboard"
GROUPS = {
    "clubes": {
        "target": {"uefa.champions": "Champions League", "uefa.europa": "Europa League", "uefa.europa.conf": "Conference League"},
        "feed": ["eng.1", "esp.1", "ita.1", "ger.1", "fra.1", "por.1", "ned.1", "bel.1", "tur.1", "sco.1", "aut.1", "sui.1",
                 "gre.1", "den.1", "nor.1", "swe.1", "cze.1", "cro.1", "ukr.1", "srb.1", "pol.1", "rou.1", "isr.1", "cyp.1"],
    },
    "selecciones": {
        "target": {"fifa.world": "Mundial", "fifa.worldq.uefa": "Eliminatorias UEFA", "fifa.worldq.conmebol": "Eliminatorias CONMEBOL",
                   "uefa.nations": "Nations League", "uefa.euro": "Eurocopa", "uefa.euroq": "Clasificación Eurocopa",
                   "conmebol.america": "Copa América", "fifa.friendly": "Amistoso"},
        "feed": ["fifa.worldq.concacaf", "fifa.worldq.afc", "fifa.worldq.caf", "concacaf.gold", "concacaf.nations.league",
                 "caf.nations", "afc.asian.cup"],
    },
}
_S = requests.Session()


def months(first_year: int, today: date | None = None):
    t = today or date.today()
    end = (t.year, t.month + 1) if t.month < 12 else (t.year + 1, 1)
    y, m = first_year, 7
    while (y, m) <= end:
        yield f"{y}{m:02d}"
        y, m = (y, m + 1) if m < 12 else (y + 1, 1)


def _parse(e: dict, code: str) -> dict | None:
    c = (e.get("competitions") or [None])[0]
    if not c:
        return None
    side = {t["homeAway"]: t for t in c.get("competitors", [])}
    if set(side) != {"home", "away"}:
        return None
    sc = lambda t: float(t["score"]) if str(t.get("score", "")).isdigit() else np.nan
    return {"event_id": str(e["id"]), "code": code, "kickoff_utc": e["date"], "status": e["status"]["type"]["name"],
            "home_id": str(side["home"]["team"]["id"]), "home": side["home"]["team"]["displayName"],
            "away_id": str(side["away"]["team"]["id"]), "away": side["away"]["team"]["displayName"],
            "hs": sc(side["home"]), "as_": sc(side["away"]), "neutral": bool(c.get("neutralSite", False))}


def fetch_month(code: str, ym: str, cache: Path, today: date | None = None) -> list:
    t = today or date.today()
    p = cache / code / f"{ym}.json"
    closed = (int(ym[:4]), int(ym[4:])) < ((t.year, t.month - 1) if t.month > 1 else (t.year - 1, 12))
    if p.exists() and closed:
        return json.loads(p.read_text())
    rows = None
    for _ in range(4):
        try:
            r = _S.get(SB.format(code=code), params={"dates": ym, "limit": 1000}, timeout=40)
        except requests.RequestException:
            continue
        if r.status_code == 200:
            rows = [x for x in (_parse(e, code) for e in r.json().get("events", [])) if x]
            break
        if r.status_code in (400, 404):          # liga o mes inexistente
            rows = []
            break
    if rows is None:
        if p.exists():
            return json.loads(p.read_text())
        raise RuntimeError(f"ESPN no respondió: {code} {ym}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows))
    return rows


def load_group(group: str, first_year: int = 2016, cache_dir: str = "data/intl", workers: int = 8, verbose=True) -> pd.DataFrame:
    G = GROUPS[group]
    codes = list(G["target"]) + G["feed"]
    jobs = [(c, ym) for c in codes for ym in months(first_year)]
    cache = Path(cache_dir)
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda j: fetch_month(j[0], j[1], cache), jobs))
    g = pd.DataFrame([r for rows in res for r in rows])
    if g.empty:
        raise RuntimeError("ESPN no devolvió partidos")
    dead = {"STATUS_CANCELED", "STATUS_POSTPONED", "STATUS_ABANDONED", "STATUS_SUSPENDED", "STATUS_FORFEIT"}
    g = g[~g["status"].isin(dead) & ~g["home_id"].str.startswith("-") & ~g["away_id"].str.startswith("-")]
    g = g.drop_duplicates("event_id").copy()
    g["final"] = g["status"].isin({"STATUS_FULL_TIME", "STATUS_FINAL", "STATUS_FINAL_AET", "STATUS_FINAL_PEN"}) & g["hs"].notna()
    g["kick"] = pd.to_datetime(g["kickoff_utc"], utc=True)
    g["date"] = g["kick"].dt.tz_localize(None).dt.normalize()
    g["target"] = g["code"].isin(G["target"])
    g["competicion"] = g["code"].map(G["target"]).fillna(g["code"])
    g["group"] = group
    if verbose:
        by = g.groupby("code").size()
        print(f"[{group}] partidos: {len(g)} ({int(g.final.sum())} finalizados) | {g.date.min().date()} -> {g.date.max().date()}")
        print("   por competición:", {k: int(v) for k, v in by.items()})
        print("   estados:", g["status"].value_counts().head(6).to_dict())
    return g.sort_values(["kick", "event_id"]).reset_index(drop=True)
