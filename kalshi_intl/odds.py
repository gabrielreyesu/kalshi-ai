"""Cuotas 1X2 y over/under de ESPN para fútbol (mediana entre casas, sin margen)."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_nba.data import _json

URL = "https://sports.core.api.espn.com/v2/sports/soccer/leagues/{code}/events/{e}/competitions/{e}/odds"


def _dec(ml):
    try:
        ml = float(str(ml).replace("+", ""))
    except (TypeError, ValueError):
        return np.nan
    if abs(ml) < 100:
        return np.nan
    return 1 + ml / 100 if ml > 0 else 1 + 100 / abs(ml)


def consensus(items: list) -> dict:
    P, O = [], []
    for it in items or []:
        if "live" in (it.get("provider", {}).get("name", "")).lower():
            continue
        h, a, d = (_dec((it.get(k) or {}).get("moneyLine")) for k in ("homeTeamOdds", "awayTeamOdds", "drawOdds"))
        if not np.isnan([h, a, d]).any():
            inv = np.array([1 / h, 1 / d, 1 / a])
            if 1.0 <= inv.sum() <= 1.2:
                P.append(inv / inv.sum())
        o, u, line = _dec(it.get("overOdds")), _dec(it.get("underOdds")), it.get("overUnder")
        if line == 2.5 and not np.isnan([o, u]).any() and 1.0 <= 1 / o + 1 / u <= 1.15:
            O.append((1 / o) / (1 / o + 1 / u))
    out = {}
    if P:
        m = np.median(np.array(P), axis=0); m = m / m.sum()
        out.update(mkt_H=float(m[0]), mkt_D=float(m[1]), mkt_A=float(m[2]), n_books=len(P))
    if O:
        out["mkt_over"] = float(np.median(O))
    return out


def fetch(code: str, event_id: str, cache: Path | None = None) -> dict:
    p = cache / f"{event_id}.json" if cache else None
    if p is not None and p.exists():
        return json.loads(p.read_text())
    out = consensus((_json(URL.format(code=code, e=event_id)) or {}).get("items", []))
    if p is not None:
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(out))
    return out


def attach(df: pd.DataFrame, cache_dir: str | None = "data/intl/odds", workers: int = 8) -> pd.DataFrame:
    cache = Path(cache_dir) if cache_dir else None
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda t: fetch(t[0], t[1], cache), zip(df["code"], df["event_id"])))
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(res)], axis=1)
