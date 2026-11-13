"""Boxscores históricos de NBA desde el resumen público de ESPN, en forma compacta y con caché.

Cada fila es un jugador en un partido: titular, si estuvo disponible (jugó o "DNP - decisión del técnico") y sus
estadísticas. Solo se guardan partidos con estado final. Un fallo de red NO se guarda en caché (se reintenta después).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"

# clave ESPN (keys) -> etiqueta alternativa (labels) -> nombre interno
FIELDS = {
    "minutes": ("MIN", "min"), "points": ("PTS", "pts"), "rebounds": ("REB", "reb"),
    "offensiveRebounds": ("OREB", "oreb"), "defensiveRebounds": ("DREB", "dreb"), "assists": ("AST", "ast"),
    "steals": ("STL", "stl"), "blocks": ("BLK", "blk"), "turnovers": ("TO", "tov"), "fouls": ("PF", "pf"),
    "plusMinus": ("+/-", "pm"),
    "fieldGoalsMade-fieldGoalsAttempted": ("FG", "fg"), "freeThrowsMade-freeThrowsAttempted": ("FT", "ft"),
}
STATS = ["min", "pts", "reb", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf", "pm", "fgm", "fga", "ftm", "fta"]


def _num(x):
    s = str(x).strip().replace("+", "")
    if not s or s in ("--", "-", "DNP"):
        return np.nan
    if ":" in s:                                   # "34:12"
        m, sec = s.split(":", 1)
        try:
            return float(m) + float(sec) / 60
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _pair(x):
    try:
        a, b = str(x).split("-", 1)
        return float(a), float(b)
    except ValueError:
        return np.nan, np.nan


def available(starter: bool, minutes: float, dnp: bool, reason: str) -> bool:
    """Disponible = jugó, o no jugó por decisión del técnico (estaba activo). Lesión/descanso/inactivo = no."""
    if starter or (np.isfinite(minutes) and minutes > 0):
        return True
    return bool(dnp) and "COACH" in str(reason).upper()


def parse_summary(payload: dict, event_id: str) -> dict | None:
    """Resumen ESPN -> {'event_id', 'players': [...]} o None si el partido no es final o no trae boxscore."""
    if not isinstance(payload, dict):
        return None
    comp = ((payload.get("header") or {}).get("competitions") or [{}])[0]
    status = ((comp.get("status") or {}).get("type") or {}).get("name")
    if status and status != "STATUS_FINAL":
        return None
    players = []
    for tb in (payload.get("boxscore") or {}).get("players") or []:
        team_id = str((tb.get("team") or {}).get("id") or "")
        for grp in tb.get("statistics") or []:
            keys, labels = grp.get("keys") or [], grp.get("labels") or grp.get("names") or []
            col = {}
            for k, (lab, name) in FIELDS.items():
                if k in keys:
                    col[name] = keys.index(k)
                elif lab in labels:
                    col[name] = labels.index(lab)
            for a in grp.get("athletes") or []:
                ath = a.get("athlete") or {}
                pid = str(ath.get("id") or "")
                if not pid or not team_id:
                    continue
                st = a.get("stats") or []
                get = lambda n: st[col[n]] if n in col and col[n] < len(st) else None
                row = {"pid": pid, "team_id": team_id, "name": ath.get("displayName", ""),
                       "starter": bool(a.get("starter", False)), "dnp": bool(a.get("didNotPlay", False)),
                       "reason": str(a.get("reason") or "")}
                for n in ("min", "pts", "reb", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf", "pm"):
                    row[n] = _num(get(n)) if get(n) is not None else np.nan
                row["fgm"], row["fga"] = _pair(get("fg")) if get("fg") is not None else (np.nan, np.nan)
                row["ftm"], row["fta"] = _pair(get("ft")) if get("ft") is not None else (np.nan, np.nan)
                row["avail"] = available(row["starter"], row["min"], row["dnp"], row["reason"])
                players.append(row)
    if not players:
        return None
    teams = {p["team_id"] for p in players}
    if len(teams) != 2:
        return None
    return {"event_id": str(event_id), "players": players}


def fetch_box(event_id: str, cache: Path, getter=None) -> dict | None:
    """Lee de caché o descarga. Se guarda en caché el resultado de un partido final, incluso vacío (ok=False)."""
    p = cache / f"{event_id}.json"
    if p.exists():
        d = json.loads(p.read_text())
        return d if d.get("ok") else None
    if getter is None:
        from kalshi_nba.data import _json as getter
    try:
        payload = getter(SUMMARY, {"event": str(event_id)})
    except RuntimeError:
        return None                                  # fallo transitorio: no se guarda
    parsed = parse_summary(payload, event_id) if payload else None
    final = bool(payload) and ((((payload.get("header") or {}).get("competitions") or [{}])[0].get("status") or {})
                               .get("type") or {}).get("name") == "STATUS_FINAL"
    if parsed is None and not final:
        return None
    out = {"ok": parsed is not None, **(parsed or {"event_id": str(event_id), "players": []})}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, allow_nan=True))
    return out if out["ok"] else None


def load_boxes(event_ids, cache_dir="data/nba/box", workers=8, getter=None) -> dict:
    cache = Path(cache_dir)
    ids = [str(e) for e in event_ids]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda e: fetch_box(e, cache, getter), ids))
    return {e: r for e, r in zip(ids, res) if r}
