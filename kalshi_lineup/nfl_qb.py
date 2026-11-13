"""NFL: valor del quarterback titular *as-of* y cambio respecto al titular habitual (investigación, solo sombra).

Titular del partido (información que se conoce antes del inicio en la gran mayoría de los casos):
  * si el titular habitual (el más frecuente en los últimos 6 partidos) o el del partido anterior lanzó al menos un
    pase hoy, él fue el titular;
  * si no aparece ninguno, el titular es el QB con más intentos hoy.
Esta regla evita la fuga más común: que una lesión DURANTE el partido convierta al suplente en "titular".
Del propio partido NO se usan yardas, touchdowns ni intercepciones para sus variables.

Valor = ANY/A (yardas netas ajustadas por intento) con memoria de 1 año y previa de 250 intentos hacia el nivel
de reemplazo. Variables (local menos visitante):
  qb_delta  valor del titular de hoy − valor del titular habitual (0 si es el mismo)
  qb_abs    valor del titular de hoy
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

SUMMARY_NFL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
QB_FEATURES = ["qb_delta", "qb_abs"]


def _split(x, sep):
    try:
        a, b = str(x).split(sep, 1)
        return float(a), float(b)
    except ValueError:
        return np.nan, np.nan


def _f(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return np.nan


def parse_nfl_summary(payload: dict, event_id: str) -> dict | None:
    """Resumen ESPN NFL -> {'event_id', 'qbs': [{pid, team_id, att, cmp, yds, td, int, sacks, sack_yds}]}."""
    if not isinstance(payload, dict):
        return None
    comp = ((payload.get("header") or {}).get("competitions") or [{}])[0]
    status = ((comp.get("status") or {}).get("type") or {}).get("name")
    if status and status != "STATUS_FINAL":
        return None
    qbs, teams = [], set()
    for tb in (payload.get("boxscore") or {}).get("players") or []:
        team_id = str((tb.get("team") or {}).get("id") or "")
        for grp in tb.get("statistics") or []:
            if str(grp.get("name", "")).lower() != "passing":
                continue
            keys, labels = grp.get("keys") or [], grp.get("labels") or []
            def idx(key, lab):
                return keys.index(key) if key in keys else (labels.index(lab) if lab in labels else None)
            i_ca, i_y, i_td, i_int, i_sk = (idx("completions/passingAttempts", "C/ATT"), idx("passingYards", "YDS"),
                                            idx("passingTouchdowns", "TD"), idx("interceptions", "INT"),
                                            idx("sacks-sackYardsLost", "SACKS"))
            for a in grp.get("athletes") or []:
                pid = str((a.get("athlete") or {}).get("id") or "")
                st = a.get("stats") or []
                g = lambda i: st[i] if i is not None and i < len(st) else None
                cmp_, att = _split(g(i_ca), "/") if g(i_ca) is not None else (np.nan, np.nan)
                sk, sky = _split(g(i_sk), "-") if g(i_sk) is not None else (0.0, 0.0)
                if not pid or not team_id or not np.isfinite(att):
                    continue
                teams.add(team_id)
                qbs.append({"pid": pid, "team_id": team_id, "att": att, "cmp": cmp_, "yds": _f(g(i_y)),
                            "td": _f(g(i_td)), "int": _f(g(i_int)), "sacks": sk if np.isfinite(sk) else 0.0,
                            "sack_yds": sky if np.isfinite(sky) else 0.0})
    if len(teams) != 2:
        return None
    return {"event_id": str(event_id), "qbs": qbs}


def anya(r: dict) -> tuple[float, float]:
    """(numerador, denominador) de ANY/A para un partido."""
    v = lambda k: 0.0 if r.get(k) is None or not np.isfinite(r.get(k)) else float(r[k])
    return v("yds") + 20 * v("td") - 45 * v("int") - v("sack_yds"), v("att") + v("sacks")


def starter_for(team_qbs: list, previous, regular=None) -> str | None:
    """Titular de hoy: el habitual o el del partido anterior si lanzó >=1 pase; si no, el de más intentos."""
    if not team_qbs:
        return None
    for cand in (regular, previous):
        if cand and any(q["pid"] == cand and q["att"] >= 1 for q in team_qbs):
            return cand
    return max(team_qbs, key=lambda q: q["att"])["pid"]


def regular_starter(history) -> str | None:
    """El titular más frecuente en los últimos partidos (empate: el más reciente)."""
    h = [x for x in history if x]
    if not h:
        return None
    counts = {p: h.count(p) for p in h}
    best = max(counts.values())
    return next(p for p in reversed(h) if counts[p] == best)


class QBRater:
    def __init__(self, halflife_days=365.0, prior_att=250.0, replacement=4.0):
        self.h, self.prior, self.repl = halflife_days, prior_att, replacement
        self.s = {}                          # pid -> [num, den, last_date]
        self.hist = defaultdict(lambda: deque(maxlen=6))   # team -> titulares recientes

    def value(self, pid, date) -> float:
        x = self.s.get(pid)
        if not x:
            return self.repl
        f = 0.5 ** (max((date - x[2]).days, 0) / self.h)
        return (x[0] * f + self.repl * self.prior) / (x[1] * f + self.prior)

    def update(self, date, box):
        for q in box["qbs"]:
            num, den = anya(q)
            if den <= 0:
                continue
            x = self.s.setdefault(q["pid"], [0.0, 0.0, date])
            f = 0.5 ** (max((date - x[2]).days, 0) / self.h)
            x[0], x[1], x[2] = x[0] * f + num, x[1] * f + den, date

    def process(self, games: pd.DataFrame, boxes: dict) -> pd.DataFrame:
        d = games[["event_id", "date", "home_id", "away_id"]].copy()
        d["event_id"] = d["event_id"].astype(str)
        rows = []
        for date, day in d.sort_values("date").groupby("date", sort=True):
            pend = []
            for r in day.itertuples():
                b = boxes.get(r.event_id)
                if b is None:
                    rows.append({"event_id": r.event_id, "has_box": 0, "qb_delta": 0.0, "qb_abs": 0.0, "qb_change_h": 0, "qb_change_a": 0})
                    continue
                side = {}
                for t in (str(r.home_id), str(r.away_id)):
                    hist = self.hist[t]
                    reg = regular_starter(hist)
                    prev = hist[-1] if hist else None
                    today = starter_for([q for q in b["qbs"] if q["team_id"] == t], prev, reg)
                    v_today = self.value(today, date) if today else self.repl
                    v_reg = self.value(reg, date) if reg else v_today
                    side[t] = (v_today - v_reg, v_today, int(reg is not None and today != reg), today)
                h, a = side[str(r.home_id)], side[str(r.away_id)]
                rows.append({"event_id": r.event_id, "has_box": 1, "qb_delta": h[0] - a[0], "qb_abs": h[1] - a[1],
                             "qb_change_h": h[2], "qb_change_a": a[2]})
                pend.append((b, {str(r.home_id): h[3], str(r.away_id): a[3]}))
            for b, starters in pend:
                self.update(date, b)
                for t, pid in starters.items():
                    if pid:
                        self.hist[t].append(pid)
        return pd.DataFrame(rows)
