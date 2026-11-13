"""Fútbol: fuerza del once titular *as-of* desde los resúmenes históricos de ESPN (investigación, solo sombra).

El once oficial se publica ~60 min antes del inicio. Del propio partido solo se usa QUIÉN es titular; goles,
asistencias y suplentes que entraron solo actualizan el estado DESPUÉS de la fecha.

Variables (local menos visitante):
  xi_cont  continuidad: peso de los titulares habituales (titularidades recientes, con decaimiento) que juegan hoy,
           sobre el peso de los 11 más habituales. 1 = once de gala, menor = rotación/bajas.
  xi_ga    goles+asistencias por titularidad (con previa) del once de hoy − media de sus últimos 10 onces.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

SUMMARY_SOCCER = "https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/summary"
XI_FEATURES = ["xi_cont", "xi_ga"]


def _stat(entry, *names):
    for s in entry.get("stats") or []:
        if isinstance(s, dict) and str(s.get("name") or s.get("abbreviation") or "") in names:
            try:
                return float(s.get("value") if s.get("value") is not None else s.get("displayValue"))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def parse_soccer_summary(payload: dict, event_id: str) -> dict | None:
    """-> {'event_id', 'players': [{pid, team_id, starter, ga}]} solo si ambos equipos tienen exactamente 11 titulares."""
    if not isinstance(payload, dict):
        return None
    comp = ((payload.get("header") or {}).get("competitions") or [{}])[0]
    status = ((comp.get("status") or {}).get("type") or {})
    if status.get("completed") is False:
        return None
    players, starters = [], defaultdict(int)
    for block in payload.get("rosters") or []:
        team_id = str((block.get("team") or {}).get("id") or "")
        for e in block.get("roster") or []:
            pid = str((e.get("athlete") or {}).get("id") or "")
            if not pid or not team_id:
                continue
            st = e.get("starter") is True
            starters[team_id] += st
            players.append({"pid": pid, "team_id": team_id, "starter": st,
                            "ga": _stat(e, "totalGoals", "G", "goals") + _stat(e, "goalAssists", "A", "assists")})
    if len(starters) != 2 or any(n != 11 for n in starters.values()):
        return None
    return {"event_id": str(event_id), "players": players}


class XIRater:
    def __init__(self, decay=0.85, window=10, prior_starts=8.0, prior_ga=0.15):
        self.decay, self.window, self.prior_starts, self.prior_ga = decay, window, prior_starts, prior_ga
        self.w = defaultdict(lambda: defaultdict(float))       # team -> pid -> peso de titularidad decaído
        self.n_xi = defaultdict(int)
        self.recent = defaultdict(lambda: deque(maxlen=window))  # team -> onces anteriores
        self.p = defaultdict(lambda: [0.0, 0.0])                # pid -> [ga, titularidades]

    def ga_rate(self, pid):
        ga, n = self.p.get(pid, (0.0, 0.0))
        return (ga + self.prior_ga * self.prior_starts) / (n + self.prior_starts)

    def team(self, team, xi) -> tuple[float, float, bool]:
        if self.n_xi[team] < 3:
            return 0.0, 0.0, False
        w = self.w[team]
        top = sorted(w.values(), reverse=True)[:11]
        cont = sum(w.get(p, 0.0) for p in xi) / sum(top) if sum(top) > 0 else 1.0
        ga_today = sum(self.ga_rate(p) for p in xi)
        base = float(np.mean([sum(self.ga_rate(p) for p in old) for old in self.recent[team]]))
        return cont, ga_today - base, True

    def update(self, box):
        by_team = defaultdict(list)
        for r in box["players"]:
            if r["starter"]:
                by_team[r["team_id"]].append(r["pid"])
            s = self.p[r["pid"]]
            if r["starter"]:
                s[1] += 1
            s[0] += r["ga"] if (r["starter"] or r["ga"]) else 0.0
        for team, xi in by_team.items():
            w = self.w[team]
            for p in list(w):
                w[p] *= self.decay
                if w[p] < 0.01:
                    del w[p]
            for p in xi:
                w[p] += 1.0
            self.n_xi[team] += 1
            self.recent[team].append(tuple(xi))

    def process(self, games: pd.DataFrame, boxes: dict) -> pd.DataFrame:
        d = games[["event_id", "date", "home_id", "away_id"]].copy()
        d["event_id"] = d["event_id"].astype(str)
        rows = []
        for date, day in d.sort_values("date").groupby("date", sort=True):
            pend = []
            for r in day.itertuples():
                b = boxes.get(r.event_id)
                row = {"event_id": r.event_id, "has_xi": int(b is not None), "xi_cont": 0.0, "xi_ga": 0.0, "xi_ok": 0}
                if b is not None:
                    h = self.team(str(r.home_id), [p["pid"] for p in b["players"] if p["team_id"] == str(r.home_id) and p["starter"]])
                    a = self.team(str(r.away_id), [p["pid"] for p in b["players"] if p["team_id"] == str(r.away_id) and p["starter"]])
                    if h[2] and a[2]:
                        row.update(xi_cont=h[0] - a[0], xi_ga=h[1] - a[1], xi_ok=1)
                    pend.append(b)
                rows.append(row)
            for b in pend:
                self.update(b)
        return pd.DataFrame(rows)
