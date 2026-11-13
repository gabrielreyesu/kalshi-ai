"""Fútbol (ligas): ¿qué se juega cada equipo? Clasificación reconstruida *as-of* y "nada en juego" (investigación).

Regla fijada antes de ver resultados (STAKES_RESEARCH.md):
  * Clasificación con los partidos de la liga jugados ANTES de la fecha (3/1/0 puntos). Partidos restantes = 2·(N−1) − jugados.
  * Rango posible de posición final, ignorando desempates: mejor = 1 + nº de rivales que ya no puede alcanzar;
    peor = 1 + nº de rivales que aún pueden igualarlo o superarlo.
  * Fronteras por liga: título (1.º), Europa (top E) y descenso (N − R). Un equipo tiene algo en juego si su rango
    posible cruza alguna frontera. "Nada en juego" = no cruza ninguna (ya decidido arriba, en medio o abajo).
Variables (local − visitante, positivo favorece al local): stake_diff = juega_algo_local − juega_algo_visitante;
también nostake_h / nostake_a por separado. Solo partidos de liga (no copas).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

# código ESPN -> (plazas europeas E, descenso R). Aproximación estable 2016-2026.
ZONES = {"eng.1": (4, 3), "esp.1": (4, 3), "ita.1": (4, 3), "ger.1": (4, 3), "fra.1": (3, 3), "por.1": (2, 3), "ned.1": (2, 3)}
STAKE_FEATURES = ["stake_diff", "nostake_h", "nostake_a"]


def season_of(date) -> int:
    d = pd.Timestamp(date)
    return d.year if d.month >= 7 else d.year - 1


def position_range(team, pts: dict, left: dict) -> tuple[int, int]:
    mine, my_max = pts[team], pts[team] + 3 * left[team]
    others = [t for t in pts if t != team]
    best = 1 + sum(1 for t in others if pts[t] > my_max)
    worst = 1 + sum(1 for t in others if pts[t] + 3 * left[t] >= mine)
    return best, worst


def has_stake(best: int, worst: int, n: int, e: int, r: int) -> bool:
    for b in (1, e, n - r):                     # frontera entre la posición b y b+1
        if best <= b < worst:
            return True
    return False


def compute(games: pd.DataFrame) -> pd.DataFrame:
    """games: event_id, code, date, home_id, away_id, hs, as_, final. Devuelve variables por partido de liga."""
    g = games[games["code"].isin(ZONES)].copy()
    g["season"] = g["date"].map(season_of)
    rows = []
    for (code, season), s in g.groupby(["code", "season"]):
        teams = sorted(set(s["home_id"]) | set(s["away_id"]))
        n = len(teams)
        if n < 10:
            continue
        total = 2 * (n - 1)
        e, r = ZONES[code]
        pts, played = defaultdict(int), defaultdict(int)
        for t in teams:
            pts[t], played[t] = 0, 0
        for date, day in s.sort_values("date").groupby("date", sort=True):
            left = {t: max(total - played[t], 0) for t in teams}
            for x in day.itertuples():
                st = {}
                for side, t in (("h", x.home_id), ("a", x.away_id)):
                    b, w = position_range(t, pts, left)
                    st[side] = has_stake(b, w, n, e, r)
                rows.append({"event_id": str(x.event_id), "stake_diff": int(st["h"]) - int(st["a"]),
                             "nostake_h": int(not st["h"]), "nostake_a": int(not st["a"]),
                             "games_left_min": min(left[x.home_id], left[x.away_id])})
            for x in day.itertuples():
                if x.final and np.isfinite(x.hs) and np.isfinite(x.as_):
                    played[x.home_id] += 1; played[x.away_id] += 1
                    if x.hs > x.as_:
                        pts[x.home_id] += 3
                    elif x.hs < x.as_:
                        pts[x.away_id] += 3
                    else:
                        pts[x.home_id] += 1; pts[x.away_id] += 1
    return pd.DataFrame(rows)


# ---- Torneos de grupos de selecciones -------------------------------------------------------------------------
# Ediciones (ventanas de fase de grupos). Nations League: todas las posiciones cuentan (1.º-2.º a cuartos/ascenso,
# 3.º promoción, 4.º desciende) -> "nada en juego" = posición final ya decidida. Eliminatorias UEFA: 1.º directo,
# 2.º repesca -> fronteras 1 y 2.
GROUP_EDITIONS = {
    "uefa.nations": [("2018-09-01", "2018-11-30"), ("2020-09-01", "2020-11-30"), ("2022-06-01", "2022-09-30"),
                     ("2024-09-01", "2024-11-30"), ("2026-09-01", "2026-11-30")],
    "fifa.worldq.uefa": [("2016-09-01", "2017-11-30"), ("2021-03-01", "2021-11-30"), ("2025-03-01", "2025-11-30")],
}
GROUP_BOUNDARIES = {"uefa.nations": None, "fifa.worldq.uefa": (1, 2)}


def infer_groups(matches: pd.DataFrame) -> dict:
    """Componentes conexas de quién juega contra quién = grupos. {team: group_key}."""
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for h, a in zip(matches["home_id"], matches["away_id"]):
        ra, rb = find(h), find(a)
        if ra != rb:
            parent[ra] = rb
    return {t: find(t) for t in parent}


def compute_groups(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, editions in GROUP_EDITIONS.items():
        g = games[games["code"] == code]
        for lo, hi in editions:
            ed = g[(g["date"] >= pd.Timestamp(lo)) & (g["date"] <= pd.Timestamp(hi))]
            if ed.empty:
                continue
            grp = infer_groups(ed)
            for key in set(grp.values()):
                teams = sorted(t for t, k in grp.items() if k == key)
                n = len(teams)
                if not 3 <= n <= 6:
                    continue
                gm = ed[ed["home_id"].isin(teams)]
                total = 2 * (n - 1)
                pts = {t: 0 for t in teams}; played = {t: 0 for t in teams}
                for date, day in gm.sort_values("date").groupby("date", sort=True):
                    left = {t: max(total - played[t], 0) for t in teams}
                    for x in day.itertuples():
                        st = {}
                        for side, t in (("h", x.home_id), ("a", x.away_id)):
                            b, w = position_range(t, pts, left)
                            bounds = GROUP_BOUNDARIES[code]
                            st[side] = (b != w) if bounds is None else any(b <= k < w for k in bounds if k < n)
                        rows.append({"event_id": str(x.event_id), "stake_diff": int(st["h"]) - int(st["a"]),
                                     "nostake_h": int(not st["h"]), "nostake_a": int(not st["a"]),
                                     "games_left_min": min(left[x.home_id], left[x.away_id])})
                    for x in day.itertuples():
                        if x.final and np.isfinite(x.hs) and np.isfinite(x.as_):
                            played[x.home_id] += 1; played[x.away_id] += 1
                            if x.hs > x.as_:
                                pts[x.home_id] += 3
                            elif x.hs < x.as_:
                                pts[x.away_id] += 3
                            else:
                                pts[x.home_id] += 1; pts[x.away_id] += 1
    return pd.DataFrame(rows)
