"""Valor de jugadores *as-of* y fuerza de la alineación del día, sin fugas de datos futuros.

Regla central (igual que kalshi_nba.engine): todos los partidos de una fecha se calculan con el estado ANTERIOR a esa
fecha, y solo después se actualiza con los boxscores del día.

Del boxscore del propio partido solo se usa lo que se conoce ANTES del salto inicial:
  * quién es titular (se anuncia ~30 min antes);
  * quién está disponible (activo). Proxy: jugó, o "DNP - decisión del técnico". Inactivo/lesión/descanso = no.
Nunca se usan minutos, puntos ni +/- del propio partido para sus variables.

Variables (positivo = favorece al local):
  lineup_gs   cambio de fuerza (game score) de los disponibles hoy vs la media de sus últimas 10 alineaciones,
              local menos visitante.
  start_gs    lo mismo, solo con los 5 titulares.
  lineup_pm   como lineup_gs pero con +/- por minuto (más ruidoso).
  abs_gs      fuerza absoluta de los disponibles hoy, local menos visitante.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

LINEUP_FEATURES = ["lineup_gs", "start_gs", "lineup_pm", "abs_gs"]


@dataclass(frozen=True)
class RaterConfig:
    halflife_days: float = 120.0     # memoria del valor por minuto
    prior_min: float = 300.0         # minutos de "previa" hacia el nivel de reemplazo (game score)
    prior_min_pm: float = 1000.0     # previa más fuerte para +/- (muy ruidoso)
    replacement: float = 0.75        # nivel de reemplazo = 75% de la media de la liga por minuto
    em_alpha: float = 0.3            # suavizado de minutos esperados
    new_player_min: float = 10.0     # minutos esperados de un jugador sin historial
    team_minutes: float = 240.0
    cap_minutes: float = 42.0
    window: int = 10                 # alineaciones recientes que definen "lo normal"


def game_score(r: dict) -> float:
    v = lambda k: 0.0 if r.get(k) is None or not np.isfinite(r.get(k)) else float(r[k])
    return (v("pts") + 0.4 * v("fgm") - 0.7 * v("fga") - 0.4 * (v("fta") - v("ftm")) + 0.7 * v("oreb")
            + 0.3 * v("dreb") + v("stl") + 0.7 * v("ast") + 0.7 * v("blk") - 0.4 * v("pf") - v("tov"))


def allocate(em: np.ndarray, total: float, cap: float) -> np.ndarray:
    """Reparte 'total' minutos en proporción a los minutos esperados, con tope por jugador."""
    em = np.maximum(np.asarray(em, float), 0.0)
    if em.sum() <= 0:
        return np.full(len(em), total / max(len(em), 1))
    m = em * total / em.sum()
    for _ in range(5):
        over = m > cap
        if not over.any():
            break
        extra = (m[over] - cap).sum()
        m[over] = cap
        free = ~over
        if not free.any() or m[free].sum() <= 0:
            break
        m[free] += extra * m[free] / m[free].sum()
    return m


class LineupRater:
    def __init__(self, cfg: RaterConfig | None = None):
        self.cfg = cfg or RaterConfig()
        self.p = {}                                           # pid -> estado
        self.lg_gs, self.lg_min = 0.0, 0.0                    # media de liga (decaída) de game score por minuto
        self.lg_date = None
        self.recent = defaultdict(lambda: deque(maxlen=self.cfg.window))   # team -> [(disponibles, titulares)]

    # ---- valores ------------------------------------------------------
    def _decay(self, date, last):
        return 0.5 ** (max((date - last).days, 0) / self.cfg.halflife_days)

    def _lg_rate(self):
        return self.lg_gs / self.lg_min if self.lg_min > 0 else 0.45

    def rate(self, pid, date) -> tuple[float, float]:
        c, s = self.cfg, self.p.get(pid)
        repl = c.replacement * self._lg_rate()
        if s is None:
            return repl, 0.0
        f = self._decay(date, s["last"])
        wm, wg, wp = s["wm"] * f, s["wg"] * f, s["wp"] * f
        gs = (wg + repl * c.prior_min) / (wm + c.prior_min)
        pm = wp / (wm + c.prior_min_pm)
        return gs, pm

    def exp_min(self, pid):
        s = self.p.get(pid)
        return s["em"] if s else self.cfg.new_player_min

    def strength(self, pids, date) -> tuple[float, float]:
        pids = list(pids)
        if not pids:
            return np.nan, np.nan
        m = allocate([self.exp_min(i) for i in pids], self.cfg.team_minutes, self.cfg.cap_minutes)
        r = np.array([self.rate(i, date) for i in pids])
        return float((m * r[:, 0]).sum()), float((m * r[:, 1]).sum())

    def starters_strength(self, pids, date) -> float:
        return float(sum(self.rate(i, date)[0] * min(self.exp_min(i), self.cfg.cap_minutes) for i in pids)) if pids else np.nan

    # ---- variables de un partido --------------------------------------
    def team_features(self, team, date, avail, starters) -> dict:
        gs, pm = self.strength(avail, date)
        st = self.starters_strength(starters, date)
        past = list(self.recent[team])
        if past and np.isfinite(gs):
            base = np.array([self.strength(a, date) for a, _ in past])
            base_st = np.nanmean([self.starters_strength(s, date) for _, s in past if s]) if any(s for _, s in past) else np.nan
            return {"gs": gs, "d_gs": gs - float(np.nanmean(base[:, 0])), "d_pm": pm - float(np.nanmean(base[:, 1])),
                    "d_st": (st - base_st) if np.isfinite(st) and np.isfinite(base_st) else 0.0, "n_past": len(past)}
        return {"gs": gs, "d_gs": 0.0, "d_pm": 0.0, "d_st": 0.0, "n_past": len(past)}

    # ---- actualización con el boxscore (DESPUÉS de la fecha) ----------
    def update(self, date, box: dict):
        c = self.cfg
        by_team = defaultdict(lambda: ([], []))
        for r in box["players"]:
            pid, team = r["pid"], r["team_id"]
            if r.get("avail"):
                by_team[team][0].append(pid)
                if r.get("starter"):
                    by_team[team][1].append(pid)
            mins = r.get("min")
            if not r.get("avail") or mins is None or not np.isfinite(mins):
                continue
            s = self.p.setdefault(pid, {"wm": 0.0, "wg": 0.0, "wp": 0.0, "em": c.new_player_min, "last": date})
            f = self._decay(date, s["last"])
            g = game_score(r)
            pm = r.get("pm") if r.get("pm") is not None and np.isfinite(r.get("pm")) else 0.0
            s["wm"], s["wg"], s["wp"] = s["wm"] * f + mins, s["wg"] * f + g, s["wp"] * f + pm
            s["em"] = (1 - c.em_alpha) * s["em"] + c.em_alpha * mins
            s["last"], s["team"] = date, team
            if mins > 0:
                if self.lg_date is not None:
                    lf = self._decay(date, self.lg_date)
                    self.lg_gs, self.lg_min = self.lg_gs * lf, self.lg_min * lf
                self.lg_date = date
                self.lg_gs += g; self.lg_min += mins
        for team, (avail, starters) in by_team.items():
            self.recent[team].append((tuple(avail), tuple(starters)))

    # ---- recorrido completo -------------------------------------------
    def process(self, games: pd.DataFrame, boxes: dict) -> pd.DataFrame:
        """games: event_id, date, home_id, away_id. boxes: event_id -> parse_summary(). Devuelve variables por partido."""
        d = games[["event_id", "date", "home_id", "away_id"]].copy()
        d["event_id"] = d["event_id"].astype(str)
        d = d.sort_values("date")
        rows = []
        for date, day in d.groupby("date", sort=True):
            pend = []
            for r in day.itertuples():
                b = boxes.get(r.event_id)
                row = {"event_id": r.event_id, "has_box": int(b is not None)}
                if b is None:
                    row.update({k: 0.0 for k in LINEUP_FEATURES}); row["n_past_min"] = 0
                    rows.append(row); continue
                side = {}
                for t in (str(r.home_id), str(r.away_id)):
                    pl = [p for p in b["players"] if p["team_id"] == t]
                    av = [p["pid"] for p in pl if p.get("avail")]
                    st = [p["pid"] for p in pl if p.get("avail") and p.get("starter")]
                    side[t] = self.team_features(t, date, av, st)
                h, a = side[str(r.home_id)], side[str(r.away_id)]
                ok = np.isfinite(h["gs"]) and np.isfinite(a["gs"])
                row.update({"lineup_gs": h["d_gs"] - a["d_gs"], "start_gs": h["d_st"] - a["d_st"],
                            "lineup_pm": h["d_pm"] - a["d_pm"], "abs_gs": (h["gs"] - a["gs"]) if ok else 0.0,
                            "n_past_min": min(h["n_past"], a["n_past"])})
                rows.append(row)
                pend.append(b)
            for b in pend:
                self.update(date, b)
        out = pd.DataFrame(rows)
        for k in LINEUP_FEATURES:
            out[k] = pd.to_numeric(out[k], errors="coerce").fillna(0.0)
        return out
