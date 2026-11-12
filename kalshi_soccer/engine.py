"""Motor de fútbol sin fugas de datos.

Port fiel de la lógica V3/V4 del notebook (Elo con regresión por temporada +
Poisson con corrección Dixon-Coles), con dos cambios:
  * el estado vive en un objeto (SoccerEngine), no en variables globales de Colab;
  * de la misma matriz de marcadores salen TODOS los mercados (1X2, totales, BTTS, spreads).

Regla central: las variables de un partido solo usan partidos de fechas ANTERIORES.
Todos los partidos de una misma fecha se predicen antes de actualizar el estado.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import poisson

LEAGUES = {
    "Premier League": "E0",
    "La Liga": "SP1",
    "Serie A": "I1",
    "Bundesliga": "D1",
    "Ligue 1": "F1",
}
TOTAL_LINES = (1.5, 2.5, 3.5)


@dataclass(frozen=True)
class EngineConfig:
    k: float = 20.0
    home_elo: float = 65.0
    season_regress: float = 0.65
    window: int = 20
    prior: float = 8.0
    rho: float = -0.08
    max_goals: int = 12
    league_window: int = 300
    lam_min: float = 0.2
    lam_max: float = 4.5
    default_home_goals: float = 1.45
    default_away_goals: float = 1.15
    league_avg_floor: float = 0.5


class UnknownTeamError(ValueError):
    pass


def season_code_for_date(date) -> str:
    """'2627' para fechas entre jul-2026 y jun-2027. (Solo para partidos futuros/API;
    el histórico usa la etiqueta de temporada del archivo.)"""
    d = pd.Timestamp(date)
    start = d.year if d.month >= 7 else d.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def score_matrix(lh: float, la: float, cfg: EngineConfig, goals: np.ndarray) -> np.ndarray:
    m = np.outer(poisson.pmf(goals, lh), poisson.pmf(goals, la))
    m /= m.sum()
    rho = cfg.rho
    m[0, 0] *= 1 - lh * la * rho
    m[0, 1] *= 1 + lh * rho
    m[1, 0] *= 1 + la * rho
    m[1, 1] *= 1 - rho
    m = np.maximum(m, 0)
    m /= m.sum()
    return m


def derive_markets(m: np.ndarray) -> dict:
    """Probabilidades de todos los mercados a partir de la matriz (filas = goles local)."""
    n = m.shape[0]
    i = np.arange(n)
    total = i[:, None] + i[None, :]
    diff = i[:, None] - i[None, :]
    out = {
        "p_H": float(np.tril(m, -1).sum()),
        "p_D": float(np.trace(m)),
        "p_A": float(np.triu(m, 1).sum()),
        "p_btts": float(m[1:, 1:].sum()),
        "p_home_m1_5": float(m[diff >= 2].sum()),   # local gana por 2+
        "p_away_m1_5": float(m[diff <= -2].sum()),  # visitante gana por 2+
    }
    for line in TOTAL_LINES:
        out["p_over_" + str(line).replace(".", "_")] = float(m[total > line].sum())
    return out


def _smooth(values, baseline: float, prior: float) -> float:
    return (sum(values) + prior * baseline) / (len(values) + prior)


class SoccerEngine:
    def __init__(self, cfg: EngineConfig | None = None):
        self.cfg = cfg or EngineConfig()
        c = self.cfg
        self.elo = defaultdict(lambda: 1500.0)
        self.season_seen: dict = {}
        self.n_played: dict = defaultdict(int)
        self.hist = defaultdict(lambda: {
            "home_gf": deque(maxlen=c.window), "home_ga": deque(maxlen=c.window),
            "away_gf": deque(maxlen=c.window), "away_ga": deque(maxlen=c.window)})
        self.league_hist = defaultdict(lambda: deque(maxlen=c.league_window))
        self.goals = np.arange(c.max_goals + 1)
        self.last_date = None

    # ---- consulta -------------------------------------------------------
    def knows(self, league: str, team: str) -> bool:
        return self.n_played.get((league, team), 0) > 0

    def known_teams(self, league: str) -> list:
        return sorted(t for (l, t), n in self.n_played.items() if l == league and n > 0)

    # ---- predicción (no modifica resultados, solo regresa de temporada, idempotente)
    def _touch(self, team, season):
        if self.season_seen.get(team) != season:
            self.elo[team] = 1500 + self.cfg.season_regress * (self.elo[team] - 1500)
            self.season_seen[team] = season

    def predict(self, league: str, season: str, home: str, away: str) -> dict:
        c = self.cfg
        h, a = (league, home), (league, away)
        self._touch(h, season)
        self._touch(a, season)
        eh, ea = self.elo[h], self.elo[a]
        hh, aa, lg = self.hist[h], self.hist[a], self.league_hist[league]
        if len(lg):
            avg_h = float(np.mean([x[0] for x in lg]))
            avg_a = float(np.mean([x[1] for x in lg]))
        else:
            avg_h, avg_a = c.default_home_goals, c.default_away_goals
        avg_h, avg_a = max(avg_h, c.league_avg_floor), max(avg_a, c.league_avg_floor)
        attack_h = _smooth(hh["home_gf"], avg_h, c.prior) / avg_h
        defense_a = _smooth(aa["away_ga"], avg_h, c.prior) / avg_h
        attack_a = _smooth(aa["away_gf"], avg_a, c.prior) / avg_a
        defense_h = _smooth(hh["home_ga"], avg_a, c.prior) / avg_a
        lh = float(np.clip(avg_h * attack_h * defense_a, c.lam_min, c.lam_max))
        la = float(np.clip(avg_a * attack_a * defense_h, c.lam_min, c.lam_max))
        out = {"elo_diff": eh - ea, "elo_h": eh, "elo_a": ea, "lambda_h": lh, "lambda_a": la,
               "n_home": self.n_played.get(h, 0), "n_away": self.n_played.get(a, 0)}
        out.update(derive_markets(score_matrix(lh, la, c, self.goals)))
        return out

    # ---- actualización --------------------------------------------------
    def update(self, league: str, home: str, away: str, hg: float, ag: float, pre: dict):
        c = self.cfg
        h, a = (league, home), (league, away)
        expected = 1 / (1 + 10 ** ((pre["elo_a"] - pre["elo_h"] - c.home_elo) / 400))
        result = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        delta = c.k * (result - expected)
        self.elo[h] += delta
        self.elo[a] -= delta
        self.hist[h]["home_gf"].append(float(hg)); self.hist[h]["home_ga"].append(float(ag))
        self.hist[a]["away_gf"].append(float(ag)); self.hist[a]["away_ga"].append(float(hg))
        self.league_hist[league].append((float(hg), float(ag)))
        self.n_played[h] += 1
        self.n_played[a] += 1

    # ---- procesamiento por lotes ---------------------------------------
    def process(self, games: pd.DataFrame) -> pd.DataFrame:
        """Devuelve una fila por partido con variables PRE-partido. Partidos con marcador
        actualizan el estado después de predecir toda su fecha; sin marcador solo se predicen."""
        d = validate_games(games)
        rows = []
        for date, day in d.groupby("date", sort=True):
            pending = []
            for rec in day.to_dict("records"):
                pre = self.predict(rec["competition"], rec["season"], rec["home"], rec["away"])
                row = {**rec, **pre}
                if pd.notna(rec["hs"]) and pd.notna(rec["as_"]):
                    row["target"] = "H" if rec["hs"] > rec["as_"] else ("D" if rec["hs"] == rec["as_"] else "A")
                    pending.append((rec, pre))
                else:
                    row["target"] = None
                rows.append(row)
            for rec, pre in pending:
                self.update(rec["competition"], rec["home"], rec["away"], rec["hs"], rec["as_"], pre)
            self.last_date = date
        return pd.DataFrame(rows)


def validate_games(df: pd.DataFrame) -> pd.DataFrame:
    need = {"date", "competition", "season", "home", "away", "hs", "as_"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["season"] = d["season"].astype(str)
    d["hs"] = pd.to_numeric(d["hs"], errors="coerce")
    d["as_"] = pd.to_numeric(d["as_"], errors="coerce")
    if d["date"].isna().any():
        raise ValueError("Fechas inválidas")
    if (d["home"] == d["away"]).any():
        raise ValueError("Equipo local = visitante")
    if (d["hs"].isna() != d["as_"].isna()).any():
        raise ValueError("Marcador incompleto")
    if ((d[["hs", "as_"]] < 0).any(axis=1)).any():
        raise ValueError("Marcador negativo")
    if d.duplicated(["date", "competition", "home", "away"]).any():
        raise ValueError("Partidos duplicados")
    long = pd.concat([d[["date", "competition", "home"]].rename(columns={"home": "team"}),
                      d[["date", "competition", "away"]].rename(columns={"away": "team"})])
    if long.duplicated(["date", "competition", "team"]).any():
        raise ValueError("Un equipo juega dos veces el mismo día")
    return d.sort_values(["date", "competition", "home", "away"]).reset_index(drop=True)
