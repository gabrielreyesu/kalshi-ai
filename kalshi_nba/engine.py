"""Motor NBA sin fugas de datos.

Elo con multiplicador por margen (estilo FiveThirtyEight), regresión a la media entre temporadas, descanso,
back-to-back, forma reciente (margen y anotación). Todo se calcula con partidos de fechas ANTERIORES: los
resultados de un día se incorporan después de generar las variables de todos los partidos de ese día.
La identidad del equipo es el id de ESPN (los nombres cambian, p. ej. LA Clippers).
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NBAConfig:
    k: float = 20.0
    hfa: float = 100.0             # ventaja de local en puntos de Elo
    season_carry: float = 0.75     # 25% de regresión a la media entre temporadas
    elo_mean: float = 1505.0
    form_window: int = 10
    pace_window: int = 15
    league_window: int = 300
    rest_cap: int = 7
    default_rest: int = 5
    default_total: float = 225.0
    mov: str = "nba"               # "nba" (potencia) o "log" (NFL)


CONFIGS = {
    "nba": NBAConfig(),
    "ncaab": NBAConfig(k=20.0, hfa=85.0, season_carry=0.6, league_window=1500, default_total=142.0),
    "nfl": NBAConfig(k=20.0, hfa=48.0, season_carry=0.67, form_window=6, pace_window=8, league_window=200,
                     rest_cap=14, default_rest=7, default_total=45.0, mov="log"),
}


FEATURES = ["elo_diff", "rest_diff", "b2b_h", "b2b_a", "g7_diff", "form_diff", "net15_diff", "playoff"]


def mov_multiplier(margin: float, elo_diff_winner: float, kind: str = "nba") -> float:
    if kind == "log":
        return float(np.log(abs(margin) + 1.0) * 2.2 / (2.2 + 0.001 * elo_diff_winner))
    return ((abs(margin) + 3.0) ** 0.8) / (7.5 + 0.006 * elo_diff_winner)


class NBAEngine:
    def __init__(self, cfg: NBAConfig | None = None):
        self.cfg = cfg or NBAConfig()
        c = self.cfg
        self.elo = defaultdict(lambda: c.elo_mean)
        self.season_seen: dict = {}
        self.games = defaultdict(lambda: deque(maxlen=max(c.pace_window, c.form_window, 12)))  # (date, pf, pa)
        self.league = deque(maxlen=c.league_window)
        self.n_played = defaultdict(int)

    def _touch(self, team, season):
        if self.season_seen.get(team) != season:
            self.elo[team] = self.cfg.season_carry * self.elo[team] + (1 - self.cfg.season_carry) * self.cfg.elo_mean
            self.season_seen[team] = season

    def _team_state(self, team, date):
        c = self.cfg
        g = list(self.games[team])
        rest = min((date - g[-1][0]).days, c.rest_cap) if g else c.default_rest
        last7 = sum(1 for x in g if 0 < (date - x[0]).days <= 7)
        fw, pw = g[-c.form_window:], g[-c.pace_window:]
        form = float(np.mean([x[1] - x[2] for x in fw])) if fw else 0.0
        pf = float(np.mean([x[1] for x in pw])) if pw else np.nan
        pa = float(np.mean([x[2] for x in pw])) if pw else np.nan
        return rest, last7, form, pf, pa

    def predict(self, date, season, home, away, playoff=0, neutral=False) -> dict:
        c = self.cfg
        self._touch(home, season); self._touch(away, season)
        eh, ea = self.elo[home], self.elo[away]
        rh, g7h, fh, pfh, pah = self._team_state(home, date)
        ra, g7a, fa, pfa, paa = self._team_state(away, date)
        lg = float(np.mean([x for x in self.league])) if self.league else c.default_total
        hfa = 0.0 if neutral else c.hfa
        return {"elo_h": eh, "elo_a": ea, "elo_diff": eh - ea + hfa,
                "rest_h": rh, "rest_a": ra, "rest_diff": rh - ra, "b2b_h": int(rh <= 1), "b2b_a": int(ra <= 1),
                "g7_diff": g7h - g7a, "form_diff": fh - fa,
                "pf_h": pfh, "pa_h": pah, "pf_a": pfa, "pa_a": paa,
                "net15_diff": (pfh - pah) - (pfa - paa) if not np.isnan(pfh + pah + pfa + paa) else 0.0,
                "league_total": lg, "playoff": int(playoff),
                "n_home": self.n_played[home], "n_away": self.n_played[away]}

    def update(self, date, home, away, hs, as_, pre, neutral=False):
        c = self.cfg
        margin = hs - as_
        exp_h = 1 / (1 + 10 ** (-(pre["elo_diff"]) / 400))
        won_h = margin > 0
        res = 0.5 if margin == 0 else (1.0 if won_h else 0.0)
        d_w = pre["elo_diff"] if won_h else -pre["elo_diff"]
        delta = c.k * (1.0 if margin == 0 else mov_multiplier(margin, d_w, c.mov)) * (res - exp_h)
        self.elo[home] += delta; self.elo[away] -= delta
        self.games[home].append((date, hs, as_)); self.games[away].append((date, as_, hs))
        self.league.append(hs + as_)
        self.n_played[home] += 1; self.n_played[away] += 1

    def process(self, games: pd.DataFrame) -> pd.DataFrame:
        """games: columnas date, season_year, season_type, home_id, away_id, hs, as_, (neutral). Orden por fecha."""
        d = games.sort_values(["date", "kickoff_utc"] if "kickoff_utc" in games else ["date"]).reset_index(drop=True)
        rows = []
        for date, day in d.groupby("date", sort=True):
            teams = pd.concat([day["home_id"], day["away_id"]])
            if teams.duplicated().any():
                raise ValueError(f"Un equipo juega dos veces el mismo día: {date.date()}")
            pend = []
            for r in day.to_dict("records"):
                pre = self.predict(date, r["season_year"], r["home_id"], r["away_id"],
                                   playoff=int(r["season_type"] == 3), neutral=bool(r.get("neutral", False)))
                row = {**r, **pre}
                if pd.notna(r["hs"]) and pd.notna(r["as_"]) and r.get("status", "STATUS_FINAL") == "STATUS_FINAL":
                    row["margin"], row["total_pts"], row["home_win"] = r["hs"] - r["as_"], r["hs"] + r["as_"], int(r["hs"] > r["as_"])
                    pend.append((r, pre))
                else:
                    row["margin"] = row["total_pts"] = row["home_win"] = np.nan
                rows.append(row)
            for r, pre in pend:
                self.update(date, r["home_id"], r["away_id"], r["hs"], r["as_"], pre, bool(r.get("neutral", False)))
        return pd.DataFrame(rows)
