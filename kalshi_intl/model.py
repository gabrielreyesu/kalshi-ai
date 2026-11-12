"""Motor para copas de clubes y selecciones: Elo (con ajuste por fuerza de liga y sede neutral) + Poisson.

Sin fugas: las variables de un partido usan solo partidos de DÍAS anteriores; los resultados de un día se
incorporan después de generar las variables de todos los partidos de ese día.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kalshi_soccer.engine import EngineConfig, derive_markets, score_matrix

FEATS = ["elo_diff", "neutral", "gf_h", "ga_h", "gf_a", "ga_a", "friendly"]
# Variables adicionales, probadas fuera de muestra (reports/variables_extra_futbol.txt)
EXTRA = ["rest_diff", "g21_diff", "h2h_gd", "split_gd", "pts5_diff"]
# Resultado: en selecciones ayudan enfrentamientos directos y rendimiento casa/fuera; en clubes ninguna aporta.
GROUP_FEATURES = {"clubes": FEATS, "selecciones": FEATS + ["h2h_gd", "split_gd"]}
_DC = EngineConfig()
_GOALS = np.arange(_DC.max_goals + 1)


@dataclass(frozen=True)
class IntlConfig:
    k: float = 20.0
    k_friendly: float = 10.0
    hfa: float = 65.0
    carry: float = 1.0            # regresión a la media cada 1 de julio (1.0 = ninguna)
    k_league: float = 0.0         # ajuste de fuerza de liga, aprendido solo en partidos entre ligas distintas
    window: int = 12
    prior: float = 6.0
    avg_goals: float = 1.35


CONFIGS = {"clubes": IntlConfig(k=20, hfa=65, carry=0.85, k_league=4.0),
           "selecciones": IntlConfig(k=30, k_friendly=15, hfa=80, carry=1.0, k_league=0.0)}


def _gmult(diff: float) -> float:
    d = abs(diff)
    return 1.0 if d <= 1 else (1.5 if d == 2 else (11 + d) / 8)


class IntlEngine:
    def __init__(self, cfg: IntlConfig):
        self.cfg = cfg
        self.elo = defaultdict(lambda: 1500.0)
        self.league_adj = defaultdict(float)
        self.home_league: dict = {}
        self.season: dict = {}
        self.form = defaultdict(lambda: deque(maxlen=cfg.window))      # (gf, ga)
        self.n = defaultdict(int)
        self.dates = defaultdict(lambda: deque(maxlen=12))             # fechas de los últimos partidos
        self.h2h = defaultdict(lambda: deque(maxlen=6))                # diferencia de goles vista desde el equipo "menor"
        self.at_home = defaultdict(lambda: deque(maxlen=8)); self.on_road = defaultdict(lambda: deque(maxlen=8))

    def _extra(self, date, home, away):
        mean = lambda d: float(np.mean(d)) if len(d) else 0.0
        rest = lambda t: min((date - self.dates[t][-1]).days, 14) if self.dates[t] else 7
        g21 = lambda t: sum(1 for x in self.dates[t] if 0 < (date - x).days <= 21)
        pts = lambda t: float(np.mean([3 if a > b else (1 if a == b else 0) for a, b in list(self.form[t])[-5:]])) if self.form[t] else 1.3
        key, sign = ((home, away), 1.0) if home < away else ((away, home), -1.0)
        return {"rest_diff": rest(home) - rest(away), "g21_diff": g21(home) - g21(away), "h2h_gd": sign * mean(self.h2h[key]),
                "split_gd": mean(self.at_home[home]) - mean(self.on_road[away]), "pts5_diff": pts(home) - pts(away)}

    def _touch(self, t, season):
        if self.season.get(t) != season:
            if t in self.season:
                self.elo[t] = 1500 + self.cfg.carry * (self.elo[t] - 1500)
            self.season[t] = season

    def _rating(self, t):
        return self.elo[t] + self.league_adj[self.home_league.get(t)]

    def _form(self, t):
        f, c = self.form[t], self.cfg
        gf = (sum(x[0] for x in f) + c.prior * c.avg_goals) / (len(f) + c.prior)
        ga = (sum(x[1] for x in f) + c.prior * c.avg_goals) / (len(f) + c.prior)
        return gf, ga

    def predict(self, date, home, away, neutral=False, friendly=False) -> dict:
        season = date.year if date.month >= 7 else date.year - 1
        self._touch(home, season); self._touch(away, season)
        gfh, gah = self._form(home); gfa, gaa = self._form(away)
        hfa = 0.0 if neutral else self.cfg.hfa
        return {**self._extra(date, home, away),
                "elo_diff": self._rating(home) - self._rating(away) + hfa, "neutral": int(neutral), "friendly": int(friendly),
                "gf_h": gfh, "ga_h": gah, "gf_a": gfa, "ga_a": gaa, "n_home": self.n[home], "n_away": self.n[away]}

    def update(self, home, away, hs, as_, pre, code, is_feed, friendly=False, date=None):
        c = self.cfg
        if date is not None:
            self.dates[home].append(date); self.dates[away].append(date)
        self.h2h[(home, away) if home < away else (away, home)].append((hs - as_) if home < away else (as_ - hs))
        self.at_home[home].append(hs - as_); self.on_road[away].append(as_ - hs)
        exp = 1 / (1 + 10 ** (-pre["elo_diff"] / 400))
        res = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        g = _gmult(hs - as_)
        d = (c.k_friendly if friendly else c.k) * g * (res - exp)
        self.elo[home] += d; self.elo[away] -= d
        lh, la = self.home_league.get(home), self.home_league.get(away)
        if c.k_league and lh and la and lh != la:
            dl = c.k_league * g * (res - exp)
            self.league_adj[lh] += dl; self.league_adj[la] -= dl
        if is_feed and c.k_league:
            self.home_league[home] = code; self.home_league[away] = code
        self.form[home].append((hs, as_)); self.form[away].append((as_, hs))
        self.n[home] += 1; self.n[away] += 1

    def process(self, g: pd.DataFrame) -> pd.DataFrame:
        rows = []
        g = g.sort_values(["date", "kick", "event_id"])
        for date, day in g.groupby("date", sort=True):
            pend = []
            for r in day.to_dict("records"):
                fr = r["code"] == "fifa.friendly"
                pre = self.predict(date, r["home_id"], r["away_id"], r["neutral"], fr)
                rows.append({**r, **pre})
                if r["final"]:
                    pend.append((r, pre, fr))
            for r, pre, fr in pend:
                self.update(r["home_id"], r["away_id"], r["hs"], r["as_"], pre, r["code"], not r["target"], fr, date)
        return pd.DataFrame(rows)


class IntlModel:
    """Goles esperados de local y visitante (Poisson) -> matriz de marcadores -> 1X2, totales, ambos anotan."""

    def __init__(self, features=None):
        self.features = list(features or FEATS)

    def fit(self, f: pd.DataFrame):
        d = f[f["final"]]
        X = d[self.features].to_numpy(float)
        mk = lambda: make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=500))
        self.mh, self.ma = mk().fit(X, d["hs"]), mk().fit(X, d["as_"])
        return self

    def predict(self, f: pd.DataFrame) -> pd.DataFrame:
        X = f[self.features].to_numpy(float)
        lh = np.clip(self.mh.predict(X), 0.15, 5.0); la = np.clip(self.ma.predict(X), 0.15, 5.0)
        rows = [derive_markets(score_matrix(a, b, _DC, _GOALS)) for a, b in zip(lh, la)]
        out = pd.DataFrame(rows, index=f.index)
        out["lambda_h"], out["lambda_a"] = lh, la
        return out


def walk_forward(f: pd.DataFrame, first_year: int = 2020, min_train: int = 1500, features=None, eval_col: str = "target") -> pd.DataFrame:
    """Entrena con todo lo anterior al 1 de julio del año y evalúa la temporada siguiente, solo en competiciones objetivo."""
    out = []
    regular = f["status"].isin({"STATUS_FULL_TIME", "STATUS_FINAL"})      # sin prórroga: el mercado 1X2 es a 90 min
    for y in range(first_year, int(f["date"].dt.year.max()) + 1):
        a, b = pd.Timestamp(f"{y}-07-01"), pd.Timestamp(f"{y + 1}-07-01")
        tr = f[f["final"] & (f["date"] < a)]
        te = f[f["final"] & regular & f[eval_col] & (f["date"] >= a) & (f["date"] < b)]
        if len(tr) < min_train or te.empty:
            continue
        pr = IntlModel(features).fit(tr).predict(te)
        rec = pd.concat([te, pr], axis=1)
        y3 = np.where(tr["hs"] > tr["as_"], "H", np.where(tr["hs"] == tr["as_"], "D", "A"))
        trt = tr[tr["target"]]
        yb = np.where(trt["hs"] > trt["as_"], "H", np.where(trt["hs"] == trt["as_"], "D", "A")) if len(trt) > 200 else y3
        for c in "HDA":
            rec[f"base_{c}"] = float((yb == c).mean())
        rec["base_over"] = float(((trt["hs"] + trt["as_"]) > 2.5).mean()) if len(trt) > 200 else float(((tr["hs"] + tr["as_"]) > 2.5).mean())
        rec["temporada"] = f"{y}/{(y + 1) % 100:02d}"
        out.append(rec)
    r = pd.concat(out, ignore_index=True)
    r["res"] = np.where(r["hs"] > r["as_"], "H", np.where(r["hs"] == r["as_"], "D", "A"))
    return r
