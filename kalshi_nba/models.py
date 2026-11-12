"""Modelos NBA: ganador (logística), margen esperado y total esperado (ridge) con dispersión normal.

Mercados derivados: P(local gana por más de x), P(total > x) = normal alrededor del valor esperado, con sigma
estimada de los residuos de ENTRENAMIENTO (nunca de la prueba).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .engine import FEATURES

TOTAL_FEATURES = ["pf_h", "pa_h", "pf_a", "pa_a", "league_total", "rest_h", "rest_a", "b2b_h", "b2b_a", "playoff"]


def total_matrix(df: pd.DataFrame) -> np.ndarray:
    half = (df["league_total"] / 2).to_numpy(float)
    cols = []
    for c in TOTAL_FEATURES:
        v = df[c].to_numpy(float)
        if c in ("pf_h", "pa_h", "pf_a", "pa_a"):
            v = np.where(np.isnan(v), half, v)
        cols.append(v)
    return np.column_stack(cols)


class NBAModels:
    def __init__(self, C: float = 1.0, alpha_m: float = 30.0, alpha_t: float = 30.0, features=None):
        self.C, self.alpha_m, self.alpha_t = C, alpha_m, alpha_t
        self.features = list(features or FEATURES)

    def fit(self, df: pd.DataFrame):
        d = df[df["margin"].notna()]
        X, Xt = d[self.features].to_numpy(float), total_matrix(d)
        self.win = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=2000)).fit(X, d["home_win"].astype(int))
        self.mar = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha_m)).fit(X, d["margin"])
        self.tot = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha_t)).fit(Xt, d["total_pts"])
        self.sig_m = float(np.std(d["margin"] - self.mar.predict(X), ddof=1))
        self.sig_t = float(np.std(d["total_pts"] - self.tot.predict(Xt), ddof=1))
        self.base_win = float(d["home_win"].mean())
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X, Xt = df[self.features].to_numpy(float), total_matrix(df)
        out = pd.DataFrame(index=df.index)
        out["p_home"] = self.win.predict_proba(X)[:, 1]
        out["mu_margin"] = self.mar.predict(X)
        out["mu_total"] = self.tot.predict(Xt)
        out["p_home_normal"] = norm.cdf(out["mu_margin"] / self.sig_m)
        return out

    # ---- mercados derivados -------------------------------------------
    def p_home_wins_by_more_than(self, mu_margin, x):
        """P(margen_local > x). Para 'Utah gana por más de 7.5' (visitante local→ usar -x con signo)."""
        return 1 - norm.cdf((np.asarray(x, float) - np.asarray(mu_margin, float)) / self.sig_m)

    def p_total_over(self, mu_total, line):
        return 1 - norm.cdf((np.asarray(line, float) - np.asarray(mu_total, float)) / self.sig_t)
