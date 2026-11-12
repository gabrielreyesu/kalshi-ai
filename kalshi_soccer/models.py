"""Capa de modelos: 1X2 híbrido (V4 del notebook) y mercados derivados (totales, BTTS, spreads)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = ["A", "D", "H"]
NUM_1X2 = ["elo_diff", "p_H", "p_D", "p_A", "lambda_h", "lambda_a"]

# nombre -> (columna con la prob. cruda de Dixon-Coles, función que define el resultado real)
MARKETS = {
    "over_1_5": ("p_over_1_5", lambda d: (d["hs"] + d["as_"]) > 1.5),
    "over_2_5": ("p_over_2_5", lambda d: (d["hs"] + d["as_"]) > 2.5),
    "over_3_5": ("p_over_3_5", lambda d: (d["hs"] + d["as_"]) > 3.5),
    "btts": ("p_btts", lambda d: (d["hs"] > 0) & (d["as_"] > 0)),
    "home_m1_5": ("p_home_m1_5", lambda d: (d["hs"] - d["as_"]) >= 2),
    "away_m1_5": ("p_away_m1_5", lambda d: (d["as_"] - d["hs"]) >= 2),
}


def _dummies(df: pd.DataFrame, leagues: list) -> np.ndarray:
    if not leagues:
        return np.empty((len(df), 0))
    return np.column_stack([(df["competition"] == l).to_numpy(float) for l in leagues])


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _pipe(C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


class Soccer1X2:
    """Regresión logística sobre Elo + Poisson/DC + liga. Equivale al V4 del notebook."""

    def __init__(self, num_cols=None, C: float = 0.3):
        self.num_cols = list(num_cols or NUM_1X2)
        self.C = C

    def _X(self, df):
        return np.hstack([df[self.num_cols].to_numpy(float), _dummies(df, self.leagues)])

    def fit(self, df: pd.DataFrame):
        self.leagues = sorted(df["competition"].unique())
        self.pipe = _pipe(self.C).fit(self._X(df), df["target"].to_numpy())
        self.classes = list(self.pipe.classes_)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Matriz (n, 3) en el orden A, D, H."""
        P = self.pipe.predict_proba(self._X(df))
        return P[:, [self.classes.index(c) for c in CLASSES]]


class DerivedMarket:
    """Recalibra la probabilidad cruda de un mercado derivado usando logit(p_DC), lambdas y liga."""

    def __init__(self, name: str, C: float = 0.3):
        self.name, self.C = name, C
        self.pcol, self.target_fn = MARKETS[name]

    def _X(self, df):
        num = np.column_stack([_logit(df[self.pcol]), df["lambda_h"], df["lambda_a"]])
        return np.hstack([num, _dummies(df, self.leagues)])

    def fit(self, df: pd.DataFrame):
        self.leagues = sorted(df["competition"].unique())
        y = self.target_fn(df).astype(int).to_numpy()
        self.pipe = _pipe(self.C).fit(self._X(df), y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict_proba(self._X(df))[:, 1]
