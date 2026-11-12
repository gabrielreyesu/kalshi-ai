"""Auditoría y calibración 1X2 temporal sobre predicciones realmente fuera de muestra.

No modifica el motor de producción. El calibrador se ajusta solo con temporadas
anteriores a la temporada evaluada; la primera temporada queda sin calibrar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

CLASSES = ("H", "D", "A")
PROB_COLS = ("p_H", "p_D", "p_A")


def probabilities(df, cols=PROB_COLS):
    p = df[list(cols)].to_numpy(dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or (p < 0).any() or (p.sum(axis=1) <= 0).any():
        raise ValueError("Probabilidades 1X2 inválidas")
    return np.clip(p / p.sum(axis=1, keepdims=True), 1e-9, 1.0)


def scores(p, outcome):
    y = pd.Series(outcome).map({"H": 0, "D": 1, "A": 2}).to_numpy()
    if pd.isna(y).any():
        raise ValueError("Resultado 1X2 inválido")
    y = y.astype(int)
    target = np.eye(3)[y]
    return {"n": len(y), "accuracy": float((p.argmax(axis=1) == y).mean()),
            "logloss": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
            "brier": float(((p - target) ** 2).sum(axis=1).mean())}


def temporal_calibrate(df, min_train=300, strength=1.0):
    """Calibración multinomial por logits; SOLO predicciones OOS de temporadas previas.

    Cada fila se evalúa con un calibrador entrenado antes del inicio de su temporada.
    No se usa el resultado de una fila para calibrar su propia temporada.
    """
    required = set(PROB_COLS) | {"res", "temporada", "date"}
    if not required.issubset(df):
        raise ValueError(f"Columnas faltantes: {sorted(required - set(df))}")
    if df.empty:
        return df.copy()
    d = df.copy()
    p = probabilities(d)
    out = p.copy()
    d["_season_start"] = d["temporada"].astype(str).str.extract(r"^(\d{4})")[0].astype(int)
    dates = pd.to_datetime(d["date"])
    if dates.isna().any():
        raise ValueError("Fechas inválidas")
    fitted = np.zeros(len(d), dtype=bool)
    for season in sorted(d["_season_start"].unique()):
        tr = (d["_season_start"].to_numpy() < season) & (dates.to_numpy() < np.datetime64(f"{season}-07-01"))
        te = d["_season_start"].to_numpy() == season
        labels = d.loc[tr, "res"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        if tr.sum() < min_train or len(np.unique(labels)) != 3:
            continue
        xtr = np.log(p[tr, :][:, [0, 2]] / p[tr, 1, None])
        xte = np.log(p[te, :][:, [0, 2]] / p[te, 1, None])
        model = LogisticRegression(C=strength, max_iter=1000)
        model.fit(xtr, labels)
        q = model.predict_proba(xte)
        out[np.where(te)[0][:, None], model.classes_[None, :]] = q
        fitted[te] = True
    for i, c in enumerate(CLASSES):
        d[f"cal_{c}"] = out[:, i]
    d["calibrated"] = fitted
    return d.drop(columns=["_season_start"])


def reliability(df, cols=PROB_COLS, bins=10):
    p = probabilities(df, cols)
    y = df["res"].to_numpy()
    records = []
    for j, cls in enumerate(CLASSES):
        for b in range(bins):
            lo, hi = b / bins, (b + 1) / bins
            sel = (p[:, j] >= lo) & ((p[:, j] < hi) if b < bins - 1 else (p[:, j] <= hi))
            if sel.sum():
                records.append({"outcome": cls, "bin": f"{lo:.1f}-{hi:.1f}", "n": int(sel.sum()),
                                "predicted": float(p[sel, j].mean()), "observed": float((y[sel] == cls).mean())})
    return pd.DataFrame(records)


def report(df, min_train=300):
    d = temporal_calibrate(df, min_train=min_train)
    result = []
    for season, part in d.groupby("temporada", sort=True):
        for variant, cols in (("original", PROB_COLS), ("temporal_calibrated", ("cal_H", "cal_D", "cal_A"))):
            result.append({"temporada": season, "version": variant, "trained_calibrator": bool(part.calibrated.all()),
                           **scores(probabilities(part, cols), part.res)})
    return d, pd.DataFrame(result)
