"""Registro prospectivo append-only: predicciones guardadas ANTES del kickoff + evaluación posterior."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import per_match_logloss, score_1x2, y_index
from .models import CLASSES

REGISTRY = "predictions/registry.csv"


def register(preds: pd.DataFrame, path: str = REGISTRY, now: pd.Timestamp | None = None) -> dict:
    """Agrega solo partidos nuevos y futuros. Nunca reescribe filas existentes."""
    now = now or pd.Timestamp.now(tz="UTC")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    preds = preds.copy()
    preds["kickoff_utc"] = pd.to_datetime(preds["kickoff_utc"], utc=True)
    existing = pd.read_csv(p, dtype={"match_id": str}) if p.exists() else pd.DataFrame()
    seen = set(existing["match_id"]) if len(existing) else set()
    new = preds[~preds["match_id"].astype(str).isin(seen)]
    started = new["kickoff_utc"] <= now
    new = new[~started]
    if len(new):
        new = new.assign(created_utc=now.isoformat())
        if len(existing):  # mismo orden de columnas que el archivo
            new = new.reindex(columns=list(existing.columns))
        new.to_csv(p, mode="a", header=not p.exists(), index=False)
    return {"nuevos": len(new), "ya_registrados": int(preds["match_id"].astype(str).isin(seen).sum()),
            "ya_iniciados_omitidos": int(started.sum())}


def evaluate(history: pd.DataFrame, path: str = REGISTRY, out_dir: str = "predictions") -> pd.DataFrame:
    """Une el registro con resultados reales (histórico) por liga/equipos y fecha ±1 día."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    reg = pd.read_csv(p, dtype={"match_id": str})
    reg["kick_date"] = pd.to_datetime(reg["kickoff_utc"], utc=True).dt.tz_localize(None).dt.normalize()
    h = history[["date", "competition", "home", "away", "hs", "as_"]].copy()
    h["date"] = pd.to_datetime(h["date"]).dt.normalize()
    m = reg.merge(h, on=["competition", "home", "away"], how="left", suffixes=("", "_res"))
    m = m[(m["date"] - m["kick_date"]).abs() <= pd.Timedelta(days=1)]
    m = m.drop_duplicates("match_id")
    if m.empty:
        print("Aún no hay partidos registrados con resultado.")
        return m
    m["target"] = np.where(m["hs"] > m["as_"], "H", np.where(m["hs"] == m["as_"], "D", "A"))
    yi = y_index(m["target"])
    for tag, pref in (("modelo", "p"), ("dc", "dc_p")):
        P = m[[f"{pref}_{c}" for c in CLASSES]].to_numpy(float)
        m[f"logloss_{tag}"] = per_match_logloss(P, yi)
    m["total_goles"] = m["hs"] + m["as_"]
    for line in ("1_5", "2_5", "3_5"):
        m[f"over_{line}_real"] = (m["total_goles"] > float(line.replace("_", "."))).astype(int)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    m.to_csv(Path(out_dir) / "evaluation.csv", index=False)
    s = score_1x2(m[[f"p_{c}" for c in CLASSES]].to_numpy(float), yi)
    print(f"Evaluados {s['n']} partidos | accuracy {s['accuracy_%']:.1f}% | logloss {s['logloss']:.4f} | brier {s['brier']:.4f}")
    return m
