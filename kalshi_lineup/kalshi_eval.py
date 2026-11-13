"""¿Acertamos mejor que el precio de Kalshi antes del partido? Protocolo fijado antes de ver resultados (KALSHI_BENCHMARK.md).

Mismos partidos (mercado de ganador de Kalshi liquidado + predicción fuera de muestra del modelo):
  * Kalshi a T-60 min, T-6 h y T-24 h (mid de bid/ask; último precio si el spread > 10¢).
  * Casas: línea de CIERRE de ESPN (posterior a T-60: referencia, no comparación justa en el tiempo).
  * Modelo actual (walk-forward por temporada) y modelo + variable de alineación (variante elegida en su backtest).
H1: modelo+alineación vs Kalshi T-60 (logloss pareado, IC95% por fecha). Se espera que Kalshi gane.
H2: mezcla logit(Kalshi T-60) + logit(modelo+alineación), entrenada SOLO con fechas anteriores (ventana creciente,
    primer 40% de fechas como calentamiento) vs Kalshi T-60 recalibrado igual. Éxito: IC95% de mejora > 0.
Descriptivo: cuando modelo y Kalshi discrepan ≥ 8 puntos, ¿quién acierta más?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from kalshi_nba.backtest import brier, logloss, paired_bootstrap

from .evaluate import ece, metrics


def _lg(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def expanding_blend(d: pd.DataFrame, cols: list, warm=0.4) -> pd.Series:
    """Logística sobre logits de `cols`, reentrenada por fecha con datos ESTRICTAMENTE anteriores."""
    d = d.sort_values("date")
    days = np.sort(d["date"].unique())
    start = days[int(len(days) * warm)]
    out = pd.Series(np.nan, index=d.index)
    X = np.column_stack([_lg(d[c]) for c in cols]); y = d["y"].to_numpy()
    for day in days[days >= start]:
        tr, te = (d["date"] < day).to_numpy(), (d["date"] == day).to_numpy()
        if tr.sum() < 150 or len(np.unique(y[tr])) < 2:
            continue
        m = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], y[tr])
        out[d.index[te]] = m.predict_proba(X[te])[:, 1]
    return out


def evaluate(d: pd.DataFrame, model_col="p_var", base_col="p_base") -> dict:
    """d: event_id, date, y (1 = gana local), k60_mid, k360_mid, k1440_mid, book_p_home, p_base, p_var."""
    d = d.dropna(subset=["k60_mid", model_col, base_col, "y"]).copy()
    d = d[(d["k60_mid"] > 0) & (d["k60_mid"] < 1)]
    y = d["y"].to_numpy()
    res = {"n": int(len(d)), "fechas": int(d["date"].nunique()), "desde": str(d["date"].min())[:10], "hasta": str(d["date"].max())[:10]}
    cand = {"Kalshi T-60 min": "k60_mid", "Kalshi T-6 h": "k360_mid", "Kalshi T-24 h": "k1440_mid",
            "Casas (cierre ESPN)": "book_p_home", "Modelo actual": base_col, "Modelo + alineación": model_col}
    tab = {}
    for name, c in cand.items():
        s = d[d[c].notna() & (d[c] > 0) & (d[c] < 1)]
        tab[name] = {**metrics(s[c], s["y"]), "n": int(len(s))}
    res["tabla"] = tab
    res["H1_modelo_vs_kalshi"] = paired_bootstrap(logloss(d["k60_mid"], y), logloss(d[model_col], y), d["date"])
    d["kal_recal"] = expanding_blend(d, ["k60_mid"])
    d["blend"] = expanding_blend(d, ["k60_mid", model_col])
    t = d.dropna(subset=["kal_recal", "blend"])
    yt = t["y"].to_numpy()
    res["H2_mezcla"] = {"n": int(len(t)), "kalshi_recalibrado": metrics(t["kal_recal"], yt), "kalshi_mas_modelo": metrics(t["blend"], yt),
                        "kalshi_crudo_mismo_tramo": metrics(t["k60_mid"], yt),
                        "mejora_logloss": paired_bootstrap(logloss(t["kal_recal"], yt), logloss(t["blend"], yt), t["date"]),
                        "mejora_vs_kalshi_crudo": paired_bootstrap(logloss(t["k60_mid"], yt), logloss(t["blend"], yt), t["date"])}
    gap = (d[model_col] - d["k60_mid"]).abs() >= 0.08
    g = d[gap]
    if len(g):
        mk = ((g["k60_mid"] >= .5) == (g["y"] == 1)).mean(); mm = ((g[model_col] >= .5) == (g["y"] == 1)).mean()
        res["discrepancias_8pts"] = {"n": int(len(g)), "acierto_kalshi": round(float(mk), 4), "acierto_modelo": round(float(mm), 4),
                                     "logloss_kalshi": round(float(logloss(g["k60_mid"], g["y"]).mean()), 4),
                                     "logloss_modelo": round(float(logloss(g[model_col], g["y"]).mean()), 4)}
    return res


def render(title: str, r: dict) -> str:
    L = [title, "=" * 72, f"Partidos: {r['n']} ({r['fechas']} fechas, {r['desde']} a {r['hasta']})", "",
         f"{'Fuente':24s} {'n':>5s} {'acierto':>8s} {'logloss':>8s} {'brier':>7s} {'ECE':>6s}"]
    for k, m in r["tabla"].items():
        L.append(f"{k:24s} {m['n']:5d} {100 * m['accuracy']:7.1f}% {m['logloss']:8.4f} {m['brier']:7.4f} {m['ece']:6.3f}")
    L += ["", "H1 modelo+alineación vs Kalshi T-60 (positivo = modelo mejor): " + str(r["H1_modelo_vs_kalshi"]),
          "", f"H2 mezcla (n={r['H2_mezcla']['n']}, entrenada solo con fechas anteriores):",
          "  Kalshi recalibrado: " + str(r["H2_mezcla"]["kalshi_recalibrado"]),
          "  Kalshi + modelo:    " + str(r["H2_mezcla"]["kalshi_mas_modelo"]),
          "  mejora vs Kalshi recalibrado: " + str(r["H2_mezcla"]["mejora_logloss"]),
          "  mejora vs Kalshi crudo:       " + str(r["H2_mezcla"]["mejora_vs_kalshi_crudo"])]
    if "discrepancias_8pts" in r:
        L += ["", "Discrepancias >= 8 pts (descriptivo): " + str(r["discrepancias_8pts"])]
    return "\n".join(L)
