"""Pick FINAL de Mocha Picks (NBA/NFL): combina Kalshi + casas + modelo propio en una sola probabilidad.

p_final = logística( a + b_k·logit(Kalshi) + b_c·logit(casas) + b_m·logit(modelo) ), con pesos aprendidos en partidos
históricos ya jugados (precios de Kalshi y línea de las casas antes del inicio, y predicción fuera de muestra del modelo).
Si falta alguna fuente se usa el submodelo entrenado con las que haya: (k,c,m) → (k,m) → (c,m) → (m).

  python -m kalshi_nba.final_pick fit --league nba --rows reports/kalshi_bench/nba_kalshi_bench_rows.csv
      → valida fuera de muestra (ventana creciente por fecha) y guarda kalshi_nba/final_weights.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

WEIGHTS = Path(__file__).with_name("final_weights.json")
SETS = [("k", "c", "m"), ("k", "m"), ("c", "m"), ("m",)]
LABEL = {("k", "c", "m"): "Kalshi + casas + modelo", ("k", "m"): "Kalshi + modelo", ("c", "m"): "casas + modelo", ("m",): "modelo"}


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def _ok(v):
    return v is not None and np.isfinite(v) and 0 < v < 1


def fit_sets(d: pd.DataFrame) -> dict:
    """d: columnas k, c, m (probabilidad del local) e y. Devuelve {"k,c,m": [a, b...]}."""
    out = {}
    for s in SETS:
        x = d.dropna(subset=list(s))
        if len(x) < 100:
            continue
        m = LogisticRegression(C=1.0, max_iter=1000).fit(np.column_stack([logit(x[c]) for c in s]), x["y"])
        out[",".join(s)] = [float(m.intercept_[0])] + [float(v) for v in m.coef_[0]]
    return out


def combine(k, c, m, w: dict) -> tuple[float | None, str]:
    vals = {"k": k, "c": c, "m": m}
    for s in SETS:
        key = ",".join(s)
        if key in w and all(_ok(vals[x]) for x in s):
            a, *b = w[key]
            z = a + sum(bi * logit(vals[x]) for bi, x in zip(b, s))
            return float(1 / (1 + np.exp(-z))), LABEL[s]
    return (float(m), "modelo") if _ok(m) else (None, "sin dato")


def load_weights(league: str) -> dict:
    if not WEIGHTS.exists():
        return {}
    return json.loads(WEIGHTS.read_text()).get(league, {}).get("pesos", {})


def walk_forward(d: pd.DataFrame, warm=0.4) -> pd.DataFrame:
    """Predicción de cada fecha con pesos entrenados SOLO con fechas anteriores."""
    d = d.sort_values("date").reset_index(drop=True)
    days = np.sort(d["date"].unique())
    out = pd.Series(np.nan, index=d.index)
    for day in days[int(len(days) * warm):]:
        tr, te = d[d["date"] < day], d[d["date"] == day]
        w = fit_sets(tr)
        out[te.index] = [combine(r.k, r.c, r.m, w)[0] for r in te.itertuples()]
    return d.assign(p_final=out)


def main(argv=None):
    from kalshi_nba.backtest import brier, logloss, paired_bootstrap
    from kalshi_lineup.evaluate import metrics
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--league", required=True)
    f.add_argument("--rows", required=True)
    f.add_argument("--model-col", default="p_base")
    a = ap.parse_args(argv)
    r = pd.read_csv(a.rows)
    d = pd.DataFrame({"date": pd.to_datetime(r["date"]), "y": r["y"], "k": r["k60_mid"], "c": r["book_p_home"], "m": r[a.model_col]})
    d.loc[~d["k"].between(0.001, 0.999), "k"] = np.nan
    wf = walk_forward(d).dropna(subset=["p_final"])
    y = wf["y"].to_numpy()
    actual = wf["c"].where(wf["c"].notna(), wf["m"])                      # pick actual: casas si existen; si no, modelo
    res = {"n": int(len(wf)), "desde": str(wf["date"].min())[:10], "hasta": str(wf["date"].max())[:10],
           "pick_actual_casas_o_modelo": metrics(actual, y), "kalshi_solo": metrics(wf["k"].fillna(wf["c"]), y),
           "modelo_solo": metrics(wf["m"], y), "pick_final_combinado": metrics(wf["p_final"], y),
           "mejora_vs_pick_actual": paired_bootstrap(logloss(actual, y), logloss(wf["p_final"], y), wf["date"]),
           "mejora_vs_kalshi": paired_bootstrap(logloss(wf["k"].fillna(wf["c"]), y), logloss(wf["p_final"], y), wf["date"]),
           "mejora_brier_vs_pick_actual": paired_bootstrap(brier(actual, y), brier(wf["p_final"], y), wf["date"])}
    pesos = fit_sets(d)
    data = json.loads(WEIGHTS.read_text()) if WEIGHTS.exists() else {}
    data[a.league] = {"pesos": pesos, "entrenado_con": int(len(d)), "modelo": a.model_col, "validacion": res,
                      "nota": "Pesos sobre logit(prob. local): [intercepto, Kalshi, casas, modelo] según el subconjunto."}
    WEIGHTS.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"pesos": pesos, **res}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
