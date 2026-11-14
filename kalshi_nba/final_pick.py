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


# ---- Fútbol (local / empate / visitante) ----------------------------------------------------------------------
# Agregación log-lineal: p_c ∝ exp(b_c + Σ_s w_s · log p_{s,c}), con b_D = 0. Pesos por fuente aprendidos minimizando
# la log loss en partidos históricos con precio de Kalshi antes del inicio, cuotas de las casas y modelo fuera de muestra.

def _logp3(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1)
    return np.log(p / p.sum(axis=-1, keepdims=True))


def _pool3(params, L: list) -> np.ndarray:
    b = np.array([params[0], 0.0, params[1]])
    z = b + sum(w * l for w, l in zip(params[2:], L))
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fit_sets3(d: pd.DataFrame, l2=1e-3) -> dict:
    """d: columnas k_H.., c_H.., m_H.. (pueden faltar con NaN) e y en {0,1,2}."""
    from scipy.optimize import minimize
    out = {}
    for s in SETS:
        cols = [[f"{x}_{c}" for c in "HDA"] for x in s]
        x = d.dropna(subset=[c for cs in cols for c in cs])
        if len(x) < 150:
            continue
        L = [_logp3(x[cs].to_numpy()) for cs in cols]
        y = x["y"].to_numpy(int)
        f = lambda p: -np.log(np.clip(_pool3(p, L)[np.arange(len(y)), y], 1e-12, 1)).mean() + l2 * np.sum(np.square(p[2:]))
        r = minimize(f, np.r_[0.0, 0.0, np.full(len(s), 1.0 / len(s))], method="L-BFGS-B")
        out[",".join(s)] = [float(v) for v in r.x]
    return out


def combine3(k, c, m, w: dict) -> tuple[list | None, str]:
    vals = {"k": k, "c": c, "m": m}
    ok = lambda v: v is not None and len(v) == 3 and all(x is not None and np.isfinite(x) and x >= 0 for x in v) and sum(v) > 0
    for s in SETS:
        key = ",".join(s)
        if key in w and all(ok(vals[x]) for x in s):
            p = _pool3(np.array(w[key]), [_logp3(np.array(vals[x], float)) for x in s])
            return [float(v) for v in p], LABEL[s]
    if ok(m):
        q = np.array(m, float); return [float(v) for v in q / q.sum()], "modelo"
    return None, "sin dato"


def walk_forward3(d: pd.DataFrame, warm=0.4) -> pd.DataFrame:
    d = d.sort_values("date").reset_index(drop=True)
    days = np.sort(d["date"].unique())
    P = np.full((len(d), 3), np.nan)
    for day in days[int(len(days) * warm):]:
        tr, te = d[d["date"] < day], d[d["date"] == day]
        w = fit_sets3(tr)
        for i, r in te.iterrows():
            g = lambda x: [r.get(f"{x}_{c}") for c in "HDA"]
            p, _ = combine3(g("k"), g("c"), g("m"), w)
            if p:
                P[i] = p
    return d.assign(f_H=P[:, 0], f_D=P[:, 1], f_A=P[:, 2])
