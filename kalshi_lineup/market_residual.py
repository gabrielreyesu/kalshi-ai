"""¿Hay información en las variables del modelo que el mercado de cierre NO incorpora? (NBA/NFL)

Protocolo pre-registrado (FACTORES.md): logística logit(prob. de las casas) + bloque completo de variables del modelo
(walk-forward por temporada, prueba ≥ 2019; confirmación NBA ≥ 2025, NFL ≥ 2024). Éxito: IC95% de mejora > 0 en
confirmación. Las variables por separado se informan solo como descripción (comparaciones múltiples).
  python -m kalshi_lineup.market_residual --league nba|nfl
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kalshi_nba.backtest import logloss, paired_bootstrap
from kalshi_nba.data import load_league
from kalshi_nba.engine import CONFIGS, EXTRA, FEATURES, NBAEngine

HOLD = {"nba": 2025, "nfl": 2024}


def lg(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def run(d, cols):
    out = []
    for s in sorted(d["season_year"].unique()):
        tr, te = d[d["season_year"] < s], d[d["season_year"] == s]
        if s < 2019 or len(tr) < 800:
            continue
        X = lambda x: np.column_stack([lg(x["line_p_home"])] + [x[c].to_numpy(float) for c in cols])
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(X(tr), tr["y"])
        out.append(te.assign(p=m.predict_proba(X(te))[:, 1]))
    return pd.concat(out)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--league", choices=["nba", "nfl"], required=True)
    a = ap.parse_args(argv)
    g = load_league(a.league, first_season=2016 if a.league == "nba" else 2015)
    f = NBAEngine(CONFIGS[a.league]).process(g)
    d = f[f["margin"].notna() & pd.to_numeric(f["line_p_home"], errors="coerce").notna() & (f["margin"] != 0)].copy()
    d["line_p_home"] = pd.to_numeric(d["line_p_home"]); d["y"] = (d["margin"] > 0).astype(int)
    block = FEATURES + EXTRA
    base, full = run(d, []), run(d, block)
    h = HOLD[a.league]
    b, fl = base[base["season_year"] >= h], full[full["season_year"] >= h]
    res = {"partidos_confirmacion": int(len(b)),
           "mercado_recalibrado": round(float(logloss(b["p"], b["y"]).mean()), 5),
           "mercado_mas_bloque": round(float(logloss(fl["p"], fl["y"]).mean()), 5),
           "mejora_bloque": paired_bootstrap(logloss(b["p"], b["y"]), logloss(fl["p"], fl["y"]), b["date"])}
    res["veredicto"] = "SUPERA_CRITERIO" if res["mejora_bloque"]["ic95_inf"] > 0 else "NO_SUPERA_CRITERIO"
    desc = {}
    for c in block:
        one = run(d, [c]); o = one[one["season_year"] >= h]
        desc[c] = paired_bootstrap(logloss(b["p"], b["y"]), logloss(o["p"], o["y"]), b["date"])["mejora"]
    res["descriptivo_por_variable"] = desc
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
