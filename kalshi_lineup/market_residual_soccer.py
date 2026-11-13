"""Fútbol: ¿aporta el bloque de variables del modelo algo a la línea 1X2 de las casas? (protocolo en FACTORES.md)

Regresión logística multinomial sobre log(casas H/D/A) + bloque (Elo, goles a favor/en contra, descanso, carga 21 días,
H2H, casa/fuera, puntos últimos 5). Walk-forward por temporada (prueba ≥ 2021/22); confirmación ≥ 2024/25. Éxito IC95% > 0.
  python -m kalshi_lineup.market_residual_soccer
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kalshi_nba.backtest import paired_bootstrap
from kalshi_intl.data import load_group
from kalshi_intl.model import CONFIGS, EXTRA, FEATS, IntlEngine
from kalshi_intl.odds import attach

CODES = ["eng.1", "esp.1", "ita.1", "ger.1", "fra.1", "uefa.champions", "uefa.europa", "uefa.europa.conf"]


def main():
    g = load_group("clubes", verbose=False)
    f = IntlEngine(CONFIGS["clubes"]).process(g)
    f = f[f["final"] & f["code"].isin(CODES) & (f["date"] >= pd.Timestamp("2019-07-01")) & f["status"].isin(["STATUS_FULL_TIME", "STATUS_FINAL"])]
    f = attach(f.assign(event_id=f["event_id"].astype(str)).reset_index(drop=True))
    d = f[f["mkt_H"].notna()].copy()
    d["y"] = np.where(d["hs"] > d["as_"], 0, np.where(d["hs"] == d["as_"], 1, 2))
    d["season"] = np.where(d["date"].dt.month >= 7, d["date"].dt.year, d["date"].dt.year - 1)
    mk = [f"mkt_{c}" for c in "HDA"]
    block = [c for c in FEATS + EXTRA if c not in ("neutral", "friendly")]
    def run(cols):
        out = []
        for s in sorted(d["season"].unique()):
            tr, te = d[d["season"] < s], d[d["season"] == s]
            if s < 2021 or len(tr) < 2000:
                continue
            X = lambda x: np.column_stack([np.log(np.clip(x[mk].to_numpy(float), 1e-3, 1))] + [x[c].to_numpy(float) for c in cols])
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000)).fit(X(tr), tr["y"])
            P = m.predict_proba(X(te))
            out.append(te.assign(ll=-np.log(np.clip(P[np.arange(len(te)), te["y"].to_numpy()], 1e-12, 1))))
        return pd.concat(out)
    a, b = run([]), run(block)
    h = a["season"] >= 2024
    res = {"partidos_confirmacion": int(h.sum()), "casas_recalibrado": round(float(a[h]["ll"].mean()), 5),
           "casas_mas_bloque": round(float(b[h]["ll"].mean()), 5),
           "desarrollo": paired_bootstrap(a[~h]["ll"], b[~h]["ll"], a[~h]["date"]),
           "confirmacion": paired_bootstrap(a[h]["ll"], b[h]["ll"], a[h]["date"])}
    res["veredicto"] = "SUPERA_CRITERIO" if res["confirmacion"]["ic95_inf"] > 0 else "NO_SUPERA_CRITERIO"
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
