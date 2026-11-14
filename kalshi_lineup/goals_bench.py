"""Goles (más/menos de 2,5): ¿combinar la línea de las casas con el modelo mejora a cada uno por separado?

Protocolo pre-registrado (FACTORES.md / KALSHI_BENCHMARK.md, sección goles): partidos de copas, 5 grandes ligas y
selecciones con línea de 2,5 de las casas (ESPN, sin margen) y probabilidad del modelo fuera de muestra
(kalshi_intl walk-forward). Combinación logística logit(casas) + logit(modelo), ventana creciente por fecha (40 %
de calentamiento). Éxito para producción: no ser peor que las casas solas (IC95% de la diferencia de log loss ≥ 0
en el límite superior) y mejor que el modelo solo.
  python -m kalshi_lineup.goals_bench
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from kalshi_nba.backtest import logloss, paired_bootstrap
from kalshi_intl.odds import attach
from kalshi_lineup.evaluate import metrics

from .soccer_kalshi_bench import BIG5
from kalshi_intl.data import load_group
from kalshi_intl.model import CONFIGS, GROUP_FEATURES, IntlEngine, walk_forward


def lg(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def main():
    parts = []
    for grp in ("clubes", "selecciones"):
        g = load_group(grp, verbose=False)
        f = IntlEngine(CONFIGS[grp]).process(g)
        f["eval"] = f["target"] | f["code"].isin(BIG5) if grp == "clubes" else f["target"]
        r = walk_forward(f, first_year=2020, features=GROUP_FEATURES[grp], eval_col="eval")
        parts.append(r[["event_id", "date", "code", "hs", "as_", "p_over_2_5"]].assign(grupo=grp))
    d = attach(pd.concat(parts, ignore_index=True).assign(event_id=lambda x: x["event_id"].astype(str)))
    d = d[d["mkt_over"].notna() & d["p_over_2_5"].notna()].copy()
    d["y"] = ((d["hs"] + d["as_"]) > 2.5).astype(int)
    d["date"] = pd.to_datetime(d["date"]); d = d.sort_values("date").reset_index(drop=True)
    days = np.sort(d["date"].unique())
    pb, pc = pd.Series(np.nan, index=d.index), pd.Series(np.nan, index=d.index)
    X1 = lambda x: lg(x["mkt_over"]).reshape(-1, 1)
    X2 = lambda x: np.column_stack([lg(x["mkt_over"]), lg(x["p_over_2_5"])])
    for i, day in enumerate(days[int(len(days) * .4):]):
        tr, te = d[d["date"] < day], d[d["date"] == day]
        if i % 7 == 0 or "m1" not in dir():                           # reentrenar semanalmente (rápido y sin mirar el futuro)
            m1 = LogisticRegression().fit(X1(tr), tr["y"]); m2 = LogisticRegression().fit(X2(tr), tr["y"])
        pb[te.index] = m1.predict_proba(X1(te))[:, 1]; pc[te.index] = m2.predict_proba(X2(te))[:, 1]
    t = d[pc.notna()].assign(pb=pb[pc.notna()], pc=pc[pc.notna()])
    y = t["y"].to_numpy()
    res = {"n": int(len(t)), "desde": str(t["date"].min())[:10], "hasta": str(t["date"].max())[:10],
           "casas_crudo": metrics(t["mkt_over"], y), "casas_recalibrado": metrics(t["pb"], y), "modelo": metrics(t["p_over_2_5"], y),
           "combinado": metrics(t["pc"], y),
           "mejora_vs_casas_recal": paired_bootstrap(logloss(t["pb"], y), logloss(t["pc"], y), t["date"]),
           "mejora_vs_casas_crudo": paired_bootstrap(logloss(t["mkt_over"], y), logloss(t["pc"], y), t["date"]),
           "mejora_vs_modelo": paired_bootstrap(logloss(t["p_over_2_5"], y), logloss(t["pc"], y), t["date"])}
    for name, mask in (("selecciones", t["grupo"] == "selecciones"), ("clubes", t["grupo"] == "clubes")):
        s = t[mask]
        if len(s) > 100:
            res[f"solo_{name}"] = {"n": int(len(s)), "mejora_vs_casas_recal": paired_bootstrap(logloss(s["pb"], s["y"]), logloss(s["pc"], s["y"]), s["date"]),
                                   "mejora_vs_modelo": paired_bootstrap(logloss(s["p_over_2_5"], s["y"]), logloss(s["pc"], s["y"]), s["date"])}
    m = LogisticRegression().fit(X2(d), d["y"])
    res["pesos"] = {"intercepto": float(m.intercept_[0]), "casas": float(m.coef_[0][0]), "modelo": float(m.coef_[0][1])}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return res


if __name__ == "__main__":
    main()
