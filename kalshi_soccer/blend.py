"""¿Aporta algo el modelo SOBRE las cuotas de cierre? Walk-forward por temporada, sin fugas.

Se comparan, en los mismos partidos con cuotas:
  mercado         : cuotas de cierre sin margen (la vara)
  mercado_recal   : regresión logística solo sobre logit(mercado)  (corrige sesgos como favorito/empate)
  mercado+modelo  : lo anterior + Elo + Poisson/DC + lambdas (+ liga)
Si 'mercado+modelo' no mejora a 'mercado_recal' con IC que excluya 0, el modelo no añade información.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .backtest import bin_logloss_each, paired_bootstrap, per_match_logloss, score_1x2, y_index
from .models import CLASSES, _dummies, _logit


def _mk(df, cols):
    return np.column_stack([_logit(df[c]) for c in cols])


def blend_walk_forward(feats: pd.DataFrame, first_test_season="2223", C=0.3, min_train=1000) -> pd.DataFrame:
    d = feats[feats["target"].notna() & feats["mkt_H"].notna()].copy()
    mk_cols = [f"mkt_{c}" for c in CLASSES]
    out = []
    for s in [x for x in sorted(d["season"].unique()) if x >= first_test_season]:
        tr, te = d[d["season"] < s], d[d["season"] == s]
        if len(tr) < min_train or te.empty:
            continue
        leagues = sorted(tr["competition"].unique())
        extra = lambda x: np.hstack([x[["elo_diff", "p_H", "p_D", "p_A", "lambda_h", "lambda_a"]].to_numpy(float),
                                     _dummies(x, leagues)])
        fits = {
            "recal": (lambda x: _mk(x, mk_cols)),
            "blend": (lambda x: np.hstack([_mk(x, mk_cols), extra(x)])),
        }
        rec = te[["date", "competition", "season", "home", "away", "target"]].copy()
        for c in CLASSES:
            rec[f"mkt_{c}"] = te[f"mkt_{c}"].to_numpy()
        for name, f in fits.items():
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(f(tr), tr["target"])
            P = m.predict_proba(f(te))
            cl = list(m.classes_)
            for j, c in enumerate(CLASSES):
                rec[f"{name}_{c}"] = P[:, cl.index(c)]
        # Over 2.5: mercado vs recal vs mercado+modelo
        if te["mkt_over_2_5"].notna().all() and tr["mkt_over_2_5"].notna().all():
            y_tr = ((tr["hs"] + tr["as_"]) > 2.5).astype(int).to_numpy()
            rec["y_over"] = ((te["hs"] + te["as_"]) > 2.5).astype(int).to_numpy()
            rec["mkt_over"] = te["mkt_over_2_5"].to_numpy()
            f_r = lambda x: _logit(x["mkt_over_2_5"]).reshape(-1, 1)
            f_b = lambda x: np.hstack([_logit(x["mkt_over_2_5"]).reshape(-1, 1),
                                       _logit(x["p_over_2_5"]).reshape(-1, 1),
                                       x[["lambda_h", "lambda_a"]].to_numpy(float), _dummies(x, leagues)])
            for name, f in (("recal", f_r), ("blend", f_b)):
                m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(f(tr), y_tr)
                rec[f"over_{name}"] = m.predict_proba(f(te))[:, 1]
        out.append(rec)
    return pd.concat(out, ignore_index=True)


def blend_report(p: pd.DataFrame) -> str:
    L = [f"PARTIDOS (fuera de muestra, con cuotas): {len(p)} | temporadas {sorted(p.season.unique())}\n",
         "=== 1X2 ==="]
    yi = y_index(p["target"])
    P = lambda pre: p[[f"{pre}_{c}" for c in CLASSES]].to_numpy(float)
    tab = {"mercado": score_1x2(P("mkt"), yi), "mercado_recal": score_1x2(P("recal"), yi),
           "mercado+modelo": score_1x2(P("blend"), yi)}
    L.append(pd.DataFrame(tab).T.round(4).to_string())
    for a, b, label in (("mkt", "recal", "recal vs mercado"), ("recal", "blend", "mercado+modelo vs recal"),
                        ("mkt", "blend", "mercado+modelo vs mercado")):
        r = paired_bootstrap(per_match_logloss(P(a), yi), per_match_logloss(P(b), yi), p["date"])
        L.append(f"  mejora de {label}: " + str({k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()}))
    L.append("\nPor temporada (logloss): mercado / recal / mercado+modelo")
    rows = []
    for s, g in p.groupby("season"):
        gi = y_index(g["target"])
        Q = lambda pre: g[[f"{pre}_{c}" for c in CLASSES]].to_numpy(float)
        rows.append({"temporada": s, "n": len(g), **{k: round(score_1x2(Q(pre), gi)["logloss"], 4)
                     for k, pre in (("mercado", "mkt"), ("recal", "recal"), ("blend", "blend"))}})
    L.append(pd.DataFrame(rows).to_string(index=False))
    if "over_blend" in p:
        q = p[p["over_blend"].notna()]
        y = q["y_over"].to_numpy(float)
        L.append(f"\n=== OVER 2.5 ({len(q)} partidos) ===")
        ll = {k: bin_logloss_each(q[c].to_numpy(), y) for k, c in
              (("mercado", "mkt_over"), ("mercado_recal", "over_recal"), ("mercado+modelo", "over_blend"))}
        L.append(pd.Series({k: round(v.mean(), 4) for k, v in ll.items()}).to_string())
        for a, b, label in (("mercado", "mercado_recal", "recal vs mercado"),
                            ("mercado_recal", "mercado+modelo", "mercado+modelo vs recal"),
                            ("mercado", "mercado+modelo", "mercado+modelo vs mercado")):
            r = paired_bootstrap(ll[a], ll[b], q["date"])
            L.append(f"  mejora de {label}: " + str({k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()}))
    return "\n".join(L)
