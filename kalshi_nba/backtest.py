"""Walk-forward por temporada de NBA y comparación contra las líneas de mercado (ESPN)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .engine import FEATURES
from .models import NBAModels, total_matrix

EPS = 1e-12


def logloss(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier(p, y):
    return (np.asarray(p, float) - np.asarray(y, float)) ** 2


def market_win_prob(df: pd.DataFrame, sigma_spread: float = 12.0) -> pd.DataFrame:
    """Prob. local implícita sin margen: mediana de moneylines entre casas; si falta, del spread (normal)."""
    p_ml = pd.to_numeric(df["line_p_home"], errors="coerce")
    sp = pd.to_numeric(df["line_spread"], errors="coerce")
    p_sp = pd.Series(norm.cdf(-sp / sigma_spread), index=df.index)     # spread local negativo = favorito
    src = np.where(p_ml.notna(), "moneyline", np.where(p_sp.notna(), "spread", None))
    return pd.DataFrame({"p_mkt": p_ml.fillna(p_sp), "mkt_src": src}, index=df.index)


def paired_bootstrap(a, b, dates, B=2000, seed=42):
    """Mejora media de b sobre a (positivo = b mejor), IC95% remuestreando por fecha."""
    d = pd.DataFrame({"x": np.asarray(a, float) - np.asarray(b, float), "date": np.asarray(dates)})
    g = d.groupby("date")["x"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(g), size=(B, len(g)))
    m = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return {"mejora": round(float(s.sum() / c.sum()), 5), "ic95_inf": round(float(lo), 5), "ic95_sup": round(float(hi), 5),
            "excluye_0": bool(lo > 0 or hi < 0)}


def walk_forward(feats: pd.DataFrame, first_test: int = 2019, min_train: int = 2000, features=None) -> pd.DataFrame:
    done = feats[feats["margin"].notna()].copy()
    out = []
    for s in [x for x in sorted(done["season_year"].unique()) if x >= first_test]:
        tr, te = done[done["season_year"] < s], done[done["season_year"] == s]
        if len(tr) < min_train or te.empty:
            continue
        m = NBAModels(features=features).fit(tr)
        pr = m.predict(te)
        rec = pd.concat([te, pr], axis=1)
        rec["sig_m"], rec["sig_t"], rec["base_win"] = m.sig_m, m.sig_t, m.base_win
        # mercado derivado: P(over línea) y P(local cubre) del modelo
        ls, lt = pd.to_numeric(te["line_spread"], errors="coerce"), pd.to_numeric(te["line_total"], errors="coerce")
        rec["p_over_model"] = m.p_total_over(pr["mu_total"], lt)
        rec["p_cover_model"] = m.p_home_wins_by_more_than(pr["mu_margin"], -ls)       # margen > -spread
        out.append(rec)
    if not out:
        raise RuntimeError("No hubo temporadas de prueba con suficiente entrenamiento")
    r = pd.concat(out, ignore_index=True)
    r = pd.concat([r, market_win_prob(r)], axis=1)
    return r


def blend_walk_forward(feats: pd.DataFrame, first_test: int = 2019, min_train: int = 1500) -> pd.DataFrame:
    """¿Aporta el modelo SOBRE la línea? Solo partidos con spread+total."""
    d = feats[feats["margin"].notna() & pd.to_numeric(feats["line_spread"], errors="coerce").notna()
              & pd.to_numeric(feats["line_total"], errors="coerce").notna()].copy()
    d = pd.concat([d, market_win_prob(d)], axis=1)
    d["line_spread"], d["line_total"] = pd.to_numeric(d["line_spread"]), pd.to_numeric(d["line_total"])
    out = []
    for s in [x for x in sorted(d["season_year"].unique()) if x >= first_test]:
        tr, te = d[d["season_year"] < s], d[d["season_year"] == s]
        if len(tr) < min_train or te.empty:
            continue
        rec = te[["date", "season_year", "home_id", "away_id", "margin", "total_pts", "home_win", "line_spread", "line_total", "p_mkt"]].copy()
        lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
        Xr = lambda x: np.column_stack([-x["line_spread"].to_numpy(float)])
        Xb = lambda x: np.column_stack([-x["line_spread"].to_numpy(float), x[FEATURES].to_numpy(float)])
        for tag, f in (("recal", Xr), ("blend", Xb)):
            mm = make_pipeline(StandardScaler(), Ridge(alpha=30)).fit(f(tr), tr["margin"])
            rec[f"mu_margin_{tag}"] = mm.predict(f(te))
        Tr = lambda x: np.column_stack([x["line_total"].to_numpy(float)])
        Tb = lambda x: np.column_stack([x["line_total"].to_numpy(float), total_matrix(x)])
        for tag, f in (("recal", Tr), ("blend", Tb)):
            mt = make_pipeline(StandardScaler(), Ridge(alpha=30)).fit(f(tr), tr["total_pts"])
            rec[f"mu_total_{tag}"] = mt.predict(f(te))
        Wr = lambda x: lg(x["p_mkt"].to_numpy(float)).reshape(-1, 1)
        Wb = lambda x: np.column_stack([lg(x["p_mkt"].to_numpy(float)), x[FEATURES].to_numpy(float)])
        for tag, f in (("recal", Wr), ("blend", Wb)):
            mw = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(f(tr), tr["home_win"].astype(int))
            rec[f"p_{tag}"] = mw.predict_proba(f(te))[:, 1]
        out.append(rec)
    return pd.concat(out, ignore_index=True)


def report(r: pd.DataFrame) -> str:
    L = [f"PARTIDOS EVALUADOS (fuera de muestra): {len(r)} | temporadas (año de cierre) {sorted(r.season_year.unique())}\n"]
    y = r["home_win"].to_numpy()
    tab = {"modelo logístico": r["p_home"], "modelo normal(margen)": r["p_home_normal"],
           "frecuencia base (local)": r["base_win"]}
    L.append("=== GANADOR: todos los partidos ===")
    L.append(pd.DataFrame({k: {"n": len(y), "accuracy_%": round(100 * float(((np.asarray(v) >= .5) == (y == 1)).mean()), 2),
                               "logloss": round(float(logloss(v, y).mean()), 4), "brier": round(float(brier(v, y).mean()), 4)}
                           for k, v in tab.items()}).T.to_string())
    m = r["p_mkt"].notna().to_numpy()
    if m.any():
        sub = r[m]; ys = sub["home_win"].to_numpy()
        t2 = {"modelo logístico": sub["p_home"], "modelo normal(margen)": sub["p_home_normal"],
              "mercado (moneyline o spread)": sub["p_mkt"], "frecuencia base": sub["base_win"]}
        L.append(f"\n=== GANADOR: mismo subconjunto con línea de mercado ({m.sum()} partidos) ===")
        L.append(pd.DataFrame({k: {"n": len(ys), "accuracy_%": round(100 * float(((np.asarray(v) >= .5) == (ys == 1)).mean()), 2),
                                   "logloss": round(float(logloss(v, ys).mean()), 4), "brier": round(float(brier(v, ys).mean()), 4)}
                               for k, v in t2.items()}).T.to_string())
        L.append("Fuente del mercado: " + str(sub["mkt_src"].value_counts().to_dict()))
        L.append("Bootstrap por fecha, mejora del MERCADO sobre el modelo (logloss): "
                 + str(paired_bootstrap(logloss(sub["p_home"], ys), logloss(sub["p_mkt"], ys), sub["date"])))
    L.append("\n=== MARGEN: error del margen esperado (RMSE, puntos; menor = mejor) ===")
    ok = pd.to_numeric(r["line_spread"], errors="coerce").notna().to_numpy()
    rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
    mk = r[ok]
    L.append(pd.Series({"n": int(ok.sum()), "modelo": round(rm(mk["margin"] - mk["mu_margin"]), 3),
                        "línea de mercado (-spread)": round(rm(mk["margin"] + pd.to_numeric(mk["line_spread"])), 3)}).to_string())
    L.append("\n=== TOTAL: error del total esperado (MAE, puntos; menor = mejor) ===")
    okt = pd.to_numeric(r["line_total"], errors="coerce").notna().to_numpy()
    mt = r[okt]
    L.append(pd.Series({"n": int(okt.sum()), "modelo": round(float(np.mean(np.abs(mt["total_pts"] - mt["mu_total"]))), 3),
                        "línea de mercado": round(float(np.mean(np.abs(mt["total_pts"] - pd.to_numeric(mt["line_total"])))), 3)}).to_string())
    L.append("\n=== OVER/UNDER y COBERTURA contra la línea (sin empates con la línea) ===")
    lt = pd.to_numeric(mt["line_total"]); np_ = mt["total_pts"] != lt
    yo = (mt["total_pts"] > lt)[np_].astype(int).to_numpy(); po = mt["p_over_model"][np_].to_numpy()
    ls = pd.to_numeric(mk["line_spread"]); npc = (mk["margin"] + ls) != 0
    yc = ((mk["margin"] + ls) > 0)[npc].astype(int).to_numpy(); pc = mk["p_cover_model"][npc].to_numpy()
    rows = []
    for name, yy, pp in (("over total", yo, po), ("local cubre spread", yc, pc)):
        rows.append({"mercado": name, "n": len(yy), "tasa_real": round(float(yy.mean()), 4),
                     "logloss_modelo": round(float(logloss(pp, yy).mean()), 4), "logloss_50%": round(float(np.log(2)), 4),
                     "accuracy_modelo_%": round(100 * float(((pp >= .5) == (yy == 1)).mean()), 2)})
    L.append(pd.DataFrame(rows).to_string(index=False))
    L.append("Lectura: las líneas buscan que over/cubre ocurra ~50%. Si 'logloss_modelo' no baja de 0.6931 con claridad, no hay ventaja.")
    L.append("\n=== POR TEMPORADA (ganador, logloss modelo / mercado) ===")
    rows = []
    for s, g in r.groupby("season_year"):
        yy = g["home_win"].to_numpy(); row = {"temporada": int(s), "n": len(g), "modelo": round(float(logloss(g["p_home"], yy).mean()), 4)}
        gm = g[g["p_mkt"].notna()]
        if len(gm):
            row["mercado"] = round(float(logloss(gm["p_mkt"], gm["home_win"]).mean()), 4)
            row["modelo_mismo_subconj"] = round(float(logloss(gm["p_home"], gm["home_win"]).mean()), 4)
        rows.append(row)
    L.append(pd.DataFrame(rows).to_string(index=False))
    L.append("\n=== CALIBRACIÓN GANADOR (modelo logístico) ===")
    cut = pd.cut(r["p_home"], bins=[0, .2, .3, .4, .5, .6, .7, .8, .9, 1.0], include_lowest=True)
    t = pd.DataFrame({"p": r["p_home"], "y": y, "b": cut}).groupby("b", observed=True).agg(n=("y", "size"), prob_media=("p", "mean"), real=("y", "mean"))
    t["dif_pp"] = 100 * (t["real"] - t["prob_media"])
    L.append(t.round(4).to_string())
    return "\n".join(L)


def blend_report(b: pd.DataFrame) -> str:
    L = [f"PARTIDOS (fuera de muestra, con spread+total): {len(b)} | temporadas {sorted(b.season_year.unique())}\n"]
    rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
    mae = lambda e: float(np.mean(np.abs(e)))
    d = b["date"]
    mm = lambda c: b["margin"] - b[c]
    L.append("=== MARGEN (RMSE) ===")
    L.append(pd.Series({"línea": round(rm(b["margin"] + b["line_spread"]), 4), "línea recalibrada": round(rm(mm("mu_margin_recal")), 4),
                        "línea+modelo": round(rm(mm("mu_margin_blend")), 4)}).to_string())
    L.append("  mejora línea+modelo vs recalibrada (error cuadrático): "
             + str(paired_bootstrap(np.square(mm("mu_margin_recal")), np.square(mm("mu_margin_blend")), d)))
    L.append("\n=== TOTAL (MAE) ===")
    tt = lambda c: b["total_pts"] - b[c]
    L.append(pd.Series({"línea": round(mae(b["total_pts"] - b["line_total"]), 4), "línea recalibrada": round(mae(tt("mu_total_recal")), 4),
                        "línea+modelo": round(mae(tt("mu_total_blend")), 4)}).to_string())
    L.append("  mejora línea+modelo vs recalibrada (error absoluto): "
             + str(paired_bootstrap(np.abs(tt("mu_total_recal")), np.abs(tt("mu_total_blend")), d)))
    L.append("\n=== GANADOR (logloss) ===")
    y = b["home_win"].to_numpy()
    ll = {k: logloss(b[c], y) for k, c in (("mercado", "p_mkt"), ("recalibrado", "p_recal"), ("mercado+modelo", "p_blend"))}
    L.append(pd.Series({k: round(float(v.mean()), 4) for k, v in ll.items()}).to_string())
    L.append("  mejora mercado+modelo vs recalibrado: " + str(paired_bootstrap(ll["recalibrado"], ll["mercado+modelo"], d)))
    L.append("  mejora mercado+modelo vs mercado: " + str(paired_bootstrap(ll["mercado"], ll["mercado+modelo"], d)))
    return "\n".join(L)
