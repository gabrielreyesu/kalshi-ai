"""Comparación pre-registrada: nba_v1 vs nba_v1 + variables de alineación, en los MISMOS partidos fuera de muestra.

Protocolo (fijado ANTES de ver resultados; ver LINEUP_RESEARCH.md):
  * Walk-forward por temporada: cada temporada se predice con un modelo entrenado solo con temporadas anteriores.
  * Desarrollo: temporadas (año de cierre) 2019-2024. Ahí se elige UNA variante (mayor mejora media de logloss).
  * Confirmación: temporadas >= 2025, evaluadas una sola vez con la variante elegida en desarrollo.
  * Éxito: mejora de logloss con IC95% (bootstrap por fecha) > 0 en confirmación, Brier no peor y ECE no peor
    en más de 0.005. Si no se cumple, la variable se descarta. Nada de esto cambia producción.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from kalshi_nba.backtest import brier, logloss, market_win_prob, paired_bootstrap, walk_forward

VARIANTS = {
    "A_disponibles_gs": ["lineup_gs"],
    "B_titulares_gs": ["start_gs"],
    "C_disponibles_gs_pm": ["lineup_gs", "lineup_pm"],
    "D_disponibles_gs_abs": ["lineup_gs", "abs_gs"],
}
DEV_LAST = 2024


def ece(p, y, bins=10) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).sum() for b in range(bins) if (idx == b).any()) / len(p))


def metrics(p, y) -> dict:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return {"n": int(len(y)), "accuracy": round(float(((p >= .5) == (y == 1)).mean()), 4),
            "logloss": round(float(logloss(p, y).mean()), 5), "brier": round(float(brier(p, y).mean()), 5),
            "ece": round(ece(p, y), 4)}


def run_variants(feats: pd.DataFrame, base_features: list, first_test=2019, min_train=2000) -> pd.DataFrame:
    """Una fila por partido OOS con p_base y p_<variante>."""
    base = walk_forward(feats, first_test=first_test, min_train=min_train, features=base_features)
    out = base[["event_id", "date", "season_year", "home_win", "p_home", "line_p_home", "line_spread"]].rename(columns={"p_home": "p_base"})
    out = pd.concat([out, market_win_prob(out)[["p_mkt"]]], axis=1)
    for name, extra in VARIANTS.items():
        r = walk_forward(feats, first_test=first_test, min_train=min_train, features=base_features + extra)
        out = out.merge(r[["event_id", "p_home"]].rename(columns={"p_home": f"p_{name}"}), on="event_id", how="inner")
    return out


def market_plus_lineup(oos: pd.DataFrame, feats: pd.DataFrame, cols: list, min_train=1500) -> pd.DataFrame:
    """¿Añade la alineación algo SOBRE el mercado de cierre? logit(p_mkt) [+ alineación], walk-forward."""
    from sklearn.linear_model import LogisticRegression
    d = oos[oos["p_mkt"].notna()].merge(feats[["event_id"] + cols], on="event_id")
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    res = []
    for s in sorted(d["season_year"].unique()):
        tr, te = d[d["season_year"] < s], d[d["season_year"] == s]
        if len(tr) < min_train:
            continue
        X = lambda x, c: np.column_stack([lg(x["p_mkt"].to_numpy(float))] + [x[k].to_numpy(float) for k in c])
        r = te[["event_id", "date", "season_year", "home_win", "p_mkt"]].copy()
        for tag, c in (("recal", []), ("lineup", cols)):
            m = LogisticRegression(C=1.0, max_iter=2000).fit(X(tr, c), tr["home_win"].astype(int))
            r[f"p_mkt_{tag}"] = m.predict_proba(X(te, c))[:, 1]
        res.append(r)
    return pd.concat(res, ignore_index=True) if res else pd.DataFrame()


def compare(oos: pd.DataFrame, col: str) -> dict:
    y = oos["home_win"].to_numpy()
    return {"base": metrics(oos["p_base"], y), "variante": metrics(oos[col], y),
            "mejora_logloss": paired_bootstrap(logloss(oos["p_base"], y), logloss(oos[col], y), oos["date"]),
            "mejora_brier": paired_bootstrap(brier(oos["p_base"], y), brier(oos[col], y), oos["date"])}


def decide(oos: pd.DataFrame, feats: pd.DataFrame | None = None) -> dict:
    dev, hold = oos[oos["season_year"] <= DEV_LAST], oos[oos["season_year"] > DEV_LAST]
    dev_res = {n: compare(dev, f"p_{n}") for n in VARIANTS} if len(dev) else {}
    chosen = max(dev_res, key=lambda n: dev_res[n]["mejora_logloss"]["mejora"]) if dev_res else "A_disponibles_gs"
    out = {"protocolo": "dev<=2024 elige variante; confirmación >=2025 una sola vez", "desarrollo": dev_res,
           "variante_elegida": chosen}
    if len(hold):
        h = compare(hold, f"p_{chosen}")
        ok = (h["mejora_logloss"]["ic95_inf"] > 0 and h["mejora_brier"]["mejora"] >= 0
              and h["variante"]["ece"] <= h["base"]["ece"] + 0.005)
        out["confirmacion"] = h
        out["veredicto"] = "SUPERA_CRITERIO_SOLO_SOMBRA" if ok else "NO_SUPERA_CRITERIO"
    else:
        out["veredicto"] = "SIN_TEMPORADAS_DE_CONFIRMACION"
    per = []
    for s, g in oos.groupby("season_year"):
        y = g["home_win"].to_numpy()
        row = {"temporada": int(s), "n": len(g), "logloss_base": round(float(logloss(g["p_base"], y).mean()), 5)}
        row.update({f"ll_{n}": round(float(logloss(g[f"p_{n}"], y).mean()), 5) for n in VARIANTS})
        gm = g[g["p_mkt"].notna()]
        if len(gm):
            row["ll_mercado_subconj"] = round(float(logloss(gm["p_mkt"], gm["home_win"]).mean()), 5)
        per.append(row)
    out["por_temporada"] = per
    m = oos[oos["p_mkt"].notna()]
    if len(m):
        y = m["home_win"].to_numpy()
        out["vs_mercado_mismo_subconjunto"] = {"base": metrics(m["p_base"], y), "variante": metrics(m[f"p_{chosen}"], y),
                                               "mercado": metrics(m["p_mkt"], y)}
    if feats is not None and len(m):
        mp = market_plus_lineup(oos, feats, VARIANTS[chosen])
        if len(mp):
            y = mp["home_win"].to_numpy()
            out["mercado_mas_alineacion"] = {
                "nota": "Mercado de cierre ya incorpora alineaciones; se espera poca o ninguna mejora.",
                "mercado_recalibrado": metrics(mp["p_mkt_recal"], y), "mercado_mas_alineacion": metrics(mp["p_mkt_lineup"], y),
                "mejora_logloss": paired_bootstrap(logloss(mp["p_mkt_recal"], y), logloss(mp["p_mkt_lineup"], y), mp["date"])}
    return out


def render(res: dict, coverage: dict) -> str:
    L = ["NBA — RATINGS AJUSTADOS POR ALINEACIÓN (investigación, producción sin cambios)", "=" * 72,
         "Cobertura de boxscores: " + json.dumps(coverage, ensure_ascii=False), "",
         "Protocolo: " + res["protocolo"], "", "--- DESARROLLO (temporadas <= 2024) ---"]
    for n, r in res["desarrollo"].items():
        L.append(f"{n:24s} base ll={r['base']['logloss']:.5f} acc={r['base']['accuracy']:.4f} | variante ll={r['variante']['logloss']:.5f} "
                 f"acc={r['variante']['accuracy']:.4f} ece={r['variante']['ece']:.4f} | mejora ll {r['mejora_logloss']}")
    L.append(f"\nVariante elegida en desarrollo: {res['variante_elegida']}")
    if "confirmacion" in res:
        c = res["confirmacion"]
        L += ["\n--- CONFIRMACIÓN (temporadas >= 2025, una sola vez) ---",
              "base:     " + json.dumps(c["base"]), "variante: " + json.dumps(c["variante"]),
              "mejora logloss: " + json.dumps(c["mejora_logloss"]), "mejora brier:   " + json.dumps(c["mejora_brier"])]
    L.append(f"\nVEREDICTO: {res['veredicto']}")
    L += ["\n--- POR TEMPORADA (logloss) ---", pd.DataFrame(res["por_temporada"]).to_string(index=False)]
    if "vs_mercado_mismo_subconjunto" in res:
        L += ["\n--- MISMO SUBCONJUNTO CON MERCADO DE CIERRE ---"]
        L += [f"{k:10s} {json.dumps(v)}" for k, v in res["vs_mercado_mismo_subconjunto"].items()]
    if "mercado_mas_alineacion" in res:
        L += ["\n--- ¿LA ALINEACIÓN AÑADE ALGO SOBRE EL MERCADO? ---", json.dumps(res["mercado_mas_alineacion"], ensure_ascii=False, indent=1)]
    return "\n".join(L)
