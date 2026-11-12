"""Walk-forward por temporada + métricas + comparación contra el mercado."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import CLASSES, MARKETS, DerivedMarket, Soccer1X2

EPS = 1e-15


# ---------------------------------------------------------------- métricas
def y_index(target) -> np.ndarray:
    idx = pd.Categorical(target, categories=CLASSES).codes
    if (idx < 0).any():
        raise ValueError("Resultado fuera de A/D/H")
    return idx


def per_match_logloss(P: np.ndarray, yi: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(P[np.arange(len(yi)), yi], EPS, 1))


def score_1x2(P: np.ndarray, yi: np.ndarray) -> dict:
    onehot = np.eye(3)[yi]
    return {"n": len(yi), "accuracy_%": 100 * float((P.argmax(1) == yi).mean()),
            "logloss": float(per_match_logloss(P, yi).mean()),
            "brier": float(((P - onehot) ** 2).sum(1).mean())}


def bin_logloss_each(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_bootstrap(loss_a, loss_b, dates, B: int = 2000, seed: int = 42) -> dict:
    """Mejora media de B sobre A (positivo = B mejor) con IC95% remuestreando por FECHA."""
    d = pd.DataFrame({"x": np.asarray(loss_a) - np.asarray(loss_b), "date": np.asarray(dates)})
    g = d.groupby("date")["x"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(g), size=(B, len(g)))
    means = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mejora": float(s.sum() / c.sum()), "ic95_inf": float(lo), "ic95_sup": float(hi),
            "excluye_0": bool(lo > 0 or hi < 0)}


def calibration_table(p: np.ndarray, y: np.ndarray, bins=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0)) -> pd.DataFrame:
    cut = pd.cut(p, bins=list(bins), include_lowest=True)
    t = pd.DataFrame({"p": p, "y": y, "bin": cut}).groupby("bin", observed=True).agg(
        n=("y", "size"), prob_media=("p", "mean"), frecuencia_real=("y", "mean"))
    t["diferencia_pp"] = 100 * (t["frecuencia_real"] - t["prob_media"])
    return t.round(4)


# ---------------------------------------------------------------- walk-forward
def walk_forward(feats: pd.DataFrame, first_test_season: str = "2223", min_train: int = 1000) -> pd.DataFrame:
    """Entrena con temporadas < s y evalúa en s. Las variables de cada fila ya son pre-partido."""
    scored = feats[feats["target"].notna()].copy()
    out = []
    for s in [x for x in sorted(scored["season"].unique()) if x >= first_test_season]:
        tr, te = scored[scored["season"] < s], scored[scored["season"] == s]
        if len(tr) < min_train or te.empty:
            continue
        full, elo = Soccer1X2().fit(tr), Soccer1X2(num_cols=["elo_diff"]).fit(tr)
        P, Pe = full.predict_proba(te), elo.predict_proba(te)
        base = tr["target"].value_counts(normalize=True).reindex(CLASSES, fill_value=0).to_numpy()
        rec = te[["date", "competition", "season", "home", "away", "hs", "as_", "target"]].copy()
        for j, c in enumerate(CLASSES):
            rec[f"m_{c}"], rec[f"elo_{c}"] = P[:, j], Pe[:, j]
            rec[f"dc_{c}"] = te[f"p_{c}"].to_numpy()
            rec[f"base_{c}"] = base[j]
            if f"mkt_{c}" in te:
                rec[f"mkt_{c}"] = te[f"mkt_{c}"].to_numpy()
        for name, (pcol, fn) in MARKETS.items():
            rec[f"y_{name}"] = fn(te).astype(int).to_numpy()
            rec[f"raw_{name}"] = te[pcol].to_numpy()
            rec[f"cal_{name}"] = DerivedMarket(name).fit(tr).predict_proba(te)
            rec[f"base_{name}"] = float(fn(tr).mean())
        for extra in ("mkt_over_2_5", "mkt_source", "mkt_ou_source"):
            if extra in te:
                rec[extra] = te[extra].to_numpy()
        out.append(rec)
    if not out:
        raise RuntimeError("No hubo temporadas de prueba con suficiente entrenamiento")
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- reporte
def _P(df, prefix):
    return df[[f"{prefix}_{c}" for c in CLASSES]].to_numpy(float)


def report(pred: pd.DataFrame) -> str:
    L = []
    yi = y_index(pred["target"])
    models = {"modelo (V4 híbrido)": "m", "solo Elo": "elo", "Poisson/DC crudo": "dc", "frecuencias base": "base"}
    has_mkt = "mkt_H" in pred and pred["mkt_H"].notna().any()
    L.append(f"PARTIDOS EVALUADOS (fuera de muestra): {len(pred)} | temporadas {sorted(pred.season.unique())}\n")

    L.append("=== 1X2: TODOS LOS PARTIDOS ===")
    L.append(pd.DataFrame({k: score_1x2(_P(pred, p), yi) for k, p in models.items()}).T.round(4).to_string())

    if has_mkt:
        m = pred["mkt_H"].notna().to_numpy()
        sub, ys = pred[m], yi[m]
        L.append(f"\n=== 1X2: MISMO SUBCONJUNTO CON CUOTAS DE MERCADO ({m.sum()} partidos) ===")
        tab = {k: score_1x2(_P(sub, p), ys) for k, p in models.items()}
        tab["mercado (sin margen)"] = score_1x2(_P(sub, "mkt"), ys)
        L.append(pd.DataFrame(tab).T.round(4).to_string())
        L.append("Lectura: si 'mercado' tiene menor logloss que 'modelo', el modelo NO aporta ventaja sobre las cuotas.")
        ll_m, ll_k = per_match_logloss(_P(sub, "m"), ys), per_match_logloss(_P(sub, "mkt"), ys)
        L.append("Bootstrap por fecha, mejora del MERCADO sobre el modelo (logloss): "
                 + str({k: round(v, 5) if isinstance(v, float) else v
                        for k, v in paired_bootstrap(ll_m, ll_k, sub["date"]).items()}))
        if "mkt_source" in sub:
            L.append("\nFuente de cuotas usada: " + str(sub["mkt_source"].value_counts().to_dict()))

    L.append("\n=== 1X2: modelo vs Elo y vs Poisson crudo (bootstrap por fecha; mejora = modelo mejor) ===")
    lm = per_match_logloss(_P(pred, "m"), yi)
    for k in ("elo", "dc"):
        r = paired_bootstrap(per_match_logloss(_P(pred, k), yi), lm, pred["date"])
        L.append(f"  vs {k}: " + str({a: round(b, 5) if isinstance(b, float) else b for a, b in r.items()}))

    L.append("\n=== 1X2 POR TEMPORADA (logloss) ===")
    rows = []
    for s, g in pred.groupby("season"):
        gi = y_index(g["target"])
        row = {"temporada": s, "n": len(g)}
        for k, p in models.items():
            row[k] = round(score_1x2(_P(g, p), gi)["logloss"], 4)
        if has_mkt and g["mkt_H"].notna().all():
            row["mercado"] = round(score_1x2(_P(g, "mkt"), gi)["logloss"], 4)
        rows.append(row)
    L.append(pd.DataFrame(rows).to_string(index=False))

    L.append("\n=== 1X2 POR LIGA (modelo; columnas explícitas A,D,H) ===")
    rows = []
    for lg, g in pred.groupby("competition"):
        sc = score_1x2(_P(g, "m"), y_index(g["target"]))
        rows.append({"liga": lg, **{k: round(v, 4) for k, v in sc.items()}})
    L.append(pd.DataFrame(rows).to_string(index=False))

    L.append("\n=== CALIBRACIÓN 1X2 (todas las probabilidades de clase, modelo) ===")
    P = _P(pred, "m")
    L.append(calibration_table(P.ravel(), np.eye(3)[yi].ravel()).to_string())

    L.append("\n=== MERCADOS DERIVADOS (logloss binario; menor = mejor) ===")
    rows = []
    for name in MARKETS:
        y = pred[f"y_{name}"].to_numpy(float)
        row = {"mercado": name, "n": len(y), "tasa_real": round(float(y.mean()), 4),
               "base": round(float(bin_logloss_each(pred[f"base_{name}"].to_numpy(), y).mean()), 4),
               "Poisson_crudo": round(float(bin_logloss_each(pred[f"raw_{name}"].to_numpy(), y).mean()), 4),
               "calibrado": round(float(bin_logloss_each(pred[f"cal_{name}"].to_numpy(), y).mean()), 4)}
        if name == "over_2_5" and "mkt_over_2_5" in pred and pred["mkt_over_2_5"].notna().any():
            mk = pred["mkt_over_2_5"].notna().to_numpy()
            row["| mismo subconj: n"] = int(mk.sum())
            for col in ("base", "raw", "cal", "mkt"):
                src = f"base_{name}" if col == "base" else (f"{col}_over_2_5" if col == "mkt" else f"{col}_{name}")
                row[f"sub_{col}"] = round(float(bin_logloss_each(pred.loc[mk, src].to_numpy(), y[mk]).mean()), 4)
        rows.append(row)
    L.append(pd.DataFrame(rows).to_string(index=False))
    L.append("Lectura: 'calibrado' < 'Poisson_crudo' => usar la columna *_cal; si no, usar *_raw. "
             "En over_2_5, 'sub_mkt' es la vara que hay que superar.")

    L.append("\n=== CALIBRACIÓN over_2_5 (modelo calibrado) ===")
    L.append(calibration_table(pred["cal_over_2_5"].to_numpy(), pred["y_over_2_5"].to_numpy(float)).to_string())
    return "\n".join(L)
