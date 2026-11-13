"""Probabilidad FINAL en modo sombra: consenso de casas cuando existe, modelo propio como respaldo.

Motivo (AUDIT_V9.md): en ~25 mil partidos fuera de muestra el consenso de mercado acierta 2-3.4 puntos más que el
modelo en los cinco motores, y mezclar no supera al mercado solo. Este módulo NO cambia la producción: registra la
probabilidad final junto a la del modelo y mide ambas, de forma prospectiva, sobre los mismos partidos.

Uso:
  python -m kalshi_external.final_shadow build      # tras cada `predict`: agrega la foto actual
  python -m kalshi_external.final_shadow evaluate   # puntúa partidos ya jugados (última foto previa al inicio)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_nba.final_pick import combine, load_weights

P = Path("predictions")
SHADOW = "final_shadow.csv"
COLS = ["snapshot_utc", "sport", "event_id", "kickoff_utc", "home", "away", "model_H", "model_D", "model_A",
        "market_H", "market_D", "market_A", "final_H", "final_D", "final_A", "final_source", "kalshi_H"]
SOURCE_LABEL = {"market_consensus": "casas", "model_fallback": "modelo"}


def source_label(src) -> str:
    """Etiqueta legible del origen del pick (compatible con filas antiguas)."""
    return SOURCE_LABEL.get(str(src), str(src))


def _kalshi_mid(x) -> float:
    """Mid de Kalshi del local solo si el mercado tiene un spread razonable (<= 10¢)."""
    b, a = pd.to_numeric(getattr(x, "bid", np.nan), errors="coerce"), pd.to_numeric(getattr(x, "ask", np.nan), errors="coerce")
    if np.isfinite(b) and np.isfinite(a) and 0 < a - b <= 0.10 and 0 < (a + b) / 2 < 1:
        return float((a + b) / 2)
    return np.nan


def _final(model, market):
    """market si es un vector válido; si no, model. Devuelve (vector normalizado, origen)."""
    m = np.asarray(market, float)
    if np.isfinite(m).all() and (m >= 0).all() and m.sum() > 0:
        return m / m.sum(), "market_consensus"
    q = np.asarray(model, float)
    return q / q.sum(), "model_fallback"


def rows_us(lg: str, folder: Path) -> list:
    reg, snap = folder / f"{lg}_registry.csv", folder / f"{lg}_kalshi_snapshots.csv"
    if not reg.exists() or not snap.exists():
        return []
    r = pd.read_csv(reg, dtype={"event_id": str}).drop_duplicates("event_id")
    s = pd.read_csv(snap, dtype={"event_id": str})
    if "p_book" not in s:
        return []
    s = s[(s["kind"] == "winner") & (s["side"] == "home")]
    last = s["snapshot_utc"].max()
    s = s[s["snapshot_utc"] == last].drop_duplicates("event_id")
    out = []
    for x in s.merge(r[["event_id", "kickoff_utc", "home", "away"]], on="event_id").itertuples():
        pm, pb = pd.to_numeric(x.p_model, errors="coerce"), pd.to_numeric(x.p_book, errors="coerce")
        if not np.isfinite(pm) or not pd.Timestamp(x.snapshot_utc) < pd.Timestamp(x.kickoff_utc):
            continue
        km = _kalshi_mid(x)
        w = load_weights(lg)
        if w:                                     # pick final = Kalshi + casas + modelo (kalshi_nba/final_pick.py)
            p, src = combine(km, pb, pm, w)
            f = [p, 1 - p]
        else:
            f, src = _final([pm, 1 - pm], [pb, 1 - pb])
        out.append(dict(snapshot_utc=x.snapshot_utc, sport=lg, event_id=x.event_id, kickoff_utc=x.kickoff_utc, home=x.home, away=x.away,
                        model_H=pm, model_D=np.nan, model_A=1 - pm, market_H=pb, market_D=np.nan, market_A=1 - pb if np.isfinite(pb) else np.nan,
                        final_H=f[0], final_D=np.nan, final_A=f[1], final_source=src, kalshi_H=km))
    return out


def rows_intl(folder: Path) -> list:
    snap = folder / "intl_kalshi_snapshots.csv"
    if not snap.exists():
        return []
    s = pd.read_csv(snap, dtype={"event_id": str})
    s = s[s["snapshot_utc"] == s["snapshot_utc"].max()].drop_duplicates("event_id")
    out = []
    for x in s.itertuples():
        model = [getattr(x, c, np.nan) for c in ("p_H", "p_D", "p_A")]
        market = [pd.to_numeric(getattr(x, c, np.nan), errors="coerce") for c in ("mkt_H", "mkt_D", "mkt_A")]
        if not np.isfinite(np.asarray(model, float)).all() or not pd.Timestamp(x.snapshot_utc) < pd.Timestamp(x.kickoff_utc):
            continue
        f, src = _final(model, market)
        out.append(dict(snapshot_utc=x.snapshot_utc, sport="intl" if x.grupo == "selecciones" else "club", event_id=x.event_id,
                        kickoff_utc=x.kickoff_utc, home=x.home, away=x.away, model_H=model[0], model_D=model[1], model_A=model[2],
                        market_H=market[0], market_D=market[1], market_A=market[2], final_H=f[0], final_D=f[1], final_A=f[2], final_source=src))
    return out


def build(folder: Path = P) -> int:
    rows = rows_us("nba", folder) + rows_us("nfl", folder) + rows_intl(folder)
    if not rows:
        return 0
    new = pd.DataFrame(rows)[COLS]
    path = folder / SHADOW
    if path.exists():
        old = pd.read_csv(path, dtype={"event_id": str})
        new = pd.concat([old, new], ignore_index=True).drop_duplicates(["sport", "event_id", "snapshot_utc"], keep="last")
    new.to_csv(path, index=False)
    return len(rows)


def score(shadow: pd.DataFrame, outcomes: pd.DataFrame, bootstrap: int = 2000, seed: int = 7) -> pd.DataFrame:
    """Modelo vs final en los MISMOS partidos, usando la última foto anterior al inicio. IC por remuestreo de fechas."""
    s = shadow.copy()
    s["snapshot_utc"], s["kickoff_utc"] = pd.to_datetime(s["snapshot_utc"], utc=True, format="ISO8601"), pd.to_datetime(s["kickoff_utc"], utc=True, format="ISO8601")
    s = s[s["snapshot_utc"] < s["kickoff_utc"]].sort_values("snapshot_utc").drop_duplicates(["sport", "event_id"], keep="last")
    d = s.merge(outcomes.assign(event_id=outcomes["event_id"].astype(str)), on=["sport", "event_id"])
    out = []
    for sport, g in d.groupby("sport"):
        three = sport in ("intl", "club")
        labels = {"H": 0, "D": 1, "A": 2} if three else {"H": 0, "A": 1}
        cols = ["H", "D", "A"] if three else ["H", "A"]
        y = g["result"].map(labels)
        g, y = g[y.notna()], y[y.notna()].to_numpy(int)
        if not len(g):
            continue
        M = g[[f"model_{c}" for c in cols]].to_numpy(float); F = g[[f"final_{c}" for c in cols]].to_numpy(float)
        M, F = M / M.sum(1, keepdims=True), F / F.sum(1, keepdims=True)
        i = np.arange(len(y)); lm, lf = -np.log(np.clip(M[i, y], 1e-12, 1)), -np.log(np.clip(F[i, y], 1e-12, 1))
        diff, days = lm - lf, g["kickoff_utc"].dt.normalize().to_numpy()
        groups = [np.where(days == u)[0] for u in np.unique(days)]
        lo = hi = np.nan
        if bootstrap and len(groups) > 1:
            rng = np.random.default_rng(seed)
            draws = [diff[np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])].mean() for _ in range(bootstrap)]
            lo, hi = np.percentile(draws, [2.5, 97.5])
        out.append(dict(sport=sport, n=len(y), con_mercado=int((~g["final_source"].isin(["model_fallback", "modelo"])).sum()),
                        acierto_modelo=float((M.argmax(1) == y).mean()), acierto_final=float((F.argmax(1) == y).mean()),
                        logloss_modelo=float(lm.mean()), logloss_final=float(lf.mean()),
                        mejora_logloss=float(diff.mean()), ic95_inf=float(lo), ic95_sup=float(hi)))
    return pd.DataFrame(out)


def evaluate(folder: Path = P) -> pd.DataFrame:
    path = folder / SHADOW
    if not path.exists():
        print("Aún no hay fotos de probabilidad final."); return pd.DataFrame()
    shadow = pd.read_csv(path, dtype={"event_id": str})
    from .outcomes import live_outcomes
    outcomes, warn = live_outcomes(shadow[["sport", "event_id"]].drop_duplicates())
    t = score(shadow, outcomes)
    (folder / "final_shadow_scoreboard.json").write_text(json.dumps({
        "status": "SHADOW_ONLY_PRODUCTION_UNCHANGED", "updated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "promotion_rule": "Por deporte: n >= 200 partidos resueltos con mercado e IC95% de mejora_logloss completamente > 0; la activación requiere autorización explícita.",
        "warnings": warn, "by_sport": t.round(5).to_dict("records")}, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(t.round(4).to_string(index=False) if len(t) else "Aún no hay partidos resueltos.")
    return t


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    print("filas agregadas:", build()) if cmd == "build" else evaluate()
