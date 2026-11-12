"""Uso:
  python -m kalshi_nba.cli backtest            # walk-forward + comparación contra líneas (descarga cuotas: lento la 1ª vez)
  python -m kalshi_nba.cli predict [--days 7]  # predicciones + precios de Kalshi, registro append-only
  python -m kalshi_nba.cli evaluate            # modelo vs Kalshi en mercados ya resueltos
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import MODEL_VERSION
from .backtest import blend_report, blend_walk_forward, logloss, paired_bootstrap, report, walk_forward
from .data import fetch_live_line, load_league, load_nba
from .engine import CONFIGS, NBAEngine
from .kalshi import LEAGUE_SERIES, attach_games, fetch_open, model_probability, parse_markets, settle
from .models import NBAModels

def _paths(lg):
    return Path(f"predictions/{lg}_registry.csv"), Path(f"predictions/{lg}_kalshi_snapshots.csv")


# Dispersión del resultado alrededor de la línea de las casas (margen, total), medida en el backtest
BOOK_SIGMA = {"nba": (13.4, 17.9), "nfl": (12.7, 13.0)}
FEE_RATE = 0.07      # SUPUESTO: comisión por contrato ~ 0.07 * precio * (1 - precio). Verifica la tarifa vigente de Kalshi.


def fee(price):
    return FEE_RATE * price * (1 - price)


class _BookPricer:
    """Misma interfaz que NBAModels para valorar spreads/totales, pero centrada en la línea de las casas."""
    def __init__(self, sig_m, sig_t):
        self.sig_m, self.sig_t = sig_m, sig_t
    p_home_wins_by_more_than = NBAModels.p_home_wins_by_more_than
    p_total_over = NBAModels.p_total_over


def _append(df: pd.DataFrame, path: Path):
    """Agrega filas; si llegan columnas nuevas, reescribe el archivo conservando todas las filas anteriores."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old_cols = list(pd.read_csv(path, nrows=0).columns)
        if set(df.columns) - set(old_cols):
            old = pd.read_csv(path, dtype=str)
            pd.concat([old, df.astype(object)], ignore_index=True).to_csv(path, index=False)
            return
        df = df.reindex(columns=old_cols)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


DEFAULTS = {"nba": dict(first_season=2016, first_test=2019, min_train=2000),
            "nfl": dict(first_season=2015, first_test=2019, min_train=1000),
            "ncaab": dict(first_season=2019, first_test=2022, min_train=5000)}


def cmd_backtest(a):
    D = DEFAULTS[a.league]
    f = NBAEngine(CONFIGS[a.league]).process(load_league(a.league, first_season=a.first_season or D["first_season"]))
    ft = a.first_test or D["first_test"]
    txt = f"LIGA: {a.league.upper()}\n" + report(walk_forward(f, first_test=ft, min_train=D["min_train"])) + "\n\n" + "=" * 70 + \
        "\n¿APORTA EL MODELO SOBRE LA LÍNEA? (mezcla)\n" + "=" * 70 + "\n" + \
        blend_report(blend_walk_forward(f, first_test=ft, min_train=D["min_train"]))
    Path("reports").mkdir(exist_ok=True)
    Path(f"reports/{a.league}_backtest_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


def cmd_predict(a):
    now = pd.Timestamp.now(tz="UTC")
    lg = a.league
    if lg not in LEAGUE_SERIES:
        raise SystemExit(f"predict aún no está disponible para {lg} (falta unir equipos con Kalshi)")
    REG, SNAP = _paths(lg)
    g = load_league(lg, first_season=a.first_season or DEFAULTS[lg]["first_season"], with_odds=False)
    f = NBAEngine(CONFIGS[lg]).process(g)
    models = NBAModels().fit(f)
    kick = pd.to_datetime(f["kickoff_utc"], utc=True)
    up = f[(f["status"] != "STATUS_FINAL") & (kick > now) & (kick <= now + pd.Timedelta(days=a.days))].copy()
    if up.empty:
        print("No hay partidos próximos."); return
    pr = models.predict(up)
    preds = pd.concat([up[["event_id", "kickoff_utc", "home", "away", "home_id", "away_id", "elo_diff", "rest_h", "rest_a"]], pr], axis=1)
    preds["sig_m"], preds["sig_t"], preds["model_version"], preds["created_utc"] = models.sig_m, models.sig_t, MODEL_VERSION, now.isoformat()
    seen = set(pd.read_csv(REG, dtype={"event_id": str})["event_id"]) if REG.exists() else set()
    new = preds[~preds["event_id"].isin(seen)]
    if len(new):
        _append(new, REG)
    print(f"Partidos próximos: {len(preds)} | nuevos en el registro: {len(new)}")
    try:
        mk = pd.concat([parse_markets(k, fetch_open(k, lg)) for k in LEAGUE_SERIES[lg]], ignore_index=True)
        mk = model_probability(attach_games(mk, up, lg), preds, models)
        # valor "justo" según las casas de apuestas (ya incluye lesiones y noticias): línea de consenso actual
        lines = {e: fetch_live_line(e, lg) for e in mk["event_id"].unique()}
        bk = pd.DataFrame([{"event_id": e, "p_home": l.get("p_home"), "mu_margin": -l["spread"], "mu_total": l["total"]}
                           for e, l in lines.items() if l and l.get("p_home") is not None])
        mk["p_book"] = np.nan
        if len(bk):
            b = model_probability(mk.drop(columns=["p_home", "mu_margin", "mu_total", "p_model", "mid", "ventaja_bruta"]), bk, _BookPricer(*BOOK_SIGMA[lg]))
            mk["p_book"] = mk["ticker"].map(dict(zip(b["ticker"], b["p_model"])))
        mk["ventaja_casas_neta"] = mk["p_book"] - mk["ask"] - fee(mk["ask"])
        mk["snapshot_utc"] = now.isoformat()
        mk = mk[pd.to_datetime(mk["when"], utc=True) > now]                 # nunca mercados de partidos ya iniciados
        cols = ["snapshot_utc", "event_id", "ticker", "kind", "side", "strike", "title", "bid", "ask", "last", "vol", "mid", "p_model", "ventaja_bruta", "p_book", "ventaja_casas_neta"]
        if len(mk):
            _append(mk[cols], SNAP)
            mk[cols].sort_values("ventaja_casas_neta", ascending=False).to_csv(f"predictions/{lg}_kalshi_edges.csv", index=False)
        print(f"Kalshi: {len(mk)} mercados unidos ({mk['kind'].value_counts().to_dict() if len(mk) else {}})")
    except Exception as ex:  # noqa: BLE001
        print("AVISO: no se pudieron obtener precios de Kalshi:", ex)


def cmd_evaluate(a):
    lg = a.league
    REG, SNAP = _paths(lg)
    if not SNAP.exists():
        print("Aún no hay fotos de precios."); return
    from .data import current_season
    g = load_league(lg, first_season=current_season(lg), with_odds=False, verbose=False)
    fin = g[g["status"] == "STATUS_FINAL"]
    s = pd.read_csv(SNAP, dtype={"event_id": str}).sort_values("snapshot_utc").drop_duplicates("ticker")   # primera foto de cada mercado
    for c in ("p_model", "mid", "ask", "bid", "p_book"):
        s[c] = pd.to_numeric(s[c], errors="coerce") if c in s else np.nan
    d = settle(s, fin).dropna(subset=["p_model", "mid"])
    d = d[(d["mid"] > 0) & (d["mid"] < 1)]
    if d.empty:
        print("Aún no hay mercados resueltos."); return
    rows = []
    for k, x in d.groupby("kind"):
        rows.append({"mercado": k, "n": len(x), "partidos": x["event_id"].nunique(),
                     "logloss_modelo": round(float(logloss(x["p_model"], x["y"]).mean()), 4),
                     "logloss_kalshi": round(float(logloss(x["mid"], x["y"]).mean()), 4),
                     **{"boot_" + kk: v for kk, v in paired_bootstrap(logloss(x["mid"], x["y"]), logloss(x["p_model"], x["y"]), x["event_id"]).items()}})
    t = pd.DataFrame(rows)
    d.to_csv(f"predictions/{lg}_evaluation.csv", index=False)
    print(t.to_string(index=False))
    print("boot_mejora > 0 con IC que excluye 0 => el modelo le gana al precio de Kalshi (remuestreo por partido).")
    b = d.dropna(subset=["p_book", "ask"])
    b = b[(b["ask"] > 0) & (b["ask"] < 1)]
    if len(b):
        print(f"\nCASAS DE APUESTAS vs KALSHI ({len(b)} mercados, {b['event_id'].nunique()} partidos)")
        rows = []
        for k, x in b.groupby("kind"):
            row = {"mercado": k, "n": len(x), "logloss_casas": round(float(logloss(x["p_book"], x["y"]).mean()), 4),
                   "logloss_kalshi": round(float(logloss(x["mid"], x["y"]).mean()), 4)}
            for thr in (0.02, 0.05):
                bet = x[x["p_book"] - x["ask"] - fee(x["ask"]) > thr]
                pnl = bet["y"] - bet["ask"] - fee(bet["ask"])
                row[f"apuestas>{thr}"] = len(bet)
                row[f"ganancia_por_contrato>{thr}"] = round(float(pnl.mean()), 4) if len(bet) else np.nan
            rows.append(row)
        print(pd.DataFrame(rows).to_string(index=False))
        print("Simulación: comprar 'Yes' al ask cuando casas - ask - comisión supuesta > umbral. Con pocos partidos es ruido.")


def main():
    ap = argparse.ArgumentParser(prog="kalshi_nba")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("backtest", cmd_backtest), ("predict", cmd_predict), ("evaluate", cmd_evaluate)):
        p = sub.add_parser(name); p.add_argument("--first-season", type=int, default=None)
        p.add_argument("--first-test", type=int, default=None); p.add_argument("--days", type=int, default=7)
        p.add_argument("--league", default="nba", choices=["nba", "ncaab", "nfl"]); p.set_defaults(f=fn)
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
