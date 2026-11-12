"""Uso:
  python -m kalshi_soccer.cli backtest [--start 2015] [--first-test 2223]
  python -m kalshi_soccer.cli predict  [--days 14] [--allow-new]
  python -m kalshi_soccer.cli evaluate
Variables de entorno: FOOTBALL_DATA_API_KEY (solo para `predict`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from . import MODEL_VERSION
from .backtest import report, walk_forward
from .blend import blend_report, blend_walk_forward
from .data import fetch_fixtures, load_history
from .engine import EngineConfig, SoccerEngine, season_code_for_date
from .models import MARKETS, CLASSES, DerivedMarket, Soccer1X2
from .kalshi import SERIES, attach_kalshi, edges, fetch_open_events
from .names import resolve_team
from .registry import REGISTRY, evaluate, register


def _features(start: int):
    hist = load_history(start_year=start)
    eng = SoccerEngine()
    feats = eng.process(hist)
    return hist, eng, feats


def cmd_backtest(a):
    _, _, feats = _features(a.start)
    pred = walk_forward(feats, first_test_season=a.first_test)
    Path("reports").mkdir(exist_ok=True)
    pred.to_csv("reports/soccer_backtest_predictions.csv", index=False)
    txt = report(pred)
    Path("reports/soccer_backtest_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


def cmd_blend(a):
    _, _, feats = _features(a.start)
    p = blend_walk_forward(feats, first_test_season=a.first_test)
    Path("reports").mkdir(exist_ok=True)
    p.to_csv("reports/soccer_blend_predictions.csv", index=False)
    txt = blend_report(p)
    Path("reports/soccer_blend_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


def cmd_predict(a):
    key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not key:
        raise SystemExit("Falta la variable de entorno FOOTBALL_DATA_API_KEY")
    hist, eng, feats = _features(a.start)
    scored = feats[feats["target"].notna()]
    model = Soccer1X2().fit(scored)
    markets = {n: DerivedMarket(n).fit(scored) for n in MARKETS}

    fx = fetch_fixtures(key, days=a.days)
    rows, unresolved = [], []
    for r in fx.itertuples():
        known = eng.known_teams(r.competition)
        home, mh = resolve_team(r.competition, r.home_api, known)
        away, ma = resolve_team(r.competition, r.away_api, known)
        if home is None or away is None:
            unresolved.append({"match_id": r.match_id, "competition": r.competition, "home_api": r.home_api,
                               "away_api": r.away_api, "home": mh if home is None else "ok", "away": ma if away is None else "ok"})
            continue
        season = season_code_for_date(r.kickoff_utc)
        pre = eng.predict(r.competition, season, home, away)
        row = pd.DataFrame([{**pre, "competition": r.competition}])
        P = model.predict_proba(row)[0]
        out = {"match_id": r.match_id, "kickoff_utc": r.kickoff_utc.isoformat(), "competition": r.competition,
               "home": home, "away": away, "model_version": MODEL_VERSION,
               "config_hash": hashlib.sha256(json.dumps(asdict(EngineConfig()), sort_keys=True).encode()).hexdigest()[:12],
               "new_team_home": pre["n_home"] == 0, "new_team_away": pre["n_away"] == 0,
               "elo_diff": pre["elo_diff"], "lambda_h": pre["lambda_h"], "lambda_a": pre["lambda_a"]}
        for j, c in enumerate(CLASSES):
            out[f"p_{c}"], out[f"dc_p_{c}"] = float(P[j]), pre[f"p_{c}"]
        for n, (pcol, _) in MARKETS.items():
            out[f"{pcol}_raw"] = pre[pcol]
            out[f"{pcol}_cal"] = float(markets[n].predict_proba(row)[0])
        rows.append(out)

    if unresolved:
        Path("predictions").mkdir(exist_ok=True)
        pd.DataFrame(unresolved).to_csv("predictions/unresolved_teams.csv", index=False)
        print(f"ATENCIÓN: {len(unresolved)} partidos con equipos sin resolver -> predictions/unresolved_teams.csv")
    preds = pd.DataFrame(rows)
    if preds.empty:
        print("No hay predicciones nuevas.")
        return
    try:
        evs = {lg: fetch_open_events(lg) for lg in SERIES}
        preds = attach_kalshi(preds, fx, evs)
        n = int(preds["k_event"].notna().sum())
        print(f"Kalshi: precios unidos a {n} de {len(preds)} partidos")
        snap = preds[preds["k_event"].notna()].assign(snapshot_utc=pd.Timestamp.now(tz="UTC").isoformat())
        if len(snap):
            sp = Path("predictions/kalshi_snapshots.csv")
            cols = ["match_id", "snapshot_utc", "k_event", "k_vol"] + [f"k_{k}_{c}" for k in ("bid", "ask", "last") for c in "HDA"]
            snap[cols].to_csv(sp, mode="a", header=not sp.exists(), index=False)
        Path("reports").mkdir(exist_ok=True)
        e = edges(preds)
        if len(e):
            e.to_csv("predictions/soccer_kalshi_edges.csv", index=False)
    except Exception as ex:  # noqa: BLE001  (Kalshi caído no debe romper el registro)
        print("AVISO: no se pudieron obtener precios de Kalshi:", ex)
    if not a.allow_new:
        preds = preds[~(preds["new_team_home"] | preds["new_team_away"])]
    print(register(preds, REGISTRY))


def cmd_evaluate(a):
    hist = load_history(start_year=max(a.start, 2024), verbose=False)
    evaluate(hist)


def main():
    ap = argparse.ArgumentParser(prog="kalshi_soccer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest"); b.add_argument("--start", type=int, default=2015); b.add_argument("--first-test", default="2223"); b.set_defaults(f=cmd_backtest)
    bl = sub.add_parser("blend"); bl.add_argument("--start", type=int, default=2015); bl.add_argument("--first-test", default="2223"); bl.set_defaults(f=cmd_blend)
    p = sub.add_parser("predict"); p.add_argument("--start", type=int, default=2015); p.add_argument("--days", type=int, default=14); p.add_argument("--allow-new", action="store_true"); p.set_defaults(f=cmd_predict)
    e = sub.add_parser("evaluate"); e.add_argument("--start", type=int, default=2024); e.set_defaults(f=cmd_evaluate)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
