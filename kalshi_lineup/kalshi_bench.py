"""python -m kalshi_lineup.kalshi_bench --league nba|nfl [--output reports/kalshi_bench]

Une precios de Kalshi antes del partido, línea de cierre de las casas y predicciones fuera de muestra del modelo actual
y del modelo + alineación, y ejecuta la comparación pre-registrada (KALSHI_BENCHMARK.md). Solo escribe informes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from kalshi_nba.data import LEAGUES, fetch_odds, load_league, pick_line
from kalshi_nba.engine import CONFIGS, LEAGUE_FEATURES, NBAEngine

from .box import load_boxes
from .evaluate import run_variants
from .kalshi_eval import evaluate, render
from .kalshi_history import pregame_prices

CFG = {"nba": dict(series="KXNBAGAME", first=2016, min_train=2000, variant=["lineup_gs", "abs_gs"]),
       "nfl": dict(series="KXNFLGAME", first=2015, min_train=1000, variant=["qb_delta", "qb_abs"])}


def oos_predictions(lg: str, games: pd.DataFrame) -> pd.DataFrame:
    c = CFG[lg]
    done = games[games["status"] == "STATUS_FINAL"]
    feats = NBAEngine(CONFIGS[lg]).process(games)
    feats["event_id"] = feats["event_id"].astype(str)
    if lg == "nba":
        from .ratings import LineupRater
        boxes = load_boxes(done["event_id"], cache_dir="data/nba/box")
        lf = LineupRater().process(games, boxes)
    else:
        from .nfl_qb import SUMMARY_NFL, QBRater, parse_nfl_summary
        boxes = load_boxes(done["event_id"], cache_dir="data/nfl/box", url=SUMMARY_NFL, parser=parse_nfl_summary)
        lf = QBRater().process(games, boxes)
    feats = feats.merge(lf, on="event_id", how="left")
    for k in c["variant"]:
        feats[k] = feats[k].fillna(0.0)
    oos = run_variants(feats, LEAGUE_FEATURES[lg], first_test=2019, min_train=c["min_train"], variants={"var": c["variant"]})
    return oos.rename(columns={"p_var": "p_var"})[["event_id", "date", "home_win", "p_base", "p_var"]]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=list(CFG), required=True)
    ap.add_argument("--output", default="reports/kalshi_bench")
    a = ap.parse_args(argv)
    lg, c = a.league, CFG[a.league]
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    games = load_league(lg, first_season=c["first"], with_odds=False)
    games["event_id"] = games["event_id"].astype(str)
    for col in ("line_spread", "line_total", "line_p_home"):
        games[col] = float("nan")                  # la línea de las casas se añade aparte, solo para los partidos de Kalshi
    k = pregame_prices(c["series"], lg, games)
    L = LEAGUES[lg]
    lines = {e: pick_line(fetch_odds(e, Path(f"data/{lg}"), league=lg), L["total_range"], L["max_spread"]) for e in k["event_id"]}
    k["book_p_home"] = k["event_id"].map(lambda e: lines.get(e, {}).get("p_home"))
    oos = oos_predictions(lg, games)
    d = k.merge(oos, on="event_id").rename(columns={"home_win": "y"})
    print(f"Kalshi emparejados: {len(k)} | con predicción fuera de muestra: {len(d)}")
    res = evaluate(d)
    txt = render(f"{lg.upper()} — MODELO vs PRECIO DE KALSHI ANTES DEL PARTIDO (investigación)", res)
    (out / f"{lg}_kalshi_bench.txt").write_text(txt, encoding="utf8")
    (out / f"{lg}_kalshi_bench.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf8")
    d.to_csv(out / f"{lg}_kalshi_bench_rows.csv", index=False)
    print(txt)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf8") as f:
            f.write("```\n" + txt + "\n```\n")


if __name__ == "__main__":
    main()
