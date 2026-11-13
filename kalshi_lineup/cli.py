"""python -m kalshi_lineup.cli backtest [--first-season 2016] [--output reports/lineup]

Descarga (con caché) resultados, líneas y boxscores de NBA desde ESPN, construye las variables de alineación y
ejecuta la comparación pre-registrada. Solo escribe informes; no toca predictions/ ni el modelo de producción.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from kalshi_nba.data import load_league
from kalshi_nba.engine import CONFIGS, LEAGUE_FEATURES, NBAEngine

from .box import load_boxes
from .evaluate import decide, render, run_variants
from .ratings import LineupRater


def cmd_backtest(a):
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    games = load_league("nba", first_season=a.first_season)
    final = games[games["status"] == "STATUS_FINAL"]
    boxes = load_boxes(final["event_id"], cache_dir=a.box_cache, workers=a.workers)
    cov = {"partidos_finales": int(len(final)), "con_boxscore": int(len(boxes)),
           "porcentaje": round(100 * len(boxes) / max(len(final), 1), 2)}
    by_season = final.assign(ok=final["event_id"].astype(str).isin(boxes)).groupby("season_year")["ok"].mean().round(3)
    cov["por_temporada"] = {int(k): float(v) for k, v in by_season.items()}
    print("Cobertura:", json.dumps(cov))
    if cov["porcentaje"] < 80:
        raise SystemExit("Cobertura de boxscores < 80%: no se evalúa (revisar esquema/red de ESPN).")
    feats = NBAEngine(CONFIGS["nba"]).process(games)
    feats["event_id"] = feats["event_id"].astype(str)
    lf = LineupRater().process(games, boxes)
    feats = feats.merge(lf, on="event_id", how="left")
    oos = run_variants(feats, LEAGUE_FEATURES["nba"], first_test=a.first_test, min_train=2000)
    res = decide(oos, feats)
    txt = render(res, cov)
    (out / "nba_lineup_report.txt").write_text(txt, encoding="utf8")
    (out / "nba_lineup_result.json").write_text(json.dumps({**res, "cobertura": cov}, indent=1, ensure_ascii=False, default=str), encoding="utf8")
    oos.to_csv(out / "nba_lineup_oos.csv", index=False)
    print(txt)
    summ = os.getenv("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf8") as f:
            f.write("```\n" + txt + "\n```\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest")
    b.add_argument("--first-season", type=int, default=2016)
    b.add_argument("--first-test", type=int, default=2019)
    b.add_argument("--box-cache", default="data/nba/box")
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--output", default="reports/lineup")
    a = ap.parse_args()
    {"backtest": cmd_backtest}[a.cmd](a)


if __name__ == "__main__":
    main()
