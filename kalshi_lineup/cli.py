"""python -m kalshi_lineup.cli backtest | backtest-nfl [--output reports/lineup]

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
from .nfl_qb import QB_FEATURES, SUMMARY_NFL, QBRater, parse_nfl_summary
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


def _write(out: Path, stem: str, res: dict, cov: dict, oos, txt: str):
    (out / f"{stem}_report.txt").write_text(txt, encoding="utf8")
    (out / f"{stem}_result.json").write_text(json.dumps({**res, "cobertura": cov}, indent=1, ensure_ascii=False, default=str), encoding="utf8")
    oos.to_csv(out / f"{stem}_oos.csv", index=False)
    print(txt)
    summ = os.getenv("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf8") as f:
            f.write("```\n" + txt + "\n```\n")


NFL_VARIANTS = {"A_qb_cambio": ["qb_delta"], "B_qb_cambio_y_nivel": QB_FEATURES}


def cmd_backtest_nfl(a):
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    games = load_league("nfl", first_season=a.first_season or 2015)
    final = games[games["status"] == "STATUS_FINAL"]
    boxes = load_boxes(final["event_id"], cache_dir=a.box_cache or "data/nfl/box", workers=a.workers,
                       url=SUMMARY_NFL, parser=parse_nfl_summary)
    cov = {"partidos_finales": int(len(final)), "con_boxscore": int(len(boxes)),
           "porcentaje": round(100 * len(boxes) / max(len(final), 1), 2)}
    print("Cobertura:", json.dumps(cov))
    if cov["porcentaje"] < 80:
        raise SystemExit("Cobertura de boxscores < 80%: no se evalúa (revisar esquema/red de ESPN).")
    feats = NBAEngine(CONFIGS["nfl"]).process(games)
    feats["event_id"] = feats["event_id"].astype(str)
    qf = QBRater().process(games, boxes)
    feats = feats.merge(qf, on="event_id", how="left")
    for c in QB_FEATURES:
        feats[c] = feats[c].fillna(0.0)
    oos = run_variants(feats, LEAGUE_FEATURES["nfl"], first_test=2019, min_train=1000, variants=NFL_VARIANTS)
    oos = oos.merge(qf[["event_id", "qb_change_h", "qb_change_a"]], on="event_id", how="left")
    res = decide(oos, feats, variants=NFL_VARIANTS, dev_last=2023)
    chg = oos[(oos["qb_change_h"] == 1) | (oos["qb_change_a"] == 1)]
    if len(chg):
        from .evaluate import compare
        res["solo_partidos_con_cambio_de_qb"] = compare(chg, f"p_{res['variante_elegida']}")
    cov["partidos_oos_con_cambio_de_qb"] = int(len(chg))
    txt = render(res, cov, title="NFL — QUARTERBACK TITULAR")
    if "solo_partidos_con_cambio_de_qb" in res:
        txt += "\n\n--- SOLO PARTIDOS CON CAMBIO DE QB (descriptivo) ---\n" + json.dumps(res["solo_partidos_con_cambio_de_qb"], ensure_ascii=False, indent=1)
    _write(out, "nfl_qb", res, cov, oos, txt)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest")
    b.add_argument("--first-season", type=int, default=2016)
    b.add_argument("--first-test", type=int, default=2019)
    b.add_argument("--box-cache", default="data/nba/box")
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--output", default="reports/lineup")
    n = sub.add_parser("backtest-nfl")
    n.add_argument("--first-season", type=int, default=2015)
    n.add_argument("--box-cache", default="data/nfl/box")
    n.add_argument("--workers", type=int, default=8)
    n.add_argument("--output", default="reports/lineup")
    a = ap.parse_args()
    {"backtest": cmd_backtest, "backtest-nfl": cmd_backtest_nfl}[a.cmd](a)


if __name__ == "__main__":
    main()
