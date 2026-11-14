"""python -m kalshi_lineup.travel_eval --league nba|nfl   (protocolo: TRAVEL_RESEARCH.md)"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kalshi_nba.data import load_league
from kalshi_nba.engine import CONFIGS, LEAGUE_FEATURES, NBAEngine

from .evaluate import decide, render, run_variants
from .travel import TRAVEL_FEATURES, compute

VARIANTS = {"A_distancia": ["travel"], "B_distancia_husos_reloj": TRAVEL_FEATURES}
CFG = {"nba": dict(first=2016, min_train=2000, dev_last=2024), "nfl": dict(first=2015, min_train=1000, dev_last=2023)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=list(CFG), required=True)
    ap.add_argument("--output", default="reports/travel")
    a = ap.parse_args(argv)
    c = CFG[a.league]
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    games = load_league(a.league, first_season=c["first"])
    feats = NBAEngine(CONFIGS[a.league]).process(games)
    feats["event_id"] = feats["event_id"].astype(str)
    feats = feats.merge(compute(games, a.league), on="event_id", how="left")
    for k in TRAVEL_FEATURES:
        feats[k] = feats[k].fillna(0.0)
    oos = run_variants(feats, LEAGUE_FEATURES[a.league], first_test=2019, min_train=c["min_train"], variants=VARIANTS)
    res = decide(oos, feats, variants=VARIANTS, dev_last=c["dev_last"])
    txt = render(res, {"partidos_oos": int(len(oos))}, title=f"{a.league.upper()} — VIAJES Y HORA DEL PARTIDO")
    (out / f"{a.league}_travel.txt").write_text(txt, encoding="utf8")
    (out / f"{a.league}_travel.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf8")
    print(txt)


if __name__ == "__main__":
    main()
