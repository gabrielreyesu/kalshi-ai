"""NFL: prueba prospectiva EN SOMBRA del modelo con quarterback titular, con datos publicados ANTES del partido.

Titular previsto (regla fijada antes de ver resultados): el primer QB del depth chart actual de ESPN que no figure
como Out / Doubtful / Injured Reserve / Suspension en el informe de lesiones capturado ahora. El "habitual" es el
titular más frecuente en los últimos 6 partidos del equipo (igual que en el backtest).

Cada ejecución agrega una foto a predictions/nfl_qb_shadow.csv (p_base = nfl_v1 reentrenado, p_lineup = + variante B)
y puntúa partidos jugados en predictions/nfl_qb_shadow_scoreboard.json. NO cambia nfl_registry.csv ni la página.

  python -m kalshi_lineup.nfl_shadow [--hours 96]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_nba.data import _json, load_league
from kalshi_nba.engine import CONFIGS, LEAGUE_FEATURES, NBAEngine
from kalshi_nba.models import NBAModels

from .box import load_boxes
from .nfl_qb import QB_FEATURES, SUMMARY_NFL, QBRater, parse_nfl_summary, regular_starter
from .shadow import OUT_STATUS, parse_injuries, score

P = Path("predictions")
SHADOW, BOARD = "nfl_qb_shadow.csv", "nfl_qb_shadow_scoreboard.json"
DEPTH = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team}/depthcharts"
INJ_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"


NAMES: dict = {}


def depth_qbs(payload: dict) -> list:
    """IDs de QB en orden del depth chart ofensivo (la primera formación que tenga 'qb')."""
    for d in (payload or {}).get("depthchart", []):
        qb = (d.get("positions") or {}).get("qb")
        if qb:
            ath = [a for a in qb.get("athletes", []) if a.get("id")]
            NAMES.update({str(a["id"]): a.get("displayName", "") for a in ath})
            return [str(a["id"]) for a in ath]
    return []


def expected_starter(qbs: list, outs: set) -> str | None:
    return next((q for q in qbs if q not in outs), None)


def snapshot(now, hours=96, games=None, boxes=None, depth=None, injuries=None, box_cache="data/nfl/box"):
    if games is None:
        games = load_league("nfl", first_season=2015, with_odds=False, verbose=False)
    games = games.copy(); games["event_id"] = games["event_id"].astype(str)
    done = games[games["status"] == "STATUS_FINAL"]
    if boxes is None:
        boxes = load_boxes(done["event_id"], cache_dir=box_cache, url=SUMMARY_NFL, parser=parse_nfl_summary)
    rater = QBRater()
    hist = rater.process(done, boxes)
    feats = NBAEngine(CONFIGS["nfl"]).process(games)
    feats["event_id"] = feats["event_id"].astype(str)
    train = feats.merge(hist, on="event_id")
    train = train[train["margin"].notna()]
    kick = pd.to_datetime(feats["kickoff_utc"], utc=True, format="ISO8601")
    up = feats[(feats["status"] != "STATUS_FINAL") & (kick > now) & (kick <= now + pd.Timedelta(hours=hours))].copy()
    if up.empty:
        return pd.DataFrame(), games
    inj = parse_injuries(injuries if injuries is not None else _json(INJ_URL))
    outs = {i["pid"] for i in inj if i["status"] in OUT_STATUS and i["pid"]}
    date = pd.Timestamp(now.tz_convert(None).normalize())
    rows = []
    for r in up.itertuples():
        side = {}
        for t in (str(r.home_id), str(r.away_id)):
            qbs = (depth or {}).get(t) if depth is not None else depth_qbs(_json(DEPTH.format(team=t)))
            today = expected_starter(qbs or [], outs)
            reg = regular_starter(rater.hist[t])
            v_today = rater.value(today, date) if today else np.nan
            v_reg = rater.value(reg, date) if reg else v_today
            side[t] = (today, reg, v_today, v_reg)
        h, a = side[str(r.home_id)], side[str(r.away_id)]
        ok = all(np.isfinite(x[2]) for x in (h, a))
        rows.append({"event_id": r.event_id, "qb_delta": ((h[2] - h[3]) - (a[2] - a[3])) if ok else 0.0,
                     "qb_abs": (h[2] - a[2]) if ok else 0.0, "qb_home": h[0], "qb_away": a[0],
                     "qb_change_home": int(bool(h[1]) and h[0] != h[1]), "qb_change_away": int(bool(a[1]) and a[0] != a[1]),
                     "depth_ok": int(ok)})
    up = up.merge(pd.DataFrame(rows), on="event_id")
    base_f = LEAGUE_FEATURES["nfl"]
    base = NBAModels(features=base_f).fit(train)
    qbm = NBAModels(features=base_f + QB_FEATURES).fit(train)
    out = pd.DataFrame({"snapshot_utc": now.isoformat(), "event_id": up["event_id"], "kickoff_utc": up["kickoff_utc"],
                        "home": up["home"], "away": up["away"],
                        "p_base": base.predict(up)["p_home"].round(5).to_numpy(),
                        "p_lineup": qbm.predict(up)["p_home"].round(5).to_numpy(),
                        "qb_home": up["qb_home"], "qb_away": up["qb_away"],
                        "qb_home_name": up["qb_home"].map(lambda x: NAMES.get(str(x), "")),
                        "qb_away_name": up["qb_away"].map(lambda x: NAMES.get(str(x), "")),
                        "qb_change_home": up["qb_change_home"], "qb_change_away": up["qb_change_away"],
                        "qb_delta": up["qb_delta"].round(3), "depth_ok": up["depth_ok"],
                        "minutes_to_tip": ((pd.to_datetime(up["kickoff_utc"], utc=True, format="ISO8601") - now).dt.total_seconds() / 60).round(0)})
    return out, games


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=96)
    a = ap.parse_args(argv)
    now = pd.Timestamp.now(tz="UTC")
    snap, games = snapshot(now, a.hours)
    path = P / SHADOW
    old = pd.read_csv(path, dtype={"event_id": str}) if path.exists() else pd.DataFrame()
    allrows = pd.concat([old, snap], ignore_index=True) if len(snap) else old
    if len(snap):
        P.mkdir(exist_ok=True); allrows.to_csv(path, index=False)
    board = {"status": "SHADOW_ONLY_PRODUCTION_UNCHANGED", "updated_utc": now.isoformat(),
             "regla": "Activar solo con n >= 200, IC95% de mejora_logloss > 0 y autorización explícita.",
             "nuevas_filas": int(len(snap)), "resultado": score(allrows, games) if len(allrows) else {"n": 0}}
    (P / BOARD).write_text(json.dumps(board, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(json.dumps(board, ensure_ascii=False))


if __name__ == "__main__":
    main()
