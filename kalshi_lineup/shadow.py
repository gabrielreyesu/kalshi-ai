"""NBA: prueba prospectiva EN SOMBRA del modelo con alineación, usando el informe de lesiones REAL antes del partido.

El backtest histórico infería la disponibilidad del boxscore final. Aquí se usa solo lo que se sabe antes del salto:
  * rotación reciente del equipo (jugadores disponibles en sus últimos 10 partidos y que siguen en el equipo);
  * menos los jugadores con estado Out / Doubtful / Injured Reserve / Suspension en el informe de ESPN capturado ahora.
Questionable y Day-To-Day cuentan como disponibles (regla fijada antes de ver resultados).

Cada ejecución:
  1. agrega una foto a predictions/nba_lineup_shadow.csv (p_base = nba_v1 reentrenado, p_lineup = + variante D);
  2. puntúa los partidos ya jugados con la última foto anterior al inicio y escribe
     predictions/nba_lineup_shadow_scoreboard.json.
NO cambia nba_registry.csv ni PREDICCIONES.md. Activar requiere n >= 200, IC95% > 0 y autorización explícita.

  python -m kalshi_lineup.shadow [--hours 36]
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_nba.backtest import logloss, paired_bootstrap
from kalshi_nba.data import _json, load_league
from kalshi_nba.engine import CONFIGS, LEAGUE_FEATURES, NBAEngine
from kalshi_nba.models import NBAModels

from .box import load_boxes
from .ratings import LineupRater

P = Path("predictions")
SHADOW, BOARD = "nba_lineup_shadow.csv", "nba_lineup_shadow_scoreboard.json"
OUT_STATUS = {"out", "doubtful", "injured reserve", "suspension", "out for season"}
EXTRA = ["lineup_gs", "abs_gs"]                 # variante D, elegida en el backtest pre-registrado
INJ_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s)
    return re.sub(r"[^a-z]", "", s)


def parse_injuries(j: dict) -> list:
    out = []
    for t in (j or {}).get("injuries", []):
        for x in t.get("injuries", []):
            a = x.get("athlete") or {}
            pid = str(a.get("id") or "")
            if not pid:
                for l in a.get("links") or []:
                    m = re.search(r"/id/([^/?#]+)", str(l.get("href", "")))
                    if m:
                        pid = m.group(1); break
            out.append({"team_id": str(t.get("id")), "pid": pid, "name": a.get("displayName", ""),
                        "status": str(x.get("status") or "").lower()})
    return out


def rotation(rater: LineupRater, team: str) -> set:
    recent = set().union(*[set(a) for a, _ in rater.recent[team]]) if rater.recent[team] else set()
    return {p for p in recent if rater.p.get(p, {}).get("team") == team}


def team_features(rater, team, date, injuries, names) -> dict:
    rot = rotation(rater, team)
    outs = [i for i in injuries if i["team_id"] == team and i["status"] in OUT_STATUS]
    by_name = {norm(names.get(p, "")): p for p in rot}
    removed, unmatched = set(), []
    for i in outs:
        pid = i["pid"] if i["pid"] in rot else by_name.get(norm(i["name"]))
        (removed.add(pid) if pid else unmatched.append(i["name"]))
    avail = sorted(rot - removed)
    f = rater.team_features(team, date, avail, [])
    return {"gs": f["gs"], "d_gs": f["d_gs"], "out": [names.get(p, p) for p in removed], "unmatched": unmatched}


def snapshot(now: pd.Timestamp, hours=36, first_season=2016, box_cache="data/nba/box", injuries=None, games=None, boxes=None):
    if games is None:
        games = load_league("nba", first_season=first_season, with_odds=False, verbose=False)
    games = games.copy()
    games["event_id"] = games["event_id"].astype(str)
    done = games[games["status"] == "STATUS_FINAL"]
    if boxes is None:
        boxes = load_boxes(done["event_id"], cache_dir=box_cache)
    names = {p["pid"]: p.get("name", "") for b in boxes.values() for p in b["players"]}
    rater = LineupRater()
    hist = rater.process(done, boxes)
    feats = NBAEngine(CONFIGS["nba"]).process(games)
    feats["event_id"] = feats["event_id"].astype(str)
    train = feats.merge(hist, on="event_id", how="inner")
    train = train[train["margin"].notna()]
    kick = pd.to_datetime(feats["kickoff_utc"], utc=True, format="ISO8601")
    up = feats[(feats["status"] != "STATUS_FINAL") & (kick > now) & (kick <= now + pd.Timedelta(hours=hours))].copy()
    if up.empty:
        return pd.DataFrame(), games
    inj = parse_injuries(injuries if injuries is not None else _json(INJ_URL))
    date = pd.Timestamp(now.tz_convert(None).normalize())
    rows = []
    for r in up.itertuples():
        h = team_features(rater, str(r.home_id), date, inj, names)
        a = team_features(rater, str(r.away_id), date, inj, names)
        rows.append({"event_id": r.event_id, "lineup_gs": h["d_gs"] - a["d_gs"],
                     "abs_gs": (h["gs"] - a["gs"]) if np.isfinite(h["gs"]) and np.isfinite(a["gs"]) else 0.0,
                     "out_home": "; ".join(h["out"]), "out_away": "; ".join(a["out"]),
                     "unmatched_out": len(h["unmatched"]) + len(a["unmatched"])})
    lf = pd.DataFrame(rows)
    up = up.merge(lf, on="event_id")
    base_f = LEAGUE_FEATURES["nba"]
    base = NBAModels(features=base_f).fit(train)
    lin = NBAModels(features=base_f + EXTRA).fit(train)
    out = pd.DataFrame({"snapshot_utc": now.isoformat(), "event_id": up["event_id"], "kickoff_utc": up["kickoff_utc"],
                        "home": up["home"], "away": up["away"],
                        "p_base": base.predict(up)["p_home"].round(5).to_numpy(),
                        "p_lineup": lin.predict(up)["p_home"].round(5).to_numpy(),
                        "lineup_gs": up["lineup_gs"].round(3), "abs_gs": up["abs_gs"].round(3),
                        "out_home": up["out_home"], "out_away": up["out_away"], "unmatched_out": up["unmatched_out"],
                        "minutes_to_tip": ((pd.to_datetime(up["kickoff_utc"], utc=True, format="ISO8601") - now).dt.total_seconds() / 60).round(0)})
    return out, games


def score(shadow: pd.DataFrame, games: pd.DataFrame, bootstrap=2000) -> dict:
    s = shadow.copy()
    s["snapshot_utc"] = pd.to_datetime(s["snapshot_utc"], utc=True, format="ISO8601")
    s["kick"] = pd.to_datetime(s["kickoff_utc"], utc=True, format="ISO8601")
    s = s[s["snapshot_utc"] < s["kick"]].sort_values("snapshot_utc").drop_duplicates("event_id", keep="last")
    g = games[games["status"] == "STATUS_FINAL"][["event_id", "hs", "as_"]].copy()
    g["event_id"] = g["event_id"].astype(str)
    d = s.assign(event_id=s["event_id"].astype(str)).merge(g, on="event_id")
    d = d[d["hs"] != d["as_"]]
    if d.empty:
        return {"n": 0}
    y = (d["hs"] > d["as_"]).astype(int).to_numpy()
    lb, ll = logloss(d["p_base"], y), logloss(d["p_lineup"], y)
    res = {"n": int(len(d)), "acierto_base": round(float(((d["p_base"] >= .5) == (y == 1)).mean()), 4),
           "acierto_lineup": round(float(((d["p_lineup"] >= .5) == (y == 1)).mean()), 4),
           "logloss_base": round(float(lb.mean()), 5), "logloss_lineup": round(float(ll.mean()), 5),
           "minutos_antes_mediana": float(d["minutes_to_tip"].median())}
    if d["kick"].dt.normalize().nunique() > 1:
        res["mejora_logloss"] = paired_bootstrap(lb, ll, d["kick"].dt.normalize(), B=bootstrap)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=36)
    ap.add_argument("--box-cache", default="data/nba/box")
    a = ap.parse_args(argv)
    now = pd.Timestamp.now(tz="UTC")
    snap, games = snapshot(now, a.hours, box_cache=a.box_cache)
    path = P / SHADOW
    old = pd.read_csv(path, dtype={"event_id": str}) if path.exists() else pd.DataFrame()
    allrows = pd.concat([old, snap], ignore_index=True) if len(snap) else old
    if len(snap):
        P.mkdir(exist_ok=True)
        allrows.to_csv(path, index=False)
    board = {"status": "SHADOW_ONLY_PRODUCTION_UNCHANGED", "updated_utc": now.isoformat(),
             "regla": "Activar solo con n >= 200, IC95% de mejora_logloss > 0 y autorización explícita.",
             "nuevas_filas": int(len(snap)), "resultado": score(allrows, games) if len(allrows) else {"n": 0}}
    (P / BOARD).write_text(json.dumps(board, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(json.dumps(board, ensure_ascii=False))


if __name__ == "__main__":
    main()
