"""Copas europeas: prueba prospectiva EN SOMBRA del once titular, con el once OFICIAL publicado antes del inicio.

ESPN publica los onces ~25-60 min antes. Cada ejecución (cada 15 min en horario de copas):
  1. busca partidos de Champions / Europa League / Conference que empiecen en los próximos 5-80 minutos;
  2. si su resumen ya trae 11 titulares por equipo y el partido sigue programado, calcula la continuidad del once
     (variante A, elegida en el backtest pre-registrado) y registra p_base (intl_v1 reentrenado) y p_xi (+ xi_cont);
  3. puntúa partidos jugados (1X2 a 90 min) en predictions/soccer_xi_shadow_scoreboard.json.
Si no hay partidos próximos, termina en segundos sin cargar historial. NO cambia intl_registry.csv ni la página.

  python -m kalshi_lineup.soccer_shadow            # ejecución normal
  python -m kalshi_lineup.soccer_shadow --warm     # solo descarga/actualiza el historial de onces (caché)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_nba.backtest import paired_bootstrap
from kalshi_nba.data import _json
from kalshi_intl.data import load_group
from kalshi_intl.model import CONFIGS, GROUP_FEATURES, IntlEngine, IntlModel

from .box import load_boxes
from .soccer_eval import CLUB_FEED
from .soccer_xi import SUMMARY_SOCCER, XIRater, parse_soccer_summary

P = Path("predictions")
SHADOW, BOARD = "soccer_xi_shadow.csv", "soccer_xi_shadow_scoreboard.json"
REGULAR = {"STATUS_FULL_TIME", "STATUS_FINAL"}
IDX = {"H": 0, "D": 1, "A": 2}


def history_boxes(g, since="2018-07-01", box_cache="data/intl_box", workers=8) -> dict:
    fin = g[g["final"] & (g["date"] >= pd.Timestamp(since)) & (g["target"] | g["code"].isin(CLUB_FEED))]
    boxes = {}
    for code, part in fin.groupby("code"):
        boxes.update(load_boxes(part["event_id"], cache_dir=f"{box_cache}/{code}", workers=workers,
                                url=SUMMARY_SOCCER.format(code=code), parser=parse_soccer_summary))
    return boxes


def upcoming(g, now, lo=5, hi=80):
    kick = g["kick"]
    return g[g["target"] & ~g["final"] & (g["status"] == "STATUS_SCHEDULED")
             & (kick > now + pd.Timedelta(minutes=lo)) & (kick <= now + pd.Timedelta(minutes=hi))]


def snapshot(now, g=None, boxes=None, lineups=None, box_cache="data/intl_box"):
    """lineups: {event_id: parse_soccer_summary(..., pregame=True)} (para pruebas); si es None se piden a ESPN."""
    if g is None:
        g = load_group("clubes", verbose=False)
    g = g.copy(); g["event_id"] = g["event_id"].astype(str)
    up = upcoming(g, now)
    if up.empty:
        return pd.DataFrame(), g
    xi_now = {}
    for r in up.itertuples():
        b = (lineups or {}).get(r.event_id) if lineups is not None else \
            parse_soccer_summary(_json(SUMMARY_SOCCER.format(code=r.code), {"event": r.event_id}) or {}, r.event_id, pregame=True)
        if b:
            xi_now[r.event_id] = b
    if not xi_now:
        return pd.DataFrame(), g
    if boxes is None:
        boxes = history_boxes(g, box_cache=box_cache)
    done = g[g["final"]]
    rater = XIRater()
    hist = rater.process(done, boxes)
    f = IntlEngine(CONFIGS["clubes"]).process(g)
    f["event_id"] = f["event_id"].astype(str)
    f = f.merge(hist[["event_id", "xi_cont"]], on="event_id", how="left")
    f["xi_cont"] = f["xi_cont"].fillna(0.0)
    rows = []
    for e, b in xi_now.items():
        r = up[up["event_id"] == e].iloc[0]
        xi = lambda t: [p["pid"] for p in b["players"] if p["team_id"] == str(t) and p["starter"]]
        h, a = rater.team(str(r.home_id), xi(r.home_id)), rater.team(str(r.away_id), xi(r.away_id))
        rows.append({"event_id": e, "xi_cont_now": (h[0] - a[0]) if (h[2] and a[2]) else 0.0, "xi_ok": int(h[2] and a[2]),
                     "cont_home": round(h[0], 3), "cont_away": round(a[0], 3)})
    x = pd.DataFrame(rows)
    target = f[f["event_id"].isin(x["event_id"])].merge(x, on="event_id")
    target["xi_cont"] = target["xi_cont_now"]
    train = f[f["final"]]
    base_f = GROUP_FEATURES["clubes"]
    pb = IntlModel(base_f).fit(train).predict(target)
    px = IntlModel(base_f + ["xi_cont"]).fit(train).predict(target)
    out = pd.DataFrame({"snapshot_utc": now.isoformat(), "event_id": target["event_id"].to_numpy(),
                        "kickoff_utc": target["kick"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy(),
                        "competicion": target["code"].to_numpy(), "home": target["home"].to_numpy(), "away": target["away"].to_numpy(),
                        **{f"base_{c}": pb[f"p_{c}"].round(5).to_numpy() for c in "HDA"},
                        **{f"xi_{c}": px[f"p_{c}"].round(5).to_numpy() for c in "HDA"},
                        "xi_cont": target["xi_cont"].round(4).to_numpy(), "xi_ok": target["xi_ok"].to_numpy(),
                        "cont_home": target["cont_home"].to_numpy(), "cont_away": target["cont_away"].to_numpy(),
                        "minutes_to_kick": ((target["kick"] - now).dt.total_seconds() / 60).round(0).to_numpy()})
    return out, g


def score(shadow: pd.DataFrame, g: pd.DataFrame, bootstrap=2000) -> dict:
    s = shadow.copy()
    s["snapshot_utc"] = pd.to_datetime(s["snapshot_utc"], utc=True, format="ISO8601")
    s["kick"] = pd.to_datetime(s["kickoff_utc"], utc=True, format="ISO8601")
    s = s[s["snapshot_utc"] < s["kick"]].sort_values("snapshot_utc").drop_duplicates("event_id", keep="last")
    res = g[g["final"] & g["status"].isin(REGULAR)][["event_id", "hs", "as_"]].copy()
    res["event_id"] = res["event_id"].astype(str)
    d = s.assign(event_id=s["event_id"].astype(str)).merge(res, on="event_id")
    if d.empty:
        return {"n": 0}
    y = np.where(d["hs"] > d["as_"], 0, np.where(d["hs"] == d["as_"], 1, 2))
    ll = lambda pre: -np.log(np.clip(d[[f"{pre}_{c}" for c in "HDA"]].to_numpy(float)[np.arange(len(d)), y], 1e-12, 1))
    acc = lambda pre: float((d[[f"{pre}_{c}" for c in "HDA"]].to_numpy(float).argmax(1) == y).mean())
    lb, lx = ll("base"), ll("xi")
    out = {"n": int(len(d)), "acierto_base": round(acc("base"), 4), "acierto_xi": round(acc("xi"), 4),
           "logloss_base": round(float(lb.mean()), 5), "logloss_xi": round(float(lx.mean()), 5),
           "minutos_antes_mediana": float(d["minutes_to_kick"].median())}
    if d["kick"].dt.normalize().nunique() > 1:
        out["mejora_logloss"] = paired_bootstrap(lb, lx, d["kick"].dt.normalize(), B=bootstrap)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", action="store_true")
    a = ap.parse_args(argv)
    now = pd.Timestamp.now(tz="UTC")
    if a.warm:
        g = load_group("clubes", verbose=False)
        print("onces históricos en caché:", len(history_boxes(g)))
        return
    snap, g = snapshot(now)
    path = P / SHADOW
    old = pd.read_csv(path, dtype={"event_id": str}) if path.exists() else pd.DataFrame()
    allrows = pd.concat([old, snap], ignore_index=True) if len(snap) else old
    if len(snap):
        P.mkdir(exist_ok=True); allrows.to_csv(path, index=False)
    board = {"status": "SHADOW_ONLY_PRODUCTION_UNCHANGED", "updated_utc": now.isoformat(),
             "regla": "Activar solo con n >= 200, IC95% de mejora_logloss > 0 y autorización explícita.",
             "nuevas_filas": int(len(snap)), "resultado": score(allrows, g) if len(allrows) else {"n": 0}}
    (P / BOARD).write_text(json.dumps(board, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(json.dumps(board, ensure_ascii=False))


if __name__ == "__main__":
    main()
