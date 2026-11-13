"""NBA: tendencia de los árbitros (investigación). Protocolo pre-registrado en REFEREES_RESEARCH.md.

Para cada árbitro, con partidos ANTERIORES: media (encogida hacia 0 con k = 40 partidos) del residuo de totales
(total real − línea de cierre) y del residuo de margen (margen local − (−spread)). Variable del partido = media del trío.
  python -m kalshi_lineup.referees
"""
from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
SUMMARIES = {"nba": SUMMARY, "nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"}
K = 40.0


def officials(event_id: str, cache: Path, url: str = SUMMARY) -> list:
    p = cache / f"{event_id}.json"
    if p.exists():
        return json.loads(p.read_text())
    from kalshi_nba.data import _json
    try:
        j = _json(url, {"event": event_id}) or {}
    except RuntimeError:
        return []
    names = [o.get("displayName") for o in (j.get("gameInfo") or {}).get("officials", []) if o.get("displayName")]
    if names:
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(names))
    return names


def features(d: pd.DataFrame) -> pd.DataFrame:
    """d ordenado por fecha con refs, res_total, res_margin. Variables con estado anterior a cada FECHA."""
    st = defaultdict(lambda: [0.0, 0.0, 0])           # árbitro -> [suma res_total, suma res_margin, n]
    out = []
    for date, day in d.sort_values("date").groupby("date", sort=True):
        for x in day.itertuples():
            ft = [st[r][0] / (st[r][2] + K) for r in x.refs] or [0.0]
            fm = [st[r][1] / (st[r][2] + K) for r in x.refs] or [0.0]
            out.append({"event_id": x.event_id, "ref_total": float(np.mean(ft)), "ref_margin": float(np.mean(fm)),
                        "ref_n": int(min((st[r][2] for r in x.refs), default=0))})
        for x in day.itertuples():
            for r in x.refs:
                st[r][0] += x.res_total; st[r][1] += x.res_margin; st[r][2] += 1
    return pd.DataFrame(out)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--league", choices=["nba", "nfl"], default="nba")
    a = ap.parse_args(argv)
    lgname = a.league
    hold_from = 2025 if lgname == "nba" else 2024
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from kalshi_nba.backtest import logloss, paired_bootstrap
    from kalshi_nba.data import load_league
    g = load_league(lgname, first_season=2016 if lgname == "nba" else 2015)
    g = g[(g["status"] == "STATUS_FINAL") & g["line_total"].notna() & g["line_spread"].notna()].copy()
    g["event_id"] = g["event_id"].astype(str)
    cache = Path(f"data/{lgname}/refs")
    with ThreadPoolExecutor(8) as ex:
        refs = dict(zip(g["event_id"], ex.map(lambda e: officials(e, cache, SUMMARIES[lgname]), g["event_id"])))
    g["refs"] = g["event_id"].map(refs)
    g = g[g["refs"].map(len) >= 2].copy()
    g["res_total"] = (g["hs"] + g["as_"]) - g["line_total"]
    g["res_margin"] = (g["hs"] - g["as_"]) + g["line_spread"]
    d = g.merge(features(g), on="event_id")
    out = {"partidos_con_arbitros": int(len(d))}
    pt, pw0, pw1 = [], [], []
    lg = lambda q: np.log(np.clip(q, 1e-3, 1 - 1e-3) / (1 - np.clip(q, 1e-3, 1 - 1e-3)))
    for s in sorted(d["season_year"].unique()):
        tr, te = d[d["season_year"] < s], d[d["season_year"] == s]
        if s < 2019 or len(tr) < (2000 if lgname == "nba" else 800):
            continue
        m = LinearRegression().fit(tr[["ref_total"]], tr["res_total"])
        pt.append(te.assign(adj=m.predict(te[["ref_total"]])))
        w = te[te["line_p_home"].notna() & (te["hs"] != te["as_"])]; wt = tr[tr["line_p_home"].notna() & (tr["hs"] != tr["as_"])]
        yt = (wt["hs"] > wt["as_"]).astype(int)
        X0 = lambda x: lg(x["line_p_home"].to_numpy(float)).reshape(-1, 1)
        X1 = lambda x: np.column_stack([lg(x["line_p_home"].to_numpy(float)), x["ref_margin"].to_numpy(float)])
        pw0.append(w.assign(p=LogisticRegression().fit(X0(wt), yt).predict_proba(X0(w))[:, 1]))
        pw1.append(w.assign(p=LogisticRegression().fit(X1(wt), yt).predict_proba(X1(w))[:, 1]))
    p = pd.concat(pt)
    hold = p[p["season_year"] >= hold_from]; dev = p[p["season_year"] < hold_from]
    sq = lambda x: (x["res_total"] ** 2, (x["res_total"] - x["adj"]) ** 2)
    out["totales_desarrollo"] = paired_bootstrap(*sq(dev), dev["date"])
    out["totales_confirmacion"] = {"n": int(len(hold)), **paired_bootstrap(*sq(hold), hold["date"])}
    a, b = pd.concat(pw0), pd.concat(pw1)
    ha, hb = a[a["season_year"] >= hold_from], b[b["season_year"] >= hold_from]
    ya = (ha["hs"] > ha["as_"]).astype(int)
    out["ganador_confirmacion"] = {"n": int(len(ha)), **paired_bootstrap(logloss(ha["p"], ya), logloss(hb["p"], ya), ha["date"])}
    out["veredicto_totales"] = "SUPERA_CRITERIO" if out["totales_confirmacion"]["ic95_inf"] > 0 else "NO_SUPERA_CRITERIO"
    out["veredicto_ganador"] = "SUPERA_CRITERIO" if out["ganador_confirmacion"]["ic95_inf"] > 0 else "NO_SUPERA_CRITERIO"
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
