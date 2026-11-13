"""Fútbol: tendencia del árbitro principal frente a las casas (goles y 1X2). Protocolo en REFEREES_RESEARCH.md.

Por árbitro, con partidos ANTERIORES: media encogida (k = 25) de (más de 2,5 real − prob. casas) y de
(gana local real − prob. casas local). Prueba: logit(casas) + tendencia, walk-forward por temporada (prueba ≥ 2021/22;
confirmación ≥ 2024/25). Éxito: IC95% > 0 en confirmación.
  python -m kalshi_lineup.referees_soccer
"""
from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from kalshi_nba.backtest import logloss, paired_bootstrap
from kalshi_nba.data import _json
from kalshi_intl.data import load_group
from kalshi_intl.odds import attach

CODES = ["eng.1", "esp.1", "ita.1", "ger.1", "fra.1", "uefa.champions", "uefa.europa", "uefa.europa.conf"]
URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/summary"
K = 25.0


def referee(code, eid, cache: Path):
    p = cache / f"{eid}.json"
    if p.exists():
        return json.loads(p.read_text())
    try:
        j = _json(URL.format(code=code), {"event": eid}) or {}
    except RuntimeError:
        return None
    o = [x for x in (j.get("gameInfo") or {}).get("officials", []) if x.get("displayName")]
    name = sorted(o, key=lambda x: x.get("order", 9))[0]["displayName"] if o else None
    if name:
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(name))
    return name


def lg(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def main():
    g = load_group("clubes", verbose=False)
    g = g[g["final"] & g["code"].isin(CODES) & (g["date"] >= pd.Timestamp("2019-07-01"))
          & g["status"].isin(["STATUS_FULL_TIME", "STATUS_FINAL"])].copy()
    g["event_id"] = g["event_id"].astype(str)
    cache = Path("data/intl/refs")
    with ThreadPoolExecutor(8) as ex:
        refs = list(ex.map(lambda t: referee(t[0], t[1], cache), zip(g["code"], g["event_id"])))
    g["ref"] = refs
    g = attach(g[g["ref"].notna()].reset_index(drop=True))
    g = g[g["mkt_over"].notna() & g["mkt_H"].notna()].copy()
    g["over"] = ((g["hs"] + g["as_"]) > 2.5).astype(int); g["hw"] = (g["hs"] > g["as_"]).astype(int)
    g["season"] = np.where(g["date"].dt.month >= 7, g["date"].dt.year, g["date"].dt.year - 1)
    st = defaultdict(lambda: [0.0, 0.0, 0])
    fo, fh = [], []
    for date, day in g.sort_values("date").groupby("date", sort=True):
        for i, x in day.iterrows():
            s = st[x["ref"]]; fo.append((i, s[0] / (s[2] + K))); fh.append((i, s[1] / (s[2] + K)))
        for i, x in day.iterrows():
            s = st[x["ref"]]; s[0] += x["over"] - x["mkt_over"]; s[1] += x["hw"] - x["mkt_H"]; s[2] += 1
    g["ref_over"] = pd.Series(dict(fo)); g["ref_home"] = pd.Series(dict(fh))
    res = {"partidos": int(len(g)), "arbitros": int(g["ref"].nunique())}
    for name, y, mk, feat in (("goles_mas_2_5", "over", "mkt_over", "ref_over"), ("gana_local", "hw", "mkt_H", "ref_home")):
        p0, p1 = [], []
        for s in sorted(g["season"].unique()):
            tr, te = g[g["season"] < s], g[g["season"] == s]
            if s < 2021 or len(tr) < 1500:
                continue
            X0 = lambda x: lg(x[mk]).reshape(-1, 1); X1 = lambda x: np.column_stack([lg(x[mk]), x[feat]])
            p0.append(te.assign(p=LogisticRegression().fit(X0(tr), tr[y]).predict_proba(X0(te))[:, 1]))
            p1.append(te.assign(p=LogisticRegression().fit(X1(tr), tr[y]).predict_proba(X1(te))[:, 1]))
        a, b = pd.concat(p0), pd.concat(p1)
        dv, hd = a["season"] < 2024, a["season"] >= 2024
        res[name] = {"desarrollo": paired_bootstrap(logloss(a[dv]["p"], a[dv][y]), logloss(b[dv]["p"], b[dv][y]), a[dv]["date"]),
                     "confirmacion": {"n": int(hd.sum()), **paired_bootstrap(logloss(a[hd]["p"], a[hd][y]), logloss(b[hd]["p"], b[hd][y]), a[hd]["date"])}}
        res[name]["veredicto"] = "SUPERA_CRITERIO" if res[name]["confirmacion"]["ic95_inf"] > 0 else "NO_SUPERA_CRITERIO"
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
