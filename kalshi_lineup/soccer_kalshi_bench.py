"""python -m kalshi_lineup.soccer_kalshi_bench

Fútbol: une precios de Kalshi antes del inicio (data/kalshi/soccer_pregame.csv), cuotas de las casas (ESPN) y el modelo
fuera de muestra (kalshi_intl walk-forward), valida el pick final combinado (local/empate/visitante) y guarda los pesos
en kalshi_nba/final_weights.json bajo "futbol". Protocolo: KALSHI_BENCHMARK.md (sección fútbol).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from kalshi_nba.backtest import paired_bootstrap
from kalshi_nba.final_pick import WEIGHTS, fit_sets3, walk_forward3
from kalshi_intl.data import load_group
from kalshi_intl.model import CONFIGS, GROUP_FEATURES, IntlEngine, walk_forward
from kalshi_intl.odds import attach

from .soccer_eval import metrics3

BIG5 = ["eng.1", "esp.1", "ita.1", "ger.1", "fra.1"]


def oos(grp: str) -> pd.DataFrame:
    g = load_group(grp, verbose=False)
    f = IntlEngine(CONFIGS[grp]).process(g)
    f["eval"] = f["target"] | f["code"].isin(BIG5) if grp == "clubes" else f["target"]
    r = walk_forward(f, first_year=2020, features=GROUP_FEATURES[grp], eval_col="eval")
    return r[["event_id", "date", "code", "res", "p_H", "p_D", "p_A"]].assign(event_id=lambda x: x["event_id"].astype(str))


def ll(d, pre):
    P = d[[f"{pre}_{c}" for c in "HDA"]].to_numpy(float); P = P / P.sum(1, keepdims=True)
    return -np.log(np.clip(P[np.arange(len(d)), d["y"].to_numpy(int)], 1e-12, 1))


def main():
    k = pd.read_csv("data/kalshi/soccer_pregame.csv", dtype={"event_id": str})
    o = pd.concat([oos("clubes"), oos("selecciones")], ignore_index=True)
    d = k.merge(o, on="event_id")
    d = attach(d)                                                  # mkt_H/D/A (casas, sin margen)
    d["y"] = d["res"].map({"H": 0, "D": 1, "A": 2})
    d = d.rename(columns={**{f"k60_{c}": f"k_{c}" for c in "HDA"}, **{f"mkt_{c}": f"c_{c}" for c in "HDA"},
                          **{f"p_{c}": f"m_{c}" for c in "HDA"}})
    d["date"] = pd.to_datetime(d["date"])
    wf = walk_forward3(d).dropna(subset=["f_H"])
    cur = wf.copy()                                               # pick actual: casas si existen; si no, modelo
    for c in "HDA":
        cur[f"a_{c}"] = cur[f"c_{c}"].where(cur["c_H"].notna(), cur[f"m_{c}"])
        cur[f"q_{c}"] = cur[f"k_{c}"].where(cur["k_H"].notna(), cur[f"a_{c}"])
    res = {"n": int(len(wf)), "desde": str(wf["date"].min())[:10], "hasta": str(wf["date"].max())[:10],
           "con_kalshi": int(wf["k_H"].notna().sum()), "con_casas": int(wf["c_H"].notna().sum()),
           "pick_actual": metrics3(cur.rename(columns={f"a_{c}": f"x_{c}" for c in "HDA"}).assign(res=cur["res"]), "x"),
           "kalshi_o_actual": metrics3(cur.rename(columns={f"q_{c}": f"x_{c}" for c in "HDA"}).assign(res=cur["res"]), "x"),
           "modelo": metrics3(wf, "m"), "combinado": metrics3(wf, "f"),
           "mejora_vs_pick_actual": paired_bootstrap(ll(cur, "a"), ll(wf, "f"), wf["date"]),
           "mejora_vs_kalshi": paired_bootstrap(ll(cur, "q"), ll(wf, "f"), wf["date"])}
    by = {}
    for name, codes in (("ligas", BIG5), ("copas", ["uefa.champions", "uefa.europa", "uefa.europa.conf"])):
        s = wf[wf["code"].isin(codes)]; cs = cur.loc[s.index]
        if len(s) > 50:
            by[name] = {"n": int(len(s)), "actual": round(float(ll(cs, "a").mean()), 4), "combinado": round(float(ll(s, "f").mean()), 4),
                        "mejora": paired_bootstrap(ll(cs, "a"), ll(s, "f"), s["date"])}
    s = wf[~wf["code"].isin(BIG5 + ["uefa.champions", "uefa.europa", "uefa.europa.conf"])]; cs = cur.loc[s.index]
    if len(s) > 50:
        by["selecciones"] = {"n": int(len(s)), "actual": round(float(ll(cs, "a").mean()), 4), "combinado": round(float(ll(s, "f").mean()), 4),
                             "mejora": paired_bootstrap(ll(cs, "a"), ll(s, "f"), s["date"])}
    res["por_grupo"] = by
    pesos = fit_sets3(d)
    data = json.loads(WEIGHTS.read_text()) if WEIGHTS.exists() else {}
    data["futbol"] = {"pesos": pesos, "entrenado_con": int(len(d)), "validacion": res,
                      "nota": "Agregación log-lineal: [b_local, b_visitante, w_fuente...] según el subconjunto."}
    WEIGHTS.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"pesos": pesos, **res}, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
