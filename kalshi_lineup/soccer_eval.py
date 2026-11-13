"""Backtest pre-registrado del once titular en fútbol (copas europeas y selecciones), 1X2 a 90 minutos.

  python -m kalshi_lineup.soccer_eval --group clubes|selecciones [--output reports/lineup]

Protocolo (fijado antes de ver resultados; ver LINEUP_RESEARCH.md):
  * walk-forward de kalshi_intl (entrena con todo lo anterior al 1 de julio, evalúa la temporada siguiente);
  * desarrollo: temporadas 2020/21–2023/24, se elige una variante; confirmación: ≥ 2024/25, una sola vez;
  * éxito: IC95% de mejora de logloss > 0, Brier (multiclase) no peor, ECE no peor en más de 0,005.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_nba.backtest import paired_bootstrap
from kalshi_intl.data import GROUPS, load_group
from kalshi_intl.model import CONFIGS, GROUP_FEATURES, IntlEngine, walk_forward

from .box import load_boxes
from .soccer_xi import SUMMARY_SOCCER, XI_FEATURES, XIRater, parse_soccer_summary

VARIANTS = {"A_continuidad": ["xi_cont"], "B_continuidad_y_ataque": XI_FEATURES}
CLUB_FEED = ["eng.1", "esp.1", "ita.1", "ger.1", "fra.1", "por.1", "ned.1"]
DEV_LAST = 2023          # temporada que EMPIEZA en 2023 (2023/24)
IDX = {"H": 0, "D": 1, "A": 2}


def _P(r, pre="p"):
    P = r[[f"{pre}_{c}" for c in "HDA"]].to_numpy(float)
    return P / P.sum(1, keepdims=True)


def ll3(r, pre="p"):
    y = r["res"].map(IDX).to_numpy()
    return -np.log(np.clip(_P(r, pre)[np.arange(len(r)), y], 1e-12, 1))


def brier3(r, pre="p"):
    y = np.eye(3)[r["res"].map(IDX).to_numpy()]
    return ((_P(r, pre) - y) ** 2).sum(1)


def ece3(r, pre="p", bins=10):
    P = _P(r, pre); conf = P.max(1); hit = (P.argmax(1) == r["res"].map(IDX).to_numpy()).astype(float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs(conf[idx == b].mean() - hit[idx == b].mean()) * (idx == b).sum() for b in range(bins) if (idx == b).any()) / len(r))


def metrics3(r, pre="p"):
    return {"n": int(len(r)), "accuracy": round(float((_P(r, pre).argmax(1) == r["res"].map(IDX).to_numpy()).mean()), 4),
            "logloss": round(float(ll3(r, pre).mean()), 5), "brier": round(float(brier3(r, pre).mean()), 5), "ece": round(ece3(r, pre), 4)}


def compare3(r, pre):
    return {"base": metrics3(r, "p"), "variante": metrics3(r, pre),
            "mejora_logloss": paired_bootstrap(ll3(r, "p"), ll3(r, pre), r["date"]),
            "mejora_brier": paired_bootstrap(brier3(r, "p"), brier3(r, pre), r["date"])}


def run(f: pd.DataFrame, base: list, first_year=2020, variants=VARIANTS) -> pd.DataFrame:
    keep = ["event_id", "date", "temporada", "res", "code", "xi_ok"]
    r = walk_forward(f, first_year=first_year, features=base)[keep + ["p_H", "p_D", "p_A"]]
    for name, extra in variants.items():
        v = walk_forward(f, first_year=first_year, features=base + extra)[["event_id", "p_H", "p_D", "p_A"]]
        r = r.merge(v.rename(columns={c: f"{name}_{c[2]}" for c in ("p_H", "p_D", "p_A")}), on="event_id")
    r["start_year"] = r["temporada"].str[:4].astype(int)
    return r


def decide(r: pd.DataFrame, variants=VARIANTS, dev_last=DEV_LAST) -> dict:
    dev, hold = r[r["start_year"] <= dev_last], r[r["start_year"] > dev_last]
    dev_res = {n: compare3(dev, n) for n in variants}
    chosen = max(dev_res, key=lambda n: dev_res[n]["mejora_logloss"]["mejora"])
    out = {"protocolo": f"dev<={dev_last}/{(dev_last + 1) % 100:02d} elige; confirmación posterior una vez", "desarrollo": dev_res,
           "variante_elegida": chosen}
    if len(hold):
        h = compare3(hold, chosen)
        ok = h["mejora_logloss"]["ic95_inf"] > 0 and h["mejora_brier"]["mejora"] >= 0 and h["variante"]["ece"] <= h["base"]["ece"] + 0.005
        out.update(confirmacion=h, veredicto="SUPERA_CRITERIO_SOLO_SOMBRA" if ok else "NO_SUPERA_CRITERIO")
    else:
        out["veredicto"] = "SIN_TEMPORADAS_DE_CONFIRMACION"
    out["por_temporada"] = [{"temporada": t, "n": len(g), "logloss_base": round(float(ll3(g).mean()), 5),
                             **{f"ll_{n}": round(float(ll3(g, n).mean()), 5) for n in variants}} for t, g in r.groupby("temporada")]
    cov = r["xi_ok"].mean() if len(r) else 0
    out["partidos_oos_con_once_valido_pct"] = round(100 * float(cov), 1)
    sub = r[r["xi_ok"] == 1]
    if len(sub) and len(hold):
        sh = sub[sub["start_year"] > dev_last]
        if len(sh):
            out["confirmacion_solo_con_once"] = compare3(sh, chosen)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=list(GROUPS), required=True)
    ap.add_argument("--cache", default="data/intl")
    ap.add_argument("--box-cache", default="data/intl_box")
    ap.add_argument("--since", default="2018-07-01")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", default="reports/lineup")
    a = ap.parse_args(argv)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    g = load_group(a.group, cache_dir=a.cache)
    fin = g[g["final"] & (g["date"] >= pd.Timestamp(a.since))]
    if a.group == "clubes":
        fin = fin[fin["target"] | fin["code"].isin(CLUB_FEED)]
    boxes = {}
    for code, part in fin.groupby("code"):
        boxes.update(load_boxes(part["event_id"], cache_dir=f"{a.box_cache}/{code}", workers=a.workers,
                                url=SUMMARY_SOCCER.format(code=code), parser=parse_soccer_summary))
    cov = {"partidos_solicitados": int(len(fin)), "con_once_valido": int(len(boxes)),
           "porcentaje": round(100 * len(boxes) / max(len(fin), 1), 2),
           "por_competicion": {c: round(float(p["event_id"].astype(str).isin(boxes).mean()), 3) for c, p in fin.groupby("code")}}
    print("Cobertura:", json.dumps(cov))
    if cov["porcentaje"] < 60:
        raise SystemExit("Cobertura de onces < 60%: no se evalúa (revisar esquema/red de ESPN).")
    f = IntlEngine(CONFIGS[a.group]).process(g)
    f["event_id"] = f["event_id"].astype(str)
    xi = XIRater().process(g, boxes)
    f = f.merge(xi, on="event_id", how="left")
    for c in XI_FEATURES + ["xi_ok"]:
        f[c] = f[c].fillna(0.0)
    r = run(f, GROUP_FEATURES[a.group])
    res = decide(r)
    res["cobertura"] = cov
    L = [f"FÚTBOL {a.group.upper()} — ONCE TITULAR (investigación, producción sin cambios)", "=" * 72,
         "Cobertura: " + json.dumps(cov, ensure_ascii=False), f"Partidos OOS con once válido: {res['partidos_oos_con_once_valido_pct']}%",
         "Protocolo: " + res["protocolo"], "", "--- DESARROLLO ---"]
    for n, x in res["desarrollo"].items():
        L.append(f"{n:24s} base {json.dumps(x['base'])}\n{'':24s} var  {json.dumps(x['variante'])}\n{'':24s} mejora ll {x['mejora_logloss']}")
    L.append(f"\nVariante elegida: {res['variante_elegida']}")
    if "confirmacion" in res:
        c = res["confirmacion"]
        L += ["\n--- CONFIRMACIÓN (una sola vez) ---", "base:     " + json.dumps(c["base"]), "variante: " + json.dumps(c["variante"]),
              "mejora logloss: " + json.dumps(c["mejora_logloss"]), "mejora brier:   " + json.dumps(c["mejora_brier"])]
    if "confirmacion_solo_con_once" in res:
        L.append("\nConfirmación, solo partidos con once válido (descriptivo): " + json.dumps(res["confirmacion_solo_con_once"]["mejora_logloss"]))
    L += [f"\nVEREDICTO: {res['veredicto']}", "\n--- POR TEMPORADA ---", pd.DataFrame(res["por_temporada"]).to_string(index=False)]
    txt = "\n".join(L)
    (out / f"soccer_xi_{a.group}_report.txt").write_text(txt, encoding="utf8")
    (out / f"soccer_xi_{a.group}_result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf8")
    r.to_csv(out / f"soccer_xi_{a.group}_oos.csv", index=False)
    print(txt)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf8") as fh:
            fh.write("```\n" + txt + "\n```\n")


if __name__ == "__main__":
    main()
