"""python -m kalshi_lineup.stakes_eval [--output reports/stakes]

Backtest pre-registrado de "qué se juega cada equipo" en ligas (STAKES_RESEARCH.md). Motor de clubes de kalshi_intl,
evaluado en partidos de liga (1X2 a 90 min). Desarrollo 2020/21–2023/24 elige variante; confirmación ≥ 2024/25 una vez.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from kalshi_intl.data import load_group
from kalshi_intl.model import CONFIGS, GROUP_FEATURES, IntlEngine, walk_forward

from .soccer_eval import compare3, decide
from .stakes import GROUP_EDITIONS, STAKE_FEATURES, ZONES, compute, compute_groups

VARIANTS = {"A_diferencia": ["stake_diff"], "B_por_equipo": ["nostake_h", "nostake_a"]}


def run(f: pd.DataFrame, base: list, first_year=2020, eval_col="liga") -> pd.DataFrame:
    keep = ["event_id", "date", "temporada", "res", "code", "nostake_h", "nostake_a", "games_left_min"]
    r = walk_forward(f, first_year=first_year, features=base, eval_col=eval_col)[keep + ["p_H", "p_D", "p_A"]]
    for name, extra in VARIANTS.items():
        v = walk_forward(f, first_year=first_year, features=base + extra, eval_col=eval_col)[["event_id", "p_H", "p_D", "p_A"]]
        r = r.merge(v.rename(columns={c: f"{name}_{c[2]}" for c in ("p_H", "p_D", "p_A")}), on="event_id")
    r["start_year"] = r["temporada"].str[:4].astype(int)
    r["xi_ok"] = 1
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/intl")
    ap.add_argument("--output", default="reports/stakes")
    ap.add_argument("--group", choices=["clubes", "selecciones"], default="clubes")
    a = ap.parse_args(argv)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    g = load_group(a.group, cache_dir=a.cache)
    st = compute(g) if a.group == "clubes" else compute_groups(g)
    f = IntlEngine(CONFIGS[a.group]).process(g)
    f["event_id"] = f["event_id"].astype(str)
    f = f.merge(st, on="event_id", how="left")
    for c in STAKE_FEATURES + ["games_left_min"]:
        f[c] = f[c].fillna(0)
    f["liga"] = f["code"].isin(ZONES) if a.group == "clubes" else f["code"].isin(GROUP_EDITIONS)
    r = run(f, GROUP_FEATURES[a.group])
    res = decide(r, variants=VARIANTS)
    ns = r[(r["nostake_h"] == 1) | (r["nostake_a"] == 1)]
    res["partidos_con_algun_equipo_sin_nada_en_juego"] = int(len(ns))
    hold = ns[ns["start_year"] > 2023]
    if len(hold) > 30:
        res["confirmacion_solo_esos_partidos"] = compare3(hold, res["variante_elegida"])
    L = [("LIGAS" if a.group == "clubes" else "NATIONS LEAGUE + ELIMINATORIAS UEFA") + " — ¿QUÉ SE JUEGA CADA EQUIPO? (investigación)", "=" * 72,
         f"Partidos evaluados: {len(r)} | con algún equipo sin nada en juego: {len(ns)}",
         "Protocolo: " + res["protocolo"], "", "--- DESARROLLO ---"]
    for n, x in res["desarrollo"].items():
        L.append(f"{n:16s} base ll={x['base']['logloss']:.5f} var ll={x['variante']['logloss']:.5f} mejora {x['mejora_logloss']}")
    L.append(f"\nVariante elegida: {res['variante_elegida']}")
    if "confirmacion" in res:
        c = res["confirmacion"]
        L += ["\n--- CONFIRMACIÓN (una sola vez) ---", "base:     " + json.dumps(c["base"]), "variante: " + json.dumps(c["variante"]),
              "mejora logloss: " + json.dumps(c["mejora_logloss"]), "mejora brier:   " + json.dumps(c["mejora_brier"])]
    if "confirmacion_solo_esos_partidos" in res:
        L.append("\nSolo partidos con algún equipo sin nada en juego (descriptivo): "
                 + json.dumps(res["confirmacion_solo_esos_partidos"]["mejora_logloss"]) + " n=" + str(res["confirmacion_solo_esos_partidos"]["base"]["n"]))
    L += [f"\nVEREDICTO: {res['veredicto']}", "\n--- POR TEMPORADA ---", pd.DataFrame(res["por_temporada"]).to_string(index=False)]
    txt = "\n".join(L)
    (out / f"stakes_{a.group}_report.txt").write_text(txt, encoding="utf8")
    (out / f"stakes_{a.group}_result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf8")
    r.to_csv(out / f"stakes_{a.group}_oos.csv", index=False)
    print(txt)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf8") as fh:
            fh.write("```\n" + txt + "\n```\n")


if __name__ == "__main__":
    main()
