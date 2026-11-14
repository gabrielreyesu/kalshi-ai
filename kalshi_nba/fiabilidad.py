"""Fiabilidad del pick final: ¿se cumplen las probabilidades? (tabla para la página de progreso)

Usa la validación fuera de muestra del pick combinado (pesos entrenados solo con fechas anteriores) sobre partidos
reales con precio de Kalshi, casas y modelo. Agrupa por la probabilidad del PICK (el lado favorito) y compara con el
acierto real. Guarda kalshi_nba/fiabilidad.json (se versiona: es un resumen agregado, sin datos de terceros).
  python -m kalshi_nba.fiabilidad
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .final_pick import walk_forward

OUT = Path(__file__).with_name("fiabilidad.json")
BINS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]


def table(p_home: pd.Series, y: pd.Series) -> list:
    p = np.where(p_home >= 0.5, p_home, 1 - p_home)                    # probabilidad del pick
    hit = np.where(p_home >= 0.5, y == 1, y == 0)
    rows = []
    for lo, hi in BINS:
        m = (p >= lo) & (p < hi)
        if m.sum() >= 15:
            rows.append({"tramo": f"{int(lo * 100)}–{min(int(hi * 100), 100)}%", "n": int(m.sum()),
                         "prob_media": round(float(p[m].mean()), 3), "acierto": round(float(hit[m].mean()), 3)})
    return rows


def main():
    out = {"nota": "Validación fuera de muestra del pick combinado (Kalshi + casas + modelo).", "deportes": {}}
    for lg in ("nba", "nfl"):
        f = Path(f"reports/kalshi_bench/{lg}_kalshi_bench_rows.csv")
        if not f.exists():
            continue
        r = pd.read_csv(f)
        d = pd.DataFrame({"date": pd.to_datetime(r["date"]), "y": r["y"], "k": r["k60_mid"], "c": r["book_p_home"], "m": r["p_base"]})
        wf = walk_forward(d).dropna(subset=["p_final"])
        out["deportes"][lg.upper()] = {"n": int(len(wf)), "desde": str(wf["date"].min())[:10], "hasta": str(wf["date"].max())[:10],
                                       "acierto_total": round(float(((wf["p_final"] >= .5) == (wf["y"] == 1)).mean()), 3),
                                       "tramos": table(wf["p_final"], wf["y"])}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
