"""Aviso informativo para la página: partidos próximos donde un equipo ya no se juega nada en la tabla.

Lee la caché de resultados de ESPN (data/intl) que ya usa soccer-daily. Escribe predictions/stakes_upcoming.csv.
NO cambia ninguna probabilidad: en el backtest (STAKES_RESEARCH.md) la variable no superó el criterio.
  python -m kalshi_lineup.stakes_now
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from kalshi_intl.data import load_group

from .stakes import compute, compute_groups


def upcoming_flags(now=None) -> pd.DataFrame:
    now = now or pd.Timestamp.now(tz="UTC")
    out = []
    for grp, fn in (("clubes", compute), ("selecciones", compute_groups)):
        g = load_group(grp, verbose=False)
        st = fn(g)
        if st.empty:
            continue
        up = g[~g["final"] & (g["kick"] > now)][["event_id", "home", "away"]].assign(event_id=lambda x: x["event_id"].astype(str))
        out.append(up.merge(st, on="event_id"))
    d = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["event_id", "nostake_h", "nostake_a"])
    return d[(d["nostake_h"] == 1) | (d["nostake_a"] == 1)]


def main():
    d = upcoming_flags()
    Path("predictions").mkdir(exist_ok=True)
    d[["event_id", "home", "away", "nostake_h", "nostake_a"]].to_csv("predictions/stakes_upcoming.csv", index=False)
    print(f"partidos próximos con algún equipo sin nada en juego: {len(d)}")


if __name__ == "__main__":
    main()
