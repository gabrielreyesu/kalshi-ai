"""Resultados finales reales (ESPN) para puntuar las fuentes externas en TODOS los deportes registrados.

Antes de V9 el flujo de ingesta solo leía resultados de un archivo que otro workflow generaba y que aquí no existe,
de modo que ninguna fuente llegaba nunca a tener casos resueltos. Este módulo cierra ese ciclo.
Solo lectura; si ESPN falla se devuelve lo que haya y se informa el aviso.
"""
from __future__ import annotations

import pandas as pd

REGULATION = {"STATUS_FULL_TIME", "STATUS_FINAL"}          # fútbol: sin prórroga ni penales (mercado a 90 min)
COLUMNS = ["event_id", "sport", "result"]


def outcomes_from_games(games: pd.DataFrame, sport: str) -> pd.DataFrame:
    """games: event_id, status, hs, as_ (+ 'final' en fútbol). Devuelve H/D/A (fútbol) o H/A (NBA/NFL; empates fuera)."""
    if games is None or games.empty:
        return pd.DataFrame(columns=COLUMNS)
    g = games.copy()
    if sport in ("intl", "club"):
        g = g[g["status"].isin(REGULATION) & g["hs"].notna() & g["as_"].notna()]
        res = g.apply(lambda r: "H" if r.hs > r.as_ else ("D" if r.hs == r.as_ else "A"), axis=1) if len(g) else pd.Series(dtype=str)
    else:
        g = g[(g["status"] == "STATUS_FINAL") & g["hs"].notna() & g["as_"].notna() & (g["hs"] != g["as_"])]
        res = g.apply(lambda r: "H" if r.hs > r.as_ else "A", axis=1) if len(g) else pd.Series(dtype=str)
    out = pd.DataFrame({"event_id": g["event_id"].astype(str), "sport": sport, "result": res})
    return out.drop_duplicates(["sport", "event_id"])[COLUMNS]


def live_outcomes(registry: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Consulta ESPN solo para los deportes presentes en el registro. Devuelve (resultados, avisos)."""
    parts, warn = [], []
    sports = set(registry["sport"].unique()) if len(registry) else set()
    ids = set(registry["event_id"].astype(str)) if len(registry) else set()
    for sport in sorted(sports):
        try:
            if sport in ("nba", "nfl"):
                from kalshi_nba.data import current_season, load_league
                g = load_league(sport, first_season=current_season(sport), with_odds=False, verbose=False)
            elif sport in ("intl", "club"):
                from kalshi_intl.data import load_group
                g = load_group("selecciones" if sport == "intl" else "clubes", first_year=pd.Timestamp.now().year - 1, verbose=False)
            else:
                continue
            o = outcomes_from_games(g, sport)
            parts.append(o[o["event_id"].isin(ids)])
        except Exception as exc:  # noqa: BLE001
            warn.append(f"Outcome lookup {sport}: {type(exc).__name__}")
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS)), warn
