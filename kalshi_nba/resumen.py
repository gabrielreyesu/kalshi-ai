"""Genera PREDICCIONES.md: una tabla legible de los próximos partidos (se ve directo en GitHub).
Solo lee los CSV de predictions/; no usa internet.   Uso:  python -m kalshi_nba.resumen
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TZ = "America/Chicago"
P = Path("predictions")
pct = lambda x: "—" if x is None or pd.isna(x) else f"{100 * float(x):.0f}%"


def _when(s):
    return pd.to_datetime(s, utc=True).dt.tz_convert(TZ)


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No hay partidos próximos registrados._\n"
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.to_numpy()) + "\n"


def liga_us(lg: str, now) -> str:
    reg = P / f"{lg}_registry.csv"
    if not reg.exists():
        return "_Aún no hay registro._\n"
    r = pd.read_csv(reg, dtype={"event_id": str})
    r["t"] = _when(r["kickoff_utc"])
    r = r[r["t"] > now].sort_values("t")
    k = c = pd.Series(dtype=float)
    snap = P / f"{lg}_kalshi_snapshots.csv"
    if snap.exists():
        s = pd.read_csv(snap, dtype={"event_id": str})
        s = s[(s["kind"] == "winner") & (s["side"] == "home")].sort_values("snapshot_utc").drop_duplicates("event_id", keep="last")
        k = pd.to_numeric(s.set_index("event_id")["mid"], errors="coerce")
        if "p_book" in s:
            c = pd.to_numeric(s.set_index("event_id")["p_book"], errors="coerce")
    pm = pd.Series(dtype=float)
    pmf = P / f"{lg}_polymarket_snapshots.csv"
    if pmf.exists():
        q = pd.read_csv(pmf, dtype={"event_id": str}).sort_values("snapshot_utc").drop_duplicates("event_id", keep="last")
        pm = pd.to_numeric(q.set_index("event_id")["pm_home"], errors="coerce")
    from .injuries import out_counts
    bajas = out_counts(lg, str(P))
    rows = []
    for x in r.itertuples():
        fav_home = x.p_home >= 0.5
        pk, pc, pp = k.get(x.event_id, np.nan), c.get(x.event_id, np.nan), pm.get(x.event_id, np.nan)
        flip = lambda v: v if fav_home or pd.isna(v) else 1 - v
        rows.append({"Fecha (hora Chicago)": x.t.strftime("%a %d %b %H:%M"), "Partido": f"{x.away} en {x.home}",
                     "Favorito del modelo": x.home if fav_home else x.away,
                     "Modelo": pct(flip(x.p_home)), "Kalshi": pct(flip(pk)), "Polymarket": pct(flip(pp)), "Casas": pct(flip(pc)),
                     "Diferencia esperada": f"{abs(x.mu_margin):.1f} pts", "Total esperado": f"{x.mu_total:.0f} pts",
                     "Bajas: local / visita": f"{bajas.get(str(getattr(x, 'home_id', '')), 0)} / {bajas.get(str(getattr(x, 'away_id', '')), 0)}" if bajas else "—"})
    return _table(pd.DataFrame(rows))


def _bajas_fut(x, lat):
    from kalshi_intl.lesiones import count_out
    a, b = count_out(lat, x.home, x.kickoff_utc), count_out(lat, x.away, x.kickoff_utc)
    return "—" if a is None and b is None else f"{'—' if a is None else a} / {'—' if b is None else b}"


def _lat():
    from kalshi_intl.lesiones import load_latest
    return load_latest(P / "futbol_lesiones.csv")


def futbol(now) -> str:
    reg = P / "registry.csv"
    if not reg.exists():
        return "_Aún no hay registro._\n"
    r = pd.read_csv(reg)
    r["t"] = _when(r["kickoff_utc"])
    r = r[r["t"] > now].sort_values("t")
    num = lambda x, c: pd.to_numeric(pd.Series([getattr(x, c, np.nan)]), errors="coerce").iloc[0]
    lat = _lat()
    rows = []
    for x in r.itertuples():
        kal = []
        for s in "HDA":
            b, a = num(x, f"k_bid_{s}"), num(x, f"k_ask_{s}")
            kal.append((b + a) / 2 if not (pd.isna(b) or pd.isna(a)) else np.nan)
        rows.append({"Fecha (hora Chicago)": x.t.strftime("%a %d %b %H:%M"), "Liga": x.competition, "Partido": f"{x.home} vs {x.away}",
                     "Modelo: local / empate / visita": f"{pct(x.p_H)} / {pct(x.p_D)} / {pct(x.p_A)}",
                     "Kalshi: local / empate / visita": " / ".join(pct(v) for v in kal),
                     "Goles: +1.5 / +2.5 / +3.5": " / ".join(pct(getattr(x, f"p_over_{k}_cal", np.nan)) for k in ("1_5", "2_5", "3_5")),
                     "Ambos anotan": pct(getattr(x, "p_btts_cal", np.nan)),
                     "Gana por 2+: local / visita": f"{pct(getattr(x, 'p_home_m1_5_cal', np.nan))} / {pct(getattr(x, 'p_away_m1_5_cal', np.nan))}",
                     "Bajas: local / visita": _bajas_fut(x, lat)})
    return _table(pd.DataFrame(rows))


def internacional(now, grupo: str) -> str:
    reg = P / "intl_registry.csv"
    if not reg.exists():
        return "_Aún no hay registro._\n"
    r = pd.read_csv(reg, dtype={"event_id": str})
    r = r[r["grupo"] == grupo].copy()
    snap = P / "intl_kalshi_snapshots.csv"
    if snap.exists():                      # precios más recientes de Kalshi y de las casas
        s = pd.read_csv(snap, dtype={"event_id": str}).sort_values("snapshot_utc").drop_duplicates("event_id", keep="last").set_index("event_id")
        for c in [x for x in s.columns if x.startswith(("k_mid_", "mkt_", "p_over_1_5", "p_over_3_5", "p_home_m1_5", "p_away_m1_5"))]:
            r[c] = r["event_id"].map(s[c]).combine_first(r[c] if c in r else pd.Series(np.nan, index=r.index))
    r["t"] = _when(r["kickoff_utc"])
    r = r[r["t"] > now].sort_values("t")
    lat = _lat()
    tri = lambda x, cols: " / ".join(pct(pd.to_numeric(pd.Series([getattr(x, c, np.nan)]), errors="coerce").iloc[0]) for c in cols)
    rows = [{"Fecha (hora Chicago)": x.t.strftime("%a %d %b %H:%M"), "Competición": x.competicion,
             "Partido": f"{x.home} vs {x.away}" + (" (sede neutral)" if x.neutral else ""),
             "Modelo: local / empate / visita": tri(x, ["p_H", "p_D", "p_A"]),
             "Kalshi": tri(x, ["k_mid_H", "k_mid_D", "k_mid_A"]), "Casas": tri(x, ["mkt_H", "mkt_D", "mkt_A"]),
             "Goles: +1.5 / +2.5 / +3.5": " / ".join(pct(getattr(x, c, np.nan)) for c in ("p_over_1_5", "p_over_2_5", "p_over_3_5")),
             "Ambos anotan": pct(getattr(x, "p_btts", np.nan)),
             "Gana por 2+: local / visita": f"{pct(getattr(x, 'p_home_m1_5', np.nan))} / {pct(getattr(x, 'p_away_m1_5', np.nan))}",
             "Bajas: local / visita": _bajas_fut(x, lat)} for x in r.itertuples()]
    return _table(pd.DataFrame(rows))


def final_prueba(now) -> str:
    """Pick con la probabilidad FINAL en modo de prueba (consenso de casas si existe; si no, el modelo)."""
    f = P / "final_shadow.csv"
    if not f.exists():
        return "_Aún no hay datos._\n"
    d = pd.read_csv(f, dtype={"event_id": str})
    d["t"] = pd.to_datetime(d["kickoff_utc"], utc=True, format="ISO8601").dt.tz_convert(TZ)
    d = d[d["t"] > now].sort_values("snapshot_utc").drop_duplicates(["sport", "event_id"], keep="last").sort_values("t")
    names = {"nba": "NBA", "nfl": "NFL", "club": "Copas europeas", "intl": "Selecciones"}
    rows = []
    for x in d.itertuples():
        opts = [(x.home, x.final_H), ("Empate", x.final_D), (x.away, x.final_A)]
        pick, prob = max([(n, v) for n, v in opts if pd.notna(v)], key=lambda z: z[1])
        rows.append({"Fecha (hora Chicago)": x.t.strftime("%a %d %b %H:%M"), "Deporte": names.get(x.sport, x.sport),
                     "Partido": f"{x.away} en {x.home}" if x.sport in ("nba", "nfl") else f"{x.home} vs {x.away}",
                     "Pick final": pick, "Probabilidad": pct(prob),
                     "Origen": "casas" if x.final_source == "market_consensus" else "modelo (sin cuotas aún)"})
    return _table(pd.DataFrame(rows))


def picks_hoy(now, horas: int = 36) -> str:
    """Vista simple: un pick por partido de las próximas horas, con nivel de confianza."""
    fin = now + pd.Timedelta(hours=horas)
    nivel = lambda v: "Alta" if v >= 0.70 else ("Media" if v >= 0.55 else "Baja")
    rows = []
    f = P / "final_shadow.csv"
    if f.exists():
        d = pd.read_csv(f, dtype={"event_id": str})
        d["t"] = pd.to_datetime(d["kickoff_utc"], utc=True, format="ISO8601").dt.tz_convert(TZ)
        d = d[(d["t"] > now) & (d["t"] <= fin)].sort_values("snapshot_utc").drop_duplicates(["sport", "event_id"], keep="last")
        names = {"nba": "NBA", "nfl": "NFL", "club": "Copa europea", "intl": "Selecciones"}
        for x in d.itertuples():
            pick, prob = max([(n, v) for n, v in ((x.home, x.final_H), ("Empate", x.final_D), (x.away, x.final_A)) if pd.notna(v)], key=lambda z: z[1])
            rows.append((x.t, names.get(x.sport, x.sport), f"{x.away} en {x.home}" if x.sport in ("nba", "nfl") else f"{x.home} vs {x.away}", pick, prob))
    reg = P / "registry.csv"
    if reg.exists():                                   # ligas domésticas: aún sin cuotas de casas, pick del modelo
        r = pd.read_csv(reg)
        r["t"] = _when(r["kickoff_utc"])
        for x in r[(r["t"] > now) & (r["t"] <= fin)].itertuples():
            pick, prob = max(((x.home, x.p_H), ("Empate", x.p_D), (x.away, x.p_A)), key=lambda z: z[1])
            rows.append((x.t, x.competition, f"{x.home} vs {x.away}", pick, prob))
    if not rows:
        return "_No hay partidos en las próximas horas._\n"
    rows.sort(key=lambda z: z[0])
    return _table(pd.DataFrame([{"Hora (Chicago)": t.strftime("%a %d %b %H:%M"), "Competición": c, "Partido": m, "Pick": k,
                                 "Probabilidad": pct(v), "Confianza": nivel(v)} for t, c, m, k, v in rows]))


def build(now=None) -> str:
    now = now or pd.Timestamp.now(tz=TZ)
    return f"""# Predicciones — actualizado {now.strftime('%d %b %Y %H:%M')} (hora Chicago)

📱 **Versión para el teléfono:** https://gabrielreyesu.github.io/kalshi-ai/ (tarjetas, filtros por deporte y día, buscador).

**Ir a:** [NBA](#nba) · [NFL](#nfl) · [Ligas de fútbol](#fútbol-ligas-premier-la-liga-serie-a-bundesliga-ligue-1) · [Copas europeas](#fútbol-copas-europeas-champions-europa-league-conference) · [Selecciones](#fútbol-selecciones-uefa-y-conmebol) · [Resumen de picks](#resumen-de-picks-de-hoy-y-mañana)

**Cómo leer las tablas**
- **Modelo**: probabilidad que da el modelo propio al favorito indicado.
- **Kalshi** / **Polymarket**: probabilidad que implica el precio en cada mercado (62¢ ≈ 62%).
- **Casas**: probabilidad según las casas de apuestas, sin su margen. Es la referencia más fiable.
- Si **Kalshi** y **Polymarket** difieren más de 3 puntos, vale la pena mirar ese partido: uno de los dos está desfasado.
- **Diferencia esperada** y **Total esperado**: por cuántos puntos ganaría el favorito y cuántos puntos habría en total.
- **Bajas**: jugadores fuera o dudosos. Es informativo: el modelo todavía NO las usa.
- En fútbol, **Goles** = probabilidad de más de 1.5, 2.5 y 3.5 goles; **Ambos anotan**; **Gana por 2+** = victoria por dos o más goles.
- Un "—" significa que aún no había precio o dato para ese partido.

## NBA
{liga_us('nba', now)}
## NFL
{liga_us('nfl', now)}
## Fútbol: ligas (Premier, La Liga, Serie A, Bundesliga, Ligue 1)
{futbol(now)}
## Fútbol: copas europeas (Champions, Europa League, Conference)
{internacional(now, 'clubes')}
## Fútbol: selecciones (UEFA y CONMEBOL)
En selecciones, las columnas de goles son poco fiables: en el backtest el "más de 2.5" fue peor que la línea base.

{internacional(now, 'selecciones')}
## Resumen de picks de hoy y mañana
Un pick por partido. **Confianza**: Alta = 70% o más, Media = 55% a 69%, Baja = menos de 55%.
Usa el consenso de las casas cuando existe y el modelo como respaldo (en prueba). No es una recomendación de apuesta.

{picks_hoy(now)}
"""


if __name__ == "__main__":
    Path("PREDICCIONES.md").write_text(build(), encoding="utf-8")
    print("PREDICCIONES.md generado")
