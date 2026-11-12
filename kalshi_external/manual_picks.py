"""Seguimiento MANUAL de analistas: el usuario anota un pick por línea en analysts/mis_picks.csv y este módulo
lo une con el partido real, registra la hora de captura, lo liquida cuando termina y mide a cada analista.

Es solo evaluación: estos picks NUNCA se usan para entrenar ni para cambiar probabilidades.
Trazabilidad: la hora de captura es la de la corrida (y queda en el historial de git); un pick anotado con menos de
15 minutos de margen antes del inicio, o después, se rechaza.

Uso:  python -m kalshi_external.manual_picks sync
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .analysts import MIN_N, wilson
from .core import team_name

SRC, LEDGER, REJECTED, STATUS = (Path("analysts") / n for n in ("mis_picks.csv", "libro.csv", "rechazados.csv", "ESTADO.md"))
SRC_COLS = ["analista", "deporte", "local", "visitante", "fecha_partido", "mercado", "seleccion", "linea", "cuota", "link"]
LEDGER_COLS = ["pick_id", "analista", "deporte", "mercado", "event_id", "local", "visitante", "kickoff_utc", "seleccion", "linea",
               "cuota", "link", "captured_utc", "uso", "resultado", "marcador"]
SPORTS = {"nfl": "nfl", "nba": "nba", "futbol": "futbol", "fútbol": "futbol", "soccer": "futbol"}
MARKETS = {"ganador", "spread", "total", "ambos_anotan"}
SEL = {"local": "home", "home": "home", "visitante": "away", "visita": "away", "away": "away", "empate": "draw", "draw": "draw",
       "mas": "over", "más": "over", "over": "over", "menos": "under", "under": "under", "si": "yes", "sí": "yes", "yes": "yes", "no": "no"}
ALLOWED = {"ganador": {"home", "away", "draw"}, "spread": {"home", "away"}, "total": {"over", "under"}, "ambos_anotan": {"yes", "no"}}


def fixtures(sport: str) -> pd.DataFrame:
    """Partidos reales (ESPN): event_id, kick, home, away, hs, as_, final, regulation."""
    if sport in ("nba", "nfl"):
        from kalshi_nba.data import current_season, load_league
        g = load_league(sport, first_season=current_season(sport), with_odds=False, verbose=False)
        g["kick"] = pd.to_datetime(g["kickoff_utc"], utc=True)
        g["final"] = g["status"].eq("STATUS_FINAL") & g["hs"].notna()
        g["regulation"] = True
    else:
        from kalshi_intl.data import load_group
        g = pd.concat([load_group(x, first_year=pd.Timestamp.now().year - 1, verbose=False) for x in ("clubes", "selecciones")], ignore_index=True)
        g["regulation"] = g["status"].isin({"STATUS_FULL_TIME", "STATUS_FINAL"})
    return g[["event_id", "kick", "home", "away", "hs", "as_", "final", "regulation"]].drop_duplicates("event_id")


def _same(a: str, b: str) -> bool:
    x, y = team_name(a), team_name(b)
    return len(x) >= 3 and len(y) >= 3 and (x == y or x in y or y in x)


def find_event(fx: pd.DataFrame, local: str, visitante: str, fecha) -> tuple:
    """(fila | None, motivo). Requiere UN solo partido con esos dos equipos a ±1 día de la fecha indicada."""
    day = pd.Timestamp(fecha)
    if pd.isna(day):
        return None, "fecha_partido inválida (usa AAAA-MM-DD)"
    day = day.tz_localize("UTC") if day.tzinfo is None else day.tz_convert("UTC")
    near = fx[(fx["kick"] >= day - pd.Timedelta(days=1)) & (fx["kick"] < day + pd.Timedelta(days=2))]
    hit = near[[_same(local, h) and _same(visitante, a) for h, a in zip(near["home"], near["away"])]]
    if len(hit) == 1:
        return hit.iloc[0], ""
    swapped = near[[_same(local, a) and _same(visitante, h) for h, a in zip(near["home"], near["away"])]]
    if len(hit) == 0 and len(swapped) == 1:
        return None, "local y visitante están al revés"
    if len(hit) > 1:
        return None, "varios partidos coinciden: escribe los nombres completos"
    wide = fx[(fx["kick"] >= day - pd.Timedelta(days=10)) & (fx["kick"] < day + pd.Timedelta(days=11))]
    wide = wide[[_same(local, h) and _same(visitante, a) for h, a in zip(wide["home"], wide["away"])]]
    if len(wide) == 1:
        return None, f"no hay partido ese día; el más cercano entre esos equipos es el {wide.iloc[0]['kick'].strftime('%Y-%m-%d')} (UTC)"
    return None, "no se encontró el partido: revisa los nombres de los equipos y la fecha"


def parse_row(r: dict) -> tuple:
    """Normaliza una línea anotada. Devuelve (dict | None, motivo)."""
    need = [c for c in ("analista", "deporte", "local", "visitante", "fecha_partido", "mercado", "seleccion") if not str(r.get(c) or "").strip() or str(r.get(c)).strip().lower() == "nan"]
    if need:
        return None, "faltan columnas: " + ", ".join(need)
    sport = SPORTS.get(str(r["deporte"]).strip().lower())
    market = str(r["mercado"]).strip().lower()
    if sport is None:
        return None, "deporte debe ser nfl, nba o futbol"
    if market not in MARKETS:
        return None, "mercado debe ser ganador, spread, total o ambos_anotan"
    raw = str(r["seleccion"]).strip()
    sel = SEL.get(raw.lower())
    if sel is None:                                       # también se acepta el nombre del equipo
        sel = "home" if _same(raw, str(r["local"])) else ("away" if _same(raw, str(r["visitante"])) else None)
    if sel not in ALLOWED[market] or (sel == "draw" and sport != "futbol") or (market == "ambos_anotan" and sport != "futbol"):
        return None, f"selección no válida para {market}"
    line = pd.to_numeric(str(r.get("linea") or "").replace(",", "."), errors="coerce")
    if market in ("spread", "total") and pd.isna(line):
        return None, f"falta la línea para {market}"
    odds = pd.to_numeric(str(r.get("cuota") or "").replace(",", "."), errors="coerce")
    if not pd.isna(odds) and odds <= 1:
        return None, "cuota debe ser decimal mayor que 1 (ej. 1.91)"
    return {"analista": str(r["analista"]).strip(), "deporte": sport, "mercado": market, "seleccion": sel,
            "linea": None if pd.isna(line) else float(line), "cuota": None if pd.isna(odds) else float(odds),
            "link": str(r.get("link") or "").strip().replace("nan", "")}, ""


def settle(market: str, sel: str, line, hs: float, as_: float, sport: str, regulation: bool) -> str:
    if sport == "futbol" and not regulation:
        return "sin_liquidar"                              # fue a prórroga: el pick era a 90 minutos y no tenemos ese marcador
    m, t = hs - as_, hs + as_
    if market == "ganador":
        res = "home" if m > 0 else ("away" if m < 0 else "draw")
        if res == "draw" and sport != "futbol":
            return "push"
        return "win" if sel == res else "loss"
    if market == "spread":
        v = (m if sel == "home" else -m) + float(line)
        return "push" if v == 0 else ("win" if v > 0 else "loss")
    if market == "total":
        if t == float(line):
            return "push"
        return "win" if (t > float(line)) == (sel == "over") else "loss"
    return "win" if ((hs > 0 and as_ > 0) == (sel == "yes")) else "loss"


def sync(src=SRC, ledger=LEDGER, rejected=REJECTED, status=STATUS, now=None, fixtures_fn=fixtures) -> dict:
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    now = now.tz_localize("UTC") if now.tzinfo is None else now
    led = pd.read_csv(ledger, dtype=str) if Path(ledger).exists() else pd.DataFrame(columns=LEDGER_COLS)
    raw = pd.read_csv(src, dtype=str).reindex(columns=SRC_COLS) if Path(src).exists() else pd.DataFrame(columns=SRC_COLS)
    cache, bad, new = {}, [], []
    fx = lambda s: cache.setdefault(s, fixtures_fn(s))
    seen = set(led["pick_id"])
    for r in raw.to_dict("records"):
        if str(r.get("analista") or "").strip().lower() in ("", "nan") or str(r["analista"]).startswith("#"):
            continue
        p, why = parse_row(r)
        ev = None
        if p:
            ev, why = find_event(fx(p["deporte"]), str(r["local"]), str(r["visitante"]), r["fecha_partido"])
        if ev is None:
            bad.append({**r, "motivo": why}); continue
        pid = "|".join([p["analista"].lower(), p["deporte"], p["mercado"], str(ev["event_id"]), p["seleccion"], str(p["linea"])])
        if pid in seen:
            continue                                        # ya registrado: el primer registro manda
        if now >= ev["kick"] - pd.Timedelta(minutes=15):
            bad.append({**r, "motivo": "anotado demasiado tarde: el partido ya empezó o faltaban menos de 15 minutos"}); continue
        seen.add(pid)
        new.append({"pick_id": pid, **p, "event_id": str(ev["event_id"]), "local": ev["home"], "visitante": ev["away"],
                    "kickoff_utc": ev["kick"].isoformat(), "captured_utc": now.isoformat(), "uso": "solo_evaluacion", "resultado": "pendiente", "marcador": ""})
    led = pd.concat([led, pd.DataFrame(new)], ignore_index=True).reindex(columns=LEDGER_COLS)
    for i, r in led[led["resultado"].isin(["pendiente"])].iterrows():           # liquidar los ya jugados
        f = fx(r["deporte"]); e = f[f["event_id"].astype(str) == str(r["event_id"])]
        if len(e) == 1 and bool(e.iloc[0]["final"]):
            e = e.iloc[0]
            led.at[i, "resultado"] = settle(r["mercado"], r["seleccion"], r["linea"], float(e["hs"]), float(e["as_"]), r["deporte"], bool(e["regulation"]))
            led.at[i, "marcador"] = f"{int(e['hs'])}-{int(e['as_'])}"
    Path(ledger).parent.mkdir(parents=True, exist_ok=True)
    led.to_csv(ledger, index=False)
    pd.DataFrame(bad).to_csv(rejected, index=False)
    Path(status).write_text(estado(led, bad, now), encoding="utf8")
    return {"nuevos": len(new), "rechazados": len(bad), "en_libro": len(led), "pendientes": int((led["resultado"] == "pendiente").sum())}


def estado(led: pd.DataFrame, bad: list, now) -> str:
    L = [f"# Seguimiento de analistas — actualizado {now.tz_convert('America/Chicago').strftime('%d %b %Y %H:%M')} (hora Chicago)", "",
         "Solo evaluación: estos picks no se usan para entrenar ni cambian las predicciones.",
         f"Con menos de {MIN_N} picks decididos, un porcentaje alto todavía es compatible con suerte.", ""]
    rows = []
    for (a, s, m), g in led.groupby(["analista", "deporte", "mercado"]):
        w, l, p = (int((g["resultado"] == k).sum()) for k in ("win", "loss", "push"))
        n = w + l
        lo, hi = wilson(w, n)
        odds = pd.to_numeric(g.loc[g["resultado"].isin(["win", "loss"]), "cuota"], errors="coerce")
        won = g.loc[g["resultado"].isin(["win", "loss"]), "resultado"].eq("win").to_numpy()
        profit = float(np.where(won, odds - 1, -1.0).sum()) if n and odds.notna().all() else None
        rows.append({"Analista": a, "Deporte": s, "Mercado": m, "Ganados": w, "Perdidos": l, "Nulos": p,
                     "Pendientes": int((g["resultado"] == "pendiente").sum()), "Acierto": "—" if not n else f"{100 * w / n:.0f}%",
                     "Acierto real probable": "—" if not n else f"{100 * lo:.0f}% a {100 * hi:.0f}%",
                     "Beneficio (unidades)": "—" if profit is None else f"{profit:+.2f}", "Muestra suficiente": "sí" if n >= MIN_N else "no"})
    if rows:
        t = pd.DataFrame(rows)
        L += ["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)] + ["| " + " | ".join(str(v) for v in r) + " |" for r in t.to_numpy()]
    else:
        L.append("_Aún no hay picks registrados._")
    if bad:
        L += ["", "## Líneas rechazadas en la última corrida", ""] + [f"- {b.get('analista')}: {b.get('local')} vs {b.get('visitante')} ({b.get('fecha_partido')}) → {b['motivo']}" for b in bad]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    print(sync() if (len(sys.argv) < 2 or sys.argv[1] == "sync") else __doc__)
