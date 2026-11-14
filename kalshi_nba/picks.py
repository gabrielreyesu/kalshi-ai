"""Picks por mercado para la página y PREDICCIONES.md. Solo lee predictions/; no usa internet ni cambia probabilidades.

Cada pick = {mercado, pick, prob, fuente, precio}. 'prob' es la probabilidad de que el pick ACIERTE:
  * NBA/NFL: ganador, hándicap y total de puntos de los contratos de Kalshi. Probabilidad de las casas (sin margen)
    cuando existe; si no, del modelo. 'precio' = lo que cuesta ese contrato en Kalshi (mid), para comparar.
  * Fútbol: ganador, doble oportunidad, goles y ambos anotan, del modelo (o de las casas en 1X2 si existen).
Aviso honesto: en el backtest, el modelo NO mostró ventaja en hándicap y totales frente a la línea de las casas;
por eso esos picks usan la probabilidad de las casas y su "confianza" refleja probabilidad, no ventaja.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

SAFE_SPREAD, SAFE_TOTAL = 0.70, 0.75


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _k(x):
    return f"{x:g}"


def pick(market, choice, prob, src, price=None):
    return {"mercado": market, "pick": choice, "prob": None if prob is None else round(prob, 4), "fuente": src,
            "precio": None if price is None else round(price, 3)}


def us_snapshot(folder: Path, lg: str) -> pd.DataFrame:
    """Última foto de cada partido (todas sus líneas de ganador, hándicap y total)."""
    p = folder / f"{lg}_kalshi_snapshots.csv"
    if not p.exists():
        return pd.DataFrame()
    s = pd.read_csv(p, dtype={"event_id": str})
    for c in ("kind", "side", "strike", "mid", "p_model", "p_book"):
        if c not in s:
            s[c] = None
    last = s.groupby("event_id")["snapshot_utc"].transform("max")
    return s[s["snapshot_utc"] == last]


def us_picks(rows: pd.DataFrame, home: str, away: str, winner: tuple | None) -> list:
    """rows: líneas de UN partido. winner: (equipo, prob, fuente) ya calculado para el ganador."""
    out = []
    if winner and winner[0]:
        price = None
        if rows is not None and not rows.empty:
            w = rows[(rows["kind"] == "winner") & (rows["side"] == ("home" if winner[0] == home else "away"))]
            price = _f(w["mid"].iloc[0]) if len(w) else None
        out.append(pick("Ganador", winner[0], winner[1], winner[2], price))
    if rows is None or rows.empty:
        return out
    prob = lambda r: (_f(r.p_book), "casas") if _f(r.p_book) is not None else (_f(r.p_model), "modelo")
    team = {"home": home, "away": away}
    other = {"home": away, "away": home}
    sp = [r for r in rows[rows["kind"] == "spread"].itertuples() if prob(r)[0] is not None and r.side in team]
    if sp:
        r = min(sp, key=lambda r: abs(prob(r)[0] - 0.5))
        p, src = prob(r); mid = _f(r.mid)
        if p >= 0.5:
            out.append(pick("Hándicap", f"{team[r.side]} gana por más de {_k(r.strike)}", p, src, mid))
        else:
            out.append(pick("Hándicap", f"{other[r.side]} +{_k(r.strike)} (gana o pierde por menos de {_k(r.strike)})",
                            1 - p, src, None if mid is None else 1 - mid))
        safe = [r for r in sp if prob(r)[0] >= SAFE_SPREAD]
        if safe:
            r = max(safe, key=lambda r: r.strike)
            p, src = prob(r)
            out.append(pick("Hándicap seguro", f"{team[r.side]} gana por más de {_k(r.strike)}", p, src, _f(r.mid)))
    tt = [r for r in rows[rows["kind"] == "total"].itertuples() if prob(r)[0] is not None]
    if tt:
        r = min(tt, key=lambda r: abs(prob(r)[0] - 0.5))
        p, src = prob(r); mid = _f(r.mid)
        out.append(pick("Total de puntos", f"Más de {_k(r.strike)}" if p >= 0.5 else f"Menos de {_k(r.strike)}",
                        max(p, 1 - p), src, mid if p >= 0.5 else (None if mid is None else 1 - mid)))
        safe = []
        over = [r for r in tt if prob(r)[0] >= SAFE_TOTAL]
        under = [r for r in tt if 1 - prob(r)[0] >= SAFE_TOTAL]
        if over:
            r = max(over, key=lambda r: r.strike); p, src = prob(r)
            safe.append(pick("Puntos (línea segura)", f"Más de {_k(r.strike)}", p, src, _f(r.mid)))
        if under:
            r = min(under, key=lambda r: r.strike); p, src = prob(r); mid = _f(r.mid)
            safe.append(pick("Puntos (línea segura)", f"Menos de {_k(r.strike)}", 1 - p, src, None if mid is None else 1 - mid))
        if safe:
            out.append(min(safe, key=lambda z: z["prob"]))      # la más cercana al 75%: la más informativa
    return out


def over25(model_p, book_p):
    """Más de 2,5 goles: casas + modelo si hay línea de las casas (pesos validados en goals_bench), si no el modelo."""
    import json
    from pathlib import Path
    f = Path(__file__).with_name("final_weights.json")
    w = json.loads(f.read_text()).get("futbol_goles_2_5", {}).get("pesos") if f.exists() else None
    ok = lambda v: v is not None and math.isfinite(v) and 0 < v < 1
    if w and ok(book_p) and ok(model_p):
        lg = lambda v: math.log(v / (1 - v))
        z = w["intercepto"] + w["casas"] * lg(book_p) + w["modelo"] * lg(model_p)
        return 1 / (1 + math.exp(-z)), "casas + modelo"
    if ok(book_p):
        return book_p, "casas"
    return (model_p, "modelo") if ok(model_p) else (None, None)


def soccer_picks(home, away, probs3, src3, over: dict, btts, goals_ok=True, kalshi3=None, book_over=None) -> list:
    """probs3 = [H, D, A] (casas o modelo); over = {1.5: p, 2.5: p}; btts = p (modelo); kalshi3 = precios 1X2;
    book_over = probabilidad de las casas de más de 2,5 goles (sin margen)."""
    out = []
    if all(v is not None for v in probs3):
        h, d, a = probs3
        i, best = max(enumerate(((home, h), ("Empate", d), (away, a))), key=lambda z: z[1][1])
        price = (kalshi3 or [None] * 3)[i]
        out.append(pick("Ganador", best[0], best[1], src3, price))
        dc = max(((f"{home} o empate", h + d), (f"{away} o empate", a + d), (f"{home} o {away} (no empate)", h + a)), key=lambda z: z[1])
        out.append(pick("Doble oportunidad", dc[0], dc[1], src3))
    p25, src25 = over25(over.get(2.5), book_over)
    if p25 is not None and (goals_ok or src25 != "modelo"):          # con línea de las casas también en selecciones
        out.append(pick("Goles", "Más de 2.5" if p25 >= 0.5 else "Menos de 2.5", max(p25, 1 - p25), src25))
    if goals_ok:
        p15 = over.get(1.5)
        if p15 is not None and p15 >= 0.5:
            out.append(pick("Goles (línea segura)", "Más de 1.5", p15, "modelo"))
        if btts is not None:
            out.append(pick("Ambos anotan", "Sí" if btts >= 0.5 else "No", max(btts, 1 - btts), "modelo"))
    return out
