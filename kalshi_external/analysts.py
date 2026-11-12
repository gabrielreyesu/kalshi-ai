"""Analistas: auditoría estadística de la lista de seguimiento y libro de pronósticos con trazabilidad temporal.

Reglas (CLAUDE_HANDOFF.md): solo se aceptan pronósticos con AUTORIZACIÓN explícita de uso, publicados y capturados
antes del partido. No se hace scraping. Todo es experimental: nada de esto modifica las probabilidades de producción.

Uso:
  python -m kalshi_external.analysts audit [analysts/watchlist.csv]       # reports/analysts/watchlist_audit.{csv,md}
  python -m kalshi_external.analysts ingest <picks.csv> [libro.jsonl]     # valida y agrega al libro (append-only)
  python -m kalshi_external.analysts metrics <libro.jsonl> <resultados.csv>
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MARKETS = {"nfl": {"moneyline", "ats", "total", "prop"}, "nba": {"moneyline", "spread", "total", "prop"},
           "soccer": {"1x2_90", "1x2_official", "btts", "goals", "handicap"}}
# Tasa que un pronosticador debe superar para aportar algo: en mercados a cuota ~1.91 el punto de equilibrio es 52.4%;
# en "ganador directo" (SU) hay que superar a quien elige siempre al favorito del mercado (medido en nuestros backtests).
BREAK_EVEN_110 = 0.524
SU_BASELINE = {"NFL": 0.663, "NBA": 0.676}
MIN_N = 200                       # por debajo, un porcentaje alto es compatible con suerte
PICK_FIELDS = ["analyst_id", "sport", "market", "event_id", "selection", "line", "odds_decimal", "probability",
               "published_utc", "captured_utc", "kickoff_utc", "source_url", "authorized_for_modeling"]


def wilson(wins: float, n: float, z: float = 1.96) -> tuple:
    """Intervalo de confianza de Wilson al 95% para una proporción."""
    if not n or n <= 0:
        return (float("nan"), float("nan"))
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def audit_watchlist(path="analysts/watchlist.csv") -> pd.DataFrame:
    d = pd.read_csv(path)
    out = []
    for r in d.itertuples(index=False):
        vals = list(r)
        analyst, sport, market = vals[2], str(vals[1]), str(vals[6])
        wins, losses, rate, n, odds = (pd.to_numeric(vals[i], errors="coerce") for i in (7, 8, 10, 11, 13))
        if pd.isna(n) and not pd.isna(wins) and not pd.isna(losses):
            n = wins + losses
        if pd.isna(wins) and not pd.isna(n) and not pd.isna(rate):
            wins = round(n * rate / 100)
        lo, hi = wilson(wins, n) if not pd.isna(n) and not pd.isna(wins) else (np.nan, np.nan)
        if market.strip().upper() == "SU":
            base, base_name = SU_BASELINE.get(sport.split("/")[0].strip(), np.nan), "elegir siempre al favorito"
        elif not pd.isna(odds) and odds > 1:
            base, base_name = 1 / odds, f"punto de equilibrio a cuota {odds:.2f}"
        else:
            base, base_name = BREAK_EVEN_110, "punto de equilibrio a cuota 1.91"
        rights = str(vals[16])
        if pd.isna(n):
            verdict = "sin tamaño de muestra: no evaluable"
        elif lo > base and n >= MIN_N:
            verdict = "supera la referencia con muestra suficiente"
        elif lo > base:
            verdict = "supera la referencia, pero muestra pequeña"
        else:
            verdict = "no distinguible de la referencia"
        out.append({"analista": analyst, "deporte": sport, "mercado": market, "n": n, "acierto_%": None if pd.isna(n) or not n else round(100 * wins / n, 1),
                    "ic95_inf_%": None if pd.isna(lo) else round(100 * lo, 1), "ic95_sup_%": None if pd.isna(hi) else round(100 * hi, 1),
                    "referencia_%": round(100 * base, 1) if not pd.isna(base) else None, "referencia": base_name, "veredicto": verdict,
                    "autorizado_para_modelos": False, "derechos": rights})
    return pd.DataFrame(out)


def validate_pick(p: dict) -> list:
    """Lista de motivos de rechazo (vacía = aceptado)."""
    why = []
    for f in ("analyst_id", "sport", "market", "event_id", "selection", "published_utc", "captured_utc", "kickoff_utc"):
        if p.get(f) in (None, "") or (isinstance(p.get(f), float) and math.isnan(p.get(f))):
            why.append(f"falta {f}")
    if why:
        return why
    if str(p.get("authorized_for_modeling", "")).strip().lower() not in ("1", "true", "yes", "si", "sí"):
        why.append("sin autorización explícita de uso en modelos")
    sport, market = str(p["sport"]).lower(), str(p["market"]).lower()
    if sport not in MARKETS or market not in MARKETS[sport]:
        why.append(f"mercado no reconocido: {sport}/{market}")
    try:
        pub, cap, ko = (pd.Timestamp(p[k]) for k in ("published_utc", "captured_utc", "kickoff_utc"))
        pub, cap, ko = (t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC") for t in (pub, cap, ko))
        if pub > cap:
            why.append("publicado después de capturado")
        if cap >= ko - pd.Timedelta(minutes=15):
            why.append("capturado sin margen antes del inicio")
    except (ValueError, TypeError):
        why.append("fechas inválidas")
    for f, lo, hi in (("odds_decimal", 1.0, 1000.0), ("probability", 0.0, 1.0)):
        v = pd.to_numeric(p.get(f), errors="coerce")
        if not pd.isna(v) and not (lo < v < hi):
            why.append(f"{f} fuera de rango")
    return why


def ingest(picks_csv, ledger="data/analysts/picks.jsonl") -> dict:
    d = pd.read_csv(picks_csv, dtype=str)
    ledger = Path(ledger)
    seen = set()
    if ledger.exists():
        for line in ledger.read_text(encoding="utf8").splitlines():
            if line.strip():
                r = json.loads(line); seen.add((r["analyst_id"], r["sport"], r["market"], str(r["event_id"]), r["selection"]))
    ok, rejected = [], []
    for row in d.to_dict("records"):
        why = validate_pick(row)
        key = (row.get("analyst_id"), str(row.get("sport", "")).lower(), str(row.get("market", "")).lower(), str(row.get("event_id")), row.get("selection"))
        if why:
            rejected.append({**row, "motivo": "; ".join(why)})
        elif key in seen:
            rejected.append({**row, "motivo": "duplicado: el primer registro se conserva"})
        else:
            seen.add(key)
            ok.append({**{f: row.get(f) for f in PICK_FIELDS}, "sport": key[1], "market": key[2]})
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf8") as f:
        for r in ok:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"aceptados": len(ok), "rechazados": len(rejected), "detalle_rechazos": rejected}


def metrics(ledger_rows: list, results: pd.DataFrame) -> pd.DataFrame:
    """results: analyst_id, sport, market, event_id, selection, outcome (win|loss|push) [, market_prob].
    Por analista+deporte+mercado: muestra, acierto con IC, beneficio a la cuota registrada, y ventaja frente al mercado."""
    if not ledger_rows or results.empty:
        return pd.DataFrame()
    keys = ["analyst_id", "sport", "market", "event_id", "selection"]
    L = pd.DataFrame(ledger_rows).astype({"event_id": str})
    d = L.merge(results.astype({"event_id": str}), on=keys, how="inner")
    out = []
    for (a, s, m), g in d.groupby(["analyst_id", "sport", "market"]):
        w, l, psh = (int((g["outcome"] == k).sum()) for k in ("win", "loss", "push"))
        n = w + l
        lo, hi = wilson(w, n)
        odds = pd.to_numeric(g["odds_decimal"], errors="coerce")
        dec = g[g["outcome"].isin(["win", "loss"])]
        profit = float(np.where(dec["outcome"] == "win", pd.to_numeric(dec["odds_decimal"], errors="coerce") - 1, -1.0).sum()) if odds.notna().all() and n else float("nan")
        row = {"analyst_id": a, "sport": s, "market": m, "n": n, "pushes": psh, "win_rate": w / n if n else float("nan"),
               "ic95_inf": lo, "ic95_sup": hi, "profit_units": profit, "roi": profit / n if n and profit == profit else float("nan"),
               "muestra_suficiente": bool(n >= MIN_N)}
        if "market_prob" in g and n:
            mp = pd.to_numeric(dec["market_prob"], errors="coerce")
            row["prob_media_mercado"] = float(mp.mean()); row["ventaja_vs_mercado"] = w / n - float(mp.mean())
        pr = pd.to_numeric(dec["probability"], errors="coerce") if "probability" in dec else pd.Series(dtype=float)
        if len(pr) and pr.notna().all():                       # Brier/logloss solo si la fuente aporta probabilidades
            y = (dec["outcome"] == "win").to_numpy(float); q = np.clip(pr.to_numpy(float), 1e-6, 1 - 1e-6)
            row["brier"] = float(np.mean((q - y) ** 2)); row["logloss"] = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
        out.append(row)
    return pd.DataFrame(out)


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "audit"
    if cmd == "audit":
        t = audit_watchlist(argv[2] if len(argv) > 2 else "analysts/watchlist.csv")
        out = Path("reports/analysts"); out.mkdir(parents=True, exist_ok=True)
        t.to_csv(out / "watchlist_audit.csv", index=False)
        print(t.drop(columns=["derechos", "autorizado_para_modelos", "referencia"]).to_string(index=False))
    elif cmd == "ingest":
        r = ingest(argv[2], argv[3] if len(argv) > 3 else "data/analysts/picks.jsonl")
        print(json.dumps({k: v for k, v in r.items() if k != "detalle_rechazos"}, ensure_ascii=False))
        for x in r["detalle_rechazos"][:20]:
            print("  rechazado:", x.get("analyst_id"), x.get("event_id"), "->", x["motivo"])
    elif cmd == "metrics":
        rows = [json.loads(l) for l in Path(argv[2]).read_text(encoding="utf8").splitlines() if l.strip()]
        print(metrics(rows, pd.read_csv(argv[3], dtype=str)).round(4).to_string(index=False))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
