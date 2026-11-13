"""Salud de las fuentes: avisa cuando una etapa corre "en verde" pero no produce nada útil.

Lee los JSON de estado que ya escriben los recolectores (no hace peticiones) y emite avisos de GitHub Actions
(`::warning::`) más una tabla en el resumen de la ejecución. Nunca hace fallar el workflow: un aviso no es un error,
pero tampoco debe pasar inadvertido.

  python -m kalshi_external.health --mode full|hourly --reports reports/external_learning
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# (archivo relativo, ruta de clave, modos donde aplica, mensaje). Se avisa si el valor es 0, falta o el archivo falta.
RULES = [
    ("status.json", "new_records", {"full"}, "The Odds API / fuentes externas: 0 registros nuevos"),
    ("weather_status.json", "new_forecast_snapshots", {"full"}, "NOAA: 0 pronósticos nuevos (puede ser normal sin NFL al aire libre en 72 h)"),
    ("player_context/coverage.json", "new_injury_snapshot_batches", {"full"}, "Lesiones NBA/NFL: 0 lotes nuevos"),
    ("soccer_players/coverage.json", "fixtures_matched_by_espn", {"full", "hourly"}, "Fútbol: 0 partidos de ESPN emparejados"),
    ("soccer_impact/coverage.json", "team_contexts_with_4_plus_prior_matches", {"full"},
     "Fútbol: 0 equipos con >=4 partidos previos con jugadores; la investigación de titulares no puede evaluarse"),
    ("soccer_impact/coverage.json", "eligible_for_prospective_model_test", {"full"},
     "Fútbol: 0 casos aptos para la prueba prospectiva de bajas/alineaciones"),
]


def _get(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def check(reports: Path, mode: str) -> list[dict]:
    out = []
    for rel, key, modes, msg in RULES:
        if mode not in modes:
            continue
        p = reports / rel
        if not p.exists():
            out.append({"archivo": rel, "clave": key, "valor": None, "estado": "FALTA_ARCHIVO", "mensaje": msg})
            continue
        try:
            v = _get(json.loads(p.read_text(encoding="utf8")), key)
        except ValueError:
            out.append({"archivo": rel, "clave": key, "valor": None, "estado": "JSON_INVALIDO", "mensaje": msg}); continue
        bad = v is None or (isinstance(v, (int, float)) and v == 0)
        out.append({"archivo": rel, "clave": key, "valor": v, "estado": "AVISO" if bad else "OK", "mensaje": msg})
    for rel in ("status.json", "soccer_players/coverage.json", "player_context/coverage.json"):
        p = reports / rel
        if p.exists():
            try:
                for w in (json.loads(p.read_text(encoding="utf8")).get("warnings") or [])[:5]:
                    out.append({"archivo": rel, "clave": "warnings", "valor": str(w)[:200], "estado": "AVISO", "mensaje": "Aviso del recolector"})
            except ValueError:
                pass
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "hourly"], default="full")
    ap.add_argument("--reports", default="reports/external_learning")
    a = ap.parse_args(argv)
    rows = check(Path(a.reports), a.mode)
    (Path(a.reports)).mkdir(parents=True, exist_ok=True)
    (Path(a.reports) / "source_health.json").write_text(json.dumps({"mode": a.mode, "checks": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    for r in rows:
        if r["estado"] != "OK":
            print(f"::warning title=Salud de fuentes::{r['mensaje']} ({r['archivo']}:{r['clave']}={r['valor']})")
    summ = os.getenv("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf8") as f:
            f.write(f"### Salud de fuentes ({a.mode})\n\n| Estado | Archivo | Clave | Valor | Detalle |\n|---|---|---|---|---|\n")
            for r in rows:
                f.write(f"| {r['estado']} | {r['archivo']} | {r['clave']} | {r['valor']} | {r['mensaje']} |\n")
    n = sum(r["estado"] != "OK" for r in rows)
    print(f"Salud de fuentes: {len(rows) - n} OK, {n} avisos")
    return rows


if __name__ == "__main__":
    main()
