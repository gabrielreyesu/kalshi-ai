"""Generate a reference-only summary of source coverage and analyst profile links.

No web scraping, no third-party pick copying, no model training.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


def _safe(s):
    return str(s or "").replace("|", "/").replace("\n", " ").replace("\r", " ").strip()[:180]


def read_watchlist(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    result = []
    seen = set()
    for r in rows:
        link = str(r.get("URL perfil") or "").strip()
        parsed = urlparse(link)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            continue
        name = _safe(r.get("Analista"))
        sport = _safe(r.get("Deporte"))
        market = _safe(r.get("Mercado específico"))
        if not (name and sport):
            continue
        key = (name.casefold(), sport.casefold(), market.casefold())
        if key in seen:
            continue
        seen.add(key)
        try:
            n = max(0, int(float(r.get("N decidido") or 0)))
        except (TypeError, ValueError):
            n = 0
        try:
            rate = f"{float(r['Win rate publicado (%)']):.1f}%"
        except (TypeError, ValueError, KeyError):
            rate = "sin dato"
        result.append({
            "name": name, "sport": sport, "market": market, "url": link,
            "sample": n, "rate": rate, "priority": 0 if
            _safe(r.get("Prioridad")).lower() == "alta" else 1,
        })
    return result


def read_status(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def report(watchlist, status, max_per_sport=5):
    if max_per_sport < 1:
        raise ValueError("max_per_sport must be positive")
    lines = [
        "# Referencias externas — Mocha Picks (solo consulta)",
        "",
        "Actualizado: " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "",
        "**Sin trabajo manual:** no hace falta copiar picks de X, Discord ni screenshots.",
        "Este informe no extrae selecciones de páginas ni entrena modelos.",
        "",
        "## Cobertura de proveedores autorizados",
        "",
    ]
    if status is None:
        lines.append("No hay resumen de la última consulta; no se confirma ninguna captura.")
    else:
        lines.extend([
            "- Observaciones prepartido en caché: " + _safe(status.get("observations_in_private_cache", "desconocido")),
            "- Nuevas observaciones de la última ejecución: " + _safe(status.get("new_records", "desconocido")),
            "- Fuentes con 30+ partidos resueltos: " + _safe(status.get("sources_with_30_plus_resolved_cases", 0)),
            "- Esos valores miden cobertura; **no** demuestran una mejora predictiva.",
        ])
        for source, count in sorted((status.get("new_by_source") or {}).items()):
            lines.append("- " + _safe(source) + ": " + _safe(count) + " observaciones reportadas")
        if status.get("warnings"):
            lines.append("- Hay advertencias; consultar el archivo status.json de esa ejecución.")
    lines.extend([
        "",
        "Proveedores enlazados: [The Odds API](https://the-odds-api.com/), "
        "[API-Football](https://www.api-football.com/), [Skorbot](https://skorbot.live/). "
        "El acceso y permiso de uso de cada fuente deben verificarse.",
        "",
        "## Directorio público de analistas — solo referencia",
        "",
        "Las tasas son registros históricos declarados por sus plataformas, no picks verificados por Mocha Picks. "
        "No se usan para entrenar ni como señales automatizadas.",
        "",
    ])
    for sport in ["NFL", "NBA", "Soccer", "Soccer / NBA", "College football"]:
        selected = [r for r in watchlist if r["sport"] == sport]
        if not selected:
            continue
        selected.sort(key=lambda r: (r["priority"], -r["sample"], r["name"].casefold()))
        lines += ["### " + sport, "", "| Perfil | Mercado | Tasa publicada | N |",
                  "|---|---|---:|---:|"]
        for r in selected[:max_per_sport]:
            sample = str(r["sample"]) if r["sample"] else "sin dato"
            lines.append(
                "| [" + r["name"] + "](" + r["url"] + ") | " + r["market"]
                + " | " + r["rate"] + " | " + sample + " |"
            )
        lines.append("")
    lines.extend([
        "## Mercados adicionales que interesa estudiar",
        "",
        "NFL: recepciones, yardas recibidas y spreads. NBA: puntos, rebotes y triples. "
        "Fútbol: 1X2 y tiros a puerta. NHL: tiros a puerta y spreads.",
        "",
        "**Cobertura actual:** The Odds API está consultando h2h (ganadores), "
        "no player props. Estos mercados necesitan endpoints, licencias, cuotas y resultados propios.",
        "Las capturas de boletos ganadores tienen sesgo de selección y no constituyen un historial verificable.",
        "",
    ])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watchlist", default="analysts/watchlist.csv")
    ap.add_argument("--status", default="reports/external_learning/status.json")
    ap.add_argument("--output", default="reports/external_learning/REFERENCIAS.md")
    args = ap.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report(read_watchlist(args.watchlist), read_status(args.status)), encoding="utf8")
    print("Reference-only summary saved to", output)


if __name__ == "__main__":
    main()
