"""Archivo histórico DURABLE de observaciones externas: fusiona (sin duplicar) la caché local con un repositorio
privado de datos. La caché de GitHub Actions caduca; un repositorio privado no.

Uso:  python -m kalshi_external.archive merge <archivo_a.jsonl> <archivo_b.jsonl> <salida.jsonl>
Los datos de proveedores NO se publican en el repositorio de código.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _key(r: dict) -> tuple:
    return (r.get("source_id"), r.get("sport"), str(r.get("event_id")), r.get("captured_utc"))


def merge(paths, out) -> int:
    seen, rows = set(), []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                       # una línea corrupta no invalida el archivo
            k = _key(r)
            if k not in seen:
                seen.add(k); rows.append(r)
    rows.sort(key=lambda r: (str(r.get("captured_utc")), str(r.get("source_id")), str(r.get("event_id"))))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf8")
    return len(rows)


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "merge":
        raise SystemExit(__doc__)
    print("observaciones en el archivo:", merge(sys.argv[2:4], sys.argv[4]))
