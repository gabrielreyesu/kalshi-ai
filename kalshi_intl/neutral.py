"""Sede neutral inferida para torneos de selecciones (investigación; NO cambia el motor de producción).

ESPN marca `neutralSite` casi siempre en falso (DATA_QUALITY_AUDIT / AUDIT_V9). Regla fijada antes de ver resultados:
un partido de fase final de torneo (Mundial, Eurocopa, Copa América, Copa Asiática, Copa Africana, Copa Oro y la fase
final de la Nations League en junio) es NEUTRAL salvo que el equipo local sea anfitrión de esa edición.

  python -m kalshi_intl.neutral [--cache data/intl] [--output reports/intl_neutral]
Criterio: igual que el resto de pruebas (IC95% de mejora de logloss > 0, Brier no peor, ECE no peor en > 0,005).
"""
from __future__ import annotations

import argparse
import json
import os
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# (código ESPN, año del partido) -> anfitriones. Ediciones desplazadas de año usan el año en que se jugaron.
HOSTS = {
    "fifa.world": {2018: {"Russia"}, 2022: {"Qatar"}, 2026: {"United States", "USA", "Mexico", "Canada"}},
    "uefa.euro": {2016: {"France"}, 2021: {"England", "Italy", "Germany", "Spain", "Netherlands", "Denmark", "Hungary",
                                          "Scotland", "Russia", "Azerbaijan", "Romania"}, 2024: {"Germany"}},
    "conmebol.america": {2016: {"United States", "USA"}, 2019: {"Brazil"}, 2021: {"Brazil"}, 2024: {"United States", "USA"}},
    "afc.asian.cup": {2019: {"United Arab Emirates"}, 2024: {"Qatar"}},
    "caf.nations": {2017: {"Gabon"}, 2019: {"Egypt"}, 2022: {"Cameroon"}, 2024: {"Ivory Coast", "Cote d'Ivoire"},
                    2025: {"Morocco"}, 2026: {"Morocco"}},
    "concacaf.gold": {2017: {"United States", "USA"}, 2019: {"United States", "USA", "Costa Rica", "Jamaica"},
                      2021: {"United States", "USA"}, 2023: {"United States", "USA", "Canada"}, 2025: {"United States", "USA"}},
}
NATIONS_FINALS = {2019: {"Portugal"}, 2021: {"Italy"}, 2023: {"Netherlands"}, 2025: {"Germany"}}


def _n(s):
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()


def infer_neutral(g: pd.DataFrame) -> pd.Series:
    """Devuelve la columna neutral inferida (bool). Conserva True si ESPN ya lo marcaba."""
    year = g["date"].dt.year
    out = g["neutral"].astype(bool).copy()
    for i, (code, y, m, home) in enumerate(zip(g["code"], year, g["date"].dt.month, g["home"])):
        hosts = None
        if code in HOSTS and y in HOSTS[code]:
            hosts = HOSTS[code][y]
        elif code == "uefa.nations" and m == 6 and y in NATIONS_FINALS:
            hosts = NATIONS_FINALS[y]
        if hosts is not None and _n(home) not in {_n(h) for h in hosts}:
            out.iloc[i] = True
    return out


IDX = {"H": 0, "D": 1, "A": 2}


def _P(r, pre):
    P = r[[f"{pre}_{c}" for c in "HDA"]].to_numpy(float)
    return P / P.sum(1, keepdims=True)


def _ll(r, pre):
    return -np.log(np.clip(_P(r, pre)[np.arange(len(r)), r["res"].map(IDX).to_numpy()], 1e-12, 1))


def _br(r, pre):
    return ((_P(r, pre) - np.eye(3)[r["res"].map(IDX).to_numpy()]) ** 2).sum(1)


def _ece(r, pre, bins=10):
    P = _P(r, pre); conf = P.max(1); hit = (P.argmax(1) == r["res"].map(IDX).to_numpy()).astype(float)
    i = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs(conf[i == b].mean() - hit[i == b].mean()) * (i == b).sum() for b in range(bins) if (i == b).any()) / len(r))


def _m(r, pre):
    return {"n": int(len(r)), "accuracy": round(float((_P(r, pre).argmax(1) == r["res"].map(IDX).to_numpy()).mean()), 4),
            "logloss": round(float(_ll(r, pre).mean()), 5), "brier": round(float(_br(r, pre).mean()), 5), "ece": round(_ece(r, pre), 4)}


def compare3(r, pre):
    from kalshi_nba.backtest import paired_bootstrap
    return {"base": _m(r, "p"), "variante": _m(r, pre),
            "mejora_logloss": paired_bootstrap(_ll(r, "p"), _ll(r, pre), r["date"]),
            "mejora_brier": paired_bootstrap(_br(r, "p"), _br(r, pre), r["date"])}


def main(argv=None):
    from .data import load_group
    from .model import CONFIGS, GROUP_FEATURES, IntlEngine, walk_forward

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/intl")
    ap.add_argument("--output", default="reports/intl_neutral")
    a = ap.parse_args(argv)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    g = load_group("selecciones", cache_dir=a.cache)
    g2 = g.copy(); g2["neutral"] = infer_neutral(g)
    changed = int((g2["neutral"] & ~g["neutral"].astype(bool)).sum())
    print(f"Partidos marcados neutrales por la regla: {changed} (ESPN ya marcaba {int(g['neutral'].sum())})")
    feats = GROUP_FEATURES["selecciones"]
    keep = ["event_id", "date", "temporada", "res", "code", "p_H", "p_D", "p_A"]
    base = walk_forward(IntlEngine(CONFIGS["selecciones"]).process(g), features=feats)[keep]
    var = walk_forward(IntlEngine(CONFIGS["selecciones"]).process(g2), features=feats)[["event_id", "p_H", "p_D", "p_A"]]
    r = base.merge(var.rename(columns={"p_H": "neu_H", "p_D": "neu_D", "p_A": "neu_A"}), on="event_id")
    flag = g2.set_index(g2["event_id"].astype(str))["neutral"] & ~g.set_index(g["event_id"].astype(str))["neutral"].astype(bool)
    r["afectado"] = r["event_id"].astype(str).map(flag).fillna(False)
    res = {"partidos_marcados": changed, "todos": compare3(r, "neu"),
           "solo_afectados": compare3(r[r["afectado"]], "neu") if r["afectado"].sum() > 30 else None}
    t = res["todos"]
    ok = t["mejora_logloss"]["ic95_inf"] > 0 and t["mejora_brier"]["mejora"] >= 0 and t["variante"]["ece"] <= t["base"]["ece"] + 0.005
    res["veredicto"] = "SUPERA_CRITERIO_SOLO_SOMBRA" if ok else "NO_SUPERA_CRITERIO"
    txt = "SELECCIONES — SEDE NEUTRAL INFERIDA (investigación, producción sin cambios)\n" + json.dumps(res, indent=1, ensure_ascii=False)
    (out / "neutral_report.json").write_text(txt.split("\n", 1)[1], encoding="utf8")
    r.to_csv(out / "neutral_oos.csv", index=False)
    print(txt)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf8") as fh:
            fh.write("```\n" + txt + "\n```\n")


if __name__ == "__main__":
    main()
