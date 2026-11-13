"""NFL: distribución REAL del margen alrededor de la línea de las casas (números clave 3, 7, 10…).

Problema: P(local gana por más de x) se calculaba con una curva normal (σ = 12,7) centrada en −spread, que ignora que
en la NFL muchos partidos terminan por 3 o 7 puntos. Aquí se usa la distribución empírica del residuo
(margen real − (−spread)) en temporadas anteriores, por tramo de |spread| (variante B) o sin tramos (variante A).
Protocolo pre-registrado en NFL_MARGINS.md.

  python -m kalshi_nba.margins            # valida (walk-forward) y guarda kalshi_nba/nfl_margin_table.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

TABLE = Path(__file__).with_name("nfl_margin_table.json")
BUCKETS = [(0, 3.25), (3.25, 7.25), (7.25, 99)]
SIGMA = 12.7


def _bucket(spread_abs: float) -> int:
    return next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= spread_abs < hi)


class EmpiricalMargin:
    """Misma interfaz que NBAModels/_BookPricer para spreads: p_home_wins_by_more_than(mu_margin, x)."""

    def __init__(self, residuals: dict, sig_t: float = 13.0):
        # residuals: {"all": [...], "0": [...], "1": [...], "2": [...]} (enteros/medios puntos)
        self.res = {k: np.sort(np.asarray(v, float)) for k, v in residuals.items()}
        self.sig_m, self.sig_t = SIGMA, sig_t

    def _surv(self, key, t):
        r = self.res[key] if len(self.res.get(key, [])) >= 150 else self.res["all"]
        # P(residuo > t) con media corrección de continuidad para empates exactos en t
        return (len(r) - np.searchsorted(r, t, side="right") + 0.5 * (np.searchsorted(r, t, side="right") - np.searchsorted(r, t, side="left"))) / len(r)

    def p_home_wins_by_more_than(self, mu_margin, x, bucketed=True):
        mu, x = np.broadcast_arrays(np.asarray(mu_margin, float), np.asarray(x, float))
        out = np.empty(mu.shape)
        for i in np.ndindex(mu.shape):
            key = str(_bucket(abs(mu[i]))) if bucketed else "all"
            out[i] = self._surv(key, x[i] - mu[i])
        return np.clip(out, 0.005, 0.995)

    def p_total_over(self, mu_total, line):
        return 1 - norm.cdf((np.asarray(line, float) - np.asarray(mu_total, float)) / self.sig_t)

    @classmethod
    def fit(cls, games: pd.DataFrame) -> "EmpiricalMargin":
        d = games.dropna(subset=["line_spread", "hs", "as_"])
        resid = (d["hs"] - d["as_"]) - (-d["line_spread"])
        out = {"all": resid.tolist()}
        for i in range(len(BUCKETS)):
            out[str(i)] = resid[d["line_spread"].abs().map(_bucket) == i].tolist()
        return cls(out)


def load() -> EmpiricalMargin | None:
    if not TABLE.exists():
        return None
    return EmpiricalMargin(json.loads(TABLE.read_text())["residuos"])


def main():
    from kalshi_nba.backtest import logloss, paired_bootstrap
    from kalshi_nba.data import load_league
    g = load_league("nfl", first_season=2015)
    g = g[(g["status"] == "STATUS_FINAL") & g["line_spread"].notna()].copy()
    g["margin"] = g["hs"] - g["as_"]
    lines = np.arange(-14.5, 15, 1.0)                       # desplazamientos alrededor de la línea
    rows = []
    for s in sorted(g["season_year"].unique()):
        tr, te = g[g["season_year"] < s], g[g["season_year"] == s]
        if s < 2019 or len(tr) < 800:
            continue
        em = EmpiricalMargin.fit(tr)
        for x in te.itertuples():
            mu = -x.line_spread
            ks = np.floor(mu) + 0.5 + np.arange(-15, 15)          # siempre líneas x,5 (como Kalshi): sin empates
            y = (x.margin > ks).astype(int)
            pn = 1 - norm.cdf((ks - mu) / SIGMA)
            pa = em.p_home_wins_by_more_than(np.full(len(ks), mu), ks, bucketed=False)
            pb = em.p_home_wins_by_more_than(np.full(len(ks), mu), ks, bucketed=True)
            rows.append({"date": x.date, "season": s, "ll_normal": float(logloss(pn, y).mean()),
                         "ll_A": float(logloss(pa, y).mean()), "ll_B": float(logloss(pb, y).mean()),
                         "ll_normal_key": float(logloss(pn[np.abs(ks - mu) <= 7.5], y[np.abs(ks - mu) <= 7.5]).mean()),
                         "ll_B_key": float(logloss(pb[np.abs(ks - mu) <= 7.5], y[np.abs(ks - mu) <= 7.5]).mean())})
    r = pd.DataFrame(rows)
    dev, hold = r[r["season"] <= 2022], r[r["season"] > 2022]
    res = {"partidos_validacion": int(len(r)),
           "desarrollo": {v: paired_bootstrap(dev["ll_normal"], dev[f"ll_{v}"], dev["date"]) for v in ("A", "B")}}
    chosen = max(("A", "B"), key=lambda v: res["desarrollo"][v]["mejora"])
    res["variante_elegida"] = chosen
    res["confirmacion"] = {"n": int(len(hold)), "logloss_normal": round(float(hold["ll_normal"].mean()), 5),
                           "logloss_empirico": round(float(hold[f"ll_{chosen}"].mean()), 5),
                           "mejora": paired_bootstrap(hold["ll_normal"], hold[f"ll_{chosen}"], hold["date"]),
                           "lineas_cercanas_±7.5": paired_bootstrap(hold["ll_normal_key"], hold["ll_B_key"], hold["date"])}
    ok = res["confirmacion"]["mejora"]["ic95_inf"] > 0
    res["veredicto"] = "SUPERA_CRITERIO" if ok else "NO_SUPERA_CRITERIO"
    if ok:
        em = EmpiricalMargin.fit(g)
        TABLE.write_text(json.dumps({"variante": chosen, "temporadas": [int(g["season_year"].min()), int(g["season_year"].max())],
                                     "residuos": {k: v.tolist() for k, v in em.res.items()}, "validacion": res}, ensure_ascii=False) + "\n")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
