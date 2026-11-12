"""Datos sintéticos de fútbol con fuerza real conocida (para probar el pipeline sin internet)."""
import numpy as np
import pandas as pd
from scipy.stats import poisson


def make_league(league, season_starts, teams=16, seed=0, signal=True, with_odds=True, home_adv=0.25):
    rng = np.random.default_rng(seed)
    names = [f"{league[:3]}_T{i}" for i in range(teams)]
    att = rng.normal(0, 0.25, teams) if signal else np.zeros(teams)
    dfn = rng.normal(0, 0.25, teams) if signal else np.zeros(teams)
    rows = []
    for y in season_starts:
        season = f"{y % 100:02d}{(y + 1) % 100:02d}"
        d0 = pd.Timestamp(f"{y}-08-10")
        fixtures = [(i, j) for i in range(teams) for j in range(teams) if i != j]
        rng.shuffle(fixtures)
        per_day = teams // 2
        used_days = {}
        day = 0
        queue = list(fixtures)
        while queue:
            busy, today, rest = set(), [], []
            for (i, j) in queue:
                if i in busy or j in busy or len(today) >= per_day:
                    rest.append((i, j)); continue
                busy.update([i, j]); today.append((i, j))
            for (i, j) in today:
                lh = np.exp(0.28 + home_adv + att[i] - dfn[j])
                la = np.exp(0.28 + att[j] - dfn[i])
                hg, ag = rng.poisson(lh), rng.poisson(la)
                row = dict(date=d0 + pd.Timedelta(days=7 * day), competition=league, season=season,
                           home=names[i], away=names[j], hs=hg, as_=ag, neutral=False)
                if with_odds:   # "mercado" = probabilidades verdaderas con 5% de margen
                    g = np.arange(13)
                    m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
                    ph, pd_, pa = np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()
                    tot = g[:, None] + g[None, :]
                    po = m[tot > 2.5].sum()
                    row.update(PSCH=1 / (ph * 1.05), PSCD=1 / (pd_ * 1.05), PSCA=1 / (pa * 1.05),
                               **{"PC>2.5": 1 / (po * 1.05), "PC<2.5": 1 / ((1 - po) * 1.05)})
                rows.append(row)
            queue = rest
            day += 1
    return pd.DataFrame(rows)


def make_history(signal=True, seasons=range(2015, 2022), leagues=("Premier League", "La Liga"), with_odds=True):
    from kalshi_soccer.data import attach_market
    parts = [make_league(l, list(seasons), seed=i + 1, signal=signal, with_odds=with_odds)
             for i, l in enumerate(leagues)]
    df = pd.concat(parts, ignore_index=True)
    # odds crudas -> mkt_* (mismo camino que con datos reales)
    return attach_market(df).sort_values(["date", "competition", "home"]).reset_index(drop=True)
