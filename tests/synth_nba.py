"""NBA sintética con fuerza real conocida y líneas = verdad + ruido (para probar el pipeline sin internet)."""
import numpy as np
import pandas as pd


def make_nba(signal=True, seasons=range(2016, 2023), teams=30, seed=0, days=140, per_day=7):
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 4.0, teams) if signal else np.zeros(teams)
    pace = rng.normal(0, 4.0, teams) if signal else np.zeros(teams)
    rows, eid = [], 0
    for y in seasons:
        strength = strength + (rng.normal(0, 1.0, teams) if signal else 0)       # la fuerza deriva entre temporadas
        d0 = pd.Timestamp(f"{y - 1}-10-22")
        for k in range(days):
            order = rng.permutation(teams)
            date = d0 + pd.Timedelta(days=int(k * 1.6))
            for i in range(per_day):
                h, a = int(order[2 * i]), int(order[2 * i + 1])
                mu_m = strength[h] - strength[a] + 2.5
                mu_t = 224 + pace[h] + pace[a]
                m, t = mu_m + rng.normal(0, 12), mu_t + rng.normal(0, 17)
                hs, as_ = round((t + m) / 2), round((t - m) / 2)
                if hs == as_:
                    hs += 1
                eid += 1
                rows.append(dict(event_id=str(eid), date=date, kickoff_utc=date.isoformat(), season_year=y, season_type=2,
                                 status="STATUS_FINAL", home_id=f"T{h}", away_id=f"T{a}", home=f"T{h}", away=f"T{a}",
                                 hs=float(hs), as_=float(as_), neutral=False,
                                 line_spread=-round((mu_m + rng.normal(0, 1.5)) * 2) / 2,
                                 line_total=round((mu_t + rng.normal(0, 2.0)) * 2) / 2,
                                 line_p_home=np.nan))
    return pd.DataFrame(rows)
