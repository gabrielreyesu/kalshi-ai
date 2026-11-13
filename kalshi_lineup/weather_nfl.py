"""NFL: clima a la hora del partido (Open-Meteo, histórico) y prueba pre-registrada (WEATHER_RESEARCH.md).

  python -m kalshi_lineup.weather_nfl            # descarga (caché por estadio) y evalúa
Variables (0 en estadios con techo fijo o retráctil, y en partidos neutrales):
  wind   viento a 10 m (km/h) · rain  precipitación (mm/h) · cold  max(0, 5 − temperatura °C)
Prueba principal: total de puntos. Residuo (total real − línea de cierre) ~ clima, walk-forward por temporada;
éxito si el error cuadrático baja frente a la línea sola con IC95% > 0. Secundaria: ganador (mercado + clima).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .travel import site

URL = "https://archive-api.open-meteo.com/v1/archive"
# Techo fijo o retráctil (equipo local, temporadas): el clima exterior no afecta.
ROOF = {"1": (2017, 2100), "6": (2009, 2100), "8": (2002, 2100), "34": (2002, 2100), "11": (2008, 2100),
        "13": (2020, 2100), "14": (2020, 2100), "24": (2020, 2100), "16": (2016, 2100), "18": (1975, 2100), "22": (2006, 2100)}
WEATHER = ["wind", "rain", "cold"]


def indoor(home: str, season: int) -> bool:
    r = ROOF.get(str(home))
    return bool(r and r[0] <= season <= r[1])


def fetch_site(lat, lon, start, end, cache: Path) -> pd.DataFrame:
    p = cache / f"{lat:.3f}_{lon:.3f}_{start}_{end}.json"
    if not p.exists():
        for i in range(5):
            try:
                r = requests.get(URL, params={"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                                              "hourly": "temperature_2m,precipitation,wind_speed_10m", "timezone": "UTC"}, timeout=90)
            except requests.RequestException:
                time.sleep(5 * (i + 1)); continue
            if r.status_code == 200:
                p.parent.mkdir(parents=True, exist_ok=True); p.write_text(r.text); break
            time.sleep(5 * (i + 1))
        else:
            raise RuntimeError("Open-Meteo no respondió")
    h = json.loads(p.read_text())["hourly"]
    return pd.DataFrame({"t": pd.to_datetime(h["time"], utc=True), "temp": h["temperature_2m"], "rain": h["precipitation"],
                         "wind": h["wind_speed_10m"]}).set_index("t")


def compute(games: pd.DataFrame, cache_dir="data/weather") -> pd.DataFrame:
    g = games[games["status"] == "STATUS_FINAL"].copy()
    g["kick"] = pd.to_datetime(g["kickoff_utc"], utc=True, format="ISO8601").dt.floor("h")
    rows, sites = [], {}
    for x in g.itertuples():
        s = int(x.season_year)
        v = site("nfl", str(x.home_id), s)
        if v is None or bool(getattr(x, "neutral", False)) or indoor(str(x.home_id), s):
            rows.append({"event_id": str(x.event_id), "wind": 0.0, "rain": 0.0, "cold": 0.0, "outdoor": 0}); continue
        sites.setdefault((v[0], v[1]), []).append((str(x.event_id), x.kick))
    for (lat, lon), evs in sites.items():                        # solo el día de cada partido (cupo diario de Open-Meteo)
        for e, k in evs:
            day = k.strftime("%Y-%m-%d"); nxt = (k + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            try:
                w = fetch_site(lat, lon, day, nxt, Path(cache_dir))
            except RuntimeError:
                w = None
            if w is not None and k in w.index:
                r = w.loc[k]
                rows.append({"event_id": e, "wind": float(r["wind"]), "rain": float(r["rain"] or 0), "cold": max(0.0, 5 - float(r["temp"])), "outdoor": 1})
            else:
                rows.append({"event_id": e, "wind": 0.0, "rain": 0.0, "cold": 0.0, "outdoor": 1})
    return pd.DataFrame(rows)


def main():
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from kalshi_nba.backtest import logloss, paired_bootstrap
    from kalshi_nba.data import load_league
    games = load_league("nfl", first_season=2015)
    games["event_id"] = games["event_id"].astype(str)
    w = compute(games)
    d = games.merge(w, on="event_id")
    d = d[d["line_total"].notna()].copy()
    d["total"] = d["hs"] + d["as_"]; d["res"] = d["total"] - d["line_total"]
    out = {"partidos": int(len(d)), "al_aire_libre": int(d["outdoor"].sum())}
    # Totales: residuo ~ clima, walk-forward por temporada (entrenar con temporadas anteriores)
    preds = []
    for s in sorted(d["season_year"].unique()):
        tr, te = d[d["season_year"] < s], d[d["season_year"] == s]
        if s < 2019 or len(tr) < 800:
            continue
        m = LinearRegression().fit(tr[WEATHER], tr["res"])
        preds.append(te.assign(adj=m.predict(te[WEATHER])))
    p = pd.concat(preds)
    e0, e1 = p["res"] ** 2, (p["res"] - p["adj"]) ** 2
    out["totales_todos"] = {"n": int(len(p)), "rmse_linea": round(float(np.sqrt(e0.mean())), 3), "rmse_linea_mas_clima": round(float(np.sqrt(e1.mean())), 3),
                            "mejora_error_cuadratico": paired_bootstrap(e0, e1, p["date"])}
    o = p[p["outdoor"] == 1]
    out["totales_aire_libre"] = {"n": int(len(o)), "mejora_error_cuadratico": paired_bootstrap(o["res"] ** 2, (o["res"] - o["adj"]) ** 2, o["date"])}
    windy = o[o["wind"] >= 25]
    out["viento_fuerte_25kmh"] = {"n": int(len(windy)), "residuo_medio_real": round(float(windy["res"].mean()), 2),
                                  "ajuste_medio_modelo": round(float(windy["adj"].mean()), 2)}
    m_all = LinearRegression().fit(d[WEATHER], d["res"])
    out["coeficientes_puntos_por_unidad"] = dict(zip(WEATHER, [round(float(c), 3) for c in m_all.coef_]))
    # Ganador: logit(mercado) + clima
    lg = lambda q: np.log(np.clip(q, 1e-3, 1 - 1e-3) / (1 - np.clip(q, 1e-3, 1 - 1e-3)))
    dw = d[d["line_p_home"].notna() & (d["hs"] != d["as_"])].copy(); dw["y"] = (dw["hs"] > dw["as_"]).astype(int)
    r0, r1 = [], []
    for s in sorted(dw["season_year"].unique()):
        tr, te = dw[dw["season_year"] < s], dw[dw["season_year"] == s]
        if s < 2019 or len(tr) < 800:
            continue
        X0 = lambda x: lg(x["line_p_home"].to_numpy(float)).reshape(-1, 1)
        X1 = lambda x: np.column_stack([lg(x["line_p_home"].to_numpy(float)), x[WEATHER].to_numpy(float)])
        r0.append(te.assign(p=LogisticRegression().fit(X0(tr), tr["y"]).predict_proba(X0(te))[:, 1]))
        r1.append(te.assign(p=LogisticRegression().fit(X1(tr), tr["y"]).predict_proba(X1(te))[:, 1]))
    a, b = pd.concat(r0), pd.concat(r1)
    out["ganador_mercado_mas_clima"] = paired_bootstrap(logloss(a["p"], a["y"]), logloss(b["p"], b["y"]), a["date"])
    Path("reports/weather").mkdir(parents=True, exist_ok=True)
    Path("reports/weather/nfl_weather.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
