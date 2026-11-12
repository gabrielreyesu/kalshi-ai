"""Descarga de históricos (football-data.co.uk), cuotas del mercado y calendario (football-data.org)."""
from __future__ import annotations

import io
import os
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .engine import LEAGUES, season_code_for_date

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
API_URL = "https://api.football-data.org/v4"
API_CODES = {"PL": "Premier League", "PD": "La Liga", "SA": "Serie A", "BL1": "Bundesliga", "FL1": "Ligue 1"}

# Prioridad: cierre (más difícil de superar) -> promedio -> Bet365. Se usa la primera fuente completa por partido.
H_TRIPLES = [("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"), ("PSH", "PSD", "PSA"),
             ("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A"), ("BbAvH", "BbAvD", "BbAvA")]
OU_PAIRS = [("PC>2.5", "PC<2.5"), ("AvgC>2.5", "AvgC<2.5"), ("P>2.5", "P<2.5"),
            ("Avg>2.5", "Avg<2.5"), ("B365>2.5", "B365<2.5"), ("BbAv>2.5", "BbAv<2.5")]
ODDS_COLS = sorted({c for t in H_TRIPLES for c in t} | {c for t in OU_PAIRS for c in t})


def attach_market(df: pd.DataFrame) -> pd.DataFrame:
    """Probabilidades implícitas del mercado sin margen (normalización proporcional)."""
    n = len(df)
    mk = np.full((n, 3), np.nan)
    ou = np.full(n, np.nan)
    src1, src2 = np.full(n, None, dtype=object), np.full(n, None, dtype=object)
    for cols in H_TRIPLES:
        if not all(c in df.columns for c in cols):
            continue
        o = df[list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(o).all(1) & (o > 1).all(1) & np.isnan(mk[:, 0])
        inv = 1 / o[ok]
        mk[ok] = inv / inv.sum(1, keepdims=True)
        src1[ok] = cols[0]
    for cols in OU_PAIRS:
        if not all(c in df.columns for c in cols):
            continue
        o = df[list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(o).all(1) & (o > 1).all(1) & np.isnan(ou)
        inv = 1 / o[ok]
        ou[ok] = inv[:, 0] / inv.sum(1)
        src2[ok] = cols[0]
    out = df.drop(columns=[c for c in ODDS_COLS if c in df.columns]).copy()
    out["mkt_A"], out["mkt_D"], out["mkt_H"] = mk[:, 2], mk[:, 1], mk[:, 0]   # orden A,D,H como el modelo
    out["mkt_over_2_5"] = ou
    out["mkt_source"] = src1
    out["mkt_ou_source"] = src2
    return out


def _get(url: str, retries: int = 3, timeout: int = 30) -> bytes:
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "kalshi-soccer/1.0"})
            r.raise_for_status()
            return r.content
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"No se pudo descargar {url}: {last}")


def _parse_csv(content: bytes, league: str, season: str) -> pd.DataFrame:
    raw = pd.read_csv(io.BytesIO(content), on_bad_lines="skip", encoding_errors="replace")
    need = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    if not set(need).issubset(raw.columns):
        raise ValueError("Columnas incompletas")
    keep = need + [c for c in ODDS_COLS if c in raw.columns]
    df = raw[keep].rename(columns={"Date": "date", "HomeTeam": "home", "AwayTeam": "away",
                                   "FTHG": "hs", "FTAG": "as_"})
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed", errors="coerce")
    df["competition"], df["season"] = league, season
    return df.dropna(subset=["date", "home", "away", "hs", "as_"])


def current_season_start(today: date | None = None) -> int:
    t = today or date.today()
    return t.year if t.month >= 7 else t.year - 1


def load_history(start_year: int = 2015, cache_dir: str = "data/raw", end_year: int | None = None,
                 verbose: bool = True) -> pd.DataFrame:
    """Histórico de las 5 ligas con cuotas. Temporadas pasadas se cachean; la actual se re-descarga."""
    end_year = end_year or current_season_start()
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    parts, errors = [], []
    for y in range(start_year, end_year + 1):
        season = f"{y % 100:02d}{(y + 1) % 100:02d}"
        for league, code in LEAGUES.items():
            path = cache / f"{season}_{code}.csv"
            try:
                if path.exists() and y < end_year:
                    content = path.read_bytes()
                else:
                    try:
                        content = _get(BASE_URL.format(season=season, code=code))
                        path.write_bytes(content)
                    except Exception:
                        if not path.exists():
                            raise
                        content = path.read_bytes()
                        errors.append((league, season, "descarga falló; se usó caché"))
                parts.append(_parse_csv(content, league, season))
            except Exception as e:  # noqa: BLE001
                errors.append((league, season, str(e)))
    if not parts:
        raise RuntimeError(f"No se descargó ningún partido. Errores: {errors[:5]}")
    df = pd.concat(parts, ignore_index=True)
    df["neutral"] = False
    df = attach_market(df)
    df = df.sort_values(["date", "competition", "home", "away"]).drop_duplicates(
        ["date", "competition", "home", "away"]).reset_index(drop=True)
    if verbose:
        print(f"Partidos: {len(df)} | {df.date.min().date()} -> {df.date.max().date()} | avisos/errores: {len(errors)}")
        for e in errors[:10]:
            print("  ", e)
    return df


def fetch_fixtures(api_key: str, days: int = 14, today: date | None = None) -> pd.DataFrame:
    """Calendario próximo (football-data.org). Solo partidos SCHEDULED/TIMED."""
    today = today or date.today()
    rows = []
    for code, league in API_CODES.items():
        r = requests.get(f"{API_URL}/competitions/{code}/matches", timeout=30,
                         headers={"X-Auth-Token": api_key},
                         params={"dateFrom": today.isoformat(), "dateTo": (today + timedelta(days=days)).isoformat()})
        r.raise_for_status()
        for m in r.json().get("matches", []):
            if m["status"] not in ("SCHEDULED", "TIMED"):
                continue
            rows.append({"match_id": str(m["id"]), "competition": league, "home_api": m["homeTeam"]["name"],
                         "away_api": m["awayTeam"]["name"], "kickoff_utc": m["utcDate"]})
    out = pd.DataFrame(rows, columns=["match_id", "competition", "home_api", "away_api", "kickoff_utc"])
    out["kickoff_utc"] = pd.to_datetime(out["kickoff_utc"], utc=True)
    return out
