"""Validate durable pre-match research archives without publishing their contents.

This hard-fails on corrupt rows, duplicate snapshot keys, impossible odds,
future information, or invalid weather. Empty archives are permitted (new user).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .core import utc

CLASSES = {"nfl": ("H","A"), "nba": ("H","A"),
           "intl": ("H","D","A"), "club": ("H","D","A")}
MIN_BEFORE_KICKOFF = pd.Timedelta(minutes=15)


def _load(path):
    path = Path(path)
    if not path.exists():
        return []
    result = []
    with path.open(encoding="utf8") as stream:
        for n, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON row {n} of {path.name}") from err
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at row {n} of {path.name}")
            result.append(obj)
    return result


def _snapshot_key(r):
    required = ("source_id", "sport", "event_id", "captured_utc")
    if not all(r.get(k) for k in required):
        raise ValueError("Missing forecast snapshot identifiers")
    return tuple(str(r[k]) for k in required)


def _times(r):
    needed = ("kickoff_utc", "captured_utc", "published_utc")
    if not all(r.get(key) for key in needed):
        raise ValueError("Missing source timestamps")
    kick, captured, issued = (utc(r[k]) for k in needed)
    if issued > captured:
        raise ValueError("Forecast publication later than archive capture")
    if captured >= kick - MIN_BEFORE_KICKOFF:
        raise ValueError("Forecast recorded too close to or after kickoff")
    return kick, captured, issued


def audit_snapshot_archive(odds, weather):
    """Use actual unique event identifiers rather than treating snapshots as games."""
    if not isinstance(odds, list) or not isinstance(weather, list):
        raise ValueError("Archives must be lists")
    seen = set()
    sports = Counter()
    sources = Counter()
    unique_events = set()
    first = last = None
    for kind, rows in (("market", odds), ("weather", weather)):
        for r in rows:
            if not isinstance(r, dict):
                raise ValueError("Snapshot must be a JSON object")
            key = _snapshot_key(r)
            if key in seen:
                raise ValueError("Duplicate captured event/source timestamp")
            seen.add(key)
            kick, cap, _ = _times(r)
            event_id, sport = str(r["event_id"]), str(r["sport"])
            unique_events.add((sport, event_id))
            sports[sport] += 1
            sources[str(r["source_id"])] += 1
            first = cap if first is None or cap < first else first
            last = cap if last is None or cap > last else last
            if kind == "market":
                if sport not in CLASSES:
                    raise ValueError("Unexpected sport in bookmaker archive")
                labels = CLASSES[sport]
                values = np.asarray([r.get("p_"+c) for c in labels], dtype=float)
                baseline = np.asarray([r.get("model_p_"+c) for c in labels], dtype=float)
                for name, probs in (("bookmaker", values), ("baseline", baseline)):
                    if (not np.isfinite(probs).all() or (probs < 0).any()
                        or not np.isclose(probs.sum(), 1.0, atol=1e-5)):
                        raise ValueError(f"Invalid {name} probabilities for {sport} event {event_id}")
                if r.get("source_kind") not in ("market","model","analyst","research_only"):
                    raise ValueError("Unknown odds source_kind")
            else:
                if sport != "nfl" or r.get("source_kind") != "environment":
                    raise ValueError("Unexpected weather sport/source")
                for field, lo, hi in (
                    ("temperature_f",-125,140),
                    ("wind_speed_max_mph",0,250),
                    ("precip_probability_pct",0,100),
                ):
                    v = r.get(field)
                    if v is None:
                        continue
                    try:
                        value = float(v)
                    except (TypeError, ValueError) as err:
                        raise ValueError(f"Invalid weather {field}") from err
                    if not np.isfinite(value) or not lo <= value <= hi:
                        raise ValueError(f"Invalid weather {field}")
    return {
        "archive_status": "VALID_NO_MODEL_CHANGE",
        "market_snapshots": len(odds),
        "weather_snapshots": len(weather),
        "unique_sport_event_pairs": len(unique_events),
        "sources": dict(sorted(sources.items())),
        "sport_snapshots": dict(sorted(sports.items())),
        "first_snapshot_utc": first.isoformat() if first is not None else None,
        "latest_snapshot_utc": last.isoformat() if last is not None else None,
        "rules": "All timestamps pre-kickoff, outcomes never inferred, probability rows normalized",
        "production_changed": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market",default="data/external_signals/history.jsonl")
    ap.add_argument("--weather",default="data/external_signals/weather.jsonl")
    ap.add_argument("--output",default="reports/external_learning/archive_health.json")
    a = ap.parse_args()
    result = audit_snapshot_archive(_load(a.market),_load(a.weather))
    path = Path(a.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print(json.dumps(result,ensure_ascii=False))


if __name__ == "__main__":
    main()
