"""Prospective paired market-source diagnostics: no model training or promotion.

Scores sources only against the model's recorded PRE-MATCH probabilities
for the SAME event, and keeps all the intraday market snapshots for auditing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OPTIONS = {"intl": ("H", "D", "A"), "club": ("H", "D", "A"),
           "nfl": ("H", "A"), "nba": ("H", "A")}


def _three_way(sport):
    return sport in ("intl", "club")


def diagnostics(history, outcomes, min_resolved=100, bootstrap=500, seed=14):
    """Return paired metrics and intraday-change summary, with date-cluster CI.

    Lower source logloss/Brier are better. For delta columns, positive means
    a source beat the matching original model, not that the source is proven.
    Each source + sport + fixture contributes at most one LAST pre-match
    observation to outcome scoring; all snapshots remain in movement stats.
    """
    if bootstrap < 0 or min_resolved < 1:
        raise ValueError("Invalid diagnostic thresholds")
    paired_cols = ["source_id", "sport", "source_kind", "n_events", "n_dates",
                   "accuracy_source", "accuracy_model", "logloss_source",
                   "logloss_model", "gain_logloss", "brier_source", "brier_model",
                   "gain_brier", "ci95_gain_low", "ci95_gain_high",
                   "sufficient_history", "candidate_research_only"]
    movement_cols = ["source_id", "sport", "events", "snapshots",
                     "events_with_multiple_snapshots", "mean_open_close_abs_p_home",
                     "latest_capture_utc"]
    empty_metrics = pd.DataFrame(columns=paired_cols)
    empty_movements = pd.DataFrame(columns=movement_cols)
    if not history:
        return empty_metrics, empty_movements

    f = pd.DataFrame(history)
    required = {"event_id", "sport", "source_id", "source_kind",
                "captured_utc", "published_utc", "kickoff_utc",
                "p_H", "p_D", "p_A", "model_p_H", "model_p_D", "model_p_A"}
    if not required.issubset(f):
        raise ValueError("Missing forecast metadata: " + repr(sorted(required-set(f))))
    if not {"sport", "event_id", "result"}.issubset(outcomes):
        raise ValueError("Outcomes must have sport,event_id,result")
    f = f[f["sport"].isin(OPTIONS)].copy()
    if f.empty:
        return empty_metrics, empty_movements

    for c in ("captured_utc", "published_utc", "kickoff_utc"):
        f[c] = pd.to_datetime(f[c], utc=True, errors="coerce")
    if f[["captured_utc", "published_utc", "kickoff_utc"]].isna().any().any():
        raise ValueError("Invalid source timestamps")
    f = f[(f.published_utc <= f.captured_utc) &
          (f.captured_utc < f.kickoff_utc - pd.Timedelta(minutes=15))]
    if f.empty:
        return empty_metrics, empty_movements

    f["event_id"] = f.event_id.astype(str)
    keys = ["sport", "event_id", "source_id"]
    f = f.sort_values("captured_utc").drop_duplicates(keys + ["captured_utc"])
    movements = []
    for (source, sport), d in f.groupby(["source_id", "sport"]):
        counts = d.groupby(keys).size()
        open_p = d.groupby(keys).first()["p_H"].astype(float)
        close_p = d.groupby(keys).last()["p_H"].astype(float)
        change = (close_p - open_p).abs().to_numpy(float)
        movements.append({
            "source_id": source, "sport": sport,
            "events": int(len(counts)), "snapshots": int(len(d)),
            "events_with_multiple_snapshots": int((counts >= 2).sum()),
            "mean_open_close_abs_p_home": float(np.nanmean(change)),
            "latest_capture_utc": d.captured_utc.max().isoformat(),
        })
    movement_df = pd.DataFrame(movements, columns=movement_cols)
    last = f.drop_duplicates(keys, keep="last")

    o = outcomes[["sport", "event_id", "result"]].copy()
    o["event_id"] = o.event_id.astype(str)
    if o.duplicated(["sport", "event_id"]).any():
        raise ValueError("Duplicate completed outcomes")
    settled = last.merge(o, on=["sport", "event_id"], how="inner", validate="many_to_one")
    results = []
    for (source, sport), d in settled.groupby(["source_id", "sport"]):
        labels = OPTIONS[sport]
        y = d.result.map({name:i for i,name in enumerate(labels)})
        if y.isna().any():
            continue
        y = y.to_numpy(dtype=int)
        p = d[["p_"+n for n in labels]].to_numpy(dtype=float, copy=True)
        model = d[["model_p_"+n for n in labels]].to_numpy(dtype=float, copy=True)
        if (not np.isfinite(p).all() or not np.isfinite(model).all() or
            (p < 0).any() or (model < 0).any() or
            (p.sum(axis=1) <= 0).any() or (model.sum(axis=1) <= 0).any()):
            continue
        p /= p.sum(axis=1, keepdims=True)
        model /= model.sum(axis=1, keepdims=True)
        idx = np.arange(len(y))
        source_loss = -np.log(np.clip(p[idx,y],1e-12,1))
        model_loss = -np.log(np.clip(model[idx,y],1e-12,1))
        diff = model_loss - source_loss
        target = np.eye(len(labels))[y]
        brier_s = ((p-target)**2).sum(axis=1)
        brier_m = ((model-target)**2).sum(axis=1)
        dates = d.kickoff_utc.dt.normalize().to_numpy()
        groups = [np.flatnonzero(dates == date) for date in np.unique(dates)]
        low = high = np.nan
        if bootstrap and len(groups) > 1:
            rng = np.random.default_rng(seed)
            samples = np.empty(bootstrap)
            for i in range(bootstrap):
                selected = rng.integers(0, len(groups), size=len(groups))
                ix = np.concatenate([groups[j] for j in selected])
                samples[i] = diff[ix].mean()
            low, high = np.percentile(samples, [2.5, 97.5])
        sufficient = len(y) >= min_resolved
        # An interval can only be considered positive with enough data AND a
        # reproducible uncertainty estimate; still NEVER auto-promote.
        research = bool(sufficient and np.isfinite(low) and low > 0)
        results.append({
            "source_id": source, "sport": sport,
            "source_kind": str(d.source_kind.iloc[0]),
            "n_events": int(len(y)), "n_dates": int(len(groups)),
            "accuracy_source": float((p.argmax(axis=1) == y).mean()),
            "accuracy_model": float((model.argmax(axis=1) == y).mean()),
            "logloss_source": float(source_loss.mean()),
            "logloss_model": float(model_loss.mean()),
            "gain_logloss": float(diff.mean()),
            "brier_source": float(brier_s.mean()),
            "brier_model": float(brier_m.mean()),
            "gain_brier": float((brier_m - brier_s).mean()),
            "ci95_gain_low": float(low), "ci95_gain_high": float(high),
            "sufficient_history": bool(sufficient),
            "candidate_research_only": research,
        })
    return pd.DataFrame(results, columns=paired_cols), movement_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="data/external_signals/history.jsonl")
    parser.add_argument("--outcomes", required=True, help="CSV: sport,event_id,result")
    parser.add_argument("--output", default="reports/external_learning")
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    rows = []
    p = Path(args.history)
    if p.exists():
        with p.open(encoding="utf8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    metrics, movement = diagnostics(rows, pd.read_csv(args.outcomes, dtype={"event_id":str}), bootstrap=args.bootstrap)
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(dest/"paired_source_metrics.csv", index=False)
    movement.to_csv(dest/"market_movement.csv", index=False)
    print(f"Reference metrics: {len(metrics)} sources with matched outcomes; {len(movement)} movement summaries")


if __name__ == "__main__":
    main()
