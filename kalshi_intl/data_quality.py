"""Inspect international historical data quality without guessing venue metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXTRA_TIME_STATUSES = {"STATUS_FINAL_AET", "STATUS_FINAL_PEN"}
REGULATION_STATUSES = {"STATUS_FULL_TIME", "STATUS_FINAL"}


def audit_history(df: pd.DataFrame) -> dict:
    required = {"event_id", "date", "status", "neutral", "final", "code", "hs", "as_"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    d = df.copy()
    dates = pd.to_datetime(d["date"], utc=True, errors="coerce")
    if dates.isna().any():
        raise ValueError("Invalid dates")
    if d.event_id.isna().any() or d.event_id.duplicated().any():
        raise ValueError("Missing or duplicated event_id")
    final = d["final"].astype(bool)
    if d.loc[final, ["hs", "as_"]].isna().any().any():
        raise ValueError("Completed matches missing scores")
    neutral = d["neutral"].fillna(False).astype(bool)
    status_counts = d["status"].value_counts().to_dict()
    by_comp = d.groupby("code", dropna=False).agg(
        matches=("event_id", "count"), neutral_count=("neutral", "sum")
    ).reset_index()
    warnings = []
    if len(d) >= 100 and not neutral.any():
        warnings.append(
            "Neutral venues absent from source for all games; audit ESPN neutralSite "
            "field before treating home advantage as reliable"
        )
    future_final = final & (dates > pd.Timestamp.now(tz="UTC"))
    if future_final.any():
        raise ValueError(f"{int(future_final.sum())} final results dated in the future")
    return {
        "n_matches": int(len(d)),
        "n_final": int(final.sum()),
        "n_regulation_final": int((final & d.status.isin(REGULATION_STATUSES)).sum()),
        "n_extra_time_final_excluded_from_goal_fit": int((final & d.status.isin(EXTRA_TIME_STATUSES)).sum()),
        "n_neutral": int(neutral.sum()),
        "neutral_by_competition": by_comp.fillna("unknown").to_dict(orient="records"),
        "status_counts": {k: int(v) for k, v in status_counts.items()},
        "warnings": warnings,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Historical event CSV with ESPN status and scores")
    ap.add_argument("--output", default="reports/intl_quality/history_quality.json")
    args = ap.parse_args()
    report = audit_history(pd.read_csv(args.input))
    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"Historical data audit: {report['n_matches']} matches, {report['n_neutral']} flagged neutral, "
        f"{report['n_extra_time_final_excluded_from_goal_fit']} extra-time/penalty matches"
    )
    for w in report["warnings"]:
        print("WARNING:", w)
    print("Report:", p)


if __name__ == "__main__":
    main()
