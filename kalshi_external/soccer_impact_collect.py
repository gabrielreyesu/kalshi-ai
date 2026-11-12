"""Archive soccer starter/replacement research with as-of win-model baselines.

The role and cover-gap fields are measured exposure proxies, not estimated
causal scoring effects. Archive only in the private DATA_REPO_TOKEN repository.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd
from .cli import append_history
from .core import utc
from .soccer_starter_impact import research_snapshot, build_role_context
from .soccer_confirmed_lineups import match_lineup_to_espn_roster


def read_rows(path):
    file=Path(path)
    if not file.exists():
        return []
    results=[]
    for line in file.read_text(encoding="utf8").splitlines():
        if line.strip():
            record=json.loads(line)
            if not isinstance(record,dict):
                raise ValueError("Invalid player-research archive row")
            results.append(record)
    return results


def baseline_asof(registry, preview):
    """Require same ESPN event/code and probability creation <= observation."""
    if registry is None or registry.empty:
        return None
    required={"event_id","code","created_utc","kickoff_utc","p_H","p_D","p_A","grupo"}
    if not required.issubset(registry):
        return None
    rows=registry[(registry.event_id.astype(str)==str(preview["event_id"])) &
                  (registry.code.astype(str)==str(preview["competition_code"]))]
    observed=utc(preview["captured_utc"])
    candidates=[]
    for r in rows.to_dict("records"):
        try:
            created=utc(r["created_utc"])
            original_kickoff=utc(r["kickoff_utc"])
            if (created>observed or created>=original_kickoff-pd.Timedelta(minutes=15)
                or abs(original_kickoff-utc(preview["kickoff_utc"]))>pd.Timedelta(minutes=60)):
                continue
            probabilities=[float(r[x]) for x in ("p_H","p_D","p_A")]
            if any(not 0<p<1 for p in probabilities) or abs(sum(probabilities)-1)>0.02:
                continue
            if str(r["grupo"]) not in ("clubes","selecciones"):
                continue
            candidates.append((created,{
                "p_H":probabilities[0],"p_D":probabilities[1],"p_A":probabilities[2],
                "model_created_utc":created.isoformat(),
                "model_group":"club" if str(r["grupo"])=="clubes" else "intl",
            }))
        except (ValueError,TypeError,KeyError):
            continue
    return max(candidates,key=lambda t:t[0])[1] if candidates else None


def build(previews, history, lineups, registry, as_of):
    now=utc(as_of)
    latest={}
    for r in previews:
        if r.get("source_id")!="soccer_player_opponent_preview":
            continue
        try:
            seen,kick=utc(r["captured_utc"]),utc(r["kickoff_utc"])
        except (ValueError,KeyError,TypeError):
            continue
        if not seen<=now or seen>=kick-pd.Timedelta(minutes=15):
            continue
        if kick<now+pd.Timedelta(minutes=15) or kick>now+pd.Timedelta(hours=48):
            continue
        key=(str(r["competition_code"]),str(r["event_id"]))
        if key not in latest or seen>utc(latest[key]["captured_utc"]):
            latest[key]=r
    output=[]
    for preview in latest.values():
        matching=[
            x for x in lineups
            if x.get("source_id")=="api_football_pregame_confirmed_XI"
            and str(x.get("competition_code"))==str(preview["competition_code"])
            and str(x.get("event_id"))==str(preview["event_id"])
            and utc(x["captured_utc"])<=utc(preview["captured_utc"])
        ]
        lineup=None
        if matching:
            observed=max(matching,key=lambda x:utc(x["captured_utc"]))
            roster=build_role_context(preview,history,preview["captured_utc"])
            lineup=match_lineup_to_espn_roster(observed,roster,preview["captured_utc"])
        impact=research_snapshot(preview,history,confirmed_lineup=lineup)
        baseline=baseline_asof(registry,preview)
        impact["model_baseline_asof"]=baseline
        impact["eligible_for_matched_1x2_evaluation"]=bool(
            baseline is not None and
            impact["home_minus_away_absence_exposure"] is not None
        )
        output.append(impact)
    return sorted(output,key=lambda x:(x["kickoff_utc"],x["event_id"]))


def run(archive="data/external_signals",
        registry="predictions/intl_registry.csv",
        output="reports/external_learning/soccer_impact/coverage.json",
        observed_at=None):
    now=utc(observed_at or pd.Timestamp.now(tz="UTC"))
    path=Path(archive)
    file=Path(registry)
    reg=pd.read_csv(file,dtype={"event_id":str}) if file.exists() else pd.DataFrame()
    impacts=build(
        read_rows(path/"soccer_previews.jsonl"),
        read_rows(path/"soccer_player_games.jsonl"),
        read_rows(path/"soccer_confirmed_lineups.jsonl"),
        reg,now
    )
    _,new=append_history(path/"soccer_starter_impact.jsonl",impacts,max_bytes=100_000_000)
    result={
        "captured_utc":now.isoformat(),
        "new_player_importance_snapshots":new,
        "future_soccer_games_examined":len(impacts),
        "team_contexts_with_4_plus_prior_matches":sum(
            1 for x in impacts for t in x["teams"] if t["team_history_games"]>=4),
        "important_absences_with_role_index":sum(
            1 for x in impacts for t in x["teams"]
            for p in t["missing_starter_replacement_research"]
            if p["player_role_index"] is not None),
        "potential_replacement_lists":sum(
            1 for x in impacts for t in x["teams"]
            for p in t["missing_starter_replacement_research"]
            if p["replacement_candidates"]),
        "verified_pregame_XI_games":sum(x["lineup_confirmed"] for x in impacts),
        "eligible_for_prospective_model_test":sum(
            x["eligible_for_matched_1x2_evaluation"] for x in impacts),
        "status":"RESEARCH_ONLY_NO_CAUSAL_WIN_ADJUSTMENTS",
        "warning":"Unknown athlete identities/minutes/official lineups are left unknown. No model changed.",
    }
    target=Path(output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(json.dumps(result))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--archive",default="data/external_signals")
    parser.add_argument("--registry",default="predictions/intl_registry.csv")
    parser.add_argument("--output",default="reports/external_learning/soccer_impact/coverage.json")
    args=parser.parse_args()
    run(args.archive,args.registry,args.output)


if __name__=="__main__":
    main()
