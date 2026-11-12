"""Soccer starter relevance, missing-player exposure, and possible cover.

RESEARCH ONLY: every number is a descriptive pregame role/availability proxy,
not a causal win-probability change. No assumed lineup, no cross-provider IDs.
"""
from __future__ import annotations

from collections import defaultdict
from math import isfinite

import pandas as pd

from .core import utc
from .soccer_player_intel import canonical_name

METRIC_MINUTES = "minutes"
MIN_TEAM_GAMES = 4
MIN_PLAYER_MINUTE_GAMES = 3
MAX_TEAM_GAMES = 15


def _historical_roster(fixture, games, observed):
    """Only fully recorded, prior, same-competition team player games."""
    now = utc(observed)
    cutoff = utc(fixture["kickoff_utc"])
    code = str(fixture["competition_code"])
    team_ids = {str(fixture["home_id"]), str(fixture["away_id"])}
    team_games = defaultdict(dict)
    for game in games:
        if (game.get("source_id") != "espn_soccer_final_player_stats"
                or str(game.get("competition_code")) != code):
            continue
        captured = utc(game["captured_utc"])
        kick = utc(game["kickoff_utc"])
        if captured > now or kick >= cutoff or kick >= captured:
            continue
        for team in game.get("teams") or []:
            tid = str(team.get("team_id") or "")
            if tid not in team_ids:
                continue
            team_games[tid][str(game["event_id"])] = (kick, team.get("members") or [])
    roster = {}
    for team_id in team_ids:
        chronological = sorted(team_games[team_id].values(), key=lambda x:x[0])[-MAX_TEAM_GAMES:]
        team_n = len(chronological)
        by_id = defaultdict(list)
        for kick, players in chronological:
            for player in players:
                pid = str(player.get("player_id") or "")
                if not pid:
                    continue
                by_id[pid].append((kick, player))
        members = {}
        for pid, appearances in by_id.items():
            appearances.sort(key=lambda x:x[0])
            name = str(appearances[-1][1].get("player") or "")
            position = str(appearances[-1][1].get("position") or "").upper()
            minute_values = []
            starts_observed = []
            for _, player in appearances:
                metrics = player.get("metrics") or {}
                minutes = metrics.get(METRIC_MINUTES)
                if isinstance(minutes, (int, float)) and isfinite(minutes) and 0 <= minutes <= 130:
                    minute_values.append(float(minutes))
                # ESPN missing starter flags are UNOBSERVED, not "did not start".
                if player.get("starter_reported") is True:
                    starts_observed.append(True)
            enough = team_n >= MIN_TEAM_GAMES and len(minute_values) >= MIN_PLAYER_MINUTE_GAMES
            minutes_mean = sum(minute_values) / len(minute_values) if minute_values else None
            participation = len(appearances) / team_n if team_n else 0.0
            # Reliance is a share of possible 90-min exposure, not points/odds.
            role_index = round(100 * min(1.0, minutes_mean / 90) * participation, 2) if enough else None
            members[pid] = {
                "espn_player_id":pid, "player":name, "position":position,
                "prior_team_games":team_n, "player_appearances":len(appearances),
                "minute_sample":len(minute_values), "avg_minutes_when_available":
                    round(minutes_mean,2) if minutes_mean is not None else None,
                "participation_fraction":round(participation,3),
                "known_starts":len(starts_observed),
                "role_index_0_100":role_index,
                "role_evaluable":enough,
            }
        roster[team_id]=members
    return roster


def build_role_context(fixture, games, as_of=None):
    now = utc(as_of or fixture["captured_utc"])
    kickoff = utc(fixture["kickoff_utc"])
    if now >= kickoff or utc(fixture["captured_utc"])>now:
        raise ValueError("Player relevance requires a genuine pre-match snapshot")
    return _historical_roster(fixture,games,now)


def _team_injury_match(injury, team_id, roster):
    """Match vendor-named athlete to a single ESPN athlete in the EXACT team.

    Missing verified team identity or ambiguous normalized player name => no
    numeric relevance attached to the injury.
    """
    if str(injury.get("espn_team_id") or "") != team_id:
        return None, "team_not_linked"
    name = canonical_name(injury.get("player"))
    if not name:
        return None, "name_missing"
    matching=[r for r in roster.values() if canonical_name(r["player"])==name]
    if len(matching)!=1:
        return None, "ambiguous_or_unknown_player"
    return matching[0], "unique_team_name_match_not_cross_provider_id"


def _same_role_candidates(roster, absent, explicit_missing, lineup_ids):
    """Candidates are suggestions unless a pre-match confirmed XI is present."""
    target = str(absent.get("position") or "")
    if not target:
        return []
    results=[]
    for candidate in roster.values():
        if candidate["espn_player_id"]==absent["espn_player_id"]:
            continue
        if candidate["espn_player_id"] in explicit_missing:
            continue
        if candidate["position"] != target or candidate["role_index_0_100"] is None:
            continue
        in_starting_xi = bool(lineup_ids is not None and candidate["espn_player_id"] in lineup_ids)
        if lineup_ids is not None and not in_starting_xi:
            continue
        results.append({
            "espn_player_id":candidate["espn_player_id"],
            "player":candidate["player"],
            "position":candidate["position"],
            "role_index_0_100":candidate["role_index_0_100"],
            "evidence":"observed_pregame_XI" if lineup_ids is not None else "historical_role_only_not_confirmed",
        })
    return sorted(results,key=lambda r:r["role_index_0_100"],reverse=True)[:3]


def research_snapshot(preview, games, confirmed_lineup=None):
    """One prospective research sample. No probabilities or causal adjustments.

    confirmed_lineup, if given, is a provider-verified *pre-kickoff* snapshot
    with ESPN player IDs matched only via unique exact names within teams.
    """
    as_of=utc(preview["captured_utc"])
    if as_of >= utc(preview["kickoff_utc"]):
        raise ValueError("Context recorded after kickoff")
    fixture={
        "competition_code":preview["competition_code"],
        "home_id":preview["home_id"],"away_id":preview["away_id"],
        "kickoff_utc":preview["kickoff_utc"],"captured_utc":preview["captured_utc"]
    }
    roster=build_role_context(fixture,games,as_of)
    known_lineup = confirmed_lineup if isinstance(confirmed_lineup,dict) and (
        confirmed_lineup.get("lineup_confirmed") is True and
        confirmed_lineup.get("event_id")==preview.get("event_id") and
        utc(confirmed_lineup.get("captured_utc"))<=as_of
    ) else None
    teams=[]
    for team_id in [str(preview["home_id"]),str(preview["away_id"])]:
        members=roster.get(team_id,{})
        unresolved=[]
        absences=[]
        matched_absent_ids=set()
        for injury in preview.get("provider_injury_context") or []:
            if str(injury.get("espn_team_id") or "")!=team_id:
                continue
            if injury.get("availability") not in ("reported_absent","uncertain"):
                continue
            person,match_reason=_team_injury_match(injury,team_id,members)
            item={
                "player":str(injury.get("player") or ""),
                "provider_status":injury.get("availability"),
                "provider_report_type":str(injury.get("type") or ""),
                "matching_confidence":match_reason,
                "role_index_0_100":person.get("role_index_0_100") if person else None,
                "position":person.get("position") if person else None,
                "espn_player_id":person.get("espn_player_id") if person else None,
            }
            if not person:
                unresolved.append(item)
            else:
                if injury["availability"]=="reported_absent":
                    matched_absent_ids.add(person["espn_player_id"])
                absences.append(item)
        lineup_ids=None
        if known_lineup is not None:
            lineup_ids=set(str(v) for v in known_lineup.get("starter_ids_by_espn_team",{}).get(team_id,[]))
        missing_roles=[]
        for item in absences:
            if item["provider_status"]!="reported_absent":
                continue
            person=members[item["espn_player_id"]]
            alternatives=_same_role_candidates(members,person,matched_absent_ids,lineup_ids)
            baseline=person["role_index_0_100"]
            replacement=alternatives[0]["role_index_0_100"] if alternatives else None
            missing_roles.append({
                "player":person["player"],"position":person["position"],
                "player_role_index":baseline,
                "replacement_candidates":alternatives,
                "cover_gap_index_points":
                    round(max(0.,baseline-replacement),2)
                    if baseline is not None and replacement is not None else None,
                "cover_confirmed":bool(lineup_ids is not None and alternatives),
            })
        verified=[
            item["role_index_0_100"] for item in absences
            if item["provider_status"]=="reported_absent"
            and item["role_index_0_100"] is not None
        ]
        all_missing=[x for x in preview.get("provider_injury_context") or []
                     if str(x.get("espn_team_id") or "")==team_id and
                        x.get("availability")=="reported_absent"]
        full_coverage=(not all_missing or
                       len([x for x in absences if x["provider_status"]=="reported_absent"
                            and x["role_index_0_100"] is not None])==len(all_missing))
        teams.append({
            "espn_team_id":team_id,
            "team_history_games":max((p["prior_team_games"] for p in members.values()),default=0),
            "player_roles_evaluable":sum(x["role_evaluable"] for x in members.values()),
            "reported_absences":len(all_missing),
            "absence_role_index_sum":round(sum(verified),2) if full_coverage and verified else None,
            "all_reported_absences_role_resolved":full_coverage,
            "linked_absences_and_uncertain":absences,
            "unresolved_provider_reports":unresolved,
            "missing_starter_replacement_research":missing_roles,
            "confirmed_lineup_at_capture":lineup_ids is not None,
        })
    home,away=teams
    # Difference only when BOTH teams have sufficient linked player roles
    # and their injury feed is actually covered (unknown != zero).
    coverage=bool(preview.get("api_injury_coverage_confirmed"))
    difference=(round((away["absence_role_index_sum"] or 0) -
                      (home["absence_role_index_sum"] or 0),2)
                if coverage and home["all_reported_absences_role_resolved"]
                and away["all_reported_absences_role_resolved"]
                and (home["absence_role_index_sum"] is not None or
                     away["absence_role_index_sum"] is not None) else None)
    return {
        "source_id":"soccer_starter_replacement_research",
        "source_kind":"soccer_player_impact_research",
        "sport":"soccer","event_id":str(preview["event_id"]),
        "competition_code":str(preview["competition_code"]),
        "captured_utc":as_of.isoformat(),
        "kickoff_utc":utc(preview["kickoff_utc"]).isoformat(),
        "home_id":str(preview["home_id"]),"away_id":str(preview["away_id"]),
        "home_minus_away_absence_exposure":difference,
        "availability_coverage_known":coverage,
        "teams":teams,
        "lineup_confirmed":known_lineup is not None,
        "status":"RESEARCH_ONLY_NOT_MODEL_INPUT",
        "warning":"Role index measures observed minutes/exposure, not wins, goals, odds or causal player absence. Replacement is only a suggestion without a verified XI.",
    }
