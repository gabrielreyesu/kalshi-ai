"""As-of player availability and opponent matchup research for NBA/NFL.

All returned statistics are either from completed games before as_of or from
an observed pre-game injury snapshot. No prospective model coefficients.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd


MEASURES = {
    "nba": ("minutes", "points", "rebounds", "assists"),
    "nfl": ("passing_yards", "rushing_yards", "receiving_yards", "receptions"),
}
OUT = {"out", "injured reserve", "ir", "out for season", "suspension", "suspended"}
UNCERTAIN = {"questionable", "doubtful", "day-to-day", "probable", "gtd", "game time decision"}


def timestamp(value):
    value = pd.Timestamp(value)
    if pd.isna(value):
        raise ValueError("Invalid timestamp")
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def as_number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if np.isfinite(value) else None
    if value is None:
        return None
    s = str(value).strip()
    if not s or s in ("--", "-", "DNP", "N/A"):
        return None
    if ":" in s:
        try:
            minutes, seconds = s.split(":")
            return float(minutes) + float(seconds) / 60.0
        except (ValueError, TypeError):
            return None
    try:
        f = float(s)
        return f if np.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def parse_injury_feed(payload, sport, captured):
    """Store all injury-listed athletes, never equate not-listed with healthy."""
    if sport not in MEASURES:
        raise ValueError("Unsupported sport")
    seen = timestamp(captured)
    if not isinstance(payload, dict) or not isinstance(payload.get("injuries"), list):
        raise ValueError("Unrecognized injury report schema")
    athletes = []
    unique = set()
    for team in payload["injuries"]:
        team_info = team.get("team") or team
        team_id = str(team_info.get("id") or team.get("id") or "")
        if not team_id:
            continue
        for item in team.get("injuries") or []:
            athlete = item.get("athlete") or {}
            athlete_id = str(athlete.get("id") or "")
            name = str(athlete.get("displayName") or athlete.get("fullName") or "").strip()
            status = str(item.get("status") or "").strip()
            if not (athlete_id or name) or not status:
                continue
            report_date = item.get("date")
            if report_date:
                try:
                    # An injury report updated after our snapshot must not leak in.
                    reported = timestamp(report_date)
                    if reported > seen:
                        continue
                except (ValueError, TypeError):
                    continue
            else:
                reported = None
            key = (team_id, athlete_id or name.casefold())
            if key in unique:
                continue
            unique.add(key)
            athletes.append({
                "team_id": team_id, "team": str(team_info.get("displayName") or team_info.get("name") or ""),
                "player_id": athlete_id, "player": name,
                "position": str((athlete.get("position") or {}).get("abbreviation") or ""),
                "status": status,
                "reported_utc": reported.isoformat() if reported is not None else None,
            })
    return {
        "source_id": "espn_official_injury_feed",
        "source_kind": "player_availability_research",
        "sport": sport, "event_id": f"{sport}_league_injuries",
        "captured_utc": seen.isoformat(), "reported_athletes": athletes,
        "note": "Only injury-listed athletes. Not listed does not imply active.",
    }


def _stat_map(stat_block, sport):
    """Handle ESPN NBA per-team and NFL per-category scorebox layouts."""
    labels = stat_block.get("labels") or []
    group = str(stat_block.get("name") or stat_block.get("displayName") or "").lower()
    if not labels:
        labels = stat_block.get("names") or []
    out = []
    for entry in stat_block.get("athletes") or []:
        athlete = entry.get("athlete") or {}
        ident = str(athlete.get("id") or "")
        if not ident:
            continue
        stat = dict(zip([str(k).upper() for k in labels], entry.get("stats") or []))
        metrics = {}
        if sport == "nba":
            names = {"minutes": ("MIN",), "points": ("PTS",),
                     "rebounds": ("REB",), "assists": ("AST",)}
        elif "passing" in group:
            names = {"passing_yards": ("YDS", "PASS YDS", "PASS_YDS")}
        elif "rushing" in group:
            names = {"rushing_yards": ("YDS", "RUSH YDS", "RUSH_YDS")}
        elif "receiving" in group:
            names = {"receiving_yards": ("YDS", "REC YDS", "REC_YDS"),
                     "receptions": ("REC",)}
        else:
            continue
        for metric, fields in names.items():
            value = next((as_number(stat[k]) for k in fields if k in stat), None)
            if value is not None:
                metrics[metric] = value
        if not metrics:
            continue
        out.append({
            "player_id": ident, "player": str(athlete.get("displayName") or ""),
            "position": str((athlete.get("position") or {}).get("abbreviation") or ""),
            "metrics": metrics,
        })
    return out


def parse_completed_boxscore(payload, sport, event_id, captured, kickoff=None):
    """Parse ESPN event summary only if its official game status is final."""
    if sport not in MEASURES:
        raise ValueError("Unsupported sport")
    if not isinstance(payload, dict):
        raise ValueError("Malformed boxscore response")
    head = payload.get("header") or {}
    contests = head.get("competitions") or []
    if len(contests) != 1:
        return None
    contest = contests[0]
    status = contest.get("status") or head.get("status") or {}
    status_name = str((status.get("type") or {}).get("name") or status.get("name") or "")
    if status_name != "STATUS_FINAL":
        return None
    game_kick = contest.get("date") or kickoff
    if not game_kick:
        return None
    if timestamp(game_kick) >= timestamp(captured):
        return None
    members = contest.get("competitors") or []
    sides = {r.get("homeAway"): r for r in members}
    if set(sides) != {"home", "away"}:
        return None
    home = str((sides["home"].get("team") or {}).get("id") or "")
    away = str((sides["away"].get("team") or {}).get("id") or "")
    if not home or not away:
        return None
    individuals = []
    for team_block in (payload.get("boxscore") or {}).get("players") or []:
        team = team_block.get("team") or {}
        team_id = str(team.get("id") or "")
        if team_id not in (home, away):
            continue
        by_id = {}
        for group in team_block.get("statistics") or []:
            for row in _stat_map(group, sport):
                pid = row["player_id"]
                if pid not in by_id:
                    by_id[pid] = {**row, "team_id": team_id,
                                  "opponent_id": away if team_id == home else home}
                else:
                    by_id[pid]["metrics"].update(row["metrics"])
        individuals.extend(by_id.values())
    if not individuals:
        return None
    return {
        "source_id": "espn_completed_player_boxscore",
        "source_kind": "historical_player_stats",
        "sport": sport, "event_id": str(event_id),
        "captured_utc": timestamp(captured).isoformat(),
        "kickoff_utc": timestamp(game_kick).isoformat(),
        "home_id": home, "away_id": away, "players": individuals,
        "note": "Postgame boxscore. Only valid as historical context for LATER fixtures.",
    }


def _latest_snapshot(records, sport, as_of):
    cutoff = timestamp(as_of)
    candidates = [
        r for r in records if r.get("sport") == sport and
        r.get("source_id") == "espn_official_injury_feed" and
        cutoff-pd.Timedelta(hours=36) <= timestamp(r["captured_utc"]) <= cutoff
    ]
    return max(candidates, key=lambda r: timestamp(r["captured_utc"])) if candidates else None


def player_matchup_context(event, boxscores, injuries, as_of, max_recent=20, shrinkage_games=6):
    """Build explanatory player context; never interpret a sample as causal impact.

    Prior games must be finalized before the observation time AND before the
    upcoming fixture. Historical opponent-specific rates are shrunk heavily
    toward a player's overall recent mean and omitted below two prior meetings.
    """
    sport = str(event["sport"])
    if sport not in MEASURES:
        raise ValueError("Unsupported sport")
    now, kickoff = timestamp(as_of), timestamp(event["kickoff_utc"])
    if now >= kickoff:
        raise ValueError("Only pregame context permitted")
    team_ids = [str(event["home_id"]), str(event["away_id"])]
    if not all(team_ids):
        raise ValueError("Missing canonical ESPN team id")
    snapshot = _latest_snapshot(injuries, sport, now)
    listed = defaultdict(list)
    if snapshot:
        for item in snapshot.get("reported_athletes", []):
            if str(item.get("team_id")) in team_ids:
                listed[str(item["team_id"])].append(item)
    history = defaultdict(list)
    for game in boxscores:
        if game.get("sport") != sport or game.get("source_id") != "espn_completed_player_boxscore":
            continue
        # A backfilled postgame record must NOT be used at timestamps before it
        # existed; this prevents retrospective performance leakage.
        if timestamp(game["captured_utc"]) > now or timestamp(game["kickoff_utc"]) >= kickoff:
            continue
        for p in game.get("players", []):
            if str(p.get("team_id")) in team_ids:
                key = (str(p["team_id"]), str(p["player_id"]))
                history[key].append({
                    "game_utc": timestamp(game["kickoff_utc"]),
                    "opponent_id": str(p.get("opponent_id") or ""),
                    "player": p.get("player"), "position": p.get("position"),
                    "metrics": p.get("metrics") or {},
                })
    rows = []
    for team_id, opponent_id in ((team_ids[0], team_ids[1]), (team_ids[1], team_ids[0])):
        # Daily CSV fallback lacks canonical athlete IDs. Match by EXACT
        # team+player name only when precisely one prior ID has that name.
        # Otherwise retain a separate name: identifier and do not guess.
        by_name = defaultdict(set)
        for (t, historic_pid), games in history.items():
            if t == team_id and games:
                pname = str(games[-1].get("player") or "").casefold().strip()
                if pname:
                    by_name[pname].add(historic_pid)
        current = {}
        for item in listed.get(team_id, []):
            pid = str(item.get("player_id") or "")
            if not pid:
                continue
            if not item.get("id_is_canonical", True):
                matches = by_name.get(str(item.get("player") or "").casefold().strip(), set())
                if len(matches) == 1:
                    pid = next(iter(matches))
            current[pid] = item
        ids = {pid for t,pid in history if t == team_id} | set(current)
        for pid in ids:
            old = sorted(history.get((team_id, pid), []), key=lambda x:x["game_utc"])[-max_recent:]
            record = current.get(pid)
            fallback = old[-1] if old else {}
            status = str(record.get("status") if record else "not_reported").lower()
            entry = {
                "team_id": team_id, "opponent_id": opponent_id,
                "player_id": pid,
                "player": (record or {}).get("player") or fallback.get("player") or "",
                "position": (record or {}).get("position") or fallback.get("position") or "",
                "injury_status": status,
                "availability": ("reported_out" if status in OUT else
                                 "uncertain" if status in UNCERTAIN else
                                 "listed_other" if record else "not_reported"),
                "last_report_utc": snapshot["captured_utc"] if record and snapshot else None,
                "prior_games": len(old),
                "metrics": {},
            }
            for metric in MEASURES[sport]:
                previous = [g["metrics"][metric] for g in old
                            if metric in g["metrics"] and g["metrics"][metric] is not None]
                if not previous:
                    continue
                base = float(np.mean(previous))
                prior = [g["metrics"][metric] for g in old
                         if g["opponent_id"] == opponent_id and metric in g["metrics"]]
                n = len(prior)
                match_mean = float(np.mean(prior)) if n >= 2 else None
                adjusted = (base + n/(n+shrinkage_games)*(match_mean-base)
                            if match_mean is not None else None)
                entry["metrics"][metric] = {
                    "recent_mean": round(base, 3),
                    "prior_opponent_games": n,
                    "opponent_mean": round(match_mean,3) if match_mean is not None else None,
                    "shrunk_opponent_mean": round(adjusted,3) if adjusted is not None else None,
                }
            rows.append(entry)
    return {
        "sport": sport, "event_id": str(event["event_id"]),
        "evaluated_utc": now.isoformat(), "kickoff_utc": kickoff.isoformat(),
        "injury_snapshot_utc": snapshot.get("captured_utc") if snapshot else None,
        "injury_data_present": snapshot is not None,
        "teams": team_ids,
        "players": sorted(rows, key=lambda p:(p["team_id"],p["player_id"])),
        "warning": "Not listed means unknown, not active. H2H stats are small-sample associations, not causal win-probability effects. Research only.",
    }
