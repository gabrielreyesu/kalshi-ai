"""Verify hourly soccer ingestion refreshes team news without hourly paid-API burn."""
import json
import tempfile
import unittest
from pathlib import Path

from kalshi_external.soccer_player_collect import (
    CODES, _hourly_competitions, _injury_refresh_days, run,
)
from kalshi_external.soccer_impact_collect import read_rows


class Response:
    status_code=200
    headers={}
    content=b'{"ok":true}'
    def __init__(self, data): self.data=data
    def json(self): return self.data


def scoreboard(date="2026-11-13T19:00:00Z"):
    return {"events":[{
        "id":"5001","date":date,
        "status":{"type":{"name":"STATUS_SCHEDULED"}},
        "competitions":[{"competitors":[
            {"homeAway":"home","team":{"id":"10","displayName":"Arsenal"}},
            {"homeAway":"away","team":{"id":"20","displayName":"Barcelona"}}
        ]}]
    }]}


def prior_snapshot(captured="2026-11-12T11:00:00Z",
                   kickoff="2026-11-13T19:00:00Z"):
    return {
        "source_id":"soccer_player_opponent_preview",
        "sport":"soccer","event_id":"5001",
        "competition_code":"uefa.champions",
        "home_id":"10","away_id":"20",
        "captured_utc":captured,"kickoff_utc":kickoff,
        "players":[],"provider_injury_context":[]
    }


class SoccerProvider:
    def __init__(self, kickoff="2026-11-13T19:00:00Z", lineups=False):
        self.requests=[]
        self.kickoff=kickoff
        self.lineups=lineups
    def get(self, url, params=None, headers=None, timeout=None, allow_redirects=False):
        self.requests.append((url,dict(params or {})))
        if url.endswith("/scoreboard"):
            when=params["dates"]
            if when==self.kickoff[:10].replace("-",""):
                return Response(scoreboard(self.kickoff))
            return Response({"events":[]})
        if url.endswith("/injuries"):
            if params["date"]!=self.kickoff[:10]:
                return Response({"response":[]})
            return Response({"response":[{
                "fixture":{"id":777,"date":self.kickoff},
                "team":{"id":55,"name":"Arsenal"},
                "league":{"name":"Champions League"},
                "player":{"id":100,"name":"Starter One",
                          "type":"Missing Fixture","reason":"Knee"}
            }]})
        if url.endswith("/fixtures"):
            if params["date"]!=self.kickoff[:10]:
                return Response({"response":[]})
            return Response({"response":[{
                "fixture":{"id":777,"date":self.kickoff},
                "teams":{"home":{"id":55,"name":"Arsenal"},
                         "away":{"id":66,"name":"Barcelona"}}
            }]})
        if url.endswith("/fixtures/lineups"):
            if not self.lineups:
                return Response({"response":[]})
            a=[]
            for side,team_id in [("A",55),("B",66)]:
                a.append({
                    "team":{"id":team_id},"formation":"4-3-3",
                    "startXI":[{"player":{"id":str(team_id*100+i),
                                           "name":f"{side} player {i}","pos":"M"}}
                               for i in range(11)],
                    "substitutes":[]
                })
            return Response({"response":a})
        raise ValueError("Unmocked provider endpoint")


def write_snapshot(path, record):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(record)+"\n")


class HourlySoccerMonitorTests(unittest.TestCase):
    def test_select_only_competition_with_upcoming_previews(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)
            write_snapshot(path/"soccer_previews.jsonl",prior_snapshot())
            from kalshi_external.core import utc
            codes=_hourly_competitions(path,utc("2026-11-12T12:00:00Z"),set(CODES))
            self.assertEqual(codes,("uefa.champions",))
            self.assertEqual(_hourly_competitions(path,utc("2026-11-15T20:00:00Z"),set(CODES)),())

    def test_injuries_only_refreshed_when_four_hours_old(self):
        from kalshi_external.core import utc
        reports=[{"source_id":"api_football_soccer_injuries",
                  "observed_day_utc":"2026-11-13",
                  "captured_utc":"2026-11-12T12:00:00Z"}]
        now=utc("2026-11-12T15:59:00Z")
        self.assertEqual(_injury_refresh_days(reports,now,{"2026-11-13"}),[])
        now=utc("2026-11-12T16:00:00Z")
        self.assertEqual(_injury_refresh_days(reports,now,{"2026-11-13"}),["2026-11-13"])

    def test_hourly_run_no_historical_or_duplicate_injury_calls(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            private=root/"signals"
            write_snapshot(private/"soccer_previews.jsonl",prior_snapshot())
            first=SoccerProvider()
            result=run(private,root/"status",observed_at="2026-11-12T12:00:00Z",
                       session=first,codes=CODES,api_key="demo",
                       mode="hourly")
            self.assertEqual(result["collection_mode"],"hourly")
            self.assertEqual(result["history_monthly_scoreboards_requested"],0)
            self.assertEqual(result["ESPN_match_summaries_queried"],0)
            self.assertEqual(result["api_football_requests_this_run"]["injuries"],2)
            self.assertEqual(result["api_football_requests_this_run"]["fixture_index"],2)
            self.assertEqual(result["new_future_match_player_contexts"],1)
            self.assertEqual({u.split("/")[-2] for u,p in first.requests if u.endswith("/scoreboard")},
                             {"uefa.champions"})
            self.assertFalse(any("202610" in str(p) or u.endswith("/summary") for u,p in first.requests))
            second=SoccerProvider()
            r2=run(private,root/"status",observed_at="2026-11-12T13:00:00Z",
                   session=second,codes=CODES,api_key="demo",mode="hourly")
            self.assertEqual(r2["api_football_requests_this_run"]["injuries"],0)
            self.assertEqual(r2["api_football_requests_this_run"]["fixture_index"],0)
            self.assertEqual(r2["new_api_football_injury_day_snapshots"],0)
            self.assertEqual(r2["new_future_match_player_contexts"],0)
            self.assertEqual(r2["unchanged_contexts_not_rearchived"],1)
            third=SoccerProvider()
            r3=run(private,root/"status",observed_at="2026-11-12T16:01:00Z",
                   session=third,codes=CODES,api_key="demo",mode="hourly")
            self.assertEqual(r3["api_football_requests_this_run"]["injuries"],2)
            self.assertEqual(r3["api_football_requests_this_run"]["fixture_index"],1)
            self.assertEqual(r3["new_future_match_player_contexts"],1)
            self.assertEqual(r3["unchanged_contexts_not_rearchived"],0)
            # Same fixture IDs preserved; not falsely considered "unknown" just
            # because we did not call /fixtures a second time.
            latest=read_rows(private/"soccer_previews.jsonl")[-1]
            self.assertTrue(latest["api_injury_coverage_confirmed"])
            self.assertEqual(latest["provider_injury_context"][0]["availability"],
                             "reported_absent")

    def test_confident_pregame_lineups_only_once_and_only_before_kickoff(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            private=root/"signals"
            kick="2026-11-12T14:00:00Z"
            write_snapshot(private/"soccer_previews.jsonl",
                           prior_snapshot("2026-11-12T11:00:00Z",kick))
            provider=SoccerProvider(kickoff=kick,lineups=True)
            r=run(private,root/"status",observed_at="2026-11-12T13:25:00Z",
                  session=provider,api_key="demo",mode="hourly")
            self.assertEqual(r["new_verified_pregame_starting_XI_snapshots"],1)
            self.assertEqual(r["api_football_requests_this_run"]["starting_XI"],1)
            records=read_rows(private/"soccer_confirmed_lineups.jsonl")
            self.assertEqual(len(records),1)
            self.assertTrue(records[0]["lineup_confirmed"])
            preview=read_rows(private/"soccer_previews.jsonl")[-1]
            self.assertEqual(preview["confirmed_XI_observed_utc"],
                             records[0]["captured_utc"])
            later=SoccerProvider(kickoff=kick,lineups=True)
            r2=run(private,root/"status",observed_at="2026-11-12T13:30:00Z",
                   session=later,api_key="demo",mode="hourly")
            self.assertEqual(r2["api_football_requests_this_run"]["starting_XI"],0)
            self.assertEqual(r2["new_verified_pregame_starting_XI_snapshots"],0)
            self.assertEqual(r2["new_future_match_player_contexts"],0)
            self.assertEqual(len(read_rows(private/"soccer_confirmed_lineups.jsonl")),1)

    def test_empty_hourly_scope_does_not_burn_provider_requests(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            p=SoccerProvider()
            r=run(root/"signals",root/"status",observed_at="2026-11-12T12:00:00Z",
                  session=p,api_key="demo",mode="hourly")
            self.assertEqual(len(p.requests),0)
            self.assertEqual(r["new_future_match_player_contexts"],0)
            self.assertEqual(r["api_football_requests_this_run"]["starting_XI"],0)

    def test_invalid_mode_fails_before_any_network(self):
        with self.assertRaisesRegex(ValueError,"hourly or full"):
            run(mode="faster")


if __name__=="__main__":
    unittest.main()
