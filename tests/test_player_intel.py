import tempfile
import unittest
from pathlib import Path

import pandas as pd

from kalshi_external.player_intel import (
    parse_injury_feed, parse_completed_boxscore, player_matchup_context, as_number
)
from kalshi_external.player_collect import run


def nba_injuries():
    return {"injuries": [
        {"id":"1","displayName":"Boston Celtics","injuries":[
            {"athlete":{"id":"101","displayName":"Star A",
                        "position":{"abbreviation":"G"}},
             "status":"Out","date":"2026-11-12T08:00:00Z"},
            {"athlete":{"id":"102","displayName":"Star B",
                        "position":{"abbreviation":"F"}},
             "status":"Questionable","date":"2026-11-12T09:00:00Z"},
            {"athlete":{"id":"103","displayName":"Star C"},
             "status":"Out","date":"2026-11-14T09:00:00Z"},
        ]},
    ]}


def nba_summary(game_id="90001", date="2026-11-11T20:00:00Z",
                status="STATUS_FINAL", away_id="2", score=31):
    return {
        "header":{"competitions":[{
            "date":date, "status":{"type":{"name":status}},
            "competitors":[
                {"homeAway":"home","team":{"id":"1"}},
                {"homeAway":"away","team":{"id":away_id}},
            ]
        }]},
        "boxscore":{"players":[
            {"team":{"id":"1"},"statistics":[{
                "labels":["MIN","PTS","REB","AST"],
                "athletes":[
                    {"athlete":{"id":"101","displayName":"Star A",
                                "position":{"abbreviation":"G"}},
                     "stats":["32:30",str(score),"8","6"]},
                ]
            }]},
            {"team":{"id":away_id},"statistics":[{
                "labels":["MIN","PTS"],
                "athletes":[
                    {"athlete":{"id":"201","displayName":"Opponent B"},
                     "stats":["35","24"]},
                ]
            }]}
        ]},
    }


def nfl_summary(date="2026-11-11T20:00:00Z"):
    return {
        "header":{"competitions":[{
            "date":date, "status":{"type":{"name":"STATUS_FINAL"}},
            "competitors":[
                {"homeAway":"home","team":{"id":"4"}},
                {"homeAway":"away","team":{"id":"5"}},
            ]
        }]},
        "boxscore":{"players":[{
            "team":{"id":"4"},"statistics":[
                {"name":"receiving","labels":["REC","YDS","AVG","TD"],
                 "athletes":[{"athlete":{"id":"11","displayName":"Receiver X"},
                              "stats":["7","110","15.7","1"]}]},
                {"name":"rushing","labels":["CAR","YDS","AVG"],
                 "athletes":[{"athlete":{"id":"11","displayName":"Receiver X"},
                              "stats":["2","18","9.0"]}]}
            ]}]}
    }


class Response:
    status_code = 200
    content = b'{"ok":true}'
    def __init__(self, body): self.body=body
    def json(self): return self.body


class FakeSession:
    def __init__(self): self.calls=[]
    def get(self,url,params=None,timeout=None,headers=None,allow_redirects=None):
        self.calls.append((url,params))
        if url.endswith("/injuries"):
            return Response(nba_injuries() if "/basketball/" in url else {"injuries":[]})
        if url.endswith("/scoreboard"):
            if "/basketball/" in url and params["dates"]=="20261111":
                return Response({"events":[{"id":"90001","date":"2026-11-11T20:00:00Z",
                                             "status":{"type":{"name":"STATUS_FINAL"}}}]})
            return Response({"events":[]})
        if url.endswith("/summary"):
            return Response(nba_summary())
        raise ValueError("unexpected route")


class PlayerIntelTests(unittest.TestCase):
    def test_injury_snapshot_carries_player_ids_and_time(self):
        r=parse_injury_feed(nba_injuries(),"nba","2026-11-12T11:00:00Z")
        self.assertEqual(len(r["reported_athletes"]),2)
        self.assertEqual(r["reported_athletes"][0]["player_id"],"101")
        self.assertEqual(r["reported_athletes"][1]["status"],"Questionable")

    def test_completed_game_boxscore_nba_and_nfl(self):
        nba=parse_completed_boxscore(nba_summary(),"nba","90001","2026-11-12T11:00:00Z")
        self.assertAlmostEqual(nba["players"][0]["metrics"]["minutes"],32.5)
        self.assertEqual(nba["players"][0]["metrics"]["points"],31)
        nfl=parse_completed_boxscore(nfl_summary(),"nfl","90002","2026-11-12T11:00:00Z")
        p=nfl["players"][0]
        self.assertEqual(p["metrics"]["receptions"],7)
        self.assertEqual(p["metrics"]["receiving_yards"],110)
        self.assertEqual(p["metrics"]["rushing_yards"],18)

    def test_rejects_unfinished_games_and_wrong_event_time(self):
        self.assertIsNone(parse_completed_boxscore(
            nba_summary(status="STATUS_IN_PROGRESS"),"nba","90001","2026-11-12T11:00:00Z"))
        self.assertIsNone(parse_completed_boxscore(
            nba_summary(date="2026-11-14T20:00:00Z"),
            "nba","90001","2026-11-12T11:00:00Z"))

    def test_matchup_splits_shrunk_without_future_leakage(self):
        games=[]
        for i,pts in enumerate((10,30,32,9)):
            date=(pd.Timestamp("2026-10-01T20:00Z")+pd.Timedelta(days=i)).isoformat()
            g=parse_completed_boxscore(
                nba_summary(game_id=str(100+i),date=date,away_id="2" if i<2 else "3",score=pts),
                "nba",str(100+i),"2026-10-07T10:00:00Z")
            games.append(g)
        injury=parse_injury_feed(nba_injuries(),"nba","2026-11-12T11:00:00Z")
        ev={"sport":"nba","event_id":"later","kickoff_utc":"2026-11-13T20:00:00Z",
            "home_id":"1","away_id":"2"}
        result=player_matchup_context(ev,games,[injury],"2026-11-12T12:00:00Z")
        star=next(p for p in result["players"] if p["player_id"]=="101")
        self.assertEqual(star["availability"],"reported_out")
        self.assertEqual(star["prior_games"],4)
        pts=star["metrics"]["points"]
        self.assertEqual(pts["prior_opponent_games"],2)
        self.assertEqual(pts["opponent_mean"],20.0)
        self.assertAlmostEqual(pts["recent_mean"],20.25)
        self.assertGreater(pts["shrunk_opponent_mean"],20.0)
        self.assertLess(pts["shrunk_opponent_mean"],20.25)

    def test_matchup_no_future_boxscore_or_future_injury(self):
        ev={"sport":"nba","event_id":"future","kickoff_utc":"2026-11-13T20:00:00Z",
            "home_id":"1","away_id":"2"}
        today=parse_injury_feed(nba_injuries(),"nba","2026-11-12T11:00:00Z")
        later=parse_injury_feed(nba_injuries(),"nba","2026-11-14T11:00:00Z")
        recent=parse_completed_boxscore(nba_summary(),"nba","90001","2026-11-14T09:00:00Z")
        result=player_matchup_context(ev,[recent],[later,today],"2026-11-12T12:00:00Z")
        p=next(p for p in result["players"] if p["player_id"]=="101")
        self.assertEqual(p["prior_games"],0)
        self.assertEqual(p["availability"],"reported_out")

    def test_no_injury_listing_does_not_mean_active(self):
        ev={"sport":"nba","event_id":"future","kickoff_utc":"2026-11-13T20:00:00Z",
            "home_id":"1","away_id":"2"}
        game=parse_completed_boxscore(nba_summary(),"nba","90001","2026-11-12T11:00:00Z")
        context=player_matchup_context(ev,[game],[],"2026-11-12T12:00:00Z")
        a=next(p for p in context["players"] if p["player_id"]=="101")
        self.assertEqual(a["availability"],"not_reported")
        self.assertFalse(context["injury_data_present"])

    def test_as_number_tolerates_minutes(self):
        self.assertAlmostEqual(as_number("30:45"),30.75)
        self.assertIsNone(as_number("3-4"))

    def test_run_writes_private_batches_and_aggregate_report_only(self):
        with tempfile.TemporaryDirectory() as td:
            a=Path(td)/"injury.jsonl"
            b=Path(td)/"box.jsonl"
            report=Path(td)/"reports"
            session=FakeSession()
            first=run(a,b,report,observed_at="2026-11-12T11:00:00Z",
                      session=session,days=2,max_new_games=3,pause_seconds=0)
            self.assertEqual(first["new_injury_snapshot_batches"],2)
            self.assertEqual(first["new_completed_game_boxscores"],1)
            self.assertEqual(len(b.read_text().splitlines()),1)
            self.assertNotIn("Star A",(report/"coverage.json").read_text())
            again=run(a,b,report,observed_at="2026-11-12T12:00:00Z",
                      session=FakeSession(),days=2,max_new_games=3,pause_seconds=0)
            self.assertEqual(again["new_completed_game_boxscores"],0)
            self.assertEqual(again["new_injury_snapshot_batches"],2)

if __name__=="__main__":
    unittest.main()
