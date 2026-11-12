"""Bounded ESPN monthly backfill for teams with scheduled soccer fixtures."""
import tempfile
import unittest
from pathlib import Path

from kalshi_external.soccer_player_collect import run
from kalshi_external.soccer_impact_collect import read_rows


def event(eid,date,status):
    return {"id":str(eid),"date":date,
            "status":{"type":{"name":status}},
            "competitions":[{"competitors":[
                {"homeAway":"home","team":{"id":"10","displayName":"Arsenal"}},
                {"homeAway":"away","team":{"id":"20","displayName":"Barcelona"}}
            ]}]}


class Response:
    status_code=200
    content=b"{}"
    headers={}
    def __init__(self,data):self.data=data
    def json(self):return self.data


class ESPNMonthlyStub:
    def __init__(self):self.requests=[]
    dates={"1001":"2026-10-15T19:00:00Z",
           "1002":"2026-11-05T19:00:00Z"}
    def get(self,url,params=None,headers=None,timeout=None,allow_redirects=False):
        self.requests.append((url,dict(params or {})))
        if url.endswith("/scoreboard"):
            d=params["dates"]
            if d=="20261113":
                return Response({"events":[event("3001","2026-11-13T19:00:00Z","STATUS_SCHEDULED")]})
            if d=="202610":
                return Response({"events":[event("1001",self.dates["1001"],"STATUS_FULL_TIME")]})
            if d=="202611":
                return Response({"events":[event("1002",self.dates["1002"],"STATUS_FULL_TIME")]})
            return Response({"events":[]})
        if url.endswith("/summary"):
            eid=str(params["event"])
            date=self.dates[eid]
            return Response({
                "header":{"competitions":[{
                    "date":date,"status":{"type":{"name":"STATUS_FULL_TIME"}},
                    "competitors":[
                        {"homeAway":"home","team":{"id":"10"}},
                        {"homeAway":"away","team":{"id":"20"}}
                    ]}]},
                "boxscore":{"players":[{
                    "team":{"id":"10"},"statistics":[{
                        "labels":["MIN","G","SH"],
                        "athletes":[{"athlete":{
                            "id":"101","displayName":"Star Forward"},
                            "stats":["90","1","3"],"starter":True}]
                    }]
                }]}
            })
        raise ValueError("Unexpected ESPN URL")


class MonthBackfillTests(unittest.TestCase):
    def test_recent_completed_matches_discovered_for_upcoming_team(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            sess=ESPNMonthlyStub()
            result=run(root/"private",root/"reports",
                       observed_at="2026-11-12T12:00:00Z",
                       session=sess,codes=("uefa.champions",),
                       api_key="",max_summaries=5)
            self.assertEqual(result["history_monthly_scoreboards_requested"],2)
            self.assertEqual(result["new_prior_fixtures_discovered"],2)
            self.assertEqual(result["new_completed_match_player_stats"],2)
            self.assertEqual(result["new_future_match_player_contexts"],1)
            self.assertEqual(result["future_contexts_with_history"],1)
            past=read_rows(root/"private"/"soccer_player_games.jsonl")
            self.assertEqual({x["event_id"] for x in past},{"1001","1002"})
            self.assertTrue(all(x["captured_utc"]>x["kickoff_utc"] for x in past))
            self.assertLessEqual(len([r for r in sess.requests
                                       if r[1].get("dates") in ("202610","202611")]),8)
            second=run(root/"private",root/"reports",
                       observed_at="2026-11-12T13:00:00Z",
                       session=ESPNMonthlyStub(),codes=("uefa.champions",),
                       api_key="",max_summaries=5)
            self.assertEqual(second["new_completed_match_player_stats"],0)

    def test_no_upcoming_games_no_historical_backfill(self):
        class Empty(ESPNMonthlyStub):
            def get(self,url,params=None,headers=None,timeout=None,allow_redirects=False):
                self.requests.append((url,dict(params or {})))
                return Response({"events":[]})
        with tempfile.TemporaryDirectory() as tmp:
            s=Empty()
            r=run(Path(tmp)/"private",Path(tmp)/"report",
                  observed_at="2026-11-12T12:00:00Z",
                  session=s,codes=("esp.1",),api_key="")
            self.assertEqual(r["history_monthly_scoreboards_requested"],0)
            self.assertEqual(r["new_prior_fixtures_discovered"],0)


if __name__=="__main__":
    unittest.main()
