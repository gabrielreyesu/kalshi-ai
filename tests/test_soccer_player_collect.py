import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kalshi_external.soccer_player_collect import run, _get, ROOT


def scoreboard(event_id,date,status):
    return {"events":[{
        "id":str(event_id),"date":date,
        "status":{"type":{"name":status}},
        "competitions":[{"competitors":[
            {"homeAway":"home","team":{"id":"10","displayName":"Arsenal"}},
            {"homeAway":"away","team":{"id":"20","displayName":"Barcelona"}}
        ]}]
    }]}


def finished_summary(date):
    return {
        "header":{"competitions":[{
            "date":date,"status":{"type":{"name":"STATUS_FULL_TIME"}},
            "competitors":[
                {"homeAway":"home","team":{"id":"10"}},
                {"homeAway":"away","team":{"id":"20"}}]
        }]},
        "boxscore":{"players":[{
            "team":{"id":"10"},"statistics":[{
                "labels":["MIN","G","SH"],
                "athletes":[{"athlete":{"id":"101","displayName":"Forward"},
                             "stats":["90","1","4"]}]
            }]
        }]}
    }


class Resp:
    def __init__(self,code,body,headers=None):
        self.status_code=code
        self.data=body
        self.headers=headers or {}
        self.content=b"{}"
    def json(self):return self.data


class MockESPN:
    def __init__(self,forward=None):
        self.urls=[]
        self.forward=forward

    def get(self,url,params=None,headers=None,timeout=None,allow_redirects=False):
        self.urls.append((url,params))
        if self.forward is not None:
            return self.forward(url,params)
        if url.endswith("/scoreboard"):
            day=params.get("dates")
            if day=="20261111":
                return Resp(200,scoreboard("1001","2026-11-11T19:00:00Z","STATUS_FULL_TIME"))
            if day=="20261113":
                return Resp(200,scoreboard("3001","2026-11-13T19:00:00Z","STATUS_SCHEDULED"))
            return Resp(200,{"events":[]})
        if url.endswith("/summary"):
            return Resp(200,finished_summary("2026-11-11T19:00:00Z"))
        raise ValueError("Unexpected HTTP")


class SoccerCollectorTests(unittest.TestCase):
    def test_research_pipeline_archives_completed_stats_and_future_context(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            data=root/"private"
            status=root/"reports"
            s=MockESPN()
            a=run(data,status,observed_at="2026-11-12T12:00:00Z",
                  session=s,codes=("uefa.champions",),api_key="",
                  max_summaries=3)
            self.assertEqual(a["new_completed_match_player_stats"],1)
            self.assertEqual(a["new_future_match_player_contexts"],1)
            self.assertEqual(a["future_contexts_with_history"],1)
            self.assertFalse(a["api_football_configured"])
            self.assertFalse(a["production_changed"])
            self.assertTrue((data/"soccer_player_games.jsonl").exists())
            self.assertTrue((data/"soccer_previews.jsonl").exists())
            self.assertNotIn("Forward",(status/"coverage.json").read_text())
            b=run(data,status,observed_at="2026-11-12T12:00:00Z",
                  session=MockESPN(),codes=("uefa.champions",),api_key="")
            self.assertEqual(b["new_completed_match_player_stats"],0)
            self.assertEqual(b["new_future_match_player_contexts"],0)

    def test_unsupported_league_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,"Unsupported soccer competition"):
                run(Path(td)/"private",Path(td)/"reports",
                    observed_at="2026-11-12T12:00:00Z",codes=("unknown",),
                    session=MockESPN(),api_key="")

    def test_redirect_outside_allowed_host_rejected(self):
        target=f"{ROOT}/uefa.champions/scoreboard"
        s=MockESPN(lambda *_:Resp(302,{},{"Location":"https://evil.example/scoreboard"}))
        with self.assertRaisesRegex(ValueError,"unauthorized endpoint"):
            _get(s,target,params={"dates":"20261112"})

    def test_missing_coverage_does_not_fabricate_players(self):
        with tempfile.TemporaryDirectory() as td:
            s=MockESPN(lambda url,p:Resp(200,{"events":[]} if url.endswith("scoreboard") else {}))
            r=run(Path(td)/"private",Path(td)/"report",
                  observed_at="2026-11-12T12:00:00Z",session=s,codes=("esp.1",),
                  api_key="")
            self.assertEqual(r["new_completed_match_player_stats"],0)
            self.assertEqual(r["new_pregame_roster_snapshots_unconfirmed"],0)
            self.assertEqual(r["new_future_match_player_contexts"],0)
            self.assertEqual(r["new_api_football_injury_day_snapshots"],0)


if __name__=="__main__":
    unittest.main()
