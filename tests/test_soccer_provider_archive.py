import json
import tempfile
import unittest
from pathlib import Path

from kalshi_external.soccer_player_collect import run
from kalshi_external.soccer_player_audit import audit, read_jsonl


class Response:
    status_code=200
    content=b'{"ok":true}'
    headers={}
    def __init__(self,obj):self.obj=obj
    def json(self):return self.obj


class MultisourceSession:
    def __init__(self):self.urls=[]
    def get(self,url,params=None,headers=None,timeout=None,allow_redirects=False):
        self.urls.append((url,params))
        if url.endswith("/scoreboard"):
            if params["dates"]!="20261113":
                return Response({"events":[]})
            return Response({"events":[{
                "id":"3001","date":"2026-11-13T19:00:00Z",
                "status":{"type":{"name":"STATUS_SCHEDULED"}},
                "competitions":[{"competitors":[
                    {"homeAway":"home","team":{"id":"10","displayName":"Arsenal"}},
                    {"homeAway":"away","team":{"id":"20","displayName":"Barcelona"}}]}]
            }]})
        if url.endswith("/injuries"):
            day=params["date"]
            if day!="2026-11-13":
                return Response({"response":[]})
            return Response({"response":[{
                "fixture":{"id":777,"date":"2026-11-13T19:00:00Z"},
                "league":{"name":"Champions League"},
                "team":{"id":55,"name":"Arsenal"},
                "player":{"id":501,"name":"Forward Out","type":"Missing Fixture","reason":"Suspended"}
            }]})
        if url.endswith("/fixtures"):
            day=params["date"]
            if day!="2026-11-13":
                return Response({"response":[]})
            return Response({"response":[{
                "fixture":{"id":777,"date":"2026-11-13T19:00:00Z"},
                "teams":{"home":{"id":55,"name":"Arsenal"},
                         "away":{"id":66,"name":"Barcelona"}}
            }]})
        raise ValueError("Unexpected provider request")


class SoccerVendorArchiveTests(unittest.TestCase):
    def test_keep_injury_context_on_second_run_same_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            session=MultisourceSession()
            first=run(root/"private",root/"report",
                      observed_at="2026-11-12T12:00:00Z",
                      session=session,codes=("uefa.champions",),
                      api_key="test-key")
            self.assertEqual(first["new_api_football_injury_day_snapshots"],2)
            second=run(root/"private",root/"report",
                       observed_at="2026-11-12T13:00:00Z",
                       session=MultisourceSession(),codes=("uefa.champions",),
                       api_key="test-key")
            self.assertEqual(second["new_api_football_injury_day_snapshots"],0)
            entries=read_jsonl(root/"private"/"soccer_previews.jsonl")
            self.assertEqual(len(entries),2)
            self.assertEqual(entries[0]["provider_injury_context"][0]["availability"],"reported_absent")
            self.assertEqual(entries[1]["provider_injury_context"][0]["player"],"Forward Out")
            reports=read_jsonl(root/"private"/"soccer_injuries.jsonl")
            self.assertEqual(len(reports),2)
            self.assertEqual(audit([],[],reports,entries,now="2026-11-12T13:10:00Z")
                             ["record_counts"]["injuries"],2)

    def test_vendor_injuries_remain_optional(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            report=run(root/"private",root/"report",
                       observed_at="2026-11-12T12:00:00Z",
                       session=MultisourceSession(),codes=("uefa.champions",),
                       api_key="")
            self.assertFalse(report["api_football_configured"])
            previews=read_jsonl(root/"private"/"soccer_previews.jsonl")
            self.assertEqual(previews[0]["provider_injury_context"],[])
            self.assertFalse(previews[0].get("lineup_confirmed",False))


if __name__=="__main__":
    unittest.main()
