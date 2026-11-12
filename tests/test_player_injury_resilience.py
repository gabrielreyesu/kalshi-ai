import unittest
from unittest.mock import patch

from kalshi_external.player_intel import parse_injury_feed
from kalshi_external.player_collect import _get, API_ROOT


class Response:
    def __init__(self,status,body,headers=None):
        self.status_code=status
        self.body=body
        self.content=b'{}'
        self.headers=headers or {}
    def json(self):return self.body


class SequenceSession:
    def __init__(self,*responses):self.responses=list(responses);self.calls=[]
    def get(self,url,params=None,headers=None,timeout=None,allow_redirects=None):
        self.calls.append(url)
        return self.responses.pop(0)


class TestInjuryProviderResilience(unittest.TestCase):
    def test_nfl_nested_team_schema_and_fullname(self):
        payload={"injuries":[{"team":{"id":"19","displayName":"New York Giants"},
            "injuries":[{"athlete":{"id":"111","fullName":"NFL Player",
                         "position":{"abbreviation":"QB"}},
                        "status":"Questionable","date":"2026-11-12T09:00:00Z"}]}]}
        r=parse_injury_feed(payload,"nfl","2026-11-12T11:00:00Z")
        self.assertEqual(len(r["reported_athletes"]),1)
        self.assertEqual(r["reported_athletes"][0]["team_id"],"19")
        self.assertEqual(r["reported_athletes"][0]["team"],"New York Giants")
        self.assertEqual(r["reported_athletes"][0]["player"],"NFL Player")

    def test_retries_transient_http_without_extra_secrets(self):
        seq=SequenceSession(Response(503,{}),Response(429,{}),
                            Response(200,{"injuries":[]}))
        with patch("kalshi_external.player_collect.time.sleep"):
            r=_get(seq,f"{API_ROOT}/football/nfl/injuries")
        self.assertEqual(r["injuries"],[])
        self.assertEqual(len(seq.calls),3)

    def test_safe_same_host_redirect(self):
        target=f"{API_ROOT}/football/nfl/injuries"
        seq=SequenceSession(Response(302,{},{"Location":target}),
                            Response(200,{"injuries":[]}))
        result=_get(seq,target)
        self.assertEqual(result,{"injuries":[]})
        self.assertEqual(len(seq.calls),2)

    def test_redirect_to_foreign_host_refused(self):
        target=f"{API_ROOT}/football/nfl/injuries"
        seq=SequenceSession(Response(302,{},{"Location":"https://other.example/injuries"}))
        with self.assertRaisesRegex(ValueError,"Unexpected ESPN data host"):
            _get(seq,target)

    def test_http_error_has_clear_code(self):
        seq=SequenceSession(Response(404,{}))
        with self.assertRaisesRegex(RuntimeError,"ESPN_HTTP_404"):
            _get(seq,f"{API_ROOT}/football/nfl/injuries")

    def test_invalid_redirect_path_refused(self):
        seq=SequenceSession(Response(302,{},{"Location":"https://site.api.espn.com/other"}))
        with self.assertRaisesRegex(ValueError,"Unexpected ESPN data path"):
            _get(seq,f"{API_ROOT}/football/nfl/injuries")


if __name__=="__main__":
    unittest.main()
