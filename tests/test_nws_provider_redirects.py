import unittest
from kalshi_external.nws_weather import _get


class Response:
    def __init__(self,code,body=None,location=None):
        self.status_code=code
        self.content=b"{}"
        self.headers={"Location":location} if location else {}
        self.body=body or {"ok":True}
    def json(self):
        return self.body


class RedirectSession:
    def __init__(self,target):
        self.calls=[]
        self.target=target
    def get(self,url,params=None,headers=None,timeout=None,allow_redirects=None):
        self.calls.append((url,params))
        if len(self.calls)==1:
            return Response(301,location=self.target)
        return Response(200,body={"valid":True})


class TestProviderRedirects(unittest.TestCase):
    def test_first_party_canonical_redirect(self):
        api="https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        target="https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard/"
        s=RedirectSession(target)
        self.assertTrue(_get(s,api,params={"dates":"20261115"})["valid"])
        self.assertEqual(len(s.calls),2)
        self.assertEqual(s.calls[1][1],{"dates":"20261115"})

    def test_redirect_to_untrusted_host_is_never_followed(self):
        api="https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
        s=RedirectSession("https://evil.example/collect?token=123")
        with self.assertRaisesRegex(ValueError,"Untrusted forecast endpoint"):
            _get(s,api,params={"event":"401873501"})
        self.assertEqual(len(s.calls),1)

    def test_trailing_slash_noaa_redirect(self):
        start="https://api.weather.gov/points/44.50097,-88.06168"
        target="https://api.weather.gov/points/44.50097,-88.06168/"
        s=RedirectSession(target)
        self.assertTrue(_get(s,start)["valid"])


if __name__=="__main__":
    unittest.main()
