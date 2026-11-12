import tempfile
import unittest
from pathlib import Path
import pandas as pd
from kalshi_external.nws_weather import run


class DeniedResponse:
    status_code=403
    content=b"SENSITIVE_RESPONSE_BODY_MUST_NOT_LEAK"
    def json(self):
        return {}


class DeniedSession:
    def get(self,*args,**kwargs):
        return DeniedResponse()


class TestProviderErrors(unittest.TestCase):
    def test_http_error_is_visible_but_not_body(self):
        with tempfile.TemporaryDirectory() as d:
            registry=Path(d)/"nfl.csv"
            pd.DataFrame([{
                "event_id":"401873501","home":"Green Bay Packers","away":"Chicago Bears",
                "kickoff_utc":"2026-11-15T18:00:00Z"
            }]).to_csv(registry,index=False)
            report=run(registry,Path(d)/"weather.jsonl",Path(d)/"status.json",
                observed_at="2026-11-12T20:00:00Z",session=DeniedSession())
            self.assertEqual(report["matched_weather_events"],0)
            self.assertEqual(len(report["warnings"]),1)
            self.assertIn("HTTP 403",report["warnings"][0])
            self.assertNotIn("SENSITIVE_RESPONSE_BODY_MUST_NOT_LEAK",
                             Path(d,"status.json").read_text())

if __name__=="__main__":
    unittest.main()
