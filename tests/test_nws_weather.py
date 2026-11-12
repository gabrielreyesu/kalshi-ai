import tempfile
import unittest
from pathlib import Path

import pandas as pd

from kalshi_external.nws_weather import (
    _get, choose_hour, forecast_event, make_weather, run,
)


class Response:
    status_code = 200
    content = b"{}"
    def __init__(self, payload):
        self.payload=payload
    def json(self):
        return self.payload


class Session:
    def __init__(self, venue="Lambeau Field", updated="2026-11-12T12:00:00Z",
                 forecast_url="https://api.weather.gov/gridpoints/GRB/10,20/forecast/hourly"):
        self.calls=[]
        self.venue=venue
        self.updated=updated
        self.forecast_url=forecast_url

    def get(self,url,params=None,headers=None,timeout=None,allow_redirects=None):
        self.calls.append((url,params))
        if "espn.com" in url:
            return Response({"header":{"competitions":[{"venue":{"fullName":self.venue}}]}})
        if "/points/" in url:
            return Response({"properties":{"forecastHourly":self.forecast_url}})
        if "/gridpoints/" in url:
            return Response({"properties":{"updated":self.updated,"periods":[{
                "startTime":"2026-11-15T12:00:00Z","endTime":"2026-11-15T13:00:00Z",
                "temperature":2,"temperatureUnit":"C","windSpeed":"5 to 15 mph",
                "windDirection":"NW","probabilityOfPrecipitation":{"value":65},
                "shortForecast":"Breezy, chance of rain"
            }]}})
        raise ValueError("unknown mock url")


def fixture():
    return {
        "event_id":"401873501","home":"Green Bay Packers","away":"Chicago Bears",
        "kickoff_utc":"2026-11-15T12:30:00Z"
    }


class TestNOAAWeather(unittest.TestCase):
    def test_stadium_and_forecast_are_verified(self):
        session=Session()
        out=forecast_event(session,fixture(),"2026-11-12T14:00:00Z")
        self.assertEqual(out["source_id"],"noaa_nws_hourly")
        self.assertEqual(out["venue"],"Lambeau Field")
        self.assertAlmostEqual(out["temperature_f"],35.6)
        self.assertEqual(out["wind_speed_max_mph"],15)
        self.assertEqual(out["precip_probability_pct"],65)
        self.assertEqual(len(session.calls),3)

    def test_nonmatching_stadium_fails_closed(self):
        session=Session(venue="Wembley Stadium")
        self.assertIsNone(forecast_event(session,fixture(),"2026-11-12T14:00:00Z"))
        self.assertEqual(len(session.calls),1)

    def test_reject_weather_api_redirect_or_untrusted_host(self):
        with self.assertRaisesRegex(ValueError,"Untrusted forecast endpoint"):
            _get(Session(),"https://evil.example/gridpoints/GRB/10,20/forecast/hourly")
        s=Session(forecast_url="https://bad.example/weather")
        with self.assertRaisesRegex(ValueError,"Untrusted forecast endpoint"):
            forecast_event(s,fixture(),"2026-11-12T14:00:00Z")

    def test_future_issue_date_rejected(self):
        s=Session(updated="2026-11-17T14:00:00Z")
        with self.assertRaisesRegex(ValueError,"issued in the future"):
            forecast_event(s,fixture(),"2026-11-12T14:00:00Z")

    def test_unknown_venue_and_dome_ignored(self):
        d=fixture()
        d["home"]="Dallas Cowboys"
        s=Session()
        self.assertIsNone(forecast_event(s,d,"2026-11-12T14:00:00Z"))
        self.assertFalse(s.calls)

    def test_only_hour_overlapping_kickoff_used(self):
        self.assertIsNone(choose_hour([{"startTime":"2026-11-15T11:00:00Z",
              "endTime":"2026-11-15T12:00:00Z"}],"2026-11-15T12:30:00Z"))

    def test_two_distinct_runs_persist_and_no_extra_api_keys(self):
        with tempfile.TemporaryDirectory() as td:
            registry=Path(td)/"nfl.csv"
            history=Path(td)/"weather.jsonl"
            status=Path(td)/"status.json"
            pd.DataFrame([fixture()]).to_csv(registry,index=False)
            a=run(registry,history,status,observed_at="2026-11-12T14:00:00Z",session=Session())
            b=run(registry,history,status,observed_at="2026-11-12T16:00:00Z",session=Session())
            self.assertEqual(a["new_forecast_snapshots"],1)
            self.assertEqual(b["new_forecast_snapshots"],1)
            self.assertEqual(len(history.read_text().splitlines()),2)
            self.assertFalse(a["production_changed"])
            c=run(registry,history,status,observed_at="2026-11-12T16:00:00Z",session=Session())
            self.assertEqual(c["new_forecast_snapshots"],0)


if __name__=="__main__":
    unittest.main()
