import unittest
from kalshi_external.nws_weather import forecast_event


class Response:
    status_code=200
    content=b"{}"
    def __init__(self,body):
        self.body=body
    def json(self):
        return self.body


class VenueSession:
    def __init__(self, alternate="Lambeau Field", no_hour=False):
        self.alternate=alternate
        self.no_hour=no_hour
        self.calls=[]

    def get(self,url,params=None,headers=None,timeout=None,allow_redirects=None):
        self.calls.append((url,params))
        if url.endswith("/summary"):
            return Response({"header":{"competitions":[{}]}})
        if url.endswith("/scoreboard"):
            return Response({"events":[{
                "id":"401873501",
                "competitions":[{"venue":{"fullName":self.alternate}}]
            }]})
        if "/points/" in url:
            return Response({"properties":{
                "forecastHourly":"https://api.weather.gov/gridpoints/GRB/10,20/forecast/hourly"
            }})
        if "/gridpoints/" in url:
            return Response({"properties":{"updated":"2026-11-12T12:00:00Z",
                "periods":[] if self.no_hour else [{
                    "startTime":"2026-11-15T12:00:00Z",
                    "endTime":"2026-11-15T13:00:00Z",
                    "temperature":31,"temperatureUnit":"F",
                    "windSpeed":"5 mph","probabilityOfPrecipitation":{"value":35}
                }]
            }})
        raise ValueError("Mock encountered unknown host")


def fixture():
    return {"event_id":"401873501",
            "home":"Green Bay Packers","away":"Chicago Bears",
            "kickoff_utc":"2026-11-15T12:30:00Z"}


class TestVenueFallback(unittest.TestCase):
    def test_scoreboard_fallback_works(self):
        session=VenueSession()
        r=forecast_event(session,fixture(),"2026-11-12T14:00:00Z")
        self.assertEqual(r["venue"],"Lambeau Field")
        self.assertTrue(any(url.endswith("/scoreboard") for url,_ in session.calls))

    def test_wrong_stadium_does_not_get_weather(self):
        session=VenueSession(alternate="Wembley Stadium")
        issues=[]
        self.assertIsNone(forecast_event(session,fixture(),"2026-11-12T14:00:00Z",diagnostics=issues))
        self.assertIn("verified venue differs",issues[0])
        self.assertFalse(any("api.weather.gov" in url for url,_ in session.calls))

    def test_unavailable_forecast_hour_logged(self):
        issues=[]
        self.assertIsNone(forecast_event(VenueSession(no_hour=True),fixture(),
                         "2026-11-12T14:00:00Z",diagnostics=issues))
        self.assertIn("forecast hour unavailable",issues[0])


if __name__=="__main__":
    unittest.main()
