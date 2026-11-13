import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd

from kalshi_lineup.travel import compute, km, site


class TestTravel(unittest.TestCase):
    def test_distance_and_moves(self):
        self.assertAlmostEqual(km((40.75, -73.99), (34.04, -118.27)), 3940, delta=40)       # NY - LA
        self.assertEqual(site("nfl", "14", 2015)[2], "America/Chicago")                      # Rams en St. Louis
        self.assertEqual(site("nfl", "14", 2020)[2], "America/Los_Angeles")

    def test_road_trip_and_clock(self):
        g = pd.DataFrame([
            dict(event_id="1", kickoff_utc="2025-01-01T03:00Z", season_year=2025, home_id="13", away_id="18", neutral=False),  # NYK en LAL
            dict(event_id="2", kickoff_utc="2025-01-03T03:00Z", season_year=2025, home_id="9", away_id="18", neutral=False),   # NYK en GSW
            dict(event_id="3", kickoff_utc="2025-01-05T00:00Z", season_year=2025, home_id="18", away_id="13", neutral=True)])
        f = compute(g, "nba").set_index("event_id")
        self.assertGreater(f.loc["1", "travel"], 3.5)                     # NYK viajó ~3.900 km; LAL en casa
        self.assertLess(f.loc["2", "travel"], 0.7)                        # de LA a SF: viaje corto para NYK
        self.assertLess(f.loc["2", "travel"], f.loc["1", "travel"])
        self.assertLess(f.loc["1", "clock"], 0)                           # visitante del este jugando en el oeste
        self.assertEqual(f.loc["3", "travel"], 0.0)                       # neutral: 0


if __name__ == "__main__":
    unittest.main()
