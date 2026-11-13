import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd

from test_lineup import make_league
from kalshi_lineup.shadow import norm, parse_injuries, score, snapshot

INJ = {"injuries": [{"id": "T0", "injuries": [
    {"status": "Out", "athlete": {"displayName": "Star Zero", "links": [{"href": "https://espn.com/nba/player/_/id/0-0/x"}]}},
    {"status": "Questionable", "athlete": {"id": "0-1", "displayName": "B"}},
    {"status": "Out", "athlete": {"displayName": "Nobody Known"}}]}]}


class TestShadow(unittest.TestCase):
    def test_parse_and_norm(self):
        rows = parse_injuries(INJ)
        self.assertEqual(rows[0]["pid"], "0-0"); self.assertEqual(rows[1]["status"], "questionable")
        self.assertEqual(norm("Luka Dončić Jr."), norm("luka doncic"))

    def test_snapshot_star_out_lowers_home_prob(self):
        g, b = make_league(seasons=range(2016, 2020))
        last = g["date"].max()
        nxt = last + pd.Timedelta(days=2)
        fut = pd.DataFrame([dict(event_id="F1", date=nxt, kickoff_utc=(nxt + pd.Timedelta(hours=23)).tz_localize("UTC").isoformat(),
                                 season_year=2019, season_type=2, status="STATUS_SCHEDULED", home_id="T0", away_id="T1",
                                 home="T0", away="T1", hs=np.nan, as_=np.nan, neutral=False, line_spread=np.nan,
                                 line_total=np.nan, line_p_home=np.nan)])
        games = pd.concat([g, fut], ignore_index=True)
        now = pd.Timestamp(nxt, tz="UTC") + pd.Timedelta(hours=20)
        healthy, _ = snapshot(now, 36, injuries={"injuries": []}, games=games, boxes=b)
        hurt, _ = snapshot(now, 36, injuries=INJ, games=games, boxes=b)
        self.assertEqual(len(hurt), 1)
        self.assertEqual(int(hurt["unmatched_out"].iloc[0]), 1)
        self.assertLess(hurt["p_lineup"].iloc[0], healthy["p_lineup"].iloc[0])
        self.assertAlmostEqual(hurt["p_base"].iloc[0], healthy["p_base"].iloc[0])

    def test_score_uses_last_pregame_snapshot(self):
        sh = pd.DataFrame([
            dict(snapshot_utc="2026-01-01T10:00Z", event_id="1", kickoff_utc="2026-01-02T00:00Z", p_base=.6, p_lineup=.7, minutes_to_tip=840),
            dict(snapshot_utc="2026-01-02T01:00Z", event_id="1", kickoff_utc="2026-01-02T00:00Z", p_base=.1, p_lineup=.1, minutes_to_tip=-60)])
        games = pd.DataFrame([dict(event_id="1", status="STATUS_FINAL", hs=100.0, as_=90.0)])
        r = score(sh, games)
        self.assertEqual(r["n"], 1)
        self.assertAlmostEqual(r["logloss_lineup"], -np.log(.7), places=4)


if __name__ == "__main__":
    unittest.main()
