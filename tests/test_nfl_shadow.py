import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd

from test_nfl_qb import make_nfl
from kalshi_lineup.nfl_shadow import depth_qbs, expected_starter, snapshot

DEPTH = {"depthchart": [{"positions": {"lde": {"athletes": [{"id": "9"}]}}},
                        {"positions": {"qb": {"athletes": [{"id": "1", "displayName": "Uno"}, {"id": "2", "displayName": "Dos"}]}}}]}


class TestNFLShadow(unittest.TestCase):
    def test_depth_and_starter(self):
        self.assertEqual(depth_qbs(DEPTH), ["1", "2"])
        self.assertEqual(expected_starter(["1", "2"], {"1"}), "2")
        self.assertIsNone(expected_starter([], set()))

    def test_backup_lowers_team_probability(self):
        g, b = make_nfl(seasons=range(2014, 2020))
        nxt = g["date"].max() + pd.Timedelta(days=7)
        fut = pd.DataFrame([dict(event_id="F", date=nxt, kickoff_utc=(nxt + pd.Timedelta(hours=18)).tz_localize("UTC").isoformat(),
                                 season_year=2019, season_type=2, status="STATUS_SCHEDULED", home_id="T0", away_id="T1",
                                 home="T0", away="T1", hs=np.nan, as_=np.nan, neutral=False, line_spread=np.nan,
                                 line_total=np.nan, line_p_home=np.nan)])
        games = pd.concat([g, fut], ignore_index=True)
        now = pd.Timestamp(nxt, tz="UTC") + pd.Timedelta(hours=10)
        starter, _ = snapshot(now, 48, games=games, boxes=b, depth={"T0": ["0-s"], "T1": ["1-s"]}, injuries={"injuries": []})
        backup, _ = snapshot(now, 48, games=games, boxes=b, depth={"T0": ["0-s", "0-new"], "T1": ["1-s"]},
                             injuries={"injuries": [{"id": "T0", "injuries": [{"status": "Out", "athlete": {"id": "0-s"}}]}]})
        self.assertEqual(backup["qb_home"].iloc[0], "0-new")
        self.assertEqual(int(backup["qb_change_home"].iloc[0]), 1)
        self.assertLess(backup["p_lineup"].iloc[0], starter["p_lineup"].iloc[0])
        self.assertAlmostEqual(backup["p_base"].iloc[0], starter["p_base"].iloc[0])


if __name__ == "__main__":
    unittest.main()
