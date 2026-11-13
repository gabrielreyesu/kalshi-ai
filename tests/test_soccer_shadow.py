import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd

from test_soccer_xi import PAY, make_soccer
from kalshi_lineup.soccer_shadow import score, snapshot, upcoming
from kalshi_lineup.soccer_xi import parse_soccer_summary


def with_future(g, minutes=40):
    now = g["kick"].max() + pd.Timedelta(days=3)
    k = now + pd.Timedelta(minutes=minutes)
    fut = dict(event_id="F1", code="uefa.champions", kickoff_utc=k.isoformat(), status="STATUS_SCHEDULED", home_id="T0",
               away_id="T1", home="T0", away="T1", hs=np.nan, as_=np.nan, neutral=False, final=False, kick=k,
               date=k.tz_localize(None).normalize(), target=True, competicion="x", group="clubes")
    return pd.concat([g, pd.DataFrame([fut])], ignore_index=True), now


def xi(team, rot):
    return [{"pid": f"{team}-{i}", "team_id": f"T{team}", "starter": True, "ga": 0.0} for i in list(range(rot, 11)) + list(range(11, 11 + rot))]


class TestSoccerShadow(unittest.TestCase):
    def test_pregame_parser(self):
        pre = {**PAY, "header": {"competitions": [{"status": {"type": {"name": "STATUS_SCHEDULED", "completed": False}}}]}}
        self.assertIsNotNone(parse_soccer_summary(pre, "1", pregame=True))
        self.assertIsNone(parse_soccer_summary(PAY, "1", pregame=True))            # ya terminado: no es pregame

    def test_window_and_rotation_effect(self):
        g, b = make_soccer(seasons=range(2016, 2021))
        g, now = with_future(g)
        self.assertEqual(list(upcoming(g, now)["event_id"]), ["F1"])
        self.assertTrue(upcoming(g, now + pd.Timedelta(minutes=38)).empty)          # menos de 5 min: fuera de ventana
        full, _ = snapshot(now, g=g, boxes=b, lineups={"F1": {"event_id": "F1", "players": xi(0, 0) + xi(1, 0)}})
        rot, _ = snapshot(now, g=g, boxes=b, lineups={"F1": {"event_id": "F1", "players": xi(0, 5) + xi(1, 0)}})
        self.assertEqual(len(full), 1)
        self.assertLess(rot["xi_cont"].iloc[0], full["xi_cont"].iloc[0])
        self.assertAlmostEqual(rot["base_H"].iloc[0], full["base_H"].iloc[0])
        none, _ = snapshot(now, g=g, boxes=b, lineups={})
        self.assertTrue(none.empty)

    def test_score(self):
        sh = pd.DataFrame([dict(snapshot_utc="2026-01-01T19:20Z", event_id="1", kickoff_utc="2026-01-01T20:00Z",
                                base_H=.5, base_D=.3, base_A=.2, xi_H=.6, xi_D=.25, xi_A=.15, minutes_to_kick=40)])
        g = pd.DataFrame([dict(event_id="1", final=True, status="STATUS_FULL_TIME", hs=2.0, as_=0.0)])
        r = score(sh, g)
        self.assertEqual(r["n"], 1); self.assertAlmostEqual(r["logloss_xi"], -np.log(.6), places=4)


if __name__ == "__main__":
    unittest.main()
