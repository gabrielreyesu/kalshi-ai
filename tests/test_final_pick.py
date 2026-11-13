import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd

from kalshi_nba.final_pick import combine, fit_sets, load_weights, walk_forward


class TestFinalPick(unittest.TestCase):
    def test_fallback_order(self):
        w = {"k,c,m": [0, .5, .5, 0], "k,m": [0, 1, 0], "c,m": [0, 1, 0], "m": [0, 1]}
        self.assertEqual(combine(.6, .6, .5, w)[1], "Kalshi + casas + modelo")
        self.assertEqual(combine(.6, None, .5, w)[1], "Kalshi + modelo")
        self.assertEqual(combine(np.nan, .6, .5, w)[1], "casas + modelo")
        self.assertEqual(combine(None, None, .55, w)[1], "modelo")
        self.assertAlmostEqual(combine(.6, None, .5, w)[0], .6, places=3)
        self.assertEqual(combine(None, None, None, w), (None, "sin dato"))

    def test_learns_to_trust_informative_source(self):
        rng = np.random.default_rng(0)
        n = 3000
        truth = rng.uniform(.15, .85, n)
        d = pd.DataFrame({"y": (rng.random(n) < truth).astype(int), "k": truth,
                          "c": np.clip(truth + rng.normal(0, .03, n), .02, .98), "m": rng.uniform(.2, .8, n)})
        w = fit_sets(d)["k,c,m"]
        self.assertGreater(w[1] + w[2], 0.7); self.assertLess(abs(w[3]), 0.2)

    def test_walk_forward_no_future(self):
        rng = np.random.default_rng(1)
        n = 1200
        p = rng.uniform(.2, .8, n)
        d = pd.DataFrame({"date": pd.to_datetime("2025-01-01") + pd.to_timedelta(np.arange(n) // 8, "D"),
                          "y": (rng.random(n) < p).astype(int), "k": p, "c": p, "m": p})
        a = walk_forward(d)
        d2 = d.copy(); late = d2["date"] > d2["date"].quantile(.8); d2.loc[late, "y"] = 1 - d2.loc[late, "y"]
        b = walk_forward(d2)
        early = (d["date"] <= d["date"].quantile(.8)).to_numpy()
        self.assertTrue(np.allclose(a["p_final"][early].fillna(-1), b["p_final"][early].fillna(-1)))

    def test_shipped_weights_exist(self):
        for lg in ("nba", "nfl"):
            self.assertIn("k,c,m", load_weights(lg))


if __name__ == "__main__":
    unittest.main()
