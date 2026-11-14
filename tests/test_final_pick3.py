import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd

from kalshi_nba.final_pick import combine3, fit_sets3, walk_forward3


def synth(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.dirichlet([4, 2.5, 3], n)
    y = np.array([rng.choice(3, p=p) for p in t])
    d = pd.DataFrame({"y": y, "date": pd.to_datetime("2025-01-01") + pd.to_timedelta(np.arange(n) // 6, "D")})
    noise = lambda s: np.clip(t + rng.normal(0, s, t.shape), .02, 1)
    for src, s in (("k", .02), ("c", .03), ("m", .15)):
        q = noise(s); q = q / q.sum(1, keepdims=True)
        for i, c in enumerate("HDA"):
            d[f"{src}_{c}"] = q[:, i]
    return d


class TestFinalPick3(unittest.TestCase):
    def test_learns_market_over_noisy_model(self):
        w = fit_sets3(synth())["k,c,m"]
        self.assertGreater(w[2] + w[3], 3 * abs(w[4]))

    def test_combine_fallbacks_and_normalized(self):
        w = {"k,c,m": [0, 0, .5, .5, 0], "c,m": [0, 0, 1, 0], "m": [0, 0, 1]}
        p, src = combine3([.5, .3, .2], [.5, .3, .2], [.4, .3, .3], w)
        self.assertEqual(src, "Kalshi + casas + modelo"); self.assertAlmostEqual(sum(p), 1)
        self.assertEqual(combine3([None] * 3, [.5, .3, .2], [.4, .3, .3], w)[1], "casas + modelo")
        self.assertEqual(combine3(None, None, [.4, .3, .3], {})[1], "modelo")

    def test_walk_forward_only_past(self):
        d = synth(900, 1)
        a = walk_forward3(d)
        d2 = d.copy(); late = d2["date"] > d2["date"].quantile(.8); d2.loc[late, "y"] = (d2.loc[late, "y"] + 1) % 3
        b = walk_forward3(d2)
        early = (d["date"] <= d["date"].quantile(.8)).to_numpy()
        self.assertTrue(np.allclose(a.loc[early, "f_H"].fillna(-1), b.loc[early, "f_H"].fillna(-1)))


if __name__ == "__main__":
    unittest.main()
