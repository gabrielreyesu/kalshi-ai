import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd

from kalshi_lineup.kalshi_eval import evaluate, expanding_blend
from kalshi_lineup.kalshi_history import price_before

K = pd.Timestamp("2026-01-10T01:00Z")


def candle(hours_before, bid, ask, last=None):
    return {"end_period_ts": int((K - pd.Timedelta(hours=hours_before)).timestamp()),
            "yes_bid": {"close": str(bid)}, "yes_ask": {"close": str(ask)}, "price": {"close": str(last if last is not None else (bid + ask) / 2)}}


class TestKalshiHistory(unittest.TestCase):
    def test_price_before_never_uses_future(self):
        c = [candle(5, .40, .42), candle(2, .50, .52), candle(0.5, .90, .92)]      # la última es a T-30: no vale para T-60
        p = price_before(c, K, 60)
        self.assertAlmostEqual(p["mid"], .51); self.assertAlmostEqual(p["age_h"], 2)
        self.assertTrue(np.isnan(price_before(c, K, 600)["mid"]))

    def test_recent_markets_dollar_fields(self):
        c = [{"end_period_ts": int((K - pd.Timedelta(hours=2)).timestamp()), "yes_bid": {"close_dollars": "0.60"},
              "yes_ask": {"close_dollars": "0.62"}, "price": {"close_dollars": "0.61"}}]
        self.assertAlmostEqual(price_before(c, K, 60)["mid"], .61)

    def test_rams_abbreviation_matches_kalshi(self):
        from kalshi_nba.kalshi import NFL_ABBR
        self.assertEqual(NFL_ABBR["14"], "LA")

    def test_wide_spread_uses_last(self):
        self.assertAlmostEqual(price_before([candle(2, .10, .60, last=.33)], K, 60)["mid"], .33)

    def test_blend_uses_only_past_dates(self):
        rng = np.random.default_rng(0)
        n = 900
        p = rng.uniform(.2, .8, n)
        d = pd.DataFrame({"date": pd.to_datetime("2025-01-01") + pd.to_timedelta(np.arange(n) // 5, "D"), "y": (rng.random(n) < p).astype(int),
                          "k60_mid": p, "p_x": np.clip(p + rng.normal(0, .05, n), .01, .99)})
        b = expanding_blend(d, ["k60_mid"])
        first = d["date"].iloc[b.first_valid_index()]
        self.assertGreater((first - d["date"].min()).days, 0)          # calentamiento sin predicción
        d2 = d.copy(); late = d2["date"] > d2["date"].quantile(.8); d2.loc[late, "y"] = 1 - d2.loc[late, "y"]
        b2 = expanding_blend(d2, ["k60_mid"])
        early = (d["date"] <= d["date"].quantile(.8)).to_numpy()
        self.assertTrue(np.allclose(b[early].fillna(-1), b2[early].fillna(-1)))   # cambiar el futuro no altera el pasado

    def test_evaluate_runs(self):
        rng = np.random.default_rng(1)
        n = 800
        p = rng.uniform(.2, .8, n)
        d = pd.DataFrame({"event_id": [str(i) for i in range(n)], "date": pd.to_datetime("2025-01-01") + pd.to_timedelta(np.arange(n) // 6, "D"),
                          "y": (rng.random(n) < p).astype(int), "k60_mid": p, "k360_mid": p, "k1440_mid": p, "book_p_home": p,
                          "p_base": np.clip(p + rng.normal(0, .1, n), .02, .98), "p_var": np.clip(p + rng.normal(0, .08, n), .02, .98)})
        r = evaluate(d)
        self.assertEqual(r["n"], n)
        self.assertLess(r["H1_modelo_vs_kalshi"]["mejora"], 0)          # el ruido empeora al modelo frente a la verdad


if __name__ == "__main__":
    unittest.main()
