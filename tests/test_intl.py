import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd

from kalshi_intl.model import IntlEngine, IntlModel, IntlConfig, FEATS, walk_forward
from kalshi_intl.odds import consensus
from kalshi_intl.data import months, _parse


def synth(n_days=420, teams=24, seed=0, signal=True):
    rng = np.random.default_rng(seed)
    s = rng.normal(0, 0.35, teams) if signal else np.zeros(teams)
    rows, eid = [], 0
    for d in range(n_days):
        date = pd.Timestamp("2018-08-01") + pd.Timedelta(days=4 * d)
        order = rng.permutation(teams)
        for i in range(6):
            h, a = int(order[2 * i]), int(order[2 * i + 1]); neutral = bool(i == 5); eid += 1
            hs = rng.poisson(np.exp(0.25 + (0 if neutral else 0.2) + s[h] - s[a])); as_ = rng.poisson(np.exp(0.25 + s[a] - s[h]))
            rows.append(dict(event_id=str(eid), code="uefa.champions" if i < 3 else "eng.1", kickoff_utc=date.isoformat(), kick=date.tz_localize("UTC"),
                             date=date, status="STATUS_FULL_TIME", home_id=f"T{h}", away_id=f"T{a}", home=f"T{h}", away=f"T{a}",
                             hs=float(hs), as_=float(as_), neutral=neutral, final=True, target=i < 3, competicion="Champions League"))
    return pd.DataFrame(rows)


class TestIntl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = synth(); cls.f = IntlEngine(IntlConfig(k_league=4.0, carry=0.85)).process(cls.g)

    def test_no_future_leak(self):
        cut = self.g["date"].quantile(0.6); g2 = self.g.copy(); late = g2["date"] > cut
        g2.loc[late, "hs"] = 9.0; g2.loc[late, "as_"] = 0.0
        f2 = IntlEngine(IntlConfig(k_league=4.0, carry=0.85)).process(g2)
        a, b = self.f[self.f["date"] <= cut], f2[f2["date"] <= cut]
        for c in FEATS:
            self.assertTrue(np.allclose(a[c].astype(float), b[c].astype(float)), c)

    def test_neutral_removes_home_advantage(self):
        e = IntlEngine(IntlConfig(hfa=80))
        d = pd.Timestamp("2024-01-01")
        self.assertEqual(e.predict(d, "A", "B", neutral=True)["elo_diff"], 0.0)
        self.assertEqual(e.predict(d, "A", "B", neutral=False)["elo_diff"], 80.0)

    def test_model_probabilities_and_signal(self):
        r = walk_forward(self.f, first_year=2020, min_train=500)
        self.assertTrue(np.allclose(r[["p_H", "p_D", "p_A"]].sum(axis=1), 1.0, atol=1e-6))
        i = r["res"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        ll = lambda pre: -np.log(r[[f"{pre}_{c}" for c in "HDA"]].to_numpy(float)[np.arange(len(r)), i]).mean()
        self.assertLess(ll("p"), ll("base"))

    def test_no_signal_no_edge(self):
        f = IntlEngine(IntlConfig()).process(synth(signal=False, seed=3))
        r = walk_forward(f, first_year=2020, min_train=500)
        i = r["res"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        ll = lambda pre: -np.log(r[[f"{pre}_{c}" for c in "HDA"]].to_numpy(float)[np.arange(len(r)), i]).mean()
        self.assertGreater(ll("p"), ll("base") - 0.01)

    def test_consensus_and_helpers(self):
        it = lambda h, d, a: {"provider": {"name": "x"}, "homeTeamOdds": {"moneyLine": h}, "drawOdds": {"moneyLine": d}, "awayTeamOdds": {"moneyLine": a},
                              "overUnder": 2.5, "overOdds": -110, "underOdds": -110}
        c = consensus([it(-150, 280, 400), it(-145, 275, 390), {"provider": {"name": "x Live"}, "homeTeamOdds": {"moneyLine": 900}}])
        self.assertAlmostEqual(c["mkt_H"] + c["mkt_D"] + c["mkt_A"], 1.0); self.assertGreater(c["mkt_H"], 0.5); self.assertAlmostEqual(c["mkt_over"], 0.5)
        self.assertEqual(consensus([]), {})
        from datetime import date
        self.assertEqual(list(months(2025, date(2025, 9, 15))), ["202507", "202508", "202509", "202510"])
        self.assertIsNone(_parse({"competitions": []}, "x"))

    def test_kalshi_matching_requires_unique_clear_match(self):
        from kalshi_intl.cli import match_kalshi
        k = pd.Timestamp("2026-11-25T20:00Z"); o = lambda n: {"name": n, "bid": 0.3, "ask": 0.34, "last": 0.32}
        ev = [{"event": "E1", "when": k, "teams": [o("Arsenal"), o("Dortmund")], "tie": o("Tie")}]
        m = match_kalshi("Arsenal", "Borussia Dortmund", k, ev)
        self.assertEqual(m["k_event"], "E1"); self.assertAlmostEqual(m["k_mid_H"], 0.32)
        self.assertEqual(match_kalshi("Arsenal", "Real Madrid", k, ev), {})
        self.assertEqual(match_kalshi("Arsenal", "Borussia Dortmund", k + pd.Timedelta(days=5), ev), {})
