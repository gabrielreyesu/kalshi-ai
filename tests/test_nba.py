import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from scipy.stats import norm

from synth_nba import make_nba
from kalshi_nba.engine import NBAEngine, FEATURES
from kalshi_nba.models import NBAModels
from kalshi_nba.backtest import walk_forward, logloss


class TestNBAEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = make_nba(signal=True, seasons=range(2016, 2020))
        cls.f = NBAEngine().process(cls.g)

    def test_no_future_leak(self):
        cut = self.g["date"].quantile(0.6)
        g2 = self.g.copy(); late = g2["date"] > cut
        rng = np.random.default_rng(3)
        g2.loc[late, "hs"] = rng.integers(80, 140, late.sum()).astype(float); g2.loc[late, "as_"] = rng.integers(80, 140, late.sum()).astype(float) + 0.5
        f2 = NBAEngine().process(g2)
        a, b = self.f[self.f["date"] <= cut].reset_index(drop=True), f2[f2["date"] <= cut].reset_index(drop=True)
        for c in FEATURES + ["pf_h", "pa_a", "league_total"]:
            self.assertTrue(np.allclose(a[c].astype(float), b[c].astype(float), equal_nan=True, atol=1e-12), c)

    def test_incremental_equals_batch(self):
        cut = self.g["date"].quantile(0.7)
        e = NBAEngine(); e.process(self.g[self.g["date"] <= cut])
        part = e.process(self.g[self.g["date"] > cut]).reset_index(drop=True)
        full = self.f[self.f["date"] > cut].reset_index(drop=True)
        for c in ("elo_diff", "form_diff", "net15_diff"):
            self.assertTrue(np.allclose(part[c], full[c], atol=1e-9), c)

    def test_same_team_twice_in_a_day_rejected(self):
        bad = self.g.head(4).copy(); bad["date"] = bad["date"].iloc[0]
        bad.loc[bad.index[1], "home_id"] = bad.loc[bad.index[0], "home_id"]
        with self.assertRaises(ValueError):
            NBAEngine().process(bad)

    def test_elo_is_zero_sum(self):
        e = NBAEngine(); e.process(self.g)
        self.assertAlmostEqual(np.mean(list(e.elo.values())), e.cfg.elo_mean, delta=15)

    def test_future_game_does_not_update_state(self):
        e = NBAEngine(); e.process(self.g)
        n0, elo0 = dict(e.n_played), dict(e.elo)
        e.predict(pd.Timestamp("2030-01-01"), 2030, "T1", "T2")
        self.assertEqual(dict(e.n_played), n0)


class TestNBABacktest(unittest.TestCase):
    def _run(self, signal):
        g = make_nba(signal=signal, seasons=range(2016, 2022))
        f = NBAEngine().process(g)
        return walk_forward(f, first_test=2018, min_train=1500)

    def test_no_edge_without_signal(self):
        r = self._run(False)
        self.assertGreater(float(logloss(r["p_home"], r["home_win"]).mean()), float(logloss(r["base_win"], r["home_win"]).mean()) - 0.004)

    def test_signal_detected_and_market_bounds(self):
        r = self._run(True)
        ll_m, ll_b = logloss(r["p_home"], r["home_win"]).mean(), logloss(r["base_win"], r["home_win"]).mean()
        self.assertLess(ll_m, ll_b - 0.01)
        mae_model = np.mean(np.abs(r["total_pts"] - r["mu_total"])); mae_flat = np.mean(np.abs(r["total_pts"] - r["total_pts"].mean()))
        self.assertLess(mae_model, mae_flat)
        self.assertGreater(r["sig_m"].min(), 5)


class TestDerived(unittest.TestCase):
    def test_probabilities_monotone(self):
        g = make_nba(signal=True, seasons=range(2016, 2019)); f = NBAEngine().process(g)
        m = NBAModels().fit(f); pr = m.predict(f.head(50))
        p1, p2 = m.p_total_over(pr["mu_total"], 215), m.p_total_over(pr["mu_total"], 235)
        self.assertTrue((p1 >= p2).all())
        c1, c2 = m.p_home_wins_by_more_than(pr["mu_margin"], 2.5), m.p_home_wins_by_more_than(pr["mu_margin"], 7.5)
        self.assertTrue((c1 >= c2).all())
        self.assertTrue(((pr["p_home"] > 0) & (pr["p_home"] < 1)).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestKalshiNBA(unittest.TestCase):
    def setUp(self):
        from kalshi_nba.kalshi import parse_markets
        mk = lambda t, ev, title, strike=None: {"ticker": t, "event_ticker": ev, "title": title, "floor_strike": strike,
                                                 "occurrence_datetime": "2026-11-12T05:00:00Z", "close_time": "2026-11-14T00:00:00Z",
                                                 "yes_bid_dollars": "0.40", "yes_ask_dollars": "0.44", "last_price_dollars": "0.42", "volume_fp": "5"}
        self.mk = pd.concat([
            parse_markets("winner", [mk("KXNBAGAME-26NOV11ORLUTA-UTA", "KXNBAGAME-26NOV11ORLUTA", "Utah wins"),
                                     mk("KXNBAGAME-26NOV11ORLUTA-ORL", "KXNBAGAME-26NOV11ORLUTA", "Orlando wins")]),
            parse_markets("spread", [mk("KXNBASPREAD-26NOV11ORLUTA-UTA8", "KXNBASPREAD-26NOV11ORLUTA", "Utah wins by over 7.5", 7.5),
                                     mk("KXNBASPREAD-26NOV11ORLUTA-ORL4", "KXNBASPREAD-26NOV11ORLUTA", "Orlando wins by over 3.5", 3.5)]),
            parse_markets("total", [mk("KXNBATOTAL-26NOV11ORLUTA-230", "KXNBATOTAL-26NOV11ORLUTA", "Over 229.5", 229.5)])], ignore_index=True)
        self.games = pd.DataFrame([dict(event_id="9", kickoff_utc="2026-11-12T05:00Z", home_id="26", away_id="19")])   # Utah local

    def test_attach_orientation_and_settle(self):
        from kalshi_nba.kalshi import attach_games, settle, model_probability
        a = attach_games(self.mk, self.games)
        self.assertEqual(len(a), 5)
        self.assertEqual(dict(zip(a.ticker.str.rsplit("-", n=1).str[1], a.side))["UTA"], "home")
        self.assertEqual(dict(zip(a.ticker.str.rsplit("-", n=1).str[1], a.side))["ORL4"], "away")
        s = settle(a, pd.DataFrame([dict(event_id="9", hs=110.0, as_=115.0)]))       # Orlando gana por 5, total 225
        y = dict(zip(s.ticker.str.rsplit("-", n=1).str[1], s.y))
        self.assertEqual(y, {"UTA": 0, "ORL": 1, "UTA8": 0, "ORL4": 1, "230": 0})
        g = make_nba(signal=True, seasons=range(2016, 2019)); m = NBAModels().fit(NBAEngine().process(g))
        preds = pd.DataFrame([dict(event_id="9", p_home=0.6, mu_margin=3.0, mu_total=225.0)])
        d = model_probability(a, preds, m).set_index(a.ticker.str.rsplit("-", n=1).str[1].values)
        self.assertAlmostEqual(d.loc["UTA", "p_model"] + d.loc["ORL", "p_model"], 1.0)
        self.assertAlmostEqual(d.loc["UTA8", "p_model"], 1 - norm.cdf((7.5 - 3.0) / m.sig_m))
        self.assertAlmostEqual(d.loc["ORL4", "p_model"], norm.cdf((-3.5 - 3.0) / m.sig_m))
        self.assertLess(d.loc["230", "p_model"], 0.5)

    def test_unmatched_event_is_dropped(self):
        from kalshi_nba.kalshi import attach_games
        far = self.games.assign(kickoff_utc="2026-11-20T05:00Z")
        self.assertEqual(len(attach_games(self.mk, far)), 0)


class TestConsensusLine(unittest.TestCase):
    def test_median_ignores_corrupt_book(self):
        from kalshi_nba.data import pick_line
        items = [dict(provider="A", spread=-5.5, total=221.5, ml_home=-200, ml_away=170),
                 dict(provider="B", spread=-5.0, total=222.0, ml_home=-210, ml_away=175),
                 dict(provider="roto", spread=-5.5, total=52.5, ml_home=-5000, ml_away=-5000)]
        l = pick_line(items)
        self.assertAlmostEqual(l["total"], 221.75); self.assertTrue(0.6 < l["p_home"] < 0.7)
        self.assertEqual(pick_line([dict(provider="x", spread=None, total=None, ml_home=None, ml_away=None)]), {})


class TestKalshiNFL(unittest.TestCase):
    def test_two_letter_abbreviations_match(self):
        from kalshi_nba.kalshi import parse_markets, attach_games
        m = {"ticker": "KXNFLSPREAD-26NOV16KCGB-GB4", "event_ticker": "KXNFLSPREAD-26NOV16KCGB", "title": "Green Bay wins by over 3.5",
             "floor_strike": 3.5, "occurrence_datetime": "2026-11-16T18:00:00Z", "close_time": "2026-11-18T00:00:00Z",
             "yes_bid_dollars": "0.30", "yes_ask_dollars": "0.33", "last_price_dollars": "0.31", "volume_fp": "1"}
        games = pd.DataFrame([dict(event_id="5", kickoff_utc="2026-11-16T18:00Z", home_id="9", away_id="12")])   # KC en GB
        a = attach_games(parse_markets("spread", [m]), games, league="nfl")
        self.assertEqual((len(a), a.side.iloc[0]), (1, "home"))
        self.assertEqual(len(attach_games(parse_markets("spread", [m]), games, league="nba")), 0)   # tabla NBA no coincide


class TestLeagueCalendar(unittest.TestCase):
    def test_season_windows(self):
        from datetime import date
        from kalshi_nba.data import season_days, current_season
        d = list(season_days(2024, today=date(2030, 1, 1), league="nfl"))
        self.assertEqual((d[0], d[-1]), (date(2024, 9, 1), date(2025, 2, 20)))
        d = list(season_days(2024, today=date(2030, 1, 1), league="nba"))
        self.assertEqual((d[0], d[-1]), (date(2023, 10, 1), date(2024, 6, 30)))
        self.assertEqual(current_season("nfl", date(2026, 11, 11)), 2026)
        self.assertEqual(current_season("nba", date(2026, 11, 11)), 2027)
        self.assertEqual(current_season("nfl", date(2027, 1, 20)), 2026)


class TestResumen(unittest.TestCase):
    def test_builds_tables_from_csv(self):
        import tempfile
        from kalshi_nba import resumen
        with tempfile.TemporaryDirectory() as tmp:
            old = resumen.P; resumen.P = __import__("pathlib").Path(tmp)
            try:
                pd.DataFrame([dict(event_id="1", kickoff_utc="2030-01-02T01:00Z", home="Utah Jazz", away="Orlando Magic", p_home=0.3, mu_margin=-5.2, mu_total=228.4)]).to_csv(f"{tmp}/nba_registry.csv", index=False)
                pd.DataFrame([dict(snapshot_utc="2030-01-01", event_id="1", kind="winner", side="home", mid=0.25, p_book=0.28)]).to_csv(f"{tmp}/nba_kalshi_snapshots.csv", index=False)
                pd.DataFrame([dict(match_id="9", kickoff_utc="2030-01-02T15:00Z", competition="La Liga", home="Betis", away="Sevilla", p_H=.5, p_D=.25, p_A=.25, p_over_2_5_cal=.55, k_bid_H=.48, k_ask_H=.52, k_bid_D=None, k_ask_D=None, k_bid_A=.2, k_ask_A=.24)]).to_csv(f"{tmp}/registry.csv", index=False)
                txt = resumen.build(now=pd.Timestamp("2030-01-01", tz=resumen.TZ))
            finally:
                resumen.P = old
        self.assertIn("| Orlando Magic en Utah Jazz | Orlando Magic | 70% | 75% | — | 72% | 5.2 pts | 228 pts | — |", txt)
        self.assertIn("50% / 25% / 25% | 50% / — / 22% | — / 55% / — | — | — / — | — |", txt)
        self.assertIn("## NFL\n_Aún no hay registro._", txt)


class TestPolymarket(unittest.TestCase):
    def test_attach_orients_and_rejects_unclear(self):
        from kalshi_nba.polymarket import attach
        pm = [{"slug": "nba-dal-min-2026-11-12", "start": pd.Timestamp("2026-11-13T01:00Z"), "bid": 0.33, "ask": 0.35,
               "outcomes": [{"name": "Mavericks", "price": 0.34}, {"name": "Timberwolves", "price": 0.66}]}]
        g = pd.DataFrame([dict(event_id="1", kickoff_utc="2026-11-13T01:00Z", home="Minnesota Timberwolves", away="Dallas Mavericks"),
                          dict(event_id="2", kickoff_utc="2026-11-13T01:00Z", home="Phoenix Suns", away="Dallas Mavericks"),
                          dict(event_id="3", kickoff_utc="2026-11-15T01:00Z", home="Minnesota Timberwolves", away="Dallas Mavericks")])
        a = attach(g, pm)
        self.assertEqual(list(a.event_id), ["1"])
        self.assertAlmostEqual(a.pm_home.iloc[0], 0.66); self.assertAlmostEqual(a.pm_bid_home.iloc[0], 0.65); self.assertAlmostEqual(a.pm_ask_home.iloc[0], 0.67)


class TestInjuries(unittest.TestCase):
    def test_out_counts_use_latest_day_and_ignore_questionable(self):
        import tempfile
        from kalshi_nba.injuries import out_counts
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame([dict(team_id="1", status="Out", day="2026-11-11"), dict(team_id="1", status="Out", day="2026-11-12"),
                          dict(team_id="1", status="Doubtful", day="2026-11-12"), dict(team_id="1", status="Questionable", day="2026-11-12"),
                          dict(team_id="2", status="Day-To-Day", day="2026-11-12")]).to_csv(f"{tmp}/nba_injuries.csv", index=False)
            self.assertEqual(out_counts("nba", tmp), {"1": 2})
            self.assertEqual(out_counts("nfl", tmp), {})
