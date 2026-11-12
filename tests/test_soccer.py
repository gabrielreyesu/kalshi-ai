import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
import unittest
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy.stats import poisson

from synth import make_history
from kalshi_soccer.engine import SoccerEngine, validate_games
from kalshi_soccer.backtest import walk_forward, score_1x2, y_index, _P, bin_logloss_each
from kalshi_soccer.models import MARKETS


def notebook_reference(d):
    """Copia literal de la lógica de la celda 32 del notebook (referencia para el test de paridad)."""
    d = d.copy().sort_values(["date", "competition", "home", "away"]).reset_index(drop=True)
    elo = defaultdict(lambda: 1500.0); season_seen = {}
    history = defaultdict(lambda: {k: deque(maxlen=20) for k in ("home_gf", "home_ga", "away_gf", "away_ga")})
    league_history = defaultdict(lambda: deque(maxlen=300))
    smooth = lambda v, b: (sum(v) + 8 * b) / (len(v) + 8)

    def dc_prob(lh, la):
        goals = np.arange(13)
        mat = np.outer(poisson.pmf(goals, lh), poisson.pmf(goals, la)); mat /= mat.sum(); rho = -0.08
        mat[0, 0] *= 1 - lh * la * rho; mat[0, 1] *= 1 + lh * rho; mat[1, 0] *= 1 + la * rho; mat[1, 1] *= 1 - rho
        mat = np.maximum(mat, 0); mat /= mat.sum()
        return np.tril(mat, -1).sum(), np.trace(mat), np.triu(mat, 1).sum()

    rows = []
    for (liga, fecha), jornada in d.groupby(["competition", "date"], sort=True):
        updates = []
        for _, r in jornada.iterrows():
            h, a = (liga, r.home), (liga, r.away)
            for team in (h, a):
                if season_seen.get(team) != r.season:
                    elo[team] = 1500 + 0.65 * (elo[team] - 1500); season_seen[team] = r.season
            eh, ea = elo[h], elo[a]
            hh, aa, lg = history[h], history[a], league_history[liga]
            if len(lg):
                avg_h = np.mean([x[0] for x in lg]); avg_a = np.mean([x[1] for x in lg])
            else:
                avg_h, avg_a = 1.45, 1.15
            avg_h = max(avg_h, 0.5); avg_a = max(avg_a, 0.5)
            attack_h = smooth(hh["home_gf"], avg_h) / avg_h; defense_a = smooth(aa["away_ga"], avg_h) / avg_h
            attack_a = smooth(aa["away_gf"], avg_a) / avg_a; defense_h = smooth(hh["home_ga"], avg_a) / avg_a
            lh = float(np.clip(avg_h * attack_h * defense_a, 0.2, 4.5)); la = float(np.clip(avg_a * attack_a * defense_h, 0.2, 4.5))
            pH, pD, pA = dc_prob(lh, la)
            rows.append(dict(date=fecha, competition=liga, home=r.home, away=r.away, elo_diff=eh - ea,
                             lambda_h=lh, lambda_a=la, p_H=pH, p_D=pD, p_A=pA))
            result = 1.0 if r.hs > r.as_ else 0.5 if r.hs == r.as_ else 0.0
            expected = 1 / (1 + 10 ** ((ea - eh - 65) / 400))
            updates.append((h, a, float(r.hs), float(r.as_), result, expected))
        for h, a, hg, ag, result, expected in updates:
            delta = 20 * (result - expected); elo[h] += delta; elo[a] -= delta
            history[h]["home_gf"].append(hg); history[h]["home_ga"].append(ag)
            history[a]["away_gf"].append(ag); history[a]["away_ga"].append(hg)
            league_history[liga].append((hg, ag))
    return pd.DataFrame(rows)


class TestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hist = make_history(signal=True, seasons=range(2015, 2020), with_odds=True)
        cls.feats = SoccerEngine().process(cls.hist)

    def test_parity_with_notebook(self):
        ref = notebook_reference(self.hist)
        keys = ["date", "competition", "home", "away"]
        m = self.feats.merge(ref, on=keys, suffixes=("", "_ref"), validate="one_to_one")
        self.assertEqual(len(m), len(self.feats))
        for c in ["elo_diff", "lambda_h", "lambda_a", "p_H", "p_D", "p_A"]:
            self.assertLess(np.abs(m[c] - m[c + "_ref"]).max(), 1e-9, c)

    def test_no_future_leak(self):
        """Cambiar resultados FUTUROS no puede alterar variables de partidos anteriores."""
        cut = self.hist["date"].quantile(0.6)
        h2 = self.hist.copy()
        late = h2["date"] > cut
        rng = np.random.default_rng(9)
        h2.loc[late, "hs"] = rng.integers(0, 6, late.sum()); h2.loc[late, "as_"] = rng.integers(0, 6, late.sum())
        f2 = SoccerEngine().process(h2)
        a = self.feats[self.feats["date"] <= cut].reset_index(drop=True)
        b = f2[f2["date"] <= cut].reset_index(drop=True)
        for c in ["elo_diff", "lambda_h", "lambda_a", "p_H", "p_over_2_5", "p_btts"]:
            self.assertTrue(np.allclose(a[c], b[c], atol=1e-12), c)

    def test_incremental_equals_batch(self):
        cut = self.hist["date"].quantile(0.7)
        e = SoccerEngine(); e.process(self.hist[self.hist["date"] <= cut])
        part2 = e.process(self.hist[self.hist["date"] > cut])
        full = self.feats[self.feats["date"] > cut].reset_index(drop=True)
        self.assertTrue(np.allclose(part2["p_H"], full["p_H"], atol=1e-12))
        self.assertTrue(np.allclose(part2["lambda_h"], full["lambda_h"], atol=1e-12))

    def test_probabilities_consistent(self):
        f = self.feats
        self.assertLess(np.abs(f.p_H + f.p_D + f.p_A - 1).max(), 1e-9)
        self.assertTrue((f.p_over_1_5 >= f.p_over_2_5 - 1e-12).all() and (f.p_over_2_5 >= f.p_over_3_5 - 1e-12).all())
        self.assertTrue(((f.p_btts >= 0) & (f.p_btts <= 1)).all())

    def test_future_fixture_does_not_update_state(self):
        e = SoccerEngine(); e.process(self.hist)
        before = dict(e.elo); n0 = dict(e.n_played)
        e.predict("Premier League", "2021", "Nuevo_A", "Nuevo_B")   # equipos desconocidos
        self.assertEqual(dict(e.n_played), n0)
        self.assertFalse(e.knows("Premier League", "Nuevo_A"))

    def test_validation_rejects_bad_data(self):
        bad = self.hist.head(5).copy(); bad.loc[bad.index[0], "hs"] = -1
        with self.assertRaises(ValueError): validate_games(bad)


class TestBacktest(unittest.TestCase):
    def _run(self, signal):
        hist = make_history(signal=signal, seasons=range(2015, 2022), with_odds=True)
        feats = SoccerEngine().process(hist)
        return walk_forward(feats, first_test_season="1819", min_train=500)

    def test_null_signal_gives_no_edge(self):
        """Sin diferencias reales entre equipos, el modelo NO debe batir la línea base."""
        p = self._run(signal=False)
        yi = y_index(p["target"])
        ll_m = score_1x2(_P(p, "m"), yi)["logloss"]; ll_b = score_1x2(_P(p, "base"), yi)["logloss"]
        self.assertGreater(ll_m, ll_b - 0.004, (ll_m, ll_b))   # sin ventaja apreciable

    def test_real_signal_is_detected_and_market_is_benchmark(self):
        p = self._run(signal=True)
        yi = y_index(p["target"]); m = p["mkt_H"].notna().to_numpy()
        ll_m = score_1x2(_P(p, "m"), yi)["logloss"]; ll_b = score_1x2(_P(p, "base"), yi)["logloss"]
        ll_k = score_1x2(_P(p[m], "mkt"), yi[m])["logloss"]
        self.assertLess(ll_m, ll_b)               # el modelo detecta la señal
        self.assertLess(ll_k, ll_m + 1e-9)        # el 'mercado' (verdad + margen) es la cota superior
        # totales: calibrado y crudo deben ser mejores que la base
        y = p["y_over_2_5"].to_numpy(float)
        self.assertLess(bin_logloss_each(p["cal_over_2_5"].to_numpy(), y).mean(),
                        bin_logloss_each(p["base_over_2_5"].to_numpy(), y).mean())


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestCliFlow(unittest.TestCase):
    """Flujo completo predict -> registro append-only -> evaluate, con datos y API simulados."""

    def test_predict_register_evaluate(self):
        import tempfile
        from argparse import Namespace
        from unittest import mock
        from kalshi_soccer import cli

        hist = make_history(signal=True, seasons=range(2015, 2021), with_odds=True)
        # make_history deja 'season' como código; la temporada vigente para fixtures se deriva de la fecha
        teams = sorted(set(hist[hist.competition == "Premier League"].home))
        kick = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=3, hours=15)
        fx = pd.DataFrame([
            dict(match_id="1", competition="Premier League", home_api=teams[0], away_api=teams[1], kickoff_utc=kick),
            dict(match_id="2", competition="Premier League", home_api="Equipo Desconocido FC", away_api=teams[2], kickoff_utc=kick),
        ])
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                os.environ["FOOTBALL_DATA_API_KEY"] = "x"
                with mock.patch.object(cli, "load_history", return_value=hist), \
                     mock.patch.object(cli, "fetch_fixtures", return_value=fx):
                    cli.cmd_predict(Namespace(start=2015, days=7, allow_new=False))
                    reg = pd.read_csv("predictions/registry.csv", dtype={"match_id": str})
                    self.assertEqual(list(reg.match_id), ["1"])                      # solo el resuelto
                    self.assertTrue(os.path.exists("predictions/unresolved_teams.csv"))  # el otro se reporta
                    self.assertAlmostEqual(reg[["p_A", "p_D", "p_H"]].sum(axis=1).iloc[0], 1.0, places=6)
                    first = open("predictions/registry.csv").read()
                    cli.cmd_predict(Namespace(start=2015, days=7, allow_new=False))   # segunda corrida
                    self.assertEqual(open("predictions/registry.csv").read(), first)  # idempotente, no reescribe
                # evaluación: el partido "se juega" y aparece en el histórico
                played = pd.DataFrame([dict(date=kick.tz_localize(None).normalize(), competition="Premier League",
                                            home=teams[0], away=teams[1], hs=2, as_=1)])
                from kalshi_soccer.registry import evaluate
                ev = evaluate(played)
                self.assertEqual(len(ev), 1)
                self.assertEqual(ev.target.iloc[0], "H")
            finally:
                os.chdir(cwd)


class TestKalshiMatching(unittest.TestCase):
    def _ev(self, a, b, kick, ask=(0.45, 0.27, 0.30)):
        mk = lambda kind, team, ask_, i: {"kind": kind, "team": team, "bid": ask_ - 0.01, "ask": ask_, "last": ask_, "vol": 10.0, "ticker": f"T{i}"}
        return {"event": "KXEPLGAME-26NOV23X", "kickoff_utc": pd.Timestamp(kick),
                "outcomes": [mk("team", a, ask[0], 1), mk("D", None, ask[1], 2), mk("team", b, ask[2], 3)]}

    def test_matches_and_orients(self):
        from kalshi_soccer.kalshi import attach_kalshi, kalshi_mid
        fx = pd.DataFrame([dict(match_id="1", home_api="Everton FC", away_api="Brentford FC")])
        preds = pd.DataFrame([dict(match_id="1", competition="Premier League", home="Everton", away="Brentford",
                                   kickoff_utc="2026-11-23T20:00:00+00:00", p_H=.3, p_D=.27, p_A=.43)])
        ev = {"Premier League": [self._ev("Brentford", "Everton", "2026-11-23T20:00:00Z")]}
        out = attach_kalshi(preds, fx, ev)
        self.assertAlmostEqual(out.k_ask_A.iloc[0], 0.45)   # Brentford = visitante en este partido
        self.assertAlmostEqual(out.k_ask_H.iloc[0], 0.30)
        self.assertAlmostEqual(kalshi_mid(out).sum(axis=1).iloc[0], 1.0)

    def test_ambiguous_or_unknown_is_left_empty(self):
        from kalshi_soccer.kalshi import attach_kalshi
        fx = pd.DataFrame([dict(match_id="1", home_api="Everton FC", away_api="Brentford FC")])
        preds = pd.DataFrame([dict(match_id="1", competition="Premier League", home="Everton", away="Brentford",
                                   kickoff_utc="2026-11-23T20:00:00+00:00", p_H=.3, p_D=.27, p_A=.43)])
        for ev in ({"Premier League": [self._ev("Fulham", "Everton", "2026-11-23T20:00:00Z")]},      # un solo equipo coincide
                   {"Premier League": [self._ev("Brentford", "Everton", "2026-11-26T20:00:00Z")]}):  # fecha lejana
            self.assertTrue(pd.isna(attach_kalshi(preds, fx, ev).k_event.iloc[0]))


class TestKalshiAliases(unittest.TestCase):
    def test_alias_resolves_ambiguous_names(self):
        from kalshi_soccer.kalshi import team_matches
        # "Barcelona" está contenido en "RCD Espanyol de Barcelona": el alias evita el cruce
        self.assertTrue(team_matches("La Liga", "Barcelona", "FC Barcelona", "Barcelona"))
        self.assertFalse(team_matches("La Liga", "Barcelona", "RCD Espanyol de Barcelona", "Espanol"))
        self.assertTrue(team_matches("Serie A", "Milan", "AC Milan", "Milan"))
        self.assertFalse(team_matches("Serie A", "Milan", "FC Internazionale Milano", "Inter"))
