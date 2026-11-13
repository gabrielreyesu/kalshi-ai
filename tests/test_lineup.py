import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd

from kalshi_lineup.box import available, fetch_box, parse_summary
from kalshi_lineup.evaluate import VARIANTS, decide, run_variants
from kalshi_lineup.ratings import LINEUP_FEATURES, LineupRater, allocate, game_score
from kalshi_nba.engine import FEATURES, NBAEngine


def make_league(seasons=range(2016, 2022), teams=12, per_team=10, seed=1, p_out=0.2, signal=True):
    """Liga sintética donde el resultado depende de qué jugadores están disponibles."""
    rng = np.random.default_rng(seed)
    val = {(t, i): rng.normal(0, 0.08) + (0.25 if i == 0 else 0.0) for t in range(teams) for i in range(per_team)}
    mins = np.array([36, 34, 32, 30, 28, 24, 20, 16, 12, 8], float)[:per_team]
    games, boxes, eid = [], {}, 0
    for y in seasons:
        d0 = pd.Timestamp(f"{y - 1}-10-22")
        for k in range(110):
            date = d0 + pd.Timedelta(days=int(k * 1.6))
            order = rng.permutation(teams)
            for g in range(teams // 2):
                h, a = int(order[2 * g]), int(order[2 * g + 1])
                eid += 1
                players, strength = [], {}
                for t in (h, a):
                    out = rng.random(per_team) < p_out
                    s = 0.0
                    for i in range(per_team):
                        pid = f"{t}-{i}"
                        if out[i]:
                            players.append(dict(pid=pid, team_id=f"T{t}", starter=False, dnp=True, reason="INJURY", avail=False,
                                                min=np.nan, pts=np.nan, pm=np.nan))
                            continue
                        m = mins[i]
                        s += val[(t, i)] * m
                        pts = max(0.0, m * (0.45 + val[(t, i)]) + rng.normal(0, 3))
                        players.append(dict(pid=pid, team_id=f"T{t}", starter=i < 5, dnp=False, reason="", avail=True,
                                            min=float(m), pts=float(pts), pm=float(rng.normal(0, 8))))
                    strength[t] = s
                margin = 2.5 + (strength[h] - strength[a]) * (1.5 if signal else 0.0) + rng.normal(0, 11)
                hs = 110 + round(margin / 2); as_ = 110 - round(margin / 2)
                if hs == as_:
                    hs += 1
                games.append(dict(event_id=str(eid), date=date, kickoff_utc=date.isoformat(), season_year=y, season_type=2,
                                  status="STATUS_FINAL", home_id=f"T{h}", away_id=f"T{a}", home=f"T{h}", away=f"T{a}",
                                  hs=float(hs), as_=float(as_), neutral=False, line_spread=np.nan, line_total=np.nan,
                                  line_p_home=np.nan))
                boxes[str(eid)] = {"event_id": str(eid), "players": players}
    return pd.DataFrame(games), boxes


ESPN = {
    "header": {"competitions": [{"status": {"type": {"name": "STATUS_FINAL"}}}]},
    "boxscore": {"players": [
        {"team": {"id": "1"}, "statistics": [{
            "keys": ["minutes", "points", "fieldGoalsMade-fieldGoalsAttempted", "freeThrowsMade-freeThrowsAttempted",
                     "rebounds", "assists", "turnovers", "steals", "blocks", "offensiveRebounds", "defensiveRebounds",
                     "fouls", "plusMinus"],
            "athletes": [
                {"athlete": {"id": "11", "displayName": "A"}, "starter": True, "didNotPlay": False,
                 "stats": ["35", "30", "11-20", "6-8", "7", "5", "3", "1", "1", "2", "5", "2", "+9"]},
                {"athlete": {"id": "12", "displayName": "B"}, "starter": False, "didNotPlay": True,
                 "reason": "DNP-COACH'S DECISION", "stats": []},
                {"athlete": {"id": "13", "displayName": "C"}, "starter": False, "didNotPlay": True,
                 "reason": "INACTIVE - INJURY/ILLNESS", "stats": []}]}]},
        {"team": {"id": "2"}, "statistics": [{
            "labels": ["MIN", "FG", "3PT", "FT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TO", "PF", "+/-", "PTS"],
            "athletes": [{"athlete": {"id": "21"}, "starter": True,
                          "stats": ["40", "8-15", "2-5", "3-4", "1", "6", "7", "4", "2", "0", "2", "3", "-9", "21"]}]}]},
    ]},
}


class TestBoxParsing(unittest.TestCase):
    def test_parse_keys_and_labels(self):
        b = parse_summary(ESPN, "99")
        p = {r["pid"]: r for r in b["players"]}
        self.assertEqual(p["11"]["pts"], 30); self.assertEqual((p["11"]["fgm"], p["11"]["fga"]), (11, 20))
        self.assertEqual(p["11"]["pm"], 9); self.assertTrue(p["11"]["avail"])
        self.assertTrue(p["12"]["avail"])          # DNP decisión del técnico = activo
        self.assertFalse(p["13"]["avail"])         # inactivo por lesión
        self.assertEqual(p["21"]["pts"], 21); self.assertEqual(p["21"]["pm"], -9); self.assertEqual(p["21"]["ftm"], 3)

    def test_not_final_is_ignored(self):
        x = {**ESPN, "header": {"competitions": [{"status": {"type": {"name": "STATUS_IN_PROGRESS"}}}]}}
        self.assertIsNone(parse_summary(x, "1"))

    def test_network_failure_not_cached(self):
        import tempfile
        from pathlib import Path
        def boom(url, params):
            raise RuntimeError("down")
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(fetch_box("5", Path(d), getter=boom))
            self.assertFalse((Path(d) / "5.json").exists())
            self.assertIsNotNone(fetch_box("6", Path(d), getter=lambda u, p: ESPN))
            self.assertTrue((Path(d) / "6.json").exists())

    def test_available_rule(self):
        self.assertFalse(available(False, np.nan, True, "DNP - REST"))
        self.assertTrue(available(False, 3.0, False, ""))

    def test_allocate_caps(self):
        m = allocate([48, 48, 48, 10, 10, 10, 10], 240, 42)
        self.assertAlmostEqual(m.sum(), 240); self.assertLessEqual(m.max(), 42 + 1e-9)

    def test_game_score(self):
        self.assertAlmostEqual(game_score({"pts": 10, "fgm": 4, "fga": 8}), 10 + 1.6 - 5.6)


class TestLineupRater(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g, cls.b = make_league()
        cls.f = LineupRater().process(cls.g, cls.b)

    def test_no_future_leak_from_same_day_stats(self):
        """Cambiar estadísticas (no la disponibilidad) desde una fecha NO cambia variables de esa fecha ni anteriores."""
        cut = self.g["date"].quantile(0.5)
        b2 = {}
        rng = np.random.default_rng(5)
        late = set(self.g.loc[self.g["date"] >= cut, "event_id"])
        for e, box in self.b.items():
            if e in late:
                box = {**box, "players": [{**p, "pts": float(rng.uniform(0, 60)) if p["avail"] else p["pts"],
                                           "min": float(rng.uniform(1, 48)) if p["avail"] else p["min"]} for p in box["players"]]}
            b2[e] = box
        f2 = LineupRater().process(self.g, b2)
        keep = set(self.g.loc[self.g["date"] <= cut, "event_id"])
        a = self.f[self.f["event_id"].isin(keep)].set_index("event_id").sort_index()
        b = f2[f2["event_id"].isin(keep)].set_index("event_id").sort_index()
        for c in LINEUP_FEATURES:
            self.assertTrue(np.allclose(a[c], b[c], atol=1e-12), c)

    def test_missing_box_is_neutral(self):
        e = self.g["event_id"].iloc[-1]
        f = LineupRater().process(self.g, {k: v for k, v in self.b.items() if k != e})
        r = f[f["event_id"] == e].iloc[0]
        self.assertEqual(r["has_box"], 0)
        self.assertTrue(all(r[c] == 0 for c in LINEUP_FEATURES))

    def test_star_absence_lowers_strength(self):
        corr = np.corrcoef(self.f["lineup_gs"], self.g.set_index("event_id").loc[self.f["event_id"], "hs"].to_numpy()
                           - self.g.set_index("event_id").loc[self.f["event_id"], "as_"].to_numpy())[0, 1]
        self.assertGreater(corr, 0.1)

    def test_variant_beats_base_when_signal_exists(self):
        feats = NBAEngine().process(self.g)
        feats["event_id"] = feats["event_id"].astype(str)
        feats = feats.merge(self.f, on="event_id")
        oos = run_variants(feats, FEATURES, first_test=2018, min_train=500)
        for n in VARIANTS:
            self.assertIn(f"p_{n}", oos)
        res = decide(oos)
        best = res["desarrollo"][res["variante_elegida"]]
        self.assertGreater(best["mejora_logloss"]["mejora"], 0)
        self.assertGreater(best["mejora_logloss"]["ic95_inf"], 0)

    def test_null_no_spurious_gain(self):
        g, b = make_league(signal=False, seed=3)
        feats = NBAEngine().process(g)
        feats["event_id"] = feats["event_id"].astype(str)
        feats = feats.merge(LineupRater().process(g, b), on="event_id")
        res = decide(run_variants(feats, FEATURES, first_test=2018, min_train=500))
        a = res["desarrollo"]["A_disponibles_gs"]["mejora_logloss"]
        self.assertLess(a["ic95_inf"], 0.002)
        self.assertLess(a["mejora"], 0.003)


if __name__ == "__main__":
    unittest.main()
