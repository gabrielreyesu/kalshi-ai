import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd

from kalshi_lineup.stakes import compute, has_stake, position_range


class TestStakes(unittest.TestCase):
    def test_range_and_stake(self):
        pts = {"A": 90, "B": 60, "C": 59, "D": 20}
        left = {"A": 2, "B": 2, "C": 2, "D": 2}
        self.assertEqual(position_range("A", pts, left), (1, 1))          # campeón decidido
        self.assertEqual(position_range("B", pts, left), (2, 3))
        self.assertFalse(has_stake(1, 1, 20, 4, 3))
        self.assertTrue(has_stake(2, 3, 4, 2, 1))                          # frontera europea entre 2.º y 3.º
        self.assertTrue(has_stake(16, 19, 20, 4, 3))                       # puede descender

    def test_uses_only_past_results(self):
        rng = np.random.default_rng(0)
        teams = [f"T{i}" for i in range(12)]
        rows, eid = [], 0
        for w in range(22):
            order = rng.permutation(teams)
            for k in range(6):
                eid += 1
                rows.append(dict(event_id=str(eid), code="eng.1", date=pd.Timestamp("2024-08-10") + pd.Timedelta(days=7 * w),
                                 home_id=order[2 * k], away_id=order[2 * k + 1], hs=float(rng.integers(0, 4)), as_=float(rng.integers(0, 4)), final=True))
        g = pd.DataFrame(rows)
        a = compute(g).set_index("event_id")
        g2 = g.copy(); late = g2["date"] >= g2["date"].quantile(.7)
        g2.loc[late, "hs"] = 9.0
        b = compute(g2).set_index("event_id")
        keep = g.loc[g["date"] <= g["date"].quantile(.7), "event_id"]
        self.assertTrue((a.loc[keep] == b.loc[keep]).all().all())
        self.assertGreater(a["nostake_h"].iloc[-6:].sum() + a["nostake_a"].iloc[-6:].sum(), 0)   # al final hay equipos sin nada en juego


class TestGroups(unittest.TestCase):
    def test_nations_league_locked_position(self):
        from kalshi_lineup.stakes import compute_groups
        rows, eid = [], 0
        teams = ["A", "B", "C", "D"]
        sched = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("A", "D"), ("B", "C"), ("B", "A"), ("D", "C"),
                 ("C", "A"), ("D", "B"), ("D", "A"), ("C", "B")]
        rank = {"A": 0, "B": 1, "C": 2, "D": 3}                       # A gana todo, D pierde todo, B gana a C
        for i, (h, a) in enumerate(sched):
            eid += 1
            final = i < 10
            hw = rank[h] < rank[a]
            rows.append(dict(event_id=str(eid), code="uefa.nations", date=pd.Timestamp("2026-09-05") + pd.Timedelta(days=7 * (i // 2)),
                             home_id=h, away_id=a, hs=(2.0 if hw else 0.0) if final else np.nan,
                             as_=(0.0 if hw else 2.0) if final else np.nan, final=final))
        st = compute_groups(pd.DataFrame(rows)).set_index("event_id")
        self.assertEqual(len(st), 12)
        self.assertEqual(st.loc["1", "nostake_h"], 0)                     # jornada 1: todo abierto
        last = st.loc[["11", "12"]]                                       # jornada 6: A 15 pts y D 0 pts, posiciones cerradas
        self.assertEqual(int(last.loc["11", "nostake_h"]), 1)             # D (local) ya es último
        self.assertEqual(int(last.loc["11", "nostake_a"]), 1)             # A ya es primero


if __name__ == "__main__":
    unittest.main()
