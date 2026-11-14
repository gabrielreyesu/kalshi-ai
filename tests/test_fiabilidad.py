import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd

from kalshi_nba.fiabilidad import table


class TestFiabilidad(unittest.TestCase):
    def test_bins_use_pick_side(self):
        p = pd.Series([0.2] * 20 + [0.75] * 20)          # 20 picks del visitante al 80%, 20 del local al 75%
        y = pd.Series([0] * 16 + [1] * 4 + [1] * 15 + [0] * 5)
        t = {r["tramo"]: r for r in table(p, y)}
        self.assertAlmostEqual(t["70–80%"]["acierto"], 0.75)
        self.assertAlmostEqual(t["80–90%"]["acierto"], 0.80)
        self.assertAlmostEqual(t["80–90%"]["prob_media"], 0.80)


if __name__ == "__main__":
    unittest.main()
