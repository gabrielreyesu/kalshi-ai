import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd

from kalshi_intl.neutral import infer_neutral


class TestNeutral(unittest.TestCase):
    def test_rule(self):
        g = pd.DataFrame([
            dict(code="fifa.world", date=pd.Timestamp("2022-11-25"), home="Qatar", neutral=False),        # anfitrión
            dict(code="fifa.world", date=pd.Timestamp("2022-11-26"), home="France", neutral=False),       # neutral
            dict(code="fifa.world", date=pd.Timestamp("2026-06-12"), home="United States", neutral=False),
            dict(code="caf.nations", date=pd.Timestamp("2024-01-20"), home="Côte d'Ivoire", neutral=False),
            dict(code="uefa.nations", date=pd.Timestamp("2023-06-14"), home="Croatia", neutral=False),    # final four
            dict(code="uefa.nations", date=pd.Timestamp("2023-09-10"), home="Croatia", neutral=False),    # fase de grupos
            dict(code="fifa.worldq.uefa", date=pd.Timestamp("2025-03-10"), home="Spain", neutral=False),
            dict(code="fifa.friendly", date=pd.Timestamp("2025-03-10"), home="Spain", neutral=True),       # ESPN ya lo marca
        ])
        self.assertEqual(infer_neutral(g).tolist(), [False, True, False, False, True, False, False, True])


if __name__ == "__main__":
    unittest.main()
