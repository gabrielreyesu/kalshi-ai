import json, os, re, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd

from kalshi_nba import pagina


class TestPagina(unittest.TestCase):
    def _data(self, html):
        return json.loads(re.search(r"const DATA = (.*?);\nconst S", html, re.S).group(1))

    def test_empty_folder_renders(self):
        with tempfile.TemporaryDirectory() as d:
            old = pagina.P
            try:
                pagina.P = Path(d)
                import kalshi_nba.resumen as r; r_old = r.P; r.P = Path(d)
                html = pagina.render(pagina.collect())
            finally:
                pagina.P = old; r.P = r_old
        self.assertEqual(self._data(html)["games"], [])
        self.assertIn('name="viewport"', html)

    def test_us_game_and_escaping(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            pd.DataFrame([dict(event_id="1", kickoff_utc="2030-01-01T00:00Z", home="</script><b>X", away="Y", home_id="10",
                               away_id="20", elo_diff=0, rest_h=1, rest_a=1, p_home=0.62, mu_margin=3.0, mu_total=220,
                               p_home_normal=0.6, sig_m=13, sig_t=17, model_version="nba_v1", created_utc="2029-12-31T00:00Z")]
                         ).to_csv(p / "nba_registry.csv", index=False)
            old = pagina.P
            try:
                pagina.P = p
                import kalshi_nba.resumen as r; r_old = r.P; r.P = p
                html = pagina.render(pagina.collect(pd.Timestamp("2029-12-31", tz="America/Chicago")))
            finally:
                pagina.P = old; r.P = r_old
        self.assertNotIn("</script><b>", html)
        g = self._data(html)["games"][0]
        self.assertEqual((g["pick"], g["prob"], g["src"]), ("</script><b>X", 0.62, "modelo"))
        self.assertEqual(g["rows"][1], ["Casas", [None, None]])

    def test_real_predictions_render(self):
        if not Path("predictions/nba_registry.csv").exists():
            self.skipTest("sin datos")
        data = pagina.collect(pd.Timestamp("2026-11-01", tz="America/Chicago"))
        json.dumps(data, allow_nan=False)
        self.assertGreater(len(data["games"]), 0)


if __name__ == "__main__":
    unittest.main()
