import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd

from kalshi_nba.picks import soccer_picks, us_picks


def rows():
    r = [dict(kind="winner", side="home", strike=None, mid=0.70, p_model=0.66, p_book=0.70),
         dict(kind="winner", side="away", strike=None, mid=0.31, p_model=0.34, p_book=0.30)]
    for k, p in ((1.5, 0.66), (3.5, 0.60), (5.5, 0.52), (7.5, 0.45)):
        r.append(dict(kind="spread", side="home", strike=k, mid=p, p_model=p + .03, p_book=p))
    r.append(dict(kind="spread", side="away", strike=2.5, mid=0.20, p_model=0.22, p_book=0.20))
    for k, p in ((200.5, 0.90), (210.5, 0.76), (220.5, 0.51), (230.5, 0.25), (240.5, 0.08)):
        r.append(dict(kind="total", side=None, strike=k, mid=p, p_model=p, p_book=p))
    return pd.DataFrame(r)


class TestPicks(unittest.TestCase):
    def test_us_markets(self):
        ps = {p["mercado"]: p for p in us_picks(rows(), "Home", "Away", ("Home", 0.70, "casas"))}
        self.assertEqual(ps["Ganador"]["precio"], 0.70)
        self.assertEqual(ps["Hándicap"]["pick"], "Home gana por más de 5.5")
        self.assertNotIn("Hándicap seguro", ps)                     # ninguna línea llega al 70%
        self.assertEqual(ps["Total de puntos"]["pick"], "Más de 220.5")
        self.assertEqual(ps["Puntos (línea segura)"]["pick"], "Menos de 230.5")
        self.assertAlmostEqual(ps["Puntos (línea segura)"]["prob"], 0.75)

    def test_underdog_plus_line_and_model_fallback(self):
        r = rows(); r = r[r["kind"] != "total"].copy()
        r.loc[r["kind"] == "spread", "p_book"] = None
        r.loc[(r["kind"] == "spread") & (r["strike"] == 7.5), "p_model"] = 0.40
        r.loc[(r["kind"] == "spread") & (r["strike"] == 5.5), "p_model"] = 0.35
        ps = {p["mercado"]: p for p in us_picks(r, "Home", "Away", None)}
        self.assertTrue(ps["Hándicap"]["pick"].startswith("Away +"))
        self.assertEqual(ps["Hándicap"]["fuente"], "modelo")
        self.assertNotIn("Ganador", ps)

    def test_soccer(self):
        ps = {p["mercado"]: p for p in soccer_picks("A", "B", [0.5, 0.3, 0.2], "casas", {1.5: 0.7, 2.5: 0.45}, 0.4, kalshi3=[0.52, None, None])}
        self.assertEqual((ps["Ganador"]["pick"], ps["Ganador"]["precio"]), ("A", 0.52))
        self.assertEqual(ps["Doble oportunidad"]["pick"], "A o empate"); self.assertAlmostEqual(ps["Doble oportunidad"]["prob"], 0.8)
        self.assertEqual((ps["Goles"]["pick"], ps["Goles"]["prob"]), ("Menos de 2.5", 0.55))
        self.assertEqual(ps["Ambos anotan"]["pick"], "No")
        none = soccer_picks("A", "B", [0.5, 0.3, 0.2], "casas", {2.5: 0.6}, 0.5, goals_ok=False)
        self.assertEqual([p["mercado"] for p in none], ["Ganador", "Doble oportunidad"])


if __name__ == "__main__":
    unittest.main()


class TestOver25(unittest.TestCase):
    def test_books_drive_goals_pick(self):
        from kalshi_nba.picks import over25
        p, src = over25(0.40, 0.62)
        self.assertEqual(src, "casas + modelo"); self.assertGreater(p, 0.6)
        self.assertEqual(over25(0.55, None), (0.55, "modelo"))
        ps = soccer_picks("A", "B", [.5, .3, .2], "casas", {2.5: .4}, .5, goals_ok=False, book_over=.6)
        self.assertIn("Goles", [x["mercado"] for x in ps])                 # selecciones: con casas sí hay pick de goles
        ps = soccer_picks("A", "B", [.5, .3, .2], "casas", {2.5: .4}, .5, goals_ok=False)
        self.assertNotIn("Goles", [x["mercado"] for x in ps])
