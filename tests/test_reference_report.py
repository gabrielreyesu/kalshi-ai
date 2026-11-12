import tempfile
import unittest
from pathlib import Path
from kalshi_external.reference_report import read_watchlist, read_status, report


class TestExternalReferenceSummary(unittest.TestCase):
    def test_watchlist_and_no_unsafe_url(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "watchlist.csv"
            p.write_text("Prioridad,Deporte,Analista,URL perfil,Mercado específico,Win rate publicado (%),N decidido\n"
                         "Alta,NFL,Expert A,https://example.org/a,SU,70,250\n"
                         "Alta,NFL,Expert B,javascript:alert(1),SU,99,10\n")
            rows = read_watchlist(p)
            self.assertEqual(len(rows), 1)
            out = report(rows, {"new_records": 7, "observations_in_private_cache": 15})
            self.assertIn("Expert A", out)
            self.assertNotIn("Expert B", out)
            self.assertIn("15", out)
            self.assertIn("no player props", out)

    def test_missing_status_is_not_success(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(read_status(Path(td) / "missing.json"))
        out = report([], None)
        self.assertIn("no se confirma ninguna captura", out)

    def test_invalid_json_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "status.json"
            p.write_text("invalid")
            self.assertIsNone(read_status(p))

    def test_limits(self):
        rows = [{"name":str(i), "sport":"NBA", "market":"moneyline",
                 "url":"https://example.org/" + str(i), "sample":250,
                 "rate":"55.0%", "priority":0} for i in range(8)]
        self.assertEqual(report(rows,None,max_per_sport=3).count("| ["),3)
        with self.assertRaises(ValueError):
            report(rows,None,max_per_sport=0)


if __name__=="__main__":
    unittest.main()
