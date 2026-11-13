import json, os, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from kalshi_external.health import check, main


class TestSourceHealth(unittest.TestCase):
    def _write(self, root, rel, obj):
        p = Path(root) / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(obj))

    def test_zero_and_missing_are_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, "soccer_impact/coverage.json", {"team_contexts_with_4_plus_prior_matches": 0,
                                                           "eligible_for_prospective_model_test": 3})
            self._write(d, "soccer_players/coverage.json", {"fixtures_matched_by_espn": 26, "warnings": ["x"]})
            rows = {(r["archivo"], r["clave"]): r["estado"] for r in check(Path(d), "full")}
            self.assertEqual(rows[("soccer_impact/coverage.json", "team_contexts_with_4_plus_prior_matches")], "AVISO")
            self.assertEqual(rows[("soccer_impact/coverage.json", "eligible_for_prospective_model_test")], "OK")
            self.assertEqual(rows[("soccer_players/coverage.json", "fixtures_matched_by_espn")], "OK")
            self.assertEqual(rows[("status.json", "new_records")], "FALTA_ARCHIVO")
            self.assertEqual(rows[("soccer_players/coverage.json", "warnings")], "AVISO")

    def test_hourly_only_checks_hourly_rules(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, "soccer_players/coverage.json", {"fixtures_matched_by_espn": 4})
            rows = check(Path(d), "hourly")
            self.assertEqual([r["clave"] for r in rows], ["fixtures_matched_by_espn"])

    def test_main_never_raises_and_writes_report(self):
        with tempfile.TemporaryDirectory() as d:
            main(["--mode", "full", "--reports", d])
            self.assertTrue((Path(d) / "source_health.json").exists())


if __name__ == "__main__":
    unittest.main()
