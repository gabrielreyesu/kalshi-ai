import tempfile
import unittest
from pathlib import Path
import pandas as pd
from kalshi_external.player_audit import audit
from kalshi_external.player_previews import build, run
from kalshi_external.player_intel import parse_injury_feed, parse_completed_boxscore, player_matchup_context


def injury(captured="2026-11-12T11:00:00Z"):
    return parse_injury_feed({"injuries":[{"id":"1","displayName":"Team A","injuries":[{
        "athlete":{"id":"100","displayName":"Starting Guard","position":{"abbreviation":"PG"}},
        "date":"2026-11-12T10:00:00Z","status":"Out"}]}]},"nba",captured)


def game(captured="2026-11-12T11:00:00Z"):
    summary={"header":{"competitions":[{
        "date":"2026-11-11T02:00:00Z","status":{"type":{"name":"STATUS_FINAL"}},
        "competitors":[{"homeAway":"home","team":{"id":"1"}},
                       {"homeAway":"away","team":{"id":"2"}}]}]},
        "boxscore":{"players":[{"team":{"id":"1"},"statistics":[{
            "labels":["MIN","PTS","REB","AST"],"athletes":[{
                "athlete":{"id":"100","displayName":"Starting Guard","position":{"abbreviation":"PG"}},
                "stats":["34:30","24","7","9"]}]}]}]}}
    return parse_completed_boxscore(summary,"nba","90001",captured)


def upcoming():
    return pd.DataFrame([{
        "event_id":"90002","home_id":"1","away_id":"2","home":"Team A","away":"Team B",
        "kickoff_utc":"2026-11-13T21:00:00Z",
    }])


class PlayerArchiveTests(unittest.TestCase):
    def test_archive_audit(self):
        report=audit([injury()],[game()])
        self.assertEqual(report["injury_batches"],1)
        self.assertEqual(report["completed_game_boxscores"],1)
        self.assertEqual(report["status"],"VALID_PLAYER_RESEARCH_NO_PRODUCTION_CHANGE")

    def test_archive_duplicate_rejected(self):
        r=injury()
        with self.assertRaisesRegex(ValueError,"Duplicate player archive"):
            audit([r,r],[])

    def test_archive_rejects_future_player_stats(self):
        g=game()
        g["captured_utc"]="2026-11-10T11:00:00Z"
        with self.assertRaisesRegex(ValueError,"before kickoff"):
            audit([],[g])

    def test_preview_has_important_reported_absence_and_matchup(self):
        data=build({"nba":upcoming()},[injury()],[game()],"2026-11-12T12:00:00Z")
        self.assertEqual(len(data),1)
        record=data[0]
        self.assertTrue(record["injury_data_present"])
        self.assertEqual(record["teams"][0]["known_reported_out"],1)
        guard=record["teams"][0]["player_context"][0]
        self.assertEqual(guard["availability"],"reported_out")
        self.assertEqual(guard["metrics"]["assists"]["recent_mean"],9.0)

    def test_stale_injury_report_not_claimed_as_current(self):
        old=injury("2026-11-09T11:00:00Z")
        ev={"sport":"nba","event_id":"90002","home_id":"1","away_id":"2",
            "kickoff_utc":"2026-11-13T21:00:00Z"}
        report=player_matchup_context(ev,[game()],[old],"2026-11-12T12:00:00Z")
        self.assertFalse(report["injury_data_present"])
        athlete=next(x for x in report["players"] if x["player_id"]=="100")
        self.assertEqual(athlete["availability"],"not_reported")

    def test_private_preview_archive_created_without_user_input(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)
            (d/"nba_registry.csv").write_text(upcoming().to_csv(index=False))
            import json
            (d/"player_injuries.jsonl").write_text(json.dumps(injury())+"\n")
            (d/"player_boxscores.jsonl").write_text(json.dumps(game())+"\n")
            status=run(registry_dir=d,injuries_path=d/"player_injuries.jsonl",
                       boxscores_path=d/"player_boxscores.jsonl",
                       previews_path=d/"private_player_previews.jsonl",
                       output=d/"status.json",observed_at="2026-11-12T12:00:00Z")
            self.assertEqual(status["new_pregame_contexts"],1)
            self.assertNotIn("Starting Guard",(d/"status.json").read_text())
            self.assertIn("Starting Guard",(d/"private_player_previews.jsonl").read_text())


if __name__=="__main__":
    unittest.main()
