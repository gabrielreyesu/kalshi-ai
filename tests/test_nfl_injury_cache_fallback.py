import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from kalshi_external.player_collect import cached_daily_injuries, run
from kalshi_external.player_intel import player_matchup_context


def cached_file(folder, date="2026-11-12"):
    pd.DataFrame([{"team_id":"19","team":"New York Giants",
        "player":"Star QB","position":"QB","status":"Out","type":"Knee",
        "reported":"2026-11-12T12:00:00Z","day":date}]).to_csv(
        Path(folder)/"nfl_injuries.csv",index=False)


class NFLDailyFallback(unittest.TestCase):
    def test_only_same_day_cache_and_no_canonical_id_assumption(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached_file(tmp)
            x=cached_daily_injuries("nfl","2026-11-12T22:30:00Z",tmp)
            self.assertIsNotNone(x)
            self.assertEqual(x["feed_origin"],"same_day_cached_espn_csv")
            p=x["reported_athletes"][0]
            self.assertFalse(p["id_is_canonical"])
            self.assertTrue(p["player_id"].startswith("name:"))
            self.assertIsNone(cached_daily_injuries("nfl","2026-11-13T22:30:00Z",tmp))

    def test_join_exact_player_name_to_prior_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached_file(tmp)
            injury=cached_daily_injuries("nfl","2026-11-12T22:30:00Z",tmp)
            history=[{"source_id":"espn_completed_player_boxscore","sport":"nfl",
                 "event_id":"old","captured_utc":"2026-11-11T22:00:00Z",
                 "kickoff_utc":"2026-11-10T19:00:00Z",
                 "players":[{"team_id":"19","opponent_id":"28",
                             "player_id":"101","player":"Star QB",
                             "position":"QB","metrics":{"passing_yards":250}}]}]
            upcoming={"sport":"nfl","event_id":"future",
                      "home_id":"19","away_id":"28",
                      "kickoff_utc":"2026-11-13T23:00:00Z"}
            ctx=player_matchup_context(upcoming,history,[injury],"2026-11-12T22:31:00Z")
            star=next(p for p in ctx["players"] if p["player_id"]=="101")
            self.assertEqual(star["availability"],"reported_out")
            self.assertEqual(star["metrics"]["passing_yards"]["recent_mean"],250.0)
            self.assertNotIn("name:star qb",[p["player_id"] for p in ctx["players"]])

    def test_no_guess_when_name_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached_file(tmp)
            injury=cached_daily_injuries("nfl","2026-11-12T22:30:00Z",tmp)
            history=[{"source_id":"espn_completed_player_boxscore","sport":"nfl",
                "event_id":"prior","captured_utc":"2026-11-11T22:00:00Z",
                "kickoff_utc":"2026-11-10T19:00:00Z",
                "players":[{"team_id":"19","opponent_id":"28","player_id":pid,
                    "player":"Star QB","metrics":{"passing_yards":n}}
                    for pid,n in [("101",250),("102",80)]]}]
            upcoming={"sport":"nfl","event_id":"future","home_id":"19",
                      "away_id":"28","kickoff_utc":"2026-11-13T23:00:00Z"}
            ctx=player_matchup_context(upcoming,history,[injury],"2026-11-12T22:31:00Z")
            data={p["player_id"]:p for p in ctx["players"]}
            self.assertEqual(data["name:star qb"]["availability"],"reported_out")
            self.assertEqual(data["101"]["availability"],"not_reported")
            self.assertEqual(data["102"]["availability"],"not_reported")

    def test_nfl_fallback_during_live_provider_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached_file(tmp)
            base=Path(tmp)
            with patch("kalshi_external.player_collect._fetch_injuries") as fetch, patch(
                "kalshi_external.player_collect._recent_completed",return_value=([],[],0)):
                fetch.side_effect=[{"source_id":"espn_official_injury_feed",
                                     "source_kind":"player_availability_research",
                                     "sport":"nba","event_id":"nba_league_injuries",
                                     "captured_utc":"2026-11-12T22:00:00Z",
                                     "reported_athletes":[]},
                                   ValueError("Unrecognized injury report schema")]
                report=run(injuries_path=base/"player_injuries.jsonl",
                           boxscores_path=base/"player_boxscores.jsonl",
                           report_dir=base/"reports",registry_dir=base,
                           observed_at="2026-11-12T22:00:00Z")
            self.assertEqual(report["new_injury_snapshot_batches"],2)
            self.assertTrue(any("cached injury CSV" in x for x in report["warnings"]))
            self.assertEqual(report["sports"][1]["players_in_latest_injury_report"],1)


if __name__=="__main__":
    unittest.main()
