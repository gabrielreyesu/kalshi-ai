import unittest
import pandas as pd

from kalshi_external.soccer_player_intel import (
    parse_espn_summary, parse_api_football_injuries, player_opponent_context,
)
from kalshi_external.soccer_player_collect import (
    parse_scoreboard,parse_api_fixtures,join_vendor_injuries,
)
from kalshi_external.soccer_player_audit import audit


def summary(event_date="2026-11-11T19:00:00Z", status="STATUS_FULL_TIME",
            goal=1, opponent="20"):
    return {
        "header":{"competitions":[{
            "date":event_date,"status":{"type":{"name":status}},
            "competitors":[
                {"homeAway":"home","team":{"id":"10"}},
                {"homeAway":"away","team":{"id":opponent}}
            ]}]},
        "boxscore":{"players":[{
            "team":{"id":"10"},"statistics":[{
                "labels":["MIN","G","A","SH","SOG"],
                "athletes":[{"athlete":{"id":"101","displayName":"Example Forward",
                                        "position":{"abbreviation":"FW"}},
                             "stats":["90",str(goal),"0","4","2"],"starter":True}]
            }]
        }]}
    }


def upcoming():
    return {"sport":"soccer","event_id":"3001","competition_code":"uefa.champions",
            "kickoff_utc":"2026-11-14T19:00:00Z",
            "home_id":"10","away_id":"20","home":"Arsenal","away":"Barcelona"}


class SoccerPlayerResearchTests(unittest.TestCase):
    def test_finished_soccer_stats_are_historical_only(self):
        r=parse_espn_summary(summary(),"uefa.champions","1001","2026-11-12T12:00:00Z")
        self.assertEqual(r["source_id"],"espn_soccer_final_player_stats")
        p=r["teams"][0]["members"][0]
        self.assertEqual(p["player_id"],"101")
        self.assertEqual(p["metrics"]["goals"],1)
        self.assertEqual(p["metrics"]["shots_on_target"],2)
        self.assertEqual(p["metrics"]["minutes"],90)
        self.assertEqual(audit([r],[],[],[])["record_counts"]["games"],1)

    def test_roster_before_kickoff_is_not_confirmed_starting_xi(self):
        p=summary("2026-11-14T19:00:00Z","STATUS_SCHEDULED")
        r=parse_espn_summary(p,"uefa.champions","3001","2026-11-14T18:00:00Z")
        self.assertEqual(r["source_id"],"espn_soccer_pregame_roster")
        self.assertFalse(r["lineup_confirmed"])
        self.assertTrue(r["teams"][0]["members"][0]["starter_reported"])
        self.assertEqual(audit([],[r],[],[])["record_counts"]["lineups"],1)

    def test_no_pre_match_reconstruction_with_future_kickoff(self):
        r=parse_espn_summary(summary(),"uefa.champions","1001","2026-11-11T18:00:00Z")
        self.assertIsNone(r)
        self.assertIsNone(parse_espn_summary(summary(),"uefa.champions","1001",
                 "2026-11-12T12:00:00Z",expected_kickoff="2026-11-08T19:00:00Z"))

    def test_no_claim_player_stats_if_only_roster(self):
        p=summary()
        p["boxscore"]["players"][0]["statistics"][0]["athletes"][0]["stats"]=[]
        self.assertIsNone(parse_espn_summary(p,"uefa.champions","1001","2026-11-12T12:00:00Z"))

    def test_availability_missing_and_questionable_separate_from_espn_ids(self):
        day="2026-11-14"
        payload={"response":[
            {"fixture":{"id":2005,"date":"2026-11-14T19:00:00Z"},
             "league":{"name":"Champions League"},
             "team":{"id":55,"name":"Arsenal"},
             "player":{"id":100,"name":"Example Forward","type":"Missing Fixture","reason":"Suspended"}},
            {"fixture":{"id":2005,"date":"2026-11-14T19:00:00Z"},
             "team":{"id":55,"name":"Arsenal"},
             "player":{"id":101,"name":"Example Reserve","type":"Questionable","reason":"Muscle"}},
        ]}
        x=parse_api_football_injuries(payload,"2026-11-12T12:00:00Z",day)
        self.assertEqual(len(x["players"]),2)
        self.assertEqual([p["availability"] for p in x["players"]],["reported_absent","uncertain"])
        self.assertEqual(x["players"][0]["provider_player_id"],"100")
        self.assertEqual(audit([],[],[x],[])["record_counts"]["injuries"],1)

    def test_unique_fixture_matching_requires_both_teams_and_kickoff(self):
        fixture=upcoming()
        vendor={"fixture_id":"2005","kickoff_utc":fixture["kickoff_utc"],
                "home":"Arsenal","away":"Barcelona",
                "home_provider_id":"55","away_provider_id":"66"}
        inj={"players":[{
             "fixture_id":"2005","provider_team_id":"55",
             "team":"Arsenal","player":"Example Forward",
             "provider_player_id":"77","availability":"reported_absent",
             "reason":"Suspended","type":"Missing Fixture"}]}
        r=join_vendor_injuries([fixture],[vendor],[inj])
        self.assertEqual(len(r["3001"]),1)
        self.assertEqual(r["3001"][0]["identity_match"],"unverified_cross_provider_player")
        other={**fixture,"event_id":"3002"}
        self.assertEqual(join_vendor_injuries([fixture,other],[vendor],[inj]),{})
        self.assertEqual(join_vendor_injuries([fixture],[{**vendor,"away":"Chelsea"}],[inj]),{})

    def test_opponent_history_uses_only_prior_as_of_and_shrinks(self):
        past=[]
        for i,goal in enumerate([2,0,1,4]):
            game=parse_espn_summary(summary(
                event_date=(pd.Timestamp("2026-10-01T19:00Z")+pd.Timedelta(days=i)).isoformat(),
                goal=goal,opponent="20" if i<2 else "30"),"eng.1",str(1000+i),
                "2026-10-08T12:00:00Z")
            past.append(game)
        result=player_opponent_context(upcoming(),past,[],
                                       "2026-11-12T12:00:00Z")
        row=result["players"][0]
        self.assertEqual(row["metrics"]["goals"]["prior_opponent_games"],2)
        self.assertEqual(row["metrics"]["goals"]["opponent_mean"],1.0)
        self.assertAlmostEqual(row["metrics"]["goals"]["recent_mean"],1.75)
        self.assertGreater(row["metrics"]["goals"]["shrunk_opponent_mean"],1.0)
        self.assertLess(row["metrics"]["goals"]["shrunk_opponent_mean"],1.75)
        self.assertEqual(audit([],[],[],[result])["record_counts"]["previews"],1)
        later={**past[0],"event_id":"new","captured_utc":"2026-11-13T12:00:00Z"}
        unchanged=player_opponent_context(upcoming(),past+[later],[],
                                          "2026-11-12T12:00:00Z")
        self.assertEqual(result["players"],unchanged["players"])

    def test_scoreboard_parses_club_and_national_id(self):
        p={"events":[{"id":"3001","date":"2026-11-14T19:00:00Z",
            "status":{"type":{"name":"STATUS_SCHEDULED"}},
            "competitions":[{"competitors":[
                {"homeAway":"home","team":{"id":"10","displayName":"Arsenal"}},
                {"homeAway":"away","team":{"id":"20","displayName":"Barcelona"}}]}]}]}
        d=parse_scoreboard(p,"uefa.champions")
        self.assertEqual(len(d),1)
        self.assertEqual(d[0]["home_id"],"10")
        self.assertEqual(parse_scoreboard(p,"uefa.nations")[0]["competition_code"],"uefa.nations")

    def test_fixture_vendor_parser_and_no_lineup_false_positives(self):
        v={"response":[{"fixture":{"id":2005,"date":"2026-11-14T19:00:00Z"},
            "teams":{"home":{"name":"Arsenal","id":55},"away":{"name":"Barcelona","id":66}}}]}
        self.assertEqual(parse_api_fixtures(v)[0]["away_provider_id"],"66")
        r=parse_espn_summary(summary("2026-11-14T19:00:00Z","STATUS_SCHEDULED"),
                             "uefa.champions","3001","2026-11-14T18:00:00Z")
        r["lineup_confirmed"]=True
        with self.assertRaisesRegex(ValueError,"incorrectly marked confirmed"):
            audit([],[r],[],[])

    def test_conflicting_capture_utc_never_accepted(self):
        game=parse_espn_summary(summary(),"uefa.champions","1001","2026-11-12T12:00:00Z")
        game["captured_utc"]="2026-11-10T12:00:00Z"
        with self.assertRaisesRegex(ValueError,"captured before kickoff"):
            audit([game],[],[],[])


if __name__=="__main__":
    unittest.main()
