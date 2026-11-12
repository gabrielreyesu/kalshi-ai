import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from kalshi_external.soccer_starter_impact import (
    research_snapshot, build_role_context,
)
from kalshi_external.soccer_confirmed_lineups import (
    parse_confirmed_lineups, match_lineup_to_espn_roster,
)
from kalshi_external.soccer_impact_collect import build, baseline_asof
from kalshi_external.soccer_impact_audit import audit
from kalshi_external.soccer_impact_shadow import evaluate, adjust_probabilities


NOW="2026-11-12T14:00:00Z"
KICK="2026-11-13T19:00:00Z"


def fixture():
    return {"sport":"soccer","event_id":"777","competition_code":"uefa.champions",
            "home_id":"10","away_id":"20","home":"Arsenal","away":"Barcelona",
            "kickoff_utc":KICK}


def past_games():
    games=[]
    for i in range(6):
        date=(pd.Timestamp("2026-10-01T19:00Z")+pd.Timedelta(days=i))
        games.append({
            "source_id":"espn_soccer_final_player_stats","sport":"soccer",
            "event_id":str(i),"competition_code":"uefa.champions",
            "kickoff_utc":date.isoformat(),
            "captured_utc":(date+pd.Timedelta(hours=3)).isoformat(),
            "teams":[{
                "team_id":"10","opponent_id":"20",
                "members":[
                    {"player_id":"101","player":"Star Forward","position":"FW",
                     "starter_reported":True,"metrics":{"minutes":90,"goals":1}},
                    *([{"player_id":"102","player":"Backup Forward","position":"FW",
                        "starter_reported":False,"metrics":{"minutes":30,"goals":0}}]
                      if i%2==0 else [])
                ]
            },{
                "team_id":"20","opponent_id":"10","members":[{
                    "player_id":"201","player":"Goalkeeper Two","position":"GK",
                    "starter_reported":True,"metrics":{"minutes":90,"saves":5}
                }]
            }]
        })
    return games


def preview():
    return {"source_id":"soccer_player_opponent_preview",
            "sport":"soccer","event_id":"777",
            "competition_code":"uefa.champions",
            "home_id":"10","away_id":"20","captured_utc":NOW,
            "kickoff_utc":KICK,
            "api_injury_coverage_confirmed":True,
            "provider_injury_context":[{
                "espn_team_id":"10","player":"Star Forward",
                "type":"Missing Fixture","availability":"reported_absent",
                "provider_team_id":"55","provider_player_id":"9001",
            }]}


def provider_teams():
    return [
        {"team":{"id":55},"formation":"4-3-3",
         "startXI":[{"player":{"id":str(i),"name":"Backup Forward" if i==1 else f"A{i}","pos":"F"}}
                    for i in range(1,12)],
         "substitutes":[]},
        {"team":{"id":66},"formation":"4-4-2",
         "startXI":[{"player":{"id":str(i),"name":f"B{i}","pos":"M"}}
                    for i in range(21,32)],
         "substitutes":[]}
    ]


def vendor_fixture():
    return {"fixture_id":"500","kickoff_utc":KICK,
            "home":"Arsenal","away":"Barcelona",
            "home_provider_id":"55","away_provider_id":"66"}


class RoleImportanceTests(unittest.TestCase):
    def test_absent_high_role_and_replacement_is_not_confirmed(self):
        r=research_snapshot(preview(),past_games())
        self.assertEqual(r["home_minus_away_absence_exposure"],-100)
        missing=r["teams"][0]["missing_starter_replacement_research"][0]
        self.assertEqual(missing["player_role_index"],100.0)
        self.assertEqual(len(missing["replacement_candidates"]),1)
        self.assertAlmostEqual(missing["replacement_candidates"][0]["role_index_0_100"],16.67)
        self.assertAlmostEqual(missing["cover_gap_index_points"],83.33)
        self.assertFalse(missing["cover_confirmed"])
        self.assertFalse(r["lineup_confirmed"])
        self.assertEqual(audit([], [r],now="2026-11-12T14:10:00Z")["pregame_player_importance_snapshots"],1)

    def test_unknown_status_or_name_never_guesses_role(self):
        p=preview()
        p["provider_injury_context"][0]["player"]="Unknown Athlete"
        r=research_snapshot(p,past_games())
        self.assertIsNone(r["home_minus_away_absence_exposure"])
        self.assertEqual(r["teams"][0]["unresolved_provider_reports"][0]["matching_confidence"],
                         "ambiguous_or_unknown_player")
        p["provider_injury_context"][0]["espn_team_id"]="99"
        r2=research_snapshot(p,past_games())
        self.assertIsNone(r2["home_minus_away_absence_exposure"])

    def test_no_minutes_stays_unevaluable_not_fictitious(self):
        games=past_games()
        for g in games:
            for t in g["teams"]:
                for player in t["members"]:
                    player["metrics"].pop("minutes",None)
        r=research_snapshot(preview(),games)
        self.assertIsNone(r["home_minus_away_absence_exposure"])
        self.assertIsNone(r["teams"][0]["missing_starter_replacement_research"][0]["player_role_index"])

    def test_other_competition_same_player_ids_never_counted(self):
        games=past_games()
        for g in games:
            g["competition_code"]="eng.1"
        result=research_snapshot(preview(),games)
        self.assertIsNone(result["home_minus_away_absence_exposure"])
        self.assertEqual(result["teams"][0]["player_roles_evaluable"],0)

    def test_future_stats_or_late_injury_not_accepted(self):
        games=past_games()
        for g in games[:4]:
            g["captured_utc"]="2026-11-15T01:00:00Z"
        r=research_snapshot(preview(),games)
        self.assertIsNone(r["home_minus_away_absence_exposure"])
        p=preview()
        p["captured_utc"]="2026-11-14T01:00:00Z"
        with self.assertRaisesRegex(ValueError,"pre-match"):
            research_snapshot(p,past_games())

    def test_pre_match_two_complete_starting_XI_may_be_confirmed(self):
        future=fixture()
        observed="2026-11-13T18:20:00Z"
        record=parse_confirmed_lineups({"response":provider_teams()},
                                       future,vendor_fixture(),observed)
        self.assertTrue(record["lineup_confirmed"])
        self.assertEqual(len(record["teams"][0]["startXI"]),11)
        self.assertEqual(audit([record],[],now="2026-11-13T18:21:00Z")
                         ["confirmed_pregame_XI_snapshots"],1)
        p=preview()
        p["captured_utc"]=observed
        roster=build_role_context(p,past_games(),observed)
        matched=match_lineup_to_espn_roster(record,roster,observed)
        r=research_snapshot(p,past_games(),matched)
        absence=r["teams"][0]["missing_starter_replacement_research"][0]
        self.assertTrue(absence["cover_confirmed"])
        self.assertEqual(absence["replacement_candidates"][0]["player"],"Backup Forward")

    def test_incomplete_or_postgame_XI_rejected(self):
        future=fixture()
        teams=provider_teams()
        teams[0]["startXI"].pop()
        self.assertIsNone(parse_confirmed_lineups(
            {"response":teams},future,vendor_fixture(),"2026-11-13T18:20:00Z"))
        self.assertIsNone(parse_confirmed_lineups(
            {"response":provider_teams()},future,vendor_fixture(),
            "2026-11-13T19:03:00Z"))

    def test_baseline_only_if_known_before_capture(self):
        r=pd.DataFrame([{"event_id":"777","code":"uefa.champions",
            "created_utc":"2026-11-12T13:00:00Z",
            "kickoff_utc":KICK,"grupo":"clubes",
            "p_H":.50,"p_D":.25,"p_A":.25}])
        self.assertIsNotNone(baseline_asof(r,preview()))
        later=r.copy()
        later["created_utc"]="2026-11-12T15:00:00Z"
        self.assertIsNone(baseline_asof(later,preview()))
        rows=build([preview()],past_games(),[],r,NOW)
        self.assertEqual(rows[0]["model_baseline_asof"]["model_group"],"club")
        self.assertTrue(rows[0]["eligible_for_matched_1x2_evaluation"])

    def test_shadow_insufficient_data_always_blocked(self):
        r=research_snapshot(preview(),past_games())
        r["model_baseline_asof"]={"p_H":.5,"p_D":.25,"p_A":.25,
                                  "model_group":"club",
                                  "model_created_utc":"2026-11-12T13:00:00Z"}
        outcomes=pd.DataFrame([{"sport":"club","event_id":"777","result":"A"}])
        result=evaluate([r],outcomes,as_of="2026-11-15T00:00:00Z",
                        min_train=2,min_test=1)
        self.assertEqual(result["status"],"INSUFFICIENT_PROSPECTIVE_OUTCOMES")
        self.assertFalse(result["model_promoted"])

    def test_shadow_later_dates_and_no_automatic_promotion(self):
        hist,results=[],[]
        for i in range(210):
            date=pd.Timestamp("2026-03-01T20:00Z")+pd.Timedelta(days=i)
            win_home=i%2==0
            hist.append({
                "source_id":"soccer_starter_replacement_research",
                "status":"RESEARCH_ONLY_NOT_MODEL_INPUT",
                "sport":"soccer","event_id":str(i),"competition_code":"uefa.champions",
                "captured_utc":(date-pd.Timedelta(hours=24)).isoformat(),
                "kickoff_utc":date.isoformat(),
                "home_minus_away_absence_exposure":100 if win_home else -100,
                "model_baseline_asof":{
                    "p_H":.38,"p_D":.24,"p_A":.38,"model_group":"club",
                    "model_created_utc":(date-pd.Timedelta(hours=25)).isoformat()
                },
            })
            results.append({"sport":"club","event_id":str(i),
                            "result":"H" if win_home else "A"})
        output=evaluate(hist,pd.DataFrame(results),
                        as_of="2026-10-15T00:00:00Z",bootstrap=80,
                        min_train=120,min_test=50,min_test_dates=20)
        self.assertEqual(output["status"],"RESEARCH_ONLY_PAIRED_HOLDOUT")
        self.assertGreater(output["holdout_logloss_gain"],0)
        self.assertTrue(output["candidate_research_only"])
        self.assertFalse(output["model_promoted"])
        self.assertGreater(output["n_test"],49)

    def test_probabilities_preserve_draw_and_normalize(self):
        x=adjust_probabilities([.4,.25,.35],100,.2)
        self.assertAlmostEqual(sum(x),1)
        self.assertGreater(x[0],.4)
        self.assertLess(x[2],.35)


if __name__=="__main__":
    unittest.main()
