import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from kalshi_external.core import future_match, make_record, historical_scores, team_name, load_registry
from kalshi_external.providers import odds_consensus, parse_football_forecast, parse_licensed_analysts
from kalshi_external.cli import append_history, load_completed_intl


def registry():
    return pd.DataFrame([
        {"sport":"intl","event_id":"soc1","home":"Türkiye","away":"Belgium",
         "kickoff_utc":"2026-11-20T15:00:00Z","created_utc":"2026-11-13T10:00:00Z",
         "p_H":.25,"p_D":.25,"p_A":.50},
        {"sport":"nfl","event_id":"nfl1","home":"New York Giants","away":"Washington Commanders",
         "kickoff_utc":"2026-11-20T20:00:00Z","created_utc":"2026-11-13T10:00:00Z",
         "p_home":.55}
    ])


def record():
    return make_record(registry().iloc[0], "expert_test",
                       "2026-11-19T12:00:00Z", [.30,.20,.50],
                       "2026-11-18T12:00:00Z")


class TestExternalContracts(unittest.TestCase):
    def test_team_alias(self):
        self.assertEqual(team_name("Türkiye"),team_name("Turkey"))

    def test_only_unique_matching(self):
        d=registry()
        found=future_match(d,"intl","Turkey","Belgium",
                           "2026-11-20T15:00:00Z","2026-11-19T12:00:00Z")
        self.assertEqual(found.event_id,"soc1")
        self.assertIsNone(future_match(d,"intl","Turkey","Spain",
                          "2026-11-20T15:00:00Z","2026-11-19T12:00:00Z"))
        self.assertIsNone(future_match(pd.concat([d,d.iloc[[0]]]),"intl",
                          "Turkey","Belgium","2026-11-20T15:00:00Z","2026-11-19T12:00:00Z"))

    def test_late_fixtures_not_accepted(self):
        d=registry()
        self.assertIsNone(future_match(d,"intl","Turkey","Belgium",
             "2026-11-20T15:00:00Z","2026-11-20T14:55:00Z"))
        with self.assertRaises(ValueError):
            make_record(d.iloc[0],"x","2026-11-19T12:00:00Z",
                        [.2,.3,.5],published="2026-11-20T14:00:00Z")

    def test_valid_probabilities_and_baseline(self):
        r=record()
        self.assertAlmostEqual(r["p_H"]+r["p_D"]+r["p_A"],1)
        self.assertAlmostEqual(r["model_p_H"]+r["model_p_D"]+r["model_p_A"],1)
        with self.assertRaises(ValueError):
            make_record(registry().iloc[0],"x","2026-11-19T12:00:00Z",[-.1,.2,.9])

    def test_nfl_two_way(self):
        r=make_record(registry().iloc[1],"consensus",
                      "2026-11-19T12:00:00Z",[.56,.44])
        self.assertIsNone(r["p_D"])
        self.assertAlmostEqual(r["model_p_A"],.45)

    def test_odds_consensus_de_vigs_and_ignores_inplay(self):
        base={"sport_key":"americanfootball_nfl","commence_time":"2026-11-20T20:00:00Z",
              "home_team":"New York Giants","away_team":"Washington Commanders"}
        books=[]
        for ident in range(3):
            books.append({"key":str(ident),"markets":[{"key":"h2h",
             "last_update":"2026-11-19T11:50:00Z","outcomes":[
             {"name":"New York Giants","price":1.80},
             {"name":"Washington Commanders","price":2.10}]}]})
        events=[{**base,"bookmakers":books}]
        r=odds_consensus(events,registry(),"2026-11-19T12:00:00Z")
        self.assertEqual(len(r),1)
        self.assertEqual(r[0]["n_bookmakers"],3)
        self.assertAlmostEqual(r[0]["p_H"]+r[0]["p_A"],1)
        self.assertEqual(odds_consensus(events,registry(),"2026-11-20T20:01:00Z"),[])

    def test_api_football_model_percent(self):
        fixture={"fixture":{"id":123,"date":"2026-11-20T15:00:00Z",
                 "status":{"short":"NS"}},"teams":{"home":{"name":"Turkey"},
                                                  "away":{"name":"Belgium"}}}
        prediction={"response":[{"predictions":{"percent":{
            "home":"22%","draw":"28%","away":"50%"}}}]}
        r=parse_football_forecast(fixture,prediction,registry(),"2026-11-19T12:00:00Z")
        self.assertAlmostEqual(r["p_D"],.28)
        fixture["fixture"]["status"]["short"]="1H"
        self.assertIsNone(parse_football_forecast(fixture,prediction,registry(),"2026-11-19T12:00:00Z"))

    def test_licensed_analyst_requires_timestamp_and_consent(self):
        item={"licensed_for_modeling":True,"sport":"intl","event_id":"soc1",
              "source_id":"analyst_A","published_utc":"2026-11-19T10:00:00Z",
              "p_H":.20,"p_D":.30,"p_A":.50}
        records=parse_licensed_analysts([item],registry(),"2026-11-19T12:00:00Z")
        self.assertEqual(len(records),1)
        self.assertEqual(parse_licensed_analysts([{**item,"licensed_for_modeling":False}],
                        registry(),"2026-11-19T12:00:00Z"),[])
        self.assertEqual(parse_licensed_analysts([{**item,"published_utc":"2026-11-21T00:00:00Z"}],
                        registry(),"2026-11-19T12:00:00Z"),[])

    def test_history_dedup_by_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"history.jsonl"
            a,n=append_history(path,[record(),record()])
            self.assertEqual((len(a),n),(1,1))
            b,n=append_history(path,[record()])
            self.assertEqual((len(b),n),(1,0))

    def test_scoring_only_resolved_and_paired(self):
        records=[record()]
        outcomes=pd.DataFrame([{"event_id":"soc1","sport":"intl","result":"A"}])
        a=historical_scores(records,outcomes,min_cases=1)
        self.assertEqual(a.n.iloc[0],1)
        self.assertGreater(a.model_logloss.iloc[0],0)
        self.assertTrue(a.eligible_for_review.iloc[0])

    def test_scores_ignore_unresolved_games(self):
        d=historical_scores([record()],pd.DataFrame(
            [{"event_id":"different","sport":"intl","result":"A"}]))
        self.assertTrue(d.empty)

    def test_no_lookahead_in_scoring(self):
        r=record();r["captured_utc"]="2026-11-21T12:00:00Z"
        out=pd.DataFrame([{"event_id":"soc1","sport":"intl","result":"A"}])
        self.assertTrue(historical_scores([r],out).empty)

    def test_load_completed_excludes_extra_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"raw.csv"
            pd.DataFrame([
                {"event_id":"a","res":"H","status":"STATUS_FULL_TIME","target":True},
                {"event_id":"b","res":"A","status":"STATUS_FINAL_AET","target":True},
                {"event_id":"c","res":"D","status":"STATUS_FINAL","target":True},
            ]).to_csv(path,index=False)
            o=load_completed_intl(path)
            self.assertEqual(set(o.event_id),{"a","c"})


if __name__=="__main__":unittest.main()
