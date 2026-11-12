import unittest
import pandas as pd
from kalshi_external.skorbot import parse_skorbot, CREDIT


def registry():
    return pd.DataFrame([{
        "event_id":"cl1","sport":"club","home":"Galatasaray","away":"Aston Villa",
        "kickoff_utc":"2026-11-24T17:45:00Z","created_utc":"2026-11-12T15:00:00Z",
        "p_H":.28,"p_D":.24,"p_A":.48
    }])


class TestSkorbot(unittest.TestCase):
    def test_pre_match_and_credit(self):
        payload={"generated_at":"2026-11-13T10:00:00Z","matches":[{
            "status":"scheduled","result":None,
            "home":"Galatasaray","away":"Aston Villa",
            "kickoff":"2026-11-24T17:45:00Z",
            "updated_at":"2026-11-13T09:00:00Z",
            "probabilities":{"home":.38,"draw":.25,"away":.37}
        }]}
        r=parse_skorbot(payload,registry(),"2026-11-13T11:00:00Z")
        self.assertEqual(len(r),1)
        self.assertEqual(r[0]["attribution"],CREDIT)
        self.assertEqual(r[0]["source_kind"],"research_only")

    def test_result_or_post_match_update_excluded(self):
        item={"status":"finished","result":{"home":1,"away":0},
              "home":"Galatasaray","away":"Aston Villa",
              "kickoff":"2026-11-24T17:45:00Z",
              "updated_at":"2026-11-25T11:00:00Z",
              "probabilities":{"home":.38,"draw":.25,"away":.37}}
        p={"matches":[item]}
        self.assertEqual(parse_skorbot(p,registry(),"2026-11-26T12:00:00Z"),[])
        item["status"]="scheduled";item["result"]=None
        self.assertEqual(parse_skorbot(p,registry(),"2026-11-23T12:00:00Z"),[])

    def test_unmatched_not_recorded(self):
        p={"matches":[{"status":"scheduled","result":None,
          "home":"Liverpool","away":"Aston Villa",
          "kickoff":"2026-11-24T17:45:00Z",
          "updated_at":"2026-11-13T09:00:00Z",
          "probabilities":{"home":.38,"draw":.25,"away":.37}}]}
        self.assertEqual(parse_skorbot(p,registry(),"2026-11-13T11:00:00Z"),[])
