import unittest
import pandas as pd
from kalshi_intl.data_quality import audit_history


def sample(n=120):
    rows=[]
    for i in range(n):
        status="STATUS_FINAL_AET" if i==0 else ("STATUS_FINAL_PEN" if i==1 else "STATUS_FULL_TIME")
        rows.append({"event_id":str(i),"date":"2024-05-01","status":status,"neutral":False,
                     "final":True,"code":"fifa.world","hs":2.,"as_":1.})
    return pd.DataFrame(rows)


class TestHistoryAudit(unittest.TestCase):
    def test_neutral_warning_and_regulation_count(self):
        audit=audit_history(sample())
        self.assertEqual(audit["n_neutral"],0)
        self.assertTrue(audit["warnings"])
        self.assertEqual(audit["n_extra_time_or_penalty_final"],2)
        self.assertEqual(audit["n_regulation_final"],118)

    def test_neutral_present_does_not_warn(self):
        d=sample();d.loc[0,"neutral"]=True
        self.assertFalse(audit_history(d)["warnings"])

    def test_duplicate_fails(self):
        d=sample()
        with self.assertRaisesRegex(ValueError,"duplicated event_id"):
            audit_history(pd.concat([d,d.iloc[[0]]]))

    def test_missing_scores_fails(self):
        d=sample();d.loc[3,"hs"]=None
        with self.assertRaisesRegex(ValueError,"missing scores"):
            audit_history(d)


if __name__=="__main__":
    unittest.main()
