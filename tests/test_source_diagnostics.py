import tempfile
import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from kalshi_external.cli import append_history
from kalshi_external.source_diagnostics import diagnostics


def example(n=110):
    rows = []
    outcomes = []
    for i in range(n):
        kickoff=(pd.Timestamp("2025-08-01T20:00Z")+pd.Timedelta(days=i))
        decision=("H","A")[i%2]
        base={"source_id":"bookmaker_consensus","source_kind":"market",
              "sport":"nfl","event_id":str(i),"kickoff_utc":kickoff.isoformat(),
              "published_utc":(kickoff-pd.Timedelta(hours=30)).isoformat(),
              "model_p_H":.51,"model_p_D":None,"model_p_A":.49,
              "p_H":.50,"p_D":None,"p_A":.50}
        for j in (0,1):
            row=dict(base)
            row["captured_utc"]=(kickoff-pd.Timedelta(hours=24-4*j)).isoformat()
            row["published_utc"]=row["captured_utc"]
            row["p_H"]=.75 if i%2==0 else .25
            if j==0:
                row["p_H"]=.60 if i%2==0 else .40
            row["p_A"]=1-row["p_H"]
            rows.append(row)
        outcomes.append({"sport":"nfl","event_id":str(i),"result":decision})
    return rows,pd.DataFrame(outcomes)


class TestMoreSnapshots(unittest.TestCase):
    def test_distinct_snapshots_same_day_preserved(self):
        r,_=example(1)
        with tempfile.TemporaryDirectory() as td:
            file=Path(td)/"history.jsonl"
            h,n=append_history(file,r)
            self.assertEqual((len(h),n),(2,2))
            newer,n=append_history(file,r)
            self.assertEqual((len(newer),n),(2,0))

    def test_paired_metrics_and_date_cluster_uncertainty(self):
        r,outcomes=example()
        metrics,mov=diagnostics(r,outcomes,bootstrap=100)
        self.assertEqual(metrics.n_events.iloc[0],110)
        self.assertEqual(metrics.n_dates.iloc[0],110)
        self.assertGreater(metrics.gain_logloss.iloc[0],0)
        self.assertTrue(metrics.sufficient_history.iloc[0])
        self.assertTrue(metrics.candidate_research_only.iloc[0])
        self.assertEqual(mov.snapshots.iloc[0],220)
        self.assertEqual(mov.events_with_multiple_snapshots.iloc[0],110)
        self.assertGreater(mov.mean_open_close_abs_p_home.iloc[0],0)

    def test_late_snapshot_excluded(self):
        rows,o=example(3)
        late=dict(rows[0])
        late["captured_utc"]=(pd.Timestamp(late["kickoff_utc"])-pd.Timedelta(minutes=1)).isoformat()
        late["published_utc"]=late["captured_utc"]
        m,mov=diagnostics(rows+[late],o,bootstrap=0)
        self.assertEqual(mov.snapshots.iloc[0],6)
        self.assertEqual(len(m),1)

    def test_duplicate_outcomes_raise(self):
        rows,o=example(2)
        with self.assertRaisesRegex(ValueError,"Duplicate completed outcomes"):
            diagnostics(rows,pd.concat([o,o.head(1)]))

    def test_empty_safe(self):
        m,mov=diagnostics([],pd.DataFrame(columns=["sport","event_id","result"]))
        self.assertTrue(m.empty and mov.empty)

    def test_small_sample_not_promoted(self):
        rows,o=example(8)
        m,_=diagnostics(rows,o,bootstrap=0)
        self.assertFalse(m.sufficient_history.iloc[0])
        self.assertFalse(m.candidate_research_only.iloc[0])

if __name__=="__main__":
    unittest.main()
