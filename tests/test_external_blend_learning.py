import unittest

import pandas as pd

from kalshi_external.learn import fit_blend_candidates


def data(n=140):
    rows=[];results=[]
    for i in range(n):
        event=f"future-{i}"
        date=(pd.Timestamp("2025-01-01T17:00Z")+pd.Timedelta(days=i)).isoformat()
        captured=(pd.Timestamp(date)-pd.Timedelta(days=1)).isoformat()
        # Deliberately informative source, weak original baseline.
        choice=("H","D","A")[i%3]
        source=([.8,.1,.1],[.1,.8,.1],[.1,.1,.8])[i%3]
        rows.append({"event_id":event,"sport":"intl","source_id":"licensed_expert",
                     "source_kind":"analyst","kickoff_utc":date,
                     "captured_utc":captured,"published_utc":captured,
                     "p_H":source[0],"p_D":source[1],"p_A":source[2],
                     "model_p_H":.35,"model_p_D":.30,"model_p_A":.35})
        results.append({"event_id":event,"sport":"intl","result":choice})
    return rows,pd.DataFrame(results)


class TestExternalLearning(unittest.TestCase):
    def test_train_on_earlier_games_test_on_future(self):
        history,outcomes=data()
        c=fit_blend_candidates(history,outcomes,min_train=50,min_test=20,bootstrap=25)
        self.assertEqual(len(c),1)
        self.assertGreater(c[0]["candidate_weight"],0)
        self.assertLess(c[0]["test_candidate_minus_baseline"],0)
        self.assertFalse("prod" in c[0].get("status","").lower() and c[0].get("production_changed",False))

    def test_future_outcomes_do_not_change_selected_weight(self):
        h,outcomes=data()
        a=fit_blend_candidates(h,outcomes,bootstrap=0)
        future=outcomes.copy()
        future.loc[future.index>115,"result"]="A"
        b=fit_blend_candidates(h,future,bootstrap=0)
        self.assertEqual(a[0]["candidate_weight"],b[0]["candidate_weight"])

    def test_read_only_observation_never_fit(self):
        history,outcomes=data()
        for h in history:
            h["source_kind"]="research_only"
        self.assertEqual(fit_blend_candidates(history,outcomes),[])

    def test_insufficient_resolved_examples(self):
        history,outcomes=data(n=35)
        self.assertEqual(fit_blend_candidates(history,outcomes),[])


if __name__=="__main__":
    unittest.main()
