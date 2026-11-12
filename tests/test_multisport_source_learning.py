import unittest
import numpy as np
import pandas as pd

from kalshi_external.learn import fit_blend_candidates


def multisport(n=130):
    records, outcomes = [], []
    for sport in ("nfl", "nba", "intl", "club"):
        tri=sport in ("intl","club")
        classes=("H","D","A") if tri else ("H","A")
        for i in range(n):
            outcome=classes[i%len(classes)]
            p=(.08*np.ones(len(classes),dtype=float))
            p[i%len(classes)]=1-.08*(len(classes)-1)
            other=np.ones(len(classes))/len(classes)
            kick=(pd.Timestamp("2025-01-01T19:00Z")+pd.Timedelta(days=i)).isoformat()
            stamp=(pd.Timestamp(kick)-pd.Timedelta(days=1)).isoformat()
            rec={"sport":sport,"source_id":"oddsapi_bookmaker_consensus",
                 "source_kind":"market","event_id":str(i),
                 "kickoff_utc":kick,"published_utc":stamp,"captured_utc":stamp}
            rec.update({"p_H":p[0],"p_A":p[-1],"p_D":p[1] if tri else None,
                        "model_p_H":other[0],"model_p_A":other[-1],
                        "model_p_D":other[1] if tri else None})
            records.append(rec)
            outcomes.append({"sport":sport,"event_id":str(i),"result":outcome})
    return records,pd.DataFrame(outcomes)


class MultiSportLearning(unittest.TestCase):
    def test_sport_specific_challengers_no_cross_contamination(self):
        hist,results=multisport()
        out=fit_blend_candidates(hist,results,min_train=60,min_test=25,bootstrap=20)
        self.assertEqual(set(x["sport"] for x in out),{"nfl","nba","intl","club"})
        self.assertEqual(len(out),4)
        self.assertTrue(all(x["n_test"]>=25 for x in out))
        self.assertTrue(all(x["candidate_weight"]>0 for x in out))
        self.assertTrue(all(x["test_candidate_minus_baseline"]<0 for x in out))

    def test_same_event_id_other_sport_is_not_duplicate(self):
        hist,results=multisport()
        out=fit_blend_candidates(hist,results,bootstrap=0)
        self.assertEqual(len(out),4)

    def test_future_results_do_not_retrain_prior_season(self):
        hist,results=multisport()
        a=fit_blend_candidates(hist,results,bootstrap=0)
        changed=results.copy()
        changed.loc[(changed.sport=="nba")&
                    (changed.event_id.astype(int)>110),"result"]="A"
        b=fit_blend_candidates(hist,changed,bootstrap=0)
        wa={x["sport"]:x["candidate_weight"] for x in a}
        wb={x["sport"]:x["candidate_weight"] for x in b}
        self.assertEqual(wa,wb)

    def test_read_only_third_party_not_trained(self):
        h,results=multisport()
        for x in h:
            x["source_kind"]="research_only"
        self.assertEqual(fit_blend_candidates(h,results),[])

    def test_unscored_sport_not_counted(self):
        h,results=multisport()
        results=results[results.sport!="nba"]
        out=fit_blend_candidates(h,results,bootstrap=0)
        self.assertEqual(set(x["sport"] for x in out),{"nfl","intl","club"})


if __name__=="__main__":
    unittest.main()
