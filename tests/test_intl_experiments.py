import unittest
import numpy as np
import pandas as pd
from kalshi_intl.experiments import evaluate_methods

def sample():
    rng=np.random.default_rng(51);rows=[]
    for season in (2021,2022,2023):
        for i in range(65):
            p=rng.dirichlet([2.5,1.5,2])
            rows.append({'event_id':f'{season}_{i}','date':f'{season}-08-{i%28+1:02d}',
                'temporada':f'{season}/{(season+1)%100:02d}',
                'res':rng.choice(['H','D','A'],p=p),'p_H':p[0],'p_D':p[1],'p_A':p[2]})
    return pd.DataFrame(rows)

class TestExperiments(unittest.TestCase):
    def test_normalized_probability_and_paired_rows(self):
        pred,season,overall,ci,settings=evaluate_methods(sample(),min_train=60,bootstrap=20)
        self.assertEqual(set(overall.method),{'original','logit_full','blend_25','blend_50','temperature'})
        self.assertEqual(set(ci.method),set(overall.method)-{'original'})
        self.assertEqual(pred.groupby('method').size().nunique(),1)
        np.testing.assert_allclose(pred[['p_H','p_D','p_A']].sum(axis=1),1)
        self.assertTrue((settings.n_training>0).all())

    def test_future_results_never_change_same_season_predictions(self):
        d=sample()
        a=evaluate_methods(d,min_train=60,bootstrap=0)[0]
        altered=d.copy();altered.loc[altered.temporada=='2023/24','res']='D'
        b=evaluate_methods(altered,min_train=60,bootstrap=0)[0]
        a=a[a.temporada=='2023/24'].sort_values(['method','event_id']).reset_index(drop=True)
        b=b[b.temporada=='2023/24'].sort_values(['method','event_id']).reset_index(drop=True)
        np.testing.assert_allclose(a[['p_H','p_D','p_A']],b[['p_H','p_D','p_A']])

    def test_reject_duplicate_ids(self):
        d=sample()
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            evaluate_methods(pd.concat([d,d.head(1)],ignore_index=True))
if __name__=='__main__':unittest.main()
