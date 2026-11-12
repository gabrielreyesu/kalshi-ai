import unittest
import numpy as np
import pandas as pd
from kalshi_intl.regression_compare import compare_models


class TestRegressionComparison(unittest.TestCase):
    def sample(self):
        rng=np.random.default_rng(10)
        rows=[]
        for i in range(55):
            p=rng.dirichlet([2,2,2])
            y=rng.choice(['H','D','A'],p=p)
            rows.append(dict(event_id=str(i),date=f'2023-09-{i%25+1:02d}',res=y,
                             p_H=p[0],p_D=p[1],p_A=p[2]))
        return pd.DataFrame(rows)

    def test_identity_returns_zero_difference(self):
        a=self.sample()
        metrics,ci=compare_models(a,a.sample(frac=1,random_state=1),bootstrap=20)
        self.assertEqual(metrics.n.iloc[0],55)
        self.assertTrue(np.allclose(ci.delta_candidate_minus_baseline,0))

    def test_mismatched_fixture_rejected(self):
        a=self.sample()
        with self.assertRaisesRegex(ValueError,'different fixtures'):
            compare_models(a,a.iloc[1:],bootstrap=0)

    def test_outcome_mismatch_rejected(self):
        a=self.sample()
        b=a.copy();b.loc[0,'res']='X'
        with self.assertRaisesRegex(ValueError,'Different dates/outcomes'):
            compare_models(a,b,bootstrap=0)

    def test_duplicate_rejected(self):
        a=self.sample()
        with self.assertRaisesRegex(ValueError,'duplicated event_id'):
            compare_models(pd.concat([a,a.iloc[[0]]]),a,bootstrap=0)


if __name__=='__main__':
    unittest.main()
