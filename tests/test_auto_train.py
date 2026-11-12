import unittest
import numpy as np
import pandas as pd
from kalshi_intl.auto_train import train_candidate, predict_candidate


def sample():
    rng = np.random.default_rng(24)
    rows = []
    for i in range(90):
        p = rng.dirichlet([2, 2, 2])
        rows.append({'event_id': str(i), 'date': '2022-11-%02d' % (i % 28 + 1),
                     'res': rng.choice(['H','D','A'], p=p),
                     'p_H': p[0], 'p_D': p[1], 'p_A': p[2]})
    return pd.DataFrame(rows)


class TestAutoTraining(unittest.TestCase):
    def test_candidate_from_prior_history(self):
        d = sample()
        candidate = train_candidate(d, as_of='2023-01-01', min_samples=50)
        self.assertEqual(candidate['n_historical_matches'], 90)
        self.assertEqual(candidate['validation_status'], 'CHALLENGER_ONLY_NOT_PRODUCTION_APPROVED')
        q = predict_candidate(candidate, d[['p_H', 'p_D', 'p_A']].to_numpy())
        self.assertTrue(np.isfinite(q).all())
        np.testing.assert_allclose(q.sum(axis=1), 1)

    def test_future_results_do_not_change_candidate(self):
        d = sample()
        d2 = pd.concat([d, d.head(1).assign(event_id='future', date='2024-01-01', res='A')], ignore_index=True)
        a = train_candidate(d, as_of='2023-01-01', min_samples=50)
        b = train_candidate(d2, as_of='2023-01-01', min_samples=50)
        self.assertEqual(a, b)

    def test_duplicate_games_rejected(self):
        d = sample()
        d = pd.concat([d, d.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            train_candidate(d, as_of='2023-01-01', min_samples=50)

    def test_not_enough_prior_games(self):
        with self.assertRaisesRegex(ValueError, 'Only 0'):
            train_candidate(sample(), as_of='2021-01-01', min_samples=50)


if __name__ == '__main__':
    unittest.main()
