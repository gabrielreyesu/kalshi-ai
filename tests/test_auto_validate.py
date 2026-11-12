import unittest
import numpy as np
import pandas as pd
from kalshi_intl.auto_validate import run


def synthetic_history():
    """Deterministic OOS-shaped examples; no report/ESPN cache dependency."""
    rng = np.random.default_rng(17)
    rows = []
    for season in (2021, 2022, 2023):
        for i in range(90):
            p = rng.dirichlet([2.0, 1.3, 1.8])
            result = rng.choice(['H', 'D', 'A'], p=p)
            rows.append({
                'temporada': f'{season}/{(season + 1) % 100:02d}',
                'date': f'{season}-08-{i % 28 + 1:02d}',
                'res': result,
                'friendly': i % 2,
                'p_H': p[0], 'p_D': p[1], 'p_A': p[2],
                'cal_H': p[0], 'cal_D': p[1], 'cal_A': p[2],
            })
    return pd.DataFrame(rows)


class TestAutoValidate(unittest.TestCase):
    def test_short_history_rejected(self):
        with self.assertRaises(ValueError):
            run(pd.DataFrame())

    def test_holdout_and_no_promotion(self):
        metrics, comparisons, decisions, hold = run(
            synthetic_history(), min_history=20, min_gain=0.005, bootstrap=100
        )
        self.assertEqual(len(hold), 90)
        self.assertEqual(int(comparisons.holdout_season.iloc[0]), 2023)
        self.assertFalse(comparisons.promote_recommended.any())
        self.assertEqual(set(metrics.method), {'original', 'calibrated', 'adaptive'})
        self.assertEqual(len(decisions), 6)

    def test_invalid_duplicate_rows(self):
        d = synthetic_history()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            run(pd.concat([d, d.iloc[[0]]], ignore_index=True), bootstrap=100)


if __name__ == '__main__':
    unittest.main()
