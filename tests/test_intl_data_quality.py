import unittest
import pandas as pd

from kalshi_intl.data_quality import audit_history
from kalshi_intl.model import IntlModel, FEATS


def sample(n=120):
    rows = []
    for i in range(n):
        status = "STATUS_FINAL_AET" if i == 0 else ("STATUS_FINAL_PEN" if i == 1 else "STATUS_FULL_TIME")
        r = {"event_id": str(i), "date": "2024-05-01", "status": status,
             "neutral": False, "final": True, "code": "fifa.world", "hs": 2., "as_": 1.}
        r.update({k: float(i % 3 + 1) for k in FEATS})
        r["neutral"] = 0
        rows.append(r)
    return pd.DataFrame(rows)


class TestHistoryDataQuality(unittest.TestCase):
    def test_audit_reports_neutral_gap_and_extra_time(self):
        a = audit_history(sample())
        self.assertEqual(a["n_neutral"], 0)
        self.assertEqual(a["n_extra_time_final_excluded_from_goal_fit"], 2)
        self.assertEqual(a["n_regulation_final"], 118)
        self.assertTrue(a["warnings"])

    def test_goal_training_excludes_extra_time(self):
        d = sample()
        before = IntlModel(FEATS).fit(d)
        changed = d.copy()
        changed.loc[changed.status.isin(["STATUS_FINAL_AET", "STATUS_FINAL_PEN"]), ["hs", "as_"]] = [30, 35]
        after = IntlModel(FEATS).fit(changed)
        import numpy as np
        np.testing.assert_allclose(
            before.predict(d.iloc[3:10])[["lambda_h", "lambda_a"]],
            after.predict(d.iloc[3:10])[["lambda_h", "lambda_a"]],
            atol=1e-10,
        )

    def test_empty_regulation_training_rejected(self):
        with self.assertRaisesRegex(ValueError, "regulation-time"):
            IntlModel(FEATS).fit(sample().iloc[:2])

    def test_duplicate_event_rejected(self):
        d = sample()
        with self.assertRaisesRegex(ValueError, "duplicated event_id"):
            audit_history(pd.concat([d, d.iloc[[0]]]))


if __name__ == "__main__":
    unittest.main()
