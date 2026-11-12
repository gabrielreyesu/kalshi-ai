import numpy as np
import pandas as pd
from kalshi_intl.validation import evaluate_oos


def sample():
    rng = np.random.default_rng(22)
    rows = []
    for year in (2021, 2022, 2023):
        for i in range(45):
            p = rng.dirichlet([2, 1.5, 2])
            rows.append(dict(date=f'{year}-09-{i%25+1:02d}', temporada=f'{year}/{(year+1)%100:02d}',
                             res=rng.choice(['H','D','A'],p=p), p_H=p[0],p_D=p[1],p_A=p[2]))
    return pd.DataFrame(rows)


def test_paired_only_calibrated():
    d, metrics, ci = evaluate_oos(sample(), min_train=30, bootstrap=20)
    assert d['calibrated'].all()
    assert len(d) == 90
    assert set(metrics.method) == {'original','calibrated'}
    assert set(ci.metric) == {'logloss','brier','accuracy_error'}
    assert np.isfinite(ci.ci95_low).all()


def test_no_training_returns_no_claim():
    d, metrics, ci = evaluate_oos(sample(), min_train=1000, bootstrap=0)
    assert d.empty and metrics.empty and ci.empty


def test_future_outcomes_cannot_change_earlier_calibration():
    a = sample()
    b = a.copy()
    b.loc[b.temporada == '2023/24', 'res'] = 'D'
    x = evaluate_oos(a, min_train=30, bootstrap=0)[0]
    z = evaluate_oos(b, min_train=30, bootstrap=0)[0]
    early = x.temporada == '2022/23'
    np.testing.assert_allclose(x.loc[early,['cal_H','cal_D','cal_A']], z.loc[early,['cal_H','cal_D','cal_A']])
