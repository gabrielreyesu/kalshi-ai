import numpy as np
import pandas as pd
import pytest
from kalshi_intl.calibration import probabilities, scores, temporal_calibrate, reliability


def sample():
    rng = np.random.default_rng(42)
    rows = []
    for year in (2021, 2022, 2023):
        for i in range(90):
            p = rng.dirichlet([3, 2, 2])
            y = rng.choice(['H', 'D', 'A'], p=p)
            rows.append(dict(temporada=f'{year}/{str(year+1)[-2:]}', date=pd.Timestamp(f'{year}-09-01') + pd.Timedelta(days=i),
                             p_H=p[0], p_D=p[1], p_A=p[2], res=y))
    return pd.DataFrame(rows)


def test_no_lookahead_future_results():
    d = sample()
    first = temporal_calibrate(d, min_train=60)
    changed = d.copy()
    changed.loc[changed.temporada == '2023/24', 'res'] = 'A'
    second = temporal_calibrate(changed, min_train=60)
    np.testing.assert_allclose(first[['cal_H','cal_D','cal_A']], second[['cal_H','cal_D','cal_A']])
    assert not first.loc[first.temporada == '2021/22', 'calibrated'].any()
    assert first.loc[first.temporada == '2022/23', 'calibrated'].all()


def test_probability_normalization_and_scores():
    d = sample()
    c = temporal_calibrate(d, min_train=60)
    np.testing.assert_allclose(probabilities(c, ['cal_H','cal_D','cal_A']).sum(axis=1), 1)
    assert np.isfinite(scores(probabilities(c), c.res)['logloss'])
    assert not reliability(c).empty


def test_invalid_probabilities():
    d = sample()
    d.loc[0, 'p_H'] = -1
    with pytest.raises(ValueError):
        probabilities(d)
