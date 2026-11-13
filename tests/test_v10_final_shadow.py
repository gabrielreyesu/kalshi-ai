import numpy as np
import pandas as pd

from kalshi_external.final_shadow import _final, build, score


def test_final_prefers_valid_market_and_falls_back_to_model():
    f, src = _final([0.6, 0.4], [0.7, 0.3]); assert src == "market_consensus" and np.allclose(f, [0.7, 0.3])
    f, src = _final([0.6, 0.4], [np.nan, np.nan]); assert src == "model_fallback" and np.allclose(f, [0.6, 0.4])
    f, src = _final([2, 1, 1], [0.5, 0.3, 0.3]); assert src == "market_consensus" and abs(f.sum() - 1) < 1e-12


def test_build_only_uses_pre_kickoff_snapshots_and_is_idempotent(tmp_path):
    pd.DataFrame([dict(event_id="1", kickoff_utc="2030-01-02T00:00:00+00:00", home="A", away="B"),
                  dict(event_id="2", kickoff_utc="2029-12-31T00:00:00+00:00", home="C", away="D")]).to_csv(tmp_path / "nba_registry.csv", index=False)
    pd.DataFrame([dict(snapshot_utc="2030-01-01T00:00:00+00:00", event_id="1", kind="winner", side="home", p_model=0.6, p_book=0.7),
                  dict(snapshot_utc="2030-01-01T00:00:00+00:00", event_id="1", kind="total", side="", p_model=0.5, p_book=0.5),
                  dict(snapshot_utc="2030-01-01T00:00:00+00:00", event_id="2", kind="winner", side="home", p_model=0.6, p_book=0.7)]).to_csv(tmp_path / "nba_kalshi_snapshots.csv", index=False)
    assert build(tmp_path) == 1 and build(tmp_path) == 1
    d = pd.read_csv(tmp_path / "final_shadow.csv", dtype={"event_id": str})
    assert list(d.event_id) == ["1"] and abs(d.final_H.iloc[0] - 0.7) < 1e-12 and d.final_source.iloc[0] == "market_consensus"


def test_score_pairs_same_games_uses_last_pre_kickoff_snapshot_and_ignores_late_ones():
    rows = []
    for k in range(200):
        ko = pd.Timestamp("2030-01-01", tz="UTC") + pd.Timedelta(days=k)
        base = dict(sport="nba", event_id=str(k), kickoff_utc=ko.isoformat(), home="A", away="B", model_D=np.nan, market_D=np.nan, final_D=np.nan, final_source="market_consensus")
        rows.append({**base, "snapshot_utc": (ko - pd.Timedelta(hours=30)).isoformat(), "model_H": 0.5, "model_A": 0.5, "market_H": 0.5, "market_A": 0.5, "final_H": 0.5, "final_A": 0.5})
        rows.append({**base, "snapshot_utc": (ko - pd.Timedelta(hours=3)).isoformat(), "model_H": 0.5, "model_A": 0.5, "market_H": 0.8, "market_A": 0.2, "final_H": 0.8, "final_A": 0.2})
        rows.append({**base, "snapshot_utc": (ko + pd.Timedelta(hours=1)).isoformat(), "model_H": 0.5, "model_A": 0.5, "market_H": 0.01, "market_A": 0.99, "final_H": 0.01, "final_A": 0.99})
    out = pd.DataFrame([dict(sport="nba", event_id=str(k), result="H" if k % 5 else "A") for k in range(200)])
    t = score(pd.DataFrame(rows), out, bootstrap=200).iloc[0]
    assert t.n == 200 and abs(t.acierto_final - 0.8) < 1e-9 and t.logloss_final < t.logloss_modelo and t.ic95_inf > 0


def test_simple_picks_table_shows_only_near_games_with_confidence(tmp_path):
    from kalshi_nba import resumen
    old = resumen.P; resumen.P = tmp_path
    try:
        base = dict(snapshot_utc="2030-01-01T00:00:00+00:00", model_H=0.5, model_D=float("nan"), model_A=0.5, market_H=0.8, market_D=float("nan"),
                    market_A=0.2, final_D=float("nan"), final_source="market_consensus")
        reg = dict(home_id="10", away_id="20", elo_diff=0, rest_h=1, rest_a=1, p_home=0.5, mu_margin=0.0, mu_total=220, p_home_normal=0.5,
                   sig_m=13, sig_t=17, model_version="nba_v1", created_utc="2029-12-31T00:00:00Z")
        pd.DataFrame([{**reg, "event_id": "1", "kickoff_utc": "2030-01-01T20:00:00+00:00", "home": "Utah Jazz", "away": "Orlando Magic"}]).to_csv(tmp_path / "nba_registry.csv", index=False)
        pd.DataFrame([{**reg, "event_id": "2", "kickoff_utc": "2030-01-09T20:00:00+00:00", "home": "Lions", "away": "Bears"}]).to_csv(tmp_path / "nfl_registry.csv", index=False)
        pd.DataFrame([{**base, "sport": "nba", "event_id": "1", "kickoff_utc": "2030-01-01T20:00:00+00:00", "home": "Utah Jazz", "away": "Orlando Magic", "final_H": 0.8, "final_A": 0.2},
                      {**base, "sport": "nfl", "event_id": "2", "kickoff_utc": "2030-01-09T20:00:00+00:00", "home": "Lions", "away": "Bears", "final_H": 0.6, "final_A": 0.4}]).to_csv(tmp_path / "final_shadow.csv", index=False)
        txt = resumen.build(now=pd.Timestamp("2030-01-01T12:00:00", tz="UTC").tz_convert(resumen.TZ))
    finally:
        resumen.P = old
    resumen_picks = txt.split("## Resumen de picks de hoy y mañana")[1]
    assert "| NBA | Orlando Magic en Utah Jazz | Ganador | Utah Jazz | 80% | Alta | casas |" in resumen_picks
    assert "Lions" not in resumen_picks                     # el partido lejano no entra en el resumen
    assert txt.index("## NBA") < txt.index("## Resumen de picks")   # las tablas detalladas van primero
