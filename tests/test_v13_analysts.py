import json

import pandas as pd

from kalshi_external.analysts import audit_watchlist, ingest, metrics, validate_pick, wilson

GOOD = dict(analyst_id="a", sport="nfl", market="ats", event_id="1", selection="home", line="-3.5", odds_decimal="1.91", probability="",
            published_utc="2030-01-01T10:00:00Z", captured_utc="2030-01-01T11:00:00Z", kickoff_utc="2030-01-02T18:00:00Z",
            source_url="https://x", authorized_for_modeling="yes")


def test_wilson_interval():
    lo, hi = wilson(53, 71)
    assert 0.63 < lo < 0.64 and 0.83 < hi < 0.84
    assert wilson(0, 0)[0] != wilson(0, 0)[0]            # NaN sin muestra


def test_validate_pick_enforces_authorization_time_and_market():
    assert validate_pick(GOOD) == []
    assert any("autorización" in w for w in validate_pick({**GOOD, "authorized_for_modeling": "no"}))
    assert any("sin margen" in w for w in validate_pick({**GOOD, "captured_utc": "2030-01-02T17:55:00Z"}))
    assert any("publicado después" in w for w in validate_pick({**GOOD, "published_utc": "2030-01-01T12:00:00Z"}))
    assert any("mercado no reconocido" in w for w in validate_pick({**GOOD, "market": "1x2_90"}))
    assert any("falta kickoff_utc" in w for w in validate_pick({**GOOD, "kickoff_utc": ""}))
    assert any("odds_decimal" in w for w in validate_pick({**GOOD, "odds_decimal": "0.5"}))


def test_ingest_is_append_only_and_rejects_duplicates(tmp_path):
    src, led = tmp_path / "p.csv", tmp_path / "d" / "picks.jsonl"
    pd.DataFrame([GOOD, {**GOOD, "event_id": "2", "authorized_for_modeling": ""}]).to_csv(src, index=False)
    r = ingest(src, led)
    assert (r["aceptados"], r["rechazados"]) == (1, 1)
    assert ingest(src, led)["aceptados"] == 0             # repetir no duplica
    assert len(led.read_text().splitlines()) == 1 and json.loads(led.read_text().splitlines()[0])["market"] == "ats"


def test_metrics_profit_ci_and_market_edge():
    rows = [{**GOOD, "event_id": str(i)} for i in range(10)]
    res = pd.DataFrame([dict(analyst_id="a", sport="nfl", market="ats", event_id=str(i), selection="home",
                             outcome="win" if i < 6 else ("push" if i == 9 else "loss"), market_prob="0.5") for i in range(10)])
    m = metrics(rows, res).iloc[0]
    assert (m.n, m.pushes) == (9, 1) and abs(m.win_rate - 6 / 9) < 1e-12
    assert abs(m.profit_units - (6 * 0.91 - 3)) < 1e-9 and abs(m.ventaja_vs_mercado - (6 / 9 - 0.5)) < 1e-12
    assert not m.muestra_suficiente and "brier" not in m.index          # sin probabilidades no hay Brier


def test_watchlist_audit_flags_small_samples():
    t = audit_watchlist("analysts/watchlist.csv").set_index(["analista", "mercado"])
    assert len(t) == 26 and not t["autorizado_para_modelos"].any()
    assert t.loc[("James McKinney", "SU"), "veredicto"] == "no distinguible de la referencia"
    assert t.loc[("Sean Koerner", "NFL tracked bets"), "veredicto"] == "supera la referencia con muestra suficiente"
    assert t.loc[("Micah Roberts", "NFL picks"), "veredicto"] == "supera la referencia, pero muestra pequeña"
