import json

import pandas as pd

from kalshi_external.archive import merge
from kalshi_external.core import load_registry
from kalshi_external.outcomes import outcomes_from_games


def test_outcomes_soccer_regulation_only_and_us_sports_no_ties():
    g = pd.DataFrame([
        dict(event_id="1", status="STATUS_FULL_TIME", hs=2.0, as_=1.0),
        dict(event_id="2", status="STATUS_FINAL_AET", hs=2.0, as_=1.0),     # prórroga: fuera
        dict(event_id="3", status="STATUS_FULL_TIME", hs=0.0, as_=0.0),
        dict(event_id="4", status="STATUS_SCHEDULED", hs=float("nan"), as_=float("nan")),
    ])
    o = outcomes_from_games(g, "intl").set_index("event_id")["result"].to_dict()
    assert o == {"1": "H", "3": "D"}
    u = pd.DataFrame([
        dict(event_id="7", status="STATUS_FINAL", hs=100.0, as_=110.0),
        dict(event_id="8", status="STATUS_FINAL", hs=20.0, as_=20.0),        # empate NFL: fuera
        dict(event_id="9", status="STATUS_IN_PROGRESS", hs=50.0, as_=40.0),
    ])
    assert outcomes_from_games(u, "nfl").set_index("event_id")["result"].to_dict() == {"7": "A"}
    assert outcomes_from_games(pd.DataFrame(), "nba").empty


def test_club_registry_is_loaded_separately(tmp_path):
    p = tmp_path / "intl_registry.csv"
    pd.DataFrame([
        dict(event_id="1", grupo="selecciones", kickoff_utc="2030-01-01T00:00Z", home="A", away="B", created_utc="2029-12-30T00:00Z"),
        dict(event_id="2", grupo="clubes", kickoff_utc="2030-01-01T00:00Z", home="C", away="D", created_utc="2029-12-30T00:00Z"),
    ]).to_csv(p, index=False)
    assert list(load_registry(p, "club").event_id) == ["2"]
    assert list(load_registry(p, "intl").event_id) == ["1"]


def test_archive_merge_is_idempotent_and_tolerates_corrupt_lines(tmp_path):
    a, b, out = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "o" / "h.jsonl"
    r1 = dict(source_id="s", sport="nba", event_id="1", captured_utc="2026-11-12T10:00:00+00:00", p_H=0.6)
    r2 = dict(source_id="s", sport="nba", event_id="1", captured_utc="2026-11-12T22:00:00+00:00", p_H=0.62)
    a.write_text(json.dumps(r1) + "\n{corrupto\n", encoding="utf8")
    b.write_text(json.dumps(r1) + "\n" + json.dumps(r2) + "\n", encoding="utf8")
    assert merge([a, b], out) == 2
    assert merge([out, b], out) == 2                      # repetir no duplica
    assert merge([tmp_path / "no_existe.jsonl", out], out) == 2
