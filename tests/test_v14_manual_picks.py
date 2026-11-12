import pandas as pd

from kalshi_external.manual_picks import find_event, parse_row, settle, sync

KO = pd.Timestamp("2030-01-02T01:00:00Z")


def fx(_sport):
    return pd.DataFrame([dict(event_id="9", kick=KO, home="Utah Jazz", away="Orlando Magic", hs=float("nan"), as_=float("nan"), final=False, regulation=True)])


def fx_final(_sport):
    return fx(_sport).assign(hs=110.0, as_=115.0, final=True)


ROW = dict(analista="Ana", deporte="NBA", local="Utah", visitante="Orlando", fecha_partido="2030-01-02", mercado="spread", seleccion="visitante", linea="-3,5", cuota="1.91", link="")


def test_parse_row_normalizes_and_rejects():
    p, why = parse_row(ROW); assert why == "" and p["seleccion"] == "away" and p["linea"] == -3.5 and p["cuota"] == 1.91
    assert parse_row({**ROW, "seleccion": "Orlando Magic"})[0]["seleccion"] == "away"
    assert "línea" in parse_row({**ROW, "linea": ""})[1]
    assert "selección" in parse_row({**ROW, "mercado": "ganador", "seleccion": "empate"})[1]          # no hay empate en NBA
    assert "deporte" in parse_row({**ROW, "deporte": "tenis"})[1]
    assert "cuota" in parse_row({**ROW, "cuota": "-110"})[1]


def test_find_event_needs_unique_match_and_detects_swapped_teams():
    assert find_event(fx(""), "Utah", "Orlando", "2030-01-01")[0]["event_id"] == "9"       # ±1 día (hora local vs UTC)
    assert "al revés" in find_event(fx(""), "Orlando", "Utah", "2030-01-02")[1]
    assert find_event(fx(""), "Utah", "Orlando", "2030-02-02")[0] is None


def test_settle_all_markets():
    assert settle("ganador", "away", None, 110, 115, "nba", True) == "win"
    assert settle("spread", "away", -3.5, 110, 115, "nba", True) == "win" and settle("spread", "away", -5, 110, 115, "nba", True) == "push"
    assert settle("spread", "home", 3.5, 110, 115, "nba", True) == "loss"
    assert settle("total", "over", 224.5, 110, 115, "nba", True) == "win" and settle("total", "under", 225, 110, 115, "nba", True) == "push"
    assert settle("ganador", "draw", None, 1, 1, "futbol", True) == "win" and settle("ambos_anotan", "yes", None, 1, 0, "futbol", True) == "loss"
    assert settle("ganador", "home", None, 2, 1, "futbol", False) == "sin_liquidar"          # prórroga: no hay marcador a 90'


def test_sync_registers_once_rejects_late_and_settles(tmp_path):
    src, led, rej, st = (tmp_path / n for n in ("mis.csv", "libro.csv", "rech.csv", "ESTADO.md"))
    pd.DataFrame([ROW, {**ROW, "analista": "Beto", "local": "Lakers"}]).to_csv(src, index=False)
    r = sync(src, led, rej, st, now="2030-01-01T12:00:00Z", fixtures_fn=fx)
    assert (r["nuevos"], r["rechazados"], r["pendientes"]) == (1, 1, 1)
    assert sync(src, led, rej, st, now="2030-01-01T13:00:00Z", fixtures_fn=fx)["nuevos"] == 0          # no duplica
    pd.DataFrame([ROW, {**ROW, "analista": "Tarde"}]).to_csv(src, index=False)
    r = sync(src, led, rej, st, now="2030-01-02T00:50:00Z", fixtures_fn=fx)                          # 10 min antes: tarde
    assert r["nuevos"] == 0 and "demasiado tarde" in rej.read_text()
    r = sync(src, led, rej, st, now="2030-01-03T00:00:00Z", fixtures_fn=fx_final)
    d = pd.read_csv(led, dtype=str)
    assert list(d.resultado) == ["win"] and d.marcador.iloc[0] == "110-115" and d.captured_utc.iloc[0].startswith("2030-01-01T12:00")
    assert "| Ana | nba | spread | 1 | 0 | 0 | 0 | 100% |" in st.read_text()
