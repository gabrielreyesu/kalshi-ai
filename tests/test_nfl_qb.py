import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd

from kalshi_lineup.evaluate import decide, run_variants
from kalshi_lineup.nfl_qb import QBRater, anya, parse_nfl_summary, regular_starter, starter_for
from kalshi_nba.engine import CONFIGS, FEATURES, NBAEngine

PAY = {
    "header": {"competitions": [{"status": {"type": {"name": "STATUS_FINAL"}}}]},
    "boxscore": {"players": [
        {"team": {"id": "1"}, "statistics": [
            {"name": "passing", "keys": ["completions/passingAttempts", "passingYards", "yardsPerPassAttempt",
                                         "passingTouchdowns", "interceptions", "sacks-sackYardsLost", "adjQBR", "QBRating"],
             "athletes": [{"athlete": {"id": "7"}, "stats": ["22/35", "260", "7.4", "2", "1", "3-21", "60", "95"]}]},
            {"name": "rushing", "keys": ["rushingAttempts"], "athletes": [{"athlete": {"id": "7"}, "stats": ["4"]}]}]},
        {"team": {"id": "2"}, "statistics": [
            {"name": "passing", "labels": ["C/ATT", "YDS", "AVG", "TD", "INT", "SACKS", "QBR", "RTG"],
             "athletes": [{"athlete": {"id": "9"}, "stats": ["10/20", "100", "5", "0", "2", "1-8", "20", "50"]}]}]},
    ]},
}


def make_nfl(seasons=range(2014, 2024), teams=16, seed=2, signal=True):
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 3, teams)
    qb_val = {t: 6.5 + rng.normal(0, 0.8) for t in range(teams)}       # titular
    games, boxes, eid = [], {}, 0
    for y in seasons:
        for w in range(17):
            date = pd.Timestamp(f"{y}-09-08") + pd.Timedelta(days=7 * w)
            order = rng.permutation(teams)
            for g in range(teams // 2):
                h, a = int(order[2 * g]), int(order[2 * g + 1])
                eid += 1
                qbs, eff = [], {}
                for t in (h, a):
                    backup = rng.random() < 0.15
                    pid, v = (f"{t}-b{y}", 4.0) if backup else (f"{t}-s", qb_val[t])
                    att = 32.0
                    yds = max(0.0, v * att + rng.normal(0, 40))
                    qbs.append({"pid": pid, "team_id": f"T{t}", "att": att, "cmp": 20.0, "yds": yds, "td": 0.0, "int": 0.0,
                                "sacks": 0.0, "sack_yds": 0.0})
                    eff[t] = v
                margin = 2 + strength[h] - strength[a] + (3.0 * (eff[h] - eff[a]) if signal else 0) + rng.normal(0, 13)
                hs, as_ = 24 + round(margin / 2), 24 - round(margin / 2)
                if hs == as_:
                    hs += 1
                games.append(dict(event_id=str(eid), date=date, kickoff_utc=date.isoformat(), season_year=y, season_type=2,
                                  status="STATUS_FINAL", home_id=f"T{h}", away_id=f"T{a}", home=f"T{h}", away=f"T{a}",
                                  hs=float(hs), as_=float(as_), neutral=False, line_spread=np.nan, line_total=np.nan, line_p_home=np.nan))
                boxes[str(eid)] = {"event_id": str(eid), "qbs": qbs}
    return pd.DataFrame(games), boxes


class TestNFLQB(unittest.TestCase):
    def test_parse(self):
        b = parse_nfl_summary(PAY, "1")
        q = {x["pid"]: x for x in b["qbs"]}
        self.assertEqual((q["7"]["cmp"], q["7"]["att"], q["7"]["yds"], q["7"]["sacks"], q["7"]["sack_yds"]), (22, 35, 260, 3, 21))
        self.assertEqual((q["9"]["att"], q["9"]["int"]), (20, 2))
        self.assertEqual(anya(q["7"]), (260 + 40 - 45 - 21, 38))

    def test_injured_in_game_starter_is_still_starter(self):
        qbs = [{"pid": "S", "att": 3}, {"pid": "B", "att": 30}]
        self.assertEqual(starter_for(qbs, "S"), "S")
        self.assertEqual(starter_for([{"pid": "B", "att": 30}], "S"), "B")
        self.assertEqual(starter_for([{"pid": "S", "att": 30}], "B", "S"), "S")     # vuelve el habitual

    def test_regular_starter_survives_one_backup_game(self):
        self.assertEqual(regular_starter(["S", "S", "S", "B"]), "S")
        self.assertEqual(regular_starter(["S", "B", "B"]), "B")

    def test_no_future_leak(self):
        g, b = make_nfl()
        f1 = QBRater().process(g, b)
        cut = g["date"].quantile(0.5)
        late = set(g.loc[g["date"] >= cut, "event_id"])
        b2 = {e: ({**x, "qbs": [{**q, "yds": 999.0, "td": 5.0} for q in x["qbs"]]} if e in late else x) for e, x in b.items()}
        f2 = QBRater().process(g, b2)
        keep = set(g.loc[g["date"] <= cut, "event_id"])
        a = f1[f1.event_id.isin(keep)].set_index("event_id").sort_index()
        c = f2[f2.event_id.isin(keep)].set_index("event_id").sort_index()
        self.assertTrue(np.allclose(a[["qb_delta", "qb_abs"]], c[["qb_delta", "qb_abs"]]))

    def test_signal_recovered(self):
        g, b = make_nfl(seasons=range(2010, 2024), teams=32)
        feats = NBAEngine(CONFIGS["nfl"]).process(g)
        feats["event_id"] = feats["event_id"].astype(str)
        feats = feats.merge(QBRater().process(g, b), on="event_id")
        v = {"A_qb_cambio": ["qb_delta"]}
        res = decide(run_variants(feats, FEATURES, first_test=2014, min_train=1000, variants=v), variants=v, dev_last=2023)
        self.assertGreater(res["desarrollo"]["A_qb_cambio"]["mejora_logloss"]["ic95_inf"], 0)


if __name__ == "__main__":
    unittest.main()
