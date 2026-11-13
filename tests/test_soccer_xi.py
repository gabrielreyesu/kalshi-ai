import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd

from kalshi_lineup.soccer_eval import decide, run
from kalshi_lineup.soccer_xi import XIRater, parse_soccer_summary
from kalshi_intl.model import CONFIGS, FEATS, IntlEngine


def roster(team, n_start=11, extra=3, goals=None):
    goals = goals or {}
    r = [{"athlete": {"id": f"{team}-{i}"}, "starter": i < n_start,
          "stats": [{"name": "totalGoals", "value": goals.get(i, 0)}, {"name": "goalAssists", "value": 0}]} for i in range(n_start + extra)]
    return {"team": {"id": team}, "roster": r}


PAY = {"header": {"competitions": [{"status": {"type": {"completed": True}}}]}, "rosters": [roster("1", goals={9: 2}), roster("2")]}


def make_soccer(seasons=range(2016, 2026), teams=20, seed=4, signal=True):
    """Clubes sintéticos: a veces rotan a sus mejores titulares y marcan menos."""
    rng = np.random.default_rng(seed)
    att = rng.normal(0, 0.25, teams)
    rows, boxes, eid = [], {}, 0
    for y in seasons:
        for w in range(38):
            date = pd.Timestamp(f"{y}-08-10") + pd.Timedelta(days=7 * w)
            order = rng.permutation(teams)
            for k in range(teams // 2):
                h, a = int(order[2 * k]), int(order[2 * k + 1])
                eid += 1
                players, mult = [], {}
                for t in (h, a):
                    rot = int(rng.integers(0, 6)) if rng.random() < 0.3 else 0        # titulares habituales que descansan
                    xi = list(range(rot, 11)) + list(range(11, 11 + rot))
                    mult[t] = np.exp(-0.06 * rot) if signal else 1.0
                    for i in range(16):
                        players.append({"pid": f"{t}-{i}", "team_id": f"T{t}", "starter": i in xi,
                                        "ga": float(rng.poisson(0.25 if i < 11 else 0.1)) if i in xi else 0.0})
                lh = np.exp(0.25 + att[h] - att[a]) * mult[h] / mult[a] ** 0.5
                la = np.exp(0.0 + att[a] - att[h]) * mult[a] / mult[h] ** 0.5
                hs, as_ = int(rng.poisson(lh)), int(rng.poisson(la))
                kick = pd.Timestamp(date, tz="UTC")
                rows.append(dict(event_id=str(eid), code="uefa.champions" if w % 4 == 0 else "eng.1", kickoff_utc=kick.isoformat(),
                                 status="STATUS_FULL_TIME", home_id=f"T{h}", away_id=f"T{a}", home=f"T{h}", away=f"T{a}",
                                 hs=float(hs), as_=float(as_), neutral=False, final=True, kick=kick, date=date,
                                 target=w % 4 == 0, competicion="x", group="clubes"))
                boxes[str(eid)] = {"event_id": str(eid), "players": players}
    return pd.DataFrame(rows), boxes


class TestSoccerXI(unittest.TestCase):
    def test_parse(self):
        b = parse_soccer_summary(PAY, "5")
        self.assertEqual(sum(p["starter"] for p in b["players"]), 22)
        self.assertEqual([p["ga"] for p in b["players"] if p["pid"] == "1-9"], [2.0])
        bad = {**PAY, "rosters": [roster("1", n_start=10), roster("2")]}
        self.assertIsNone(parse_soccer_summary(bad, "5"))
        live = {**PAY, "header": {"competitions": [{"status": {"type": {"completed": False}}}]}}
        self.assertIsNone(parse_soccer_summary(live, "5"))

    def test_no_future_leak(self):
        g, b = make_soccer(seasons=range(2016, 2019))
        f1 = XIRater().process(g, b)
        cut = g["date"].quantile(0.5)
        late = set(g.loc[g["date"] >= cut, "event_id"])
        b2 = {e: ({**x, "players": [{**p, "ga": 9.0} for p in x["players"]]} if e in late else x) for e, x in b.items()}
        f2 = XIRater().process(g, b2)
        keep = set(g.loc[g["date"] <= cut, "event_id"])
        x = f1[f1.event_id.isin(keep)].set_index("event_id").sort_index()
        y = f2[f2.event_id.isin(keep)].set_index("event_id").sort_index()
        self.assertTrue(np.allclose(x[["xi_cont", "xi_ga"]], y[["xi_cont", "xi_ga"]]))

    def test_signal_recovered(self):
        g, b = make_soccer()
        f = IntlEngine(CONFIGS["clubes"]).process(g)
        f["event_id"] = f["event_id"].astype(str)
        f = f.merge(XIRater().process(g, b), on="event_id")
        res = decide(run(f, FEATS, first_year=2019), dev_last=2023)
        self.assertGreater(res["desarrollo"]["A_continuidad"]["mejora_logloss"]["mejora"], 0)


if __name__ == "__main__":
    unittest.main()
