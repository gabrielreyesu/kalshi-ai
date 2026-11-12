"""Uso:
  python -m kalshi_intl.cli backtest    # walk-forward (clubes y selecciones) contra cuotas de ESPN
  python -m kalshi_intl.cli predict     # próximos 14 días + precios de Kalshi + cuotas de casas; registro append-only
  python -m kalshi_intl.cli evaluate    # modelo vs Kalshi vs casas en partidos ya jugados
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_soccer.kalshi import _f, _get, _same
from . import MODEL_VERSION
from .data import GROUPS, load_group
from .model import CONFIGS, GROUP_FEATURES, IntlEngine, IntlModel, walk_forward
from .odds import attach, fetch

REG, SNAP = Path("predictions/intl_registry.csv"), Path("predictions/intl_kalshi_snapshots.csv")
KALSHI = {"clubes": ["KXUCLGAME", "KXUELGAME", "KXUECLGAME"],
          "selecciones": ["KXUEFANLGAME", "KXINTLFRIENDLYGAME", "KXWCGAME", "KXFIFAGAME"]}


def kalshi_events(series: list) -> list:
    """[{event, when, teams:[{name,bid,ask,last}], tie:{...}}] de los mercados abiertos de ganador/empate."""
    ev = {}
    for s in series:
        cur = None
        while True:
            p = {"series_ticker": s, "status": "open", "limit": 200}
            if cur:
                p["cursor"] = cur
            j = _get("/markets", p)
            for m in j.get("markets", []):
                e = ev.setdefault(m["event_ticker"], {"event": m["event_ticker"], "teams": [], "tie": None,
                                                      "when": pd.Timestamp(m.get("occurrence_datetime") or m["close_time"])})
                o = {"name": (m.get("yes_sub_title") or "").strip(), "bid": _f(m.get("yes_bid_dollars")),
                     "ask": _f(m.get("yes_ask_dollars")), "last": _f(m.get("last_price_dollars"))}
                if o["name"].lower() in ("tie", "draw"):
                    e["tie"] = o
                else:
                    e["teams"].append(o)
            cur = j.get("cursor")
            if not cur or not j.get("markets"):
                break
    return [e for e in ev.values() if len(e["teams"]) == 2 and e["tie"]]


def match_kalshi(home: str, away: str, kick, events: list) -> dict:
    """Precios H/D/A si exactamente UN evento coincide en hora (±36 h) y en ambos equipos; si no, vacío."""
    hits = []
    for e in events:
        if abs((e["when"] - kick).total_seconds()) > 36 * 3600:
            continue
        side = {}
        for o in e["teams"]:
            mh, ma = _same(o["name"], home), _same(o["name"], away)
            if mh != ma:
                side["H" if mh else "A"] = o
        if set(side) == {"H", "A"}:
            hits.append({**side, "D": e["tie"], "event": e["event"]})
    if len(hits) != 1:
        return {}
    h = hits[0]
    out = {"k_event": h["event"]}
    for c in "HDA":
        out[f"k_bid_{c}"], out[f"k_ask_{c}"] = h[c]["bid"], h[c]["ask"]
        out[f"k_mid_{c}"] = (h[c]["bid"] + h[c]["ask"]) / 2 if h[c]["bid"] is not None and h[c]["ask"] else h[c]["last"]
    return out


def _fit(grp):
    g = load_group(grp, verbose=False)
    f = IntlEngine(CONFIGS[grp]).process(g)
    return f, IntlModel(GROUP_FEATURES[grp]).fit(f)


def cmd_predict(days=14):
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for grp in GROUPS:
        f, model = _fit(grp)
        up = f[f["target"] & ~f["final"] & (f["kick"] > now) & (f["kick"] <= now + pd.Timedelta(days=days))]
        if up.empty:
            continue
        pr = model.predict(up)
        try:
            events = kalshi_events(KALSHI[grp])
        except Exception as ex:  # noqa: BLE001
            print("AVISO Kalshi:", ex); events = []
        for (_, r), (_, p) in zip(up.iterrows(), pr.iterrows()):
            row = {"event_id": r["event_id"], "grupo": grp, "competicion": r["competicion"], "code": r["code"],
                   "kickoff_utc": r["kick"].isoformat(), "home": r["home"], "away": r["away"], "neutral": int(r["neutral"]),
                   "p_H": p["p_H"], "p_D": p["p_D"], "p_A": p["p_A"], "p_over_2_5": p["p_over_2_5"], "p_btts": p["p_btts"],
                   "p_over_1_5": p["p_over_1_5"], "p_over_3_5": p["p_over_3_5"], "p_home_m1_5": p["p_home_m1_5"], "p_away_m1_5": p["p_away_m1_5"],
                   "lambda_h": p["lambda_h"], "lambda_a": p["lambda_a"], "elo_diff": r["elo_diff"],
                   "partidos_previos_local": r["n_home"], "partidos_previos_visita": r["n_away"],
                   "model_version": MODEL_VERSION, "created_utc": now.isoformat()}
            row.update(match_kalshi(r["home"], r["away"], r["kick"], events))
            try:
                row.update(fetch(r["code"], r["event_id"]))            # cuotas de casas actuales (sin caché)
            except Exception:  # noqa: BLE001
                pass
            rows.append(row)
    if not rows:
        print("No hay partidos próximos."); return
    d = pd.DataFrame(rows)
    cols = ["event_id", "grupo", "competicion", "code", "kickoff_utc", "home", "away", "neutral", "p_H", "p_D", "p_A", "p_over_2_5",
            "p_btts", "lambda_h", "lambda_a", "elo_diff", "partidos_previos_local", "partidos_previos_visita", "model_version",
            "created_utc", "k_event"] + [f"k_{a}_{c}" for a in ("bid", "ask", "mid") for c in "HDA"] + ["mkt_H", "mkt_D", "mkt_A", "mkt_over", "n_books"] + ["p_over_1_5", "p_over_3_5", "p_home_m1_5", "p_away_m1_5"]
    d = d.reindex(columns=cols)
    REG.parent.mkdir(parents=True, exist_ok=True)
    seen = set(pd.read_csv(REG, dtype={"event_id": str})["event_id"]) if REG.exists() else set()
    new = d[~d["event_id"].isin(seen)]
    from kalshi_nba.cli import _append          # conserva filas anteriores aunque lleguen columnas nuevas
    if len(new):
        _append(new, REG)
    _append(d.assign(snapshot_utc=now.isoformat()), SNAP)
    print(f"Próximos: {len(d)} | nuevos en registro: {len(new)} | con precio Kalshi: {int(d.k_event.notna().sum())} | con cuotas de casas: {int(d.mkt_H.notna().sum())}")
    print("   por competición:", d["competicion"].value_counts().to_dict())


def cmd_evaluate():
    if not REG.exists():
        print("Aún no hay registro."); return
    reg = pd.read_csv(REG, dtype={"event_id": str})
    fin = pd.concat([load_group(g, first_year=pd.Timestamp.now().year - 1, verbose=False) for g in GROUPS])
    fin = fin[fin["final"] & fin["status"].isin({"STATUS_FULL_TIME", "STATUS_FINAL"})][["event_id", "hs", "as_"]]
    d = reg.merge(fin, on="event_id")
    if d.empty:
        print("Aún no hay partidos registrados con resultado."); return
    i = np.where(d["hs"] > d["as_"], 0, np.where(d["hs"] == d["as_"], 1, 2))
    ll = lambda cols, x: -np.log(np.clip(x[cols].to_numpy(float)[np.arange(len(x)), i[x.index]], 1e-9, 1))
    d = d.reset_index(drop=True)
    out = [{"comparación": "modelo (todos)", "n": len(d), "logloss": round(float(ll(["p_H", "p_D", "p_A"], d).mean()), 4)}]
    for name, cols in (("Kalshi", ["k_mid_H", "k_mid_D", "k_mid_A"]), ("casas", ["mkt_H", "mkt_D", "mkt_A"])):
        s = d.dropna(subset=cols)
        if len(s):
            P = s[cols].to_numpy(float); s = s.copy(); s[cols] = P / P.sum(1, keepdims=True)
            out.append({"comparación": f"{name} (mismos {len(s)} partidos)", "n": len(s), "logloss": round(float(ll(cols, s).mean()), 4),
                        "logloss_modelo": round(float(ll(["p_H", "p_D", "p_A"], s).mean()), 4)})
    print(pd.DataFrame(out).to_string(index=False))


def cmd_backtest():
    from kalshi_nba.backtest import paired_bootstrap
    L = []
    for grp in GROUPS:
        f = IntlEngine(CONFIGS[grp]).process(load_group(grp))
        r = attach(walk_forward(f, features=GROUP_FEATURES[grp]))
        P = lambda x, pre: x[[f"{pre}_{c}" for c in "HDA"]].to_numpy(float)
        idx = lambda x: x["res"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        ll = lambda x, pre: -np.log(np.clip(P(x, pre)[np.arange(len(x)), idx(x)], 1e-12, 1))
        acc = lambda x, pre: 100 * float((P(x, pre).argmax(1) == idx(x)).mean())
        L.append(f"=== {grp.upper()} === partidos fuera de muestra: {len(r)} ({r.temporada.min()} a {r.temporada.max()})")
        L.append(f"1X2: modelo {acc(r, 'p'):.1f}% logloss {ll(r, 'p').mean():.4f} | frecuencias base {acc(r, 'base'):.1f}% {ll(r, 'base').mean():.4f}")
        s = r[r["mkt_H"].notna()] if "mkt_H" in r else r.iloc[:0]
        if len(s):
            L.append(f"Con cuotas ({len(s)}): modelo {acc(s, 'p'):.1f}% {ll(s, 'p').mean():.4f} | mercado {acc(s, 'mkt'):.1f}% {ll(s, 'mkt').mean():.4f}")
            L.append("  mejora del mercado sobre el modelo: " + str(paired_bootstrap(ll(s, "p"), ll(s, "mkt"), s["date"])))
        L.append("")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/intl_backtest_report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    {"predict": cmd_predict, "evaluate": cmd_evaluate, "backtest": cmd_backtest}[sys.argv[1] if len(sys.argv) > 1 else "predict"]()
