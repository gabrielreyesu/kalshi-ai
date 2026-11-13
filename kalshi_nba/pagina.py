"""Página web de predicciones pensada para el teléfono: docs/index.html (un solo archivo, sin dependencias).

Lee los mismos CSV de predictions/ que PREDICCIONES.md y no usa internet. Se publica con GitHub Pages
(Settings → Pages → rama main, carpeta /docs).   Uso:  python -m kalshi_nba.pagina
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .injuries import OUT
from .picks import soccer_picks, us_picks, us_snapshot
from .resumen import TZ, _bajas_fut, _lat, _when

P = Path("predictions")
SPORT = {"nba": "NBA", "nfl": "NFL", "liga": "Ligas", "club": "Copas", "intl": "Selecciones"}


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, 4) if math.isfinite(v) else None


def _latest(path: Path, key="event_id") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_csv(path, dtype={key: str})
    return d.sort_values("snapshot_utc").drop_duplicates(key, keep="last").set_index(key) if "snapshot_utc" in d else d


def _final_picks() -> dict:
    f = P / "final_shadow.csv"
    if not f.exists():
        return {}
    d = pd.read_csv(f, dtype={"event_id": str}).sort_values("snapshot_utc").drop_duplicates(["sport", "event_id"], keep="last")
    return {(r.sport, r.event_id): r for r in d.itertuples()}


def _injured(lg: str) -> dict:
    """{team_id: [nombres fuera/dudosos]} de la foto más reciente."""
    path = P / f"{lg}_injuries.csv"
    if not path.exists():
        return {}
    d = pd.read_csv(path, dtype={"team_id": str})
    d = d[(d["day"] == d["day"].max()) & d["status"].str.lower().isin(OUT)]
    return {t: [f"{r.player} ({str(r.status).lower()})" for r in g.itertuples()] for t, g in d.groupby("team_id")}


def _pick(names, probs):
    ok = [(n, p) for n, p in zip(names, probs) if p is not None]
    return max(ok, key=lambda z: z[1]) if ok else (None, None)


def games_us(lg: str, now, finals) -> list:
    reg = P / f"{lg}_registry.csv"
    if not reg.exists():
        return []
    r = pd.read_csv(reg, dtype={"event_id": str, "home_id": str, "away_id": str})
    r = (r.sort_values("created_utc") if "created_utc" in r else r).drop_duplicates("event_id", keep="last")
    r["t"] = _when(r["kickoff_utc"])
    r = r[r["t"] > now].sort_values("t")
    s = pd.DataFrame()
    snap = P / f"{lg}_kalshi_snapshots.csv"
    if snap.exists():
        s = pd.read_csv(snap, dtype={"event_id": str})
        s = s[(s["kind"] == "winner") & (s["side"] == "home")].sort_values("snapshot_utc").drop_duplicates("event_id", keep="last").set_index("event_id")
    pm = _latest(P / f"{lg}_polymarket_snapshots.csv")
    inj = _injured(lg)
    lines = us_snapshot(P, lg)
    lines = {e: g for e, g in lines.groupby("event_id")} if len(lines) else {}
    out = []
    for x in r.itertuples():
        kh = _f(s["mid"].get(x.event_id)) if "mid" in s else None
        bh = _f(s["p_book"].get(x.event_id)) if "p_book" in s else None
        ph = _f(pm["pm_home"].get(x.event_id)) if "pm_home" in pm else None
        model = [_f(x.p_home), _f(1 - x.p_home)]
        fin = finals.get((lg, x.event_id))
        if fin is not None and pd.notna(fin.final_H):
            final, src = [_f(fin.final_H), _f(fin.final_A)], ("casas" if fin.final_source == "market_consensus" else "modelo")
        else:
            final, src = model, "modelo"
        pick, prob = _pick([x.home, x.away], final)
        out.append({"sport": lg, "comp": SPORT[lg], "t": x.t.isoformat(), "home": x.home, "away": x.away, "three": False,
                    "pick": pick, "prob": prob, "src": src,
                    "rows": [["Modelo", model], ["Casas", [bh, None if bh is None else 1 - bh]],
                             ["Kalshi", [kh, None if kh is None else 1 - kh]], ["Polymarket", [ph, None if ph is None else 1 - ph]]],
                    "extra": [["Diferencia esperada", f"{x.home if x.mu_margin >= 0 else x.away} por {abs(x.mu_margin):.1f} pts"],
                              ["Total esperado", f"{x.mu_total:.0f} pts"]],
                    "inj": {"home": inj.get(str(getattr(x, "home_id", "")), []), "away": inj.get(str(getattr(x, "away_id", "")), [])},
                    "picks": us_picks(lines.get(x.event_id), x.home, x.away, (pick, prob, src))})
    return out


def _tri(x, cols):
    return [_f(getattr(x, c, None)) for c in cols]


def _norm(v):
    if any(p is None for p in v):
        return v if any(p is not None for p in v) else [None, None, None]
    s = sum(v)
    return [round(p / s, 4) for p in v] if s > 0 else v


def games_ligas(now) -> list:
    reg = P / "registry.csv"
    if not reg.exists():
        return []
    r = pd.read_csv(reg)
    r = r.drop_duplicates("match_id", keep="last") if "match_id" in r else r
    r["t"] = _when(r["kickoff_utc"])
    r = r[r["t"] > now].sort_values("t")
    lat = _lat()
    out = []
    for x in r.itertuples():
        model = _tri(x, ["p_H", "p_D", "p_A"])
        kal = []
        for s in "HDA":
            b, a = _f(getattr(x, f"k_bid_{s}", None)), _f(getattr(x, f"k_ask_{s}", None))
            kal.append(None if b is None or a is None else round((b + a) / 2, 4))
        pick, prob = _pick([x.home, "Empate", x.away], model)
        out.append({"sport": "liga", "comp": x.competition, "t": x.t.isoformat(), "home": x.home, "away": x.away, "three": True,
                    "pick": pick, "prob": prob, "src": "modelo",
                    "rows": [["Modelo", model], ["Kalshi", kal]],
                    "extra": _soccer_extra(x, cal=True) + [["Bajas local / visita", _bajas_fut(x, lat)]], "inj": None,
                    "picks": soccer_picks(x.home, x.away, model, "modelo",
                                          {1.5: _f(getattr(x, "p_over_1_5_cal", None)), 2.5: _f(getattr(x, "p_over_2_5_cal", None))},
                                          _f(getattr(x, "p_btts_cal", None)), kalshi3=kal)})
    return out


def _soccer_extra(x, cal):
    g = lambda c: _f(getattr(x, c, None))
    sfx = "_cal" if cal else ""
    pc = lambda v: "—" if v is None else f"{100 * v:.0f}%"
    return [["Más de 1.5 / 2.5 / 3.5 goles", " / ".join(pc(g(f"p_over_{k}{sfx}")) for k in ("1_5", "2_5", "3_5"))],
            ["Ambos anotan", pc(g(f"p_btts{sfx}"))],
            ["Gana por 2+ local / visita", f"{pc(g(f'p_home_m1_5{sfx}'))} / {pc(g(f'p_away_m1_5{sfx}'))}"]]


def games_intl(now, finals) -> list:
    reg = P / "intl_registry.csv"
    if not reg.exists():
        return []
    r = pd.read_csv(reg, dtype={"event_id": str})
    r = (r.sort_values("created_utc") if "created_utc" in r else r).drop_duplicates("event_id", keep="last")
    snap = P / "intl_kalshi_snapshots.csv"
    if snap.exists():
        s = _latest(snap)
        for c in [c for c in s.columns if c.startswith(("k_mid_", "mkt_", "p_over_1_5", "p_over_3_5", "p_home_m1_5", "p_away_m1_5"))]:
            r[c] = r["event_id"].map(s[c]).combine_first(r[c] if c in r else pd.Series(np.nan, index=r.index))
    r["t"] = _when(r["kickoff_utc"])
    r = r[r["t"] > now].sort_values("t")
    lat = _lat()
    out = []
    for x in r.itertuples():
        sport = "intl" if x.grupo == "selecciones" else "club"
        model, books = _tri(x, ["p_H", "p_D", "p_A"]), _norm(_tri(x, ["mkt_H", "mkt_D", "mkt_A"]))
        fin = finals.get((sport, x.event_id))
        if fin is not None and pd.notna(fin.final_H):
            final, src = [_f(fin.final_H), _f(fin.final_D), _f(fin.final_A)], ("casas" if fin.final_source == "market_consensus" else "modelo")
        else:
            final, src = model, "modelo"
        pick, prob = _pick([x.home, "Empate", x.away], final)
        extra = _soccer_extra(x, cal=False) + [["Bajas local / visita", _bajas_fut(x, lat)]]
        if sport == "intl":
            extra.append(["Aviso", "En selecciones, los mercados de goles del modelo son poco fiables."])
        out.append({"sport": sport, "comp": x.competicion + (" · sede neutral" if x.neutral else ""), "t": x.t.isoformat(),
                    "home": x.home, "away": x.away, "three": True, "pick": pick, "prob": prob, "src": src,
                    "rows": [["Modelo", model], ["Casas", books], ["Kalshi", _tri(x, ["k_mid_H", "k_mid_D", "k_mid_A"])]],
                    "extra": extra, "inj": None,
                    "picks": soccer_picks(x.home, x.away, final, src,
                                          {1.5: _f(getattr(x, "p_over_1_5", None)), 2.5: _f(getattr(x, "p_over_2_5", None))},
                                          _f(getattr(x, "p_btts", None)), goals_ok=sport != "intl",
                                          kalshi3=_tri(x, ["k_mid_H", "k_mid_D", "k_mid_A"]))})
    return out


def collect(now=None) -> dict:
    global P
    from . import resumen as _r
    P = _r.P                                   # misma carpeta que PREDICCIONES.md (las pruebas la cambian)
    now = now or pd.Timestamp.now(tz=TZ)
    finals = _final_picks()
    games = games_us("nba", now, finals) + games_us("nfl", now, finals) + games_ligas(now) + games_intl(now, finals)
    games.sort(key=lambda g: g["t"])
    return {"updated": now.isoformat(), "tz": TZ, "games": games}


def render(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    return TEMPLATE.replace("__DATA__", payload).replace("__UPDATED__", html.escape(pd.Timestamp(data["updated"]).strftime("%d %b %Y %H:%M")))


TEMPLATE = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="robots" content="noindex">
<meta name="theme-color" content="#0f172a">
<title>Kalshi AI · Predicciones</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#0f172a;--muted:#5b6474;--line:#e3e6eb;--accent:#2563eb;--home:#2563eb;--draw:#9aa3af;--away:#e11d48;
--hi:#15803d;--mid:#b45309;--lo:#6b7280;--chip:#eef1f5;--onaccent:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#0b1020;--card:#141a2e;--ink:#e8ecf4;--muted:#9aa4b8;--line:#232b45;--accent:#60a5fa;--home:#60a5fa;
--draw:#6b7280;--away:#fb7185;--hi:#4ade80;--mid:#fbbf24;--lo:#9ca3af;--chip:#1c2440;--onaccent:#0b1020}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:12px 16px 4px}.sticky{position:sticky;top:0;z-index:5;background:var(--bg);padding:6px 16px 8px;border-bottom:1px solid var(--line)}
h1{font-size:18px;margin:0}.sub{color:var(--muted);font-size:12px;margin-top:2px}
.bar{display:flex;gap:6px;overflow-x:auto;margin-top:6px;scrollbar-width:none}.bar::-webkit-scrollbar{display:none}
button.chip{flex:0 0 auto;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:999px;padding:6px 11px;font:inherit;font-size:13px}
button.chip[aria-pressed=true]{background:var(--accent);border-color:var(--accent);color:var(--onaccent);font-weight:600}
input[type=search]{width:100%;margin-top:8px;padding:9px 12px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--ink);font:inherit}
main{padding:12px 16px 40px;max-width:720px;margin:0 auto}
.day{font-weight:700;font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin:18px 2px 8px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:12px 14px;margin-bottom:10px}
.top{display:flex;justify-content:space-between;gap:8px;font-size:12px;color:var(--muted)}
.teams{font-weight:650;font-size:16px;margin:6px 0 8px}.teams small{font-weight:400;color:var(--muted)}
.pick{display:flex;align-items:center;justify-content:space-between;gap:8px;background:var(--chip);border-radius:10px;padding:8px 10px}
.pick b{font-size:16px}.pct{font-size:20px;font-weight:750;font-variant-numeric:tabular-nums}
.conf{font-size:11px;font-weight:700;border-radius:999px;padding:2px 8px;border:1px solid currentColor}
.c-Alta{color:var(--hi)}.c-Media{color:var(--mid)}.c-Baja{color:var(--lo)}
.src{font-size:11px;color:var(--muted)}.warn{display:inline-block;margin-top:6px;font-size:11px;color:var(--mid)}.count{color:var(--muted);font-size:12px;margin:10px 2px 0}
details{margin-top:8px}summary{cursor:pointer;color:var(--accent);font-size:13px;list-style:none}summary::-webkit-details-marker{display:none}
.row{display:grid;grid-template-columns:78px 1fr;gap:8px;align-items:center;margin:7px 0;font-size:12px}
.stack{display:flex;height:20px;border-radius:6px;overflow:hidden;background:var(--chip);font-size:11px;font-weight:650;color:#fff}
.stack span{display:flex;align-items:center;justify-content:center;min-width:0;white-space:nowrap;overflow:hidden}
.h{background:var(--home)}.d{background:var(--draw)}.a{background:var(--away)}.na{color:var(--muted);font-size:12px}
.legend{display:flex;gap:12px;font-size:11px;color:var(--muted);margin-top:4px}.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px}
dl{display:grid;grid-template-columns:auto 1fr;gap:4px 10px;font-size:13px;margin:8px 0 0}dt{color:var(--muted)}dd{margin:0;text-align:right}
.inj{font-size:12px;color:var(--muted);margin-top:6px}
.mk{margin-top:8px;border-top:1px solid var(--line)}.mk>div{display:grid;grid-template-columns:1fr auto;gap:2px 10px;padding:7px 0;border-bottom:1px solid var(--line)}
.mk>div:last-child{border-bottom:0}.mk .r{text-align:right}.mk .lab{font-size:11px;color:var(--muted)}.mk .ch{font-size:14px;font-weight:600}
.mk .pp{text-align:right;font-weight:700;font-variant-numeric:tabular-nums}.mk .meta{text-align:right;font-size:11px;color:var(--muted)}
.view{display:flex;background:var(--chip);border-radius:10px;padding:3px;margin-top:8px}
.view button{flex:1;border:0;background:transparent;color:var(--ink);font:inherit;font-size:14px;padding:7px;border-radius:8px}
.view button[aria-pressed=true]{background:var(--card);font-weight:650;box-shadow:0 1px 2px rgba(0,0,0,.15)}
.pk{display:grid;grid-template-columns:56px 1fr;gap:10px;align-items:center}.pk .big{font-size:20px;font-weight:750;font-variant-numeric:tabular-nums;text-align:center}
.pk .t1{font-weight:650}.pk .t2{font-size:12px;color:var(--muted)}.empty{color:var(--muted);text-align:center;padding:40px 0}
footer{color:var(--muted);font-size:12px;padding:0 16px 30px;max-width:720px;margin:0 auto}
</style>
</head>
<body>
<header>
  <h1>Predicciones</h1>
  <div class="sub">Actualizado __UPDATED__ (Chicago) · horas en tu zona horaria</div>
  <div class="view" id="view"><button data-v="games">Partidos</button><button data-v="picks">Mejores picks</button></div>
  <input type="search" id="q" placeholder="Buscar equipo o competición" aria-label="Buscar">
</header>
<div class="sticky">
  <div class="bar" id="sports"></div>
  <div class="bar" id="when"></div>
  <div class="bar" id="mkts"></div>
</div>
<main id="list"></main>
<footer>
  <p><b>Pick</b>: resultado más probable. Usa el consenso de casas (sin margen) cuando existe y el modelo propio si no; está en prueba.
  <b>Confianza</b>: Alta ≥ 70%, Media 55–69%, Baja &lt; 55%. Kalshi y Polymarket: precio ≈ probabilidad (62¢ ≈ 62%).</p>
  <p><b>Mercados</b>: Hándicap = ganar por más de X puntos (o "+X": ganar o perder por menos de X). Total = puntos o goles del partido.
  "Línea segura" = la línea más exigente que aún tiene ≥ 70–75% de probabilidad. Doble oportunidad = uno de dos resultados.
  En hándicap y totales la probabilidad viene de las casas: el modelo propio no mostró ventaja ahí.</p>
  <p>Bajas: informativas; el modelo de producción todavía no las usa. Esto no es una recomendación de apuesta.</p>
</footer>
<script>
const DATA = __DATA__;
const S = {sport: "all", when: "48h", q: "", hi: "0", view: "games", mkt: "all"};
const MKTS = [["all","Todos los mercados"],["Ganador","Ganador"],["Doble","Doble oportunidad"],["Hándicap","Hándicap"],["Total","Totales"],["Goles","Goles"],["Ambos","Ambos anotan"]];
const mktOk = p => S.mkt === "all" || (S.mkt === "Total" ? /^(Total|Puntos)/.test(p.mercado) : p.mercado.startsWith(S.mkt));
const SPORTS = [["all","Todo"],["nba","NBA"],["nfl","NFL"],["liga","Ligas"],["club","Copas"],["intl","Selecciones"]];
const WHEN = [["24h","Hoy"],["48h","Hoy y mañana"],["7d","7 días"],["all","Todo"]];
const HI = [["0","Toda confianza"],["1","Solo Alta"]];
try { const s = JSON.parse(localStorage.getItem("kai-filters") || "{}"); Object.assign(S, s, {q: ""}); } catch (e) {}
const pc = v => v == null ? "—" : Math.round(100 * v) + "%";
const level = v => v >= .70 ? "Alta" : v >= .55 ? "Media" : "Baja";
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function chips(id, items, key) {
  const el = document.getElementById(id);
  el.innerHTML = items.map(([v, l]) => `<button class="chip" data-v="${v}" aria-pressed="${S[key] === v}">${l}</button>`).join("");
  el.onclick = e => { const b = e.target.closest("button"); if (!b) return; S[key] = b.dataset.v;
    try { localStorage.setItem("kai-filters", JSON.stringify({sport: S.sport, when: S.when, hi: S.hi, view: S.view, mkt: S.mkt})); } catch (e) {}
    chips(id, items, key); draw(); };
}
function stack(v, three) {
  if (!v || v.every(x => x == null)) return `<span class="na">sin dato</span>`;
  const parts = three ? [["h", v[0]], ["d", v[1]], ["a", v[2]]] : [["h", v[0]], ["a", v[1]]];
  return `<div class="stack">${parts.map(([c, p]) => p == null ? "" : `<span class="${c}" style="width:${(100 * p).toFixed(1)}%">${p >= .12 ? pc(p) : ""}</span>`).join("")}</div>`;
}
function card(g) {
  const t = new Date(g.t);
  const hour = t.toLocaleTimeString("es", {hour: "2-digit", minute: "2-digit"});
  const lv = g.prob == null ? "Baja" : level(g.prob);
  const vs = g.sport === "nba" || g.sport === "nfl" ? `${esc(g.away)} <small>en</small> ${esc(g.home)}` : `${esc(g.home)} <small>vs</small> ${esc(g.away)}`;
  const legend = `<div class="legend"><span><i class="h"></i>${esc(g.home)}</span>${g.three ? '<span><i class="d"></i>Empate</span>' : ""}<span><i class="a"></i>${esc(g.away)}</span></div>`;
  const inj = g.inj && (g.inj.home.length || g.inj.away.length)
    ? `<div class="inj"><b>Bajas ${esc(g.home)}:</b> ${esc(g.inj.home.join(", ") || "ninguna")}<br><b>Bajas ${esc(g.away)}:</b> ${esc(g.inj.away.join(", ") || "ninguna")}</div>` : "";
  return `<article class="card">
    <div class="top"><span>${esc(g.comp)}</span><span>${hour}</span></div>
    <div class="teams">${vs}</div>
    <div class="pick"><div><div class="src">Pick (${esc(g.src)})</div><b>${esc(g.pick ?? "—")}</b></div>
      <div style="text-align:right"><div class="pct">${pc(g.prob)}</div><span class="conf c-${lv}">${lv}</span></div></div>
    ${gap(g)}
    ${mk(g)}
    <details><summary>Ver comparación y detalles</summary>
      ${g.rows.map(([n, v]) => `<div class="row"><span>${esc(n)}</span>${stack(v, g.three)}</div>`).join("")}${legend}
      <dl>${g.extra.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>${inj}
    </details></article>`;
}
const money = v => v == null ? "" : ` · Kalshi ${Math.round(100 * v)}¢`;
function mk(g) {
  const extra = (g.picks || []).filter(p => p.mercado !== "Ganador");
  if (!extra.length) return "";
  return `<div class="mk">${extra.map(p => `<div><span><span class="lab">${esc(p.mercado)}</span><br><span class="ch">${esc(p.pick)}</span></span>
    <span class="r"><span class="pp">${pc(p.prob)}</span><br><span class="meta">${level(p.prob ?? 0)} · ${esc(p.fuente)}${money(p.precio)}</span></span></div>`).join("")}</div>`;
}
function pickRow(g, p) {
  const t = new Date(g.t), lv = level(p.prob ?? 0);
  const when = t.toLocaleDateString("es", {weekday: "short", day: "numeric"}) + " " + t.toLocaleTimeString("es", {hour: "2-digit", minute: "2-digit"});
  const vs = g.sport === "nba" || g.sport === "nfl" ? `${esc(g.away)} en ${esc(g.home)}` : `${esc(g.home)} vs ${esc(g.away)}`;
  return `<article class="card pk"><div><div class="big">${pc(p.prob)}</div><div style="text-align:center"><span class="conf c-${lv}">${lv}</span></div></div>
    <div><div class="t2">${esc(p.mercado)} · ${esc(g.comp)}</div><div class="t1">${esc(p.pick)}</div>
    <div class="t2">${vs} · ${when}</div><div class="t2">Fuente: ${esc(p.fuente)}${money(p.precio)}</div></div></article>`;
}
function gap(g) {
  const m = g.rows.find(r => r[0] === "Modelo"), c = g.rows.find(r => r[0] === "Casas");
  if (!m || !c || c[1].some(v => v == null) || m[1].some(v => v == null)) return "";
  const d = Math.max(...m[1].map((v, i) => Math.abs(v - c[1][i])));
  return d >= 0.10 ? `<span class="warn">⚠ El modelo difiere ${Math.round(100 * d)} pts de las casas (posible baja o noticia no incorporada)</span>` : "";
}
function draw() {
  const now = Date.now(), lim = {"24h": 24, "48h": 48, "7d": 168, "all": 1e9}[S.when] * 3600e3, q = S.q.trim().toLowerCase();
  const gs = DATA.games.filter(g => { const t = Date.parse(g.t);
    return t > now - 3 * 3600e3 && t - now <= lim && (S.sport === "all" || g.sport === S.sport)
      && (S.view === "picks" || S.hi !== "1" || (g.prob ?? 0) >= .70)
      && (!q || (g.home + " " + g.away + " " + g.comp).toLowerCase().includes(q)); });
  const el = document.getElementById("list");
  if (S.view === "picks") {
    const ps = gs.flatMap(g => (g.picks || []).map(p => [g, p])).filter(([g, p]) => p.prob != null && mktOk(p) && (S.hi !== "1" || p.prob >= .70))
      .sort((a, b) => b[1].prob - a[1].prob);
    el.innerHTML = ps.length ? `<p class="count">${ps.length} picks, de mayor a menor probabilidad</p>` + ps.map(([g, p]) => pickRow(g, p)).join("")
      : `<p class="empty">No hay picks con estos filtros.</p>`;
    return;
  }
  if (!gs.length) { el.innerHTML = `<p class="empty">No hay partidos con estos filtros.</p>`; return; }
  let html = `<p class="count">${gs.length} partido${gs.length === 1 ? "" : "s"}</p>`, last = "";
  for (const g of gs) {
    const d = new Date(g.t).toLocaleDateString("es", {weekday: "long", day: "numeric", month: "short"});
    if (d !== last) { html += `<div class="day">${d}</div>`; last = d; }
    html += card(g);
  }
  el.innerHTML = html;
}
const viewBar = document.getElementById("view");
function drawView() {
  for (const b of viewBar.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.v === S.view));
  document.getElementById("mkts").style.display = S.view === "picks" ? "flex" : "none";
}
viewBar.onclick = e => { const b = e.target.closest("button"); if (!b) return; S.view = b.dataset.v;
  try { localStorage.setItem("kai-filters", JSON.stringify({sport: S.sport, when: S.when, hi: S.hi, view: S.view, mkt: S.mkt})); } catch (e) {}
  drawView(); draw(); };
chips("sports", SPORTS, "sport"); chips("mkts", MKTS, "mkt");
drawView();
const whenBar = document.getElementById("when");
function chipsWhen() {
  whenBar.innerHTML = WHEN.map(([v, l]) => `<button class="chip" data-k="when" data-v="${v}" aria-pressed="${S.when === v}">${l}</button>`).join("")
    + HI.slice(1).map(([v, l]) => `<button class="chip" data-k="hi" data-v="${S.hi === "1" ? "0" : "1"}" aria-pressed="${S.hi === "1"}">${l}</button>`).join("");
}
whenBar.onclick = e => { const b = e.target.closest("button"); if (!b) return; S[b.dataset.k] = b.dataset.v;
  try { localStorage.setItem("kai-filters", JSON.stringify({sport: S.sport, when: S.when, hi: S.hi, view: S.view, mkt: S.mkt})); } catch (e) {}
  chipsWhen(); draw(); };
chipsWhen();
document.getElementById("q").oninput = e => { S.q = e.target.value; draw(); };
draw();
</script>
</body>
</html>
"""


def main(out="docs/index.html"):
    data = collect()
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(data), encoding="utf-8")
    (p.parent / ".nojekyll").write_text("", encoding="utf-8")
    print(f"{out} generado: {len(data['games'])} partidos")


if __name__ == "__main__":
    main()
