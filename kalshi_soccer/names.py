"""Equivalencias de nombres: football-data.org (API) -> football-data.co.uk (histórico).

Regla: nunca se asigna un equipo por coincidencia aproximada. Si no está en MAPEO ni coincide
exacto/normalizado, queda SIN RESOLVER y se reporta con sugerencias para que lo confirmes.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import get_close_matches

MAPEO = {
    "Bundesliga": {
        "FC Bayern München": "Bayern Munich", "1. FC Köln": "FC Koln",
        "1. FC Union Berlin": "Union Berlin", "RB Leipzig": "RB Leipzig",
        "SV 07 Elversberg": "Elversberg", "SC Freiburg": "Freiburg",
        "Bayer 04 Leverkusen": "Leverkusen", "SC Paderborn 07": "Paderborn",
        "Borussia Mönchengladbach": "M'gladbach", "Borussia Dortmund": "Dortmund",
        "Eintracht Frankfurt": "Ein Frankfurt", "TSG 1899 Hoffenheim": "Hoffenheim",
        "FC Schalke 04": "Schalke 04", "VfB Stuttgart": "Stuttgart", "FC Augsburg": "Augsburg",
        "1. FSV Mainz 05": "Mainz", "SV Werder Bremen": "Werder Bremen", "Hamburger SV": "Hamburg",
    },
    "Premier League": {
        "Manchester City FC": "Man City", "Fulham FC": "Fulham", "Tottenham Hotspur FC": "Tottenham",
        "Ipswich Town FC": "Ipswich", "Coventry City FC": "Coventry",
        "Crystal Palace FC": "Crystal Palace", "Aston Villa FC": "Aston Villa",
        "Sunderland AFC": "Sunderland", "AFC Bournemouth": "Bournemouth",
        "Nottingham Forest FC": "Nott'm Forest", "Chelsea FC": "Chelsea", "Leeds United FC": "Leeds",
        "Newcastle United FC": "Newcastle", "Arsenal FC": "Arsenal", "Hull City AFC": "Hull",
        "Brighton & Hove Albion FC": "Brighton", "Liverpool FC": "Liverpool",
        "Manchester United FC": "Man United", "Brentford FC": "Brentford", "Everton FC": "Everton",
    },
    "La Liga": {
        "Athletic Club": "Ath Bilbao", "RCD Espanyol de Barcelona": "Espanol",
        "Deportivo Alavés": "Alaves", "RC Deportivo La Coruña": "La Coruna",
        "Rayo Vallecano de Madrid": "Vallecano", "Valencia CF": "Valencia", "Sevilla FC": "Sevilla",
        "Real Betis Balompié": "Betis", "Real Madrid CF": "Real Madrid", "RC Celta de Vigo": "Celta",
        "FC Barcelona": "Barcelona", "Villarreal CF": "Villarreal",
        "Real Racing Club de Santander": "Santander", "Real Sociedad de Fútbol": "Sociedad",
        "CA Osasuna": "Osasuna", "Málaga CF": "Malaga", "Getafe CF": "Getafe",
        "Club Atlético de Madrid": "Ath Madrid", "Levante UD": "Levante", "Elche CF": "Elche",
    },
    "Serie A": {
        "SS Lazio": "Lazio", "US Lecce": "Lecce", "Como 1907": "Como", "Cagliari Calcio": "Cagliari",
        "Parma Calcio 1913": "Parma", "AS Roma": "Roma", "SSC Napoli": "Napoli", "Torino FC": "Torino",
        "Juventus FC": "Juventus", "Venezia FC": "Venezia", "US Sassuolo Calcio": "Sassuolo",
        "Genoa CFC": "Genoa", "Bologna FC 1909": "Bologna", "Udinese Calcio": "Udinese",
        "AC Milan": "Milan", "Frosinone Calcio": "Frosinone", "Atalanta BC": "Atalanta",
        "FC Internazionale Milano": "Inter", "AC Monza": "Monza", "ACF Fiorentina": "Fiorentina",
    },
    "Ligue 1": {
        "ES Troyes AC": "Troyes", "Le Havre AC": "Le Havre", "OGC Nice": "Nice",
        "Paris Saint-Germain FC": "Paris SG", "AS Monaco FC": "Monaco", "AJ Auxerre": "Auxerre",
        "Racing Club de Lens": "Lens", "Toulouse FC": "Toulouse", "Lille OSC": "Lille",
        "Olympique Lyonnais": "Lyon", "RC Strasbourg Alsace": "Strasbourg", "Stade Brestois 29": "Brest",
        "Paris FC": "Paris FC", "Angers SCO": "Angers", "FC Lorient": "Lorient",
        "Stade Rennais FC 1901": "Rennes", "Olympique de Marseille": "Marseille", "Le Mans FC": "Le Mans",
    },
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def resolve_team(league: str, api_name: str, known: list):
    """-> (nombre_historico | None, motivo). `known` = equipos del histórico para esa liga."""
    target = MAPEO.get(league, {}).get(api_name)
    if target is not None:
        return (target, "mapeo") if target in known else (None, f"mapeo apunta a '{target}' que no está en el histórico")
    if api_name in known:
        return api_name, "exacto"
    hits = [t for t in known if _norm(t) == _norm(api_name)]
    if len(hits) == 1:
        return hits[0], "normalizado"
    sug = get_close_matches(api_name, known, n=3, cutoff=0.5)
    return None, "sin resolver; sugerencias: " + ", ".join(sug)
