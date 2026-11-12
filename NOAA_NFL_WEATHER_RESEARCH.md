# Kalshi AI — NOAA NWS outdoor weather research (v15)

This is a new official, free, public-domain source: [National Weather Service](https://www.weather.gov/documentation/services-web-api). No API key or sports odds credits.

## What is recorded

- For upcoming NFL fixtures in the local `predictions/nfl_registry.csv`, collect official NOAA hourly weather around kickoff when the game is within the next 7 days.
- **Venue must be corroborated by ESPN for that exact event**, matching a vetted open-air field. International games, roofed venues and unexpected/unknown stadium names are skipped; no city-center guess.
- Record both forecast issuance and collection timestamp, along with kickoff, forecast hour, temperature F, maximum wind mph, wind direction, precipitation probability, and source attribution.
- Weather snapshots are stored separately in private `kalshi-ai-data/external_signals/weather.jsonl`, using the existing versioned private archive and the GitHub Actions cache. Aggregate coverage is in `reports/external_learning/weather_status.json`.
- NWS provider problems are reported as warnings; the system does not fabricate weather or alter model predictions.

## Initial restricted venue catalog

**Green Bay Packers** (Lambeau Field), **Kansas City Chiefs** (Arrowhead Stadium), **New York Giants/Jets** (MetLife Stadium), **Philadelphia Eagles** (Lincoln Financial Field), **Pittsburgh Steelers** (Acrisure Stadium), **Seattle Seahawks** (Lumen Field). Coordinates in `kalshi_external/nws_weather.py` were checked against public mapping references. Outdoor games at other venues currently receive no weather signal.

## Sources

- Weather API usage, free and public-domain: https://www.weather.gov/documentation/services-web-api
- NWS weather point-to-grid method: https://www.weather.gov/documentation/services-web-api
- Stadium coordinate references: https://mapcarta.com/22864550 , https://www.latlong.net/place/arrowhead-stadium-mo-usa-31869.html , https://mapcarta.com/W24221553 , https://mapcarta.com/21464102 , https://mapcarta.com/es/22731928 , https://en.wikipedia.org/wiki/Lumen_Field

## Safety and limits

All forecasts are collected prospectively and stored as environmental observations, **not as probability predictions**. They are not used by any production model or weighted by a trainer. The value of weather (wind/precipitation/temperature) must be established in prospective NFL logloss/accuracy experiments before even considering promotion. No paid services are enabled or requested.


## Coverage-fallback amendment

First real NOAA workflow found 3 eligible games and 0 matched forecasts without warning. The event summary may omit venue metadata. The adapter now checks the ESPN scoreboard for the same event id as a fallback, while still requiring the actual venue name to match a curated open-air stadium. No location is guessed.

Every omitted candidate now reports whether it lacked a verified venue, had the wrong stadium, had no hourly link, or lacked a forecast period covering kickoff. This is an instrumentation improvement, not an accuracy improvement.
