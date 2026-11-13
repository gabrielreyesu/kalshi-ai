# Mocha Picks — instrucciones del proyecto

## Meta
Una **página de predicciones fiable** (https://gabrielreyesu.github.io/mocha-picks/) que el dueño pueda usar con sus
amigos, para NFL, NBA y fútbol (clubes y selecciones). Se juzga por la **probabilidad real de acertar**, no por pagos.

## Principio central: juntar TODA la información útil
El pick final debe salir de combinar todas las fuentes disponibles, no de una sola:
- **Mercados**: precio de Kalshi (lo más cerca posible del inicio), consenso de casas sin margen, Polymarket.
- **Modelos propios**: Elo/Poisson/regresiones por deporte.
- **Contexto de jugadores**: lesiones, alineaciones/onces oficiales, quarterback titular, descansos.
- **Contexto de competición**: clasificación, qué se juega cada equipo, sede, calendario.
- Cualquier fuente nueva se **prueba** y, si aporta, se suma al combinador (`kalshi_nba/final_pick.py`).

## Reglas de trabajo
- Sin fugas de datos futuros: todo dato de un partido debe conocerse antes del inicio.
- Validación temporal fuera de muestra; comparar con log loss, Brier, calibración (ECE) y acierto, con IC95% por fecha.
- Protocolo pre-registrado (escrito antes de ver resultados); reportar también los resultados negativos.
- No cambiar el modelo principal sin pruebas sólidas; lo nuevo va primero en sombra.
- Ramas y Pull Requests; los PRs **no incluyen archivos generados** (`docs/*.html`, `PREDICCIONES.md`): los escriben los bots.
- Nunca ejecutar apuestas. No pedir ni exponer secretos.
- Comunicación con el dueño en español, clara y sin jerga innecesaria.

## Dónde está cada cosa
- `kalshi_nba/` NBA/NFL (motor, página `pagina.py`, progreso `progreso.py`, picks por mercado `picks.py`, pick final `final_pick.py`).
- `kalshi_soccer/` ligas; `kalshi_intl/` copas y selecciones; `kalshi_external/` fuentes externas y pick final en sombra.
- `kalshi_lineup/` investigación: alineaciones, QB, onces, sombras prospectivas, Kalshi histórico, clasificación.
- Resultados de investigación: `LINEUP_RESEARCH.md`, `KALSHI_BENCHMARK.md`, `STAKES_RESEARCH.md`.
- Datos privados: repositorio `gabrielreyesu/mocha-picks-data` (no publicar datos de terceros aquí).
