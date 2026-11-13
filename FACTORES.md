# Registro de factores — Mocha Picks

Cada factor se prueba con el mismo protocolo: datos conocidos antes del inicio, validación fuera de muestra, criterio
fijado antes de ver resultados. **Aporta** = mejora con IC95% > 0. Dos niveles: frente al **modelo propio** (útil donde
no hay mercado) y frente al **mercado** (Kalshi/casas: la vara más alta, porque el mercado ya incorpora casi todo).

Estados: ✅ en producción · 🟡 aporta al modelo / en sombra · ⚪ no concluyente · ❌ descartado · ⏳ pendiente.

## Mercados y combinación
| Factor | Deporte | Estado | Resultado | Dónde |
|---|---|---|---|---|
| Consenso de casas sin margen | todos | ✅ | Referencia más fiable junto con Kalshi | `final_shadow.py` |
| Precio de Kalshi antes del inicio | NBA/NFL | ✅ | Kalshi T-60 ≈ casas; mejora al acercarse el partido | `KALSHI_BENCHMARK.md` |
| Pick final combinado Kalshi + casas + modelo | NBA/NFL | ✅ | Empata con casas; mejora mucho sin casas | `final_pick.py` |
| Pick final combinado | fútbol | ⏳ | Validación en curso (3.120 partidos) | `soccer_kalshi_bench.py` |
| Polymarket | NBA/NFL | ⏳ | Se captura; falta añadirlo al combinador | `polymarket.py` |
| Línea de goles de las casas (más/menos de 2,5) + modelo | fútbol | ✅ | Empata con las casas (+0,0001); mejora al modelo solo +0,0113 [0,0079; 0,0145] en 7.769 partidos; picks de goles en copas y selecciones | `goals_bench.py` |
| Movimiento del precio de Kalshi (T-24 h → T-60 min) | NBA/NFL | ❌ | NBA −0,0001 [−0,0020; +0,0021]; NFL −0,0012: el precio final ya lo incluye | `KALSHI_BENCHMARK.md` |

## Jugadores
| Factor | Deporte | Estado | Resultado | Dónde |
|---|---|---|---|---|
| Jugadores disponibles (lesiones) | NBA | 🟡 | Backtest −0,0098 log loss; sombra en curso; no supera a Kalshi ni aporta en el combinador (peso −0,15) | `LINEUP_RESEARCH.md` |
| Quarterback titular | NFL | 🟡 | Backtest −0,0133; sombra en curso; ≈ Kalshi. En el combinador sube el peso del modelo a 0,44 y mejora el pick +0,0014 [−0,0035; +0,0062] (no significativo, 246 partidos): activar si la sombra lo confirma | `LINEUP_RESEARCH.md` |
| Once titular (continuidad) | copas | 🟡 | Backtest −0,0030 (débil); sombra en curso | `LINEUP_RESEARCH.md` |
| Once titular | selecciones | ❌ | Sin efecto | `LINEUP_RESEARCH.md` |
| Suspensiones (tarjetas) | fútbol | ⏳ | — | — |
| Valor individual de jugador (no solo presencia) | fútbol | ⏳ | — | — |
| Descanso / back-to-back | NBA | ✅ | En el modelo base | `engine.py` |
| Viajes, husos horarios y reloj del visitante | NBA/NFL | ❌ | NBA −0,0002 (vs mercado −0,0004); NFL −0,0005: el descanso ya está en el modelo y el mercado lo incorpora | `TRAVEL_RESEARCH.md` |

## Contexto del partido
| Factor | Deporte | Estado | Resultado | Dónde |
|---|---|---|---|---|
| Localía | todos | ✅ | En los modelos base | motores |
| Sede neutral inferida | selecciones | ⚪ | Pocos partidos; repetir tras torneos | `research/intl-neutral-venues` |
| Qué se juega cada equipo (clasificación) | ligas | ⚪ | +0,0003, no significativo; aviso informativo | `STAKES_RESEARCH.md` |
| Qué se juega cada equipo | Nations League | ❌ | Sin efecto | `STAKES_RESEARCH.md` |
| Racha / forma reciente | todos | ✅/❌ | En el modelo base; variantes extra solo ayudan en NCAA y selecciones | `reports/variables_extra*.txt` |
| Enfrentamientos directos | todos | ✅/❌ | Solo aporta en selecciones (activo) | `kalshi_intl/model.py` |
| Clima (viento, lluvia, frío) | NFL | ⏳ | Protocolo listo (`WEATHER_RESEARCH.md`); descarga histórica Open-Meteo pausada hasta que se reinicie el cupo diario | `weather_nfl.py` |
| Hora del partido / zona horaria (reloj biológico del visitante) | NBA/NFL | ❌ | Incluido en la prueba de viajes (variante B) | `TRAVEL_RESEARCH.md` |
| Árbitro | fútbol/NFL | ⏳ | — | — |
| Prórrogas en entrenamiento (90 min) | selecciones | ❌ | Empeora | PR #3 |
| Distribución real de márgenes (números clave 3/7) para hándicaps | NFL | ❌ | +0,0001 [−0,0010; +0,0012] frente a la normal: se mantiene la normal | `NFL_MARGINS.md` |
