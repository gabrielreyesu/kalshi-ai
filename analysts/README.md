# Analistas (experimental)

- `watchlist.csv`: lista de seguimiento (26 registros). Ninguno tiene permiso de uso confirmado, así que **ninguno
  alimenta el modelo**. La auditoría (`python -m kalshi_external.analysts audit`) calcula el intervalo de confianza de
  cada récord y lo compara con la referencia que debe superar.
- `picks_template.csv`: formato del libro de pronósticos. Solo se aceptan filas con `authorized_for_modeling=yes`,
  publicadas y capturadas antes del partido (`python -m kalshi_external.analysts ingest <archivo>`).
- El libro (`data/analysts/picks.jsonl`) no se publica en este repositorio: es contenido de terceros.

Mercados separados: NFL `moneyline|ats|total|prop`; NBA `moneyline|spread|total|prop`;
fútbol `1x2_90` (90 minutos) | `1x2_official` (con prórroga) | `btts|goals|handicap`.

Referencias que un analista debe superar: 52.4% en mercados a cuota ~1.91; en "ganador directo", el acierto de elegir
siempre al favorito (66.3% NFL, 67.6% NBA en nuestros backtests); en tips con cuota media conocida, 1/cuota.
Con menos de 200 decisiones, un porcentaje alto es compatible con suerte. Además, los primeros puestos de una
clasificación están inflados por selección: de muchos analistas, alguno siempre tendrá una buena racha.

## Seguimiento manual (opcional, sin ejecución diaria)
1. Antes de que empiece el partido, añade una línea a `mis_picks.csv`:
   `analista,deporte,local,visitante,fecha_partido,mercado,seleccion,linea,cuota,link`
   - `deporte`: nfl, nba o futbol. `mercado`: ganador, spread, total o ambos_anotan.
   - `seleccion`: local, visitante, empate, mas, menos, si, no (o el nombre del equipo).
   - `linea`: obligatoria en spread y total (ej. -3.5, 224.5). `cuota`: decimal, opcional (ej. 1.91).
2. Si decides guardar un pick, el flujo `analistas-seguimiento` une la línea con el partido real, registra la hora y la añade a
   `libro.csv`. Lo que no se pueda unir o llegue tarde (menos de 15 minutos antes del inicio) aparece en `ESTADO.md`
   con el motivo.
3. Cuando ejecutes el workflow, se liquidan los partidos jugados y `ESTADO.md` muestra el acierto de cada analista con su margen de error.
Anota solo picks que el analista haya publicado de forma abierta; no pegues contenido de pago.
Estos picks son solo para evaluar: no entrenan el modelo ni cambian las predicciones.

## Referencias automáticas: flujo recomendado

No necesitas transcribir picks de X o Discord. El workflow manual ya no tiene un horario diario; solo se ejecuta si cambias `mis_picks.csv` o lo activas voluntariamente. El workflow `External forecast research snapshots` sigue recogiendo datos permitidos por API y genera `reports/external_learning/REFERENCIAS.md` con cobertura y enlaces de consulta sin copiar apuestas de terceros. La lista de analistas es una referencia, no un feed autorizado para entrenar.

Las capturas muestran los tipos de mercados que interesan (NFL receptions/rec yards/spreads, NBA puntos/rebotes/3PM, fútbol 1X2/tiros a puerta, NHL SOG/spreads), pero no prueban un historial rentable o habilidad predictiva. El conector actual The Odds API solo consulta `h2h`, no props de jugadores.
