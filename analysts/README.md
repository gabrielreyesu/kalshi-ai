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
