# Auditoría de V8 y cambios de V9 (12 nov 2026)

## Hallazgos principales
1. **El ciclo de aprendizaje externo no podía cerrarse.** `kalshi_external.cli` solo leía resultados de
   `reports/intl_real_validation/oos_raw.csv`, archivo que genera OTRO workflow y que no existe en `external_learning`.
   Consecuencia: ninguna fuente podía llegar a tener casos resueltos, por muchos días que corriera.
2. **Las copas de clubes nunca coincidían.** Se pedía `soccer_uefa_champs_league` a The Odds API (gasta cuota) pero el
   registro de clubes no se cargaba, así que el 100% de esos registros se descartaba.
3. **Sin almacenamiento durable.** `history.jsonl` vive solo en la caché de Actions, que caduca.
4. **Etiqueta de versión incorrecta.** Los registros de NFL salían como `nba_v1`. El motor NFL sí existe: vive en
   `kalshi_nba` con `--league nfl` (configuración propia en `engine.CONFIGS`).
5. **El aprendiz (`learn.py`) solo cubre selecciones** y limita el peso de la fuente a 0.40. Con la evidencia de abajo,
   ese tope es demasiado bajo para fuentes de mercado. Pendiente (ver siguiente PR).
6. **Sede neutral:** ESPN marca `neutralSite` casi siempre en falso en fútbol; el motor trata como local a equipos que
   juegan en sede neutral (Mundial, Eurocopa, Copa América). Pendiente: inferir neutralidad por competición y probarlo.
7. **Coincidencia de nombres exacta:** variantes como "LA Clippers" / "Los Angeles Clippers" se pierden. Pendiente: medir
   la tasa de coincidencia por fuente y añadir alias verificados.
8. Cuota de The Odds API: 3 deportes x 1 región x 1 mercado x 2 corridas/día = ~180 créditos/mes.

## Evidencia ya disponible (no hay que esperar a acumular casos)
Mismos partidos, fuera de muestra, validación cronológica (cada temporada se predice con datos anteriores).
"Mercado" = mediana de casas sin margen (cuotas históricas de ESPN / football-data.co.uk).

| Motor | Partidos | Acierto modelo | Acierto mercado | Logloss modelo | Logloss mercado |
|---|---|---|---|---|---|
| Fútbol 5 ligas | 7,582 | 52.5% | 54.6% | 0.9871 | 0.9651 |
| Copas europeas | 2,488 | 56.8% | 60.2% | 0.9473 | 0.9102 |
| Selecciones | 2,463 | 59.5% | 62.4% | 0.9006 | 0.8335 |
| NBA | 10,225 | 65.1% | 67.6% | 0.6216 | 0.5983 |
| NFL | 2,095 | 64.2% | 66.3% | 0.6352 | 0.6093 |

En los cinco motores el consenso de mercado acierta entre 2 y 3.4 puntos más que el modelo, y las mezclas modelo+mercado
no superan al mercado solo (ver reportes `*_backtest_report.txt`). Conclusión operativa: si la prioridad es acertar, la
probabilidad final debería anclarse al consenso de mercado cuando exista, con el modelo como respaldo. Esto NO se activa
en este PR: es la propuesta del siguiente, en modo sombra y con evaluación prospectiva.

## Cambios de este PR (no tocan las probabilidades de producción)
- `kalshi_external/outcomes.py`: resultados reales de ESPN para intl, club, NBA y NFL (fútbol solo 90 min; empates NFL fuera).
- `kalshi_external/cli.py`: carga también el registro de clubes; con `KALSHI_RESOLVE_LIVE=1` puntúa todas las fuentes;
  el resumen informa casos resueltos por fuente.
- `kalshi_external/archive.py` + workflow: archivo durable opcional en un repositorio PRIVADO `kalshi-ai-data`
  (secreto `DATA_REPO_TOKEN`). Sin el secreto, el comportamiento es el de antes.
- `kalshi_nba/cli.py`: `model_version` por liga (`nba_v1`, `nfl_v1`).
- 3 pruebas nuevas (80 en total).
