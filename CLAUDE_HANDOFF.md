# Kalshi AI — traspaso técnico para Claude
**Estado verificado: 12 de noviembre de 2026 (UTC).**
**Repositorio:** https://github.com/gabrielreyesu/kalshi-ai
**Rama principal previa a este documento:** `db8f9a8c3a80dc92dab1a8e8b866addd34ce8aa4`
**Objetivo del usuario:** construir un sistema autónomo para NFL, NBA y fútbol internacional que recoja predicciones y datos externos de forma repetida, entrene candidatos y mejore la calidad de las probabilidades **solo cuando existan pruebas independientes**, con máxima prioridad a la probabilidad de acertar, no a los pagos altos. Desea delegar pruebas e implementación al asistente, no ejecutar paso a paso Colab.

## Lo integrado y verificado en main
- **V5, PR #1, fusionado** https://github.com/gabrielreyesu/kalshi-ai/pull/1  
  `kalshi_intl/calibration.py`, `validation.py`, `evaluate_history.py`, `segment_policy.py`, `auto_validate.py`, `auto_train.py`. Evaluación 1X2 walk-forward, CI bootstrap por fecha, política adaptativa exploratoria, calibrador candidato JSON. GitHub workflow semanal `.github/workflows/intl_auto_validation.yml` los lunes 10:15 UTC. **No activa candidatos ni apuestas automáticamente**.
- **V6, PR #2, fusionado** https://github.com/gabrielreyesu/kalshi-ai/pull/2  
  `kalshi_intl/experiments.py`: torneo temporal entre probabilidades originales, calibrador logístico, mezclas 25/50% y temperatura, con backtest por temporada, reportes y CI. **Investigación, no producción**. 53 tests completos aprobados en ese momento.
- **Prueba V7 de filtrado de prórrogas, PR #3, descartada sin fusionar** https://github.com/gabrielreyesu/kalshi-ai/pull/3  
  El cambio para excluir STATUS_FINAL_AET/STATUS_FINAL_PEN del entrenamiento del Poisson se evaluó sobre los **mismos 3,703 partidos OOS**. Base accuracy 0.579260, logloss 0.921028, Brier 0.542284; variante accuracy 0.579530, logloss 0.921368, Brier 0.542438. Diferencia logloss +0.000341 (IC 95% [+0.000150,+0.000526]); Brier peor. Se preservó motor base por resultados. Ver https://github.com/gabrielreyesu/kalshi-ai/actions/runs/43262939927. **Esta prueba no significa que el manejo de prórrogas sea conceptualmente correcto: hay que separar mercados a 90 minutos de los de resultado oficial y auditar los datos.**
- **Auditoría de calidad, PR #4, fusionada** https://github.com/gabrielreyesu/kalshi-ai/pull/4  
  `kalshi_intl/data_quality.py` guarda `data_quality.json` y alerta si no hay sedes neutrales; verifica duplicados, marcadores, estados. **No cambia el motor original.** En el backtest evaluado: 8,255 partidos en origen, 3,703 OOS; `neutral` aparecía siempre 0. No inventar sedes neutras.
- **V8, PR #5, fusionado** https://github.com/gabrielreyesu/kalshi-ai/pull/5  
  `kalshi_external/`: `core.py` (matching exacto y marcas temporales), `providers.py` (The Odds API, API-Football, feed licenciado), `skorbot.py` (observación API pública; no usar para entrenamiento sin permiso confirmado), `cli.py` (ingesta y seguimiento), `learn.py` (pesos candidatos de mezcla, sin despliegue). Workflow `.github/workflows/external_learning.yml` cada día a las **01:35 y 13:35 UTC**, además de ejecución manual; el semanal realiza evaluación. **77/77 tests pasaron** en CI antes del merge: https://github.com/gabrielreyesu/kalshi-ai/actions/runs/43268039967.

## Estado MÁS RECIENTE de API externa: confirmado
El usuario agregó `THE_ODDS_API_KEY` como GitHub Actions Secret. **No pedirle que pegue su clave en chat ni en commits.**
Se ejecutó manualmente `External forecast research snapshots`: https://github.com/gabrielreyesu/kalshi-ai/actions/runs/43276556716
- **Run successful** a 2026-11-12 18:08 UTC.
- `new_records: 18`, `observations_in_private_cache: 18`.
- `new_by_source: {"oddsapi_bookmaker_consensus": 18}`.
- `sources_with_30_plus_resolved_cases: 0`. **La ingesta funciona pero no hay evidencia de que terceros ya hayan mejorado el motor**.
- Advertencia: `No licensed analyst feed configured; skipped`. API-Football/Skorbot no produjeron registros aceptados en esa ejecución. Comprobar cobertura, licencias, tokens, cuotas y matching.
- Los snapshots crudos se almacenan en `data/external_signals/history.jsonl` restaurada por Actions cache, **no es almacenamiento durable**. `reports/external_learning/status.json` y `source_performance.csv` son reportes de métricas.

## Resultados históricos exploratorios del calibrador internacional
Antes de V8:
- 3,703 predicciones OOS crudas y 3,101 pares originales/calibrados comparables.
- En 3,101: original accuracy ~58.30%, logloss 0.91813, Brier 0.53959; calibrado ~58.08%, logloss 0.91685, Brier 0.54011. Diferencias pequeñas/no concluyentes. En 199 partidos de 2026, política adaptativa accuracy 58.29% vs 57.79%, logloss 0.91295 vs 0.91577, sin ventaja estadísticamente concluyente. 2026 ya se inspeccionó durante desarrollo: **NO es holdout verdaderamente virgen**.
- Exploración V6 de mezcla/temperatura mejoró algunos logloss agregados pero empeoró en otros períodos; no activar por selección retrospectiva ni afirmar éxito sin validación prospectiva.

## Arquitectura y controles actuales
- Repositorio contiene `kalshi_intl`, `kalshi_nba`, `kalshi_soccer`. Tiene `predictions/nfl_registry.csv` y otros CSV de NFL, pero **verificar si existe un motor NFL en el repositorio** antes de asumirlo: no se veía paquete `kalshi_nfl` al inspeccionar el árbol. Un registro NFL observado tenía `model_version=nba_v1`; auditar etiquetas.
- GitHub Actions valida cada PR con `pytest` y programa backtesting/reentrenamiento semanal de **candidatos**. No significa que exista un sistema de promoción automática de producción.
- La ingesta V8 acepta solo snapshots prepartido, exige partido coincidente de manera inequívoca y almacena fuente + captura + publicación; para mercados de casas desmarjina probabilidades y agrega consenso. Feed de analistas necesita autorización expresa `licensed_for_modeling`, con `LICENSED_ANALYST_FEED_URL`/`HOST`. **No scrapear sitios, infringir TOS ni asumir que cuotas = expertos independientes**.
- La ingesta de NFL/NBA tiene soporte para capturar probabilidades, pero el score automático de fuentes externas verificadas está centrado en internacionales con resultados OOS a tiempo reglamentario. Los pesos aprendidos son de investigación. Skorbot permanece `research_only`.
- Una ejecución verde no prueba que haya señales capturadas ni que haya mejorado accuracy.

## Próximas prioridades técnicas, en orden
1. **Verificar costo/estabilidad de API y match rate real** (las 18 capturas de The Odds API y el resto de las fuentes) y la credibilidad de publicación antes del kickoff. Contabilizar cobertura, cuota, fuentes ausentes y latencia. No revelar secretos.
2. **Resolver almacenamiento privado y persistente para snapshots** (Actions cache puede expirar y/o perder datos), con derechos de redistribución y políticas de proveedor. Hacer una solución apropiada a permisos y cuota, no añadir datos de API a repo público.
3. **Reforzar validez temporal y cobertura del dataset**: source publish time vs capturado, created_utc del modelo, kickoff efectivo, identidades de equipos, sedes neutrales, bajas/cambios de horario, deduplicación por evento y por proveedor, diferenciar marcador a 90' vs ET/PK.
4. **Aprendizaje en sombra con comparación justa**: resultados de cada fuente vs baseline y mercado en los *mismos partidos*, logloss/Brier/calibración/accuracy, bootstrap temporal, corrección por comparación de múltiples candidatos, split cronológico verdaderamente independiente. Solo entrenar un blending/stacking con datos pasados de fuentes autorizadas; evitar información futura y correlaciones entre casas/analistas.
5. **Expandir evaluador real de NFL y NBA**; auditar estructura NFL del repositorio y etiquetas de versiones; aportar señales de lesiones, quintetos/alineaciones confirmadas, descanso, localía, clima cuando corresponda, con timestamps.
6. **Incorporar precios históricos pre-partido de Kalshi** para evaluar probabilidad y rendimiento neto tras fees/spread/slippage; nunca asumir odds actuales como históricos. Prioridad de usuario: acertar, no perseguir cuotas.
7. **Automatización de informes y alertas** si faltan datos, cuota, fallan jobs, deriva de modelo o métricas empeoran. Mantener champion protegido; promover automáticamente solo tras criterios estrictos y reversibles **y con autorización explícita para activación en producción**.

## Cómo trabajar
El usuario quiere que Claude **inspeccione el código real del GitHub, ejecute pruebas, cambie archivos en una rama y abra PRs**, sin pedirle notebooks y sin narrarle cada comando. Preferir acciones concretas y reproducibles sobre recomendaciones generales. Mantener resguardos, explicitar cuando falta evidencia o datos. Asegurar que no se dupliquen cambios ya fusionados. No asumir que un origen gratuito permite entrenamiento de IA: verificar licencia del proveedor.
