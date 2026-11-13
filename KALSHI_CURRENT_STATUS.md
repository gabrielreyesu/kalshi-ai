# Kalshi AI — Estado verificado y próximos pasos
**Actualizado:** 12 de noviembre de 2026.
**Repositorio de código:** https://github.com/gabrielreyesu/mocha-picks (antes `kalshi-ai`; GitHub redirige el nombre antiguo)
**Archivo privado:** `gabrielreyesu/mocha-picks-data` (antes `kalshi-ai-data`; los workflows prueban ambos nombres) (PRIVADO; no publicar históricos ni tokens).

## Preferencia y restricciones del usuario
El objetivo es maximizar la probabilidad de predicción correcta, con probabilidades bien calibradas, en NFL, NBA, fútbol internacional y de clubes. Reunir fuentes externas legales y reproducibles de modo automático, sin copiar picks de X/Discord, sin gastos adicionales no autorizados y sin falsas afirmaciones de mejora. No apostar automáticamente ni modificar champion/producción por mejoras retrospectivas. El usuario quiere que el asistente opere en GitHub con ramas, PRs, tests, backtests e informes, sin instrucciones manuales de Colab.

## Infraestructura existente
- `.github/workflows/external_learning.yml`: consulta fuentes externas a las 01:35 y 13:35 UTC; se ejecuta también al modificar fuentes y manualmente.
- `.github/workflows/intl_auto_validation.yml`: pruebas por PR, evaluación y entrenamiento experimental semanal los lunes.
- `THE_ODDS_API_KEY` y `DATA_REPO_TOKEN` están guardados como GitHub Actions Secrets (NO solicitarlos ni exponerlos). El archivo privado contiene odds y NOAA weather JSONL.
- Registro de analistas `analysts/watchlist.csv` para referencias, sin scraping, entrenamiento ni entrada manual necesaria; manual es opcional.
- Monitoreo periódico también configurado en ChatGPT para consultar GitHub y notificar fallos importantes.

## Cambios y validación del 12 de noviembre de 2026
| Pull request | Estado | Función y protección |
|---|---|---|
| [#14](https://github.com/gabrielreyesu/kalshi-ai/pull/14) | fusionado | Guardar capturas por timestamp completo, no una por día. `source_diagnostics.py` compara mercado vs modelo por evento resuelto con logloss, Brier, accuracy, bootstrap por fecha y cobertura de movimientos de precio. |
| [#15](https://github.com/gabrielreyesu/kalshi-ai/pull/15) | fusionado | Nuevo adaptador de NOAA/National Weather Service para estadios NFL al aire libre, con sede corroborada por ESPN y conservación en archivo privado. Sin costo ni cambios de producción. |
| [#16](https://github.com/gabrielreyesu/kalshi-ai/pull/16) | fusionado | Pesos candidatos por separado para NFL, NBA, clubes y selecciones, con resultados reales y train/test cronológico. Workflow semanal restaura el archivo privado. Solo investigación. |
| [#17](https://github.com/gabrielreyesu/kalshi-ai/pull/17) | fusionado | Fallback de sede en ESPN scoreboard cuando el resumen carece de metadatos y causas de omisión explícitas. |
| [#18](https://github.com/gabrielreyesu/kalshi-ai/pull/18) | fusionado | Errores NOAA explícitos sin revelar secretos; CI cubre cambios a `kalshi_external/**`. |
| [#19](https://github.com/gabrielreyesu/kalshi-ai/pull/19) | fusionado | Resolver redirecciones HTTPS solo hacia rutas y hosts oficiales ESPN/NOAA, máximo 2. El proveedor NOAA finalmente registró 3 pronósticos válidos para 3 partidos en la ejecución [43313261689](https://github.com/gabrielreyesu/kalshi-ai/actions/runs/43313261689). |
| [#20](https://github.com/gabrielreyesu/kalshi-ai/pull/20) | fusionado | `archive_audit.py`: controles de JSON, duplicados, normalización de probabilidades, fechas anteriores al inicio y rangos de clima; bloquea guardar nuevos datos corruptos. 131 pruebas aprobadas en PR, además de ensayos previos. |

## Última muestra y limitaciones
- El archivo privado tenía **203 capturas de cuotas** para **29 partidos distintos** (NFL 14, NBA 4, clubes 11), y **3 capturas meteorológicas NOAA** verificadas en la ejecución posterior al PR19. Las capturas son observaciones, no aciertos, y no acreditan mejoras predictivas.
- No hay evidencia estadística prospectiva suficiente para promover fuentes externas ni reemplazar los pronósticos de los motores.
- La actual The Odds API consulta `h2h`; **no** contiene todavía player props de recepciones NFL, rebotes/puntos/3PM NBA, tiros a puerta de fútbol o NHL. Añadir mercados exige evaluar planes, cuota, permisos, timestamps y liquidación; nunca activar compras o gasto de API sin autorización.
- NOAA es fuente de condiciones meteorológicas independientes, no probabilidades. Registrar clima NO equivale a haber demostrado que mejora la NFL.
- El historial histórico debe continuar en `kalshi-ai-data` privado; revisa compatibilidad con términos de cada proveedor y posibles periodos de retención. No publicar data de terceros en este repositorio.
- Verificar la ejecución recurrente completa del workflow semanal, que todavía no se ha observado después de integrar el archivo privado.
- Las comparaciones de fuentes requieren resultados resueltos idénticos, resultados propios *as-of* preservados, evaluación por deporte/mercado, corrección de sesgo de selección e intervalos de confianza.

## Siguientes prioridades de investigación
1. **Recolección confiable**: comprobar cobertura real por fuente, timestamps, identidades de equipos, resúmenes sin lecturas, errores de API, cuota restante y diferencias entre pronósticos originales y resultados.
2. **Calidad prospectiva**: acumular 100+ eventos resueltos por fuente/mercado y evaluar logloss, Brier, accuracy vs champion y vs consenso contemporáneo, con bootstrap por día y sensibilidad a selección de múltiples candidatos.
3. **Clima NFL**: construir conjunto de features con tiempo de captura, comprobar si la observación añade poder predictivo al consenso/elo con split temporal; no conectar al modelo aún.
4. **Otros datos gratuitos/autorizados**: reportes oficiales de lesiones y alineaciones con hora/estado, NFL weather en sedes adicionales SOLO verificadas; fuentes de player props solo cuando los límites y derechos sean compatibles. No scraping prohibido.
5. **Operación y alertas**: validar historial privado, fallos del workflow, tasa de cobertura y deriva de pronósticos, con status y seguimiento. No auto-promoción ni apuesta real.
