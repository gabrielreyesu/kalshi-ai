# Kalshi AI V5 — validación automática de selecciones internacionales

## Qué cambia
Esta propuesta añade calibración 1X2 fuera de muestra, comparación por temporada, una política exploratoria que selecciona probabilidades originales/calibradas por segmento, e informes semanales mediante GitHub Actions.

**No realiza apuestas, no modifica los motores originales y no sustituye automáticamente un modelo.** El motor de producción continúa sin cambios.

## Evidencia histórica exploratoria
- 3,101 partidos internacionales emparejados en el backtest 2016–2026.
- Temporada 2026: 199 partidos. Original accuracy 57.79%, log loss 0.91577, Brier 0.54043.
- Política adaptativa: accuracy 58.29%, log loss 0.91295, Brier 0.53951.
- IC al 95% de la diferencia de log loss (adaptativo menos original): [-0.01216, 0.00523]. **No es una mejora concluyente**.
- La temporada 2026 se consultó durante el desarrollo: no debe presentarse como holdout prospectivo independiente.

## Ejecución
- Pruebas existentes: `python -m unittest discover -s tests`
- Pruebas nuevas: `python -m pytest -q tests`
- Backtest real: `python -m kalshi_intl.evaluate_history --cache data/intl --first-year 2016 --output reports/intl_real_validation`
- Evaluación automatizada: `python -m kalshi_intl.auto_validate --input reports/intl_real_validation/oos_paired.csv --output reports/intl_auto_validation`
- Entrenamiento de candidato: `python -m kalshi_intl.auto_train --input reports/intl_real_validation/oos_raw.csv --output reports/intl_training/calibrator_candidate.json`

Los datos de ESPN necesitan conexión de red y su descarga puede fallar; GitHub Actions guarda los reportes como artefactos.

## Entrenamiento automático de candidatos
El workflow de GitHub Actions ahora ejecuta pruebas en cada PR y cada lunes reconstruye datos, realiza backtesting, evalúa políticas y entrena un **calibrador candidato** en JSON. No promueve automáticamente el candidato, no modifica el motor de producción ni apuesta. El entrenamiento excluye todos los resultados del día UTC actual y posteriores. El archivo `reports/intl_training/calibrator_candidate.json` contiene parámetros reproducibles sin pickle. Para activar la recurrencia hay que fusionar el PR a la rama principal.
