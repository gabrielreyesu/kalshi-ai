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
- Pruebas existentes: \`python -m unittest discover -s tests\`
- Pruebas nuevas: \`python -m pytest -q tests/test_intl_calibration.py tests/test_intl_validation.py tests/test_intl_segment_policy.py tests/test_auto_validate.py\`
- Backtest real: \`python -m kalshi_intl.evaluate_history --cache data/intl --first-year 2016 --output reports/intl_real_validation\`
- Evaluación automatizada: \`python -m kalshi_intl.auto_validate --input reports/intl_real_validation/oos_paired.csv --output reports/intl_auto_validation\`

Los datos de ESPN necesitan conexión de red y su descarga puede fallar; GitHub Actions guarda los reportes como artefactos. Si se recomienda promoción, requiere validación prospectiva antes de aplicarse. **No hay entrenamiento recurrente de un motor de producción en este cambio.**
