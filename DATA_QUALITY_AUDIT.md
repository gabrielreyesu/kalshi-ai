# Auditoría de calidad de datos internacionales (sin cambios de motor)

- El backtest internacional registra `data_quality.json` con resumen de partidos finalizados, marcadores en prórroga/penales y sedes neutrales por competición.
- En la muestra evaluada por GitHub (8,255 partidos; 3,703 evaluaciones OOS) ESPN no marcó ninguna sede neutral. La auditoría genera advertencia, sin asignar sedes ficticias.
- Verifica duplicados y que los partidos finalizados tengan marcador. No altera los motores, su entrenamiento ni ninguna probabilidad.
- Los ensayos para excluir partidos de prórroga del ajuste de goles se descartaron al empeorar log loss y Brier en 3,703 partidos. Ver PR #3 cerrado.

El flujo automático semanal guardará el informe junto al backtest. Pendiente evaluar una fuente de sedes neutrales confiable antes de utilizar ese indicador en producción.
