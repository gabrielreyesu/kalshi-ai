# V7 — Regulación de 90 minutos y calidad de datos

## Ajustes
- Poisson y el conjunto de entrenamiento walk-forward excluyen partidos cuyo marcador puede incluir prórroga/penales (STATUS_FINAL_AET y STATUS_FINAL_PEN).
- El backtesting guarda un informe adicional \`reports/intl_real_validation/data_quality.json\`, con conteos por tipo de resultado, cantidad de partidos neutrales y advertencia si ESPN no identifica sedes neutrales.
- Los partidos de sede neutral NO se inventan. Si ESPN omite el indicador, la auditoría informa el problema.
- Las 4 pruebas nuevas comprueban el filtro del entrenamiento, datos duplicados y ausencia de partidos neutros. En conjunto con la suite existente se verificaron 57 pruebas localmente, 2 advertencias no bloqueantes.

## Limitaciones
Todavía no se ha demostrado si estos ajustes aumentan el log loss o aciertos reales: se requiere reconstruir las 3,703+ predicciones OOS históricas desde ESPN y comparar con el motor previo.

Esta corrección modifica el entrenamiento de goles que usa el CLI en producción, por lo que será sometida a validación antes del merge. La auditoría no promueve candidatos ni apuesta automáticamente.
